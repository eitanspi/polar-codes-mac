"""
ncg_isi_gpu_curriculum.py
=========================
Optimized NCG ISI-MAC training pipeline.

Goal: push GPU utilization from ~30% to 70-90% by moving all data generation
to GPU (torch) and using mixed precision. Keeps the existing model
(`ISIMACNeuralDecoder` from `neural.ncg_isi_mac`) intact.

Key changes vs `scripts/local_analysis/ncg_isi_chain_local.py`:
  * GPU-batched data-gen: bits sampled, polar-encoded (XOR butterfly), and
    pushed through the ISI-MAC channel all on the same device as the model
    (no numpy ↔ torch ping-pong, no host->device transfers per iter).
  * Mixed precision: torch.amp.autocast(bf16) on CUDA; fp32 on CPU.
    bf16 chosen over fp16 because the per-step logits/logsumexp pipeline is
    sensitive to underflow and bf16 has the same exponent range as fp32 so we
    don't need GradScaler.
  * Larger batch size (default 512 on GPU, 32 on CPU).
  * Optional `torch.compile` (off by default — model has data-dependent
    Python control flow on `b`/`frozen_u`/`frozen_v`).
  * Bench mode (`--bench`) and verify mode (`--verify`).

CLI examples
------------
Bench at N=64 for 1000 iters (current vs optimized):
  python ncg_isi_gpu_curriculum.py --bench --N 64 --iters 1000
  python ncg_isi_gpu_curriculum.py --bench --N 64 --iters 1000 --baseline

Verify N=64 from existing ckpt (5000 iters then eval BLER):
  python ncg_isi_gpu_curriculum.py --verify --N 64 --iters 5000

Full curriculum on cluster (do NOT run locally — slow):
  python ncg_isi_gpu_curriculum.py --curriculum --batch 512
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from polar.channels_memory import ISIMAC  # used only for sigma2/h sanity
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.encoder import bit_reversal_perm, polar_encode_batch
from neural.ncg_isi_mac import ISIMACNeuralDecoder


# ─────────────────────────────────────────────────────────────────────────────
#  GPU-resident data generation
# ─────────────────────────────────────────────────────────────────────────────

class GPUDataGen:
    """
    All-on-device data generator for NCG ISI-MAC training.

    Samples random (u, v) ∈ {0,1}^{B×N}, polar-encodes via torch XOR butterfly,
    runs the ISI-MAC channel:

        Z_t = (1-2 X_t) + (1-2 Y_t) + h*((1-2 X_{t-1}) + (1-2 Y_{t-1})) + W_t,
        W_t ~ N(0, sigma2),

    all without leaving the GPU. The polar encoder is a fixed bit-reversal +
    n butterfly XOR stages, perfectly vectorizable.

    Encoder verified bit-exact against `polar.encoder.polar_encode_batch` in
    `_self_test_encoder()`.
    """

    def __init__(self, N, sigma2, h, device, dtype=torch.float32):
        self.N = int(N)
        self.n = int(self.N).bit_length() - 1
        assert (1 << self.n) == self.N
        self.sigma2 = float(sigma2)
        self.h = float(h)
        self.sigma = math.sqrt(self.sigma2)
        self.device = device
        self.dtype = dtype  # dtype for z (channel output)

        br_np = bit_reversal_perm(self.n).astype(np.int64)
        self.br = torch.from_numpy(br_np).to(device)

    @torch.no_grad()
    def sample(self, B):
        """Return (z, u, v) on self.device. u,v are int64; z is self.dtype."""
        N = self.N
        # Sample bits directly on GPU.
        u = torch.randint(0, 2, (B, N), device=self.device, dtype=torch.int64)
        v = torch.randint(0, 2, (B, N), device=self.device, dtype=torch.int64)

        x = self._polar_encode(u)
        y = self._polar_encode(v)

        # BPSK in dtype (fp32 always — channel sim is light)
        sx = 1.0 - 2.0 * x.to(self.dtype)
        sy = 1.0 - 2.0 * y.to(self.dtype)

        # Previous symbols (initial state = 0 → BPSK = +1)
        ones_col = torch.ones((B, 1), device=self.device, dtype=self.dtype)
        sx_prev = torch.cat([ones_col, sx[:, :-1]], dim=1)
        sy_prev = torch.cat([ones_col, sy[:, :-1]], dim=1)

        mu = sx + sy + self.h * (sx_prev + sy_prev)
        w = torch.randn((B, N), device=self.device, dtype=self.dtype) * self.sigma
        z = mu + w
        return z, u, v

    @torch.no_grad()
    def _polar_encode(self, u):
        """Polar-encode u (B,N) int64 → x (B,N) int64. Fully vectorized.

        Uses in-place XOR (`x[:, :, 0, :] ^= x[:, :, 1, :]`) to match the numpy
        encoder exactly and avoid an extra `stack` per stage.
        """
        N = self.N
        # Bit-reversal permutation (out-of-place, makes the buffer writable)
        x = u.index_select(1, self.br).contiguous()
        B = x.shape[0]
        step = 1
        while step < N:
            x_r = x.view(B, N // (2 * step), 2, step)
            x_r[:, :, 0, :] ^= x_r[:, :, 1, :]
            # x_r is a view of x, so x already reflects the XOR
            step *= 2
        return x


def _self_test_encoder(N=64, B=4, seed=0):
    """Verify torch encoder matches numpy encoder bit-exactly."""
    rng = np.random.default_rng(seed)
    u_np = rng.integers(0, 2, (B, N)).astype(np.int32)
    x_np = polar_encode_batch(u_np)

    gen = GPUDataGen(N=N, sigma2=1.0, h=0.3, device=torch.device("cpu"))
    u_t = torch.from_numpy(u_np).long()
    x_t = gen._polar_encode(u_t).numpy().astype(np.int32)
    assert np.array_equal(x_np, x_t), "torch polar encoder mismatch vs numpy!"


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline (numpy) data generator — for benchmark comparison
# ─────────────────────────────────────────────────────────────────────────────

class NumpyDataGen:
    """Replicates the data-gen loop from ncg_isi_chain_local.py."""

    def __init__(self, N, sigma2, h, device):
        self.N = int(N)
        self.ch = ISIMAC(sigma2=sigma2, h=h)
        self.device = device

    def sample(self, B):
        rng = np.random.default_rng()
        u = rng.integers(0, 2, (B, self.N)).astype(np.int32)
        v = rng.integers(0, 2, (B, self.N)).astype(np.int32)
        x = polar_encode_batch(u)
        y = polar_encode_batch(v)
        z = self.ch.sample_batch(x, y).astype(np.float32)
        z_t = torch.from_numpy(z).to(self.device)
        u_t = torch.from_numpy(u).long().to(self.device)
        v_t = torch.from_numpy(v).long().to(self.device)
        return z_t, u_t, v_t


# ─────────────────────────────────────────────────────────────────────────────
#  Training step
# ─────────────────────────────────────────────────────────────────────────────

def _make_model(device):
    return ISIMACNeuralDecoder(
        d=16, hidden=64, n_layers=2, z_hidden=32, z_encoder_type="window"
    ).to(device)


def _warm_start(model, ckpt_path):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return 0
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    own = model.state_dict()
    loaded = 0
    for k, w in sd.items():
        if k in own and own[k].shape == w.shape:
            own[k] = w
            loaded += 1
    model.load_state_dict(own)
    return loaded


def _autocast_ctx(device, enabled, dtype):
    """Return an autocast context appropriate to the device."""
    if not enabled:
        return torch.amp.autocast(device_type="cpu", enabled=False)
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def train_loop(model, datagen, b, opt, n_iters, device, *,
               amp=True, amp_dtype=torch.bfloat16, grad_clip=1.0,
               log_every=None, label="", profile=False):
    """Run n_iters of training, return ce loss list and elapsed seconds.

    If `profile`, also accumulates time spent in data-gen vs model forward/backward.
    """
    model.train()
    ce_losses = []
    if log_every is None:
        log_every = max(50, n_iters // 20)

    # Warm-up: discard first 5 iters from timing on GPU (CUDA init / autotune)
    warmup = 5 if device.type == "cuda" else 0

    t0 = None
    iters_timed = 0
    t_data = 0.0
    t_fwd = 0.0
    t_bwd = 0.0

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for it in range(1, n_iters + 1):
        if profile and it > warmup:
            _sync()
            t_a = time.time()
        z_t, u_t, v_t = datagen.sample(B=datagen_B)

        if profile and it > warmup:
            _sync()
            t_b = time.time()
            t_data += t_b - t_a

        with _autocast_ctx(device, amp, amp_dtype):
            root = model.encode_z(z_t)
            all_logits, all_targets, _, _, _ = model.tree(
                z=None, b=b, frozen_u={}, frozen_v={},
                u_true=u_t, v_true=v_t, root_emb=root, distill_alpha=0.0)
            logits = torch.stack(all_logits, dim=1).reshape(-1, 4)
            targets = torch.stack(all_targets, dim=1).reshape(-1)
            ce = F.cross_entropy(logits, targets)

        if profile and it > warmup:
            _sync()
            t_c = time.time()
            t_fwd += t_c - t_b

        opt.zero_grad(set_to_none=True)
        ce.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        ce_losses.append(float(ce.item()))

        if profile and it > warmup:
            _sync()
            t_d = time.time()
            t_bwd += t_d - t_c

        if it == warmup + 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            iters_timed = 0
        elif it > warmup + 1:
            iters_timed += 1

        if it % log_every == 0:
            avg = float(np.mean(ce_losses[-200:]))
            if t0 is not None and iters_timed > 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                rate = iters_timed / (time.time() - t0)
            else:
                rate = float("nan")
            print(f"  {label}iter {it}/{n_iters}  ce={avg:.4f}  rate={rate:.1f} it/s", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    if t0 is not None and iters_timed > 0:
        elapsed = time.time() - t0
        rate = iters_timed / elapsed
    else:
        rate = float("nan")
        elapsed = float("nan")
    if profile and iters_timed > 0:
        tot = t_data + t_fwd + t_bwd
        print(f"  profile: data={t_data/iters_timed*1000:.1f}ms ({t_data/tot*100:.0f}%)"
              f"  fwd={t_fwd/iters_timed*1000:.1f}ms ({t_fwd/tot*100:.0f}%)"
              f"  bwd={t_bwd/iters_timed*1000:.1f}ms ({t_bwd/tot*100:.0f}%)", flush=True)
    return ce_losses, rate, elapsed


# Module-level so train_loop can pick it up via closure (set in main)
datagen_B = 32


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation (target rate, frozen sets)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_bler(model, N, Au, Av, fu, fv, b, datagen_gpu, n_cw, device,
              batch_eval=128):
    """
    Batched BLER eval at target rate. Generates frozen-aware (u,v) on GPU.

    Uses inference mode (no autocast — eval is cheap).
    """
    model.eval()
    Au_t = torch.tensor([p - 1 for p in Au], device=device, dtype=torch.long)
    Av_t = torch.tensor([p - 1 for p in Av], device=device, dtype=torch.long)
    fu_pos = torch.tensor([p - 1 for p in fu.keys()], device=device, dtype=torch.long) if fu else None
    fv_pos = torch.tensor([p - 1 for p in fv.keys()], device=device, dtype=torch.long) if fv else None
    fu_val = torch.tensor([int(v) for v in fu.values()], device=device, dtype=torch.int64) if fu else None
    fv_val = torch.tensor([int(v) for v in fv.values()], device=device, dtype=torch.int64) if fv else None

    errs = 0
    total = 0
    t0 = time.time()
    # Need per-codeword decoder pass because the SC loop has data-dependent
    # frozen bit branching inside. We batch the channel sim but call the
    # model with B=batch_eval at once (the inner loop is already batched).
    while total < n_cw:
        B = min(batch_eval, n_cw - total)
        u = torch.zeros((B, N), device=device, dtype=torch.int64)
        v = torch.zeros((B, N), device=device, dtype=torch.int64)
        # Random info bits at info positions
        info_u = torch.randint(0, 2, (B, len(Au)), device=device, dtype=torch.int64)
        info_v = torch.randint(0, 2, (B, len(Av)), device=device, dtype=torch.int64)
        u[:, Au_t] = info_u
        v[:, Av_t] = info_v
        if fu_pos is not None:
            u[:, fu_pos] = fu_val.unsqueeze(0).expand(B, -1)
        if fv_pos is not None:
            v[:, fv_pos] = fv_val.unsqueeze(0).expand(B, -1)
        # Encode + channel
        x = datagen_gpu._polar_encode(u)
        y = datagen_gpu._polar_encode(v)
        sx = 1.0 - 2.0 * x.float()
        sy = 1.0 - 2.0 * y.float()
        ones_col = torch.ones((B, 1), device=device)
        sx_prev = torch.cat([ones_col, sx[:, :-1]], dim=1)
        sy_prev = torch.cat([ones_col, sy[:, :-1]], dim=1)
        mu = sx + sy + datagen_gpu.h * (sx_prev + sy_prev)
        w = torch.randn((B, N), device=device) * datagen_gpu.sigma
        z = mu + w

        _, _, u_hat, v_hat, _ = model(z, b, frozen_u=fu, frozen_v=fv)
        # u_hat, v_hat are dicts {1..N: tensor(B,)}; gather at Au, Av
        u_hat_at_Au = torch.stack([u_hat[p] for p in Au], dim=1).long()  # (B, |Au|)
        v_hat_at_Av = torch.stack([v_hat[p] for p in Av], dim=1).long()
        # True bits at info positions
        u_true_at_Au = info_u
        v_true_at_Av = info_v
        u_err = (u_hat_at_Au != u_true_at_Au).any(dim=1)
        v_err = (v_hat_at_Av != v_true_at_Av).any(dim=1)
        err = (u_err | v_err)
        errs += int(err.sum().item())
        total += B
        if total % max(1, n_cw // 5) < B:
            print(f"    eval {total}/{n_cw}  errs={errs}  ({(time.time()-t0)/60:.1f}min)", flush=True)
    bler = errs / n_cw
    return bler, errs


# ─────────────────────────────────────────────────────────────────────────────
#  Config defaults
# ─────────────────────────────────────────────────────────────────────────────

# (N, ku, kv, n, default_iters, default_eval_cw)
CURRICULUM_DEFAULT = [
    (16,   4,   7,  4,  50_000, 5000),
    (32,   7,  15,  5,  80_000, 5000),
    (64,  15,  29,  6, 100_000, 5000),
    (128, 30,  58,  7, 120_000, 3000),
    (256, 61, 118,  8, 150_000, 2000),
    (512,123, 237,  9, 200_000, 1500),
    (1024,246,474, 10, 250_000, 1000),
]

DEFAULT_SIGMA2 = 10 ** (-0.6)
DEFAULT_H = 0.3
SNR_DB_FOR_DESIGN = 6


def _resolve_design(N, ku, kv, n):
    gmac_file = os.path.join(
        REPO_ROOT, "designs", f"gmac_C_n{n}_snr{SNR_DB_FOR_DESIGN}dB.npz"
    )
    Au, Av, fu_set, fv_set, _, _, _ = design_from_file(gmac_file, n, ku, kv)
    Au = [int(p) for p in Au]
    Av = [int(p) for p in Av]
    fu = {int(p): 0 for p in fu_set}
    fv = {int(p): 0 for p in fv_set}
    return Au, Av, fu, fv


def _pick_device(force=None):
    if force is not None:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry points
# ─────────────────────────────────────────────────────────────────────────────

def cmd_bench(args):
    """Compare baseline (numpy data-gen, fp32) vs optimized (gpu data-gen, amp)."""
    global datagen_B
    device = _pick_device(args.device)
    print(f"[bench] device={device}  N={args.N}  iters={args.iters}  batch={args.batch}", flush=True)

    # Find n, ku, kv from curriculum for the requested N
    cfg = next((c for c in CURRICULUM_DEFAULT if c[0] == args.N), None)
    if cfg is None:
        raise ValueError(f"No curriculum entry for N={args.N}")
    N, ku, kv, n, _, _ = cfg
    b_path = make_path(N, N)

    if args.baseline:
        print("  mode: BASELINE (numpy data-gen, fp32)", flush=True)
        datagen = NumpyDataGen(N=N, sigma2=DEFAULT_SIGMA2, h=DEFAULT_H, device=device)
        amp_on = False
    else:
        # AMP only meaningful on cuda. CPU autocast(bf16) hurts perf because
        # most CPU kernels don't have bf16 paths and fall back to fp32 + cast.
        amp_on = device.type == "cuda" and not args.no_amp
        print(f"  mode: OPTIMIZED (gpu data-gen, amp={amp_on})", flush=True)
        datagen = GPUDataGen(N=N, sigma2=DEFAULT_SIGMA2, h=DEFAULT_H, device=device)

    datagen_B = args.batch
    model = _make_model(device)
    if args.warm_start_ckpt:
        loaded = _warm_start(model, args.warm_start_ckpt)
        print(f"  warm-started {loaded} tensors from {args.warm_start_ckpt}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.bfloat16
    ce_losses, rate, elapsed = train_loop(
        model, datagen, b_path, opt, args.iters, device,
        amp=amp_on, amp_dtype=amp_dtype, grad_clip=1.0,
        log_every=max(50, args.iters // 10),
        label=f"N={N} ", profile=args.profile)

    final = float(np.mean(ce_losses[-200:])) if ce_losses else float("nan")
    print(f"\n[bench RESULT] N={N} mode={'baseline' if args.baseline else 'optimized'}", flush=True)
    print(f"  iters/s = {rate:.2f}", flush=True)
    print(f"  iters/min = {rate*60:.1f}", flush=True)
    print(f"  final ce (last 200) = {final:.4f}", flush=True)
    print(f"  elapsed = {elapsed:.1f}s for ~{args.iters} iters", flush=True)
    if args.bench_out:
        rec = dict(N=N, batch=args.batch, iters=args.iters, mode='baseline' if args.baseline else 'optimized',
                   device=str(device), rate_it_s=rate, final_ce=final,
                   amp=amp_on)
        try:
            prev = json.load(open(args.bench_out))
        except Exception:
            prev = []
        prev.append(rec)
        json.dump(prev, open(args.bench_out, "w"), indent=2)
        print(f"  wrote {args.bench_out}", flush=True)


def cmd_verify(args):
    """Short-train from existing checkpoint, then eval BLER at target rate."""
    global datagen_B
    device = _pick_device(args.device)
    cfg = next((c for c in CURRICULUM_DEFAULT if c[0] == args.N), None)
    if cfg is None:
        raise ValueError(f"No curriculum entry for N={args.N}")
    N, ku, kv, n, _, default_eval = cfg
    n_cw = args.eval_cw if args.eval_cw else default_eval

    b_path = make_path(N, N)
    Au, Av, fu, fv = _resolve_design(N, ku, kv, n)

    print(f"[verify] device={device}  N={N}  iters={args.iters}  eval_cw={n_cw}", flush=True)

    datagen = GPUDataGen(N=N, sigma2=DEFAULT_SIGMA2, h=DEFAULT_H, device=device)
    datagen_B = args.batch
    model = _make_model(device)

    ckpt_path = args.warm_start_ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(REPO_ROOT, "scripts", "local_analysis",
                                 "ncg_models", f"ncg_isi_N{N}.pt")
    loaded = _warm_start(model, ckpt_path)
    print(f"  warm-started {loaded} tensors from {ckpt_path}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_on = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16
    ce_losses, rate, _ = train_loop(
        model, datagen, b_path, opt, args.iters, device,
        amp=amp_on, amp_dtype=amp_dtype, grad_clip=1.0,
        log_every=max(50, args.iters // 10), label=f"N={N} ")
    final_ce = float(np.mean(ce_losses[-200:])) if ce_losses else float("nan")
    print(f"  trained {args.iters} iters @ {rate:.1f} it/s, final ce={final_ce:.4f}", flush=True)

    print(f"  evaluating BLER at target rate ({n_cw} CW)...", flush=True)
    bler, errs = eval_bler(model, N, Au, Av, fu, fv, b_path, datagen,
                           n_cw, device, batch_eval=args.eval_batch)
    print(f"\n[verify RESULT] N={N}  BLER = {bler:.4f}  ({errs}/{n_cw})", flush=True)
    return bler


def cmd_curriculum(args):
    """Full N=16 → ... → 1024 chain. Mainly intended for cluster GPU."""
    global datagen_B
    device = _pick_device(args.device)
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    results_file = os.path.join(ckpt_dir, "ncg_chain_results.json")
    try:
        results = json.load(open(results_file))
    except Exception:
        results = {}

    # Allow override of iters via CLI
    iters_overrides = {}
    if args.iters_csv:
        for kv in args.iters_csv.split(","):
            k, v = kv.split(":")
            iters_overrides[int(k)] = int(v)

    n_max = args.n_max if args.n_max else 1024
    configs = [c for c in CURRICULUM_DEFAULT if c[0] <= n_max]
    if args.n_min:
        configs = [c for c in configs if c[0] >= args.n_min]

    prev_state = None
    # If we're resuming and earlier entries exist, load prev state from the last completed
    for c in CURRICULUM_DEFAULT:
        key = f"N{c[0]}"
        if key in results and results[key].get("done"):
            prev_path = os.path.join(ckpt_dir, f"ncg_isi_{key}.pt")
            if os.path.exists(prev_path):
                prev_state = torch.load(prev_path, map_location="cpu", weights_only=False).get("state_dict")

    for cfg in configs:
        N, ku, kv, n, default_iters, n_cw = cfg
        n_iters = iters_overrides.get(N, default_iters)
        key = f"N{N}"
        ck_path = os.path.join(ckpt_dir, f"ncg_isi_{key}.pt")
        if key in results and results[key].get("done") and not args.force:
            print(f"  {key} done, skipping. BLER={results[key]['bler']:.4f}", flush=True)
            prev_state = torch.load(ck_path, map_location="cpu", weights_only=False)["state_dict"]
            continue

        b_path = make_path(N, N)
        Au, Av, fu, fv = _resolve_design(N, ku, kv, n)
        print(f"\n=== Training NCG ISI N={N}, ku={ku}, kv={kv} ({n_iters} iters, batch={args.batch}) ===", flush=True)

        datagen = GPUDataGen(N=N, sigma2=DEFAULT_SIGMA2, h=DEFAULT_H, device=device)
        datagen_B = args.batch
        model = _make_model(device)
        if prev_state is not None:
            own = model.state_dict()
            loaded = 0
            for k, w in prev_state.items():
                if k in own and own[k].shape == w.shape:
                    own[k] = w
                    loaded += 1
            model.load_state_dict(own)
            print(f"  warm-started: {loaded} tensors from prev model", flush=True)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        amp_on = device.type == "cuda" and not args.no_amp
        ce_losses, rate, elapsed = train_loop(
            model, datagen, b_path, opt, n_iters, device,
            amp=amp_on, amp_dtype=torch.bfloat16, grad_clip=1.0,
            log_every=max(1, n_iters // 50), label=f"N={N} ")
        print(f"  trained in {elapsed/60:.1f}min @ {rate:.1f} it/s", flush=True)

        print(f"  evaluating BLER ({n_cw} CW)...", flush=True)
        bler, errs = eval_bler(model, N, Au, Av, fu, fv, b_path, datagen,
                               n_cw, device, batch_eval=args.eval_batch)
        print(f"  >>> N={N}  NCG BLER = {bler:.4f}  ({errs}/{n_cw})", flush=True)
        results[key] = dict(N=N, ku=ku, kv=kv, bler=bler, errs=errs, n_cw=n_cw,
                            iters=n_iters, done=True, rate_it_s=rate,
                            elapsed_min=elapsed / 60)
        json.dump(results, open(results_file, "w"), indent=2)
        torch.save({"state_dict": model.state_dict(),
                    "N": N, "ku": ku, "kv": kv}, ck_path)
        print(f"  saved {ck_path}", flush=True)
        prev_state = model.state_dict()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", action="store_true", help="Run benchmark")
    p.add_argument("--verify", action="store_true", help="Verify (short train + eval)")
    p.add_argument("--curriculum", action="store_true", help="Full curriculum chain")
    p.add_argument("--self-test", action="store_true", help="Encoder self-test")
    p.add_argument("--baseline", action="store_true", help="Bench: use numpy data-gen baseline")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=None, help="cuda|cpu (default auto)")
    p.add_argument("--warm-start-ckpt", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str,
                   default=os.path.join(REPO_ROOT, "scripts", "training", "ncg_models_gpu"))
    p.add_argument("--n-min", type=int, default=None)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--iters-csv", type=str, default=None,
                   help="Override iters per N as 'N:iters,N:iters,...'")
    p.add_argument("--eval-cw", type=int, default=None)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--force", action="store_true", help="Re-train even if done")
    p.add_argument("--bench-out", type=str, default=None,
                   help="Append benchmark result as JSON to this file")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable autocast even on CUDA")
    p.add_argument("--profile", action="store_true",
                   help="Per-iter breakdown: data-gen / fwd / bwd")
    args = p.parse_args()

    if args.batch is None:
        # Default batch: 32 on cpu, 512 on cuda
        device = _pick_device(args.device)
        args.batch = 32 if device.type == "cpu" else 512

    if args.self_test:
        _self_test_encoder()
        print("Encoder self-test: OK", flush=True)
        return

    if args.bench:
        cmd_bench(args)
    elif args.verify:
        cmd_verify(args)
    elif args.curriculum:
        cmd_curriculum(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
