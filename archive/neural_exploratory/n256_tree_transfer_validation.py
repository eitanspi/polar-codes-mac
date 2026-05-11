"""
n256_tree_transfer_validation.py

Reproduce + rigorously validate the N=128 -> N=256 tree-transfer experiment.

Steps
-----
1. Build a fresh GmacNeuralCompGraphDecoder(d=16, ...) for N=256.
2. Load the N=128 checkpoint into all matching tree ops, skipping the
   discrete embedding_z. Re-init the (continuous) z_encoder fresh and
   scale the second linear weights by 30x to match pretrained norms.
3. Train for 5000 iterations with Adam(1e-3), batch=32, grad-clip 1.0.
4. Every 500 iters, evaluate with 200 codewords; track BEST and save it
   to saved_models/n256_tree_transfer_best.pt.
5. After training, evaluate the BEST checkpoint with 5000 codewords for
   a reliable BLER number.

Logs to neural/n256_tree_transfer.log via tee-style writes (and stdout).
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_PATH = os.path.join(HERE, 'n256_tree_transfer.log')
BEST_PT = os.path.join(ROOT, 'saved_models', 'n256_tree_transfer_best.pt')

_LOG_FH = open(LOG_PATH, 'w', buffering=1)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _LOG_FH.write(line + "\n")


# ── Configuration ───────────────────────────────────────────────────────────
N = 256
n_bits = 8
ku = 123
kv = 123
SNR_DB = 6.0
sigma2 = 10.0 ** (-SNR_DB / 10.0)
BATCH = 32
N_ITERS = 5000
EVAL_EVERY = 500
EVAL_NCW = 200
FINAL_NCW = 5000
LR = 1e-3
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

log(f"N={N} ku={ku} kv={kv} sigma2={sigma2:.6f} batch={BATCH} iters={N_ITERS}")

# ── Channel + design ────────────────────────────────────────────────────────
channel = GaussianMAC(sigma2=sigma2)
b = make_path(N, N // 2)

design_file = os.path.join(ROOT, 'designs', f'gmac_B_n{n_bits}_snr6dB.npz')
Au, Av, _, _, _, _, _ = design_from_file(design_file, n_bits, ku, kv)
fu_seq = {i: 0 for i in range(1, N + 1) if i not in Au}
fv_seq = {i: 0 for i in range(1, N + 1) if i not in Av}
log(f"|Au|={len(Au)} |Av|={len(Av)} |fu|={len(fu_seq)} |fv|={len(fv_seq)}")

# ── Model: N=256, load N=128 tree ops, fresh z_encoder ──────────────────────
model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
log(f"Total params: {model.count_parameters()}")

ckpt_path = os.path.join(ROOT, 'saved_models', 'ncg_gmac_mlp_N128.pt')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
mapped = {}
for k, v in ckpt.items():
    if k.startswith('tree.embedding_z'):
        continue  # Skip discrete embedding_z from N=128 (BEMAC-flavored)
    mapped[k[5:] if k.startswith('tree.') else k] = v
model_sd = model.state_dict()
loaded = {k: v for k, v in mapped.items()
          if k in model_sd and model_sd[k].shape == v.shape}
model_sd.update(loaded)
model.load_state_dict(model_sd)
log(f"Loaded {len(loaded)}/{len(model_sd)} params from N=128 checkpoint")

# Fresh re-init of z_encoder + scale-up
with torch.no_grad():
    torch.nn.init.xavier_uniform_(model.z_encoder[0].weight)
    torch.nn.init.zeros_(model.z_encoder[0].bias)
    torch.nn.init.xavier_uniform_(model.z_encoder[2].weight)
    torch.nn.init.zeros_(model.z_encoder[2].bias)
    model.z_encoder[2].weight *= 30.0
log("Re-initialised z_encoder fresh, scaled second-layer weight by 30x")

# ── Data generation helpers ─────────────────────────────────────────────────

def gen_batch(B):
    u = torch.zeros(B, N)
    v = torch.zeros(B, N)
    for a in Au:
        u[:, a - 1] = torch.randint(0, 2, (B,)).float()
    for a in Av:
        v[:, a - 1] = torch.randint(0, 2, (B,)).float()
    cu = polar_encode_batch(u.numpy().astype(int))
    cv = polar_encode_batch(v.numpy().astype(int))
    z = torch.from_numpy(channel.sample_batch(cu, cv)).float()
    return u, v, z


def eval_bler(model, n_cw, seed, eval_batch=32):
    """Batched-evaluation BLER. eval_batch decides decoder batch size."""
    model.eval()
    rng = np.random.default_rng(seed)
    errs = 0
    done = 0
    Au_idx = np.array(sorted(Au), dtype=int) - 1  # 0-indexed
    Av_idx = np.array(sorted(Av), dtype=int) - 1
    while done < n_cw:
        B = min(eval_batch, n_cw - done)
        u_b = np.zeros((B, N), dtype=int)
        v_b = np.zeros((B, N), dtype=int)
        u_b[:, Au_idx] = rng.integers(0, 2, size=(B, len(Au_idx)))
        v_b[:, Av_idx] = rng.integers(0, 2, size=(B, len(Av_idx)))
        x_b = polar_encode_batch(u_b)
        y_b = polar_encode_batch(v_b)
        z_b = torch.from_numpy(channel.sample_batch(x_b, y_b)).float()
        with torch.no_grad():
            _, _, uh, vh, _ = model(z_b, b, fu_seq, fv_seq)
        # uh[p] is (B,) tensor; stack info positions to (B, |Au|)
        uh_stack = torch.stack([uh[int(p)] for p in sorted(Au)], dim=1).cpu().numpy().astype(int)
        vh_stack = torch.stack([vh[int(p)] for p in sorted(Av)], dim=1).cpu().numpy().astype(int)
        u_true_info = u_b[:, Au_idx]
        v_true_info = v_b[:, Av_idx]
        u_err = np.any(uh_stack != u_true_info, axis=1)
        v_err = np.any(vh_stack != v_true_info, axis=1)
        errs += int(np.sum(u_err | v_err))
        done += B
    model.train()
    return errs / n_cw


# ── Training ────────────────────────────────────────────────────────────────
opt = torch.optim.Adam(model.parameters(), lr=LR)

best_bler = float('inf')
best_iter = -1
trajectory = []  # list of (iter, bler200)
losses_window = []
t_start = time.time()

log("Starting training...")
model.train()
for it in range(1, N_ITERS + 1):
    u, v, z = gen_batch(BATCH)
    al, at, _, _, _ = model(z, b, fu_seq, fv_seq, u_true=u, v_true=v)
    if al:
        loss = F.cross_entropy(torch.cat(al), torch.cat(at))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses_window.append(loss.item())

    if it % 100 == 0:
        avg_loss = float(np.mean(losses_window[-100:])) if losses_window else float('nan')
        elapsed = time.time() - t_start
        log(f"  iter {it:4d}  loss={avg_loss:.4f}  elapsed={elapsed:.1f}s")

    if it % EVAL_EVERY == 0:
        bler = eval_bler(model, EVAL_NCW, seed=12345 + it, eval_batch=64)
        trajectory.append((it, bler))
        log(f"  >> EVAL @iter {it}: BLER({EVAL_NCW} cw) = {bler:.4f}")
        if bler < best_bler:
            best_bler = bler
            best_iter = it
            torch.save(model.state_dict(), BEST_PT)
            log(f"  ** New best ({bler:.4f}) saved to {BEST_PT}")

log(f"Training complete in {time.time() - t_start:.1f}s")
log(f"Best intermediate BLER: {best_bler:.4f} @ iter {best_iter}")
log(f"Trajectory: {trajectory}")

# ── Final evaluation with 5000 codewords on best checkpoint ─────────────────
log(f"Loading best checkpoint @ iter {best_iter} for final eval...")
model.load_state_dict(torch.load(BEST_PT, map_location='cpu', weights_only=False))

log(f"Final evaluation: {FINAL_NCW} codewords")
t0 = time.time()
final_bler = eval_bler(model, FINAL_NCW, seed=999, eval_batch=64)
log(f"FINAL BLER ({FINAL_NCW} cw) = {final_bler:.5f}  ({time.time() - t0:.1f}s)")

# ── Comparison vs baselines ─────────────────────────────────────────────────
sc_baseline = 0.005
curriculum_baseline = 0.015
log("=" * 60)
log(f"FINAL BLER (N=256, tree-transfer, 5000 cw) = {final_bler:.5f}")
log(f"  vs SC analytical baseline       = {sc_baseline:.5f}")
log(f"  vs curriculum d=16 baseline     = {curriculum_baseline:.5f}")
beats_sc = "YES" if final_bler < sc_baseline else "NO"
beats_curr = "YES" if final_bler < curriculum_baseline else "NO"
log(f"Beats SC?           {beats_sc}")
log(f"Beats curriculum?   {beats_curr}")

# ── Persist final summary in JSON-ish form ──────────────────────────────────
import json
summary = {
    "N": N,
    "ku": ku,
    "kv": kv,
    "snr_db": SNR_DB,
    "iters_trained": N_ITERS,
    "batch": BATCH,
    "best_iter": best_iter,
    "best_bler_200cw": best_bler,
    "final_bler_5000cw": final_bler,
    "trajectory_200cw": trajectory,
    "sc_baseline": sc_baseline,
    "curriculum_d16_baseline": curriculum_baseline,
}
sum_path = os.path.join(HERE, 'n256_tree_transfer_summary.json')
with open(sum_path, 'w') as f:
    json.dump(summary, f, indent=2)
log(f"Summary written to {sum_path}")

_LOG_FH.close()
