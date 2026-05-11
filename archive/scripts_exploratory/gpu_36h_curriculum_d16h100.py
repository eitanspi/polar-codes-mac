#!/usr/bin/env python3
"""
36-hour GPU curriculum: d=16 h=100 BiGRU chained NPD for ISI-MAC.
Matches the NPD paper's architecture (d=8 h=100 but we use d=16 for MAC).

Curriculum: N=16 → 32 → 64 → 128 → 256 → 512
Each step warm-starts from the previous.
Correct eval: info-only positions, proper Av.
Saves checkpoints every 50K iters.
Anti-idle: continuous training, no pauses.
"""
import sys, os, time, math, json, numpy as np, torch
sys.path.insert(0, ".")
from neural.npd_memory_mac import ChainedNPD_MAC
from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.design_mc import design_from_file

SIGMA2 = 10 ** (-0.6)
device = torch.device("cuda")
ch = ISIMAC(sigma2=SIGMA2, h=0.3)
OUT = "class_c_npd/results/npd_memory_mac"
os.makedirs(OUT, exist_ok=True)
LOG = "/tmp/gpu_36h_curriculum.log"

CURRICULUM = [
    {"N": 16,  "ku": 4,   "kv": 7,   "iters": 100000,   "batch": 64, "lr": 3e-4},
    {"N": 32,  "ku": 7,   "kv": 15,  "iters": 200000,   "batch": 32, "lr": 2e-4},
    {"N": 64,  "ku": 15,  "kv": 29,  "iters": 500000,   "batch": 32, "lr": 1e-4},
    {"N": 128, "ku": 30,  "kv": 58,  "iters": 1000000,  "batch": 16, "lr": 1e-4},
    {"N": 256, "ku": 59,  "kv": 117, "iters": 1000000,  "batch": 8,  "lr": 5e-5},
    {"N": 512, "ku": 119, "kv": 233, "iters": 1000000,  "batch": 4,  "lr": 3e-5},
]

# d=16 h=100 — paper-like architecture for MAC
model = ChainedNPD_MAC(d=16, hidden=100, n_layers=2, encoder_type="bigru", gru_layers=1).to(device)
total_params = sum(p.numel() for p in model.stage1.parameters())
print(f"Architecture: d=16 h=100 BiGRU. Params: {total_params:,}", flush=True)

total_t0 = time.time()
all_results = {}


def correct_eval(model, ch, N, n, ku, kv, Au, Av, n_cw=2000):
    """Correct BLER eval: info-only positions, proper Av, on CPU."""
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    frozen_u_set = set(i - 1 for i in range(1, N + 1) if i not in Au)

    model_cpu = ChainedNPD_MAC(d=16, hidden=100, n_layers=2, encoder_type="bigru", gru_layers=1)
    model_cpu.stage1.load_state_dict(
        {k: v.cpu() for k, v in model.stage1.state_dict().items()})
    model_cpu.eval()

    rng = np.random.default_rng(999)
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            bs = min(16, n_cw - total)
            eu = np.zeros((bs, N), dtype=int)
            ev = rng.integers(0, 2, (bs, N)).astype(int)
            for p in Au:
                eu[:, p - 1] = rng.integers(0, 2, bs)
            for p in Av:
                ev[:, p - 1] = rng.integers(0, 2, bs)
            ex = polar_encode_batch(eu)
            ey = polar_encode_batch(ev)
            ez = torch.from_numpy(ch.sample_batch(ex, ey)).float()
            emb = model_cpu.stage1.encode_channel(ez)
            emb_npd = emb[:, br_t, :]
            u_dec = model_cpu.stage1.tree.decode(emb_npd, frozen_u_set)
            for i in range(bs):
                if any(int(u_dec[i, p - 1].item()) != int(eu[i, p - 1]) for p in Au):
                    errs += 1
            total += bs
    del model_cpu
    return errs / total


for step_i, cfg in enumerate(CURRICULUM):
    N = cfg["N"]
    n = int(math.log2(N))
    ku, kv = cfg["ku"], cfg["kv"]
    ITERS, BATCH, LR = cfg["iters"], cfg["batch"], cfg["lr"]

    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long().to(device)

    design_file = f"designs/gmac_C_n{n}_snr6dB.npz"
    if not os.path.exists(design_file):
        print(f"SKIP N={N}: no design file {design_file}", flush=True)
        continue

    Au_list, Av_list, _, _, _, _, _ = design_from_file(design_file, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)

    print(f"\n{'=' * 60}", flush=True)
    print(f"STEP {step_i + 1}: N={N} ku={ku} kv={kv} iters={ITERS} "
          f"batch={BATCH} lr={LR}", flush=True)
    print(f"{'=' * 60}", flush=True)

    opt = torch.optim.AdamW(model.stage1.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=ITERS, eta_min=LR / 100)
    rng = np.random.default_rng(42 + N)
    t0 = time.time()
    best_loss = 999.0

    for it in range(1, ITERS + 1):
        model.train()
        u = np.zeros((BATCH, N), dtype=int)
        v = np.zeros((BATCH, N), dtype=int)
        for p in Au:
            u[:, p - 1] = rng.integers(0, 2, BATCH)
        for p in Av:
            v[:, p - 1] = rng.integers(0, 2, BATCH)
        x = polar_encode_batch(u)
        y = polar_encode_batch(v)
        z = torch.from_numpy(
            ch.sample_batch(x, y).astype(np.float32)).to(device)

        emb = model.stage1.encode_channel(z)
        emb_npd = emb[:, br_t, :]  # CRITICAL: bit-reverse embeddings
        x_cw_npd = torch.from_numpy(x[:, br]).long().to(device)

        loss = model.stage1.tree.fast_ce(emb_npd, x_cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), 1.0)
        opt.step()
        sched.step()

        # Log every 5% of iters or every 50K, whichever is smaller
        log_every = min(max(ITERS // 20, 1000), 50000)
        if it % log_every == 0:
            elapsed = (time.time() - t0) / 60
            total_elapsed = (time.time() - total_t0) / 60
            print(f"  [{it}/{ITERS}] loss={loss.item():.4f} "
                  f"step={elapsed:.0f}min total={total_elapsed:.0f}min",
                  flush=True)

        # Save checkpoint every 50K
        save_every = 50000
        if it % save_every == 0:
            ckpt_path = os.path.join(
                OUT, f"d16h100_s1_N{N}_iter{it}.pt")
            torch.save(model.stage1.state_dict(), ckpt_path)

    # Save final checkpoint
    final_path = os.path.join(OUT, f"d16h100_s1_N{N}_final.pt")
    torch.save(model.stage1.state_dict(), final_path)

    # CORRECT eval
    step_elapsed = (time.time() - t0) / 60
    print(f"\n  Evaluating N={N} (correct info-only eval)...", flush=True)
    eval_cw = min(2000, max(500, 10000 // N))
    bler = correct_eval(model, ch, N, n, ku, kv, Au, Av, n_cw=eval_cw)
    total_elapsed = (time.time() - total_t0) / 60

    print(f"  N={N} S1 BLER = {bler:.4f} ({int(bler * eval_cw)}/{eval_cw}) | "
          f"step={step_elapsed:.0f}min | total={total_elapsed:.0f}min",
          flush=True)

    all_results[str(N)] = {
        "N": N, "ku": ku, "kv": kv,
        "s1_bler": bler, "iters": ITERS,
        "step_time_min": step_elapsed,
        "total_time_min": total_elapsed,
        "loss_final": float(loss.item()),
        "checkpoint": final_path,
    }

    # Save results incrementally
    with open(os.path.join(OUT, "d16h100_curriculum_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

print(f"\n{'=' * 60}", flush=True)
print(f"CURRICULUM COMPLETE. Total: {(time.time() - total_t0) / 60:.0f}min",
      flush=True)
print(f"{'=' * 60}", flush=True)
print("\nFinal results:", flush=True)
for N_str, r in all_results.items():
    print(f"  N={r['N']:4d}: BLER={r['s1_bler']:.4f} "
          f"loss={r['loss_final']:.4f} ({r['step_time_min']:.0f}min)",
          flush=True)
