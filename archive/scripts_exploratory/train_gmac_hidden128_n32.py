#!/usr/bin/env python3
"""
train_gmac_hidden128_n32.py — Quick test of hidden=128 at N=32 GMAC Class B.

Compare to hidden=64 baseline (BLER ~0.040 at N=32).
If BLER drops below 0.010, the capacity increase helps.
"""

import os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac

N = 32; KU = KV = 15
SNR_DB = 6.0; SIGMA2 = 10 ** (-SNR_DB / 10)
BATCH = 32; LR = 3e-4; ITERS = 30000
EVAL_EVERY = 2000; EVAL_CW = 500


def load_design():
    n = int(math.log2(N))
    d = np.load(os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz'))
    su = np.argsort(d['u_error_rates']); sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:KU]]); Av = sorted([int(i+1) for i in sv[:KV]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def evaluate(model, channel, Au, Av, fu, fv, b, n_cw, batch_size=50):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf).astype(np.float32)).float()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / max(total, 1)


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)

    model = SimpleMLP_Gmac(d=16, hidden=128, n_layers=2, z_hidden=32)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}")
    print(f"  GMAC Class B N={N} — hidden=128 test ({params:,} params)")
    print(f"  Baseline (hidden=64): BLER ~0.040, SC ~0.047")
    print(f"{'='*70}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=LR * 0.01)

    rng = np.random.default_rng()
    t0 = time.time()
    best_bler = 1.0
    best_path = os.path.join(BASE, 'saved_models', f'ncg_gmac_h128_N{N}_best.pt')

    model.train()
    for it in range(1, ITERS + 1):
        uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf).astype(np.float32)).float()
        ut = torch.from_numpy(uf).float(); vt = torch.from_numpy(vf).float()

        logits, targets, _, _, _ = model(zf, b, fu, fv, u_true=ut, v_true=vt)
        if not logits:
            continue
        loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4),
                               torch.stack(targets).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % EVAL_EVERY == 0 or it == ITERS:
            bler = evaluate(model, channel, Au, Av, fu, fv, b, EVAL_CW)
            elapsed = (time.time() - t0) / 60
            improved = ""
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), best_path)
                improved = " *BEST*"
            print(f"  [{it:6d}/{ITERS}] loss={loss.item():.4f} BLER={bler:.4f} "
                  f"best={best_bler:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{elapsed:.1f}min{improved}", flush=True)

    final = evaluate(model, channel, Au, Av, fu, fv, b, 2000)
    print(f"\n  Done. Final BLER={final:.4f} Best={best_bler:.4f} "
          f"[{(time.time()-t0)/60:.1f} min]")
    print(f"  Baseline (hidden=64): ~0.040")
    print(f"  SC reference: ~0.047")
    if best_bler < 0.030:
        print(f"  VERDICT: hidden=128 HELPS ({best_bler:.3f} < 0.040)")
    else:
        print(f"  VERDICT: hidden=128 does NOT help meaningfully ({best_bler:.3f} vs 0.040)")


if __name__ == '__main__':
    main()
