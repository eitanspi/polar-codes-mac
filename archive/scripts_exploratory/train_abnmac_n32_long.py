#!/usr/bin/env python3
"""
train_abnmac_n32_long.py — Long retrain of ABNMAC NCG at N=32.

Prior training stopped at 5K iters with loss stuck at 0.87. This script runs
for 60K iters with cosine LR schedule + periodic evaluation. If loss drops
below 0.4 by 30K iters (confirming learning), we save the model for use in
downstream evaluation; otherwise we document that ABNMAC NCG cannot converge
even with 12x more iterations.

Output: saved_models/ncg_abnmac_N32_long.pt   (only if loss < 0.5)
Log:    /tmp/abnmac_n32_long.log
"""

import os, sys, math, time, json
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

D = 16; HIDDEN = 64; N_LAYERS = 2
BATCH = 32
LR = 3e-4
ITERS = 60_000
N = 32
KU = 10; KV = 10


def encode_abnmac_output(z_tuples):
    result = np.empty(z_tuples.shape, dtype=np.int64)
    for idx in np.ndindex(z_tuples.shape):
        zx, zy = z_tuples[idx]
        result[idx] = zx * 2 + zy
    return result


def evaluate_nn(model, channel, b, Au, Av, fu, fv, n_cw=300, batch_size=25):
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
            zf = channel.sample_batch(xf, yf)
            zf_int = encode_abnmac_output(zf)
            zt = torch.from_numpy(zf_int).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                err = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                      any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if err: errs += 1
            total += actual
    model.train()
    return errs / max(total, 1)


def main():
    channel = ABNMAC()
    n = int(math.log2(N))

    design_file = os.path.join(BASE, 'designs', f'abnmac_B_n{n}.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, KU, KV)
    b = make_path(N, N // 2)

    print(f"\n{'='*78}")
    print(f"  ABNMAC N={N} long retrain — {ITERS} iters, bs={BATCH}, lr={LR}")
    print(f"  ku={KU}, kv={KV}, |Au|={len(Au)}, |Av|={len(Av)}")
    print(f"{'='*78}", flush=True)

    model = PureNeuralCompGraphDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS, vocab_size=4)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=LR * 0.01)

    rng = np.random.default_rng()
    t0 = time.time()
    best_bler = 1.0
    loss_history = []
    best_path = os.path.join(BASE, 'saved_models', f'ncg_abnmac_N{N}_long.pt')

    model.train()
    for it in range(1, ITERS + 1):
        uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        zf_int = encode_abnmac_output(zf)
        zt = torch.from_numpy(zf_int).long()
        ut = torch.from_numpy(uf).float(); vt = torch.from_numpy(vf).float()

        logits, targets, _, _, _ = model(zt, b, fu, fv, u_true=ut, v_true=vt)
        if not logits:
            continue
        loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4),
                               torch.stack(targets).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_history.append(loss.item())

        if it % 2000 == 0:
            bler = evaluate_nn(model, channel, b, Au, Av, fu, fv, 300)
            elapsed = (time.time() - t0) / 60
            recent_loss = float(np.mean(loss_history[-1000:]))
            print(f"  [{it:6d}/{ITERS}]  loss={loss.item():.4f} "
                  f"(1K avg {recent_loss:.4f})  BLER={bler:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  {elapsed:.1f}min",
                  flush=True)
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), best_path)

    total_min = (time.time() - t0) / 60
    final_bler = evaluate_nn(model, channel, b, Au, Av, fu, fv, 1000)
    print(f"\n  Done in {total_min:.1f} min.  Final BLER={final_bler:.4f}  "
          f"Best BLER={best_bler:.4f}", flush=True)
    # SC reference is ~0.04 at this rate
    if final_bler < 0.2:
        print(f"  SUCCESS — model learned ({final_bler:.3f} << random 1.0)")
        torch.save(model.state_dict(), best_path)
    else:
        print(f"  FAILURE — model did not converge (BLER={final_bler:.3f}, random=1.0)")

    out = os.path.join(BASE, 'results', 'crc_scl_expansion', 'abnmac_N32_retrain.json')
    with open(out, 'w') as f:
        json.dump({
            'N': N, 'ku': KU, 'kv': KV, 'iters': ITERS,
            'final_bler': final_bler, 'best_bler': best_bler,
            'time_min': total_min, 'final_loss_1K_avg':
                float(np.mean(loss_history[-1000:])) if loss_history else None,
        }, f, indent=2)


if __name__ == '__main__':
    main()
