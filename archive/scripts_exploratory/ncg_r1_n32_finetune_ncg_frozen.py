#!/usr/bin/env python3
"""
Phase-2: warm-start from iter300000.pt and fine-tune at rate 0.594 with
the NCG-selected frozen set for 40 minutes wall-clock time.

NCG frozen = SC frozen with two swaps (from stable MI ranking):
  Au: SC={20..32}\{25} + {25}  →  NCG={20..32}\{25} + {19}
  Av: SC={3..16,20,22..24,26..32}   →  NCG={2..16,20,23,24,26..32}
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn.functional as F

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

torch.set_num_threads(2)   # leave cores for the N=256 training

N = 32
SIGMA2 = 10 ** (-6.0 / 10)
BATCH = 32
WALL_SEC = 40 * 60
LR = 2e-5   # smaller than 5e-5 since we're warm-starting

CKPT_IN = 'class_c_npd/results/ncg_r1_32/iter300000.pt'
OUT_DIR = 'class_c_npd/results/ncg_r1_32_ft_ncgfrozen'
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    device = torch.device('cpu')
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT_IN, map_location=device, weights_only=True))

    ch = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)

    # SC design
    Au_sc, Av_sc, _, _, _, _, _ = design_from_file(
        'designs/gmac_B_n5_snr6dB.npz', 5, 13, 25)
    # NCG = SC with swaps
    Au = sorted((set(Au_sc) - {25}) | {19})
    Av = sorted((set(Av_sc) - {22}) | {2})
    fu = {p: 0 for p in range(1, N + 1) if p not in Au}
    fv = {p: 0 for p in range(1, N + 1) if p not in Av}
    print(f'ku={len(Au)} kv={len(Av)}  rate={(len(Au)+len(Av))/(2*N):.3f}')
    print(f'Au={Au}')
    print(f'Av={Av}')

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(2026)

    t0 = time.time()
    it = 0
    loss_accum = 0.0; loss_cnt = 0
    while True:
        it += 1
        # Random info bits, zero at frozen positions
        u = np.zeros((BATCH, N), dtype=int); v = np.zeros((BATCH, N), dtype=int)
        for p in Au: u[:, p - 1] = rng.integers(0, 2, BATCH)
        for p in Av: v[:, p - 1] = rng.integers(0, 2, BATCH)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float()
        ut = torch.from_numpy(u).float(); vt = torch.from_numpy(v).float()

        logits, targets, _, _, _ = model(
            z, b, fu, fv, u_true=ut, v_true=vt)
        if logits:
            loss = F.cross_entropy(
                torch.stack(logits).reshape(-1, 4),
                torch.stack(targets).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_accum += loss.item(); loss_cnt += 1

        if it % 500 == 0:
            avg = loss_accum / max(1, loss_cnt)
            mins = (time.time() - t0) / 60
            print(f'[{it}]  avg_loss_last_500={avg:.4f}  {mins:.1f}min', flush=True)
            loss_accum = 0.0; loss_cnt = 0

        if it % 5000 == 0:
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f'iter{it}.pt'))

        if time.time() - t0 >= WALL_SEC:
            break

    mins = (time.time() - t0) / 60
    torch.save(model.state_dict(), os.path.join(OUT_DIR, 'final.pt'))
    print(f'\nDONE. {it} iters, {mins:.1f} min.  Saved final.pt')


if __name__ == '__main__':
    main()
