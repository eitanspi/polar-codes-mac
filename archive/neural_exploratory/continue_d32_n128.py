#!/usr/bin/env python3
"""Continue d=32 training at N=128 from best checkpoint."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
D = 32; HIDDEN = 128; N_LAYERS = 2; Z_HIDDEN = 64
N = 128; n = 7; ku = 62; kv = 62; sc_bler = 0.016

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG = os.path.join(os.path.dirname(__file__), 'continue_d32_n128.log')

channel = GaussianMAC(sigma2=SIGMA2)
b = make_path(N, N//2)
Au, Av, fu, fv, _, _, _ = design_from_file(
    os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr6dB.npz'), n, ku, kv)
frozen_u = {i: 0 for i in range(1, N+1) if i not in Au}
frozen_v = {i: 0 for i in range(1, N+1) if i not in Av}

model = GmacNeuralCompGraphDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_hidden=Z_HIDDEN)
ckpt = os.path.join(SAVE_DIR, 'd32_30hr_best.pt')
model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False))
print(f'Loaded {ckpt}', flush=True)
print(f'N={N}, d={D}, params={sum(p.numel() for p in model.parameters()):,}', flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
rng = np.random.default_rng(12345)
t0 = time.time(); best = 0.0185  # previous best
ITERS = 60000; BATCH = 4

def log(msg):
    with open(LOG, 'a') as f: f.write(msg + '\n')
    print(msg, flush=True)

log(f'Continuing d=32 N=128 from BLER={best}, {ITERS} iters, batch={BATCH}')

model.train()
for it in range(1, ITERS+1):
    progress = it / ITERS
    lr = 5e-5 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
    for pg in opt.param_groups: pg['lr'] = lr

    uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
    for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
    for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
    xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
    zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

    all_logits, all_targets, _, _, _ = model(
        zf, b, frozen_u, frozen_v,
        u_true=torch.from_numpy(uf).long(), v_true=torch.from_numpy(vf).long())
    loss = F.cross_entropy(torch.cat(all_logits, 0), torch.cat(all_targets, 0))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    if it % 5000 == 0:
        model.eval()
        errs = 0; total = 200; trng = np.random.default_rng(999)
        for _ in range(total):
            u1 = np.zeros((1, N), dtype=int); v1 = np.zeros((1, N), dtype=int)
            for p in Au: u1[:, p-1] = trng.integers(0, 2, 1)
            for p in Av: v1[:, p-1] = trng.integers(0, 2, 1)
            x1 = polar_encode_batch(u1); y1 = polar_encode_batch(v1)
            z1 = torch.from_numpy(channel.sample_batch(x1, y1)).float()
            with torch.no_grad():
                _, _, ud, vd, _ = model(z1, b, frozen_u, frozen_v)
            ue = any(ud[0, p-1].item() != u1[0, p-1] for p in Au)
            ve = any(vd[0, p-1].item() != v1[0, p-1] for p in Av)
            if ue or ve: errs += 1
        bler = errs / total
        improved = ''
        if bler < best:
            best = bler
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'd32_30hr_best.pt'))
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'd32_30hr_N128_best.pt'))
            improved = ' *BEST*'
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'd32_30hr_latest.pt'))
        model.train()
        elapsed = time.time() - t0
        log(f'  [{it:>6}/{ITERS}] loss={loss.item():.4f} BLER={bler:.4f} '
            f'(best={best:.4f}, SC={sc_bler}, ratio={bler/sc_bler:.2f}x) '
            f'{elapsed/60:.0f}min lr={lr:.1e}{improved}')

log(f'\nDONE: best={best:.4f} ratio={best/sc_bler:.2f}x')
