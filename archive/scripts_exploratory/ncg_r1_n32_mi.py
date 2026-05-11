#!/usr/bin/env python3
"""
MI per position for NCG N=32 trained at rate 1 (iter300000.pt).

Teacher-forced forward pass; extract binary posterior per position from 4-class logits,
MI = 1 - H2(p_hat) in bits.
"""
import sys, os, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn.functional as F

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 32
n = 5
SIGMA2 = 10 ** (-6.0 / 10)
PATH_I = N // 2          # same non-corner path used during training
CKPT = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd',
                    'results', 'ncg_r1_32', 'iter300000.pt')
OUT = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd',
                   'results', 'ncg_r1_32', 'mi_per_pos.json')

# MAC capacities (GMAC @ 6 dB)
CAP_XZ = 0.465
CAP_YZ_X = 0.912
CAP_XYZ = 1.376


def binary_entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def main():
    device = torch.device('cpu')
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2).to(device)
    sd = torch.load(CKPT, map_location=device)
    model.load_state_dict(sd)     # this script uses the plain state dict
    model.eval()

    ch = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, PATH_I)

    BATCH = 128
    N_BATCH = 40     # 5120 codewords
    rng = np.random.default_rng(12345)

    h_u = np.zeros(N); cnt_u = np.zeros(N)
    h_v = np.zeros(N); cnt_v = np.zeros(N)

    t0 = time.time()
    for bi in range(N_BATCH):
        u = rng.integers(0, 2, (BATCH, N)).astype(int)
        v = rng.integers(0, 2, (BATCH, N)).astype(int)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float()

        ut = torch.from_numpy(u).float(); vt = torch.from_numpy(v).float()
        with torch.no_grad():
            logits_list, _, _, _, _ = model(z, b, {}, {}, u_true=ut, v_true=vt)

        # Walk the 2N steps; each non-frozen step pops one logits tensor (here: all 2N).
        idx_u = 0; idx_v = 0
        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                idx_u += 1; pos = idx_u
            else:
                idx_v += 1; pos = idx_v

            lg = logits_list[step]                    # (B, 4)
            if gamma == 0:
                log_p0 = torch.logsumexp(lg[:, :2], dim=1)
                log_p1 = torch.logsumexp(lg[:, 2:], dim=1)
            else:
                log_p0 = torch.logsumexp(lg[:, [0, 2]], dim=1)
                log_p1 = torch.logsumexp(lg[:, [1, 3]], dim=1)
            # stable: p0 = sigmoid(log_p0 - log_p1)
            p0 = torch.sigmoid(log_p0 - log_p1).cpu().numpy().astype(np.float64)
            p0 = np.clip(p0, 1e-12, 1 - 1e-12)

            h = binary_entropy(p0).sum()
            if gamma == 0:
                h_u[pos - 1] += h; cnt_u[pos - 1] += BATCH
            else:
                h_v[pos - 1] += h; cnt_v[pos - 1] += BATCH

        if (bi + 1) % 5 == 0:
            print(f'  batch {bi+1}/{N_BATCH}  {time.time()-t0:.1f}s', flush=True)

    mi_u = 1.0 - np.where(cnt_u > 0, h_u / cnt_u, 1.0)
    mi_v = 1.0 - np.where(cnt_v > 0, h_v / cnt_v, 1.0)

    avg_u = float(mi_u.mean()); avg_v = float(mi_v.mean())
    print(f'\nSamples: {BATCH*N_BATCH}')
    print(f'avg MI_U = {avg_u:.4f}   (capacity I(X;Z)    = {CAP_XZ})')
    print(f'avg MI_V = {avg_v:.4f}   (capacity I(Y;Z|X)  = {CAP_YZ_X})')
    print(f'avg sum  = {avg_u+avg_v:.4f}   (capacity I(X,Y;Z) = {CAP_XYZ})')
    print(f'MI_U > 0.5: {(mi_u>0.5).sum()}/{N}')
    print(f'MI_V > 0.5: {(mi_v>0.5).sum()}/{N}')

    # Rank NCG-info positions: highest-MI are "good" (info). Lowest-MI are frozen.
    rank_u = np.argsort(-mi_u)      # descending
    rank_v = np.argsort(-mi_v)

    data = {
        'N': N, 'path_i': PATH_I,
        'n_samples': BATCH * N_BATCH,
        'mi_u': mi_u.tolist(),
        'mi_v': mi_v.tolist(),
        'avg_mi_u': avg_u, 'avg_mi_v': avg_v,
        'rank_u_desc_0indexed': rank_u.tolist(),
        'rank_v_desc_0indexed': rank_v.tolist(),
        'capacities': {'I_XZ': CAP_XZ, 'I_YZ_X': CAP_YZ_X, 'I_XYZ': CAP_XYZ},
    }
    with open(OUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
