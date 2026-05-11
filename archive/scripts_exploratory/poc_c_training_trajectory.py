#!/usr/bin/env python3
"""
POC-C: Training trajectory. For checkpoints iter50k, iter100k, iter200k, iter300k,
compute teacher-forced MI per position. Does MI plateau or keep improving?
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch

torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 32
SIGMA2 = 10 ** (-6.0 / 10)
CKPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd',
                        'results', 'ncg_r1_32')
CHECKPOINTS = ['iter50000.pt', 'iter100000.pt', 'iter200000.pt',
               'iter250000.pt', 'iter280000.pt', 'iter290000.pt', 'iter300000.pt']
N_SAMPLES = 2000
BATCH = 256
SEED = 42


def binary_entropy(p):
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def measure_teacher(model, ch, b, n_samples, batch, device, seed):
    rng = np.random.default_rng(seed)
    h_u = np.zeros(N); cnt_u = np.zeros(N)
    h_v = np.zeros(N); cnt_v = np.zeros(N)
    done = 0
    while done < n_samples:
        B = min(batch, n_samples - done)
        u = rng.integers(0, 2, (B, N)).astype(int)
        v = rng.integers(0, 2, (B, N)).astype(int)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float().to(device)
        ut = torch.from_numpy(u).float().to(device)
        vt = torch.from_numpy(v).float().to(device)
        with torch.no_grad():
            L, _, _, _, _ = model(z, b, {}, {}, u_true=ut, v_true=vt)
        idx_u = idx_v = 0
        for step in range(2 * N):
            gamma = b[step]
            lg = L[step]
            if gamma == 0:
                lp0 = torch.logsumexp(lg[:, :2], 1)
                lp1 = torch.logsumexp(lg[:, 2:], 1)
            else:
                lp0 = torch.logsumexp(lg[:, [0, 2]], 1)
                lp1 = torch.logsumexp(lg[:, [1, 3]], 1)
            p0 = torch.sigmoid(lp0 - lp1).cpu().numpy()
            h = binary_entropy(p0).sum()
            if gamma == 0:
                idx_u += 1
                h_u[idx_u - 1] += h; cnt_u[idx_u - 1] += B
            else:
                idx_v += 1
                h_v[idx_v - 1] += h; cnt_v[idx_v - 1] += B
        done += B
    mi_u = 1.0 - h_u / np.maximum(cnt_u, 1)
    mi_v = 1.0 - h_v / np.maximum(cnt_v, 1)
    return mi_u, mi_v


def main():
    device = torch.device('cpu')
    ch = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)

    results = {}
    for ckpt in CHECKPOINTS:
        model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2).to(device)
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, ckpt),
                                         map_location=device, weights_only=True))
        model.eval()
        t0 = time.time()
        mi_u, mi_v = measure_teacher(model, ch, b, N_SAMPLES, BATCH, device, SEED)
        results[ckpt] = (mi_u, mi_v)
        print(f'{ckpt}: avg MI_U={mi_u.mean():.4f}  avg MI_V={mi_v.mean():.4f}  '
              f'sum={(mi_u+mi_v).mean():.4f}  ({time.time()-t0:.1f}s)')

    print()
    print('Per-checkpoint summary table:')
    print(f'{"checkpoint":<15} {"avg MI_U":<10} {"avg MI_V":<10} {"avg sum":<10} '
          f'{"max MI_U":<10} {"max MI_V":<10}')
    for ckpt in CHECKPOINTS:
        mi_u, mi_v = results[ckpt]
        print(f'{ckpt:<15} {mi_u.mean():<10.4f} {mi_v.mean():<10.4f} '
              f'{(mi_u+mi_v).mean():<10.4f} {mi_u.max():<10.4f} {mi_v.max():<10.4f}')


if __name__ == '__main__':
    main()
