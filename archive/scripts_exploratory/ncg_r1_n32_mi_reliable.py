#!/usr/bin/env python3
"""
Reliable per-position MI for NCG N=32 rate-1 (iter300000.pt).
  - 200K samples per seed
  - 3 seeds
  - Report stability of the ranking (top-13 U, top-25 V)
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 32
SIGMA2 = 10 ** (-6.0 / 10)
CKPT = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd',
                    'results', 'ncg_r1_32', 'iter300000.pt')
OUT = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd',
                   'results', 'ncg_r1_32', 'mi_per_pos_reliable.json')

# Capacities at 6 dB
CAP_XZ = 0.465; CAP_YZ_X = 0.912; CAP_XYZ = 1.376


def binary_entropy(p):
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def measure(model, ch, b, seed, n_samples, batch, device):
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
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    ch = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)

    SEEDS = [12345, 67890, 54321]
    N_SAMPLES = 200_000
    BATCH = 256

    all_mi_u, all_mi_v = [], []
    for s in SEEDS:
        t0 = time.time()
        mi_u, mi_v = measure(model, ch, b, s, N_SAMPLES, BATCH, device)
        print(f'seed={s}: avg MI_U={mi_u.mean():.4f} avg MI_V={mi_v.mean():.4f}  '
              f'sum={(mi_u+mi_v).mean():.4f}  ({time.time()-t0:.1f}s)')
        all_mi_u.append(mi_u); all_mi_v.append(mi_v)

    mi_u_mean = np.mean(all_mi_u, axis=0); mi_u_std = np.std(all_mi_u, axis=0)
    mi_v_mean = np.mean(all_mi_v, axis=0); mi_v_std = np.std(all_mi_v, axis=0)

    print(f'\nMEAN over {len(SEEDS)} seeds:')
    print(f'  avg MI_U = {mi_u_mean.mean():.4f}  (cap {CAP_XZ})')
    print(f'  avg MI_V = {mi_v_mean.mean():.4f}  (cap {CAP_YZ_X})')
    print(f'  avg sum  = {(mi_u_mean+mi_v_mean).mean():.4f}  (cap {CAP_XYZ})')
    print(f'  Max MI_U std across positions: {mi_u_std.max():.5f}')
    print(f'  Max MI_V std across positions: {mi_v_std.max():.5f}')

    # SC design for reference
    Au_sc, Av_sc, _, _, _, _, _ = design_from_file(
        'designs/gmac_B_n5_snr6dB.npz', 5, 13, 25)
    sc_u = sorted(Au_sc); sc_v = sorted(Av_sc)

    # Top-K ranking per seed
    ku, kv = 13, 25
    ranks_u = []; ranks_v = []
    for s_mi_u, s_mi_v in zip(all_mi_u, all_mi_v):
        ranks_u.append(set(np.argsort(-s_mi_u)[:ku] + 1))
        ranks_v.append(set(np.argsort(-s_mi_v)[:kv] + 1))

    # Stability: intersection over all seeds
    stable_u = ranks_u[0] & ranks_u[1] & ranks_u[2]
    stable_v = ranks_v[0] & ranks_v[1] & ranks_v[2]
    print(f'\nRanking stability (seeds agree):')
    print(f'  U: {len(stable_u)}/{ku} positions agreed across all 3 seeds')
    print(f'  V: {len(stable_v)}/{kv} positions agreed across all 3 seeds')

    # Mean-MI ranking
    ncg_u = sorted(int(x) for x in (np.argsort(-mi_u_mean)[:ku] + 1))
    ncg_v = sorted(int(x) for x in (np.argsort(-mi_v_mean)[:kv] + 1))

    diff_u = sorted(set(ncg_u) ^ set(sc_u))
    diff_v = sorted(set(ncg_v) ^ set(sc_v))
    print(f'\nNCG vs SC top-K (from mean MI):')
    print(f'  |ncg_u ^ sc_u| = {len(diff_u)}  symmetric diff: {diff_u}')
    print(f'  |ncg_v ^ sc_v| = {len(diff_v)}  symmetric diff: {diff_v}')
    print(f'  ncg_u: {ncg_u}')
    print(f'  sc_u : {sc_u}')
    print(f'  ncg_v: {ncg_v}')
    print(f'  sc_v : {sc_v}')

    # Contested positions: show MI mean±std
    contested = set(diff_u) | set(diff_v)
    if contested:
        print(f'\nContested positions (MI mean ± std across seeds):')
        for p in sorted(contested):
            u_vals = [m[p - 1] for m in all_mi_u]
            v_vals = [m[p - 1] for m in all_mi_v]
            print(f'  pos {p:2d}: MI_U = {np.mean(u_vals):.4f} ± {np.std(u_vals):.4f}   '
                  f'MI_V = {np.mean(v_vals):.4f} ± {np.std(v_vals):.4f}')

    data = {
        'N': N, 'path_i': N // 2,
        'seeds': SEEDS, 'n_samples_per_seed': N_SAMPLES,
        'mi_u_per_seed': [m.tolist() for m in all_mi_u],
        'mi_v_per_seed': [m.tolist() for m in all_mi_v],
        'mi_u_mean': mi_u_mean.tolist(), 'mi_v_mean': mi_v_mean.tolist(),
        'mi_u_std': mi_u_std.tolist(),   'mi_v_std': mi_v_std.tolist(),
        'ncg_info_u_top13': ncg_u,
        'ncg_info_v_top25': ncg_v,
        'sc_info_u': sc_u, 'sc_info_v': sc_v,
        'stable_u_across_seeds': sorted(stable_u),
        'stable_v_across_seeds': sorted(stable_v),
    }
    with open(OUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nSaved: {OUT}')


if __name__ == '__main__':
    main()
