#!/usr/bin/env python3
"""
POC-A: Teacher-forced vs Free-running MI per position.
Same pre-trained NCG model, same codewords, two forward passes.
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
CKPT = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd',
                    'results', 'ncg_r1_32', 'iter300000.pt')
N_SAMPLES = 5000
BATCH = 256
SEED = 42


def binary_entropy(p):
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def extract_mi_from_logits(all_logits, b):
    """Accumulate entropy per position. Returns (h_u_list, cnt_u, h_v_list, cnt_v)."""
    h_u = np.zeros(N); cnt_u = np.zeros(N)
    h_v = np.zeros(N); cnt_v = np.zeros(N)
    idx_u = idx_v = 0
    for step in range(2 * N):
        gamma = b[step]
        lg = all_logits[step]
        if gamma == 0:
            lp0 = torch.logsumexp(lg[:, :2], 1)
            lp1 = torch.logsumexp(lg[:, 2:], 1)
        else:
            lp0 = torch.logsumexp(lg[:, [0, 2]], 1)
            lp1 = torch.logsumexp(lg[:, [1, 3]], 1)
        p0 = torch.sigmoid(lp0 - lp1).cpu().numpy()
        h = binary_entropy(p0).sum()
        B = lg.shape[0]
        if gamma == 0:
            idx_u += 1
            h_u[idx_u - 1] += h; cnt_u[idx_u - 1] += B
        else:
            idx_v += 1
            h_v[idx_v - 1] += h; cnt_v[idx_v - 1] += B
    return h_u, cnt_u, h_v, cnt_v


def main():
    device = torch.device('cpu')
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    ch = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)

    rng = np.random.default_rng(SEED)

    # Accumulators for two passes
    h_u_tf = np.zeros(N); cnt_u_tf = np.zeros(N)
    h_v_tf = np.zeros(N); cnt_v_tf = np.zeros(N)
    h_u_fr = np.zeros(N); cnt_u_fr = np.zeros(N)
    h_v_fr = np.zeros(N); cnt_v_fr = np.zeros(N)

    done = 0
    t0 = time.time()
    while done < N_SAMPLES:
        B = min(BATCH, N_SAMPLES - done)
        u = rng.integers(0, 2, (B, N)).astype(int)
        v = rng.integers(0, 2, (B, N)).astype(int)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float().to(device)
        ut = torch.from_numpy(u).float().to(device)
        vt = torch.from_numpy(v).float().to(device)

        # Pass 1: teacher-forced
        with torch.no_grad():
            L_tf, _, _, _, _ = model(z, b, {}, {}, u_true=ut, v_true=vt)
        hu, cu, hv, cv = extract_mi_from_logits(L_tf, b)
        h_u_tf += hu; cnt_u_tf += cu; h_v_tf += hv; cnt_v_tf += cv

        # Pass 2: free-running
        with torch.no_grad():
            L_fr, _, _, _, _ = model(z, b, {}, {})
        hu, cu, hv, cv = extract_mi_from_logits(L_fr, b)
        h_u_fr += hu; cnt_u_fr += cu; h_v_fr += hv; cnt_v_fr += cv

        done += B

    mi_u_tf = 1.0 - h_u_tf / np.maximum(cnt_u_tf, 1)
    mi_v_tf = 1.0 - h_v_tf / np.maximum(cnt_v_tf, 1)
    mi_u_fr = 1.0 - h_u_fr / np.maximum(cnt_u_fr, 1)
    mi_v_fr = 1.0 - h_v_fr / np.maximum(cnt_v_fr, 1)

    gap_u = mi_u_tf - mi_u_fr
    gap_v = mi_v_tf - mi_v_fr

    print(f'POC-A: N=32, samples={N_SAMPLES}, seed={SEED}, elapsed={time.time()-t0:.1f}s')
    print()
    print('Per-position MI (pos 1..32)')
    print('pos  MI_tf_U   MI_fr_U   gap_U    MI_tf_V   MI_fr_V   gap_V')
    for p in range(N):
        print(f'{p+1:3d}  {mi_u_tf[p]:.4f}    {mi_u_fr[p]:.4f}    {gap_u[p]:+.4f}  '
              f'{mi_v_tf[p]:.4f}    {mi_v_fr[p]:.4f}    {gap_v[p]:+.4f}')

    print()
    print('=== Summary ===')
    print(f'avg MI_U teacher: {mi_u_tf.mean():.4f}   avg MI_U free: {mi_u_fr.mean():.4f}')
    print(f'avg MI_V teacher: {mi_v_tf.mean():.4f}   avg MI_V free: {mi_v_fr.mean():.4f}')
    print(f'mean gap U: {gap_u.mean():+.4f}   mean gap V: {gap_v.mean():+.4f}')

    for thr in [0.05, 0.20, 0.50]:
        nu = int((gap_u > thr).sum()); nv = int((gap_v > thr).sum())
        print(f'# positions with gap > {thr:.2f}:  U={nu}/32   V={nv}/32')

    # Top-5 gap positions
    top_u = np.argsort(-gap_u)[:5] + 1
    top_v = np.argsort(-gap_v)[:5] + 1
    print(f'\nTop-5 gap positions U: {top_u.tolist()}  gaps={[f"{gap_u[p-1]:+.4f}" for p in top_u]}')
    print(f'Top-5 gap positions V: {top_v.tolist()}  gaps={[f"{gap_v[p-1]:+.4f}" for p in top_v]}')

    # Path order: steps 0-15 = U pos 1-16, 16-47 = V 1-32, 48-63 = U 17-32
    # Mean gap in decoding order (only information positions, i.e., all positions here since frozen={})
    # U early (pos 1-16, steps 0-15) vs U late (pos 17-32, steps 48-63)
    print(f'\nDecoding-order gap analysis (path = [0]*16 + [1]*32 + [0]*16):')
    print(f'  U early (pos 1-16, steps 0-15)  : mean gap = {gap_u[:16].mean():+.4f}')
    print(f'  V       (pos 1-32, steps 16-47) : mean gap = {gap_v.mean():+.4f}')
    print(f'  U late  (pos 17-32, steps 48-63): mean gap = {gap_u[16:].mean():+.4f}')


if __name__ == '__main__':
    main()
