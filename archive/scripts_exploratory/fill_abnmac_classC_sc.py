#!/usr/bin/env python3
"""Fill ABNMAC Class C SC baselines at N=16, 256, 512, 1024 with 5K CW each."""
import os, sys, time, json, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import ABNMAC
from polar.design import design_abnmac, make_path
from polar.decoder import decode_batch

import torch
torch.set_num_threads(4)

channel = ABNMAC()
CONFIGS = {
    16:   {'ku': 3,   'kv': 6},
    256:  {'ku': 51,  'kv': 102},
    512:  {'ku': 102, 'kv': 205},
    1024: {'ku': 205, 'kv': 410},
}

def run_sc(N, ku, kv, n_cw=5000, batch_sz=50, seed=42):
    n = int(math.log2(N))
    path_i = N  # Class C
    Au, Av, fu, fv, _, _ = design_abnmac(n, ku, kv)
    b = make_path(N, path_i=path_i)
    rng = np.random.default_rng(seed)
    errs_u = errs_v = errs_total = 0
    total = 0
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    t0 = time.time()
    while total < n_cw:
        actual = min(batch_sz, n_cw - total)
        info_u = rng.integers(0, 2, size=(actual, ku))
        info_v = rng.integers(0, 2, size=(actual, kv))
        U = build_message_batch(N, info_u, Au)
        V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U)
        Y = polar_encode_batch(V)
        Z = channel.sample_batch(X, Y)
        results = decode_batch(N, Z.tolist(), b, fu, fv, channel, vectorized=True)
        for i, (u_dec, v_dec) in enumerate(results):
            u_dec_arr = np.array(u_dec)
            v_dec_arr = np.array(v_dec)
            ue = int(np.sum(u_dec_arr[u_info_idx] != info_u[i]))
            ve = int(np.sum(v_dec_arr[v_info_idx] != info_v[i]))
            if ue > 0: errs_u += 1
            if ve > 0: errs_v += 1
            if ue > 0 or ve > 0: errs_total += 1
        total += actual
        if total % 500 == 0:
            elapsed = time.time() - t0
            print(f'  N={N} [{total}/{n_cw}] errs={errs_total} ({elapsed:.1f}s)', flush=True)
    elapsed = time.time() - t0
    bler = errs_total / n_cw
    print(f'  N={N} BLER={bler:.4f} errs={errs_total}/{n_cw} ({elapsed:.1f}s)')
    return {
        'N': N, 'ku': ku, 'kv': kv,
        'bler': bler, 'errs_total': errs_total, 'errs_u': errs_u, 'errs_v': errs_v,
        'n_cw': n_cw, 'time_s': elapsed,
    }

def main():
    results = {}
    for N in [16, 256, 512, 1024]:
        cfg = CONFIGS[N]
        print(f'\n=== N={N} ku={cfg["ku"]} kv={cfg["kv"]} ===')
        # Adjust batch size and CW count for large N
        batch_sz = 50 if N <= 64 else (20 if N <= 256 else (10 if N <= 512 else 5))
        n_cw = 5000
        r = run_sc(N, cfg['ku'], cfg['kv'], n_cw=n_cw, batch_sz=batch_sz)
        results[str(N)] = r
        # Save incrementally
        out = os.path.join(os.path.dirname(__file__), '..', 'results', 'abnmac_classC_sc_fill.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Saved: {out}')

if __name__ == '__main__':
    main()
