#!/usr/bin/env python3
"""Run more CW for GMAC Class C SC with MC design at N=256,512,1024."""
import os, sys, math, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
torch.set_num_threads(4)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_batch

SNR_DB = 6.0
sigma2 = 10 ** (-SNR_DB / 10)
channel = GaussianMAC(sigma2=sigma2)
BASE = os.path.join(os.path.dirname(__file__), '..')

CONFIGS = {
    256:  {'ku': 59,  'kv': 117, 'target_cw': 50000},
    512:  {'ku': 119, 'kv': 233, 'target_cw': 50000},
    1024: {'ku': 238, 'kv': 467, 'target_cw': 20000},
}


def run_sc(N, ku, kv, n_cw, batch_sz=10, seed=42):
    n = int(math.log2(N))
    design_file = os.path.join(BASE, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, ku, kv)
    b = make_path(N, N)
    rng = np.random.default_rng(seed)
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    errs = 0; total = 0
    t0 = time.time()
    while total < n_cw:
        actual = min(batch_sz, n_cw - total)
        info_u = rng.integers(0, 2, size=(actual, ku))
        info_v = rng.integers(0, 2, size=(actual, kv))
        U = build_message_batch(N, info_u, Au)
        V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U); Y = polar_encode_batch(V)
        Z_sig = (1 - 2*X).astype(float) + (1 - 2*Y).astype(float) + np.random.default_rng().standard_normal(X.shape) * np.sqrt(sigma2)
        results = decode_batch(N, Z_sig.tolist(), b, fu, fv, channel, vectorized=True)
        for i, (u_dec, v_dec) in enumerate(results):
            ue = int(np.sum(np.array(u_dec)[u_info_idx] != info_u[i]))
            ve = int(np.sum(np.array(v_dec)[v_info_idx] != info_v[i]))
            if ue > 0 or ve > 0: errs += 1
        total += actual
        if total % 5000 == 0:
            elapsed = time.time() - t0
            bler_est = errs / total
            print(f'  N={N} [{total}/{n_cw}] errs={errs} bler~{bler_est:.5f} ({elapsed:.1f}s)', flush=True)
    elapsed = time.time() - t0
    bler = errs / n_cw
    print(f'  N={N} FINAL: BLER={bler:.6f} ({errs}/{n_cw}) ({elapsed:.1f}s)')
    return {'N': N, 'bler': bler, 'errs': errs, 'n_cw': n_cw, 'time_s': elapsed, 'ku': ku, 'kv': kv}


def main():
    results = {}
    for N in [256, 512, 1024]:
        cfg = CONFIGS[N]
        print(f'\n{"="*50}')
        print(f'GMAC C SC (MC design): N={N} ku={cfg["ku"]} kv={cfg["kv"]} CW={cfg["target_cw"]}')
        print(f'{"="*50}')
        batch_sz = max(2, 20 // max(1, N // 128))
        r = run_sc(N, cfg['ku'], cfg['kv'], cfg['target_cw'], batch_sz=batch_sz)
        results[str(N)] = r
        out = os.path.join(BASE, 'results', 'gmac_classC_sc_highcw.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Saved: {out}')

if __name__ == '__main__':
    main()
