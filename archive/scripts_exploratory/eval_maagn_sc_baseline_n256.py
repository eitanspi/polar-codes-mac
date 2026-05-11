#!/usr/bin/env python3
"""Eval memoryless SC baseline for MA-AGN at N=256, 512 with 5K CW."""
import os, sys, math, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
torch.set_num_threads(4)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.channels_memory_new import MAAGNMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder import decode_single as gmac_decode_single

SNR_DB = 6.0
ALPHA = 0.3
sigma2 = 10 ** (-SNR_DB / 10)

CONFIGS = {
    256: {'ku': 59, 'kv': 117},
    512: {'ku': 119, 'kv': 233},
}

BASE = os.path.join(os.path.dirname(__file__), '..')

def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(BASE, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    fu = {p: 0 for p in range(1, N+1) if p not in Au}
    fv = {p: 0 for p in range(1, N+1) if p not in Av}
    return Au, Av, fu, fv


def eval_memoryless_sc(N, Au, Av, fu, fv, n_cw=5000, seed=555):
    ch = MAAGNMAC(sigma2=sigma2, alpha=ALPHA)
    gmac = GaussianMAC(sigma2=sigma2)
    b = make_path(N, N)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    t0 = time.time()
    for cw_idx in range(n_cw):
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au: u[p-1] = rng.integers(0, 2)
        for p in Av: v[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u[None, :])[0]
        y = polar_encode_batch(v[None, :])[0]
        z = ch.sample_batch(x[None, :].astype(int), y[None, :].astype(int))
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 2: z = z[0]
        u_dec, v_dec = gmac_decode_single(N, z.tolist(), b, fu, fv, gmac, log_domain=True)
        u_wrong = any(u_dec[p-1] != u[p-1] for p in Au)
        v_wrong = any(v_dec[p-1] != v[p-1] for p in Av)
        if u_wrong: errs_u += 1
        if v_wrong: errs_v += 1
        if u_wrong or v_wrong: errs_total += 1
        if (cw_idx + 1) % 500 == 0:
            elapsed = (time.time() - t0) / 60
            print(f'  N={N} [{cw_idx+1}/{n_cw}] errs={errs_total} ({elapsed:.1f}min)', flush=True)
    elapsed = (time.time() - t0) / 60
    bler = errs_total / n_cw
    print(f'  N={N} BLER={bler:.4f} ({errs_total}/{n_cw}) ({elapsed:.1f}min)')
    return {
        'N': N, 'bler': bler, 'errs_total': errs_total,
        'errs_u': errs_u, 'errs_v': errs_v, 'n_cw': n_cw,
        'time_min': elapsed,
    }


def main():
    results = {}
    for N in [256, 512]:
        cfg = CONFIGS[N]
        print(f'\n{"="*50}')
        print(f'MA-AGN memoryless SC: N={N} ku={cfg["ku"]} kv={cfg["kv"]}')
        print(f'{"="*50}')
        Au, Av, fu, fv = load_design(N, cfg['ku'], cfg['kv'])
        r = eval_memoryless_sc(N, Au, Av, fu, fv, n_cw=5000)
        results[str(N)] = r
        out = os.path.join(BASE, 'results', 'maagn_sc_baselines_large_N.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Saved: {out}')

if __name__ == '__main__':
    main()
