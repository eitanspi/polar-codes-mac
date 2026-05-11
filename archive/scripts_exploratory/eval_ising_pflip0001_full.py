#!/usr/bin/env python3
"""
Ising MAC p_flip=0.001: full eval at N=16,32,64,128,256 with 5-10K CW.
Compare trellis SC, memoryless SC, and also check if NPD could be trained.
"""
import os, sys, math, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(4)

from polar.channels_memory_new import IsingMAC
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.encoder import polar_encode_batch
from polar.decoder import decode_single as gmac_decode_single
from polar.decoder_trellis_ising_chained import bler_chained

P_FLIP = 0.001
SIGMA2 = 0.251

CONFIGS = {
    16:  {'ku': 4,  'kv': 7,  'n_cw': 10000},
    32:  {'ku': 7,  'kv': 15, 'n_cw': 10000},
    64:  {'ku': 15, 'kv': 29, 'n_cw': 10000},
    128: {'ku': 30, 'kv': 58, 'n_cw': 5000},
    256: {'ku': 59, 'kv': 117, 'n_cw': 5000},
}

RESULTS_FILE = os.path.join(BASE, 'results', 'ising_pflip0001_full_results.json')


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(BASE, 'designs', f'gmac_C_n{n}_snr6dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list); Av = sorted(Av_list)
    fu = {p: 0 for p in range(1, N+1) if p not in Au}
    fv = {p: 0 for p in range(1, N+1) if p not in Av}
    return Au, Av, fu, fv


def eval_memoryless_sc(N, Au, Av, fu, fv, n_cw, seed=555):
    ch = IsingMAC(sigma2=SIGMA2, p_flip=P_FLIP)
    gmac = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    t0 = time.time()
    for cw in range(n_cw):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
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
        if (cw + 1) % 2000 == 0:
            elapsed = (time.time() - t0) / 60
            print(f'    Memoryless SC N={N} [{cw+1}/{n_cw}] errs={errs_total} ({elapsed:.1f}min)', flush=True)
    elapsed = (time.time() - t0) / 60
    return {
        'bler': errs_total / n_cw, 'errs_total': errs_total,
        'errs_u': errs_u, 'errs_v': errs_v, 'n_cw': n_cw, 'time_min': round(elapsed, 1),
    }


def main():
    results = {}
    t_global = time.time()

    for N in [16, 32, 64, 128, 256]:
        cfg = CONFIGS[N]
        ku, kv, n_cw = cfg['ku'], cfg['kv'], cfg['n_cw']
        Au, Av, fu, fv = load_design(N, ku, kv)

        print(f"\n{'='*60}", flush=True)
        print(f"  Ising MAC p_flip={P_FLIP}, N={N}, ku={ku}, kv={kv}", flush=True)
        print(f"{'='*60}", flush=True)

        # Trellis SC
        ch = IsingMAC(sigma2=SIGMA2, p_flip=P_FLIP)
        print(f"  Trellis SC ({n_cw} CW)...", flush=True)
        t0 = time.time()
        r_trellis = bler_chained(ch, N, fu, fv, Au, Av, n_cw, seed=1001)
        r_trellis['time_min'] = round((time.time() - t0) / 60, 1)
        print(f"  Trellis SC BLER={r_trellis['chained_bler']:.4f} "
              f"({r_trellis['errs_total']}/{n_cw}) [{r_trellis['time_min']:.1f}min]", flush=True)

        # Memoryless SC
        print(f"  Memoryless SC ({n_cw} CW)...", flush=True)
        r_memless = eval_memoryless_sc(N, Au, Av, fu, fv, n_cw, seed=555)
        print(f"  Memoryless SC BLER={r_memless['bler']:.4f} "
              f"({r_memless['errs_total']}/{n_cw}) [{r_memless['time_min']:.1f}min]", flush=True)

        results[str(N)] = {
            'N': N, 'ku': ku, 'kv': kv, 'p_flip': P_FLIP,
            'trellis_sc': r_trellis,
            'memoryless_sc': r_memless,
        }

        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {RESULTS_FILE}", flush=True)

        if (time.time() - t_global) / 3600 > 1.5:
            print("TIME LIMIT (1.5h)", flush=True)
            break

    elapsed_h = (time.time() - t_global) / 3600
    print(f"\n  Done. Total: {elapsed_h:.2f}h", flush=True)


if __name__ == '__main__':
    main()
