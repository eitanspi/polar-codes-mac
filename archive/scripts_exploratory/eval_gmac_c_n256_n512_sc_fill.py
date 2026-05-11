#!/usr/bin/env python3
"""
GMAC C SC reliability fill at N=256 (55/50K) and N=512 (19/50K).
These are slow single-CW SC evals.
"""
import os, sys, math, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(4)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder import decode_single

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

RESULTS_FILE = os.path.join(BASE, 'results', 'gmac_c_sc_fill_n256_n512.json')


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(BASE, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list); Av = sorted(Av_list)
    fu = {p: 0 for p in range(1, N+1) if p not in Au}
    fv = {p: 0 for p in range(1, N+1) if p not in Av}
    b = make_path(N, N)
    return Au, Av, fu, fv, b


def eval_sc(N, ku, kv, n_cw, seed=42):
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv, b = load_design(N, ku, kv)
    rng = np.random.default_rng(seed)
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    errs = 0; total = 0
    t0 = time.time()
    for cw in range(n_cw):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
        for p in Au: u[p-1] = rng.integers(0, 2)
        for p in Av: v[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u[None, :])[0]
        y = polar_encode_batch(v[None, :])[0]
        z = channel.sample_batch(x[None, :].astype(int), y[None, :].astype(int))
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.ndim == 2: z_arr = z_arr[0]
        u_dec, v_dec = decode_single(N, z_arr.tolist(), b, fu, fv, channel, log_domain=True)
        u_dec_arr = np.array(u_dec); v_dec_arr = np.array(v_dec)
        if np.sum(u_dec_arr[u_info_idx] != u[u_info_idx]) > 0 or \
           np.sum(v_dec_arr[v_info_idx] != v[v_info_idx]) > 0:
            errs += 1
        total += 1
        if total % 5000 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  N={N} [{total}/{n_cw}] errs={errs} BLER={errs/total:.4f} ({elapsed:.1f}min)", flush=True)
    elapsed = (time.time() - t0) / 60
    bler = errs / total
    return {'N': N, 'ku': ku, 'kv': kv, 'bler': bler, 'errs': errs, 'n_cw': total, 'time_min': round(elapsed, 1)}


def main():
    results = {}
    t0 = time.time()

    # N=256: 55/50K -> run 50K more to get ~110 total
    print("=== GMAC C N=256 SC (50K CW) ===", flush=True)
    r = eval_sc(256, 59, 117, 50000, seed=400)
    results['N256'] = r
    print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']}) [{r['time_min']:.1f}min]", flush=True)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {RESULTS_FILE}", flush=True)

    # Only do N=512 if we have time
    if (time.time() - t0) / 3600 < 3.0:
        print("\n=== GMAC C N=512 SC (50K CW) ===", flush=True)
        r = eval_sc(512, 119, 233, 50000, seed=401)
        results['N512'] = r
        print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']}) [{r['time_min']:.1f}min]", flush=True)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"Total: {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
