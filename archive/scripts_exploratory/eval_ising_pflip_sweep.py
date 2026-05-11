#!/usr/bin/env python3
"""
Ising MAC p_flip sweep: try p_flip=0.001 and p_flip=0.05 at N=32
to understand the sensitivity. Also try GMAC B (symmetric) rates instead of C (corner).
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

SIGMA2 = 0.251
N_CW = 3000
RESULTS_FILE = os.path.join(BASE, 'results', 'ising_pflip_sweep_results.json')


def load_design_C(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(BASE, 'designs', f'gmac_C_n{n}_snr6dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list); Av = sorted(Av_list)
    fu = {p: 0 for p in range(1, N+1) if p not in Au}
    fv = {p: 0 for p in range(1, N+1) if p not in Av}
    return Au, Av, fu, fv


def load_design_B(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list); Av = sorted(Av_list)
    fu = {p: 0 for p in range(1, N+1) if p not in Au}
    fv = {p: 0 for p in range(1, N+1) if p not in Av}
    return Au, Av, fu, fv


def main():
    results = {}
    t_global = time.time()

    # Experiment 1: N=32, Class C rates, sweep p_flip
    N = 32; ku = 7; kv = 15
    Au, Av, fu, fv = load_design_C(N, ku, kv)

    for p_flip in [0.001, 0.005, 0.05]:
        ch = IsingMAC(sigma2=SIGMA2, p_flip=p_flip)
        key = f'N{N}_C_pflip{p_flip}'
        print(f"\n=== {key} ===", flush=True)
        r = bler_chained(ch, N, fu, fv, Au, Av, N_CW, seed=42)
        r['time_min'] = round((time.time() - t_global) / 60, 1)
        results[key] = {'p_flip': p_flip, 'N': N, 'ku': ku, 'kv': kv, 'class': 'C', **r}
        print(f"  Trellis SC BLER={r['chained_bler']:.4f} ({r['errs_total']}/{N_CW})", flush=True)

    # Experiment 2: N=32, Class B rates (lower rate), p_flip=0.01
    N = 32; ku_b = 15; kv_b = 15  # symmetric, lower total rate
    b_path = make_path(N, N // 2)
    Au_b, Av_b, fu_b, fv_b = load_design_B(N, ku_b, kv_b)
    for p_flip in [0.01, 0.001]:
        ch = IsingMAC(sigma2=SIGMA2, p_flip=p_flip)
        key = f'N{N}_B_pflip{p_flip}'
        print(f"\n=== {key} ===", flush=True)
        r = bler_chained(ch, N, fu_b, fv_b, Au_b, Av_b, N_CW, seed=42)
        r['time_min'] = round((time.time() - t_global) / 60, 1)
        results[key] = {'p_flip': p_flip, 'N': N, 'ku': ku_b, 'kv': kv_b, 'class': 'B', **r}
        print(f"  Trellis SC BLER={r['chained_bler']:.4f} ({r['errs_total']}/{N_CW})", flush=True)

    # Experiment 3: N=64, p_flip=0.001 Class C
    N = 64; ku = 15; kv = 29
    Au, Av, fu, fv = load_design_C(N, ku, kv)
    for p_flip in [0.001]:
        ch = IsingMAC(sigma2=SIGMA2, p_flip=p_flip)
        key = f'N{N}_C_pflip{p_flip}'
        print(f"\n=== {key} ===", flush=True)
        r = bler_chained(ch, N, fu, fv, Au, Av, N_CW, seed=42)
        results[key] = {'p_flip': p_flip, 'N': N, 'ku': ku, 'kv': kv, 'class': 'C', **r}
        print(f"  Trellis SC BLER={r['chained_bler']:.4f} ({r['errs_total']}/{N_CW})", flush=True)

    # Experiment 4: Try very low rate with N=64: ku=5, kv=10 (half the original)
    N = 64; ku_lo = 8; kv_lo = 15
    Au_lo, Av_lo, fu_lo, fv_lo = load_design_C(N, ku_lo, kv_lo)
    for p_flip in [0.01, 0.1]:
        ch = IsingMAC(sigma2=SIGMA2, p_flip=p_flip)
        key = f'N{N}_C_lowrate_pflip{p_flip}'
        print(f"\n=== {key} (ku={ku_lo},kv={kv_lo}) ===", flush=True)
        r = bler_chained(ch, N, fu_lo, fv_lo, Au_lo, Av_lo, N_CW, seed=42)
        results[key] = {'p_flip': p_flip, 'N': N, 'ku': ku_lo, 'kv': kv_lo, 'class': 'C_lowrate', **r}
        print(f"  Trellis SC BLER={r['chained_bler']:.4f} ({r['errs_total']}/{N_CW})", flush=True)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_FILE}")
    print(f"Total time: {(time.time()-t_global)/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
