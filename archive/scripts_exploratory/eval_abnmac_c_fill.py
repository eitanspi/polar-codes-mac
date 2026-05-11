#!/usr/bin/env python3
"""
ABNMAC Class C SC reliability fill: run more CW at N=512 (57/5K) and N=1024 (35/5K).
"""
import os, sys, math, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(4)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_batch

RESULTS_FILE = os.path.join(BASE, 'results', 'abnmac_c_sc_fill.json')


def load_design(N, ku, kv):
    n = int(math.log2(N))
    design_file = os.path.join(BASE, 'designs', f'abnmac_C_n{n}.npz')
    Au, Av, fu, fv, _, _, path_i = design_from_file(design_file, n, ku, kv)
    b = make_path(N, path_i)
    return sorted(Au), sorted(Av), fu, fv, b


def eval_sc(N, ku, kv, n_cw, seed=42):
    channel = ABNMAC()
    Au, Av, fu, fv, b = load_design(N, ku, kv)
    rng = np.random.default_rng(seed)
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    errs = 0; total = 0
    t0 = time.time()
    batch_sz = max(1, min(10, 128 // N))
    while total < n_cw:
        actual = min(batch_sz, n_cw - total)
        info_u = rng.integers(0, 2, size=(actual, len(Au)))
        info_v = rng.integers(0, 2, size=(actual, len(Av)))
        U = build_message_batch(N, info_u, Au)
        V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U)
        Y = polar_encode_batch(V)
        Z = channel.sample_batch(X, Y)
        results = decode_batch(N, Z.tolist(), b, fu, fv, channel, vectorized=True)
        for i, (u_dec, v_dec) in enumerate(results):
            u_dec_arr = np.array(u_dec); v_dec_arr = np.array(v_dec)
            if np.sum(u_dec_arr[u_info_idx] != info_u[i]) > 0 or \
               np.sum(v_dec_arr[v_info_idx] != info_v[i]) > 0:
                errs += 1
        total += actual
        if total % 2000 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  N={N} [{total}/{n_cw}] errs={errs} ({elapsed:.1f}min)", flush=True)
    elapsed = (time.time() - t0) / 60
    bler = errs / total
    return {'N': N, 'ku': ku, 'kv': kv, 'bler': bler, 'errs': errs, 'n_cw': total, 'time_min': round(elapsed, 1)}


def main():
    results = {}

    # N=512: need 10K more CW (57/5K -> target ~100+ combined)
    print("=== ABNMAC C N=512 SC (10K CW) ===", flush=True)
    r = eval_sc(512, 102, 205, 10000, seed=300)
    results['N512'] = r
    print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']}) [{r['time_min']:.1f}min]", flush=True)

    # N=1024: need 15K more CW (35/5K -> target ~100+ combined)
    print("\n=== ABNMAC C N=1024 SC (15K CW) ===", flush=True)
    r = eval_sc(1024, 205, 410, 15000, seed=301)
    results['N1024'] = r
    print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']}) [{r['time_min']:.1f}min]", flush=True)

    # N=64: 239/5K errs from table — borderline enough, but let's run more
    print("\n=== ABNMAC C N=64 SC (5K CW) ===", flush=True)
    r = eval_sc(64, 13, 26, 5000, seed=302)
    results['N64'] = r
    print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']}) [{r['time_min']:.1f}min]", flush=True)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_FILE}", flush=True)


if __name__ == '__main__':
    main()
