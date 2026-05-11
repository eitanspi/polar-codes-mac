#!/usr/bin/env python3
"""ABNMAC B N=256 SC fill: 98/2K -> need 3K more CW."""
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


def main():
    N = 256; ku = 102; kv = 102
    n = int(math.log2(N))
    channel = ABNMAC()
    design_file = os.path.join(BASE, 'designs', f'abnmac_B_n{n}.npz')
    Au, Av, fu, fv, _, _, path_i = design_from_file(design_file, n, ku, kv)
    Au = sorted(Au); Av = sorted(Av)
    b = make_path(N, path_i)
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])

    n_cw = 5000
    rng = np.random.default_rng(700)
    errs = 0; total = 0
    t0 = time.time()
    batch_sz = 5
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
        if total % 1000 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  [{total}/{n_cw}] errs={errs} ({elapsed:.1f}min)", flush=True)

    elapsed = (time.time() - t0) / 60
    bler = errs / total
    result = {'N': N, 'ku': ku, 'kv': kv, 'bler': bler, 'errs': errs, 'n_cw': total, 'time_min': round(elapsed, 1)}
    print(f"  BLER={bler:.4f} ({errs}/{total}) [{elapsed:.1f}min]", flush=True)

    out = os.path.join(BASE, 'results', 'abnmac_b_n256_sc_fill.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out}", flush=True)


if __name__ == '__main__':
    main()
