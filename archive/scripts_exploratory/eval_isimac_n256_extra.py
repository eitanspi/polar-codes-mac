#!/usr/bin/env python3
"""
ISI-MAC N=256 SC extra eval: run chained trellis SC with 5K more CW.
Uses polar.channels_memory.ISIMAC (1-tap, h=0.3).
"""
import os, sys, math, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(4)

from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.encoder import polar_encode_batch
from polar.decoder_trellis_mac_chained import decode_chained

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
H_TAP = 0.3
N_CW = 5000

CONFIGS = {
    256: {'ku': 59, 'kv': 117},
}


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(BASE, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list); Av = sorted(Av_list)
    fu = {p: 0 for p in range(1, N+1) if p not in Au}
    fv = {p: 0 for p in range(1, N+1) if p not in Av}
    return Au, Av, fu, fv


def main():
    results = {}
    ch = ISIMAC(sigma2=SIGMA2, h=H_TAP)

    for N, cfg in CONFIGS.items():
        ku, kv = cfg['ku'], cfg['kv']
        Au, Av, fu, fv = load_design(N, ku, kv)
        b = make_path(N, N)

        print(f"\n{'='*60}", flush=True)
        print(f"  ISI-MAC chained SC: N={N}, ku={ku}, kv={kv}, {N_CW} CW", flush=True)
        print(f"{'='*60}", flush=True)

        rng = np.random.default_rng(200)
        np.random.seed(200)
        errs_u = errs_v = errs_total = 0
        t0 = time.time()

        for cw in range(N_CW):
            u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
            for p in Au: u[p-1] = rng.integers(0, 2)
            for p in Av: v[p-1] = rng.integers(0, 2)
            x = polar_encode_batch(u[None, :])[0]
            y = polar_encode_batch(v[None, :])[0]
            z = ch.sample_batch(x[None, :].astype(int), y[None, :].astype(int))
            z_arr = np.asarray(z, dtype=np.float64)
            if z_arr.ndim == 2: z_arr = z_arr[0]
            u_hat, v_hat = decode_chained(z_arr, N, fu, fv, ch)
            ue = any(int(u_hat[p-1]) != int(u[p-1]) for p in Au)
            ve = any(int(v_hat[p-1]) != int(v[p-1]) for p in Av)
            if ue: errs_u += 1
            if ve: errs_v += 1
            if ue or ve: errs_total += 1
            if (cw+1) % 500 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  [{cw+1}/{N_CW}] errs={errs_total} ({elapsed:.1f}min)", flush=True)

        elapsed = (time.time() - t0) / 60
        bler = errs_total / N_CW
        results[str(N)] = {
            'N': N, 'ku': ku, 'kv': kv,
            'bler': bler, 'errs_total': errs_total,
            'errs_u': errs_u, 'errs_v': errs_v,
            'n_cw': N_CW, 'time_min': round(elapsed, 1),
        }
        print(f"  BLER={bler:.4f} ({errs_total}/{N_CW}) [{elapsed:.1f}min]", flush=True)

    out = os.path.join(BASE, 'results', 'isimac_n256_extra_sc.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}", flush=True)


if __name__ == '__main__':
    main()
