#!/usr/bin/env python3
"""
eval_ising_mac_baselines.py
===========================
Compute baselines for Ising MAC:
1. Trellis SC (chained) — uses forward-backward on 2-state Markov trellis
2. Memoryless GMAC SC — ignores memory entirely
3. Joint trellis SC — uses the full (x,y,channel_state) trellis

All at N=16, 32 with 3000+ CW for reliable estimates.
"""
import json
import math
import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.channels_memory_new import IsingMAC
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder import decode_single as gmac_decode_single
from polar.encoder import polar_encode_batch
from polar.decoder_trellis_ising_chained import bler_chained as ising_bler_chained

SIGMA2 = 0.251
P_FLIP = 0.1
SNR_DB = 6

CONFIGS = {
    16: {'ku': 4, 'kv': 7},
    32: {'ku': 7, 'kv': 15},
}

RESULTS_DIR = os.path.join(_ROOT, 'results', 'ising_mac_baselines')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    fu = {p: 0 for p in range(1, N+1) if p not in Au}
    fv = {p: 0 for p in range(1, N+1) if p not in Av}
    return Au, Av, fu, fv


def eval_memoryless_sc(N, Au, Av, fu, fv, n_cw=3000, seed=555):
    """Baseline: memoryless GMAC SC on Ising MAC samples."""
    ch = IsingMAC(sigma2=SIGMA2, p_flip=P_FLIP)
    gmac = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au: u[p-1] = rng.integers(0, 2)
        for p in Av: v[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u[None, :])[0]
        y = polar_encode_batch(v[None, :])[0]
        z = ch.sample_batch(x[None, :].astype(int), y[None, :].astype(int))
        z_vec = np.asarray(z, dtype=np.float64)
        if z_vec.ndim == 2:
            z_vec = z_vec[0]
        u_dec, v_dec = gmac_decode_single(N, z_vec.tolist(), b, fu, fv, gmac, log_domain=True)
        ue = any(u_dec[p-1] != u[p-1] for p in Au)
        ve = any(v_dec[p-1] != v[p-1] for p in Av)
        if ue: errs_u += 1
        if ve: errs_v += 1
        if ue or ve: errs_total += 1
    return {
        'bler_total': errs_total / n_cw,
        'bler_u': errs_u / n_cw,
        'bler_v': errs_v / n_cw,
        'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
    }


def main():
    all_results = {}
    out_json = os.path.join(RESULTS_DIR, 'ising_mac_baselines.json')

    for N in [16, 32]:
        cfg = CONFIGS[N]
        ku, kv = cfg['ku'], cfg['kv']
        Au, Av, fu, fv = load_design(N, ku, kv)
        ch = IsingMAC(sigma2=SIGMA2, p_flip=P_FLIP)

        print(f'\n{"="*50}\nN={N} ku={ku} kv={kv}\n{"="*50}')

        # Trellis SC (chained with Markov FB)
        print(f'  Trellis SC (chained, 5000 CW)...')
        t0 = time.time()
        trellis = ising_bler_chained(ch, N, fu, fv, Au, Av, n_cw=5000, seed=1001)
        t_trellis = time.time() - t0
        print(f'  trellis BLER={trellis["chained_bler"]:.4f} '
              f'(U={trellis["u_err_rate"]:.4f}, V={trellis["v_err_rate"]:.4f}) '
              f'({t_trellis:.1f}s)')

        # Memoryless GMAC SC
        print(f'  Memoryless SC (3000 CW)...')
        t0 = time.time()
        memless = eval_memoryless_sc(N, Au, Av, fu, fv, n_cw=3000, seed=555)
        t_memless = time.time() - t0
        print(f'  memoryless BLER={memless["bler_total"]:.4f} '
              f'(U={memless["bler_u"]:.4f}, V={memless["bler_v"]:.4f}) '
              f'({t_memless:.1f}s)')

        all_results[str(N)] = {
            'N': N, 'ku': ku, 'kv': kv,
            'sigma2': SIGMA2, 'p_flip': P_FLIP,
            'trellis_chained': trellis,
            'memoryless_sc': memless,
        }

        with open(out_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'  Saved: {out_json}')

    # Summary
    print(f'\n{"="*50}\nSummary:\n{"="*50}')
    print(f'{"N":<6}{"Trellis SC":<14}{"Memoryless SC":<16}')
    for Ns, r in all_results.items():
        t = r['trellis_chained']['chained_bler']
        m = r['memoryless_sc']['bler_total']
        print(f'{Ns:<6}{t:<14.4f}{m:<16.4f}')


if __name__ == '__main__':
    main()
