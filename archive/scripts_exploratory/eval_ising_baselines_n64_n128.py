#!/usr/bin/env python3
"""Eval Ising MAC trellis SC and memoryless SC baselines at N=64, 128 with 5K CW."""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.channels_memory_new import IsingMAC
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder import decode_single as gmac_decode_single
from polar.encoder import polar_encode_batch

SIGMA2 = 0.251
P_FLIP = 0.1
SNR_DB = 6
BASE = os.path.join(os.path.dirname(__file__), '..')

CONFIGS = {
    64:  {'ku': 15, 'kv': 29},
    128: {'ku': 30, 'kv': 58},
}


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
    ch = IsingMAC(sigma2=SIGMA2, p_flip=P_FLIP)
    gmac = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    t0 = time.time()
    for cw in range(n_cw):
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
        if (cw + 1) % 1000 == 0:
            elapsed = (time.time() - t0) / 60
            print(f'  SC [{cw+1}/{n_cw}] errs={errs_total} ({elapsed:.1f}min)', flush=True)
    elapsed = (time.time() - t0) / 60
    return {
        'bler': errs_total / n_cw, 'errs_total': errs_total,
        'errs_u': errs_u, 'errs_v': errs_v, 'n_cw': n_cw, 'time_min': elapsed,
    }


def main():
    # Check if trellis decoder exists for Ising
    try:
        from polar.decoder_trellis_ising_chained import bler_chained
        has_trellis = True
    except ImportError:
        has_trellis = False
        print('WARNING: Trellis Ising decoder not available, memoryless SC only.')

    results = {}
    for N in [64, 128]:
        cfg = CONFIGS[N]
        ku, kv = cfg['ku'], cfg['kv']
        Au, Av, fu, fv = load_design(N, ku, kv)
        print(f'\n{"="*50}')
        print(f'Ising MAC N={N} ku={ku} kv={kv}')
        print(f'{"="*50}')

        entry = {'N': N, 'ku': ku, 'kv': kv}

        if has_trellis:
            print(f'  Trellis SC (5000 CW)...')
            ch = IsingMAC(sigma2=SIGMA2, p_flip=P_FLIP)
            t0 = time.time()
            trellis = bler_chained(ch, N, fu, fv, Au, Av, n_cw=5000, seed=1001)
            elapsed = time.time() - t0
            print(f'  Trellis SC: BLER={trellis["chained_bler"]:.4f} '
                  f'(U={trellis["u_err_rate"]:.4f}, V={trellis["v_err_rate"]:.4f}) '
                  f'({elapsed:.1f}s)')
            entry['trellis_sc'] = trellis

        print(f'  Memoryless SC (5000 CW)...')
        memless = eval_memoryless_sc(N, Au, Av, fu, fv, n_cw=5000)
        print(f'  Memoryless SC: BLER={memless["bler"]:.4f} ({memless["errs_total"]}/{memless["n_cw"]})')
        entry['memoryless_sc'] = memless

        results[str(N)] = entry
        out = os.path.join(BASE, 'results', 'ising_mac_baselines_n64_n128.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Saved: {out}')


if __name__ == '__main__':
    main()
