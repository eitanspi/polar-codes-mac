#!/usr/bin/env python3
"""
eval_trellis_sc_all_N.py
========================
Priority 2: Run trellis SC with 10K CW at N=16, 32, 64, 128, 256
to get all ISI-MAC baselines reliable (>=100 errors).

Uses polar.decoder_trellis_mac_chained.decode_chained() which runs
2-state forward-backward + single-user SC at each stage.

Saves to results/reliable_evals/isi_mac_sc_10kcw.json
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

from polar.encoder import polar_encode_batch
from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from polar.decoder_trellis_mac_chained import decode_chained

SNR_DB = 6.0
ISI_H = 0.3
RATES = {
    16:  (4, 7),
    32:  (7, 15),
    64:  (15, 29),
    128: (30, 58),
    256: (59, 117),
}

OUT_DIR = os.path.join(_ROOT, 'results', 'reliable_evals')
os.makedirs(OUT_DIR, exist_ok=True)


def make_channel():
    return ISIMAC.from_snr_db(SNR_DB, h=ISI_H)


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, _path_i = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_dict = {p: 0 for p in range(1, N+1) if p not in Au}
    frozen_v_dict = {p: 0 for p in range(1, N+1) if p not in Av}
    return Au, Av, frozen_u_dict, frozen_v_dict


def wilson_ci(k, n, z=1.96):
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    spread = z * math.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


def eval_trellis_sc(channel, N, Au, Av, fu_dict, fv_dict, n_cw=10000, seed=555):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    t0 = time.time()
    for i in range(n_cw):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for p in Au:
            u_msg[p-1] = rng.integers(0, 2)
        for p in Av:
            v_msg[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u_msg[None, :])[0]
        y = polar_encode_batch(v_msg[None, :])[0]
        z = channel.sample_batch(x[None, :].astype(int), y[None, :].astype(int))[0]
        try:
            u_dec, v_dec = decode_chained(z, N, fu_dict, fv_dict, channel)
        except Exception as e:
            print(f'  N={N} iter {i} raised: {e}')
            continue
        u_wrong = any(int(u_dec[p-1]) != int(u_msg[p-1]) for p in Au)
        v_wrong = any(int(v_dec[p-1]) != int(v_msg[p-1]) for p in Av)
        if u_wrong: errs_u += 1
        if v_wrong: errs_v += 1
        if u_wrong or v_wrong: errs_total += 1
        if (i+1) % 2000 == 0:
            elapsed = (time.time() - t0) / 60
            print(f'    N={N}: {i+1}/{n_cw}, errs_total={errs_total} '
                  f'({elapsed:.1f}min)', flush=True)
    elapsed = (time.time() - t0) / 60
    return {
        'N': N, 'decoder': 'trellis_sc_chained',
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v,
        'errs_total': errs_total,
        'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw,
        'bler_total': errs_total / n_cw,
        'time_min': elapsed,
    }


def main():
    channel = make_channel()
    all_results = {}

    # Try loading existing results
    out_path = os.path.join(OUT_DIR, 'isi_mac_sc_10kcw.json')
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        print(f'Loaded existing results: {list(all_results.keys())}')

    for N, (ku, kv) in sorted(RATES.items()):
        label = f'N{N}'
        if label in all_results:
            print(f'\nSkipping N={N} (already done)')
            continue

        print(f'\n{"="*60}')
        print(f'Trellis SC: N={N}, ku={ku}, kv={kv}, 10K CW')
        print(f'{"="*60}')

        Au, Av, fu_dict, fv_dict = load_design(N, ku, kv)

        result = eval_trellis_sc(channel, N, Au, Av, fu_dict, fv_dict, n_cw=10000)
        ci_lo, ci_hi = wilson_ci(result['errs_total'], result['n_cw'])
        result['ci'] = [ci_lo, ci_hi]

        print(f'  BLER={result["bler_total"]:.4f} ({result["errs_total"]}/{result["n_cw"]}) '
              f'CI=[{ci_lo:.4f}, {ci_hi:.4f}] '
              f'U={result["bler_u"]:.4f} V={result["bler_v"]:.4f} '
              f'({result["time_min"]:.1f}min)')

        all_results[label] = result

        # Save incrementally
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  Saved to {out_path}')

    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY: ISI-MAC Trellis SC Baselines (10K CW)')
    print(f'{"="*60}')
    for label in sorted(all_results.keys(), key=lambda x: int(x[1:])):
        r = all_results[label]
        print(f'  {label}: BLER={r["bler_total"]:.4f} '
              f'({r["errs_total"]}/{r["n_cw"]}) '
              f'U={r["bler_u"]:.4f} V={r["bler_v"]:.4f}')


if __name__ == '__main__':
    main()
