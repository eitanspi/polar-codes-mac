#!/usr/bin/env python3
"""
Task 2: Fill reliability gaps for entries near 100 errors.
Uses MC-based design files (same as original evals).
- BEMAC B N=128 SC: 81/50K -> run 20K more
- GMAC B N=256 SC: 30/5K -> run 15K more
- GMAC C N=128 SC: 71/10K -> run 5K more
"""
import sys, os, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.eval import MACEval
from polar.channels import BEMAC, GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file

OUT_DIR = os.path.join(BASE, 'results', 'reliable_evals')
DESIGNS_DIR = os.path.join(BASE, 'designs')
os.makedirs(OUT_DIR, exist_ok=True)

sigma2 = 10 ** (-6 / 10)  # SNR = 6 dB


def run_eval_mc(label, channel, design_file, n, ku, kv, n_cw, seed=12345):
    """Run SC eval using MC design file."""
    N = 2 ** n
    Au, Av, fu, fv, pe_u, pe_v, path_i = design_from_file(design_file, n, ku, kv)
    Au = sorted(Au); Av = sorted(Av)
    b = make_path(N, path_i)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  N={N}, ku={ku}, kv={kv}, path_i={path_i}, n_cw={n_cw}")
    print(f"  design: {os.path.basename(design_file)}")
    print(f"{'='*60}", flush=True)

    ev = MACEval(channel, log_domain=True, rng=np.random.default_rng(seed))
    t0 = time.time()
    ber_u, ber_v, bler = ev.run(N, b, Au, Av, fu, fv, n_codewords=n_cw, batch_size=1, verbose=True)
    elapsed = (time.time() - t0) / 60

    errs = round(bler * n_cw)
    print(f"  BLER={bler:.6f} ({errs}/{n_cw}) [{elapsed:.1f}min]", flush=True)

    result = {
        'label': label, 'N': N, 'ku': ku, 'kv': kv,
        'bler': bler, 'ber_u': ber_u, 'ber_v': ber_v,
        'errs': errs, 'n_cw': n_cw,
        'time_min': round(elapsed, 1), 'seed': seed,
        'design_file': os.path.basename(design_file),
    }
    return result


def main():
    all_results = {}
    t_start = time.time()
    out = os.path.join(OUT_DIR, 'reliable_fill_2026_04_16.json')

    # 1. BEMAC B N=128 SC: need 20K more CW
    channel_bemac = BEMAC()
    design_file = os.path.join(DESIGNS_DIR, 'bemac_B_n7.npz')
    r = run_eval_mc('BEMAC_B_N128_SC_extra', channel_bemac, design_file,
                    n=7, ku=64, kv=90, n_cw=20000, seed=77777)
    all_results['bemac_b_n128_sc'] = r
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Incremental save: {out}", flush=True)

    # 2. GMAC B N=256 SC: need 15K more CW
    channel_gmac = GaussianMAC(sigma2=sigma2)
    design_file = os.path.join(DESIGNS_DIR, 'gmac_B_n8_snr6dB.npz')
    r = run_eval_mc('GMAC_B_N256_SC_extra', channel_gmac, design_file,
                    n=8, ku=123, kv=123, n_cw=15000, seed=88888)
    all_results['gmac_b_n256_sc'] = r
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Incremental save: {out}", flush=True)

    # 3. GMAC C N=128 SC: need 5K more CW
    design_file = os.path.join(DESIGNS_DIR, 'gmac_C_n7_snr6dB.npz')
    r = run_eval_mc('GMAC_C_N128_SC_extra', channel_gmac, design_file,
                    n=7, ku=30, kv=58, n_cw=5000, seed=99999)
    all_results['gmac_c_n128_sc'] = r
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)

    elapsed_total = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"  All done in {elapsed_total:.1f}min")
    print(f"  Results: {out}")
    print(f"{'='*60}")

    # Summary with combined counts
    print("\nCombined results (old + new):")
    combos = [
        ('BEMAC B N=128 SC', 81, 50000, 'bemac_b_n128_sc'),
        ('GMAC B N=256 SC', 30, 5000, 'gmac_b_n256_sc'),
        ('GMAC C N=128 SC', 71, 10000, 'gmac_c_n128_sc'),
    ]
    for label, old_errs, old_cw, key in combos:
        r = all_results[key]
        new_errs = r['errs']; new_cw = r['n_cw']
        total_errs = old_errs + new_errs
        total_cw = old_cw + new_cw
        combined_bler = total_errs / total_cw
        print(f"  {label}: {total_errs}/{total_cw} = {combined_bler:.6f}")


if __name__ == '__main__':
    main()
