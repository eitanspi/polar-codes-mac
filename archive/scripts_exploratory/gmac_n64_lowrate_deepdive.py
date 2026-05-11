#!/usr/bin/env python3
"""
Focused verification: GMAC N=64 at low rates.

The 5K-cw rate sweep showed nominal NN advantage at ku=kv ∈ {16, 20, 24}
(R=0.5-0.75) but CIs overlapped. This script re-runs those points with
50K cw each — 10x more — to determine if the advantage is real.

If the advantage holds, this is the only GMAC operating point in the
project where NN-SC clearly beats analytical SC.
"""

import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'scripts'))

from polar.encoder import polar_encode, polar_encode_batch
from polar.decoder import decode_single
from polar.channels import GaussianMAC
from polar.design import make_path
from gmac_rate_sweep import load_design, load_gmac_model, eval_nn_sc, eval_sc, wilson_ci


def main():
    SNR_DB = 6
    sigma2 = 10 ** (-SNR_DB / 10)
    channel = GaussianMAC(sigma2=sigma2)

    N = 64
    n_cw = 50000  # 10x the original sweep
    ks = [16, 20, 24, 28, 32]  # low rates + the published rate point as control

    print(f"\n{'='*72}")
    print(f"  GMAC N={N} low-rate deep dive — {n_cw} cw per point")
    print(f"  Hypothesis: NN-SC beats analytical SC at low rates")
    print(f"{'='*72}\n", flush=True)

    model = load_gmac_model(N)
    if model is None:
        print(f"FATAL: no model for N={N}")
        return
    b = make_path(N, N // 2)

    out_path = os.path.join(BASE, 'results', 'gmac_n64_lowrate_deepdive.json')
    results = []

    for k in ks:
        try:
            Au, Av, fu, fv = load_design(N, k, k)
        except Exception as e:
            print(f"k={k}: design error: {e}")
            continue
        sum_rate = 2 * k / N
        print(f"\nku=kv={k}  (R_sum={sum_rate:.3f})", flush=True)

        # NN-SC
        t0 = time.perf_counter()
        rng = np.random.default_rng(2026 + k)
        nn_errs = eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, n_cw, rng, batch_size=100)
        nn_time = time.perf_counter() - t0
        nn_bler, nn_lo, nn_hi = wilson_ci(nn_errs, n_cw)
        print(f"  NN-SC:  {nn_errs} errs / {n_cw}  =  {nn_bler:.3e}  CI[{nn_lo:.2e}, {nn_hi:.2e}]  ({nn_time:.0f}s)", flush=True)

        # SC
        t0 = time.perf_counter()
        rng2 = np.random.default_rng(2026 + k + 100000)
        sc_errs = eval_sc(N, channel, b, Au, Av, fu, fv, n_cw, rng2)
        sc_time = time.perf_counter() - t0
        sc_bler, sc_lo, sc_hi = wilson_ci(sc_errs, n_cw)
        print(f"  SC:     {sc_errs} errs / {n_cw}  =  {sc_bler:.3e}  CI[{sc_lo:.2e}, {sc_hi:.2e}]  ({sc_time:.0f}s)", flush=True)

        ratio = nn_bler / sc_bler if sc_bler > 0 else None
        # Statistical significance (CIs do not overlap)
        ci_overlap = nn_hi >= sc_lo and sc_hi >= nn_lo
        sig_str = "statistically tied (CIs overlap)" if ci_overlap else "STATISTICALLY SIGNIFICANT"
        print(f"  ratio:  {ratio:.3f}x  [{sig_str}]", flush=True)

        results.append({
            'N': N, 'ku': k, 'kv': k, 'sum_rate': sum_rate,
            'nn_errors': nn_errs, 'nn_cw': n_cw,
            'nn_bler': nn_bler, 'nn_ci_lo': nn_lo, 'nn_ci_hi': nn_hi,
            'sc_errors': sc_errs, 'sc_cw': n_cw,
            'sc_bler': sc_bler, 'sc_ci_lo': sc_lo, 'sc_ci_hi': sc_hi,
            'ratio_nn_over_sc': ratio,
            'ci_overlap': ci_overlap,
        })

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {out_path}")
    print("\n=== SUMMARY ===")
    print(f"{'k':>4}  {'R_sum':>6}  {'NN BLER':>12}  {'SC BLER':>12}  {'ratio':>8}  {'sig?':>6}")
    for r in results:
        print(f"{r['ku']:>4}  {r['sum_rate']:>6.3f}  {r['nn_bler']:>12.3e}  {r['sc_bler']:>12.3e}  {r['ratio_nn_over_sc']:>7.3f}x  {'OVERLAP' if r['ci_overlap'] else 'SIGNIF':>6}")


if __name__ == '__main__':
    main()
