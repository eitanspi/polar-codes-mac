#!/usr/bin/env python3
"""
Benchmark NN-SC vs analytical SC on BEMAC Class C at the operating points
used in the "NN advantage" results:
  - Class C (Ru=0.30, Rv=0.60)
  - Class C (Ru=0.35, Rv=0.70)

For each (rate, N), measures wall-clock time per codeword for both
decoders. Uses a small batch (200 cw) so it finishes quickly.

Class C uses path_i=N (i.e., decode all of V first, then all of U).
The analytical SC takes the _decode_extreme_llr fast path.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'nn_mac'))

import math
import time
import json
import numpy as np
import torch

from polar.encoder import polar_encode_batch
from polar.channels import BEMAC
from polar.design import make_path
from polar.decoder import decode_single
from eval_bemac_nn_scl import load_bemac_model, load_bemac_design


def bench_nn_sc(model, N, b, Au, Av, fu, fv, n_cw, batch_size, seed=2025):
    """Returns (errors, total, time_per_cw_sec)."""
    model.eval()
    rng = np.random.default_rng(seed)
    errs, total = 0, 0
    t_total = 0.0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy((xf + yf).astype(np.int64)).long()

            t0 = time.perf_counter()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            t_total += time.perf_counter() - t0

            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e:
                    errs += 1
            total += actual
    return errs, total, t_total / total


def bench_sc(N, b, Au, Av, fu, fv, channel, n_cw, seed=2025):
    """Returns (errors, total, time_per_cw_sec)."""
    rng = np.random.default_rng(seed)
    errs, total = 0, 0
    t_total = 0.0
    while total < n_cw:
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        xf = polar_encode_batch(uf.reshape(1, N))[0]
        yf = polar_encode_batch(vf.reshape(1, N))[0]
        z = (xf + yf).astype(int).tolist()

        t0 = time.perf_counter()
        u_dec, v_dec = decode_single(N, z, b, fu, fv, channel)
        t_total += time.perf_counter() - t0

        e = any(u_dec[p-1] != uf[p-1] for p in Au) or \
            any(v_dec[p-1] != vf[p-1] for p in Av)
        if e:
            errs += 1
        total += 1
    return errs, total, t_total / total


def make_classC_design(N, Ru, Rv):
    """Build Class C frozen sets manually from the bemac_C_n*.npz designs.

    The eval_bemac_nn_scl.load_bemac_design hardcodes Class B (bemac_B_n*).
    For Class C we use bemac_C_n*.npz which has the same fields.
    """
    n = int(math.log2(N))
    ku = int(N * Ru)
    kv = int(N * Rv)
    designs_dir = os.path.join(os.path.dirname(__file__), '..', 'designs')
    dp = os.path.join(designs_dir, f'bemac_C_n{n}.npz')
    if not os.path.exists(dp):
        # Fall back to root project designs/
        dp2 = os.path.join(os.path.dirname(__file__), '..', '..', 'designs', f'bemac_C_n{n}.npz')
        if os.path.exists(dp2):
            dp = dp2
        else:
            raise FileNotFoundError(f"No Class C design at {dp} or {dp2}")
    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:ku]])
    Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv, ku, kv


def main():
    NS = [16, 32, 64, 128, 256, 512]
    N_CW_BENCH = 200  # small benchmark — just for timing

    rate_points = [
        (0.30, 0.60),
        (0.35, 0.70),
    ]

    channel = BEMAC()
    results = {}

    for Ru, Rv in rate_points:
        key = f"Ru{int(Ru*100)}_Rv{int(Rv*100)}"
        results[key] = {}
        print(f"\n{'='*72}")
        print(f"  BEMAC Class C, Ru={Ru}, Rv={Rv}")
        print(f"{'='*72}", flush=True)

        for N in NS:
            print(f"\n  N={N}", flush=True)
            try:
                Au, Av, fu, fv, ku, kv = make_classC_design(N, Ru, Rv)
            except FileNotFoundError as e:
                print(f"    SKIP design: {e}", flush=True)
                continue
            b = make_path(N, N)  # Class C: path_i = N
            try:
                model = load_bemac_model(N)
            except FileNotFoundError as e:
                print(f"    SKIP model: {e}", flush=True)
                continue

            # NN-SC timing
            try:
                bs = max(1, min(50, N_CW_BENCH))
                nn_errs, nn_total, nn_t = bench_nn_sc(model, N, b, Au, Av, fu, fv, N_CW_BENCH, bs)
                nn_ms = nn_t * 1000
                print(f"    NN-SC: {nn_total} cw, {nn_errs} err, {nn_ms:.3f} ms/cw", flush=True)
            except Exception as e:
                print(f"    NN-SC ERROR: {e}", flush=True)
                nn_t, nn_errs = None, None
                nn_ms = None

            # Analytical SC timing
            try:
                sc_errs, sc_total, sc_t = bench_sc(N, b, Au, Av, fu, fv, channel, N_CW_BENCH)
                sc_ms = sc_t * 1000
                print(f"    SC:    {sc_total} cw, {sc_errs} err, {sc_ms:.3f} ms/cw", flush=True)
            except Exception as e:
                print(f"    SC ERROR: {e}", flush=True)
                sc_t, sc_errs = None, None
                sc_ms = None

            results[key][str(N)] = {
                'N': N, 'ku': ku, 'kv': kv,
                'nn_ms_per_cw': nn_ms,
                'sc_ms_per_cw': sc_ms,
                'speedup_sc_over_nn': (nn_ms / sc_ms) if (nn_ms and sc_ms) else None,
            }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'bemac',
                            'bemac_classC_timing.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
