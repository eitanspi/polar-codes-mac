#!/usr/bin/env python3
"""
eval_bemac_nn_scl_large_N.py — NN-SCL on BEMAC at N=256, 512, 1024.

BEMAC Class B, Ru=0.50, Rv=0.70.
Uses ncg_pure_neural_N{N}.pt models with BemacNeuralSCLDecoder.

Codeword counts: N=256: 1000, N=512: 500, N=1024: 200
Tests L=1 (NN-SC) and L=4 (NN-SCL).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'nn_mac'))

import math
import time
import json
import numpy as np
import torch

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import BEMAC
from polar.design import make_path
from ncg_pure_neural import PureNeuralCompGraphDecoder

# Import the BEMAC SCL decoder from existing script
from eval_bemac_nn_scl import (
    BemacNeuralSCLDecoder, load_bemac_model, load_bemac_design,
    evaluate_nn_sc_bemac, evaluate_nn_scl_bemac
)


def main():
    RU, RV = 0.50, 0.70

    # Reference SC and NN-SC BLERs from bemac_nn_vs_sc_complete.json
    SC_REF = {256: 8e-05, 512: 0.0, 1024: 0.0001}
    NN_SC_REF = {256: 4e-05, 512: 0.0, 1024: 0.0001}

    # Codeword counts and batch sizes
    CONFIG = {
        256:  {'n_cw_sc': 1000, 'n_cw_scl': 1000, 'bs': 25},
        512:  {'n_cw_sc': 500,  'n_cw_scl': 500,  'bs': 10},
        1024: {'n_cw_sc': 200,  'n_cw_scl': 200,  'bs': 5},
    }

    # Load existing results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'bemac',
                           'bemac_classB_Ru50_Rv70_nn_scl')
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, 'bemac_nn_scl_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}

    print(f"\n{'='*72}")
    print(f"BEMAC NN-SCL Large-N Evaluation — Class B, Ru={RU}, Rv={RV}")
    print(f"{'='*72}")

    for N in [256, 512, 1024]:
        n = int(math.log2(N))
        cfg = CONFIG[N]

        # Load design
        Au, Av, fu, fv, ku, kv = load_bemac_design(N, RU, RV)
        b = make_path(N, N // 2)  # Class B

        # Load model
        try:
            model = load_bemac_model(N)
        except FileNotFoundError as e:
            print(f"\n  N={N}: {e} -- SKIPPING")
            continue

        print(f"\n{'~'*72}")
        print(f"  N={N}, ku={ku}, kv={kv}")
        print(f"  Model params: {model.count_parameters():,}")
        print(f"  SC ref BLER:    {SC_REF.get(N, '?')}")
        print(f"  NN-SC ref BLER: {NN_SC_REF.get(N, '?')}")

        res = {'N': N, 'ku': ku, 'kv': kv,
               'sc_ref': SC_REF.get(N), 'nn_sc_ref': NN_SC_REF.get(N)}

        # NN-SC (greedy, L=1)
        print(f"\n  Running NN-SC (L=1) with {cfg['n_cw_sc']} codewords...")
        t0 = time.time()
        bler_sc = evaluate_nn_sc_bemac(model, N, b, Au, Av, fu, fv,
                                        cfg['n_cw_sc'], batch_size=cfg['bs'])
        t_sc = time.time() - t0
        print(f"  NN-SC  (L=1):  BLER={bler_sc:.6f}  [{t_sc:.1f}s]")
        res['nn_sc_bler'] = bler_sc
        res['nn_sc_time'] = round(t_sc, 1)

        # NN-SCL with L=4
        print(f"\n  Running NN-SCL (L=4) with {cfg['n_cw_scl']} codewords...")
        t0 = time.time()
        bler_scl4 = evaluate_nn_scl_bemac(model, N, b, Au, Av, fu, fv,
                                           cfg['n_cw_scl'], L=4)
        t_scl4 = time.time() - t0
        print(f"  NN-SCL (L=4):  BLER={bler_scl4:.6f}  [{t_scl4:.1f}s, {cfg['n_cw_scl']} cw]")
        res['nn_scl4_bler'] = bler_scl4
        res['nn_scl4_cw'] = cfg['n_cw_scl']
        res['nn_scl4_time'] = round(t_scl4, 1)

        # Compute ratios
        sc_ref = SC_REF.get(N, 0)
        if sc_ref and sc_ref > 0:
            res['nn_sc_vs_sc'] = round(bler_sc / sc_ref, 3)
            res['nn_scl4_vs_sc'] = round(bler_scl4 / sc_ref, 3) if bler_scl4 > 0 else 0
        if bler_sc > 0:
            res['nn_scl4_vs_nn_sc'] = round(bler_scl4 / bler_sc, 3) if bler_scl4 > 0 else 0

        # Summary
        print(f"\n  Summary for N={N}:")
        print(f"    {'Decoder':<18s}  {'BLER':>10s}  {'vs SC':>8s}")
        if sc_ref and sc_ref > 0:
            print(f"    {'SC (analytical)':<18s}  {sc_ref:>10.6f}  {'1.00x':>8s}")
            print(f"    {'NN-SC (L=1)':<18s}  {bler_sc:>10.6f}  {bler_sc/sc_ref:>7.2f}x")
            print(f"    {'NN-SCL (L=4)':<18s}  {bler_scl4:>10.6f}  {bler_scl4/sc_ref:>7.2f}x")
        else:
            print(f"    {'SC (analytical)':<18s}  {sc_ref:>10.6f}  {'N/A':>8s}")
            print(f"    {'NN-SC (L=1)':<18s}  {bler_sc:>10.6f}")
            print(f"    {'NN-SCL (L=4)':<18s}  {bler_scl4:>10.6f}")

        results[str(N)] = res

        # Save after each N (in case of crash)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {results_path}")

    print(f"\n{'='*72}")
    print("Done.")


if __name__ == '__main__':
    main()
