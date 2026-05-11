#!/usr/bin/env python3
"""
Investigate: why does GMAC Class C SC BLER rise at N=512,1024?
Hypothesis: analytical design degrades at large N. Try MC design.
"""
import os, sys, math, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.set_num_threads(4)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_batch

SNR_DB = 6.0
sigma2 = 10 ** (-SNR_DB / 10)
channel = GaussianMAC(sigma2=sigma2)

def run_sc_eval(N, ku, kv, Au, Av, fu, fv, n_cw=5000, batch_sz=10, seed=42):
    b = make_path(N, N)  # Class C: corner path
    rng = np.random.default_rng(seed)
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    errs = 0; total = 0
    t0 = time.time()
    while total < n_cw:
        actual = min(batch_sz, n_cw - total)
        info_u = rng.integers(0, 2, size=(actual, ku))
        info_v = rng.integers(0, 2, size=(actual, kv))
        U = build_message_batch(N, info_u, Au)
        V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U); Y = polar_encode_batch(V)
        Z_sig = (1 - 2*X).astype(float) + (1 - 2*Y).astype(float) + np.random.default_rng().standard_normal(X.shape) * np.sqrt(sigma2)
        results = decode_batch(N, Z_sig.tolist(), b, fu, fv, channel, vectorized=True)
        for i, (u_dec, v_dec) in enumerate(results):
            ue = int(np.sum(np.array(u_dec)[u_info_idx] != info_u[i]))
            ve = int(np.sum(np.array(v_dec)[v_info_idx] != info_v[i]))
            if ue > 0 or ve > 0: errs += 1
        total += actual
        if total % 1000 == 0:
            elapsed = time.time() - t0
            print(f'    [{total}/{n_cw}] errs={errs} ({elapsed:.1f}s)', flush=True)
    elapsed = time.time() - t0
    bler = errs / n_cw
    return {'bler': bler, 'errs': errs, 'n_cw': n_cw, 'time_s': elapsed, 'ku': ku, 'kv': kv}

def main():
    results = {}
    BASE = os.path.join(os.path.dirname(__file__), '..')

    for N in [256, 512, 1024]:
        n = int(math.log2(N))
        print(f'\n{"="*60}')
        print(f'N={N}, GMAC Class C, SNR={SNR_DB}dB')
        print(f'{"="*60}')

        # Method 1: Analytical design (from file if available)
        design_file = os.path.join(BASE, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
        if os.path.exists(design_file):
            # Load MC design
            from polar.design_mc import design_from_file
            # Get rates from table
            if N == 256: ku, kv = 59, 117
            elif N == 512: ku, kv = 119, 233
            elif N == 1024: ku, kv = 238, 467
            Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, ku, kv)
            print(f'  MC design: ku={ku}, kv={kv}, |Au|={len(Au)}, |Av|={len(Av)}')
            r = run_sc_eval(N, ku, kv, Au, Av, fu, fv, n_cw=5000, batch_sz=max(2, 20//max(1,N//128)))
            print(f'  MC design SC: BLER={r["bler"]:.4f} ({r["errs"]}/{r["n_cw"]})')
            results[f'N{N}_mc'] = r

        # Method 2: Analytical design
        from polar.design import design_gmac
        Au_a, Av_a, fu_a, fv_a, _, _ = design_gmac(n, ku, kv, sigma2=sigma2)
        print(f'  Analytical design: ku={ku}, kv={kv}, |Au|={len(Au_a)}, |Av|={len(Av_a)}')
        r2 = run_sc_eval(N, ku, kv, Au_a, Av_a, fu_a, fv_a, n_cw=5000, batch_sz=max(2, 20//max(1,N//128)))
        print(f'  Analytical SC: BLER={r2["bler"]:.4f} ({r2["errs"]}/{r2["n_cw"]})')
        results[f'N{N}_analytical'] = r2

        # Check agreement
        if os.path.exists(design_file):
            agree = set(Au) == set(Au_a) and set(Av) == set(Av_a)
            print(f'  Designs agree: {agree}')
            if not agree:
                only_mc_u = set(Au) - set(Au_a)
                only_anal_u = set(Au_a) - set(Au)
                print(f'  Au diff: MC-only={sorted(only_mc_u)[:10]}, Anal-only={sorted(only_anal_u)[:10]}')

        # Save incrementally
        out = os.path.join(BASE, 'results', 'gmac_classC_sc_investigation.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)

    # Method 3: Try lower rate (more frozen bits) to see if rate is too high
    print(f'\n{"="*60}')
    print(f'Rate analysis: current rates vs capacity')
    print(f'{"="*60}')
    I_ZX = 0.465
    I_ZY_X = 0.912
    for N in [256, 512, 1024]:
        if N == 256: ku, kv = 59, 117
        elif N == 512: ku, kv = 119, 233
        elif N == 1024: ku, kv = 238, 467
        Ru = ku / N; Rv = kv / N
        print(f'  N={N}: Ru={Ru:.3f} (cap={I_ZX:.3f}, util={Ru/I_ZX:.2f}), '
              f'Rv={Rv:.3f} (cap={I_ZY_X:.3f}, util={Rv/I_ZY_X:.2f})')

    print(f'\nSaved: {out}')

if __name__ == '__main__':
    main()
