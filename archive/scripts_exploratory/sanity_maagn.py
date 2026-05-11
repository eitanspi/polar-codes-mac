#!/usr/bin/env python3
"""
sanity_maagn.py
===============
Sanity checks for the MA-AGN MAC channel before launching training.

1. Verify stationarity: Var[Z_i] ≈ sigma2 across positions.
2. Verify memoryless GMAC SC BLER on MA-AGN samples (with α > 0) is WORSE
   than on memoryless GMAC samples (α = 0) at the same sigma2.

Run:
    python scripts/sanity_maagn.py
"""
from __future__ import annotations
import os
import sys
import math
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.channels_memory_new import MAAGNMAC
from polar.channels import GaussianMAC
from polar.encoder import polar_encode_batch
from polar.decoder import decode_single
from polar.design import make_path
from polar.design_mc import design_from_file


def stationarity_check():
    print("[1] Stationarity check")
    print("-" * 40)
    sigma2 = 0.25
    alpha = 0.3
    N = 256  # Longer so statistics converge
    B = 500
    ch = MAAGNMAC(sigma2=sigma2, alpha=alpha)
    X = np.zeros((B, N), dtype=np.int32)
    Y = np.zeros((B, N), dtype=np.int32)  # noise-only
    # Sample z noise only (BPSK = +1 +1), so subtract the mean =2.
    np.random.seed(0)
    Z = ch.sample_batch(X, Y)
    noise = Z - 2.0  # sx+sy = 2 when X=Y=0
    var_per_pos = noise.var(axis=0)
    print(f"  sigma2 = {sigma2}  alpha = {alpha}")
    print(f"  Var[Z_i] across positions (B={B}):")
    print(f"    mean : {var_per_pos.mean():.4f} (target {sigma2:.4f})")
    print(f"    min  : {var_per_pos.min():.4f}")
    print(f"    max  : {var_per_pos.max():.4f}")
    print(f"  Var[N_0] : {noise[:, 0].var():.4f}")
    print(f"  Var[N_{N-1}] : {noise[:, -1].var():.4f}")
    # Also check autocorrelation at lag 1
    corr = np.corrcoef(noise[:, :-1].ravel(), noise[:, 1:].ravel())[0, 1]
    print(f"  lag-1 autocorr : {corr:.4f} (target {alpha:.4f})")


def bler_check(N=16, n_cw=500, sigma2=0.25, alpha=0.3, snr_db_design=6):
    print(f"\n[2] Memoryless SC BLER on MA-AGN vs GMAC  (N={N})")
    print("-" * 60)
    n = int(math.log2(N))
    # Use GMAC_C design at 6 dB (matches training config)
    path = os.path.join(_ROOT, 'designs',
                        f'gmac_C_n{n}_snr{snr_db_design}dB.npz')
    # GMAC Class C rates (aligning with RATES used for ISI MAC)
    ku, kv = {16: (4, 7), 32: (7, 15), 64: (15, 29)}[N]
    Au, Av, fu_1idx, fv_1idx, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au); Av = sorted(Av)
    b = make_path(N, N)  # Class C (all U then all V)

    # Two channels: memoryless GMAC for baseline decoding
    gmac = GaussianMAC(sigma2=sigma2)

    # (a) Send via MA-AGN, decode with memoryless GMAC SC
    ma_agn = MAAGNMAC(sigma2=sigma2, alpha=alpha)
    rng = np.random.default_rng(42)
    np.random.seed(42)
    errs_total_a = errs_total_b = 0

    for trial in range(n_cw):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for p in Au: u_msg[p - 1] = rng.integers(0, 2)
        for p in Av: v_msg[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u_msg[None, :])[0]
        y = polar_encode_batch(v_msg[None, :])[0]

        # (a) MA-AGN sample, decode as GMAC
        z_a = ma_agn.sample_batch(x[None, :], y[None, :])[0]
        u_dec_a, v_dec_a = decode_single(N, z_a.tolist(), b, fu_1idx, fv_1idx, gmac,
                                         log_domain=True)
        if (any(u_dec_a[p - 1] != u_msg[p - 1] for p in Au) or
                any(v_dec_a[p - 1] != v_msg[p - 1] for p in Av)):
            errs_total_a += 1

        # (b) pure GMAC sample, decode as GMAC
        z_b = gmac.sample_batch(x[None, :], y[None, :])[0]
        u_dec_b, v_dec_b = decode_single(N, z_b.tolist(), b, fu_1idx, fv_1idx, gmac,
                                         log_domain=True)
        if (any(u_dec_b[p - 1] != u_msg[p - 1] for p in Au) or
                any(v_dec_b[p - 1] != v_msg[p - 1] for p in Av)):
            errs_total_b += 1

    bler_a = errs_total_a / n_cw
    bler_b = errs_total_b / n_cw
    print(f"  sigma2={sigma2}  alpha={alpha}  ku={ku}  kv={kv}  n_cw={n_cw}")
    print(f"  (a) MA-AGN sample -> memoryless SC : BLER = {bler_a:.4f}")
    print(f"  (b) GMAC   sample -> memoryless SC : BLER = {bler_b:.4f}")
    if bler_a > bler_b:
        print(f"  OK : MA-AGN BLER > GMAC BLER (ratio {bler_a/max(bler_b,1e-6):.2f}x)")
    else:
        print(f"  WARN : MA-AGN BLER not higher than GMAC BLER")
    return bler_a, bler_b


if __name__ == "__main__":
    t0 = time.time()
    stationarity_check()
    bler_check(N=16)
    print(f"\nTotal time: {(time.time()-t0):.1f} s")
