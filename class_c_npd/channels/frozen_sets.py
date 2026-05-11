"""
Frozen set loader for Class C designs, reusing the existing `designs/*.npz` files.

For each channel, Class C (path_i = N) designs give two per-position error rate
arrays:
  - u_error_rates[i]: Pe of the i-th U-bit when decoded on the MARGINAL channel
  - v_error_rates[i]: Pe of the i-th V-bit when decoded on the CLEAN conditional
                      channel given all of U

This is exactly what Stage 1 (U on marginal) and Stage 2 (V on clean) need.

We select Au and Av as the positions with lowest error rates, with target ku
and kv either specified explicitly or chosen automatically by a reliability
threshold.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Tuple


# Default designs directory (relative to to_git_v2 root)
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'designs')


def load_class_c_design(
    channel_name: str,
    n: int,
    snr_db: float = 6.0,
    ku: int = None,
    kv: int = None,
    pe_threshold: float = 0.01,
) -> Tuple[list, list, set, set, np.ndarray, np.ndarray]:
    """
    Load a Class C (path_i = N) design from the existing `designs/` files.

    Args:
        channel_name: one of 'gmac', 'bemac', 'abnmac'
        n: log2(N)
        snr_db: SNR in dB (used for GMAC file naming)
        ku: number of U info bits. If None, auto-select from pe_threshold.
        kv: number of V info bits. If None, auto-select from pe_threshold.
        pe_threshold: if ku/kv are None, select positions with Pe < threshold.

    Returns:
        Au: list of 1-indexed U info positions
        Av: list of 1-indexed V info positions
        frozen_u: set of 0-indexed U frozen positions (for NPD decoder)
        frozen_v: set of 0-indexed V frozen positions
        pe_u: per-position Pe for U on marginal channel (shape (N,))
        pe_v: per-position Pe for V on clean channel given X (shape (N,))
    """
    N = 1 << n

    # Locate the design file
    if channel_name.lower() in ('gmac', 'gaussian'):
        fname = f'gmac_C_n{n}_snr{int(round(snr_db))}dB.npz'
    elif channel_name.lower() == 'bemac':
        fname = f'bemac_C_n{n}.npz'
    elif channel_name.lower() == 'abnmac':
        fname = f'abnmac_C_n{n}.npz'
    elif channel_name.lower() in ('isi_mac', 'isi'):
        # No pre-computed design for ISI — would need fresh MC design
        raise NotImplementedError(
            "No pre-computed Class C design for ISI-MAC. "
            "Run Monte Carlo density evolution first."
        )
    else:
        raise ValueError(f"Unknown channel: {channel_name}")

    path = os.path.join(DESIGNS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Design not found: {path}")

    # CRITICAL: use design_from_file from polar.design_mc, which handles
    # ties in Pe values with a polar-specific tiebreak rule. Naive argsort
    # produces the wrong info set at large N where many positions tie at Pe=0.
    from polar.design_mc import design_from_file as _dff, load_design

    # If ku/kv not specified, auto-select by threshold, then defer to design_from_file
    if ku is None or kv is None:
        sorted_u, sorted_v, pe_u, pe_v, _ = load_design(path)
        if ku is None:
            ku = int((pe_u < pe_threshold).sum())
        if kv is None:
            kv = int((pe_v < pe_threshold).sum())

    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, path_i = _dff(path, n, ku, kv)

    # Sanity check: the file should be a Class C design
    if path_i is not None and path_i != N:
        print(f'WARNING: design file path_i={path_i} != N={N} (expected Class C)')

    Au = sorted(Au_list)
    Av = sorted(Av_list)
    # Convert 1-indexed frozen dicts to 0-indexed sets for NPD decoder
    frozen_u = {p - 1 for p in fu_dict.keys()}
    frozen_v = {p - 1 for p in fv_dict.keys()}

    return Au, Av, frozen_u, frozen_v, pe_u, pe_v


def design_summary(Au, Av, pe_u, pe_v) -> str:
    N = len(pe_u)
    ku = len(Au)
    kv = len(Av)
    max_pe_u = max(pe_u[p - 1] for p in Au) if Au else 0
    max_pe_v = max(pe_v[p - 1] for p in Av) if Av else 0
    sum_pe_u = sum(pe_u[p - 1] for p in Au)
    sum_pe_v = sum(pe_v[p - 1] for p in Av)
    return (
        f"  N={N}  ku={ku} ({ku/N:.3f})  kv={kv} ({kv/N:.3f})  total rate={(ku+kv)/N:.3f}\n"
        f"  max U Pe (mixture): {max_pe_u:.5f}\n"
        f"  max V Pe (clean):   {max_pe_v:.5f}\n"
        f"  union-bound BLER:   U~{sum_pe_u:.4f}  V~{sum_pe_v:.4f}  total~{sum_pe_u+sum_pe_v:.4f}"
    )
