"""
design_mc.py
============
Monte Carlo polar code design for two-user MAC channels.

Instead of using analytical Bhattacharyya parameters (as in design.py),
this module estimates the quality of each synthetic bit-channel via
Monte Carlo simulation with genie-aided SC decoding.

References
----------
Tal & Vardy (2013) "How to Construct Polar Codes"
Önay (ISIT 2013) "Successive Cancellation Decoding of Polar Codes for the
  Two-User Binary-Input MAC"
"""

import os
import time
import numpy as np

from .encoder import polar_encode
from .channels import BEMAC, ABNMAC
from ._decoder_numba import _build_z_tree, _coord_prob_u_log, _coord_prob_v_log
from .design import make_path


# ─────────────────────────────────────────────────────────────────────────────
#  Genie-aided SC decoder for design
# ─────────────────────────────────────────────────────────────────────────────

def _genie_decode_single(N, z, b, u_true, v_true, channel):
    """
    Genie-aided SC MAC decoder for estimating per-bit error rates.

    Decodes each bit using the SC coordinate probabilities, but always
    feeds the TRUE value of each bit (not the decoded value) to
    subsequent decoding steps.  This prevents error propagation and
    gives the true error probability of each synthetic channel.

    Parameters
    ----------
    N       : int — block length
    z       : list — channel output symbols, length N
    b       : list[int] — path vector, length 2N  (0=U step, 1=V step)
    u_true  : list[int] — true U bits, length N (0-indexed)
    v_true  : list[int] — true V bits, length N (0-indexed)
    channel : MACChannel instance

    Returns
    -------
    u_errors : np.ndarray shape (N,) — 1 where decoder erred, 0 otherwise
    v_errors : np.ndarray shape (N,) — same for V
    """
    u_hat = {}          # 1-indexed, filled with TRUE values (genie)
    v_hat = {}
    i, j = 0, 0
    cache = {}
    z_tree = _build_z_tree(list(z))

    u_errors = np.zeros(N, dtype=np.int32)
    v_errors = np.zeros(N, dtype=np.int32)

    for k in range(1, 2 * N + 1):
        bk = b[k - 1]

        if bk == 0:                         # ── U step ──
            i += 1
            p0 = _coord_prob_u_log(N, i, j, z_tree, u_hat, v_hat,
                                   0, channel, cache)
            p1 = _coord_prob_u_log(N, i, j, z_tree, u_hat, v_hat,
                                   1, channel, cache)
            decoded = 1 if p1 > p0 else 0
            true_val = u_true[i - 1]
            if decoded != true_val:
                u_errors[i - 1] = 1
            u_hat[i] = true_val             # genie: feed true value

        else:                                # ── V step ──
            j += 1
            p0 = _coord_prob_v_log(N, i, j, z_tree, u_hat, v_hat,
                                   0, channel, cache)
            p1 = _coord_prob_v_log(N, i, j, z_tree, u_hat, v_hat,
                                   1, channel, cache)
            decoded = 1 if p1 > p0 else 0
            true_val = v_true[j - 1]
            if decoded != true_val:
                v_errors[j - 1] = 1
            v_hat[j] = true_val

    return u_errors, v_errors


# ─────────────────────────────────────────────────────────────────────────────
#  MC Design core
# ─────────────────────────────────────────────────────────────────────────────

def mc_design(n, channel, mc_trials=1000, seed=None, verbose=True,
              time_budget=None, path_i=None):
    """
    Estimate per-bit error probabilities via genie-aided SC Monte Carlo.

    Parameters
    ----------
    n           : int — polarization stages, block length N = 2^n
    channel     : MACChannel
    mc_trials   : int — number of MC trials (may stop early if time_budget hit)
    seed        : int or None — RNG seed for message generation
    verbose     : bool — print progress
    time_budget : float or None — max seconds; stops early if exceeded
    path_i      : int or None — path parameter for 0^i 1^N 0^{N-i}
                  (None defaults to N, i.e. code class C: 0^N 1^N)

    Returns
    -------
    error_rates_u : np.ndarray (N,) — estimated P_e per U channel
    error_rates_v : np.ndarray (N,) — estimated P_e per V channel
    sorted_u      : np.ndarray (N,) — U channel indices, best (lowest P_e) first
    sorted_v      : np.ndarray (N,) — V channel indices, best first
    """
    N = 1 << n
    if path_i is None:
        path_i = N
    b = make_path(N, path_i)
    rng = np.random.default_rng(seed)

    u_err_counts = np.zeros(N, dtype=np.float64)
    v_err_counts = np.zeros(N, dtype=np.float64)
    completed = 0

    t0 = time.time()
    for trial in range(mc_trials):
        if time_budget is not None and time.time() - t0 > time_budget:
            if verbose:
                print(f"  time budget ({time_budget}s) reached after "
                      f"{trial} trials")
            break

        # Random message bits
        u = rng.integers(0, 2, size=N).tolist()
        v = rng.integers(0, 2, size=N).tolist()

        # Encode
        x = polar_encode(u)
        y = polar_encode(v)

        # Transmit
        z_arr = channel.sample_batch(np.array(x), np.array(y))
        z = list(z_arr.flat) if z_arr.dtype != object else z_arr.tolist()

        # Genie-aided decode
        u_err, v_err = _genie_decode_single(N, z, b, u, v, channel)
        u_err_counts += u_err
        v_err_counts += v_err
        completed += 1

        if verbose and (trial + 1) % max(1, mc_trials // 10) == 0:
            elapsed = time.time() - t0
            print(f"  trial {trial+1}/{mc_trials}  ({elapsed:.1f}s)")

    error_rates_u = u_err_counts / max(completed, 1)
    error_rates_v = v_err_counts / max(completed, 1)

    sorted_u = np.argsort(error_rates_u)    # ascending P_e = best first
    sorted_v = np.argsort(error_rates_v)

    if verbose:
        total = time.time() - t0
        print(f"  MC design done: {completed} trials in {total:.1f}s")
        print(f"  U: {int(np.sum(error_rates_u < 0.01))} ch P_e<0.01, "
              f"{int(np.sum(error_rates_u < 0.1))} ch P_e<0.1")
        print(f"  V: {int(np.sum(error_rates_v < 0.01))} ch P_e<0.01, "
              f"{int(np.sum(error_rates_v < 0.1))} ch P_e<0.1")

    return error_rates_u, error_rates_v, sorted_u, sorted_v


# ─────────────────────────────────────────────────────────────────────────────
#  Design functions matching design.py interface
# ─────────────────────────────────────────────────────────────────────────────

def _select_info_frozen(N, sorted_indices, k):
    """Pick best k channels → 1-indexed info set A and frozen dict."""
    A_0idx = sorted(sorted_indices[:k].tolist())
    A = [i + 1 for i in A_0idx]
    all_pos = set(range(1, N + 1))
    frozen = {pos: 0 for pos in sorted(all_pos - set(A))}
    return A, frozen


def design_bemac_mc(n, ku, kv, mc_trials=1000, seed=None, verbose=True,
                    time_budget=None, path_i=None):
    """
    MC polar code design for BE-MAC.

    Same return format as design.design_bemac:
        Au, Av, frozen_u, frozen_v, error_rates_u, error_rates_v

    error_rates replace Bhattacharyya parameters as the reliability metric.
    """
    N = 1 << n
    channel = BEMAC()
    pe_u, pe_v, sorted_u, sorted_v = mc_design(
        n, channel, mc_trials, seed, verbose, time_budget, path_i=path_i)

    Au, frozen_u = _select_info_frozen(N, sorted_u, ku)
    Av, frozen_v = _select_info_frozen(N, sorted_v, kv)
    return Au, Av, frozen_u, frozen_v, pe_u, pe_v


def design_abnmac_mc(n, ku, kv, p_noise=None, mc_trials=1000, seed=None,
                     verbose=True, time_budget=None, path_i=None):
    """
    MC polar code design for ABN-MAC.

    Same return format as design.design_abnmac:
        Au, Av, frozen_u, frozen_v, error_rates_u, error_rates_v
    """
    N = 1 << n
    channel = ABNMAC(p_noise)
    pe_u, pe_v, sorted_u, sorted_v = mc_design(
        n, channel, mc_trials, seed, verbose, time_budget, path_i=path_i)

    Au, frozen_u = _select_info_frozen(N, sorted_u, ku)
    Av, frozen_v = _select_info_frozen(N, sorted_v, kv)
    return Au, Av, frozen_u, frozen_v, pe_u, pe_v


# ─────────────────────────────────────────────────────────────────────────────
#  Save / Load designs
# ─────────────────────────────────────────────────────────────────────────────

def save_design(filepath, sorted_u, sorted_v,
                error_rates_u=None, error_rates_v=None, path_i=None):
    """Persist MC design results to a .npz file."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    data = dict(sorted_u=np.asarray(sorted_u),
                sorted_v=np.asarray(sorted_v))
    if error_rates_u is not None:
        data['error_rates_u'] = np.asarray(error_rates_u)
    if error_rates_v is not None:
        data['error_rates_v'] = np.asarray(error_rates_v)
    if path_i is not None:
        data['path_i'] = np.array(path_i)
    np.savez(filepath, **data)


def load_design(filepath):
    """Load a saved MC design."""
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    data = np.load(filepath, allow_pickle=True)

    # Support both formats:
    #   old: sorted_u, sorted_v, error_rates_u, error_rates_v
    #   new: u_error_rates, v_error_rates (sorted lists derived from these)
    if 'sorted_u' in data:
        sorted_u = data['sorted_u']
        sorted_v = data['sorted_v']
        pe_u = data['error_rates_u'] if 'error_rates_u' in data else None
        pe_v = data['error_rates_v'] if 'error_rates_v' in data else None
    elif 'u_error_rates' in data:
        pe_u = data['u_error_rates']
        pe_v = data['v_error_rates']
        # Sort positions by error rate (ascending = best first), 0-indexed
        sorted_u = np.argsort(pe_u)
        sorted_v = np.argsort(pe_v)
    else:
        raise KeyError(f"Unrecognized design file format: {list(data.keys())}")

    path_i = int(data['path_i']) if 'path_i' in data else None
    return sorted_u, sorted_v, pe_u, pe_v, path_i


def design_from_file(filepath, n, ku, kv):
    """
    Load a saved MC design and build info / frozen sets.

    Returns Au, Av, frozen_u, frozen_v, error_rates_u, error_rates_v, path_i
    """
    N = 1 << n
    sorted_u, sorted_v, pe_u, pe_v, path_i = load_design(filepath)
    Au, frozen_u = _select_info_frozen(N, sorted_u, ku)
    Av, frozen_v = _select_info_frozen(N, sorted_v, kv)
    return Au, Av, frozen_u, frozen_v, pe_u, pe_v, path_i


# ─────────────────────────────────────────────────────────────────────────────
#  Random frozen-bit utility
# ─────────────────────────────────────────────────────────────────────────────

def build_frozen_dict(frozen_positions, seed=None):
    """Build a frozen-bit dict with random (but reproducible) values."""
    rng = np.random.default_rng(seed)
    positions = sorted(frozen_positions)
    bits = rng.integers(0, 2, size=len(positions))
    return {pos: int(b) for pos, b in zip(positions, bits)}


# ─────────────────────────────────────────────────────────────────────────────
#  Comparison utility
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_analytical(n, ku, kv, channel_type='bemac',
                            mc_trials=1000, p_noise=None, seed=None):
    """
    Run both analytical and MC designs and report whether they agree.

    Returns dict with Au/Av for both methods and match flags.
    """
    from .design import design_bemac, design_abnmac

    if channel_type == 'bemac':
        Au_a, Av_a, _, _, z_u, z_v = design_bemac(n, ku, kv)
        Au_m, Av_m, _, _, pe_u, pe_v = design_bemac_mc(
            n, ku, kv, mc_trials, seed)
    else:
        Au_a, Av_a, _, _, z_u, z_v = design_abnmac(n, ku, kv, p_noise)
        Au_m, Av_m, _, _, pe_u, pe_v = design_abnmac_mc(
            n, ku, kv, p_noise, mc_trials, seed)

    return {
        'Au_analytical': Au_a, 'Au_mc': Au_m,
        'Av_analytical': Av_a, 'Av_mc': Av_m,
        'Au_match': Au_a == Au_m, 'Av_match': Av_a == Av_m,
        'z_u': z_u, 'z_v': z_v, 'pe_u': pe_u, 'pe_v': pe_v,
    }
