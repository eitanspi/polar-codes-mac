"""
_decoder_base.py
================
O(N²) recursive SC MAC decoder for two-user binary-input polar codes.

Internal fallback decoder — used by decoder.py for intermediate paths
(path_i not in {0, N}). Users should import from decoder.py instead.

Implements Algorithm 1 from Önay ISIT 2013:
  "Successive Cancellation Decoding of Polar Codes for the Two-User Binary-Input MAC"
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


_LOG_025 = np.log(0.25)
_LOG_05  = np.log(0.5)


# ─────────────────────────────────────────────────────────────────────────────
#  Z-tree precomputation
# ─────────────────────────────────────────────────────────────────────────────

def _build_z_tree(z: list) -> dict:
    """
    Precompute all z-subblock slices needed by the polar decode tree.

    Returns {(level, block_idx): tuple(z_slice)}.
      level 0 = root, block size N
      level n = leaves, N blocks of size 1
    """
    N = len(z)
    tree = {(0, 0): tuple(z)}
    level = 0
    block_size = N
    while block_size > 1:
        half = block_size // 2
        num_blocks = N // block_size
        for b_idx in range(num_blocks):
            z_block = tree[(level, b_idx)]
            tree[(level + 1, 2 * b_idx)]     = z_block[:half]
            tree[(level + 1, 2 * b_idx + 1)] = z_block[half:]
        level += 1
        block_size //= 2
    return tree


# ─────────────────────────────────────────────────────────────────────────────
#  Butterfly bit operations
# ─────────────────────────────────────────────────────────────────────────────

def _butterfly_ops(x_int: int, half_k: int):
    """
    Apply polar butterfly to a bit-packed integer.

    Given x_int encoding [x_0, x_1, ..., x_{2*half_k - 1}] MSB-first:
        x_bar[m]    = x[2m] ^ x[2m+1]   (XOR pairs)
        x_barbar[m] = x[2m+1]            (right halves)

    Returns (x_bar_int, x_barbar_int) packed MSB-first.
    """
    k = 2 * half_k
    x_bar = x_barbar = 0
    for m in range(half_k):
        b_even = (x_int >> (k - 1 - 2 * m)) & 1
        b_odd  = (x_int >> (k - 2 - 2 * m)) & 1
        x_bar    = (x_bar    << 1) | (b_even ^ b_odd)
        x_barbar = (x_barbar << 1) | b_odd
    return x_bar, x_barbar


# ─────────────────────────────────────────────────────────────────────────────
#  Recursive joint probability  W_N^{(i,j)}  — log domain
# ─────────────────────────────────────────────────────────────────────────────

def _W_joint_log(N: int, i: int, j: int,
                 z_tree: dict, level: int, block_idx: int,
                 u_int: int, v_int: int,
                 channel,
                 cache: dict) -> float:
    """
    log W_N^{(i,j)}(z^N, u^{i-1}, v^{j-1} | u_i, v_j)  — memoized recursion.

    Key = (level, block_idx, i, j, u_int, v_int).  All fields are O(1) to hash.
    Returns -inf when the probability is exactly zero.
    """
    key = (level, block_idx, i, j, u_int, v_int)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if N == 1:
        z_val = z_tree[(level, block_idx)][0]
        p = channel.transition_prob(z_val, u_int, v_int)
        result = np.log(p) if p > 0.0 else -np.inf
        cache[key] = result
        return result

    half_N  = N >> 1
    half_i  = (i + 1) >> 1
    half_j  = (j + 1) >> 1
    i_odd   = i & 1
    j_odd   = j & 1
    nxt_lv  = level + 1
    left_b  = block_idx << 1
    right_b = left_b | 1

    log_terms = []
    for u_x in ([0, 1] if i_odd else [None]):
        u_ext = ((u_int << 1) | u_x) if i_odd else u_int
        u_bar, u_barbar = _butterfly_ops(u_ext, half_i)

        for v_x in ([0, 1] if j_odd else [None]):
            v_ext = ((v_int << 1) | v_x) if j_odd else v_int
            v_bar, v_barbar = _butterfly_ops(v_ext, half_j)

            lw1 = _W_joint_log(half_N, half_i, half_j, z_tree, nxt_lv, left_b,
                               u_bar, v_bar, channel, cache)
            lw2 = _W_joint_log(half_N, half_i, half_j, z_tree, nxt_lv, right_b,
                               u_barbar, v_barbar, channel, cache)
            log_terms.append(lw1 + lw2)

    result = _LOG_025 + float(np.logaddexp.reduce(log_terms))
    cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Recursive joint probability — linear domain (use only for small N)
# ─────────────────────────────────────────────────────────────────────────────

def _W_joint(N: int, i: int, j: int,
             z_tree: dict, level: int, block_idx: int,
             u_int: int, v_int: int,
             channel,
             cache: dict) -> float:
    """
    Linear-domain version of _W_joint_log.  Faster for small N (no logaddexp).
    WARNING: underflows to 0.0 for large N or noisy channels.
    """
    key = (level, block_idx, i, j, u_int, v_int)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if N == 1:
        z_val = z_tree[(level, block_idx)][0]
        result = channel.transition_prob(z_val, u_int, v_int)
        cache[key] = result
        return result

    half_N  = N >> 1
    half_i  = (i + 1) >> 1
    half_j  = (j + 1) >> 1
    i_odd   = i & 1
    j_odd   = j & 1
    nxt_lv  = level + 1
    left_b  = block_idx << 1
    right_b = left_b | 1

    total = 0.0
    for u_x in ([0, 1] if i_odd else [None]):
        u_ext = ((u_int << 1) | u_x) if i_odd else u_int
        u_bar, u_barbar = _butterfly_ops(u_ext, half_i)
        for v_x in ([0, 1] if j_odd else [None]):
            v_ext = ((v_int << 1) | v_x) if j_odd else v_int
            v_bar, v_barbar = _butterfly_ops(v_ext, half_j)
            w1 = _W_joint(half_N, half_i, half_j, z_tree, nxt_lv, left_b,
                          u_bar, v_bar, channel, cache)
            w2 = _W_joint(half_N, half_i, half_j, z_tree, nxt_lv, right_b,
                          u_barbar, v_barbar, channel, cache)
            total += 0.25 * w1 * w2

    cache[key] = total
    return total


# ─────────────────────────────────────────────────────────────────────────────
#  Coordinate channel probabilities  (Eqs. 13 / 14 of Önay 2013)
# ─────────────────────────────────────────────────────────────────────────────

def _coord_prob_u_log(N, i, j, z_tree, u_hat, v_hat, u_test, channel, cache):
    """log P(u_i = u_test | z^N, û^{i-1}, v̂^j)."""
    u_int = 0
    for k in range(1, i):
        u_int = (u_int << 1) | u_hat[k]
    u_int = (u_int << 1) | u_test

    if j == 0:
        lw0 = _W_joint_log(N, i, 1, z_tree, 0, 0, u_int, 0, channel, cache)
        lw1 = _W_joint_log(N, i, 1, z_tree, 0, 0, u_int, 1, channel, cache)
        return _LOG_05 + float(np.logaddexp(lw0, lw1))
    else:
        v_int = 0
        for k in range(1, j):
            v_int = (v_int << 1) | v_hat[k]
        v_int = (v_int << 1) | v_hat[j]
        return _LOG_05 + _W_joint_log(N, i, j, z_tree, 0, 0, u_int, v_int, channel, cache)


def _coord_prob_v_log(N, i, j, z_tree, u_hat, v_hat, v_test, channel, cache):
    """log P(v_j = v_test | z^N, û^i, v̂^{j-1})."""
    v_int = 0
    for k in range(1, j):
        v_int = (v_int << 1) | v_hat[k]
    v_int = (v_int << 1) | v_test

    if i == 0:
        lw0 = _W_joint_log(N, 1, j, z_tree, 0, 0, 0, v_int, channel, cache)
        lw1 = _W_joint_log(N, 1, j, z_tree, 0, 0, 1, v_int, channel, cache)
        return _LOG_05 + float(np.logaddexp(lw0, lw1))
    else:
        u_int = 0
        for k in range(1, i):
            u_int = (u_int << 1) | u_hat[k]
        u_int = (u_int << 1) | u_hat[i]
        return _LOG_05 + _W_joint_log(N, i, j, z_tree, 0, 0, u_int, v_int, channel, cache)


def _coord_prob_u(N, i, j, z_tree, u_hat, v_hat, u_test, channel, cache):
    """Linear-domain P(u_i = u_test | ...)."""
    u_int = 0
    for k in range(1, i):
        u_int = (u_int << 1) | u_hat[k]
    u_int = (u_int << 1) | u_test

    if j == 0:
        return 0.5 * _W_joint(N, i, 1, z_tree, 0, 0, u_int, 0, channel, cache) + \
               0.5 * _W_joint(N, i, 1, z_tree, 0, 0, u_int, 1, channel, cache)
    else:
        v_int = 0
        for k in range(1, j):
            v_int = (v_int << 1) | v_hat[k]
        v_int = (v_int << 1) | v_hat[j]
        return 0.5 * _W_joint(N, i, j, z_tree, 0, 0, u_int, v_int, channel, cache)


def _coord_prob_v(N, i, j, z_tree, u_hat, v_hat, v_test, channel, cache):
    """Linear-domain P(v_j = v_test | ...)."""
    v_int = 0
    for k in range(1, j):
        v_int = (v_int << 1) | v_hat[k]
    v_int = (v_int << 1) | v_test

    if i == 0:
        return 0.5 * _W_joint(N, 1, j, z_tree, 0, 0, 0, v_int, channel, cache) + \
               0.5 * _W_joint(N, 1, j, z_tree, 0, 0, 1, v_int, channel, cache)
    else:
        u_int = 0
        for k in range(1, i):
            u_int = (u_int << 1) | u_hat[k]
        u_int = (u_int << 1) | u_hat[i]
        return 0.5 * _W_joint(N, i, j, z_tree, 0, 0, u_int, v_int, channel, cache)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-sample SC MAC decoder  (Algorithm 1 — Önay 2013)
# ─────────────────────────────────────────────────────────────────────────────

def decode_single(N: int, z, b: list, frozen_u: dict, frozen_v: dict,
                  channel, log_domain: bool = True):
    """
    SC MAC decoder for one received vector z^N.

    Parameters
    ----------
    N         : int — block length
    z         : list, length N — channel output symbols
    b         : list[int], length 2N — path vector (0=U step, 1=V step)
    frozen_u  : dict {1-indexed position: value} — U frozen bits
    frozen_v  : dict {1-indexed position: value} — V frozen bits
    channel   : MACChannel — implements transition_prob(z, x, y)
    log_domain: bool — use log-domain arithmetic (required for N≥128 ABN-MAC)

    Returns
    -------
    u_dec : list[int] of length N — decoded U bits (0-indexed)
    v_dec : list[int] of length N — decoded V bits (0-indexed)
    """
    if log_domain:
        coord_u = _coord_prob_u_log
        coord_v = _coord_prob_v_log
        cmp = lambda p1, p0: p1 > p0   # log-domain: p1>p0 ↔ log(p1)>log(p0)
    else:
        coord_u = _coord_prob_u
        coord_v = _coord_prob_v
        cmp = lambda p1, p0: p1 > p0

    u_hat = {}   # 1-indexed running decisions for U
    v_hat = {}   # 1-indexed running decisions for V
    i, j  = 0, 0
    cache  = {}
    z_tree = _build_z_tree(list(z))

    for k in range(1, 2 * N + 1):
        bk = b[k - 1]

        if bk == 0:           # U step: decide u_{i+1}
            i += 1
            p0 = coord_u(N, i, j, z_tree, u_hat, v_hat, 0, channel, cache)
            p1 = coord_u(N, i, j, z_tree, u_hat, v_hat, 1, channel, cache)
            u_hat[i] = frozen_u[i] if i in frozen_u else (1 if cmp(p1, p0) else 0)

        else:                  # V step: decide v_{j+1}
            j += 1
            p0 = coord_v(N, i, j, z_tree, u_hat, v_hat, 0, channel, cache)
            p1 = coord_v(N, i, j, z_tree, u_hat, v_hat, 1, channel, cache)
            v_hat[j] = frozen_v[j] if j in frozen_v else (1 if cmp(p1, p0) else 0)

    u_dec = [u_hat[k] for k in range(1, N + 1)]
    v_dec = [v_hat[k] for k in range(1, N + 1)]
    return u_dec, v_dec


# ─────────────────────────────────────────────────────────────────────────────
#  Helper for multiprocessing (must be module-level for pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _decode_worker(args):
    N, z, b, frozen_u, frozen_v, channel, log_domain = args
    return decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_batch(N: int, Z_list, b: list, frozen_u: dict, frozen_v: dict,
                 channel, log_domain: bool = True,
                 n_workers: int = 1) -> list:
    """
    Decode a list of received vectors.

    Parameters
    ----------
    N         : int — block length
    Z_list    : list of received vectors, each of length N
    b         : path vector, length 2N
    frozen_u  : dict {1-indexed pos: value}
    frozen_v  : dict {1-indexed pos: value}
    channel   : MACChannel
    log_domain: use log-domain arithmetic (recommended for N≥128 and ABN-MAC)
    n_workers : number of parallel worker processes (1 = sequential)
                Set > 1 for large N to exploit multi-core CPUs.

    Returns
    -------
    results : list of (u_dec, v_dec) tuples, one per input vector
    """
    if n_workers <= 1 or len(Z_list) <= 1:
        return [decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)
                for z in Z_list]

    args = [(N, z, b, frozen_u, frozen_v, channel, log_domain) for z in Z_list]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_decode_worker, args, chunksize=max(1, len(Z_list) // n_workers)))
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  MACPolarDecoder — convenience class
# ─────────────────────────────────────────────────────────────────────────────

class MACPolarDecoder:
    """
    Convenience wrapper around decode_single / decode_batch.

    Parameters
    ----------
    channel    : MACChannel — BE-MAC or ABN-MAC
    log_domain : bool — use log-domain arithmetic (default True; required for ABN-MAC N≥128)
    n_workers  : int — parallel decoder processes (default 1 = sequential)
    """

    def __init__(self, channel, log_domain: bool = True, n_workers: int = 1):
        self.channel    = channel
        self.log_domain = log_domain
        self.n_workers  = n_workers

    def decode(self, N: int, z, b: list, frozen_u: dict, frozen_v: dict):
        """Decode a single received vector. Returns (u_dec, v_dec)."""
        return decode_single(N, z, b, frozen_u, frozen_v,
                             self.channel, self.log_domain)

    def decode_batch(self, N: int, Z_list, b: list, frozen_u: dict, frozen_v: dict):
        """Decode a list of received vectors. Returns list of (u_dec, v_dec)."""
        return decode_batch(N, Z_list, b, frozen_u, frozen_v,
                            self.channel, self.log_domain, self.n_workers)
