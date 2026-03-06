"""
_decoder_numba.py
=================
Numba-accelerated SC MAC decoder for two-user binary-input polar codes.

Internal module — same algorithm as _decoder_base.py (O(N²)),
but uses np.int8 arrays instead of big Python integers, enabling
numba JIT on the butterfly and bit-assembly hot paths.

Users should import from decoder.py instead.
"""

import math
import numpy as np
from numba import njit
from concurrent.futures import ProcessPoolExecutor

_LOG_025 = math.log(0.25)
_LOG_05  = math.log(0.5)
_NEG_INF = float('-inf')
_EMPTY   = np.empty(0, dtype=np.int8)
_ARR_0   = np.array([0], dtype=np.int8)
_ARR_1   = np.array([1], dtype=np.int8)


def _logaddexp(a, b):
    """Fast scalar logaddexp — replaces np.logaddexp for Python floats."""
    if a > b:
        if b == _NEG_INF:
            return a
        return a + math.log1p(math.exp(b - a))
    else:
        if a == _NEG_INF:
            return b
        return b + math.log1p(math.exp(a - b))


def _logaddexp_reduce(terms):
    """Fast reduce of logaddexp over a small list (1-4 elements)."""
    r = terms[0]
    for k in range(1, len(terms)):
        r = _logaddexp(r, terms[k])
    return r


# ─────────────────────────────────────────────────────────────────────────────
#  Z-tree precomputation  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _build_z_tree(z: list) -> dict:
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
#  Butterfly — numba JIT on int8 arrays
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _butterfly_ops(x_arr):
    """
    Polar butterfly on an int8 array of length 2*half_k.

    x_bar[m]    = x[2m] ^ x[2m+1]
    x_barbar[m] = x[2m+1]
    """
    half_k = len(x_arr) // 2
    x_bar    = np.empty(half_k, dtype=np.int8)
    x_barbar = np.empty(half_k, dtype=np.int8)
    for m in range(half_k):
        x_bar[m]    = x_arr[2 * m] ^ x_arr[2 * m + 1]
        x_barbar[m] = x_arr[2 * m + 1]
    return x_bar, x_barbar


@njit(cache=True)
def _extend_arr(arr, bit):
    """Append one bit to an int8 array."""
    n = len(arr)
    out = np.empty(n + 1, dtype=np.int8)
    for k in range(n):
        out[k] = arr[k]
    out[n] = bit
    return out


# Warm up JIT at import time
_butterfly_ops(np.zeros(2, dtype=np.int8))
_extend_arr(np.zeros(1, dtype=np.int8), np.int8(0))


# ─────────────────────────────────────────────────────────────────────────────
#  Recursive joint probability  W_N^{(i,j)}  — log domain
# ─────────────────────────────────────────────────────────────────────────────

def _W_joint_log(N, i, j, z_tree, level, block_idx,
                 u_arr, v_arr, channel, cache):
    """
    log W_N^{(i,j)}  — memoized recursion, array-based.

    u_arr, v_arr are np.int8 arrays of lengths ceil(i/2) and ceil(j/2)
    at each recursion level.
    """
    key = (level, block_idx, i, j, u_arr.tobytes(), v_arr.tobytes())
    cached = cache.get(key)
    if cached is not None:
        return cached

    if N == 1:
        z_val = z_tree[(level, block_idx)][0]
        u_val = int(u_arr[0]) if len(u_arr) > 0 else 0
        v_val = int(v_arr[0]) if len(v_arr) > 0 else 0
        p = channel.transition_prob(z_val, u_val, v_val)
        result = math.log(p) if p > 0.0 else _NEG_INF
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
        # Inline _extend_arr
        if i_odd:
            u_ext = np.empty(len(u_arr) + 1, dtype=np.int8)
            u_ext[:len(u_arr)] = u_arr
            u_ext[len(u_arr)] = u_x
        else:
            u_ext = u_arr
        # Inline _butterfly_ops
        u_bar = u_ext[0::2] ^ u_ext[1::2]
        u_barbar = u_ext[1::2].copy()

        for v_x in ([0, 1] if j_odd else [None]):
            if j_odd:
                v_ext = np.empty(len(v_arr) + 1, dtype=np.int8)
                v_ext[:len(v_arr)] = v_arr
                v_ext[len(v_arr)] = v_x
            else:
                v_ext = v_arr
            v_bar = v_ext[0::2] ^ v_ext[1::2]
            v_barbar = v_ext[1::2].copy()

            lw1 = _W_joint_log(half_N, half_i, half_j, z_tree, nxt_lv, left_b,
                               u_bar, v_bar, channel, cache)
            lw2 = _W_joint_log(half_N, half_i, half_j, z_tree, nxt_lv, right_b,
                               u_barbar, v_barbar, channel, cache)
            log_terms.append(lw1 + lw2)

    result = _LOG_025 + _logaddexp_reduce(log_terms)
    cache[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Recursive joint probability — linear domain
# ─────────────────────────────────────────────────────────────────────────────

def _W_joint(N, i, j, z_tree, level, block_idx,
             u_arr, v_arr, channel, cache):
    """Linear-domain version. Faster for small N, underflows at large N."""
    key = (level, block_idx, i, j, u_arr.tobytes(), v_arr.tobytes())
    cached = cache.get(key)
    if cached is not None:
        return cached

    if N == 1:
        z_val = z_tree[(level, block_idx)][0]
        u_val = int(u_arr[0]) if len(u_arr) > 0 else 0
        v_val = int(v_arr[0]) if len(v_arr) > 0 else 0
        result = channel.transition_prob(z_val, u_val, v_val)
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
        if i_odd:
            u_ext = np.empty(len(u_arr) + 1, dtype=np.int8)
            u_ext[:len(u_arr)] = u_arr
            u_ext[len(u_arr)] = u_x
        else:
            u_ext = u_arr
        u_bar = u_ext[0::2] ^ u_ext[1::2]
        u_barbar = u_ext[1::2].copy()
        for v_x in ([0, 1] if j_odd else [None]):
            if j_odd:
                v_ext = np.empty(len(v_arr) + 1, dtype=np.int8)
                v_ext[:len(v_arr)] = v_arr
                v_ext[len(v_arr)] = v_x
            else:
                v_ext = v_arr
            v_bar = v_ext[0::2] ^ v_ext[1::2]
            v_barbar = v_ext[1::2].copy()
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
    u_arr = np.empty(i, dtype=np.int8)
    u_arr[:i - 1] = u_hat[1:i]
    u_arr[i - 1] = u_test

    if j == 0:
        lw0 = _W_joint_log(N, i, 1, z_tree, 0, 0, u_arr, _ARR_0, channel, cache)
        lw1 = _W_joint_log(N, i, 1, z_tree, 0, 0, u_arr, _ARR_1, channel, cache)
        return _LOG_05 + _logaddexp(lw0, lw1)
    else:
        v_arr = np.empty(j, dtype=np.int8)
        v_arr[:j] = v_hat[1:j + 1]
        return _LOG_05 + _W_joint_log(N, i, j, z_tree, 0, 0, u_arr, v_arr, channel, cache)


def _coord_prob_v_log(N, i, j, z_tree, u_hat, v_hat, v_test, channel, cache):
    """log P(v_j = v_test | z^N, û^i, v̂^{j-1})."""
    v_arr = np.empty(j, dtype=np.int8)
    v_arr[:j - 1] = v_hat[1:j]
    v_arr[j - 1] = v_test

    if i == 0:
        lw0 = _W_joint_log(N, 1, j, z_tree, 0, 0, _ARR_0, v_arr, channel, cache)
        lw1 = _W_joint_log(N, 1, j, z_tree, 0, 0, _ARR_1, v_arr, channel, cache)
        return _LOG_05 + _logaddexp(lw0, lw1)
    else:
        u_arr = np.empty(i, dtype=np.int8)
        u_arr[:i] = u_hat[1:i + 1]
        return _LOG_05 + _W_joint_log(N, i, j, z_tree, 0, 0, u_arr, v_arr, channel, cache)


def _coord_prob_u(N, i, j, z_tree, u_hat, v_hat, u_test, channel, cache):
    """Linear-domain P(u_i = u_test | ...)."""
    u_arr = np.empty(i, dtype=np.int8)
    u_arr[:i - 1] = u_hat[1:i]
    u_arr[i - 1] = u_test

    if j == 0:
        return 0.5 * _W_joint(N, i, 1, z_tree, 0, 0, u_arr, _ARR_0, channel, cache) + \
               0.5 * _W_joint(N, i, 1, z_tree, 0, 0, u_arr, _ARR_1, channel, cache)
    else:
        v_arr = np.empty(j, dtype=np.int8)
        v_arr[:j] = v_hat[1:j + 1]
        return 0.5 * _W_joint(N, i, j, z_tree, 0, 0, u_arr, v_arr, channel, cache)


def _coord_prob_v(N, i, j, z_tree, u_hat, v_hat, v_test, channel, cache):
    """Linear-domain P(v_j = v_test | ...)."""
    v_arr = np.empty(j, dtype=np.int8)
    v_arr[:j - 1] = v_hat[1:j]
    v_arr[j - 1] = v_test

    if i == 0:
        return 0.5 * _W_joint(N, 1, j, z_tree, 0, 0, _ARR_0, v_arr, channel, cache) + \
               0.5 * _W_joint(N, 1, j, z_tree, 0, 0, _ARR_1, v_arr, channel, cache)
    else:
        u_arr = np.empty(i, dtype=np.int8)
        u_arr[:i] = u_hat[1:i + 1]
        return 0.5 * _W_joint(N, i, j, z_tree, 0, 0, u_arr, v_arr, channel, cache)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-sample SC MAC decoder  (Algorithm 1 — Önay 2013)
# ─────────────────────────────────────────────────────────────────────────────

def decode_single(N: int, z, b: list, frozen_u: dict, frozen_v: dict,
                  channel, log_domain: bool = True):
    """
    SC MAC decoder for one received vector z^N.
    Same API as _decoder_base.decode_single — drop-in replacement.
    """
    if log_domain:
        coord_u = _coord_prob_u_log
        coord_v = _coord_prob_v_log
    else:
        coord_u = _coord_prob_u
        coord_v = _coord_prob_v

    # u_hat / v_hat as arrays (1-indexed: index 0 unused)
    u_hat = np.zeros(N + 1, dtype=np.int8)
    v_hat = np.zeros(N + 1, dtype=np.int8)
    i, j  = 0, 0
    cache  = {}
    z_tree = _build_z_tree(list(z))

    for k in range(1, 2 * N + 1):
        bk = b[k - 1]

        if bk == 0:           # U step
            i += 1
            p0 = coord_u(N, i, j, z_tree, u_hat, v_hat, 0, channel, cache)
            p1 = coord_u(N, i, j, z_tree, u_hat, v_hat, 1, channel, cache)
            u_hat[i] = frozen_u[i] if i in frozen_u else (1 if p1 > p0 else 0)

        else:                  # V step
            j += 1
            p0 = coord_v(N, i, j, z_tree, u_hat, v_hat, 0, channel, cache)
            p1 = coord_v(N, i, j, z_tree, u_hat, v_hat, 1, channel, cache)
            v_hat[j] = frozen_v[j] if j in frozen_v else (1 if p1 > p0 else 0)

    u_dec = [int(u_hat[k]) for k in range(1, N + 1)]
    v_dec = [int(v_hat[k]) for k in range(1, N + 1)]
    return u_dec, v_dec


# ─────────────────────────────────────────────────────────────────────────────
#  Batch decoder
# ─────────────────────────────────────────────────────────────────────────────

def _decode_worker(args):
    N, z, b, frozen_u, frozen_v, channel, log_domain = args
    return decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)


def decode_batch(N: int, Z_list, b: list, frozen_u: dict, frozen_v: dict,
                 channel, log_domain: bool = True,
                 n_workers: int = 1) -> list:
    if n_workers <= 1 or len(Z_list) <= 1:
        return [decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)
                for z in Z_list]

    args = [(N, z, b, frozen_u, frozen_v, channel, log_domain) for z in Z_list]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_decode_worker, args,
                                    chunksize=max(1, len(Z_list) // n_workers)))
    return results
