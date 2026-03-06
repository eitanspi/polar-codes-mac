"""
decoder_scl.py
==============
O(L * N log N) SCL (Successive Cancellation List) decoder for two-user
MAC polar codes on extreme paths (path_i=N and path_i=0).

Combines:
  - The efficient O(N log N) tree structure from decoder.py
  - The SCL path-forking logic

Uses absolute log-probability propagation (not LLR) to match the
metric used by _decoder_scl_base exactly.

Public API:
    decode_single_list(N, z, b, frozen_u, frozen_v, channel, log_domain=True, L=4)
    decode_batch_list(N, Z_list, ..., L=4, n_workers=1)
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .decoder import (
    build_log_W_leaf,
    _detect_path_i,
    _u_marginal_llr, _v_marginal_llr,
    _u_conditional_llr, _v_conditional_llr,
)
from .encoder import polar_encode

_NEG_INF = -np.inf


# ─────────────────────────────────────────────────────────────────────────────
#  Log-probability f/g nodes (vectorised across paths)
# ─────────────────────────────────────────────────────────────────────────────

def _f_logprob(La0, La1, Lb0, Lb1):
    """
    f-node in log-probability domain, vectorised.

    P_out(0) = P_L(0)*P_R(0) + P_L(1)*P_R(1)
    P_out(1) = P_L(0)*P_R(1) + P_L(1)*P_R(0)

    All inputs/outputs: log-domain, shape (paths, positions).
    Returns (log P_out(0), log P_out(1)).
    """
    out0 = np.logaddexp(La0 + Lb0, La1 + Lb1)
    out1 = np.logaddexp(La0 + Lb1, La1 + Lb0)
    return out0, out1


def _g_logprob(La0, La1, Lb0, Lb1, u):
    """
    g-node in log-probability domain, vectorised.

    P_out(v) = P_L(u XOR v) * P_R(v)
      v=0: P_out(0) = P_L(u) * P_R(0)
      v=1: P_out(1) = P_L(1-u) * P_R(1)

    u: int8 array shape (paths, positions).
    Returns (log P_out(0), log P_out(1)).
    """
    La_u = np.where(u == 0, La0, La1)      # La[u]
    La_1mu = np.where(u == 0, La1, La0)    # La[1-u]
    out0 = La_u + Lb0
    out1 = La_1mu + Lb1
    return out0, out1


# ─────────────────────────────────────────────────────────────────────────────
#  Path-vectorised Tal-Vardy log-prob propagation
# ─────────────────────────────────────────────────────────────────────────────

def _calc_P(lam, phi, P0, P1, C, m):
    """
    Compute log-probs at layer lam for ALL paths simultaneously.

    P0 : shape (max_paths, m+1, N) — log P(bit=0) at each (layer, position)
    P1 : shape (max_paths, m+1, N) — log P(bit=1)
    C  : shape (max_paths, m+1, N, 2) — partial sums (int8)
    """
    if lam == 0:
        return

    psi = phi >> (m - lam)
    size = 1 << (m - lam)

    if psi % 2 == 0:
        _calc_P(lam - 1, phi, P0, P1, C, m)
        La0 = P0[:, lam - 1, 0:2 * size:2]
        La1 = P1[:, lam - 1, 0:2 * size:2]
        Lb0 = P0[:, lam - 1, 1:2 * size:2]
        Lb1 = P1[:, lam - 1, 1:2 * size:2]
        o0, o1 = _f_logprob(La0, La1, Lb0, Lb1)
        P0[:, lam, :size] = o0
        P1[:, lam, :size] = o1
    else:
        La0 = P0[:, lam - 1, 0:2 * size:2]
        La1 = P1[:, lam - 1, 0:2 * size:2]
        Lb0 = P0[:, lam - 1, 1:2 * size:2]
        Lb1 = P1[:, lam - 1, 1:2 * size:2]
        u_bits = C[:, lam, :size, 0]
        o0, o1 = _g_logprob(La0, La1, Lb0, Lb1, u_bits)
        P0[:, lam, :size] = o0
        P1[:, lam, :size] = o1


def _update_C(lam, phi, C, m):
    """Propagate partial sums for ALL paths simultaneously."""
    if lam == 0:
        return

    psi = phi >> (m - lam)

    if psi % 2 == 1:
        size = 1 << (m - lam)
        s_lower = (phi >> (m - lam + 1)) % 2

        left = C[:, lam, :size, 0]
        right = C[:, lam, :size, 1]

        C[:, lam - 1, 0:2 * size:2, s_lower] = left ^ right
        C[:, lam - 1, 1:2 * size:2, s_lower] = right

        _update_C(lam - 1, phi, C, m)


# ─────────────────────────────────────────────────────────────────────────────
#  Leaf log-probability computation
# ─────────────────────────────────────────────────────────────────────────────

def _u_marginal_logprob(log_W):
    """
    Compute (log P(u=0), log P(u=1)) per position for U-marginal channel.

    log P(u|z_t) = log sum_v W(z_t|u,v) * 0.5

    Returns (lp0, lp1) each shape (N,).
    """
    lp0 = np.log(0.5) + np.logaddexp(log_W[:, 0, 0], log_W[:, 0, 1])
    lp1 = np.log(0.5) + np.logaddexp(log_W[:, 1, 0], log_W[:, 1, 1])
    return lp0, lp1


def _v_marginal_logprob(log_W):
    """
    Compute (log P(v=0), log P(v=1)) per position for V-marginal channel.
    """
    lp0 = np.log(0.5) + np.logaddexp(log_W[:, 0, 0], log_W[:, 1, 0])
    lp1 = np.log(0.5) + np.logaddexp(log_W[:, 0, 1], log_W[:, 1, 1])
    return lp0, lp1


def _v_conditional_logprob(log_W, x_enc):
    """
    Compute (log P(v=0), log P(v=1)) per position, conditioned on known x.

    log P(v=0|z_t, x_t) = log(0.5) + log W(z_t|x_t, 0)
    log P(v=1|z_t, x_t) = log(0.5) + log W(z_t|x_t, 1)
    """
    idx = np.arange(len(x_enc))
    x = np.asarray(x_enc, dtype=np.intp)
    lp0 = np.log(0.5) + log_W[idx, x, 0]
    lp1 = np.log(0.5) + log_W[idx, x, 1]
    return lp0, lp1


def _u_conditional_logprob(log_W, y_enc):
    """
    Compute (log P(u=0), log P(u=1)) per position, conditioned on known y.
    """
    idx = np.arange(len(y_enc))
    y = np.asarray(y_enc, dtype=np.intp)
    lp0 = np.log(0.5) + log_W[idx, 0, y]
    lp1 = np.log(0.5) + log_W[idx, 1, y]
    return lp0, lp1


# ─────────────────────────────────────────────────────────────────────────────
#  SCL phase decoder
# ─────────────────────────────────────────────────────────────────────────────

def _scl_decode_phase(N, m, L, leaf_lp0, leaf_lp1, frozen_0idx):
    """
    SCL decode N bits using log-probability tree, vectorised across paths.

    Parameters
    ----------
    leaf_lp0, leaf_lp1 : shape (N,) — log P(bit=0), log P(bit=1) per leaf
    frozen_0idx : dict {0-indexed position: frozen value}

    Returns
    -------
    P0, P1, C, PM, bits, active — state arrays.
    """
    max_paths = 2 * L

    P0 = np.full((max_paths, m + 1, N), _NEG_INF, dtype=np.float64)
    P1 = np.full((max_paths, m + 1, N), _NEG_INF, dtype=np.float64)
    C = np.zeros((max_paths, m + 1, N, 2), dtype=np.int8)
    PM = np.full(max_paths, _NEG_INF, dtype=np.float64)
    bits = np.zeros((max_paths, N), dtype=np.int8)
    active = np.zeros(max_paths, dtype=bool)

    P0[0, 0, :N] = leaf_lp0
    P1[0, 0, :N] = leaf_lp1
    PM[0] = 0.0
    active[0] = True

    for phi in range(N):
        # Step 1: log-prob propagation
        _calc_P(m, phi, P0, P1, C, m)

        # Step 2: decision
        is_frozen = phi in frozen_0idx

        if is_frozen:
            fval = frozen_0idx[phi]
            aidx = np.where(active)[0]
            if fval == 0:
                PM[aidx] += P0[aidx, m, 0]
            else:
                PM[aidx] += P1[aidx, m, 0]
            bits[aidx, phi] = fval
            C[aidx, m, 0, phi % 2] = fval
        else:
            _fork_and_prune(phi, P0, P1, C, PM, bits, active, L, m)

        # Step 3: partial-sum propagation
        _update_C(m, phi, C, m)

        # Normalise metrics
        aidx = np.where(active)[0]
        if len(aidx) > 0:
            max_pm = np.max(PM[aidx])
            if max_pm != _NEG_INF:
                PM[aidx] -= max_pm

    return P0, P1, C, PM, bits, active


def _fork_and_prune(phi, P0, P1, C, PM, bits, active, L, m,
                    extra_bits_list=None):
    """
    Fork each active path into bit=0 and bit=1, prune to best L.
    Matches _decoder_scl_base's ordering: for each source path in order,
    append bit=0 first, then bit=1. Use stable sort for tie-breaking.

    extra_bits_list: optional list of additional (max_paths, N) arrays to copy
    when forking (e.g. u_bits during V-phase).
    """
    max_paths = 2 * L
    aidx = np.where(active)[0]
    n_active = len(aidx)
    if n_active == 0:
        return

    # Build candidates in same order as old decoder
    candidates = []
    for l in aidx:
        lp0_val = P0[l, m, 0]
        lp1_val = P1[l, m, 0]
        candidates.append((PM[l] + lp0_val, int(l), 0))
        candidates.append((PM[l] + lp1_val, int(l), 1))

    # Stable sort by metric descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    keep = candidates[:min(len(candidates), L)]

    # Assign to slots
    src_used = set()
    inactive_iter = iter(list(np.where(~active)[0]))
    assignments = []

    for met, src, bit in keep:
        if src not in src_used:
            src_used.add(src)
            assignments.append((src, src, met, bit, False))
        else:
            dst = next(inactive_iter)
            assignments.append((dst, src, met, bit, True))

    # Execute copies before modifying
    for dst, src, _, _, nc in assignments:
        if nc:
            P0[dst] = P0[src].copy()
            P1[dst] = P1[src].copy()
            C[dst] = C[src].copy()
            bits[dst] = bits[src].copy()
            if extra_bits_list:
                for eb in extra_bits_list:
                    eb[dst] = eb[src].copy()

    active[:] = False
    for dst, _, met, bit, _ in assignments:
        PM[dst] = met
        bits[dst, phi] = bit
        C[dst, m, 0, phi % 2] = bit
        active[dst] = True


# ─────────────────────────────────────────────────────────────────────────────
#  Main API
# ─────────────────────────────────────────────────────────────────────────────

def decode_single_list(N: int, z, b: list, frozen_u: dict, frozen_v: dict,
                       channel, log_domain: bool = True, L: int = 4):
    """
    O(L * N log N) SCL MAC decoder for one received vector z^N.

    Only supports extreme paths (path_i=0 or path_i=N).

    Parameters
    ----------
    N         : int — block length (power of 2)
    z         : list, length N — channel output symbols
    b         : list[int], length 2N — path vector
    frozen_u  : dict {1-indexed position: value}
    frozen_v  : dict {1-indexed position: value}
    channel   : MACChannel
    log_domain: bool — must be True
    L         : int — list size

    Returns
    -------
    u_dec : list[int] of length N
    v_dec : list[int] of length N
    """
    if not log_domain:
        raise ValueError("SCL decoder requires log_domain=True")

    path_i = _detect_path_i(N, b)
    if path_i != 0 and path_i != N:
        raise ValueError(
            'efficient SCL only supports extreme paths (path_i=0 or path_i=N)')

    log_W = build_log_W_leaf(z, channel)
    m = N.bit_length() - 1

    if path_i == N:
        return _decode_u_first(N, m, L, log_W, frozen_u, frozen_v)
    else:
        return _decode_v_first(N, m, L, log_W, frozen_u, frozen_v)


def _decode_u_first(N, m, L, log_W, frozen_u, frozen_v):
    """U-first (0^N 1^N): SCL decode U marginal, then V conditional."""
    frozen_u_0 = {k - 1: v for k, v in frozen_u.items()}
    frozen_v_0 = {k - 1: v for k, v in frozen_v.items()}
    max_paths = 2 * L

    # Phase 1: SCL decode U
    lp0, lp1 = _u_marginal_logprob(log_W)
    P0, P1, C, PM, u_bits, active = _scl_decode_phase(
        N, m, L, lp0, lp1, frozen_u_0)

    # Transition: for each surviving U path, set up V-conditional leaf probs
    aidx = np.where(active)[0]
    for l in aidx:
        x_l = np.array(polar_encode(u_bits[l].tolist()), dtype=np.int8)
        vlp0, vlp1 = _v_conditional_logprob(log_W, x_l)
        P0[l, 0, :N] = vlp0
        P1[l, 0, :N] = vlp1
    # Reset upper layers and partial sums for V phase
    P0[:, 1:, :] = _NEG_INF
    P1[:, 1:, :] = _NEG_INF
    C[:, :, :, :] = 0

    # Phase 2: SCL decode V
    v_bits = np.zeros((max_paths, N), dtype=np.int8)

    for phi in range(N):
        _calc_P(m, phi, P0, P1, C, m)

        is_frozen = phi in frozen_v_0

        if is_frozen:
            fval = frozen_v_0[phi]
            aidx = np.where(active)[0]
            if fval == 0:
                PM[aidx] += P0[aidx, m, 0]
            else:
                PM[aidx] += P1[aidx, m, 0]
            v_bits[aidx, phi] = fval
            C[aidx, m, 0, phi % 2] = fval
        else:
            _fork_and_prune(phi, P0, P1, C, PM, v_bits, active, L, m,
                            extra_bits_list=[u_bits])

        _update_C(m, phi, C, m)

        aidx = np.where(active)[0]
        if len(aidx) > 0:
            max_pm = np.max(PM[aidx])
            if max_pm != _NEG_INF:
                PM[aidx] -= max_pm

    aidx = np.where(active)[0]
    best_l = aidx[np.argmax(PM[aidx])]
    return u_bits[best_l].tolist(), v_bits[best_l].tolist()


def _decode_v_first(N, m, L, log_W, frozen_u, frozen_v):
    """V-first (1^N 0^N): SCL decode V marginal, then U conditional."""
    frozen_u_0 = {k - 1: v for k, v in frozen_u.items()}
    frozen_v_0 = {k - 1: v for k, v in frozen_v.items()}
    max_paths = 2 * L

    # Phase 1: SCL decode V
    lp0, lp1 = _v_marginal_logprob(log_W)
    P0, P1, C, PM, v_bits, active = _scl_decode_phase(
        N, m, L, lp0, lp1, frozen_v_0)

    # Transition: set up U-conditional leaf probs
    aidx = np.where(active)[0]
    for l in aidx:
        y_l = np.array(polar_encode(v_bits[l].tolist()), dtype=np.int8)
        ulp0, ulp1 = _u_conditional_logprob(log_W, y_l)
        P0[l, 0, :N] = ulp0
        P1[l, 0, :N] = ulp1
    P0[:, 1:, :] = _NEG_INF
    P1[:, 1:, :] = _NEG_INF
    C[:, :, :, :] = 0

    # Phase 2: SCL decode U
    u_bits = np.zeros((max_paths, N), dtype=np.int8)

    for phi in range(N):
        _calc_P(m, phi, P0, P1, C, m)

        is_frozen = phi in frozen_u_0

        if is_frozen:
            fval = frozen_u_0[phi]
            aidx = np.where(active)[0]
            if fval == 0:
                PM[aidx] += P0[aidx, m, 0]
            else:
                PM[aidx] += P1[aidx, m, 0]
            u_bits[aidx, phi] = fval
            C[aidx, m, 0, phi % 2] = fval
        else:
            _fork_and_prune(phi, P0, P1, C, PM, u_bits, active, L, m,
                            extra_bits_list=[v_bits])

        _update_C(m, phi, C, m)

        aidx = np.where(active)[0]
        if len(aidx) > 0:
            max_pm = np.max(PM[aidx])
            if max_pm != _NEG_INF:
                PM[aidx] -= max_pm

    aidx = np.where(active)[0]
    best_l = aidx[np.argmax(PM[aidx])]
    return u_bits[best_l].tolist(), v_bits[best_l].tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  Batch API
# ─────────────────────────────────────────────────────────────────────────────

def _decode_list_worker(args):
    N, z, b, frozen_u, frozen_v, channel, log_domain, L = args
    return decode_single_list(N, z, b, frozen_u, frozen_v, channel, log_domain, L)


def decode_batch_list(N: int, Z_list, b: list, frozen_u: dict, frozen_v: dict,
                      channel, log_domain: bool = True, L: int = 4,
                      n_workers: int = 1) -> list:
    """
    Decode a list of received vectors using the efficient SCL decoder.
    Same API as _decoder_scl_base.decode_batch_list.
    """
    if n_workers <= 1 or len(Z_list) <= 1:
        return [decode_single_list(N, z, b, frozen_u, frozen_v,
                                   channel, log_domain, L)
                for z in Z_list]

    args = [(N, z, b, frozen_u, frozen_v, channel, log_domain, L)
            for z in Z_list]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_decode_list_worker, args,
                                    chunksize=max(1, len(Z_list) // n_workers)))
    return results
