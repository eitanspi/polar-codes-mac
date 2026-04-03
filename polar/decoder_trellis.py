"""
decoder_trellis.py
==================
SC decoder for two-user MAC polar codes on channels with memory (ISI, etc.).

Approach: Forward-backward trellis equalization produces per-position
(N, 2, 2) marginal log-likelihoods that account for the state chain across
all N positions. These effective memoryless marginals are then fed into the
standard computational-graph SC MAC decoder from decoder_interleaved.py.

The FB algorithm computes:
    log_marginal[t, x, y] = log sum_{states} alpha[t] * W(z_t|x,y,s) * beta[t+1]

This properly accounts for ISI through the state chain.

For Class C/A paths, an iterative refinement optionally conditions the FB
on the first-decoded user's codeword before decoding the second user.

Public API:
    decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True)
    decode_batch (N, Z_list, b, frozen_u, frozen_v, channel, log_domain=True,
                  n_workers=1)
"""

import numpy as np

_NEG_INF = -np.inf


# ─────────────────────────────────────────────────────────────────────────────
#  Forward-backward on the ISI trellis
# ─────────────────────────────────────────────────────────────────────────────

def _forward_backward_joint(log_W, N, S):
    """
    FB on the full MAC trellis producing (N, 2, 2) marginal log-likelihoods.
    Uses uniform P(x,y) = 1/4 at each position.
    """
    log_quarter = np.log(0.25)

    alpha = np.full((N + 1, S), _NEG_INF, dtype=np.float64)
    alpha[0, 0] = 0.0

    for t in range(N):
        for s in range(S):
            if not np.isfinite(alpha[t, s]):
                continue
            for x in range(2):
                for y in range(2):
                    for sp in range(S):
                        val = log_W[t, x, y, s, sp]
                        if np.isfinite(val):
                            alpha[t + 1, sp] = np.logaddexp(
                                alpha[t + 1, sp],
                                alpha[t, s] + val + log_quarter)

    beta = np.full((N + 1, S), _NEG_INF, dtype=np.float64)
    beta[N, :] = 0.0

    for t in range(N - 1, -1, -1):
        for s in range(S):
            for x in range(2):
                for y in range(2):
                    for sp in range(S):
                        val = log_W[t, x, y, s, sp]
                        if np.isfinite(val) and np.isfinite(beta[t + 1, sp]):
                            beta[t, s] = np.logaddexp(
                                beta[t, s],
                                val + log_quarter + beta[t + 1, sp])

    log_marginal = np.full((N, 2, 2), _NEG_INF, dtype=np.float64)
    for t in range(N):
        for x in range(2):
            for y in range(2):
                for s in range(S):
                    if not np.isfinite(alpha[t, s]):
                        continue
                    for sp in range(S):
                        val = log_W[t, x, y, s, sp]
                        if np.isfinite(val) and np.isfinite(beta[t + 1, sp]):
                            log_marginal[t, x, y] = np.logaddexp(
                                log_marginal[t, x, y],
                                alpha[t, s] + val + beta[t + 1, sp])

    return log_marginal


def _forward_backward_conditioned(log_W, N, S, known_codeword, known_user):
    """
    FB conditioned on one user's codeword, producing (N, 2, 2) marginals.

    Parameters
    ----------
    known_codeword  : list[int] length N — the known user's ENCODED bits
    known_user      : 0 = U is known (decode V), 1 = V is known (decode U)
    """
    log_half = np.log(0.5)

    alpha = np.full((N + 1, S), _NEG_INF, dtype=np.float64)
    alpha[0, 0] = 0.0

    for t in range(N):
        known_bit = int(known_codeword[t])
        for s in range(S):
            if not np.isfinite(alpha[t, s]):
                continue
            for bit in range(2):
                if known_user == 0:
                    x, y = known_bit, bit
                else:
                    x, y = bit, known_bit
                for sp in range(S):
                    val = log_W[t, x, y, s, sp]
                    if np.isfinite(val):
                        alpha[t + 1, sp] = np.logaddexp(
                            alpha[t + 1, sp],
                            alpha[t, s] + val + log_half)

    beta = np.full((N + 1, S), _NEG_INF, dtype=np.float64)
    beta[N, :] = 0.0

    for t in range(N - 1, -1, -1):
        known_bit = int(known_codeword[t])
        for s in range(S):
            for bit in range(2):
                if known_user == 0:
                    x, y = known_bit, bit
                else:
                    x, y = bit, known_bit
                for sp in range(S):
                    val = log_W[t, x, y, s, sp]
                    if np.isfinite(val) and np.isfinite(beta[t + 1, sp]):
                        beta[t, s] = np.logaddexp(
                            beta[t, s],
                            val + log_half + beta[t + 1, sp])

    log_marginal = np.full((N, 2, 2), _NEG_INF, dtype=np.float64)
    for t in range(N):
        known_bit = int(known_codeword[t])
        for bit in range(2):
            if known_user == 0:
                x, y = known_bit, bit
            else:
                x, y = bit, known_bit
            for s in range(S):
                if not np.isfinite(alpha[t, s]):
                    continue
                for sp in range(S):
                    val = log_W[t, x, y, s, sp]
                    if np.isfinite(val) and np.isfinite(beta[t + 1, sp]):
                        log_marginal[t, x, y] = np.logaddexp(
                            log_marginal[t, x, y],
                            alpha[t, s] + val + beta[t + 1, sp])

    return log_marginal


# ─────────────────────────────────────────────────────────────────────────────
#  Memoryless SC decode using interleaved computational graph
# ─────────────────────────────────────────────────────────────────────────────

def _decode_with_marginals(N, log_marginal, b, frozen_u, frozen_v):
    """
    SC MAC decode using (N, 2, 2) marginal tensors via the interleaved
    computational graph.
    """
    from polar.decoder_interleaved import (
        _CompGraph, _norm_prod_single, _LOG_HALF, _LOG_QUARTER
    )

    n = N.bit_length() - 1
    graph = _CompGraph(n, log_marginal)

    u_hat = {}
    v_hat = {}
    i_u = 0
    i_v = 0

    for step in range(2 * N):
        gamma = b[step]

        if gamma == 0:
            i_u += 1
            i_t = i_u
            frozen_dict = frozen_u
        else:
            i_v += 1
            i_t = i_v
            frozen_dict = frozen_v

        leaf_edge = i_t + N - 1
        target_vertex = leaf_edge >> 1

        graph.step_to(target_vertex)

        temp = graph.edge_data[leaf_edge][0].copy()

        if leaf_edge & 1 == 0:
            graph.calc_left(target_vertex)
        else:
            graph.calc_right(target_vertex)

        top_down = graph.edge_data[leaf_edge][0]
        combined = _norm_prod_single(top_down, temp)

        if i_t in frozen_dict:
            bit = frozen_dict[i_t]
        else:
            if gamma == 0:
                p0 = np.logaddexp(combined[0, 0], combined[0, 1])
                p1 = np.logaddexp(combined[1, 0], combined[1, 1])
                bit = 1 if p1 > p0 else 0
            else:
                p0 = np.logaddexp(combined[0, 0], combined[1, 0])
                p1 = np.logaddexp(combined[0, 1], combined[1, 1])
                bit = 1 if p1 > p0 else 0

        if gamma == 0:
            u_hat[i_t] = bit
        else:
            v_hat[i_t] = bit

        new_leaf = np.full((2, 2), _NEG_INF, dtype=np.float64)
        u_val = u_hat.get(i_t)
        v_val = v_hat.get(i_t)

        if u_val is not None and v_val is not None:
            new_leaf[u_val, v_val] = 0.0
        elif u_val is not None:
            new_leaf[u_val, 0] = _LOG_HALF
            new_leaf[u_val, 1] = _LOG_HALF
        elif v_val is not None:
            new_leaf[0, v_val] = _LOG_HALF
            new_leaf[1, v_val] = _LOG_HALF
        else:
            new_leaf[:, :] = _LOG_QUARTER

        graph.edge_data[leaf_edge][0] = new_leaf

    u_dec = [u_hat.get(k, 0) for k in range(1, N + 1)]
    v_dec = [v_hat.get(k, 0) for k in range(1, N + 1)]
    return u_dec, v_dec


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True,
                  n_iter=0):
    """
    SC MAC decoder for channels with memory (ISI).

    Stage 1: FB on full MAC trellis -> (N, 2, 2) marginals -> SC decode
    Stage 2 (if n_iter > 0): Iteratively condition FB on decoded users.

    Parameters
    ----------
    N         : int — block length (power of 2)
    z         : array-like, length N — channel output symbols
    b         : list[int], length 2N — path vector (0=U step, 1=V step)
    frozen_u  : dict {1-indexed position: value} — U frozen bits
    frozen_v  : dict {1-indexed position: value} — V frozen bits
    channel   : ISIMAC or similar channel with memory
    log_domain: bool — ignored
    n_iter    : int — number of iterative refinement passes (default 0)

    Returns
    -------
    u_dec : list[int] of length N — decoded U bits (0-indexed)
    v_dec : list[int] of length N — decoded V bits (0-indexed)
    """
    n = N.bit_length() - 1
    assert (1 << n) == N

    S = channel.num_states
    log_W = channel.build_leaf_tensors(z)  # (N, 2, 2, S, S)

    # Stage 1: Joint FB -> marginals -> SC decode
    log_marginal = _forward_backward_joint(log_W, N, S)
    u_dec, v_dec = _decode_with_marginals(N, log_marginal, b, frozen_u, frozen_v)

    if n_iter <= 0:
        return u_dec, v_dec

    # Iterative refinement for Class C/A paths
    from polar.design import make_path
    from polar.encoder import polar_encode

    b_classC = make_path(N, N)
    b_classA = make_path(N, 0)

    for it in range(n_iter):
        if b == b_classC:
            # Condition on U -> re-decode V
            u_cw = polar_encode(u_dec)
            log_marg_v = _forward_backward_conditioned(
                log_W, N, S, u_cw, known_user=0)
            fu_fixed = {i: u_dec[i - 1] for i in range(1, N + 1)}
            _, v_dec = _decode_with_marginals(
                N, log_marg_v, b, fu_fixed, frozen_v)

            # Condition on V -> re-decode U
            v_cw = polar_encode(v_dec)
            log_marg_u = _forward_backward_conditioned(
                log_W, N, S, v_cw, known_user=1)
            fv_fixed = {i: v_dec[i - 1] for i in range(1, N + 1)}
            u_dec, _ = _decode_with_marginals(
                N, log_marg_u, b, frozen_u, fv_fixed)

        elif b == b_classA:
            # Condition on V -> re-decode U
            v_cw = polar_encode(v_dec)
            log_marg_u = _forward_backward_conditioned(
                log_W, N, S, v_cw, known_user=1)
            fv_fixed = {i: v_dec[i - 1] for i in range(1, N + 1)}
            u_dec, _ = _decode_with_marginals(
                N, log_marg_u, b, frozen_u, fv_fixed)

            # Condition on U -> re-decode V
            u_cw = polar_encode(u_dec)
            log_marg_v = _forward_backward_conditioned(
                log_W, N, S, u_cw, known_user=0)
            fu_fixed = {i: u_dec[i - 1] for i in range(1, N + 1)}
            _, v_dec = _decode_with_marginals(
                N, log_marg_v, b, fu_fixed, frozen_v)

    return u_dec, v_dec


# ─────────────────────────────────────────────────────────────────────────────
#  Batch decode
# ─────────────────────────────────────────────────────────────────────────────

def _decode_worker(args):
    N, z, b, frozen_u, frozen_v, channel, log_domain = args
    return decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)


def decode_batch(N, Z_list, b, frozen_u, frozen_v, channel,
                 log_domain=True, n_workers=1):
    """Decode a list of received vectors (sequential or multiprocessing)."""
    if n_workers <= 1 or len(Z_list) <= 1:
        return [decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)
                for z in Z_list]
    from concurrent.futures import ProcessPoolExecutor
    args = [(N, z, b, frozen_u, frozen_v, channel, log_domain) for z in Z_list]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_decode_worker, args,
                                    chunksize=max(1, len(Z_list) // n_workers)))
    return results
