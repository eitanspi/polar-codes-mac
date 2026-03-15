"""
decoder_scl.py
==============
Unified SCL (Successive Cancellation List) decoder for two-user MAC polar codes.

Auto-dispatches based on path type:
  - path_i=0 or path_i=N  -> O(L * N log N) efficient SCL decoder
  - intermediate paths     -> O(L * N log N) tensor-based SCL decoder

Batch mode (vectorized=True) runs all codewords in a tight single-process
loop, eliminating multiprocessing overhead.

Public API:
    decode_single_list(N, z, b, frozen_u, frozen_v, channel, log_domain=True, L=4)
    decode_batch_list (N, Z_list, b, frozen_u, frozen_v, channel, log_domain=True,
                       L=4, n_workers=1, vectorized=True)
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor

try:
    from .decoder import (
        build_log_W_leaf,
        build_log_W_leaf_batch,
        _detect_path_i,
        _circ_conv_batch,
        _norm_prod_batch,
    )
    from .encoder import polar_encode
except ImportError:
    from decoder import (
        build_log_W_leaf,
        build_log_W_leaf_batch,
        _detect_path_i,
        _circ_conv_batch,
        _norm_prod_batch,
    )
    from encoder import polar_encode

_NEG_INF = -np.inf


# =============================================================================
#  PART A: O(L * N log N) efficient SCL (extreme paths)
# =============================================================================

# ── Log-probability f/g nodes (vectorised across paths) ──────────────────────

def _f_logprob(La0, La1, Lb0, Lb1):
    """f-node in log-probability domain, vectorised across paths."""
    out0 = np.logaddexp(La0 + Lb0, La1 + Lb1)
    out1 = np.logaddexp(La0 + Lb1, La1 + Lb0)
    return out0, out1


def _g_logprob(La0, La1, Lb0, Lb1, u):
    """g-node in log-probability domain, vectorised across paths."""
    La_u = np.where(u == 0, La0, La1)
    La_1mu = np.where(u == 0, La1, La0)
    out0 = La_u + Lb0
    out1 = La_1mu + Lb1
    return out0, out1


# ── Tal-Vardy log-prob propagation ───────────────────────────────────────────

def _calc_P(lam, phi, P0, P1, C, m):
    """Compute log-probs at layer lam for ALL paths simultaneously."""
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


# ── Leaf log-probability computation ─────────────────────────────────────────

def _u_marginal_logprob(log_W):
    lp0 = np.log(0.5) + np.logaddexp(log_W[:, 0, 0], log_W[:, 0, 1])
    lp1 = np.log(0.5) + np.logaddexp(log_W[:, 1, 0], log_W[:, 1, 1])
    return lp0, lp1


def _v_marginal_logprob(log_W):
    lp0 = np.log(0.5) + np.logaddexp(log_W[:, 0, 0], log_W[:, 1, 0])
    lp1 = np.log(0.5) + np.logaddexp(log_W[:, 0, 1], log_W[:, 1, 1])
    return lp0, lp1


def _v_conditional_logprob(log_W, x_enc):
    idx = np.arange(len(x_enc))
    x = np.asarray(x_enc, dtype=np.intp)
    lp0 = np.log(0.5) + log_W[idx, x, 0]
    lp1 = np.log(0.5) + log_W[idx, x, 1]
    return lp0, lp1


def _u_conditional_logprob(log_W, y_enc):
    idx = np.arange(len(y_enc))
    y = np.asarray(y_enc, dtype=np.intp)
    lp0 = np.log(0.5) + log_W[idx, 0, y]
    lp1 = np.log(0.5) + log_W[idx, 1, y]
    return lp0, lp1


# ── SCL phase decoder ────────────────────────────────────────────────────────

def _scl_decode_phase(N, m, L, leaf_lp0, leaf_lp1, frozen_0idx):
    """SCL decode N bits using log-probability tree, vectorised across paths."""
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
        _calc_P(m, phi, P0, P1, C, m)
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
        _update_C(m, phi, C, m)
        aidx = np.where(active)[0]
        if len(aidx) > 0:
            max_pm = np.max(PM[aidx])
            if max_pm != _NEG_INF:
                PM[aidx] -= max_pm

    return P0, P1, C, PM, bits, active


def _fork_and_prune(phi, P0, P1, C, PM, bits, active, L, m,
                    extra_bits_list=None):
    """Fork each active path into bit=0 and bit=1, prune to best L."""
    max_paths = 2 * L
    aidx = np.where(active)[0]
    n_active = len(aidx)
    if n_active == 0:
        return

    candidates = []
    for l in aidx:
        lp0_val = P0[l, m, 0]
        lp1_val = P1[l, m, 0]
        candidates.append((PM[l] + lp0_val, int(l), 0))
        candidates.append((PM[l] + lp1_val, int(l), 1))

    candidates.sort(key=lambda x: x[0], reverse=True)
    keep = candidates[:min(len(candidates), L)]

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


# ── Efficient SCL: U-first and V-first ───────────────────────────────────────

def _decode_extreme_u_first(N, m, L, log_W, frozen_u, frozen_v):
    """U-first (0^N 1^N): SCL decode U marginal, then V conditional."""
    frozen_u_0 = {k - 1: v for k, v in frozen_u.items()}
    frozen_v_0 = {k - 1: v for k, v in frozen_v.items()}
    max_paths = 2 * L

    lp0, lp1 = _u_marginal_logprob(log_W)
    P0, P1, C, PM, u_bits, active = _scl_decode_phase(
        N, m, L, lp0, lp1, frozen_u_0)

    aidx = np.where(active)[0]
    for l in aidx:
        x_l = np.array(polar_encode(u_bits[l].tolist()), dtype=np.int8)
        vlp0, vlp1 = _v_conditional_logprob(log_W, x_l)
        P0[l, 0, :N] = vlp0
        P1[l, 0, :N] = vlp1
    P0[:, 1:, :] = _NEG_INF
    P1[:, 1:, :] = _NEG_INF
    C[:, :, :, :] = 0

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


def _decode_extreme_v_first(N, m, L, log_W, frozen_u, frozen_v):
    """V-first (1^N 0^N): SCL decode V marginal, then U conditional."""
    frozen_u_0 = {k - 1: v for k, v in frozen_u.items()}
    frozen_v_0 = {k - 1: v for k, v in frozen_v.items()}
    max_paths = 2 * L

    lp0, lp1 = _v_marginal_logprob(log_W)
    P0, P1, C, PM, v_bits, active = _scl_decode_phase(
        N, m, L, lp0, lp1, frozen_v_0)

    aidx = np.where(active)[0]
    for l in aidx:
        y_l = np.array(polar_encode(v_bits[l].tolist()), dtype=np.int8)
        ulp0, ulp1 = _u_conditional_logprob(log_W, y_l)
        P0[l, 0, :N] = ulp0
        P1[l, 0, :N] = ulp1
    P0[:, 1:, :] = _NEG_INF
    P1[:, 1:, :] = _NEG_INF
    C[:, :, :, :] = 0

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


# =============================================================================
#  PART B: O(L * N^2) recursive SCL (intermediate paths fallback)
# =============================================================================

def _decode_recursive_scl(N, z, b, frozen_u, frozen_v, channel, L):
    """
    O(L * N^2) recursive SCL MAC decoder for arbitrary paths.
    Uses array-based probability functions from _decoder_numba.
    """
    try:
        from ._decoder_numba import (
            _build_z_tree,
            _coord_prob_u_log,
            _coord_prob_v_log,
        )
    except ImportError:
        from _decoder_numba import (
            _build_z_tree,
            _coord_prob_u_log,
            _coord_prob_v_log,
        )

    z_tree = _build_z_tree(list(z))
    cache = {}

    paths = [(np.zeros(N + 1, dtype=np.int8),
              np.zeros(N + 1, dtype=np.int8),
              0.0, 0, 0)]

    for k in range(1, 2 * N + 1):
        bk = b[k - 1]

        if bk == 0:
            new_paths = []
            for (u_hat, v_hat, metric, i, j) in paths:
                i_new = i + 1
                is_frozen = i_new in frozen_u

                p0 = _coord_prob_u_log(N, i_new, j, z_tree, u_hat, v_hat,
                                       0, channel, cache)
                p1 = _coord_prob_u_log(N, i_new, j, z_tree, u_hat, v_hat,
                                       1, channel, cache)

                if is_frozen:
                    fval = frozen_u[i_new]
                    u_hat[i_new] = fval
                    m = metric + (p0 if fval == 0 else p1)
                    new_paths.append((u_hat, v_hat, m, i_new, j))
                else:
                    u_hat_0 = u_hat.copy()
                    u_hat_0[i_new] = 0
                    new_paths.append((u_hat_0, v_hat.copy(), metric + p0,
                                      i_new, j))
                    u_hat_1 = u_hat.copy()
                    u_hat_1[i_new] = 1
                    new_paths.append((u_hat_1, v_hat.copy(), metric + p1,
                                      i_new, j))
        else:
            new_paths = []
            for (u_hat, v_hat, metric, i, j) in paths:
                j_new = j + 1
                is_frozen = j_new in frozen_v

                p0 = _coord_prob_v_log(N, i, j_new, z_tree, u_hat, v_hat,
                                       0, channel, cache)
                p1 = _coord_prob_v_log(N, i, j_new, z_tree, u_hat, v_hat,
                                       1, channel, cache)

                if is_frozen:
                    fval = frozen_v[j_new]
                    v_hat[j_new] = fval
                    m = metric + (p0 if fval == 0 else p1)
                    new_paths.append((u_hat, v_hat, m, i, j_new))
                else:
                    v_hat_0 = v_hat.copy()
                    v_hat_0[j_new] = 0
                    new_paths.append((u_hat.copy(), v_hat_0, metric + p0,
                                      i, j_new))
                    v_hat_1 = v_hat.copy()
                    v_hat_1[j_new] = 1
                    new_paths.append((u_hat.copy(), v_hat_1, metric + p1,
                                      i, j_new))

        if len(new_paths) > L:
            new_paths.sort(key=lambda x: x[2], reverse=True)
            new_paths = new_paths[:L]

        max_metric = max(p[2] for p in new_paths)
        if max_metric != _NEG_INF:
            paths = [(u, v, m - max_metric, i, j)
                     for (u, v, m, i, j) in new_paths]
        else:
            paths = new_paths

    best = max(paths, key=lambda x: x[2])
    u_hat, v_hat = best[0], best[1]
    u_dec = [int(u_hat[k]) for k in range(1, N + 1)]
    v_dec = [int(v_hat[k]) for k in range(1, N + 1)]
    return u_dec, v_dec


# =============================================================================
#  PART C: O(L * N log N) tensor-based SCL (all paths)
#  Batched computational graph — all L paths processed simultaneously.
# =============================================================================

_LOG_HALF = np.log(0.5)
_LOG_QUARTER = np.log(0.25)


def _circ_conv_4d(A, B):
    """Circ conv on (L, M, 2, 2) arrays — batch over L*M."""
    s = A.shape
    return _circ_conv_batch(A.reshape(-1, 2, 2),
                            B.reshape(-1, 2, 2)).reshape(s)


def _norm_prod_4d(A, B):
    """Norm prod on (L, M, 2, 2) arrays — batch over L*M."""
    s = A.shape
    return _norm_prod_batch(A.reshape(-1, 2, 2),
                            B.reshape(-1, 2, 2)).reshape(s)


class _BatchCompGraph:
    """
    Computational graph for SC decoding that processes max_paths paths
    simultaneously. All edge data packed into a single contiguous array
    for fast path copying.
    """

    def __init__(self, n, log_W_leaf, max_paths):
        self.n = n
        self.N = 1 << n
        N = self.N
        self.max_paths = max_paths

        # Compute packed layout: offset and size for each edge
        self._off = np.zeros(2 * N, dtype=np.intp)
        self._sz = np.zeros(2 * N, dtype=np.intp)
        total = 0
        for beta in range(1, 2 * N):
            self._off[beta] = total
            level = beta.bit_length() - 1
            self._sz[beta] = N >> level
            total += N >> level

        # Single packed array: (max_paths, total_tensors, 2, 2)
        self._data = np.full((max_paths, total, 2, 2),
                             _LOG_QUARTER, dtype=np.float64)

        # Initialize root
        try:
            from .encoder import bit_reversal_perm
        except ImportError:
            from encoder import bit_reversal_perm
        br = bit_reversal_perm(n)
        root = log_W_leaf[br].copy()
        totals = np.logaddexp(
            np.logaddexp(root[:, 0, 0], root[:, 0, 1]),
            np.logaddexp(root[:, 1, 0], root[:, 1, 1])
        )
        finite = np.isfinite(totals)
        root[finite] -= totals[finite, None, None]
        o1, s1 = int(self._off[1]), int(self._sz[1])
        self._data[:, o1:o1 + s1] = root[None]

        self.dec_head = 1

    def _e(self, beta):
        """View of edge beta: shape (max_paths, M, 2, 2)."""
        o, s = int(self._off[beta]), int(self._sz[beta])
        return self._data[:, o:o + s]

    def calc_left(self, beta):
        parent = self._e(beta)
        right = self._e(2 * beta + 1)
        size = right.shape[1]
        temp = _norm_prod_4d(parent[:, size:], right)
        self._e(2 * beta)[:] = _circ_conv_4d(parent[:, :size], temp)

    def calc_right(self, beta):
        parent = self._e(beta)
        left = self._e(2 * beta)
        size = left.shape[1]
        temp = _circ_conv_4d(left, parent[:, :size])
        self._e(2 * beta + 1)[:] = _norm_prod_4d(parent[:, size:], temp)

    def calc_parent(self, beta):
        left = self._e(2 * beta)
        right = self._e(2 * beta + 1)
        size = left.shape[1]
        parent = self._e(beta)
        parent[:, :size] = _circ_conv_4d(left, right)
        parent[:, size:] = right

    def step_to(self, target):
        current = self.dec_head
        if current == target:
            return
        path = self._get_path(current, target)
        for beta in path:
            self._step_one(beta)
        self.dec_head = target

    def _step_one(self, beta):
        current = self.dec_head
        if current == beta >> 1:
            if beta & 1 == 0:
                self.calc_left(current)
            else:
                self.calc_right(current)
            self.dec_head = beta
        elif beta == current >> 1:
            self.calc_parent(current)
            self.dec_head = beta

    @staticmethod
    def _get_path(current, target):
        if current == target:
            return []
        path_up = []
        path_down = []
        c, t = current, target
        while c != t:
            if c > t:
                c = c >> 1
                path_up.append(c)
            else:
                path_down.append(t)
                t = t >> 1
        path_down.reverse()
        return path_up + path_down

    def copy_path(self, dst, src):
        """Copy all edge data from path src to path dst (single array copy)."""
        self._data[dst] = self._data[src]


def _decode_tensor_scl(N, log_W, b, frozen_u, frozen_v, L):
    """
    O(L * N log N) tensor-based SCL decoder for arbitrary monotone chain paths.
    Returns (u_dec, v_dec) as lists.
    """
    n = N.bit_length() - 1
    max_paths = 2 * L

    graph = _BatchCompGraph(n, log_W, max_paths)
    PM = np.full(max_paths, _NEG_INF, dtype=np.float64)
    PM[0] = 0.0
    active = np.zeros(max_paths, dtype=bool)
    active[0] = True
    # Decided bits stored as (max_paths, N) arrays, 1-indexed via +1 offset
    u_bits = np.zeros((max_paths, N + 1), dtype=np.int8)
    v_bits = np.zeros((max_paths, N + 1), dtype=np.int8)
    # Track which positions have been decided per path
    u_decided = np.zeros((max_paths, N + 1), dtype=bool)
    v_decided = np.zeros((max_paths, N + 1), dtype=bool)

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
        is_left = (leaf_edge & 1 == 0)

        # Navigate (same path for all slots)
        graph.step_to(target_vertex)

        # Save bottom-up data and compute top-down for all paths at once
        leaf_view = graph._e(leaf_edge)
        temp = leaf_view[:, 0].copy()  # (max_paths, 2, 2)
        if is_left:
            graph.calc_left(target_vertex)
        else:
            graph.calc_right(target_vertex)
        leaf_view = graph._e(leaf_edge)
        top_down = leaf_view[:, 0]  # (max_paths, 2, 2)

        # Combined = norm_prod(top_down, temp)
        combined = _norm_prod_batch(top_down, temp)  # (max_paths, 2, 2)

        is_frozen = i_t in frozen_dict

        if is_frozen:
            bit = frozen_dict[i_t]
            aidx = np.where(active)[0]
            # Path metric increment
            if gamma == 0:
                pb = np.logaddexp(combined[aidx, bit, 0],
                                  combined[aidx, bit, 1])
            else:
                pb = np.logaddexp(combined[aidx, 0, bit],
                                  combined[aidx, 1, bit])
            PM[aidx] += pb

            # Record decision
            if gamma == 0:
                u_bits[aidx, i_t] = bit
                u_decided[aidx, i_t] = True
            else:
                v_bits[aidx, i_t] = bit
                v_decided[aidx, i_t] = True

            # Update leaf tensors for active paths
            _set_leaves_batched(graph, leaf_edge, i_t, aidx,
                                u_bits, v_bits, u_decided, v_decided)
        else:
            aidx = np.where(active)[0]

            # Compute metrics for bit=0 and bit=1 for all active paths
            if gamma == 0:
                pb0 = np.logaddexp(combined[aidx, 0, 0], combined[aidx, 0, 1])
                pb1 = np.logaddexp(combined[aidx, 1, 0], combined[aidx, 1, 1])
            else:
                pb0 = np.logaddexp(combined[aidx, 0, 0], combined[aidx, 1, 0])
                pb1 = np.logaddexp(combined[aidx, 0, 1], combined[aidx, 1, 1])

            met0 = PM[aidx] + pb0
            met1 = PM[aidx] + pb1

            # Build candidate list interleaved: (path0_b0, path0_b1, path1_b0, ...)
            n_active = len(aidx)
            all_mets = np.empty(2 * n_active, dtype=np.float64)
            all_srcs = np.empty(2 * n_active, dtype=np.intp)
            all_bits = np.empty(2 * n_active, dtype=np.int8)
            all_mets[0::2] = met0
            all_mets[1::2] = met1
            all_srcs[0::2] = aidx
            all_srcs[1::2] = aidx
            all_bits[0::2] = 0
            all_bits[1::2] = 1

            # Select best L (stable sort: bit=0 before bit=1 on ties)
            n_keep = min(len(all_mets), L)
            order = np.argsort(-all_mets, kind='stable')
            top_idx = order[:n_keep]

            keep_mets = all_mets[top_idx]
            keep_srcs = all_srcs[top_idx]
            keep_bits = all_bits[top_idx]

            # Assign to slots
            src_used = set()
            inactive_iter = iter(list(np.where(~active)[0]))
            assignments = []  # (dst, src, met, bit, need_copy)

            for k in range(n_keep):
                src = int(keep_srcs[k])
                if src not in src_used:
                    src_used.add(src)
                    assignments.append((src, src, keep_mets[k],
                                        keep_bits[k], False))
                else:
                    dst = next(inactive_iter)
                    assignments.append((dst, src, keep_mets[k],
                                        keep_bits[k], True))

            # Copy paths before updates
            for dst, src, _, _, need_copy in assignments:
                if need_copy:
                    graph.copy_path(dst, src)
                    u_bits[dst] = u_bits[src]
                    v_bits[dst] = v_bits[src]
                    u_decided[dst] = u_decided[src]
                    v_decided[dst] = v_decided[src]

            # Deactivate all, activate and update survivors
            active[:] = False
            for dst, _, met, bit, _ in assignments:
                PM[dst] = met
                active[dst] = True
                if gamma == 0:
                    u_bits[dst, i_t] = bit
                    u_decided[dst, i_t] = True
                else:
                    v_bits[dst, i_t] = bit
                    v_decided[dst, i_t] = True

            # Update leaves for all surviving paths
            surv = np.array([a[0] for a in assignments], dtype=np.intp)
            _set_leaves_batched(graph, leaf_edge, i_t, surv,
                                u_bits, v_bits, u_decided, v_decided)

        # Normalize path metrics
        aidx = np.where(active)[0]
        if len(aidx) > 0:
            max_pm = np.max(PM[aidx])
            if np.isfinite(max_pm):
                PM[aidx] -= max_pm

    # Return best path
    aidx = np.where(active)[0]
    best_l = aidx[np.argmax(PM[aidx])]
    u_dec = u_bits[best_l, 1:N + 1].tolist()
    v_dec = v_bits[best_l, 1:N + 1].tolist()
    return u_dec, v_dec


def _set_leaves_batched(graph, leaf_edge, i_t, path_indices,
                        u_bits, v_bits, u_decided, v_decided):
    """Update leaf tensors for the given path indices."""
    leaf_view = graph._e(leaf_edge)
    for l in path_indices:
        u_known = u_decided[l, i_t]
        v_known = v_decided[l, i_t]
        new_leaf = np.full((2, 2), _NEG_INF, dtype=np.float64)
        if u_known and v_known:
            new_leaf[u_bits[l, i_t], v_bits[l, i_t]] = 0.0
        elif u_known:
            new_leaf[u_bits[l, i_t], 0] = _LOG_HALF
            new_leaf[u_bits[l, i_t], 1] = _LOG_HALF
        elif v_known:
            new_leaf[0, v_bits[l, i_t]] = _LOG_HALF
            new_leaf[1, v_bits[l, i_t]] = _LOG_HALF
        else:
            new_leaf[:, :] = _LOG_QUARTER
        leaf_view[l, 0] = new_leaf


# =============================================================================
#  Public API
# =============================================================================

def decode_single_list(N: int, z, b: list, frozen_u: dict, frozen_v: dict,
                       channel, log_domain: bool = True, L: int = 4):
    """
    SCL MAC decoder for one received vector z^N.

    Auto-dispatches:
      - path_i=0 or path_i=N  -> O(L * N log N) efficient SCL
      - intermediate paths     -> O(L * N log N) tensor-based SCL

    Parameters
    ----------
    N         : int -- block length (power of 2)
    z         : list, length N -- channel output symbols
    b         : list[int], length 2N -- path vector
    frozen_u  : dict {1-indexed position: value}
    frozen_v  : dict {1-indexed position: value}
    channel   : MACChannel
    log_domain: bool -- must be True
    L         : int -- list size

    Returns
    -------
    u_dec : list[int] of length N
    v_dec : list[int] of length N
    """
    if not log_domain:
        raise ValueError("SCL decoder requires log_domain=True")

    path_i = _detect_path_i(N, b)

    log_W = build_log_W_leaf(z, channel)

    if path_i == 0 or path_i == N:
        m = N.bit_length() - 1
        if path_i == N:
            return _decode_extreme_u_first(N, m, L, log_W, frozen_u, frozen_v)
        else:
            return _decode_extreme_v_first(N, m, L, log_W, frozen_u, frozen_v)
    else:
        return _decode_tensor_scl(N, log_W, b, frozen_u, frozen_v, L)


def _decode_list_worker(args):
    N, z, b, frozen_u, frozen_v, channel, log_domain, L = args
    return decode_single_list(N, z, b, frozen_u, frozen_v, channel, log_domain, L)


def decode_batch_list(N: int, Z_list, b: list, frozen_u: dict, frozen_v: dict,
                      channel, log_domain: bool = True, L: int = 4,
                      n_workers: int = 1, vectorized: bool = True) -> list:
    """
    Decode a list of received vectors using the SCL decoder.

    Parameters
    ----------
    N, b, frozen_u, frozen_v, channel, log_domain, L : same as decode_single_list
    Z_list     : list of received vectors
    n_workers  : int -- parallel worker processes (only used when vectorized=False)
    vectorized : bool -- if True (default), decode in a tight single-process
                 loop with pre-built log_W arrays (avoids multiprocessing overhead)

    Returns
    -------
    results : list of (u_dec, v_dec) tuples
    """
    if vectorized and len(Z_list) >= 1:
        path_i = _detect_path_i(N, b)
        log_W_batch = build_log_W_leaf_batch(Z_list, channel)
        m = N.bit_length() - 1

        results = []
        for i in range(log_W_batch.shape[0]):
            log_W = log_W_batch[i]
            if path_i == 0 or path_i == N:
                if path_i == N:
                    r = _decode_extreme_u_first(N, m, L, log_W,
                                                frozen_u, frozen_v)
                else:
                    r = _decode_extreme_v_first(N, m, L, log_W,
                                                frozen_u, frozen_v)
            else:
                r = _decode_tensor_scl(N, log_W, b, frozen_u, frozen_v, L)
            results.append(r)
        return results

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
