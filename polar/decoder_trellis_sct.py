"""
decoder_trellis_sct.py
======================
SC Trellis (SCT) decoder for two-user MAC polar codes on channels with memory.

Extends the computational graph approach from Ren et al. (2025) to channels
with memory, following the theory from Wang et al. (2015).

Each edge carries (L, 2, 2, S, S) log-probability tensors where:
  - L = number of tensor entries at this tree level
  - (2, 2) = joint (u, v) user bits (XOR convolution domain)
  - (S, S) = (s_in, s_out) boundary states of the block this edge represents

Public API:
    decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True)
    decode_batch (N, Z_list, b, frozen_u, frozen_v, channel, log_domain=True,
                  n_workers=1)
"""

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp

_NEG_INF = -np.inf
_LOG_HALF = np.log(0.5)


# ─────────────────────────────────────────────────────────────────────────────
#  Trellis tensor operations in log-domain
#  All operate on arrays of shape (L, 2, 2, S, S)
# ─────────────────────────────────────────────────────────────────────────────

def _circ_conv_trellis(A, B, S):
    """
    Circular convolution with state contraction for (L, 2, 2, S, S) tensors.

    out[l, a, b, s0, s2] = logsumexp_{a', b', s1} A[l, a^a', b^b', s0, s1] + B[l, a', b', s1, s2]

    XOR convolution over (a, b) AND contraction of intermediate state s1.
    """
    L = A.shape[0]
    out = np.full((L, 2, 2, S, S), _NEG_INF, dtype=np.float64)

    # There are 4 (a', b') pairs to sum over, and S intermediate states s1.
    # For each output (a, b), we need A[a^a', b^b', s0, s1] + B[a', b', s1, s2]
    # summed over a', b', s1.

    for ap in range(2):
        for bp in range(2):
            # B_term[l, s1, s2] = B[l, ap, bp, s1, s2]
            B_term = B[:, ap, bp, :, :]  # (L, S, S)

            for a in range(2):
                for b in range(2):
                    # A_term[l, s0, s1] = A[l, a^ap, b^bp, s0, s1]
                    A_term = A[:, a ^ ap, b ^ bp, :, :]  # (L, S, S)

                    # Contract s1: sum_s1 A[l, s0, s1] + B[l, s1, s2]
                    # A_term: (L, S, S) indexed as [l, s0, s1]
                    # B_term: (L, S, S) indexed as [l, s1, s2]
                    # Result: (L, S, S) indexed as [l, s0, s2]
                    # Use einsum-like contraction via logsumexp over s1

                    # Expand for broadcasting: A[l, s0, s1, newaxis] + B[l, newaxis, s1, s2]
                    contrib = A_term[:, :, :, None] + B_term[:, None, :, :]  # (L, S, S, S) = [l, s0, s1, s2]
                    # logsumexp over s1 (axis=2)
                    contracted = _logsumexp_axis(contrib, axis=2)  # (L, S, S) = [l, s0, s2]

                    out[:, a, b, :, :] = np.logaddexp(out[:, a, b, :, :], contracted)

    return out


def _logsumexp_axis(arr, axis):
    """Numerically stable logsumexp along an axis."""
    m = np.max(arr, axis=axis, keepdims=True)
    m_squeeze = np.squeeze(m, axis=axis)
    # Handle all -inf case
    finite = np.isfinite(m_squeeze)
    result = np.full(m_squeeze.shape, _NEG_INF, dtype=np.float64)
    if np.any(finite):
        exp_sum = np.sum(np.exp(arr - m), axis=axis)
        log_sum = np.log(exp_sum) + m_squeeze
        result[finite] = log_sum[finite]
    return result


def _norm_prod_trellis(A, B):
    """
    Normalized elementwise product for (L, 2, 2, S, S) arrays.
    (A ⊙ B)[...] = A[...] + B[...] - log(sum exp(A+B))
    Normalize per L-entry (sum over a, b, s_in, s_out).
    """
    raw = A + B
    L = raw.shape[0]
    # Flatten over (2, 2, S, S) for each L entry
    flat = raw.reshape(L, -1)
    total = _logsumexp_axis(flat, axis=1)  # (L,)
    finite = np.isfinite(total)
    result = raw.copy()
    result[finite] -= total[finite, None, None, None, None]
    return result


def _norm_prod_trellis_single(A, B):
    """Normalized elementwise product for single (2, 2, S, S) tensors."""
    raw = A + B
    flat = raw.ravel()
    total = _logsumexp_axis(flat[None, :], axis=1)[0]
    if np.isfinite(total):
        return raw - total
    return raw.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Computational graph with trellis tensors (Algorithms 3-6 extended)
# ─────────────────────────────────────────────────────────────────────────────

class _CompGraphTrellis:
    """
    Computational graph for SC decoding of MAC polar codes on channels
    with memory. Same structure as _CompGraph but edges carry
    (L, 2, 2, S, S) tensors.

    Indexing:
      - Edges: 1..2N-1. Edge 1 = root. Edges N..2N-1 = leaves.
      - Vertices: 1..N-1. Vertex beta has edges beta (parent),
        2*beta (left child), 2*beta+1 (right child).
    """

    def __init__(self, n, log_W_leaf, S):
        """
        Parameters
        ----------
        n          : int — log2(N)
        log_W_leaf : ndarray (N, 2, 2, S, S) — channel leaf tensors
        S          : int — number of states
        """
        self.n = n
        self.N = 1 << n
        N = self.N
        self.S = S

        log_uniform = -np.log(4.0 * S * S)

        self.edge_data = [None] * (2 * N)

        # Edge 1 (root): bit-reversed channel posteriors, normalized
        from polar.encoder import bit_reversal_perm
        br = bit_reversal_perm(n)
        root = log_W_leaf[br].copy()  # (N, 2, 2, S, S)
        # Normalize each entry
        for t in range(N):
            flat = root[t].ravel()
            total = _logsumexp_axis(flat[None, :], axis=1)[0]
            if np.isfinite(total):
                root[t] -= total
        self.edge_data[1] = root

        # Initialize all other edges to uniform
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            self.edge_data[beta] = np.full((size, 2, 2, S, S),
                                           log_uniform, dtype=np.float64)

        self.dec_head = 1

    def calc_left(self, beta):
        """calcLeft: left = circ_conv(parent[:l], norm_prod(parent[l:], right))"""
        parent = self.edge_data[beta]
        right = self.edge_data[2 * beta + 1]
        l = right.shape[0]
        temp = _norm_prod_trellis(parent[l:], right)
        self.edge_data[2 * beta] = _circ_conv_trellis(parent[:l], temp, self.S)

    def calc_right(self, beta):
        """calcRight: right = norm_prod(parent[l:], circ_conv(left, parent[:l]))"""
        parent = self.edge_data[beta]
        left = self.edge_data[2 * beta]
        l = left.shape[0]
        temp = _circ_conv_trellis(left, parent[:l], self.S)
        self.edge_data[2 * beta + 1] = _norm_prod_trellis(parent[l:], temp)

    def calc_parent(self, beta):
        """calcParent: parent[:l] = circ_conv(left, right), parent[l:] = right"""
        left = self.edge_data[2 * beta]
        right = self.edge_data[2 * beta + 1]
        l = left.shape[0]
        S = self.S
        parent = np.empty((2 * l, 2, 2, S, S), dtype=np.float64)
        parent[:l] = _circ_conv_trellis(left, right, S)
        parent[l:] = right
        self.edge_data[beta] = parent

    def step_to(self, target):
        """Navigate decHead to target vertex."""
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

    def _get_path(self, current, target):
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


# ─────────────────────────────────────────────────────────────────────────────
#  SC Decoder using computational graph with trellis (Algorithm 4 extended)
# ─────────────────────────────────────────────────────────────────────────────

def decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True):
    """
    O(N log N) SC MAC decoder for channels with memory.

    Parameters
    ----------
    N         : int — block length (power of 2)
    z         : array-like, length N — channel output symbols
    b         : list[int], length 2N — path vector (0=U step, 1=V step)
    frozen_u  : dict {1-indexed position: value} — U frozen bits
    frozen_v  : dict {1-indexed position: value} — V frozen bits
    channel   : ISIMAC or similar channel with build_leaf_tensors()
    log_domain: bool — ignored

    Returns
    -------
    u_dec : list[int] of length N — decoded U bits (0-indexed)
    v_dec : list[int] of length N — decoded V bits (0-indexed)
    """
    n = N.bit_length() - 1
    assert (1 << n) == N

    S = channel.num_states
    log_W = channel.build_leaf_tensors(z)  # (N, 2, 2, S, S)

    graph = _CompGraphTrellis(n, log_W, S)

    log_uniform_state = -np.log(float(S))
    log_uniform_bit_state = -np.log(2.0 * S)

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

        # Save leaf's getAsParent (bottom-up / partial decisions)
        temp = graph.edge_data[leaf_edge][0].copy()  # (2, 2, S, S)

        # Compute top-down at leaf
        if leaf_edge & 1 == 0:
            graph.calc_left(target_vertex)
        else:
            graph.calc_right(target_vertex)

        top_down = graph.edge_data[leaf_edge][0]  # (2, 2, S, S)

        # Combine top-down with saved bottom-up
        combined = _norm_prod_trellis_single(top_down, temp)  # (2, 2, S, S)

        # Make hard decision by marginalizing over states
        # Marginalize over (s_in, s_out) to get (2, 2) bit posteriors
        bit_post = _logsumexp_axis(
            combined.reshape(2, 2, -1), axis=2)  # (2, 2)

        if i_t in frozen_dict:
            bit = frozen_dict[i_t]
        else:
            if gamma == 0:  # U decision: marginalize over V
                p0 = np.logaddexp(bit_post[0, 0], bit_post[0, 1])
                p1 = np.logaddexp(bit_post[1, 0], bit_post[1, 1])
                bit = 1 if p1 > p0 else 0
            else:  # V decision: marginalize over U
                p0 = np.logaddexp(bit_post[0, 0], bit_post[1, 0])
                p1 = np.logaddexp(bit_post[0, 1], bit_post[1, 1])
                bit = 1 if p1 > p0 else 0

        if gamma == 0:
            u_hat[i_t] = bit
        else:
            v_hat[i_t] = bit

        # Set leaf to partially deterministic tensor.
        #
        # CRITICAL: After tree propagation, the leaf's (S, S) dimensions
        # represent BLOCK boundary states, not single-position transitions.
        # The decided leaf should only constrain the bit dimensions and
        # leave states completely uniform. The state chain information
        # comes from the observation tensors (root edge).
        log_uniform_ss = -np.log(float(S * S))
        new_leaf = np.full((2, 2, S, S), _NEG_INF, dtype=np.float64)
        u_val = u_hat.get(i_t)
        v_val = v_hat.get(i_t)

        if u_val is not None and v_val is not None:
            new_leaf[u_val, v_val, :, :] = log_uniform_ss
        elif u_val is not None:
            new_leaf[u_val, :, :, :] = -np.log(2.0 * S * S)
        elif v_val is not None:
            new_leaf[:, v_val, :, :] = -np.log(2.0 * S * S)
        else:
            new_leaf[:, :, :, :] = -np.log(4.0 * S * S)

        graph.edge_data[leaf_edge][0] = new_leaf

    u_dec = [u_hat.get(k, 0) for k in range(1, N + 1)]
    v_dec = [v_hat.get(k, 0) for k in range(1, N + 1)]
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
