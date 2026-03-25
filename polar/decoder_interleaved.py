"""
decoder_interleaved.py
======================
O(N log N) SC decoder for two-user MAC polar codes with arbitrary
monotone chain paths, including intermediate paths like 0^{pi} 1^N 0^{N-pi}.

Implements the computational graph approach from Ren et al. (2025).
Each edge in the binary tree carries an array of 2x2 log-probability tensors.
The stepTo() mechanism allows efficient navigation between any two leaves.

Public API (drop-in for decoder.py):
    decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True)
    decode_batch (N, Z_list, b, frozen_u, frozen_v, channel, log_domain=True,
                  n_workers=1)
"""

import numpy as np

_NEG_INF = -np.inf
_LOG_HALF = np.log(0.5)
_LOG_QUARTER = np.log(0.25)


# ─────────────────────────────────────────────────────────────────────────────
#  Vectorized 2x2 tensor operations in log-domain
#  All operate on arrays of shape (L, 2, 2)
# ─────────────────────────────────────────────────────────────────────────────

def _circ_conv_batch(A, B):
    """
    Vectorized circular convolution ⊛ of (L, 2, 2) log-prob tensor arrays.
    out[a,b] = logaddexp over (a',b') of A[a^a', b^b'] + B[a', b']
    """
    A00 = A[:, 0, 0]; A01 = A[:, 0, 1]
    A10 = A[:, 1, 0]; A11 = A[:, 1, 1]
    B00 = B[:, 0, 0]; B01 = B[:, 0, 1]
    B10 = B[:, 1, 0]; B11 = B[:, 1, 1]

    out = np.empty_like(A)

    # out[0,0] = lae(A00+B00, A01+B01, A10+B10, A11+B11)
    out[:, 0, 0] = np.logaddexp(
        np.logaddexp(A00 + B00, A01 + B01),
        np.logaddexp(A10 + B10, A11 + B11))

    # out[0,1] = lae(A01+B00, A00+B01, A11+B10, A10+B11)
    out[:, 0, 1] = np.logaddexp(
        np.logaddexp(A01 + B00, A00 + B01),
        np.logaddexp(A11 + B10, A10 + B11))

    # out[1,0] = lae(A10+B00, A11+B01, A00+B10, A01+B11)
    out[:, 1, 0] = np.logaddexp(
        np.logaddexp(A10 + B00, A11 + B01),
        np.logaddexp(A00 + B10, A01 + B11))

    # out[1,1] = lae(A11+B00, A10+B01, A01+B10, A00+B11)
    out[:, 1, 1] = np.logaddexp(
        np.logaddexp(A11 + B00, A10 + B01),
        np.logaddexp(A01 + B10, A00 + B11))

    return out


def _norm_prod_batch(A, B):
    """
    Vectorized normalized elementwise product ⊙ of (L, 2, 2) arrays.
    (A ⊙ B)[a,b] = A[a,b] + B[a,b] - log(sum exp(A+B))
    """
    raw = A + B
    # Compute total per tensor: logsumexp over the 4 entries
    total = np.logaddexp(
        np.logaddexp(raw[:, 0, 0], raw[:, 0, 1]),
        np.logaddexp(raw[:, 1, 0], raw[:, 1, 1])
    )  # shape (L,)
    finite = np.isfinite(total)
    result = raw.copy()
    result[finite] -= total[finite, None, None]
    return result


def _norm_prod_single(A, B):
    """Normalized elementwise product for single 2x2 tensors."""
    raw = A + B
    total = np.logaddexp(
        np.logaddexp(raw[0, 0], raw[0, 1]),
        np.logaddexp(raw[1, 0], raw[1, 1])
    )
    if np.isfinite(total):
        return raw - total
    return raw.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Computational graph (Algorithms 3-6 from paper)
# ─────────────────────────────────────────────────────────────────────────────

class _CompGraph:
    """
    Computational graph for SC decoding of monotone chain polar codes.

    Indexing:
      - Edges: 1..2N-1. Edge 1 = root (channel posterior). Edges N..2N-1 = leaves.
      - Vertices: 1..N-1. Vertex β has parent edge β, left child edge 2β,
        right child edge 2β+1.
      - Edge β at level j = floor(log2(β)) stores N/2^j tensors of shape (2,2).

    Edge data invariant (relative to decHead at vertex β):
      - Edge β (parent of decHead): getAsChild result (top-down from root)
      - Edges 2β, 2β+1 (children of decHead): getAsParent results (bottom-up from leaves)

    Operations at vertex β:
      calcLeft:   reads edge[β] + edge[2β+1] → writes edge[2β]     (top-down)
      calcRight:  reads edge[β] + edge[2β]   → writes edge[2β+1]   (top-down)
      calcParent: reads edge[2β] + edge[2β+1] → writes edge[β]     (bottom-up)
    """

    def __init__(self, n, log_W_leaf):
        self.n = n
        self.N = 1 << n
        N = self.N

        # Edge data arrays: edge_data[beta] for beta = 1..2N-1
        self.edge_data = [None] * (2 * N)

        # Edge 1 (root): channel posterior P(X|Z), normalized (Theorem 3).
        # Bit-reverse root entries to align tree pairing with standard SC structure.
        from polar.encoder import bit_reversal_perm
        br = bit_reversal_perm(n)
        root = log_W_leaf[br].copy()  # shape (N, 2, 2), vectorized gather
        # Normalize each 2x2 tensor
        totals = np.logaddexp(
            np.logaddexp(root[:, 0, 0], root[:, 0, 1]),
            np.logaddexp(root[:, 1, 0], root[:, 1, 1])
        )  # shape (N,)
        finite = np.isfinite(totals)
        root[finite] -= totals[finite, None, None]
        self.edge_data[1] = root

        # Edges 2..2N-1: initialized to uniform (getAsParent of undecided leaves)
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            self.edge_data[beta] = np.full((size, 2, 2), _LOG_QUARTER,
                                           dtype=np.float64)

        # decHead: current vertex index (starts at root)
        self.dec_head = 1

    def calc_left(self, beta):
        """
        calcLeft at vertex beta (eq 13). Vectorized over all l entries.
        left = circ_conv(parent[:l], norm_prod(parent[l:], right))
        """
        parent = self.edge_data[beta]
        right = self.edge_data[2 * beta + 1]
        l = right.shape[0]
        temp = _norm_prod_batch(parent[l:], right)
        self.edge_data[2 * beta] = _circ_conv_batch(parent[:l], temp)

    def calc_right(self, beta):
        """
        calcRight at vertex beta (eq 14). Vectorized over all l entries.
        right = norm_prod(parent[l:], circ_conv(left, parent[:l]))
        """
        parent = self.edge_data[beta]
        left = self.edge_data[2 * beta]
        l = left.shape[0]
        temp = _circ_conv_batch(left, parent[:l])
        self.edge_data[2 * beta + 1] = _norm_prod_batch(parent[l:], temp)

    def calc_parent(self, beta):
        """
        calcParent at vertex beta (eq 15). Vectorized over all l entries.
        parent[:l] = circ_conv(left, right), parent[l:] = right
        """
        left = self.edge_data[2 * beta]
        right = self.edge_data[2 * beta + 1]
        l = left.shape[0]
        parent = np.empty((2 * l, 2, 2), dtype=np.float64)
        parent[:l] = _circ_conv_batch(left, right)
        parent[l:] = right
        self.edge_data[beta] = parent

    def step_to(self, target):
        """Navigate decHead to target vertex (Algorithm 4 step 3 + Algorithm 6)."""
        current = self.dec_head
        if current == target:
            return

        path = self._get_path(current, target)
        for beta in path:
            self._step_one(beta)

        self.dec_head = target

    def _step_one(self, beta):
        """
        Execute one navigation step toward vertex beta (Algorithm 6).

        Key: when going UP, calcParent is executed at the CURRENT vertex
        (decHead), not the target. This converts the current vertex's edge
        from getAsChild to getAsParent, preserving the root edge.
        """
        current = self.dec_head

        if current == beta >> 1:
            # Going DOWN: current is parent of beta
            if beta & 1 == 0:
                self.calc_left(current)
            else:
                self.calc_right(current)
            self.dec_head = beta
        elif beta == current >> 1:
            # Going UP: beta is parent of current
            self.calc_parent(current)
            self.dec_head = beta

    def _get_path(self, current, target):
        """
        Compute the traversal path from current to target (Algorithm 5).
        Returns list of vertices to step through.
        """
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
#  SC Decoder using computational graph (Algorithm 4)
# ─────────────────────────────────────────────────────────────────────────────

def decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True):
    """
    O(N log N) SC MAC decoder for one received vector z^N.

    Parameters
    ----------
    N         : int — block length (power of 2)
    z         : list, length N — channel output symbols
    b         : list[int], length 2N — path vector (0=U step, 1=V step)
    frozen_u  : dict {1-indexed position: value} — U frozen bits
    frozen_v  : dict {1-indexed position: value} — V frozen bits
    channel   : MACChannel
    log_domain: bool — ignored (always log-domain internally)

    Returns
    -------
    u_dec : list[int] of length N — decoded U bits (0-indexed)
    v_dec : list[int] of length N — decoded V bits (0-indexed)
    """
    from polar.efficient_decoder import build_log_W_leaf

    n = N.bit_length() - 1
    assert (1 << n) == N

    log_W = build_log_W_leaf(z, channel)
    graph = _CompGraph(n, log_W)

    u_hat = {}  # 1-indexed position → decided value
    v_hat = {}  # 1-indexed position → decided value
    i_u = 0     # number of U bits decided
    i_v = 0     # number of V bits decided

    for step in range(2 * N):
        gamma = b[step]  # 0=U, 1=V

        if gamma == 0:
            i_u += 1
            i_t = i_u
            frozen_dict = frozen_u
        else:
            i_v += 1
            i_t = i_v
            frozen_dict = frozen_v

        # Leaf edge index and its parent vertex
        leaf_edge = i_t + N - 1
        target_vertex = leaf_edge >> 1

        # Navigate decHead to target vertex
        graph.step_to(target_vertex)

        # Save leaf's getAsParent data (partial decisions)
        temp = graph.edge_data[leaf_edge][0].copy()

        # Compute top-down message at the leaf
        if leaf_edge & 1 == 0:
            graph.calc_left(target_vertex)
        else:
            graph.calc_right(target_vertex)

        top_down = graph.edge_data[leaf_edge][0]

        # Combine top-down with saved bottom-up
        combined = _norm_prod_single(top_down, temp)

        # Make hard decision
        if i_t in frozen_dict:
            bit = frozen_dict[i_t]
        else:
            if gamma == 0:  # U decision: marginalize over V
                p0 = np.logaddexp(combined[0, 0], combined[0, 1])
                p1 = np.logaddexp(combined[1, 0], combined[1, 1])
                bit = 1 if p1 > p0 else 0
            else:  # V decision: marginalize over U
                p0 = np.logaddexp(combined[0, 0], combined[1, 0])
                p1 = np.logaddexp(combined[0, 1], combined[1, 1])
                bit = 1 if p1 > p0 else 0

        # Record decision
        if gamma == 0:
            u_hat[i_t] = bit
        else:
            v_hat[i_t] = bit

        # Set leaf to partially deterministic tensor (eq 16)
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
#  Vectorized batch decode (all codewords in parallel via NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def _circ_conv_batched(A, B):
    """Circular convolution for (batch, L, 2, 2) arrays."""
    A00 = A[:, :, 0, 0]; A01 = A[:, :, 0, 1]
    A10 = A[:, :, 1, 0]; A11 = A[:, :, 1, 1]
    B00 = B[:, :, 0, 0]; B01 = B[:, :, 0, 1]
    B10 = B[:, :, 1, 0]; B11 = B[:, :, 1, 1]

    out = np.empty_like(A)
    out[:, :, 0, 0] = np.logaddexp(np.logaddexp(A00+B00, A01+B01),
                                    np.logaddexp(A10+B10, A11+B11))
    out[:, :, 0, 1] = np.logaddexp(np.logaddexp(A01+B00, A00+B01),
                                    np.logaddexp(A11+B10, A10+B11))
    out[:, :, 1, 0] = np.logaddexp(np.logaddexp(A10+B00, A11+B01),
                                    np.logaddexp(A00+B10, A01+B11))
    out[:, :, 1, 1] = np.logaddexp(np.logaddexp(A11+B00, A10+B01),
                                    np.logaddexp(A01+B10, A00+B11))
    return out


def _norm_prod_batched(A, B):
    """Normalized product for (batch, L, 2, 2) arrays."""
    raw = A + B
    total = np.logaddexp(
        np.logaddexp(raw[:, :, 0, 0], raw[:, :, 0, 1]),
        np.logaddexp(raw[:, :, 1, 0], raw[:, :, 1, 1])
    )
    finite = np.isfinite(total)
    result = raw.copy()
    result[finite] -= total[finite, None, None]
    return result


def _norm_prod_single_batched(A, B):
    """Normalized product for (batch, 2, 2) arrays."""
    raw = A + B
    total = np.logaddexp(
        np.logaddexp(raw[:, 0, 0], raw[:, 0, 1]),
        np.logaddexp(raw[:, 1, 0], raw[:, 1, 1])
    )
    finite = np.isfinite(total)
    result = raw.copy()
    result[finite] -= total[finite, None, None]
    return result


class _CompGraphBatched:
    """Computational graph with batch dimension for vectorized decoding."""

    def __init__(self, n, log_W_batch):
        """
        Parameters
        ----------
        n          : int — log2(N)
        log_W_batch: ndarray (batch, N, 2, 2) — channel log-posteriors
        """
        self.n = n
        self.N = 1 << n
        N = self.N
        self.batch = log_W_batch.shape[0]

        self.edge_data = [None] * (2 * N)

        from polar.encoder import bit_reversal_perm
        br = bit_reversal_perm(n)
        root = log_W_batch[:, br].copy()  # (batch, N, 2, 2)
        totals = np.logaddexp(
            np.logaddexp(root[:, :, 0, 0], root[:, :, 0, 1]),
            np.logaddexp(root[:, :, 1, 0], root[:, :, 1, 1])
        )
        finite = np.isfinite(totals)
        root[finite] -= totals[finite, None, None]
        self.edge_data[1] = root

        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            self.edge_data[beta] = np.full(
                (self.batch, size, 2, 2), _LOG_QUARTER, dtype=np.float64)

        self.dec_head = 1

    def calc_left(self, beta):
        parent = self.edge_data[beta]
        right = self.edge_data[2 * beta + 1]
        l = right.shape[1]
        temp = _norm_prod_batched(parent[:, l:], right)
        self.edge_data[2 * beta] = _circ_conv_batched(parent[:, :l], temp)

    def calc_right(self, beta):
        parent = self.edge_data[beta]
        left = self.edge_data[2 * beta]
        l = left.shape[1]
        temp = _circ_conv_batched(left, parent[:, :l])
        self.edge_data[2 * beta + 1] = _norm_prod_batched(parent[:, l:], temp)

    def calc_parent(self, beta):
        left = self.edge_data[2 * beta]
        right = self.edge_data[2 * beta + 1]
        parent_left = _circ_conv_batched(left, right)
        self.edge_data[beta] = np.concatenate(
            [parent_left, right], axis=1)

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


def decode_vectorized(N, Z_batch, b, frozen_u, frozen_v, channel,
                      log_domain=True):
    """
    Vectorized SC MAC decoder: all codewords decoded in parallel via NumPy.

    Parameters
    ----------
    N         : int — block length
    Z_batch   : ndarray (batch, N) — channel outputs
    b         : list[int], length 2N — path vector
    frozen_u  : dict {1-indexed: value}
    frozen_v  : dict {1-indexed: value}
    channel   : MACChannel
    log_domain: bool — ignored

    Returns
    -------
    u_dec : ndarray (batch, N) int
    v_dec : ndarray (batch, N) int
    """
    from polar.efficient_decoder import build_log_W_leaf

    n = N.bit_length() - 1
    assert (1 << n) == N
    Z_batch = np.asarray(Z_batch)
    batch = Z_batch.shape[0]

    # Build batched log_W: (batch, N, 2, 2)
    log_W_batch = np.stack(
        [build_log_W_leaf(Z_batch[i], channel) for i in range(batch)])

    graph = _CompGraphBatched(n, log_W_batch)

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

        temp = graph.edge_data[leaf_edge][:, 0].copy()  # (batch, 2, 2)

        if leaf_edge & 1 == 0:
            graph.calc_left(target_vertex)
        else:
            graph.calc_right(target_vertex)

        top_down = graph.edge_data[leaf_edge][:, 0]
        combined = _norm_prod_single_batched(top_down, temp)

        if i_t in frozen_dict:
            bit = np.full(batch, frozen_dict[i_t], dtype=np.int32)
        else:
            if gamma == 0:
                p0 = np.logaddexp(combined[:, 0, 0], combined[:, 0, 1])
                p1 = np.logaddexp(combined[:, 1, 0], combined[:, 1, 1])
            else:
                p0 = np.logaddexp(combined[:, 0, 0], combined[:, 1, 0])
                p1 = np.logaddexp(combined[:, 0, 1], combined[:, 1, 1])
            bit = (p1 > p0).astype(np.int32)

        if gamma == 0:
            u_hat[i_t] = bit
        else:
            v_hat[i_t] = bit

        # Update leaf (vectorized over batch)
        new_leaf = np.full((batch, 2, 2), _NEG_INF, dtype=np.float64)
        u_val = u_hat.get(i_t)
        v_val = v_hat.get(i_t)

        if u_val is not None and v_val is not None:
            new_leaf[np.arange(batch), u_val, v_val] = 0.0
        elif u_val is not None:
            new_leaf[np.arange(batch), u_val, 0] = _LOG_HALF
            new_leaf[np.arange(batch), u_val, 1] = _LOG_HALF
        elif v_val is not None:
            new_leaf[np.arange(batch), 0, v_val] = _LOG_HALF
            new_leaf[np.arange(batch), 1, v_val] = _LOG_HALF
        else:
            new_leaf[:, :, :] = _LOG_QUARTER

        graph.edge_data[leaf_edge] = new_leaf[:, None, :, :]

    u_dec = np.zeros((batch, N), dtype=np.int32)
    v_dec = np.zeros((batch, N), dtype=np.int32)
    for k in range(1, N + 1):
        if k in u_hat:
            u_dec[:, k - 1] = u_hat[k]
        if k in v_hat:
            v_dec[:, k - 1] = v_hat[k]

    return u_dec, v_dec


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy batch decode (sequential / multiprocessing)
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
