"""
decoder.py
==========
Unified SC decoder for two-user binary-input MAC polar codes.

Auto-dispatches based on path type:
  - path_i=0 or path_i=N  -> O(N log N) LLR-based SC decoder (faster, ~2x)
  - intermediate paths     -> O(N log N) tensor-based computational graph SC
                              (Ren et al. 2025)

Both single-codeword and batch-vectorised decoding are supported.
The batch decoder processes all codewords in parallel via NumPy,
achieving ~30-40x speedup over sequential decoding.

Public API:
    decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True)
    decode_batch (N, Z_list, b, frozen_u, frozen_v, channel, log_domain=True,
                  n_workers=1, vectorized=True)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Shared: vectorised leaf log-probability array
# ─────────────────────────────────────────────────────────────────────────────

def build_log_W_leaf(z, channel) -> np.ndarray:
    """
    Build the (N, 2, 2) array of leaf log-probabilities.

        log_W[t, x, y] = log W(z_t | x, y)

    Parameters
    ----------
    z       : list of N channel output symbols
                BE-MAC  -> integers in {0, 1, 2}
                ABN-MAC -> (zx, zy) tuples
    channel : MACChannel

    Returns
    -------
    log_W : np.ndarray shape (N, 2, 2), dtype float64
    """
    N = len(z)

    if channel.name == "be_mac":
        xy_sum = np.array([[0, 1], [1, 2]], dtype=np.int32)
        z_arr = np.asarray(z, dtype=np.int32)
        match = z_arr[:, None, None] == xy_sum[None, :, :]
        log_W = np.where(match, 0.0, -np.inf)

    elif channel.name == "abn_mac":
        zx = np.array([zi[0] for zi in z], dtype=np.int32)
        zy = np.array([zi[1] for zi in z], dtype=np.int32)
        ex = np.array([0, 1])[None, :] ^ zx[:, None]
        ey = np.array([0, 1])[None, :] ^ zy[:, None]
        log_p = np.log(channel.p_noise)
        log_W = log_p[ex[:, :, None], ey[:, None, :]]

    elif channel.name == "gaussian_mac":
        z_arr = np.asarray(z, dtype=np.float64)
        sigma2 = channel.sigma2
        log_norm = -0.5 * np.log(2.0 * np.pi * sigma2)
        # mu[x,y] = (1-2x) + (1-2y): [[2, 0], [0, -2]]
        mu = np.array([[2.0, 0.0], [0.0, -2.0]], dtype=np.float64)
        log_W = log_norm - (z_arr[:, None, None] - mu[None, :, :]) ** 2 / (2.0 * sigma2)

    else:
        log_W = np.empty((N, 2, 2), dtype=np.float64)
        for t in range(N):
            for x in range(2):
                for y in range(2):
                    p = channel.transition_prob(z[t], x, y)
                    log_W[t, x, y] = np.log(p) if p > 0.0 else -np.inf

    return log_W


# ─────────────────────────────────────────────────────────────────────────────
#  Path detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_path_i(N, b):
    """
    Detect path_i from path vector b of length 2N.

    Returns path_i in [0, N] if b = 0^{path_i} 1^N 0^{N-path_i},
    or -1 for unrecognised structure.
    """
    if len(b) != 2 * N:
        return -1
    pi = 0
    while pi < len(b) and b[pi] == 0:
        pi += 1
    if pi > N:
        return -1
    end_ones = pi + N
    if end_ones > 2 * N:
        return -1
    if (all(b[k] == 1 for k in range(pi, end_ones)) and
            all(b[k] == 0 for k in range(end_ones, 2 * N))):
        return pi
    return -1


# =============================================================================
#  PART A: O(N log N) LLR-based SC decoder (extreme paths only)
# =============================================================================

def _f_llr(La, Lb):
    """
    f-node (check node / box-plus): marginalises over one unknown bit.
    Vectorised: La, Lb may be scalars or arrays.
    """
    result = np.logaddexp(La + Lb, 0.0) - np.logaddexp(La, Lb)
    nan_mask = np.isnan(result)
    if np.ndim(nan_mask) == 0:
        if nan_mask:
            result = float(np.sign(La) * np.sign(Lb)
                           * np.minimum(np.abs(La), np.abs(Lb)))
    elif np.any(nan_mask):
        fallback = (np.sign(La) * np.sign(Lb)
                    * np.minimum(np.abs(La), np.abs(Lb)))
        result = np.where(nan_mask, fallback, result)
    return result


def _g_llr(La, Lb, u):
    """
    g-node (variable node): conditions on already-decided bit u.
    Vectorised: all arguments may be arrays of the same shape.
    """
    result = Lb + (1.0 - 2.0 * u) * La
    nan_mask = np.isnan(result)
    if np.ndim(nan_mask) == 0:
        if nan_mask:
            result = 0.0
    elif np.any(nan_mask):
        result = np.where(nan_mask, 0.0, result)
    return result


def _u_marginal_llr(log_W_leaf):
    """U-marginal leaf LLRs: marginalise over Y. Returns shape (N,)."""
    return (np.logaddexp(log_W_leaf[:, 0, 0], log_W_leaf[:, 0, 1]) -
            np.logaddexp(log_W_leaf[:, 1, 0], log_W_leaf[:, 1, 1]))


def _v_marginal_llr(log_W_leaf):
    """V-marginal leaf LLRs: marginalise over X. Returns shape (N,)."""
    return (np.logaddexp(log_W_leaf[:, 0, 0], log_W_leaf[:, 1, 0]) -
            np.logaddexp(log_W_leaf[:, 0, 1], log_W_leaf[:, 1, 1]))


def _u_conditional_llr(log_W_leaf, y_enc):
    """U-conditional leaf LLRs given fully known encoded V codeword y."""
    idx = np.arange(log_W_leaf.shape[0])
    y = np.asarray(y_enc, dtype=np.intp)
    return log_W_leaf[idx, 0, y] - log_W_leaf[idx, 1, y]


def _v_conditional_llr(log_W_leaf, x_enc):
    """V-conditional leaf LLRs given fully known encoded U codeword x."""
    idx = np.arange(log_W_leaf.shape[0])
    x = np.asarray(x_enc, dtype=np.intp)
    return log_W_leaf[idx, x, 0] - log_W_leaf[idx, x, 1]


class _SCNode:
    """
    Streaming SC sub-channel node for the Arikan polar code.
    O(N) total size, O(N log N) total f/g-node evaluations.
    """
    __slots__ = ('N', 'ch0', 'left', 'right', '_Ll', '_Lr', 'decisions')

    def __init__(self, channel_llr):
        self.N = len(channel_llr)
        if self.N == 1:
            self.ch0 = float(channel_llr[0])
        else:
            h = self.N >> 1
            self.left = _SCNode(channel_llr[:h])
            self.right = _SCNode(channel_llr[h:])
            self._Ll = 0.0
            self._Lr = 0.0
            self.decisions = np.zeros(self.N, dtype=np.int8)

    def get_llr(self, k):
        if self.N == 1:
            return self.ch0
        if k & 1 == 0:
            self._Ll = self.left.get_llr(k >> 1)
            self._Lr = self.right.get_llr(k >> 1)
            return _f_llr(self._Ll, self._Lr)
        else:
            return _g_llr(self._Ll, self._Lr, self.decisions[k - 1])

    def feed(self, k, bit):
        if self.N == 1:
            return
        self.decisions[k] = bit
        if k & 1 == 1:
            self.left.feed(k >> 1, self.decisions[k - 1] ^ bit)
            self.right.feed(k >> 1, bit)


def _sc_decode_from_llr(leaf_llr, frozen_1idx):
    """
    Arikan-order O(N log N) SC decode from arbitrary leaf LLRs.
    Returns decoded bits as np.ndarray shape (N,) int8.
    """
    N = len(leaf_llr)
    frozen_0 = {k - 1: v for k, v in frozen_1idx.items()}

    node = _SCNode(np.asarray(leaf_llr, dtype=np.float64))
    u = np.zeros(N, dtype=np.int8)

    for i in range(N):
        L = node.get_llr(i)
        if i in frozen_0:
            u[i] = frozen_0[i]
        else:
            u[i] = 0 if L >= 0 else 1
        node.feed(i, u[i])

    return u


def _decode_extreme_llr(N, log_W, path_i, frozen_u, frozen_v):
    """
    O(N log N) LLR-based SC for extreme paths (path_i=0 or path_i=N).
    Returns (u_dec, v_dec) as lists.
    """
    from .encoder import polar_encode

    if path_i == N:
        u_hat = _sc_decode_from_llr(_u_marginal_llr(log_W), frozen_u)
        x_hat = np.array(polar_encode(u_hat.tolist()), dtype=np.int8)
        v_hat = _sc_decode_from_llr(_v_conditional_llr(log_W, x_hat), frozen_v)
        return u_hat.tolist(), v_hat.tolist()
    else:  # path_i == 0
        v_hat = _sc_decode_from_llr(_v_marginal_llr(log_W), frozen_v)
        y_hat = np.array(polar_encode(v_hat.tolist()), dtype=np.int8)
        u_hat = _sc_decode_from_llr(_u_conditional_llr(log_W, y_hat), frozen_u)
        return u_hat.tolist(), v_hat.tolist()


# Expose sub-decoders for use by decoder_scl.py
sc_decode_u_marginal = lambda log_W_leaf, frozen_u_1idx: \
    _sc_decode_from_llr(_u_marginal_llr(log_W_leaf), frozen_u_1idx)
sc_decode_v_marginal = lambda log_W_leaf, frozen_v_1idx: \
    _sc_decode_from_llr(_v_marginal_llr(log_W_leaf), frozen_v_1idx)
sc_decode_u_conditional = lambda log_W_leaf, y_enc, frozen_u_1idx: \
    _sc_decode_from_llr(_u_conditional_llr(log_W_leaf, y_enc), frozen_u_1idx)
sc_decode_v_conditional = lambda log_W_leaf, x_enc, frozen_v_1idx: \
    _sc_decode_from_llr(_v_conditional_llr(log_W_leaf, x_enc), frozen_v_1idx)

# =============================================================================
#  PART B: O(N log N) tensor-based SC decoder (all paths)
#  Computational graph approach from Ren et al. (2025)
# =============================================================================

_NEG_INF = -np.inf
_LOG_HALF = np.log(0.5)
_LOG_QUARTER = np.log(0.25)


def _circ_conv_batch(A, B):
    """
    Vectorized circular convolution of (L, 2, 2) log-prob tensor arrays.
    out[a,b] = logaddexp over (a',b') of A[a^a', b^b'] + B[a', b']
    """
    A00 = A[:, 0, 0];
    A01 = A[:, 0, 1]
    A10 = A[:, 1, 0];
    A11 = A[:, 1, 1]
    B00 = B[:, 0, 0];
    B01 = B[:, 0, 1]
    B10 = B[:, 1, 0];
    B11 = B[:, 1, 1]

    out = np.empty_like(A)
    out[:, 0, 0] = np.logaddexp(
        np.logaddexp(A00 + B00, A01 + B01),
        np.logaddexp(A10 + B10, A11 + B11))
    out[:, 0, 1] = np.logaddexp(
        np.logaddexp(A01 + B00, A00 + B01),
        np.logaddexp(A11 + B10, A10 + B11))
    out[:, 1, 0] = np.logaddexp(
        np.logaddexp(A10 + B00, A11 + B01),
        np.logaddexp(A00 + B10, A01 + B11))
    out[:, 1, 1] = np.logaddexp(
        np.logaddexp(A11 + B00, A10 + B01),
        np.logaddexp(A01 + B10, A00 + B11))
    return out


def _norm_prod_batch(A, B):
    """Vectorized normalized elementwise product of (L, 2, 2) arrays."""
    raw = A + B
    total = np.logaddexp(
        np.logaddexp(raw[:, 0, 0], raw[:, 0, 1]),
        np.logaddexp(raw[:, 1, 0], raw[:, 1, 1])
    )
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


class _CompGraph:
    """
    Computational graph for SC decoding of monotone chain polar codes.

    Indexing:
      - Edges: 1..2N-1. Edge 1 = root. Edges N..2N-1 = leaves.
      - Vertices: 1..N-1. Vertex beta has edges beta, 2*beta, 2*beta+1.
    """

    def __init__(self, n, log_W_leaf):
        self.n = n
        self.N = 1 << n
        N = self.N

        self.edge_data = [None] * (2 * N)

        from .encoder import bit_reversal_perm
        br = bit_reversal_perm(n)
        root = log_W_leaf[br].copy()
        totals = np.logaddexp(
            np.logaddexp(root[:, 0, 0], root[:, 0, 1]),
            np.logaddexp(root[:, 1, 0], root[:, 1, 1])
        )
        finite = np.isfinite(totals)
        root[finite] -= totals[finite, None, None]
        self.edge_data[1] = root

        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            self.edge_data[beta] = np.full((size, 2, 2), _LOG_QUARTER,
                                           dtype=np.float64)

        self.dec_head = 1

    def calc_left(self, beta):
        parent = self.edge_data[beta]
        right = self.edge_data[2 * beta + 1]
        l = right.shape[0]
        temp = _norm_prod_batch(parent[l:], right)
        self.edge_data[2 * beta] = _circ_conv_batch(parent[:l], temp)

    def calc_right(self, beta):
        parent = self.edge_data[beta]
        left = self.edge_data[2 * beta]
        l = left.shape[0]
        temp = _circ_conv_batch(left, parent[:l])
        self.edge_data[2 * beta + 1] = _norm_prod_batch(parent[l:], temp)

    def calc_parent(self, beta):
        left = self.edge_data[2 * beta]
        right = self.edge_data[2 * beta + 1]
        l = left.shape[0]
        parent = np.empty((2 * l, 2, 2), dtype=np.float64)
        parent[:l] = _circ_conv_batch(left, right)
        parent[l:] = right
        self.edge_data[beta] = parent

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


def _decode_general_tensor(N, log_W, b, frozen_u, frozen_v):
    """
    O(N log N) tensor-based SC decoder for arbitrary monotone chain paths.
    Returns (u_dec, v_dec) as lists.
    """
    n = N.bit_length() - 1

    graph = _CompGraph(n, log_W)

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


# =============================================================================
#  PART C: Batch-vectorised SC decoders (all codewords in parallel)
# =============================================================================

def build_log_W_leaf_batch(Z_batch, channel) -> np.ndarray:
    """
    Build (batch, N, 2, 2) leaf log-probability array for a batch of
    received vectors.  Currently supports BE-MAC only.

    Parameters
    ----------
    Z_batch : array-like, shape (batch, N)
    channel : MACChannel

    Returns
    -------
    log_W : ndarray shape (batch, N, 2, 2), float64
    """
    Z = np.asarray(Z_batch, dtype=np.int32)
    if channel.name == "be_mac":
        xy_sum = np.array([[0, 1], [1, 2]], dtype=np.int32)
        match = Z[:, :, None, None] == xy_sum[None, None, :, :]
        return np.where(match, 0.0, -np.inf)
    else:
        return np.stack([build_log_W_leaf(Z[i], channel)
                         for i in range(Z.shape[0])])


# -- batched tensor operations (batch, L, 2, 2) -----------------------------

def _circ_conv_batched(A, B):
    """Circular convolution on (batch, L, 2, 2) arrays."""
    s = A.shape
    return _circ_conv_batch(
        A.reshape(-1, 2, 2), B.reshape(-1, 2, 2)).reshape(s)


def _norm_prod_batched(A, B):
    """Normalized product on (batch, L, 2, 2) arrays."""
    s = A.shape
    return _norm_prod_batch(
        A.reshape(-1, 2, 2), B.reshape(-1, 2, 2)).reshape(s)


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


# -- batched computational graph --------------------------------------------

class _CompGraphBatched:
    """Computational graph with batch dimension for vectorised SC decoding."""

    def __init__(self, n, log_W_batch):
        """
        Parameters
        ----------
        n            : int -- log2(N)
        log_W_batch  : ndarray (batch, N, 2, 2)
        """
        self.n = n
        self.N = 1 << n
        N = self.N
        self.batch = log_W_batch.shape[0]

        self.edge_data = [None] * (2 * N)

        from .encoder import bit_reversal_perm
        br = bit_reversal_perm(n)
        root = log_W_batch[:, br].copy()
        totals = np.logaddexp(
            np.logaddexp(root[:, :, 0, 0], root[:, :, 0, 1]),
            np.logaddexp(root[:, :, 1, 0], root[:, :, 1, 1]))
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
        path = _CompGraph._get_path(None, current, target)
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


def _decode_general_tensor_batch(N, log_W_batch, b, frozen_u, frozen_v):
    """
    Vectorised SC decoder for general monotone-chain paths.
    Processes all codewords in parallel via NumPy.

    Parameters
    ----------
    N            : int
    log_W_batch  : ndarray (batch, N, 2, 2)
    b            : list[int], length 2N
    frozen_u, frozen_v : dict {1-indexed: value}

    Returns
    -------
    u_dec, v_dec : ndarray (batch, N), int32
    """
    n = N.bit_length() - 1
    batch = log_W_batch.shape[0]

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

        # Update leaf to partially deterministic (eq 16)
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


# -- batched LLR-based SC (extreme paths) -----------------------------------

class _SCNodeBatched:
    """Batched SC node -- all LLRs are (batch,) arrays."""
    __slots__ = ('N', 'ch0', 'left', 'right', '_Ll', '_Lr', 'decisions')

    def __init__(self, channel_llr_batch):
        """channel_llr_batch : ndarray (batch, N)"""
        self.N = channel_llr_batch.shape[1]
        if self.N == 1:
            self.ch0 = channel_llr_batch[:, 0].copy()
        else:
            h = self.N >> 1
            self.left = _SCNodeBatched(channel_llr_batch[:, :h])
            self.right = _SCNodeBatched(channel_llr_batch[:, h:])
            batch = channel_llr_batch.shape[0]
            self._Ll = np.zeros(batch, dtype=np.float64)
            self._Lr = np.zeros(batch, dtype=np.float64)
            self.decisions = np.zeros((batch, self.N), dtype=np.int8)

    def get_llr(self, k):
        if self.N == 1:
            return self.ch0
        if k & 1 == 0:
            self._Ll = self.left.get_llr(k >> 1)
            self._Lr = self.right.get_llr(k >> 1)
            return _f_llr(self._Ll, self._Lr)
        else:
            return _g_llr(self._Ll, self._Lr, self.decisions[:, k - 1])

    def feed(self, k, bit):
        if self.N == 1:
            return
        self.decisions[:, k] = bit
        if k & 1 == 1:
            self.left.feed(k >> 1, self.decisions[:, k - 1] ^ bit)
            self.right.feed(k >> 1, bit)


def _sc_decode_from_llr_batch(leaf_llr_batch, frozen_1idx):
    """
    Batch SC decode from (batch, N) leaf LLRs.
    Returns (batch, N) int8 decoded bits.
    """
    batch, N = leaf_llr_batch.shape
    frozen_0 = {k - 1: v for k, v in frozen_1idx.items()}

    node = _SCNodeBatched(leaf_llr_batch.astype(np.float64))
    u = np.zeros((batch, N), dtype=np.int8)

    for i in range(N):
        L = node.get_llr(i)
        if i in frozen_0:
            u[:, i] = frozen_0[i]
        else:
            u[:, i] = np.where(L >= 0, 0, 1)
        node.feed(i, u[:, i])

    return u


def _decode_extreme_llr_batch(N, log_W_batch, path_i, frozen_u, frozen_v):
    """
    Batch LLR-based SC for extreme paths.
    log_W_batch : (batch, N, 2, 2)
    Returns u_hat, v_hat as (batch, N) int arrays.
    """
    from .encoder import polar_encode_batch

    if path_i == N:
        # U-marginal LLRs: (batch, N)
        u_llr = (np.logaddexp(log_W_batch[:, :, 0, 0], log_W_batch[:, :, 0, 1]) -
                 np.logaddexp(log_W_batch[:, :, 1, 0], log_W_batch[:, :, 1, 1]))
        u_hat = _sc_decode_from_llr_batch(u_llr, frozen_u)
        x_hat = polar_encode_batch(u_hat.astype(np.int32))
        # V-conditional LLRs
        bidx = np.arange(log_W_batch.shape[0])[:, None]
        tidx = np.arange(N)[None, :]
        v_llr = (log_W_batch[bidx, tidx, x_hat, 0] -
                 log_W_batch[bidx, tidx, x_hat, 1])
        v_hat = _sc_decode_from_llr_batch(v_llr, frozen_v)
        return u_hat.astype(np.int32), v_hat.astype(np.int32)

    else:  # path_i == 0
        v_llr = (np.logaddexp(log_W_batch[:, :, 0, 0], log_W_batch[:, :, 1, 0]) -
                 np.logaddexp(log_W_batch[:, :, 0, 1], log_W_batch[:, :, 1, 1]))
        v_hat = _sc_decode_from_llr_batch(v_llr, frozen_v)
        y_hat = polar_encode_batch(v_hat.astype(np.int32))
        bidx = np.arange(log_W_batch.shape[0])[:, None]
        tidx = np.arange(N)[None, :]
        u_llr = (log_W_batch[bidx, tidx, 0, y_hat] -
                 log_W_batch[bidx, tidx, 1, y_hat])
        u_hat = _sc_decode_from_llr_batch(u_llr, frozen_u)
        return u_hat.astype(np.int32), v_hat.astype(np.int32)


def _decode_batch_vectorized(N, Z_batch, b, frozen_u, frozen_v, channel):
    """
    Vectorised batch SC decoder.  Dispatches to extreme or general
    path decoder based on path structure.

    For Class B (path_i = N/2), uses the Numba-JIT parallel decoder
    for ~5-7x speedup when available.

    Parameters
    ----------
    Z_batch : array-like (batch, N)

    Returns
    -------
    u_dec, v_dec : ndarray (batch, N), int32
    """
    log_W = build_log_W_leaf_batch(Z_batch, channel)
    path_i = _detect_path_i(N, b)

    if path_i == 0 or path_i == N:
        return _decode_extreme_llr_batch(N, log_W, path_i, frozen_u, frozen_v)

    # Try Numba-JIT parallel decoder for Class B (path_i = N/2)
    if path_i == N // 2:
        try:
            from .decoder_parallel import decode_parallel_batch
            return decode_parallel_batch(N, log_W, b, frozen_u, frozen_v)
        except ImportError:
            pass  # Fall through to general tensor decoder

    return _decode_general_tensor_batch(N, log_W, b, frozen_u, frozen_v)


# =============================================================================
#  Public API
# =============================================================================

def decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True):
    """
    SC MAC decoder for one received vector z^N.

    Auto-dispatches:
      - path_i=0 or path_i=N  -> O(N log N) LLR-based (faster)
      - intermediate paths     -> O(N log N) tensor-based

    Parameters
    ----------
    N         : int -- block length (power of 2)
    z         : list, length N -- channel output symbols
    b         : list[int], length 2N -- path vector (0=U step, 1=V step)
    frozen_u  : dict {1-indexed position: value} -- U frozen bits
    frozen_v  : dict {1-indexed position: value} -- V frozen bits
    channel   : MACChannel
    log_domain: bool -- ignored (always log-domain internally)

    Returns
    -------
    u_dec : list[int] of length N
    v_dec : list[int] of length N
    """
    path_i = _detect_path_i(N, b)
    log_W = build_log_W_leaf(z, channel)

    if path_i == 0 or path_i == N:
        return _decode_extreme_llr(N, log_W, path_i, frozen_u, frozen_v)

    # Try Numba-JIT parallel decoder for Class B (path_i = N/2)
    if path_i == N // 2:
        try:
            from .decoder_parallel import decode_parallel_single
            return decode_parallel_single(N, log_W, b, frozen_u, frozen_v)
        except ImportError:
            pass

    return _decode_general_tensor(N, log_W, b, frozen_u, frozen_v)


def _decode_worker(args):
    N, z, b, frozen_u, frozen_v, channel, log_domain = args
    return decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)


def decode_batch(N, Z_list, b, frozen_u, frozen_v, channel,
                 log_domain=True, n_workers=1, vectorized=True):
    """
    Decode a list (or array) of received vectors.

    Parameters
    ----------
    N, b, frozen_u, frozen_v, channel, log_domain : same as decode_single
    Z_list     : list or (batch, N) array of received vectors
    n_workers  : int -- parallel worker processes (only used when vectorized=False)
    vectorized : bool -- if True (default), decode all codewords in parallel
                 via NumPy vectorisation (~30-40x faster than sequential)

    Returns
    -------
    results : list of (u_dec, v_dec) tuples
    """
    if vectorized and len(Z_list) >= 1:
        Z_arr = np.asarray(Z_list)
        u_dec, v_dec = _decode_batch_vectorized(
            N, Z_arr, b, frozen_u, frozen_v, channel)
        return [(u_dec[i].tolist(), v_dec[i].tolist())
                for i in range(u_dec.shape[0])]

    if n_workers <= 1 or len(Z_list) <= 1:
        return [decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)
                for z in Z_list]

    from concurrent.futures import ProcessPoolExecutor
    args = [(N, z, b, frozen_u, frozen_v, channel, log_domain) for z in Z_list]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_decode_worker, args,
                                    chunksize=max(1, len(Z_list) // n_workers)))
    return results
