"""
decoder.py
==========
O(N log N) LLR-based SC decoder for the two-user binary-input MAC.

Replaces the recursive probability approach in _decoder_base.py with a standard
Arikan factor-graph traversal, extended for MAC by choosing marginal vs
conditional leaf LLRs at each decode step.

Public API:
    decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True)
    decode_batch (N, Z_list, b, frozen_u, frozen_v, channel, log_domain=True,
                  n_workers=1)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1 — vectorised leaf log-probability array
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
        z_arr  = np.asarray(z, dtype=np.int32)
        match  = z_arr[:, None, None] == xy_sum[None, :, :]
        log_W  = np.where(match, 0.0, -np.inf)

    elif channel.name == "abn_mac":
        zx    = np.array([zi[0] for zi in z], dtype=np.int32)
        zy    = np.array([zi[1] for zi in z], dtype=np.int32)
        ex    = np.array([0, 1])[None, :] ^ zx[:, None]
        ey    = np.array([0, 1])[None, :] ^ zy[:, None]
        log_p = np.log(channel.p_noise)
        log_W = log_p[ex[:, :, None], ey[:, None, :]]

    else:
        log_W = np.empty((N, 2, 2), dtype=np.float64)
        for t in range(N):
            for x in range(2):
                for y in range(2):
                    p = channel.transition_prob(z[t], x, y)
                    log_W[t, x, y] = np.log(p) if p > 0.0 else -np.inf

    return log_W


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2 — f/g nodes and recursive LLR tree (single-user U skeleton)
# ─────────────────────────────────────────────────────────────────────────────

def _f(La, Lb):
    """
    f-node (check node / box-plus): marginalises over one unknown bit.

        f(La, Lb) = logaddexp(La+Lb, 0) - logaddexp(La, Lb)

    NaN arises when La+Lb = ±inf ∓ inf (e.g. BE-MAC leaf LLRs hit {±inf, 0}).
    Fallback: sign(La)*sign(Lb)*min(|La|,|Lb|) — exact at ±inf, small error
    elsewhere (never triggered for finite ABN-MAC LLRs).

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


def _g(La, Lb, u):
    """
    g-node (variable node): conditions on already-decided bit u.

        g(La, Lb, u) = Lb + (1 - 2*u) * La

    NaN arises when Lb and (1-2u)*La are both ±inf with opposite signs,
    indicating contradictory observations (e.g. a frozen bit forced to a
    value that contradicts the channel).  Fallback: 0.0 (maximum
    uncertainty), matching the probability-domain result that both
    conditional probabilities are zero.

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
    """
    Compute U-marginal leaf LLRs from (N,2,2) log-prob array.

        L[t] = log sum_y W(z_t|0,y) - log sum_y W(z_t|1,y)

    Returns shape (N,).
    """
    return (np.logaddexp(log_W_leaf[:, 0, 0], log_W_leaf[:, 0, 1]) -
            np.logaddexp(log_W_leaf[:, 1, 0], log_W_leaf[:, 1, 1]))


class _SCNode:
    """
    Streaming SC sub-channel node for the Arikan polar code.

    Provides LLRs one at a time (get_llr) and accepts decisions (feed).
    At each level, even-indexed positions use f-nodes, odd-indexed use g-nodes,
    matching the Arikan recursion with even/odd interleaving (butterfly_ops).

    The tree of _SCNode objects is O(N) total size and processes N bits
    with O(N log N) total f/g-node evaluations.
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
        """Return the LLR for natural-order position *k*."""
        if self.N == 1:
            return self.ch0
        if k & 1 == 0:                         # even → f-node
            self._Ll = self.left.get_llr(k >> 1)
            self._Lr = self.right.get_llr(k >> 1)
            return _f(self._Ll, self._Lr)
        else:                                   # odd  → g-node
            return _g(self._Ll, self._Lr, self.decisions[k - 1])

    def feed(self, k, bit):
        """Feed back the decision for position *k*."""
        if self.N == 1:
            return
        self.decisions[k] = bit
        if k & 1 == 1:                         # pair complete → propagate
            self.left.feed(k >> 1, self.decisions[k - 1] ^ bit)
            self.right.feed(k >> 1, bit)


def _sc_decode_from_llr(leaf_llr, frozen_1idx):
    """
    Arikan-order O(N log N) SC decode from arbitrary leaf LLRs.

    Processes bits in natural order using the Arikan recursive
    decomposition (even/odd interleaving via butterfly_ops).  This
    matches the virtual channels of _decoder_base.py exactly.

    Parameters
    ----------
    leaf_llr    : np.ndarray shape (N,) — per-position channel LLRs
    frozen_1idx : dict {1-indexed position: frozen value}

    Returns
    -------
    decoded : np.ndarray shape (N,) int8 — decoded bits in natural order
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


def sc_decode_u_marginal(log_W_leaf, frozen_u_1idx):
    """
    Decode U using only the marginal channel (V marginalised out).

    This is the single-user SC decoder on W_1(z|x) = sum_y W(z|x,y)/2.
    Equivalent to the old decoder with path 0^N 1^N during the U phase
    (all V bits undecided, j=0 throughout).

    Parameters
    ----------
    log_W_leaf   : np.ndarray shape (N,2,2) from build_log_W_leaf
    frozen_u_1idx: dict {1-indexed position: frozen value}

    Returns
    -------
    u_hat : np.ndarray shape (N,) int8  — in natural (1-indexed) bit order
    """
    return _sc_decode_from_llr(_u_marginal_llr(log_W_leaf), frozen_u_1idx)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 3 — MAC leaf dispatch: marginal vs conditional LLRs
# ─────────────────────────────────────────────────────────────────────────────

def _v_marginal_llr(log_W_leaf):
    """
    V-marginal leaf LLRs: marginalise over X.

        L[t] = log sum_x W(z_t|x,0) - log sum_x W(z_t|x,1)

    Returns shape (N,).
    """
    return (np.logaddexp(log_W_leaf[:, 0, 0], log_W_leaf[:, 1, 0]) -
            np.logaddexp(log_W_leaf[:, 0, 1], log_W_leaf[:, 1, 1]))


def _u_conditional_llr(log_W_leaf, y_enc):
    """
    U-conditional leaf LLRs given fully known encoded V codeword y.

        L[t] = log W(z_t|0,y[t]) - log W(z_t|1,y[t])

    Parameters
    ----------
    log_W_leaf : np.ndarray shape (N,2,2)
    y_enc      : np.ndarray shape (N,) int — encoded V codeword bits

    Returns shape (N,).
    """
    idx = np.arange(log_W_leaf.shape[0])
    y   = np.asarray(y_enc, dtype=np.intp)
    return log_W_leaf[idx, 0, y] - log_W_leaf[idx, 1, y]


def _v_conditional_llr(log_W_leaf, x_enc):
    """
    V-conditional leaf LLRs given fully known encoded U codeword x.

        L[t] = log W(z_t|x[t],0) - log W(z_t|x[t],1)

    Parameters
    ----------
    log_W_leaf : np.ndarray shape (N,2,2)
    x_enc      : np.ndarray shape (N,) int — encoded U codeword bits

    Returns shape (N,).
    """
    idx = np.arange(log_W_leaf.shape[0])
    x   = np.asarray(x_enc, dtype=np.intp)
    return log_W_leaf[idx, x, 0] - log_W_leaf[idx, x, 1]


def sc_decode_v_conditional(log_W_leaf, x_enc, frozen_v_1idx):
    """
    Decode V conditioned on fully known U (encoded codeword x_enc).

    Parameters
    ----------
    log_W_leaf    : np.ndarray shape (N,2,2)
    x_enc         : np.ndarray shape (N,) int — encoded U codeword
    frozen_v_1idx : dict {1-indexed position: frozen value}

    Returns
    -------
    v_hat : np.ndarray shape (N,) int8
    """
    return _sc_decode_from_llr(_v_conditional_llr(log_W_leaf, x_enc),
                               frozen_v_1idx)


def sc_decode_u_conditional(log_W_leaf, y_enc, frozen_u_1idx):
    """
    Decode U conditioned on fully known V (encoded codeword y_enc).

    Parameters
    ----------
    log_W_leaf    : np.ndarray shape (N,2,2)
    y_enc         : np.ndarray shape (N,) int — encoded V codeword
    frozen_u_1idx : dict {1-indexed position: frozen value}

    Returns
    -------
    u_hat : np.ndarray shape (N,) int8
    """
    return _sc_decode_from_llr(_u_conditional_llr(log_W_leaf, y_enc),
                               frozen_u_1idx)


def sc_decode_v_marginal(log_W_leaf, frozen_v_1idx):
    """
    Decode V using only the marginal channel (U marginalised out).

    Parameters
    ----------
    log_W_leaf    : np.ndarray shape (N,2,2)
    frozen_v_1idx : dict {1-indexed position: frozen value}

    Returns
    -------
    v_hat : np.ndarray shape (N,) int8
    """
    return _sc_decode_from_llr(_v_marginal_llr(log_W_leaf), frozen_v_1idx)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 4 — arbitrary path b
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


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 5/6 — drop-in API
# ─────────────────────────────────────────────────────────────────────────────

def decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True):
    """
    O(N log N) SC MAC decoder for one received vector z^N.

    For path 0^N 1^N (all U first) or 1^N 0^N (all V first), uses the
    efficient LLR-based SC decoder.  For other path structures, falls back
    to the recursive decoder in _decoder_base.py.

    Parameters
    ----------
    N         : int — block length (power of 2)
    z         : list, length N — channel output symbols
    b         : list[int], length 2N — path vector (0=U step, 1=V step)
    frozen_u  : dict {1-indexed position: value} — U frozen bits
    frozen_v  : dict {1-indexed position: value} — V frozen bits
    channel   : MACChannel
    log_domain: bool — ignored (always log-domain internally); kept for compat

    Returns
    -------
    u_dec : list[int] of length N — decoded U bits (0-indexed)
    v_dec : list[int] of length N — decoded V bits (0-indexed)
    """
    from .encoder import polar_encode

    path_i = _detect_path_i(N, b)
    log_W  = build_log_W_leaf(z, channel)

    if path_i == N:
        # 0^N 1^N — decode all U (marginal), then all V (conditional on U)
        u_hat = _sc_decode_from_llr(_u_marginal_llr(log_W), frozen_u)
        x_hat = np.array(polar_encode(u_hat.tolist()), dtype=np.int8)
        v_hat = _sc_decode_from_llr(_v_conditional_llr(log_W, x_hat), frozen_v)
        return u_hat.tolist(), v_hat.tolist()

    elif path_i == 0:
        # 1^N 0^N — decode all V (marginal), then all U (conditional on V)
        v_hat = _sc_decode_from_llr(_v_marginal_llr(log_W), frozen_v)
        y_hat = np.array(polar_encode(v_hat.tolist()), dtype=np.int8)
        u_hat = _sc_decode_from_llr(_u_conditional_llr(log_W, y_hat), frozen_u)
        return u_hat.tolist(), v_hat.tolist()

    else:
        # General path — fall back to recursive decoder
        from ._decoder_base import decode_single as _old_decode_single
        return _old_decode_single(N, z, b, frozen_u, frozen_v,
                                  channel, log_domain)


def _decode_worker(args):
    """Module-level worker for multiprocessing (must be picklable)."""
    N, z, b, frozen_u, frozen_v, channel, log_domain = args
    return decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)


def decode_batch(N, Z_list, b, frozen_u, frozen_v, channel,
                 log_domain=True, n_workers=1):
    """
    Decode a list of received vectors.

    Parameters / returns match _decoder_base.decode_batch exactly.

    Parameters
    ----------
    N, b, frozen_u, frozen_v, channel, log_domain : same as decode_single
    Z_list    : list of received vectors, each of length N
    n_workers : int — parallel worker processes (1 = sequential)

    Returns
    -------
    results : list of (u_dec, v_dec) tuples
    """
    if n_workers <= 1 or len(Z_list) <= 1:
        return [decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain)
                for z in Z_list]

    from concurrent.futures import ProcessPoolExecutor
    args = [(N, z, b, frozen_u, frozen_v, channel, log_domain) for z in Z_list]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_decode_worker, args,
                                    chunksize=max(1, len(Z_list) // n_workers)))
    return results
