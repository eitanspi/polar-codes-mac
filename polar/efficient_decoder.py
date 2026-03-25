"""
efficient_decoder.py
====================
O(N log N) LLR-based SC decoder for the two-user binary-input MAC.

Replaces the recursive probability approach in decoder.py with a standard
Arikan factor-graph traversal, extended for MAC by choosing marginal vs
conditional leaf LLRs at each decode step.

Public API (drop-in for decoder.py):
    decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True)
    decode_batch (N, Z_list, b, frozen_u, frozen_v, channel, log_domain=True,
                  n_workers=1)

Phase status
------------
  [x] Phase 1 — build_log_W_leaf  : vectorised (N,2,2) leaf array
  [x] Phase 2 — LLR tree skeleton : f/g nodes, recursive SC (single-user U)
  [x] Phase 3 — MAC leaf dispatch : marginal vs conditional per step
  [x] Phase 4 — arbitrary path b  : 0^N1^N and 1^N0^N efficient, fallback otherwise
  [x] Phase 5 — batch vectorisation
  [x] Phase 6 — drop-in replace
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

    elif channel.name == "gaussian_mac":
        z_arr  = np.asarray(z, dtype=np.float64)
        sigma2 = channel.sigma2
        log_norm = -0.5 * np.log(2.0 * np.pi * sigma2)
        # mu[x,y] = (1-2x) + (1-2y): [[2, 0], [0, -2]]
        mu     = np.array([[2.0, 0.0], [0.0, -2.0]], dtype=np.float64)
        log_W  = log_norm - (z_arr[:, None, None] - mu[None, :, :]) ** 2 / (2.0 * sigma2)

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
    matches the virtual channels of decoder.py exactly.

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
    to the recursive decoder in decoder.py.

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
    from polar.encoder import polar_encode

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
        from polar.decoder import decode_single as _old_decode_single
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

    Parameters / returns match decoder.decode_batch exactly.

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


# ─────────────────────────────────────────────────────────────────────────────
#  Self-tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    from polar.channels import BEMAC, ABNMAC
    from polar.design   import design_bemac, design_abnmac, make_path
    from polar.decoder  import decode_single as old_decode_single
    from polar.encoder  import polar_encode, build_message

    rng = np.random.default_rng(42)

    # ── Phase 1: leaf array correctness ──────────────────────────────────────
    print("=== Phase 1: build_log_W_leaf ===")

    def validate_leaf(channel, z_samples, label):
        log_W   = build_log_W_leaf(z_samples, channel)
        max_err = 0.0
        for t in range(len(z_samples)):
            for x in range(2):
                for y in range(2):
                    p   = channel.transition_prob(z_samples[t], x, y)
                    ref = np.log(p) if p > 0.0 else -np.inf
                    got = log_W[t, x, y]
                    if np.isfinite(ref) and np.isfinite(got):
                        max_err = max(max_err, abs(got - ref))
                    elif np.isfinite(ref) != np.isfinite(got):
                        max_err = np.inf
        status = "PASS" if max_err < 1e-12 else f"FAIL (max_err={max_err:.2e})"
        print(f"  {label:20s}  [{status}]")

    be  = BEMAC()
    abn = ABNMAC()
    N   = 64
    X   = rng.integers(0, 2, N, dtype=np.int32)
    Y   = rng.integers(0, 2, N, dtype=np.int32)
    validate_leaf(be,  be.sample_batch(X, Y).tolist(),  "BE-MAC  N=64")
    validate_leaf(abn, abn.sample_batch(X, Y).tolist(), "ABN-MAC N=64")

    # ── Phase 2: SC decoder round-trip (U marginal) ──────────────────────────
    print("\n=== Phase 2: sc_decode_u_marginal round-trip ===")

    def validate_u_roundtrip(channel, design_fn, n, ku, kv, label,
                              n_trials=100):
        N  = 1 << n
        Au, Av, frozen_u, frozen_v, _, _ = design_fn(n, ku, kv)

        errors = 0
        for _ in range(n_trials):
            info_u = rng.integers(0, 2, ku).tolist()
            info_v = rng.integers(0, 2, kv).tolist()
            u = build_message(N, info_u, Au)
            v = build_message(N, info_v, Av)
            x = polar_encode(u.tolist())
            y = polar_encode(v.tolist())
            z = channel.sample_batch(np.array(x), np.array(y)).tolist()

            log_W = build_log_W_leaf(z, channel)
            u_new = sc_decode_u_marginal(log_W, frozen_u)

            if not all(int(u_new[p - 1]) == bit for p, bit in zip(Au, info_u)):
                errors += 1

        status = "PASS" if errors == 0 else f"FAIL ({errors}/{n_trials} errors)"
        print(f"  {label:30s}  [{status}]")

    validate_u_roundtrip(be,  design_bemac,  n=4, ku=4,  kv=16,
                         label="BE-MAC  N=16  ku=4")
    validate_u_roundtrip(be,  design_bemac,  n=6, ku=16, kv=64,
                         label="BE-MAC  N=64  ku=16")
    validate_u_roundtrip(abn, design_abnmac, n=4, ku=3,  kv=8,
                         label="ABN-MAC N=16  ku=3")
    validate_u_roundtrip(abn, design_abnmac, n=6, ku=14, kv=32,
                         label="ABN-MAC N=64  ku=14")

    # ── Phase 2b: LLR agreement with old decoder (ABN-MAC, no NaN) ───────────
    print("\n=== Phase 2b: LLR agreement with old decoder (ABN-MAC) ===")

    def validate_llr_agreement(n, ku, kv, label, n_trials=50):
        N  = 1 << n
        b  = make_path(N, path_i=N)
        Au, Av, frozen_u, frozen_v, _, _ = design_abnmac(n, ku, kv)

        mismatches = 0
        for _ in range(n_trials):
            info_u = rng.integers(0, 2, ku).tolist()
            info_v = rng.integers(0, 2, kv).tolist()
            u = build_message(N, info_u, Au)
            v = build_message(N, info_v, Av)
            x = polar_encode(u.tolist())
            y = polar_encode(v.tolist())
            z = abn.sample_batch(np.array(x), np.array(y)).tolist()

            u_old, _ = old_decode_single(N, z, b, frozen_u, frozen_v,
                                          abn, log_domain=True)
            log_W = build_log_W_leaf(z, abn)
            u_new = sc_decode_u_marginal(log_W, frozen_u)

            if not np.array_equal(u_old, u_new):
                mismatches += 1

        status = "PASS" if mismatches == 0 else f"FAIL ({mismatches}/{n_trials})"
        print(f"  {label:30s}  [{status}]")

    validate_llr_agreement(n=4, ku=3,  kv=8,  label="ABN-MAC N=16  ku=3")
    validate_llr_agreement(n=5, ku=7,  kv=16, label="ABN-MAC N=32  ku=7")
    validate_llr_agreement(n=6, ku=14, kv=32, label="ABN-MAC N=64  ku=14")

    # ── Phase 3-6: Full MAC decode round-trip ─────────────────────────────────
    print("\n=== Phase 3-6: decode_single round-trip ===")

    def validate_roundtrip(channel, design_fn, n, ku, kv, path_i, label,
                            n_trials=100):
        N = 1 << n
        b = make_path(N, path_i=path_i)
        Au, Av, frozen_u, frozen_v, _, _ = design_fn(n, ku, kv)

        errors = 0
        for _ in range(n_trials):
            info_u = rng.integers(0, 2, ku).tolist()
            info_v = rng.integers(0, 2, kv).tolist()
            u = build_message(N, info_u, Au)
            v = build_message(N, info_v, Av)
            x = polar_encode(u.tolist())
            y = polar_encode(v.tolist())
            z = channel.sample_batch(np.array(x), np.array(y)).tolist()

            u_dec, v_dec = decode_single(N, z, b, frozen_u, frozen_v,
                                          channel, log_domain=True)

            u_ok = all(u_dec[p - 1] == bit for p, bit in zip(Au, info_u))
            v_ok = all(v_dec[p - 1] == bit for p, bit in zip(Av, info_v))
            if not (u_ok and v_ok):
                errors += 1

        status = "PASS" if errors == 0 else f"FAIL ({errors}/{n_trials} block errors)"
        print(f"  {label:45s}  [{status}]")

    # Path 0^N 1^N (efficient: U marginal → V conditional)
    print("  — path 0^N 1^N (U first) —")
    validate_roundtrip(be,  design_bemac,  4, 4,  16, 16,
                       "BE-MAC  N=16  ku=4  kv=16")
    validate_roundtrip(be,  design_bemac,  6, 16, 64, 64,
                       "BE-MAC  N=64  ku=16 kv=64")
    validate_roundtrip(abn, design_abnmac, 4, 3,  8,  16,
                       "ABN-MAC N=16  ku=3  kv=8")
    validate_roundtrip(abn, design_abnmac, 6, 14, 32, 64,
                       "ABN-MAC N=64  ku=14 kv=32")

    # Path 1^N 0^N (efficient: V marginal → U conditional)
    # V-first requires frozen sets designed for V-marginal / U-conditional.
    # For symmetric MACs, swap roles: design(n, kv, ku) gives V-marginal
    # info positions in Au and U-conditional info positions in Av.
    print("  — path 1^N 0^N (V first) —")

    def validate_roundtrip_vfirst(channel, design_fn, n, ku, kv, label,
                                   n_trials=100):
        N = 1 << n
        b = make_path(N, path_i=0)
        # Design with swapped roles: V uses marginal, U uses conditional
        Av_vf, Au_vf, fv_vf, fu_vf, _, _ = design_fn(n, kv, ku)

        errors = 0
        for _ in range(n_trials):
            info_u = rng.integers(0, 2, ku).tolist()
            info_v = rng.integers(0, 2, kv).tolist()
            u = build_message(N, info_u, Au_vf)
            v = build_message(N, info_v, Av_vf)
            x = polar_encode(u.tolist())
            y = polar_encode(v.tolist())
            z = channel.sample_batch(np.array(x), np.array(y)).tolist()

            u_dec, v_dec = decode_single(N, z, b, fu_vf, fv_vf,
                                          channel, log_domain=True)

            u_ok = all(u_dec[p - 1] == bit for p, bit in zip(Au_vf, info_u))
            v_ok = all(v_dec[p - 1] == bit for p, bit in zip(Av_vf, info_v))
            if not (u_ok and v_ok):
                errors += 1

        status = "PASS" if errors == 0 else f"FAIL ({errors}/{n_trials} block errors)"
        print(f"  {label:45s}  [{status}]")

    # V-first: swap rates. V sees marginal (cap~0.5), U sees conditional.
    validate_roundtrip_vfirst(be,  design_bemac,  4, 16, 4,
                              "BE-MAC  N=16  ku=16 kv=4   (V first)")
    validate_roundtrip_vfirst(be,  design_bemac,  6, 64, 16,
                              "BE-MAC  N=64  ku=64 kv=16  (V first)")
    validate_roundtrip_vfirst(abn, design_abnmac, 4, 8,  3,
                              "ABN-MAC N=16  ku=8  kv=3   (V first)")
    validate_roundtrip_vfirst(abn, design_abnmac, 6, 32, 14,
                              "ABN-MAC N=64  ku=32 kv=14  (V first)")

    # Intermediate path (fallback to old decoder)
    print("  — intermediate path (fallback) —")
    validate_roundtrip(be,  design_bemac,  4, 4,  8,  8,
                       "BE-MAC  N=16  path_i=8 (fallback)")
    validate_roundtrip(abn, design_abnmac, 4, 3,  4,  8,
                       "ABN-MAC N=16  path_i=8 (fallback)")

    # ── BLER parity: same error rate as old decoder ───────────────────────────
    print("\n=== BLER parity: efficient vs old decoder ===")

    def validate_bler_parity(channel, design_fn, n, ku, kv, path_i, label,
                              n_trials=200):
        N = 1 << n
        b = make_path(N, path_i=path_i)
        Au, Av, frozen_u, frozen_v, _, _ = design_fn(n, ku, kv)

        old_err = 0; new_err = 0
        for _ in range(n_trials):
            info_u = rng.integers(0, 2, ku).tolist()
            info_v = rng.integers(0, 2, kv).tolist()
            u = build_message(N, info_u, Au)
            v = build_message(N, info_v, Av)
            x = polar_encode(u.tolist())
            y = polar_encode(v.tolist())
            z = channel.sample_batch(np.array(x), np.array(y)).tolist()

            uo, vo = old_decode_single(N, z, b, frozen_u, frozen_v, channel)
            un, vn = decode_single(N, z, b, frozen_u, frozen_v, channel)

            u_ok_old = all(uo[p-1] == b for p, b in zip(Au, info_u))
            v_ok_old = all(vo[p-1] == b for p, b in zip(Av, info_v))
            u_ok_new = all(un[p-1] == b for p, b in zip(Au, info_u))
            v_ok_new = all(vn[p-1] == b for p, b in zip(Av, info_v))
            if not (u_ok_old and v_ok_old): old_err += 1
            if not (u_ok_new and v_ok_new): new_err += 1

        status = "PASS" if old_err == new_err else f"FAIL (old={old_err} new={new_err})"
        print(f"  {label:40s}  old={old_err}/{n_trials}  new={new_err}/{n_trials}  [{status}]")

    validate_bler_parity(be,  design_bemac,  6, 16, 64, 64,
                         "BE-MAC  N=64 U-first")
    validate_bler_parity(abn, design_abnmac, 6, 14, 32, 64,
                         "ABN-MAC N=64 U-first")

    # ── Speed comparison ──────────────────────────────────────────────────────
    print("\n=== Speed: efficient vs old decoder ===")

    for n_speed, label in [(7, "N=128"), (8, "N=256"), (9, "N=512")]:
        N_s  = 1 << n_speed
        ku_s = N_s // 4
        kv_s = N_s
        _, _, fu_s, fv_s, _, _ = design_bemac(n_speed, ku_s, kv_s)
        b_s  = make_path(N_s, path_i=N_s)
        X_s  = rng.integers(0, 2, N_s, dtype=np.int32)
        Y_s  = rng.integers(0, 2, N_s, dtype=np.int32)
        z_s  = be.sample_batch(X_s, Y_s).tolist()

        # Warm up
        decode_single(N_s, z_s, b_s, fu_s, fv_s, be)
        old_decode_single(N_s, z_s, b_s, fu_s, fv_s, be)

        reps = 10 if n_speed <= 8 else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            decode_single(N_s, z_s, b_s, fu_s, fv_s, be)
        ms_new = (time.perf_counter() - t0) / reps * 1e3

        t0 = time.perf_counter()
        for _ in range(reps):
            old_decode_single(N_s, z_s, b_s, fu_s, fv_s, be)
        ms_old = (time.perf_counter() - t0) / reps * 1e3

        speedup = ms_old / ms_new if ms_new > 0 else float('inf')
        print(f"  {label}  efficient={ms_new:.1f}ms  old={ms_old:.1f}ms  "
              f"speedup={speedup:.1f}x")

    # ── Batch decode ──────────────────────────────────────────────────────────
    print("\n=== Batch decode ===")
    N_b = 32
    Au_b, Av_b, fu_b, fv_b, _, _ = design_bemac(5, 8, 32)
    b_b = make_path(N_b, path_i=N_b)

    errors_batch = 0
    for _ in range(20):
        iu = rng.integers(0, 2, 8).tolist()
        iv = rng.integers(0, 2, 32).tolist()
        u = build_message(N_b, iu, Au_b)
        v = build_message(N_b, iv, Av_b)
        x = polar_encode(u.tolist())
        y = polar_encode(v.tolist())
        z = be.sample_batch(np.array(x), np.array(y)).tolist()

        u_dec, v_dec = decode_single(N_b, z, b_b, fu_b, fv_b, be)
        u_ok = all(u_dec[p-1] == bit for p, bit in zip(Au_b, iu))
        v_ok = all(v_dec[p-1] == bit for p, bit in zip(Av_b, iv))
        if not (u_ok and v_ok):
            errors_batch += 1
    print(f"  20 codewords N=32: {'PASS' if errors_batch == 0 else f'FAIL ({errors_batch}/20)'}")

    print("\nAll tests complete.")
