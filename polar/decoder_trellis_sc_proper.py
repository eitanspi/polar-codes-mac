"""
decoder_trellis_sc_proper.py
============================
Proper trellis SC decoder for ISI-MAC that carries state through the tree.

Unlike decoder_trellis_mac_chained.py (which collapses FB → scalar LLR → SC),
this decoder propagates per-position tensors of shape (2, |S|, |S|) through
the polar tree using trellis-aware CheckNode and BitNode operations.

For ISI-MAC Stage 1 (decode U, marginalize V):
  State: S_t = X_{t-1} ∈ {0,1}, so |S| = 2
  Per-position tensor: log_W[x, s_in, s_out] where s_out = x (deterministic)
  → effectively (2, 2) per position since s_out is determined by x

We represent each "embedding" as a tensor of shape (2, S, S):
  e[u, s_prev, s_next] = log P(z_block | u, s_prev→s_next)

CheckNode (f-operation): combines two positions, marginalizes over intermediate state
BitNode (g-operation): conditions on decided bit, updates state

At each leaf: marginalize over states to get LLR, make decision.
"""
import numpy as np
from polar.encoder import polar_encode, bit_reversal_perm
from polar.channels_memory import ISIMAC


def build_leaf_tensors_stage1(z, channel):
    """
    Build per-position log-probability tensors for Stage 1 (U, V marginalized).

    Returns: leaf_log[t, x, s_prev, s_next] shape (N, 2, 2, 2)
    where s_next = x (deterministic transition).

    Since V and V_prev are marginalized as uniform:
    log_W[t, x, s_prev] = log (1/4) sum_{y,b} N(z_t; mu(x,y,s_prev,b), sigma2)
    and s_next = x.
    """
    sigma2 = channel.sigma2
    h = channel.h
    z = np.asarray(z, dtype=np.float64)
    N = z.shape[0]
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma2)

    # leaf[t, x, s_prev, s_next]: s_next must equal x, otherwise -inf
    leaf = np.full((N, 2, 2, 2), -np.inf, dtype=np.float64)

    for x in range(2):
        bx = 1.0 - 2.0 * x
        s_next = x  # deterministic transition
        for s_prev in range(2):
            bxp = 1.0 - 2.0 * s_prev
            # Marginalize over y, y_prev (4 terms)
            terms = []
            for y in range(2):
                by = 1.0 - 2.0 * y
                for b in range(2):  # b = y_prev
                    byp = 1.0 - 2.0 * b
                    mu = bx + by + h * (bxp + byp)
                    log_p = log_norm - (z - mu) ** 2 / (2.0 * sigma2)
                    terms.append(log_p)
            # logsumexp of 4 terms + log(1/4)
            stacked = np.stack(terms, axis=-1)  # (N, 4)
            max_val = stacked.max(axis=-1)
            log_sum = max_val + np.log(np.exp(stacked - max_val[..., None]).sum(axis=-1))
            leaf[:, x, s_prev, s_next] = log_sum + np.log(0.25)

    return leaf


def build_leaf_tensors_stage2(z, x_known, channel):
    """
    Build per-position tensors for Stage 2 (V, given known X).

    State: s_prev = Y_{t-1}, s_next = y (deterministic).
    X_t and X_{t-1} are known.
    """
    sigma2 = channel.sigma2
    h = channel.h
    z = np.asarray(z, dtype=np.float64)
    x_known = np.asarray(x_known, dtype=np.int64)
    N = z.shape[0]
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma2)

    bx = 1.0 - 2.0 * x_known.astype(np.float64)
    bxp = np.concatenate([np.ones(1), bx[:-1]])  # X_{t-1}, padded with x_{-1}=0 → bpsk=+1

    leaf = np.full((N, 2, 2, 2), -np.inf, dtype=np.float64)

    for y in range(2):
        by = 1.0 - 2.0 * y
        s_next = y  # deterministic
        for s_prev in range(2):  # s_prev = Y_{t-1}
            byp = 1.0 - 2.0 * s_prev
            mu = bx + by + h * (bxp + byp)
            log_p = log_norm - (z - mu) ** 2 / (2.0 * sigma2)
            leaf[:, y, s_prev, s_next] = log_p

    return leaf


def _safe_logsumexp(stacked):
    """logsumexp along last axis, handling -inf properly."""
    max_val = stacked.max(axis=-1)
    # Where max is -inf, result is -inf
    mask = np.isfinite(max_val)
    result = np.full_like(max_val, -np.inf)
    if mask.any():
        diff = stacked[mask] - max_val[mask][..., None]
        result[mask] = max_val[mask] + np.log(np.exp(diff).sum(axis=-1))
    return result


def checknode_trellis(e1, e2, n_states=2):
    """
    Trellis-aware check node (f-operation).

    e1[pos, u, s_in, s_out]: left block log-prob tensor, shape (N/2, 2, S, S)
    e2[pos, u, s_in, s_out]: right block tensor, shape (N/2, 2, S, S)

    Output: combined tensor for the "top" (XOR) sub-problem, shape (N/2, 2, S, S)

    For the check node, u_top = u1 XOR u2. We marginalize over u2 and
    the intermediate state s_mid:

    out[u_top, s_in, s_out] = logsumexp over u2, s_mid of:
        e1[u1=u_top^u2, s_in, s_mid] + e2[u2, s_mid, s_out]
    """
    half_N = e1.shape[0]
    S = n_states
    out = np.full((half_N, 2, S, S), -np.inf, dtype=np.float64)

    for u_top in range(2):
        for s_in in range(S):
            for s_out in range(S):
                terms = []
                for u2 in range(2):
                    u1 = u_top ^ u2
                    for s_mid in range(S):
                        val = e1[:, u1, s_in, s_mid] + e2[:, u2, s_mid, s_out]
                        terms.append(val)
                # logsumexp over all terms
                stacked = np.stack(terms, axis=-1)  # (half_N, n_terms)
                out[:, u_top, s_in, s_out] = _safe_logsumexp(stacked)

    return out


def bitnode_trellis(e1, e2, u_hat, n_states=2):
    """
    Trellis-aware bit node (g-operation).

    Conditions on decided bits u_hat (the "top" codeword bits).
    e1: left block, shape (N/2, 2, S, S)
    e2: right block, shape (N/2, 2, S, S)
    u_hat: decided top bits, shape (N/2,) integers

    Output: tensor for the "bottom" sub-problem, shape (N/2, 2, S, S)

    out[u2, s_in, s_out] = logsumexp over s_mid of:
        e1[u1=u_hat^u2, s_in, s_mid] + e2[u2, s_mid, s_out]

    (No marginalization over u2 since u1 is now known = u_hat XOR u2)
    Wait, u1 = u_top = u_hat. So u_hat XOR u2 gives us which u1 to use.
    Actually: u_top = u1 XOR u2, so u1 = u_hat XOR u2 (u_hat is the decided top bit).
    """
    half_N = e1.shape[0]
    S = n_states
    out = np.full((half_N, 2, S, S), -np.inf, dtype=np.float64)

    for u2 in range(2):
        for s_in in range(S):
            for s_out in range(S):
                terms = []
                for s_mid in range(S):
                    u1 = u_hat ^ u2  # element-wise XOR, shape (N/2,)
                    # Need to index e1 per position with different u1 values
                    val = np.where(u1 == 0,
                                   e1[:, 0, s_in, s_mid],
                                   e1[:, 1, s_in, s_mid]) + e2[:, u2, s_mid, s_out]
                    terms.append(val)
                stacked = np.stack(terms, axis=-1)
                out[:, u2, s_in, s_out] = _safe_logsumexp(stacked)

    return out


def tensor_to_llr(e):
    """
    Convert a single-position tensor (2, S, S) to scalar LLR.
    Marginalize over all states: LLR = log P(u=0) - log P(u=1)
    where P(u) = sum_{s_in, s_out} e[u, s_in, s_out]
    """
    # e shape: (2, S, S)
    log_p0 = _logsumexp_all(e[0])  # marginalize over states for u=0
    log_p1 = _logsumexp_all(e[1])  # marginalize over states for u=1
    return log_p0 - log_p1


def _logsumexp_all(arr):
    """logsumexp over all elements of arr."""
    flat = arr.flatten()
    m = flat.max()
    return m + np.log(np.exp(flat - m).sum())


def decode_trellis_sc(leaf_tensors, frozen_dict, n_states=2):
    """
    Full trellis SC decoder. Carries state tensors through the polar tree.

    leaf_tensors: (N, 2, S, S) — per-position log-prob tensors
    frozen_dict: {0-indexed position: value} for frozen bits

    Returns: u_hat (N,) decoded bits
    """
    N = leaf_tensors.shape[0]
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=np.int8)
    S = n_states

    def _decode(block):
        """
        block: (block_size, 2, S, S) tensor
        Returns: codeword bits (block_size,)
        """
        block_size = block.shape[0]

        if block_size == 1:
            # Leaf: compute LLR from tensor, decide
            pos = _decode.leaf_idx
            _decode.leaf_idx += 1
            llr = tensor_to_llr(block[0])
            if pos in frozen_dict:
                bit = frozen_dict[pos]
            else:
                bit = 0 if llr >= 0 else 1
            u_hat[pos] = bit
            return np.array([bit], dtype=np.int8)

        half = block_size // 2
        e_odd = block[0::2]   # (half, 2, S, S)
        e_even = block[1::2]  # (half, 2, S, S)

        # CheckNode: combine for top sub-problem
        e_top = checknode_trellis(e_odd, e_even, S)
        cw_top = _decode(e_top)

        # BitNode: condition on top bits for bottom sub-problem
        e_bot = bitnode_trellis(e_odd, e_even, cw_top, S)
        cw_bot = _decode(e_bot)

        # Combine codeword
        cw = np.zeros(block_size, dtype=np.int8)
        cw[0::2] = cw_top ^ cw_bot
        cw[1::2] = cw_bot
        return cw

    _decode.leaf_idx = 0
    _decode(leaf_tensors)
    return u_hat


def genie_decode_trellis_sc(leaf_tensors, true_bits, n_states=2):
    """
    Genie-aided trellis SC: check decisions, feed true bits.
    Returns per-position errors (0 or 1).
    """
    N = leaf_tensors.shape[0]
    S = n_states
    errors = np.zeros(N, dtype=np.int32)
    true_bits = np.asarray(true_bits, dtype=np.int8)

    def _decode(block):
        block_size = block.shape[0]
        if block_size == 1:
            pos = _decode.leaf_idx
            _decode.leaf_idx += 1
            llr = tensor_to_llr(block[0])
            decision = 0 if llr >= 0 else 1
            if decision != true_bits[pos]:
                errors[pos] = 1
            return np.array([true_bits[pos]], dtype=np.int8)  # genie

        half = block_size // 2
        e_odd = block[0::2]
        e_even = block[1::2]

        e_top = checknode_trellis(e_odd, e_even, S)
        cw_top = _decode(e_top)

        e_bot = bitnode_trellis(e_odd, e_even, cw_top, S)
        cw_bot = _decode(e_bot)

        cw = np.zeros(block_size, dtype=np.int8)
        cw[0::2] = cw_top ^ cw_bot
        cw[1::2] = cw_bot
        return cw

    _decode.leaf_idx = 0
    _decode(leaf_tensors)
    return errors


# ── Public API ──

def decode_stage1(z, N, fu, channel):
    """Decode U with proper trellis SC (state carried through tree)."""
    leaf = build_leaf_tensors_stage1(z, channel)
    frozen_0idx = {k - 1: v for k, v in fu.items()}
    u_hat = decode_trellis_sc(leaf, frozen_0idx)
    return u_hat


def decode_stage2(z, u_hat, N, fv, channel):
    """Decode V given known X with proper trellis SC."""
    x_hat = np.array(polar_encode(list(u_hat)), dtype=np.int64)
    leaf = build_leaf_tensors_stage2(z, x_hat, channel)
    frozen_0idx = {k - 1: v for k, v in fv.items()}
    v_hat = decode_trellis_sc(leaf, frozen_0idx)
    return v_hat


def decode_chained(z, N, fu, fv, channel):
    """Full chained proper trellis SC."""
    u_hat = decode_stage1(z, N, fu, channel)
    v_hat = decode_stage2(z, u_hat, N, fv, channel)
    return u_hat, v_hat


if __name__ == "__main__":
    # Quick test
    import time
    ch = ISIMAC(sigma2=10**(-6/10), h=0.3)
    N = 16; n = 4
    from polar.design_mc import design_from_file
    Au, Av, fu, fv, _, _, _ = design_from_file('designs/gmac_C_n4_snr6dB.npz', n, 4, 7)

    rng = np.random.default_rng(0)
    errs = 0; n_cw = 500
    t0 = time.time()
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
        for p in Au: u[p-1] = rng.integers(0, 2)
        for p in Av: v[p-1] = rng.integers(0, 2)
        from polar.encoder import polar_encode_batch
        x = polar_encode_batch(u.reshape(1,-1))[0]
        y = polar_encode_batch(v.reshape(1,-1))[0]
        z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
        uh, vh = decode_chained(z, N, fu, fv, ch)
        if any(uh[p-1] != u[p-1] for p in Au) or any(vh[p-1] != v[p-1] for p in Av):
            errs += 1
    elapsed = time.time() - t0
    print(f"N={N}: BLER={errs/n_cw:.4f} ({elapsed:.1f}s, {elapsed/n_cw*1000:.1f}ms/CW)")
