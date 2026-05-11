"""
decoder_trellis_mac_chained.py
==============================
Chained (two-stage) SC decoder for two-user MAC polar codes on channels
with memory (ISI-MAC).

This decoder implements the "corner rate" decomposition from Onay (2013):
Stage 1 decodes U while treating V as uniform i.i.d. random noise, and
Stage 2 decodes V conditioned on the Stage-1 decoded U codeword. In
contrast to the joint trellis SC decoder in ``decoder_trellis.py`` (which
performs forward-backward on the full |S|=4 MAC trellis), the chained
decoder only ever runs single-user forward-backward on a reduced
|S|=2 trellis at each stage.

For ISI-MAC with tap h:
    Z[i] = (1-2X[i]) + (1-2Y[i]) + h*((1-2X[i-1]) + (1-2Y[i-1])) + W[i]
                                                    W ~ N(0, sigma2)

Stage 1 (decode U, marginalise Y as uniform i.i.d.):
  State at position i:  S_i = X_{i-1}  ∈ {0, 1}
  Y_i and Y_{i-1} are marginalised as independent Bernoulli(1/2), so
      P(Z_i | X_i=x, S=a) = (1/4) sum_{y,b} N(Z; mu(x,y,a,b), sigma2)
  Transition:  s' = x (deterministic).

Stage 2 (decode V given X-hat):
  State at position i:  S_i = Y_{i-1}  ∈ {0, 1}
  X_i and X_{i-1} are known (from Stage-1 codeword, zero-padded for i=0):
      P(Z_i | Y_i=y, S=b, X_i=x_i, X_{i-1}=a_i)
            = N(Z_i; mu(x_i, y, a_i, b), sigma2)
  Transition:  s' = y (deterministic).

In each stage we run forward-backward on the 2-state trellis to produce
per-position marginal LLRs, then feed those LLRs into the standard
single-user Arikan-order SC decoder from ``polar.decoder``.

Public API:
    decode_stage1_u(z, N, Au, Av, fu, fv, channel)
    decode_stage2_v(z, u_hat, N, Au, Av, fu, fv, channel)
    decode_chained(z, N, b, fu, fv, channel)
"""

import numpy as np

from polar.encoder import polar_encode
from polar.decoder import _sc_decode_from_llr

_NEG_INF = -np.inf
_LOG_HALF = np.log(0.5)
_LOG_QUARTER = np.log(0.25)


# ---------------------------------------------------------------------------
#  Stage 1: per-position transition P(Z | X=x, S=a) with Y, Y_prev uniform
# ---------------------------------------------------------------------------

def _log_W_stage1(z, channel):
    """
    Return (N, 2, 2) array:  log_W1[t, x, a] = log P(Z_t | X_t=x, S_t=a)
    where S_t = X_{t-1} ∈ {0,1}, and Y_t, Y_{t-1} are independently uniform
    Bernoulli(1/2). The 1/4 from marginalisation is included.
    """
    sigma2 = channel.sigma2
    h = channel.h
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma2)
    z = np.asarray(z, dtype=np.float64)
    N = z.shape[0]

    log_W = np.full((N, 2, 2), _NEG_INF, dtype=np.float64)

    for x in range(2):
        bx = 1.0 - 2.0 * x
        for a in range(2):
            bxp = 1.0 - 2.0 * a
            # Marginalise Y_t and Y_{t-1}
            terms = []
            for y in range(2):
                by = 1.0 - 2.0 * y
                for b in range(2):
                    byp = 1.0 - 2.0 * b
                    mu = bx + by + h * (bxp + byp)
                    log_p = log_norm - (z - mu) ** 2 / (2.0 * sigma2)
                    terms.append(log_p)
            # logsumexp of four terms + log(1/4)
            m = np.maximum(np.maximum(terms[0], terms[1]),
                           np.maximum(terms[2], terms[3]))
            log_sum = m + np.log(
                np.exp(terms[0] - m) + np.exp(terms[1] - m)
                + np.exp(terms[2] - m) + np.exp(terms[3] - m))
            log_W[:, x, a] = log_sum + np.log(0.25)

    return log_W


def _log_W_stage2(z, x_hat, channel):
    """
    Return (N, 2, 2) array:  log_W2[t, y, b] = log P(Z_t | Y_t=y, S_t=b)
    where X_t and X_{t-1} are KNOWN (from x_hat; zero-padded for t=0),
    and S_t = Y_{t-1} ∈ {0,1}.
    """
    sigma2 = channel.sigma2
    h = channel.h
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma2)
    z = np.asarray(z, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.int64)
    N = z.shape[0]

    # Encoded BPSK for known X
    bx = 1.0 - 2.0 * x_hat.astype(np.float64)           # (N,)
    # X_{t-1} padded with zeros → BPSK amplitude +1 for t=0 (consistent with
    # ISIMAC.sample_batch, which pads with zeros before BPSK: the BPSK of 0 is
    # +1, so bxp for t=0 is +1).
    # ISIMAC.sample_batch does: sx_prev = concat(ones, sx[:-1]) — the previous
    # BPSK symbol for t=0 is +1 (equivalent to x_{-1}=0).
    bxp = np.concatenate([np.ones(1), bx[:-1]])         # (N,)

    log_W = np.full((N, 2, 2), _NEG_INF, dtype=np.float64)
    for y in range(2):
        by = 1.0 - 2.0 * y
        for b in range(2):
            byp = 1.0 - 2.0 * b
            mu = bx + by + h * (bxp + byp)              # (N,)
            log_p = log_norm - (z - mu) ** 2 / (2.0 * sigma2)
            log_W[:, y, b] = log_p

    return log_W


# ---------------------------------------------------------------------------
#  2-state forward-backward to get per-position bit marginals
# ---------------------------------------------------------------------------

def _logsumexp2(a, b):
    return np.logaddexp(a, b)


def _forward_backward_2state(log_W, deterministic_next_state=True):
    """
    Forward-backward on a 2-state trellis with deterministic transitions
    s_{t+1} = bit_t.

    log_W : (N, 2, 2) — log_W[t, x, s] = log P(Z_t | bit=x, state=s)

    Returns
    -------
    log_marg : (N, 2) — log-marginal posterior of bit_t (unnormalised)
    """
    N = log_W.shape[0]
    # S=2.  Initial state at t=0 is known = 0  (x_{-1}=0 / y_{-1}=0).
    log_alpha = np.full((N + 1, 2), _NEG_INF, dtype=np.float64)
    log_alpha[0, 0] = 0.0

    log_beta = np.full((N + 1, 2), _NEG_INF, dtype=np.float64)
    log_beta[N, :] = 0.0  # terminal: any state is acceptable

    # Forward: alpha[t+1, s'] = logsumexp_{x, s : next(x,s)=s'} (
    #     alpha[t, s] + log_W[t, x, s] + log P(x)   ← uniform prior 1/2
    # )
    log_half = _LOG_HALF
    for t in range(N):
        a0 = log_alpha[t, 0]
        a1 = log_alpha[t, 1]
        # For each input bit x, next state is x (deterministic), from either
        # previous state s=0 or s=1.
        for x in range(2):
            # contribution from s=0 and s=1
            c0 = a0 + log_W[t, x, 0] + log_half
            c1 = a1 + log_W[t, x, 1] + log_half
            log_alpha[t + 1, x] = np.logaddexp(
                log_alpha[t + 1, x], _logsumexp2(c0, c1))

    # Backward: beta[t, s] = logsumexp_{x} (
    #     log_W[t, x, s] + log P(x) + beta[t+1, next=x]
    # )
    for t in range(N - 1, -1, -1):
        for s in range(2):
            c0 = log_W[t, 0, s] + log_half + log_beta[t + 1, 0]
            c1 = log_W[t, 1, s] + log_half + log_beta[t + 1, 1]
            log_beta[t, s] = _logsumexp2(c0, c1)

    # Marginal: log p(Z, bit_t = x) = logsumexp_s alpha[t, s] + log_W[t, x, s]
    #                                 + log P(x) + beta[t+1, next=x]
    log_marg = np.full((N, 2), _NEG_INF, dtype=np.float64)
    for t in range(N):
        for x in range(2):
            c0 = log_alpha[t, 0] + log_W[t, x, 0] + log_half + log_beta[t + 1, x]
            c1 = log_alpha[t, 1] + log_W[t, x, 1] + log_half + log_beta[t + 1, x]
            log_marg[t, x] = _logsumexp2(c0, c1)

    return log_marg


def _marg_to_llr(log_marg):
    """Convert (N, 2) log-marginal to a length-N LLR vector  log P(0|Z)/P(1|Z)."""
    return log_marg[:, 0] - log_marg[:, 1]


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def decode_stage1_u(z, N, fu, channel):
    """
    Stage 1: decode U from the ISI-MAC observation z, treating V as uniform
    i.i.d. Bernoulli(1/2) noise.

    Parameters
    ----------
    z : array-like, length N
    N : int
    fu: dict {1-indexed position: value} — U frozen bits
    channel : ISIMAC instance

    Returns
    -------
    u_hat : np.ndarray (N,) int8 — decoded U bits (natural order, 0-indexed)
    """
    log_W1 = _log_W_stage1(z, channel)              # (N, 2, 2)
    log_marg = _forward_backward_2state(log_W1)     # (N, 2)
    llr = _marg_to_llr(log_marg)                    # (N,)
    u_hat = _sc_decode_from_llr(llr, fu)
    return u_hat


def decode_stage2_v(z, u_hat, N, fv, channel):
    """
    Stage 2: decode V given the Stage-1 decoded U, by polar-encoding U to get
    X-hat and running FB on the 2-state Y-trellis.

    Parameters
    ----------
    z : array-like, length N
    u_hat : np.ndarray (N,) — Stage-1 decoded U bits (natural order, 0-indexed)
    N : int
    fv : dict {1-indexed position: value} — V frozen bits
    channel : ISIMAC instance

    Returns
    -------
    v_hat : np.ndarray (N,) int8 — decoded V bits
    """
    x_hat = np.array(polar_encode(list(map(int, u_hat))), dtype=np.int64)
    log_W2 = _log_W_stage2(z, x_hat, channel)       # (N, 2, 2)
    log_marg = _forward_backward_2state(log_W2)     # (N, 2)
    llr = _marg_to_llr(log_marg)                    # (N,)
    v_hat = _sc_decode_from_llr(llr, fv)
    return v_hat


def decode_chained(z, N, fu, fv, channel):
    """
    Two-stage chained trellis SC decoder for ISI-MAC.

    Returns
    -------
    u_hat, v_hat : np.ndarray (N,) int8
    """
    u_hat = decode_stage1_u(z, N, fu, channel)
    v_hat = decode_stage2_v(z, u_hat, N, fv, channel)
    return u_hat, v_hat


# ---------------------------------------------------------------------------
#  Utility: evaluate BLER
# ---------------------------------------------------------------------------

def bler_stage1_only(channel, N, fu, fv, Au, Av, n_cw, seed=999):
    """Measure Stage-1 U-only BLER (assume V side is genie-aided correct)."""
    from polar.encoder import polar_encode_batch
    rng = np.random.default_rng(seed)
    errs = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au:
            u[p - 1] = rng.integers(0, 2)
        for p in Av:
            v[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        u_hat = decode_stage1_u(z, N, fu, channel)
        if any(int(u_hat[p - 1]) != int(u[p - 1]) for p in Au):
            errs += 1
    return errs / n_cw


def bler_stage2_only(channel, N, fu, fv, Au, Av, n_cw, seed=1000):
    """Measure Stage-2 V-only BLER assuming the TRUE U codeword is known."""
    from polar.encoder import polar_encode_batch
    rng = np.random.default_rng(seed)
    errs = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au:
            u[p - 1] = rng.integers(0, 2)
        for p in Av:
            v[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        v_hat = decode_stage2_v(z, u.astype(np.int8), N, fv, channel)
        if any(int(v_hat[p - 1]) != int(v[p - 1]) for p in Av):
            errs += 1
    return errs / n_cw


def bler_chained(channel, N, fu, fv, Au, Av, n_cw, seed=1001):
    """Measure Joint chained BLER (any U info mismatch OR any V info mismatch)."""
    from polar.encoder import polar_encode_batch
    rng = np.random.default_rng(seed)
    errs = 0
    u_only_errs = 0
    v_only_errs = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au:
            u[p - 1] = rng.integers(0, 2)
        for p in Av:
            v[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        u_hat, v_hat = decode_chained(z, N, fu, fv, channel)
        ue = any(int(u_hat[p - 1]) != int(u[p - 1]) for p in Au)
        ve = any(int(v_hat[p - 1]) != int(v[p - 1]) for p in Av)
        if ue:
            u_only_errs += 1
        if ve:
            v_only_errs += 1
        if ue or ve:
            errs += 1
    return {
        'chained_bler': errs / n_cw,
        'u_err_rate': u_only_errs / n_cw,
        'v_err_rate': v_only_errs / n_cw,
    }


if __name__ == "__main__":
    # Quick smoke test
    import time
    from polar.channels_memory import ISIMAC
    from polar.design_mc import design_from_file
    from polar.design import make_path

    N = 16
    n = 4
    ch = ISIMAC(sigma2=10 ** (-6 / 10), h=0.3)
    Au, Av, fu, fv, _, _, _ = design_from_file(
        f'designs/gmac_C_n{n}_snr6dB.npz', n, 4, 7)
    print(f'N={N} ku={len(Au)} kv={len(Av)}')

    t0 = time.time()
    r = bler_chained(ch, N, fu, fv, Au, Av, n_cw=500, seed=0)
    print(f'BLER={r["chained_bler"]:.4f} u_err={r["u_err_rate"]:.4f} '
          f'v_err={r["v_err_rate"]:.4f}  ({time.time()-t0:.1f}s)')
