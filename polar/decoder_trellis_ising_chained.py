"""
decoder_trellis_ising_chained.py
================================
Chained (two-stage) SC decoder for the Ising MAC channel.

The Ising MAC has 2 states (GOOD=0, BAD=1) with Markov transitions
independent of inputs:
    P(s'=s|s) = 1-p_flip, P(s'!=s|s) = p_flip

GOOD state: Z = (1-2X) + (1-2Y) + W, W ~ N(0, sigma2)
BAD  state: Z = W (pure noise)

Stage 1 (decode U, marginalize Y):
  Forward-backward on 2-state Markov trellis.
  At each position: P(z|bit=x, state=s) = sum_y 0.5 * P(z|x,y,s)
  Transition: P(s'|s) = Markov (independent of x)

Stage 2 (decode V given X_hat):
  Forward-backward on 2-state Markov trellis.
  At each position: P(z|bit=y, state=s) = P(z|x_hat, y, s)  (x_hat known)
  Transition: P(s'|s) = Markov (independent of y)

Public API:
    decode_stage1_u(z, N, fu, channel)
    decode_stage2_v(z, u_hat, N, fv, channel)
    decode_chained(z, N, fu, fv, channel)
    bler_chained(channel, N, fu, fv, Au, Av, n_cw, seed)
"""
import numpy as np
from polar.encoder import polar_encode
from polar.decoder import _sc_decode_from_llr

_NEG_INF = -np.inf
_LOG_HALF = np.log(0.5)


def _log_emission_stage1(z, channel):
    """
    Stage 1 emission: marginalize Y as uniform.
    Returns (N, 2, 2): log_W[t, x, s] = log P(z_t | X=x, state=s)
    with Y marginalized as Bernoulli(1/2).
    """
    sigma2 = channel.sigma2
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma2)
    z = np.asarray(z, dtype=np.float64)
    N = z.shape[0]
    log_W = np.full((N, 2, 2), _NEG_INF, dtype=np.float64)

    for x in range(2):
        for s in range(2):
            if s == 0:  # GOOD
                # marginalize Y
                terms = []
                for y in range(2):
                    mu = (1 - 2 * x) + (1 - 2 * y)
                    log_p = log_norm - (z - mu) ** 2 / (2.0 * sigma2)
                    terms.append(log_p)
                # logsumexp of 2 terms + log(1/2)
                m = np.maximum(terms[0], terms[1])
                log_sum = m + np.log(np.exp(terms[0] - m) + np.exp(terms[1] - m))
                log_W[:, x, s] = log_sum + _LOG_HALF
            else:  # BAD: Z = W, independent of x,y
                log_p = log_norm - z ** 2 / (2.0 * sigma2)
                log_W[:, x, s] = log_p
    return log_W


def _log_emission_stage2(z, x_hat, channel):
    """
    Stage 2 emission: X is known.
    Returns (N, 2, 2): log_W[t, y, s] = log P(z_t | X=x_hat, Y=y, state=s)
    """
    sigma2 = channel.sigma2
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma2)
    z = np.asarray(z, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.int64)
    N = z.shape[0]
    log_W = np.full((N, 2, 2), _NEG_INF, dtype=np.float64)

    for y in range(2):
        for s in range(2):
            if s == 0:  # GOOD
                mu = (1 - 2 * x_hat).astype(np.float64) + (1 - 2 * y)
                log_p = log_norm - (z - mu) ** 2 / (2.0 * sigma2)
            else:  # BAD: Z = W
                log_p = log_norm - z ** 2 / (2.0 * sigma2)
            log_W[:, y, s] = log_p
    return log_W


def _forward_backward_markov(log_W, channel):
    """
    Forward-backward on a 2-state Markov trellis with state transitions
    independent of the input bit.

    log_W : (N, 2, 2) — log_W[t, bit, s] = log P(z_t | bit, state=s)
    channel: has p_flip attribute

    Returns log_marg : (N, 2) — unnormalized log-marginal of bit_t.
    """
    N = log_W.shape[0]
    p_flip = channel.p_flip
    log_stay = np.log(max(1 - p_flip, 1e-15))
    log_flip = np.log(max(p_flip, 1e-15))

    # Forward: alpha[t, s] = log P(Z_{0:t-1}, S_t=s)
    log_alpha = np.full((N + 1, 2), _NEG_INF, dtype=np.float64)
    log_alpha[0, 0] = 0.0  # initial state = GOOD

    for t in range(N):
        for bit in range(2):
            for s in range(2):
                val = log_alpha[t, s] + log_W[t, bit, s] + _LOG_HALF
                for s_next in range(2):
                    trans = log_stay if s_next == s else log_flip
                    log_alpha[t + 1, s_next] = np.logaddexp(
                        log_alpha[t + 1, s_next], val + trans)

    # Backward: beta[t, s] = log P(Z_{t:N-1} | S_t=s)
    log_beta = np.full((N + 1, 2), _NEG_INF, dtype=np.float64)
    log_beta[N, :] = 0.0

    for t in range(N - 1, -1, -1):
        for s in range(2):
            # sum over bits and next states
            acc = _NEG_INF
            for bit in range(2):
                for s_next in range(2):
                    trans = log_stay if s_next == s else log_flip
                    val = log_W[t, bit, s] + _LOG_HALF + trans + log_beta[t + 1, s_next]
                    acc = np.logaddexp(acc, val)
            log_beta[t, s] = acc

    # Marginal: log P(bit_t=b, Z)
    log_marg = np.full((N, 2), _NEG_INF, dtype=np.float64)
    for t in range(N):
        for bit in range(2):
            acc = _NEG_INF
            for s in range(2):
                for s_next in range(2):
                    trans = log_stay if s_next == s else log_flip
                    val = (log_alpha[t, s] + log_W[t, bit, s] + _LOG_HALF
                           + trans + log_beta[t + 1, s_next])
                    acc = np.logaddexp(acc, val)
            log_marg[t, bit] = acc

    return log_marg


def _marg_to_llr(log_marg):
    return log_marg[:, 0] - log_marg[:, 1]


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def decode_stage1_u(z, N, fu, channel):
    log_W1 = _log_emission_stage1(z, channel)
    log_marg = _forward_backward_markov(log_W1, channel)
    llr = _marg_to_llr(log_marg)
    u_hat = _sc_decode_from_llr(llr, fu)
    return u_hat


def decode_stage2_v(z, u_hat, N, fv, channel):
    x_hat = np.array(polar_encode(list(map(int, u_hat))), dtype=np.int64)
    log_W2 = _log_emission_stage2(z, x_hat, channel)
    log_marg = _forward_backward_markov(log_W2, channel)
    llr = _marg_to_llr(log_marg)
    v_hat = _sc_decode_from_llr(llr, fv)
    return v_hat


def decode_chained(z, N, fu, fv, channel):
    u_hat = decode_stage1_u(z, N, fu, channel)
    v_hat = decode_stage2_v(z, u_hat, N, fv, channel)
    return u_hat, v_hat


def bler_chained(channel, N, fu, fv, Au, Av, n_cw, seed=1001):
    """Measure chained BLER for Ising MAC."""
    from polar.encoder import polar_encode_batch
    rng = np.random.default_rng(seed)
    errs_u = errs_v = errs_total = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au:
            u[p - 1] = rng.integers(0, 2)
        for p in Av:
            v[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))
        z_vec = np.asarray(z, dtype=np.float64)
        if z_vec.ndim == 2:
            z_vec = z_vec[0]
        u_hat, v_hat = decode_chained(z_vec, N, fu, fv, channel)
        ue = any(int(u_hat[p - 1]) != int(u[p - 1]) for p in Au)
        ve = any(int(v_hat[p - 1]) != int(v[p - 1]) for p in Av)
        if ue: errs_u += 1
        if ve: errs_v += 1
        if ue or ve: errs_total += 1
    return {
        'chained_bler': errs_total / n_cw,
        'u_err_rate': errs_u / n_cw,
        'v_err_rate': errs_v / n_cw,
        'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
    }


if __name__ == "__main__":
    import time
    from polar.channels_memory_new import IsingMAC
    from polar.design_mc import design_from_file
    import math

    N = 16
    n = int(math.log2(N))
    ch = IsingMAC(sigma2=0.251, p_flip=0.1)
    Au_list, Av_list, fu_dict, fv_dict, _, _, _ = design_from_file(
        f'designs/gmac_C_n{n}_snr6dB.npz', n, 4, 7)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    fu = {p: 0 for p in range(1, N + 1) if p not in Au}
    fv = {p: 0 for p in range(1, N + 1) if p not in Av}

    print(f'N={N} ku={len(Au)} kv={len(Av)}')
    t0 = time.time()
    r = bler_chained(ch, N, fu, fv, Au, Av, n_cw=500, seed=0)
    print(f'BLER={r["chained_bler"]:.4f} u_err={r["u_err_rate"]:.4f} '
          f'v_err={r["v_err_rate"]:.4f} ({time.time()-t0:.1f}s)')
