"""Sanity-check the chained SCT decoder against:

(A) Brute-force per-position log-likelihood:
    log P(Z, X_t = x_t) = LSE over all (X_{0..N-1} with X_t=x_t fixed, Y_0..N-1)
                          of  log P(Z | X, Y) + (-N log 2 - N log 2)
    Verified against `_log_W_stage1` + FB output at N=8.

(B) h=0 limit: chained SCT stage-1 LLR should equal memoryless SC's
    U-marginal LLR (since the ISI term vanishes — Y_prev becomes irrelevant).

(C) Initial-state convention: x_{-1}=0, y_{-1}=0 matches channel padding
    (BPSK of 0 = +1 → bxp[0]=1, byp[0]=1).
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.channels_memory import ISIMAC
from polar.decoder_trellis_mac_chained import (
    _log_W_stage1, _log_W_stage2, _forward_backward_2state, _marg_to_llr,
)
from polar.decoder import _u_marginal_llr


def brute_force_stage1_llr(z, ch):
    """Brute-force compute log P(Z, X_t=x_t) by summing over all (X, Y) ∈ {0,1}^(2N)
    with X_t fixed. Returns (N, 2) log-marginals.
    """
    N = len(z)
    sigma2 = ch.sigma2
    h = ch.h
    log_norm = -0.5 * np.log(2 * np.pi * sigma2)

    log_marg = np.full((N, 2), -np.inf)
    log_unif = -np.log(2.0)  # 1/2 prior per bit

    for X_int in range(2 ** N):
        X = np.array([(X_int >> i) & 1 for i in range(N)])
        bx = 1.0 - 2.0 * X
        for Y_int in range(2 ** N):
            Y = np.array([(Y_int >> i) & 1 for i in range(N)])
            by = 1.0 - 2.0 * Y
            bxp = np.concatenate([[1.0], bx[:-1]])  # x_{-1}=0
            byp = np.concatenate([[1.0], by[:-1]])  # y_{-1}=0
            mu = bx + by + h * (bxp + byp)
            log_p_z_given_xy = (
                log_norm * N - np.sum((z - mu) ** 2) / (2 * sigma2)
            )
            # log P(X) + log P(Y) = -N log 4
            log_joint = log_p_z_given_xy + 2 * N * log_unif
            for t in range(N):
                xt = int(X[t])
                log_marg[t, xt] = np.logaddexp(log_marg[t, xt], log_joint)

    return log_marg


def test_A_brute_force(N=6, h=0.3, snr_db=6, seed=0):
    """FB output for chained-SCT stage-1 must match exhaustive marginal."""
    print(f"=== Test A: N={N} brute-force vs FB stage 1 (h={h}, SNR={snr_db}dB) ===")
    sigma2 = 10 ** (-snr_db / 10)
    ch = ISIMAC(sigma2=sigma2, h=h)
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=N)
    Y = rng.integers(0, 2, size=N)
    z = ch.sample_batch(X.reshape(1, -1), Y.reshape(1, -1))[0]

    log_W1 = _log_W_stage1(z, ch)
    log_marg_fb = _forward_backward_2state(log_W1)  # (N, 2)
    llr_fb = _marg_to_llr(log_marg_fb)

    log_marg_bf = brute_force_stage1_llr(z, ch)  # (N, 2)
    llr_bf = log_marg_bf[:, 0] - log_marg_bf[:, 1]

    print(f"  LLR FB     : {np.array2string(llr_fb, precision=4)}")
    print(f"  LLR BF     : {np.array2string(llr_bf, precision=4)}")
    print(f"  max |diff| : {np.max(np.abs(llr_fb - llr_bf)):.2e}")
    ok = np.allclose(llr_fb, llr_bf, atol=1e-6)
    print(f"  RESULT     : {'PASS' if ok else 'FAIL'}")
    return ok


def test_B_h_zero_memoryless(N=8, snr_db=6, seed=1):
    """At h=0, chained-SCT stage-1 LLR must equal memoryless SC's U-marginal LLR."""
    print(f"\n=== Test B: h=0 limit, chained stage-1 vs memoryless U-marginal LLR ===")
    sigma2 = 10 ** (-snr_db / 10)
    ch = ISIMAC(sigma2=sigma2, h=0.0)
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=N)
    Y = rng.integers(0, 2, size=N)
    z = ch.sample_batch(X.reshape(1, -1), Y.reshape(1, -1))[0]

    # Chained-SCT stage 1 LLR
    log_W1 = _log_W_stage1(z, ch)
    log_marg = _forward_backward_2state(log_W1)
    llr_chained = _marg_to_llr(log_marg)

    # Memoryless SC U-marginal LLR (manual: build (N,2,2) log_W)
    log_norm = -0.5 * np.log(2 * np.pi * sigma2)
    log_W_mem = np.empty((N, 2, 2))
    for x in range(2):
        for y in range(2):
            mu = (1.0 - 2 * x) + (1.0 - 2 * y)
            log_W_mem[:, x, y] = log_norm - (z - mu) ** 2 / (2 * sigma2)
    llr_mem = _u_marginal_llr(log_W_mem)

    print(f"  LLR chained: {np.array2string(llr_chained, precision=4)}")
    print(f"  LLR memless: {np.array2string(llr_mem, precision=4)}")
    print(f"  max |diff| : {np.max(np.abs(llr_chained - llr_mem)):.2e}")
    ok = np.allclose(llr_chained, llr_mem, atol=1e-9)
    print(f"  RESULT     : {'PASS' if ok else 'FAIL'}")
    return ok


def test_C_initial_state(N=4, h=0.3, snr_db=6):
    """Stage-1 mu at t=0 should equal bx + by + h*(1 + 1) = bx + by + 2h.
    At t=1, mu should depend on X_{0}, Y_{0} via the previous BPSK symbols.
    Test by varying the channel realization.
    """
    print(f"\n=== Test C: initial-state convention ===")
    sigma2 = 10 ** (-snr_db / 10)
    ch = ISIMAC(sigma2=sigma2, h=h)

    # Sample with all bits = 0 → all BPSK +1, mu at t=0 = 1 + 1 + h*(1+1) = 2 + 2h
    X = np.zeros(N, dtype=int)
    Y = np.zeros(N, dtype=int)
    np.random.seed(0)
    z = ch.sample_batch(X.reshape(1, -1), Y.reshape(1, -1))[0]
    expected_mu_t0 = 1.0 + 1.0 + h * (1.0 + 1.0)
    print(f"  All-zeros input, z[0] (sampled) close to {expected_mu_t0:.3f}? "
          f"z[0]={z[0]:.3f}")

    # Build chained log_W1, check that mu used for (x=0, a=0) matches
    # mu_at_t0(x=0, x_prev=0, Y marginalised uniform)
    log_W1 = _log_W_stage1(z, ch)
    # Reverse-engineer the mu being implicitly used: log_W1[t, x, a] is the LSE
    # over 4 (y, b) terms. At t=0, x=0, a=0, the 4 terms have mu values:
    #   (y=0, b=0): 1+1+h(1+1) = 2+2h
    #   (y=0, b=1): 1+1+h(1-1) = 2
    #   (y=1, b=0): 1-1+h(1+1) = 2h
    #   (y=1, b=1): 1-1+h(1-1) = 0
    # We can check log_W1[0, 0, 0] = log [ (1/4) * sum of exp(-(z[0]-mu_i)^2/2sigma2) * norm ]
    log_norm = -0.5 * np.log(2 * np.pi * sigma2)
    expected_mus = [2 + 2*h, 2.0, 2*h, 0.0]
    log_terms = [log_norm - (z[0] - m)**2 / (2 * sigma2) for m in expected_mus]
    m = max(log_terms)
    expected_log_W = m + np.log(sum(np.exp(t - m) for t in log_terms)) + np.log(0.25)
    got = log_W1[0, 0, 0]
    print(f"  log_W1[t=0, x=0, a=0]: got={got:.4f}, expected={expected_log_W:.4f}, "
          f"diff={abs(got - expected_log_W):.2e}")
    ok = abs(got - expected_log_W) < 1e-9
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    r1 = test_A_brute_force(N=6)
    r2 = test_B_h_zero_memoryless(N=8)
    r3 = test_C_initial_state(N=4)
    print(f"\n=== Summary ===")
    print(f"  A (brute force):       {'PASS' if r1 else 'FAIL'}")
    print(f"  B (h=0 → memoryless):  {'PASS' if r2 else 'FAIL'}")
    print(f"  C (initial-state μ):   {'PASS' if r3 else 'FAIL'}")
