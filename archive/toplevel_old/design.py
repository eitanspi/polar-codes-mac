"""
design.py
=========
Analytical Bhattacharyya polar code design for MAC channels.

For path 0^N 1^N (code class C):
  - U decoded with no side information: uses W_1(z|x) = Σ_y ½ W(z|x,y)
  - V decoded with full U side information: uses W_2(z|y;x) = W(z|x,y)

Arikan polar code recursion (Arikan 2009):
    Z(W^-) = 2Z - Z²    (bad split, higher Z = less reliable)
    Z(W^+) = Z²         (good split, lower Z = more reliable)

References
----------
Arikan (2009) "Channel Polarization: A Method for Constructing Capacity-
  Achieving Codes for Symmetric Binary-Input Memoryless Channels"
Önay (ISIT 2013) "Successive Cancellation Decoding of Polar Codes for the
  Two-User Binary-Input MAC"
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Core Bhattacharyya recursion
# ─────────────────────────────────────────────────────────────────────────────

def bhattacharyya_recursion(Z0: float, n: int) -> np.ndarray:
    """
    Apply n stages of Arikan polar code recursion starting from Z0.

    Returns array of shape (2^n,) with Bhattacharyya parameter per channel.
    Channel index i corresponds to polar sub-channel W^{(i+1)}.

    Ordering: even indices ← Z^- (bad split), odd indices ← Z^+ (good split).
    After n stages, channels are in natural order matching the path 0^N 1^N.

    Parameters
    ----------
    Z0 : float in [0, 1] — starting Bhattacharyya parameter
    n  : int             — number of polarization stages (N = 2^n)
    """
    z = np.array([float(Z0)])
    for _ in range(n):
        z_bad  = np.minimum(1.0, 2.0 * z - z ** 2)  # Z^-
        z_good = z ** 2                               # Z^+
        z_new  = np.empty(2 * len(z), dtype=np.float64)
        z_new[0::2] = z_bad    # even positions ← bad (min) split
        z_new[1::2] = z_good   # odd  positions ← good (max) split
        z = z_new
    return z


# ─────────────────────────────────────────────────────────────────────────────
#  BE-MAC design
# ─────────────────────────────────────────────────────────────────────────────

def bhattacharyya_bemac(n: int):
    """
    Analytical Bhattacharyya parameters for BE-MAC with path 0^N 1^N.

    BE-MAC: Z = X + Y.  Output alphabet {0, 1, 2}.

    U effective channel:  W_1(z|x) = Σ_y ½ W(z|x,y)
      W_1(0|0) = ½,  W_1(1|0) = ½,  W_1(2|0) = 0
      W_1(0|1) = 0,  W_1(1|1) = ½,  W_1(2|1) = ½
      Z₀_u = Σ_z √(W_1(z|0)·W_1(z|1)) = √(½·0)+√(½·½)+√(0·½) = 0.5

    V effective channel (X fully known, decoded first):
      W_2(z|y;x) = W(z|x,y) = 1_{z=x+y}
      Z₀_v = 0  →  ALL V coordinate channels are perfect

    Returns
    -------
    z_u : np.ndarray shape (N,) — Bhattacharyya params for U channels
    z_v : np.ndarray shape (N,) — Bhattacharyya params for V channels (all 0)
    """
    N = 1 << n
    z_u = bhattacharyya_recursion(0.5, n)
    z_v = np.zeros(N, dtype=np.float64)
    return z_u, z_v


def design_bemac(n: int, ku: int, kv: int):
    """
    Design polar code for BE-MAC with path 0^N 1^N.

    For code class C: kv = N (all V bits are information, Z₀_v = 0).
    U information bits: best ku positions (lowest Bhattacharyya Z).

    Parameters
    ----------
    n  : log2(N)
    ku : number of U information bits
    kv : number of V information bits (should be N for BE-MAC)

    Returns
    -------
    Au      : list[int] — 1-indexed U info positions, sorted
    Av      : list[int] — 1-indexed V info positions, sorted
    frozen_u: dict {1-indexed pos: 0} — U frozen positions
    frozen_v: dict {1-indexed pos: 0} — V frozen positions
    z_u     : np.ndarray — U Bhattacharyya parameters
    z_v     : np.ndarray — V Bhattacharyya parameters
    """
    N = 1 << n
    z_u, z_v = bhattacharyya_bemac(n)

    # Select best ku U channels (smallest Z = most reliable)
    Au_0idx = sorted(np.argsort(z_u)[:ku].tolist())
    Au = [i + 1 for i in Au_0idx]

    # All kv best V channels (all perfect, pick first kv)
    Av = list(range(1, kv + 1))

    all_pos = set(range(1, N + 1))
    frozen_u = {pos: 0 for pos in sorted(all_pos - set(Au))}
    frozen_v = {pos: 0 for pos in sorted(all_pos - set(Av))}

    return Au, Av, frozen_u, frozen_v, z_u, z_v


# ─────────────────────────────────────────────────────────────────────────────
#  ABN-MAC design
# ─────────────────────────────────────────────────────────────────────────────

def bhattacharyya_abnmac(n: int, p_noise=None):
    """
    Analytical Bhattacharyya parameters for ABN-MAC with path 0^N 1^N.

    ABN-MAC: Z=(X⊕Ex, Y⊕Ey), (Ex,Ey)~p_noise.

    U effective channel:  W_1((zx,zy)|x) = ½ p_ex[x⊕zx]
      Z₀_u = 2√(p_ex[0]·p_ex[1])

    V effective channel (X decoded, x fixed):  W_2((zx,zy)|y;x) = p_noise[x⊕zx][y⊕zy]
      Z₀_v = 2[√(p00·p01) + √(p10·p11)]  (same value for x=0 and x=1)

    Parameters
    ----------
    n       : log2(N)
    p_noise : 2×2 array-like — joint noise distribution P(Ex,Ey)
              Default: [[0.1286,0.0175],[0.0175,0.8364]]

    Returns
    -------
    z_u, z_v : np.ndarray shape (N,) each
    """
    if p_noise is None:
        p_noise = [[0.1286, 0.0175], [0.0175, 0.8364]]
    p = np.array(p_noise, dtype=np.float64)
    p_ex = p.sum(axis=1)

    Z0_u = 2.0 * np.sqrt(p_ex[0] * p_ex[1])
    Z0_v = 2.0 * (np.sqrt(p[0, 0] * p[0, 1]) + np.sqrt(p[1, 0] * p[1, 1]))

    z_u = bhattacharyya_recursion(float(Z0_u), n)
    z_v = bhattacharyya_recursion(float(Z0_v), n)
    return z_u, z_v


def design_abnmac(n: int, ku: int, kv: int, p_noise=None):
    """
    Design polar code for ABN-MAC with path 0^N 1^N.

    For code class C:
        ku = round(I(Z;X)   · N)   (U capacity constraint)
        kv = round(I(Z;Y|X) · N)   (V capacity constraint — NOT N)
        kv should additionally be capped at sum(z_v < 0.5) for finite N.

    Returns
    -------
    Same as design_bemac: Au, Av, frozen_u, frozen_v, z_u, z_v
    """
    N = 1 << n
    z_u, z_v = bhattacharyya_abnmac(n, p_noise)

    Au_0idx = sorted(np.argsort(z_u)[:ku].tolist())
    Au = [i + 1 for i in Au_0idx]

    Av_0idx = sorted(np.argsort(z_v)[:kv].tolist())
    Av = [i + 1 for i in Av_0idx]

    all_pos = set(range(1, N + 1))
    frozen_u = {pos: 0 for pos in sorted(all_pos - set(Au))}
    frozen_v = {pos: 0 for pos in sorted(all_pos - set(Av))}

    return Au, Av, frozen_u, frozen_v, z_u, z_v


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian Approximation (GA) density evolution  (Trifonov 2012)
# ─────────────────────────────────────────────────────────────────────────────

def _phi_ga(x):
    """
    phi(x) for Gaussian Approximation density evolution.

    phi(x) = 1 - (1/sqrt(4*pi*x)) * integral exp(-(t-x)^2/(4x)) * tanh(t/2) dt
            for x > 0; phi(0) = 1.

    For the GA, LLRs under the all-zeros codeword assumption are
    N(mu, 2*mu), and phi(mu) gives the error probability.

    Approximation from Chung et al. / Vangala et al.:
      phi(x) ≈ exp(-0.4527*x^0.86 + 0.0218)   for x in (0, 10)
      phi(x) ≈ sqrt(pi/x) * exp(-x/4)          for x >= 10
    """
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.ones_like(x)

    # x > 0 region
    mask_small = (x > 0) & (x < 10)
    mask_large = x >= 10

    result[mask_small] = np.exp(-0.4527 * x[mask_small] ** 0.86 + 0.0218)
    result[mask_large] = np.sqrt(np.pi / x[mask_large]) * np.exp(-x[mask_large] / 4.0)
    result = np.clip(result, 0.0, 1.0)

    return float(result) if scalar else result


def _phi_inv_ga(y):
    """
    Inverse of phi for GA density evolution.

    Given y = phi(x), find x. Uses bisection since phi is monotonically
    decreasing. For y close to 0 (large x), uses the asymptotic form
    phi(x) ~ sqrt(pi/x) * exp(-x/4) to handle very small y.
    """
    y = float(y)
    if y >= 1.0:
        return 0.0
    if y <= 0.0:
        return 1e8

    # For very small y, use asymptotic inversion of phi(x) ~ sqrt(pi/x)*exp(-x/4)
    # => x/4 ~ -ln(y) + 0.5*ln(pi/x), for large x: x ~ -4*ln(y)
    if y < 1e-10:
        # Newton refinement on log(phi(x)) = log(y)
        # log(phi(x)) ~ 0.5*log(pi) - 0.5*log(x) - x/4
        x = -4.0 * np.log(y)  # initial guess
        for _ in range(10):
            log_phi = 0.5 * np.log(np.pi / x) - x / 4.0
            target = np.log(y)
            # d(log_phi)/dx = -0.5/x - 0.25
            deriv = -0.5 / x - 0.25
            x = x - (log_phi - target) / deriv
            x = max(x, 1.0)
        return x

    # Bisection for moderate y
    lo, hi = 1e-8, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _phi_ga(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def ga_recursion(mu0: float, n: int) -> np.ndarray:
    """
    Gaussian Approximation density evolution for polar codes.

    Tracks the LLR mean mu (assuming LLR ~ N(mu, 2*mu)) through n
    polarization stages. The error probability of channel i is phi(mu[i]).

    Recursion (Trifonov 2012):
        mu^- = phi_inv(1 - (1 - phi(mu))^2)     (check node)
        mu^+ = 2 * mu                             (variable node)

    Parameters
    ----------
    mu0 : float > 0 — initial LLR mean of the base channel
    n   : int       — number of polarization stages (N = 2^n)

    Returns
    -------
    mu : np.ndarray shape (2^n,) — LLR means per synthetic channel
    """
    mu = np.array([float(mu0)])
    for _ in range(n):
        phi_mu = _phi_ga(mu)
        # Check node (bad split): mu_bad = phi_inv(1 - (1-phi(mu))^2)
        p_bad = 1.0 - (1.0 - phi_mu) ** 2
        p_bad = np.clip(p_bad, 1e-15, 1.0)
        mu_bad = np.array([_phi_inv_ga(float(p)) for p in p_bad])
        # Variable node (good split): mu_good = 2*mu
        mu_good = 2.0 * mu

        mu_new = np.empty(2 * len(mu), dtype=np.float64)
        mu_new[0::2] = mu_bad
        mu_new[1::2] = mu_good
        mu = mu_new
    return mu


def _mu0_biawgn(sigma2: float) -> float:
    """Initial LLR mean for BI-AWGN channel with noise variance sigma2.

    For BPSK ±1 and noise N(0, sigma2): LLR = 2*z/sigma2,
    and under x=+1: E[LLR] = 2/sigma2.
    """
    return 2.0 / sigma2


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian MAC design
# ─────────────────────────────────────────────────────────────────────────────

def _Z0_u_gmac(sigma2: float) -> float:
    """
    Bhattacharyya parameter Z₀ for the U-marginal channel of the Gaussian MAC.

        Z₀_u = ∫ √(W₁(z|0) · W₁(z|1)) dz

    where W₁(z|x) = ½ N(z; (1-2x)+1, σ²) + ½ N(z; (1-2x)-1, σ²).

    Computed by numerical integration.
    """
    sigma = np.sqrt(sigma2)
    z_max = max(8.0, 4 + 6 * sigma)
    z = np.linspace(-z_max, z_max, 20001)

    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma2)

    def gauss(zz, mu):
        return norm * np.exp(-0.5 * (zz - mu) ** 2 / sigma2)

    W1_0 = 0.5 * gauss(z, 2) + 0.5 * gauss(z, 0)
    W1_1 = 0.5 * gauss(z, 0) + 0.5 * gauss(z, -2)

    integrand = np.sqrt(W1_0 * W1_1)
    return float(np.trapz(integrand, z))


def bhattacharyya_gmac(n: int, sigma2: float = 1.0):
    """
    Bhattacharyya parameters for Gaussian MAC with path 0^N 1^N.

    NOTE: Bhattacharyya recursion is loose for Gaussian channels.
    Use ga_gmac() or design_gmac(method='ga') for tighter designs.

    Returns z_u, z_v : np.ndarray shape (N,) each
    """
    Z0_u = _Z0_u_gmac(sigma2)
    Z0_v = np.exp(-1.0 / (2.0 * sigma2))

    z_u = bhattacharyya_recursion(float(Z0_u), n)
    z_v = bhattacharyya_recursion(float(Z0_v), n)
    return z_u, z_v


def _mu0_u_gmac(sigma2: float) -> float:
    """
    Initial LLR mean for the U-marginal channel of Gaussian MAC.

    The U-marginal channel is a Gaussian mixture, not a simple BI-AWGN.
    W₁(z|0) = ½N(z;2,σ²) + ½N(z;0,σ²)
    W₁(z|1) = ½N(z;0,σ²) + ½N(z;-2,σ²)

    LLR = log(W₁(z|0)/W₁(z|1)).

    Under x=0 (sent +1), compute E[LLR] by numerical integration:
        mu0 = E_z[LLR | x=0] = ∫ W₁(z|0) * log(W₁(z|0)/W₁(z|1)) dz
    """
    sigma = np.sqrt(sigma2)
    z_max = max(8.0, 4 + 6 * sigma)
    z = np.linspace(-z_max, z_max, 20001)

    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma2)

    def gauss(zz, mu):
        return norm * np.exp(-0.5 * (zz - mu) ** 2 / sigma2)

    W0 = 0.5 * gauss(z, 2) + 0.5 * gauss(z, 0)
    W1 = 0.5 * gauss(z, 0) + 0.5 * gauss(z, -2)

    # LLR = log(W0/W1), avoid log(0)
    safe = (W0 > 1e-300) & (W1 > 1e-300)
    llr = np.where(safe, np.log(W0 / np.where(safe, W1, 1.0)), 0.0)

    # E[LLR | x=0] = integral W0 * llr dz
    mu0 = float(np.trapz(W0 * llr, z))
    return max(1e-10, mu0)


def ga_gmac(n: int, sigma2: float = 1.0):
    """
    Gaussian Approximation density evolution for Gaussian MAC with path 0^N 1^N.

    Uses GA recursion (Trifonov 2012) instead of Bhattacharyya bounds.
    Much tighter for Gaussian channels — gives correct channel quality
    estimates that match Monte Carlo design.

    U effective channel: mu0_u from numerical integration of mixture LLR.
    V effective channel: mu0_v = 2/σ² (standard BI-AWGN).

    Parameters
    ----------
    n      : log2(N)
    sigma2 : noise variance σ²

    Returns
    -------
    pe_u, pe_v : np.ndarray shape (N,) — error probability per channel
                 (lower = more reliable, use as ranking metric like Bhattacharyya Z)
    """
    mu0_u = _mu0_u_gmac(sigma2)
    mu0_v = _mu0_biawgn(sigma2)

    mu_u = ga_recursion(mu0_u, n)
    mu_v = ga_recursion(mu0_v, n)

    pe_u = _phi_ga(mu_u)
    pe_v = _phi_ga(mu_v)
    return pe_u, pe_v


def design_gmac(n: int, ku: int, kv: int, sigma2: float = 1.0,
                method: str = 'ga'):
    """
    Design polar code for Gaussian MAC with path 0^N 1^N.

    Parameters
    ----------
    n      : log2(N)
    ku     : number of U information bits
    kv     : number of V information bits
    sigma2 : noise variance σ²
    method : 'ga' (Gaussian Approximation, recommended) or
             'bhatt' (Bhattacharyya, loose upper bound)

    Returns
    -------
    Au, Av, frozen_u, frozen_v, rel_u, rel_v
        rel_u/rel_v: reliability metric per channel (lower = more reliable)
        For GA: error probability.  For Bhatt: Bhattacharyya Z parameter.
    """
    N = 1 << n

    if method == 'ga':
        rel_u, rel_v = ga_gmac(n, sigma2)
    else:
        rel_u, rel_v = bhattacharyya_gmac(n, sigma2)

    Au_0idx = sorted(np.argsort(rel_u)[:ku].tolist())
    Au = [i + 1 for i in Au_0idx]

    Av_0idx = sorted(np.argsort(rel_v)[:kv].tolist())
    Av = [i + 1 for i in Av_0idx]

    all_pos = set(range(1, N + 1))
    frozen_u = {pos: 0 for pos in sorted(all_pos - set(Au))}
    frozen_v = {pos: 0 for pos in sorted(all_pos - set(Av))}

    return Au, Av, frozen_u, frozen_v, rel_u, rel_v


# ─────────────────────────────────────────────────────────────────────────────
#  Path construction
# ─────────────────────────────────────────────────────────────────────────────

def make_path(N: int, path_i: int) -> list:
    """
    Build path b^{2N} = 0^{path_i} 1^N 0^{N-path_i}   (Önay 2013, Sec. V).

    path_i = N → 0^N 1^N  (code class C: all U first, then all V)
    path_i = 0 → 1^N 0^N  (all V first, then all U)

    Each 0 in b corresponds to decoding one U bit; each 1 decodes one V bit.
    """
    assert 0 <= path_i <= N
    return [0] * path_i + [1] * N + [0] * (N - path_i)


# ─────────────────────────────────────────────────────────────────────────────
#  Summary utility
# ─────────────────────────────────────────────────────────────────────────────

def summarize_design(n: int, ku: int, kv: int, z_u: np.ndarray, z_v: np.ndarray):
    """Print a compact summary of the code design."""
    N = 1 << n
    print(f"  N={N}  n={n}  ku={ku} (Ru={ku/N:.3f})  kv={kv} (Rv={kv/N:.3f})")
    print(f"  U: {int(np.sum(z_u < 0.1))} channels Z<0.1  "
          f"({int(np.sum(z_u < 0.01))} Z<0.01)  "
          f"min_Z={z_u.min():.4f}  max_Z_info={np.sort(z_u)[ku-1]:.4f}")
    print(f"  V: {int(np.sum(z_v < 0.1))} channels Z<0.1  "
          f"({int(np.sum(z_v < 0.01))} Z<0.01)  "
          f"min_Z={z_v.min():.4f}  "
          + (f"max_Z_info={np.sort(z_v)[kv-1]:.4f}" if kv > 0 else "kv=0"))


if __name__ == "__main__":
    import sys

    print("=== BE-MAC design (N=16, ku=4, kv=16) ===")
    Au, Av, fu, fv, zu, zv = design_bemac(n=4, ku=4, kv=16)
    print(f"Au = {Au}")
    print(f"Av = {Av}")
    summarize_design(4, 4, 16, zu, zv)

    print("\n=== ABN-MAC design (N=16, ku=3, kv=8) ===")
    Au2, Av2, fu2, fv2, zu2, zv2 = design_abnmac(n=4, ku=3, kv=8)
    print(f"Au = {Au2}")
    print(f"Av = {Av2}")
    summarize_design(4, 3, 8, zu2, zv2)

    print("\n=== Gaussian MAC design (N=16, sigma2=0.5) ===")
    Au3, Av3, fu3, fv3, zu3, zv3 = design_gmac(n=4, ku=3, kv=8, sigma2=0.5)
    print(f"Au = {Au3}")
    print(f"Av = {Av3}")
    summarize_design(4, 3, 8, zu3, zv3)

    print("\n=== Gaussian MAC Bhattacharyya Z₀ vs SNR ===")
    for snr_db in [0, 3, 6, 10]:
        s2 = 10.0 ** (-snr_db / 10.0)
        Z0u = _Z0_u_gmac(s2)
        Z0v = np.exp(-1.0 / (2.0 * s2))
        print(f"  SNR={snr_db:2d} dB  σ²={s2:.4f}  Z₀_u={Z0u:.4f}  Z₀_v={Z0v:.4f}")

    print("\n=== Bhattacharyya polarization (N=64) ===")
    zu64, zv64 = bhattacharyya_bemac(6)
    print(f"BE-MAC U: min={zu64.min():.4e}  max={zu64.max():.4e}  "
          f"frac<0.1={np.mean(zu64<0.1):.3f}")
