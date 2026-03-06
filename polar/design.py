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

    print("\n=== Bhattacharyya polarization (N=64) ===")
    zu64, zv64 = bhattacharyya_bemac(6)
    print(f"BE-MAC U: min={zu64.min():.4e}  max={zu64.max():.4e}  "
          f"frac<0.1={np.mean(zu64<0.1):.3f}")
