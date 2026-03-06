"""
channels.py
===========
Self-contained MAC channel models for two-user binary-input polar codes.

Implements the channels from Önay ISIT 2013:
  - BEMAC : Binary Erasure MAC,  Z = X + Y  ∈ {0,1,2}
  - ABNMAC: Additive Binary Noise MAC, Z = (X⊕Ex, Y⊕Ey)

No external dependencies beyond NumPy.
"""

import numpy as np


class MACChannel:
    """Abstract base class for two-user MAC channels."""
    output_alphabet: list = []
    name: str = "mac"

    def transition_prob(self, z, x: int, y: int) -> float:
        """Return W(z | x, y) — scalar probability."""
        raise NotImplementedError

    def sample(self, x: int, y: int):
        """Sample one output symbol given inputs x, y."""
        raise NotImplementedError

    def sample_batch(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Sample channel outputs for a batch of codeword pairs.

        Parameters
        ----------
        X : np.ndarray, shape (batch, N) or (N,), dtype int
        Y : np.ndarray, shape (batch, N) or (N,), dtype int

        Returns
        -------
        Z : same shape as X — channel output symbols (dtype may differ)
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  BE-MAC
# ─────────────────────────────────────────────────────────────────────────────

class BEMAC(MACChannel):
    """
    Binary Erasure MAC  (BE-MAC).

    Channel model
    -------------
        Z = X + Y  ∈ {0, 1, 2}

    Capacity region (uniform inputs):
        Rx ≤ 0.5,  Ry ≤ 0.5,  Rx + Ry ≤ 1.0

    With path 0^N 1^N (code class C — decode all U then all V):
        I(Z;X)    = 0.5 bits   (U capacity: V marginalized)
        I(Z;Y|X)  = 1.0 bits   (V capacity: X perfectly known)

    Bhattacharyya starting values (analytical):
        Z₀_u = 0.5   (equivalent to BEC(0.5) for U effective channel)
        Z₀_v = 0.0   (all V channels are perfect → kv = N always valid)
    """
    output_alphabet = [0, 1, 2]
    name = "be_mac"

    def transition_prob(self, z: int, x: int, y: int) -> float:
        return 1.0 if z == x + y else 0.0

    def sample(self, x: int, y: int) -> int:
        return x + y

    def sample_batch(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Z = X + Y, vectorized."""
        return np.asarray(X, dtype=np.int32) + np.asarray(Y, dtype=np.int32)

    def capacity(self):
        """Return (I_ZX, I_ZY_given_X, I_ZXY) in bits."""
        return 0.5, 1.0, 1.5


# ─────────────────────────────────────────────────────────────────────────────
#  ABN-MAC
# ─────────────────────────────────────────────────────────────────────────────

class ABNMAC(MACChannel):
    """
    Additive Binary Noise MAC  (ABN-MAC).

    Channel model
    -------------
        Z = (Zx, Zy)   where   Zx = X ⊕ Ex,   Zy = Y ⊕ Ey
        (Ex, Ey) ~ p_noise   (correlated binary noise, independent of X, Y)

    Output alphabet: {0,1} × {0,1}  encoded as integer tuples (zx, zy).

    Default noise (from Önay ISIT 2013, Table I):
        p_noise = [[0.1286, 0.0175],
                   [0.0175, 0.8364]]

    Analytical capacities (default noise):
        I(Z;X)    ≈ 0.400 bits
        I(Z;Y|X)  ≈ 0.800 bits
        I(Z;X,Y)  ≈ 1.200 bits

    Bhattacharyya starting values:
        Z₀_u = 2 √(p_ex[0] · p_ex[1])            ≈ 0.706
        Z₀_v = 2[√(p00·p01) + √(p10·p11)]        ≈ 0.337
    """
    output_alphabet = [(0, 0), (0, 1), (1, 0), (1, 1)]
    name = "abn_mac"

    DEFAULT_NOISE = [
        [0.1286, 0.0175],
        [0.0175, 0.8364],
    ]

    def __init__(self, p_noise=None):
        self.p_noise = np.array(p_noise if p_noise is not None else self.DEFAULT_NOISE,
                                dtype=np.float64)
        assert abs(self.p_noise.sum() - 1.0) < 1e-9, "p_noise must sum to 1"
        # Flat CDF for sampling
        flat = [(ex, ey, self.p_noise[ex, ey])
                for ex in [0, 1] for ey in [0, 1]]
        self._ex_vals = np.array([f[0] for f in flat], dtype=np.int32)
        self._ey_vals = np.array([f[1] for f in flat], dtype=np.int32)
        self._cdf = np.cumsum([f[2] for f in flat])

    def transition_prob(self, z, x: int, y: int) -> float:
        zx, zy = z
        ex = x ^ zx
        ey = y ^ zy
        return float(self.p_noise[ex, ey])

    def sample(self, x: int, y: int):
        r = np.random.random()
        idx = int(np.searchsorted(self._cdf, r, side='right'))
        idx = min(idx, len(self._ex_vals) - 1)
        ex, ey = int(self._ex_vals[idx]), int(self._ey_vals[idx])
        return (x ^ ex, y ^ ey)

    def sample_batch(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Vectorized batch sampling.

        Returns Z as a numpy array of Python tuples (zx, zy), shape matching X.
        """
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        shape = X.shape
        n = X.size

        # Sample noise indices
        r = np.random.random(n)
        idx = np.searchsorted(self._cdf, r, side='right').clip(0, 3)
        Ex = self._ex_vals[idx].reshape(shape)
        Ey = self._ey_vals[idx].reshape(shape)

        Zx = X ^ Ex
        Zy = Y ^ Ey

        # Pack into tuple array — keep as (Zx, Zy) pair per element
        Z = np.empty(shape, dtype=object)
        it = np.nditer([Zx, Zy], flags=['multi_index'])
        while not it.finished:
            Z[it.multi_index] = (int(it[0]), int(it[1]))
            it.iternext()
        return Z

    def capacity(self):
        """Compute (I_ZX, I_ZY_given_X, I_ZXY) from noise distribution."""
        p = self.p_noise
        p_ex = p.sum(axis=1)  # marginal P(Ex=0), P(Ex=1)

        def h(p_val):
            if p_val <= 0 or p_val >= 1:
                return 0.0
            return -p_val * np.log2(p_val) - (1 - p_val) * np.log2(1 - p_val)

        # I(Z;X) = I(Zx;X) since Zy is uniform and independent of X
        # I(Zx;X) = 1 - H(Ex) = 1 - h(p_ex[1])
        I_ZX = 1.0 - h(float(p_ex[1]))

        # H(Z|X,Y) = H(Ex, Ey)
        H_noise = float(-np.sum(p * np.log2(np.where(p > 0, p, 1.0))))
        # H(Z) = log2(4) = 2 bits (Z is uniform due to uniform X,Y)
        I_ZXY = 2.0 - H_noise

        I_ZY_given_X = I_ZXY - I_ZX
        return float(I_ZX), float(I_ZY_given_X), float(I_ZXY)


if __name__ == "__main__":
    be = BEMAC()
    print("BE-MAC transition_prob(1, 0, 1) =", be.transition_prob(1, 0, 1))  # 1.0
    print("BE-MAC transition_prob(0, 1, 0) =", be.transition_prob(0, 1, 0))  # 0.0
    print("BE-MAC capacity:", be.capacity())

    abn = ABNMAC()
    print("\nABN-MAC transition_prob((0,0), 0, 0) =",
          abn.transition_prob((0, 0), 0, 0))  # p[0][0] = 0.1286
    I_ZX, I_ZY_X, I_ZXY = abn.capacity()
    print(f"ABN-MAC capacity: I_ZX={I_ZX:.4f}  I_ZY|X={I_ZY_X:.4f}  "
          f"I_ZXY={I_ZXY:.4f}")
