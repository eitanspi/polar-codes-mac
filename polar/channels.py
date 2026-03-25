"""
channels.py
===========
Self-contained MAC channel models for two-user binary-input polar codes.

Implements the channels from Önay ISIT 2013:
  - BEMAC      : Binary Erasure MAC,  Z = X + Y  ∈ {0,1,2}
  - ABNMAC     : Additive Binary Noise MAC, Z = (X⊕Ex, Y⊕Ey)
  - GaussianMAC: Gaussian MAC with BPSK, Z = (1-2X) + (1-2Y) + W

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


# ─────────────────────────────────────────────────────────────────────────────
#  Gaussian MAC
# ─────────────────────────────────────────────────────────────────────────────

class GaussianMAC(MACChannel):
    """
    Gaussian MAC with BPSK modulation.

    Channel model
    -------------
        Z = (1-2X) + (1-2Y) + W,   W ~ N(0, sigma2)

    X, Y ∈ {0, 1} → BPSK signals s_x = 1-2X, s_y = 1-2Y ∈ {-1, +1}
    Possible noiseless outputs: {-2, 0, 0, +2}  for (X,Y) ∈ {(1,1),(0,1),(1,0),(0,0)}
    Z ∈ ℝ (continuous real-valued output)

    Parametrisation
    ---------------
    sigma2 : noise variance σ² (per-user SNR = 1/σ² since signal power = 1)
    SNR_dB : 10·log10(1/σ²)

    Capacity region (uniform binary inputs, code class C: path 0^N 1^N)
    -------------------------------------------------------------------
        I(Z;X)    — U marginal capacity  (V marginalised)
        I(Z;Y|X)  — V conditional capacity (standard BI-AWGN with means ±1)
        I(Z;X,Y)  = I(Z;X) + I(Z;Y|X)

    Bhattacharyya starting values (analytical/numerical)
    ----------------------------------------------------
        Z₀_v = exp(-1/(2σ²))     (BI-AWGN, closed form)
        Z₀_u = ∫ √(W₁(z|0)·W₁(z|1)) dz   (numerical)
    """
    name = "gaussian_mac"

    def __init__(self, sigma2=1.0):
        self.sigma2 = float(sigma2)
        self._log_norm = -0.5 * np.log(2 * np.pi * self.sigma2)

    @classmethod
    def from_snr_db(cls, snr_db: float):
        """Create a GaussianMAC from per-user SNR in dB."""
        sigma2 = 10.0 ** (-snr_db / 10.0)
        return cls(sigma2=sigma2)

    @property
    def snr_db(self):
        return -10.0 * np.log10(self.sigma2)

    def transition_prob(self, z: float, x: int, y: int) -> float:
        """Return W(z|x,y) — Gaussian PDF value."""
        mu = (1 - 2 * x) + (1 - 2 * y)
        return float(np.exp(self._log_norm - (z - mu) ** 2 / (2 * self.sigma2)))

    def sample(self, x: int, y: int) -> float:
        mu = (1 - 2 * x) + (1 - 2 * y)
        return float(mu + np.random.normal(0, np.sqrt(self.sigma2)))

    def sample_batch(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Vectorised batch sampling. Returns float64 array."""
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        mu = (1 - 2 * X).astype(np.float64) + (1 - 2 * Y).astype(np.float64)
        W = np.random.normal(0, np.sqrt(self.sigma2), size=mu.shape)
        return mu + W

    def capacity(self):
        """Compute (I_ZX, I_ZY_given_X, I_ZXY) by numerical integration."""
        sigma2 = self.sigma2
        sigma = np.sqrt(sigma2)

        z_max = max(8.0, 4 + 6 * sigma)
        z = np.linspace(-z_max, z_max, 20001)

        def gauss(zz, mu):
            return np.exp(-0.5 * (zz - mu) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)

        def _mi_binary(W0, W1, zz):
            """Mutual information of a binary-input channel with outputs W0, W1."""
            pz = 0.5 * W0 + 0.5 * W1
            mi = 0.0
            for Wx in [W0, W1]:
                safe = (pz > 1e-300) & (Wx > 1e-300)
                ratio = np.where(safe, Wx / np.where(safe, pz, 1.0), 1.0)
                mi += 0.5 * np.trapz(
                    np.where(safe, Wx * np.log2(np.maximum(ratio, 1e-300)), 0.0), zz
                )
            return float(max(0.0, mi))

        # V conditional channel (given X=0): W(z|y=0;x=0) = N(z;2,σ²),
        #                                    W(z|y=1;x=0) = N(z;0,σ²)
        # By symmetry I(Z;Y|X=0) = I(Z;Y|X=1)
        I_ZY_X = _mi_binary(gauss(z, 2), gauss(z, 0), z)

        # U marginal channel: W₁(z|0) = ½N(z;2,σ²)+½N(z;0,σ²)
        #                      W₁(z|1) = ½N(z;0,σ²)+½N(z;-2,σ²)
        W1_0 = 0.5 * gauss(z, 2) + 0.5 * gauss(z, 0)
        W1_1 = 0.5 * gauss(z, 0) + 0.5 * gauss(z, -2)
        I_ZX = _mi_binary(W1_0, W1_1, z)

        I_ZXY = I_ZX + I_ZY_X
        return I_ZX, I_ZY_X, I_ZXY


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

    print("\n--- Gaussian MAC ---")
    for snr_db in [0, 3, 6, 10]:
        g = GaussianMAC.from_snr_db(snr_db)
        I_ZX, I_ZY_X, I_ZXY = g.capacity()
        print(f"  SNR={snr_db:2d} dB  σ²={g.sigma2:.4f}  "
              f"I_ZX={I_ZX:.4f}  I_ZY|X={I_ZY_X:.4f}  I_ZXY={I_ZXY:.4f}")
    g3 = GaussianMAC.from_snr_db(3)
    print(f"\n  GaussianMAC(3dB) transition_prob(0.5, 0, 0) = "
          f"{g3.transition_prob(0.5, 0, 0):.6f}")
    z_test = g3.sample_batch(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))
    print(f"  sample_batch: {z_test}")
