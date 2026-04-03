"""
channels_memory.py
==================
MAC channel models with memory for two-user binary-input polar codes.

Implements:
  - ISIMAC: ISI MAC with BPSK, Z[i] = s_x[i] + s_y[i] + h*(s_x[i-1] + s_y[i-1]) + W[i]

State representation:
  For ISI-MAC with tap length 1, the state is (X[i-1], Y[i-1]).
  Encoded as s = 2*x_prev + y_prev, so |S| = 4.
"""

import numpy as np
from polar.channels import MACChannel


class ISIMAC(MACChannel):
    """
    ISI (Inter-Symbol Interference) MAC with BPSK modulation.

    Channel model
    -------------
        Z[i] = (1-2X[i]) + (1-2Y[i]) + h*((1-2X[i-1]) + (1-2Y[i-1])) + W[i]

    where W[i] ~ N(0, sigma2), h is the ISI coefficient.

    State: s = 2*X[i-1] + Y[i-1]  ∈ {0, 1, 2, 3}
    State transition: s' = 2*X[i] + Y[i]  (deterministic given current inputs)

    Parameters
    ----------
    sigma2 : float — noise variance
    h      : float — ISI tap coefficient (default 0.5)
    """
    name = "isi_mac"

    def __init__(self, sigma2=1.0, h=0.5):
        self.sigma2 = float(sigma2)
        self.h = float(h)
        self.num_states = 4  # |S| = 4 for (x_prev, y_prev) ∈ {0,1}²
        self._log_norm = -0.5 * np.log(2 * np.pi * self.sigma2)

    @classmethod
    def from_snr_db(cls, snr_db: float, h=0.5):
        """Create an ISIMAC from per-user SNR in dB."""
        sigma2 = 10.0 ** (-snr_db / 10.0)
        return cls(sigma2=sigma2, h=h)

    @property
    def snr_db(self):
        return -10.0 * np.log10(self.sigma2)

    @staticmethod
    def _decode_state(s):
        """Decode state integer to (x_prev, y_prev)."""
        return s >> 1, s & 1

    @staticmethod
    def _encode_state(x, y):
        """Encode (x, y) pair to state integer."""
        return 2 * x + y

    def _mu(self, x, y, s):
        """Noiseless output given current inputs (x, y) and previous state s."""
        x_prev, y_prev = self._decode_state(s)
        return ((1 - 2 * x) + (1 - 2 * y)
                + self.h * ((1 - 2 * x_prev) + (1 - 2 * y_prev)))

    def transition_prob(self, z, x: int, y: int, state: int = 0) -> float:
        """Return W(z | x, y, state) — Gaussian PDF value."""
        mu = self._mu(x, y, state)
        return float(np.exp(self._log_norm - (z - mu) ** 2 / (2 * self.sigma2)))

    def sample(self, x: int, y: int, state: int = 0):
        """Sample one output, return (z, new_state)."""
        mu = self._mu(x, y, state)
        z = float(mu + np.random.normal(0, np.sqrt(self.sigma2)))
        new_state = self._encode_state(x, y)
        return z, new_state

    def sample_batch(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Vectorised batch sampling with sequential state tracking.

        Parameters
        ----------
        X : ndarray (batch, N) or (N,) — user 1 bits
        Y : ndarray (batch, N) or (N,) — user 2 bits

        Returns
        -------
        Z : ndarray, same shape — continuous channel outputs
        """
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)

        if X.ndim == 1:
            X = X[None, :]
            Y = Y[None, :]
            squeeze = True
        else:
            squeeze = False

        batch, N = X.shape
        sx = (1 - 2 * X).astype(np.float64)  # BPSK
        sy = (1 - 2 * Y).astype(np.float64)

        # Previous symbols (initial state = 0 → x_prev=0, y_prev=0 → BPSK = +1)
        sx_prev = np.concatenate([np.ones((batch, 1)), sx[:, :-1]], axis=1)
        sy_prev = np.concatenate([np.ones((batch, 1)), sy[:, :-1]], axis=1)

        mu = sx + sy + self.h * (sx_prev + sy_prev)
        W = np.random.normal(0, np.sqrt(self.sigma2), size=mu.shape)
        Z = mu + W

        if squeeze:
            return Z[0]
        return Z

    def build_leaf_tensors(self, z_seq) -> np.ndarray:
        """
        Build (N, 2, 2, |S|, |S|) leaf log-probability tensors.

        Parameters
        ----------
        z_seq : array-like, length N — observed channel outputs

        Returns
        -------
        log_W : ndarray (N, 2, 2, |S|, |S|)
            log_W[t, x, y, s, s'] = log P(z_t | x, y, s) if s' == encode(x,y)
                                   = -inf otherwise
            For t=0: log_W[0, x, y, s, s'] = -inf for s != 0
                     (initial state is known to be 0)
        """
        z_arr = np.asarray(z_seq, dtype=np.float64)
        N = len(z_arr)
        S = self.num_states
        log_W = np.full((N, 2, 2, S, S), -np.inf, dtype=np.float64)

        for x in range(2):
            for y in range(2):
                s_next = self._encode_state(x, y)
                for s in range(S):
                    mu = self._mu(x, y, s)
                    # log N(z; mu, sigma2)
                    log_p = self._log_norm - (z_arr - mu) ** 2 / (2 * self.sigma2)
                    log_W[:, x, y, s, s_next] = log_p

        # Initial state constraint: s_in = 0 for t=0
        log_W[0, :, :, 1:, :] = -np.inf

        return log_W


if __name__ == "__main__":
    print("=== ISI-MAC ===")
    ch = ISIMAC.from_snr_db(6, h=0.5)
    print(f"SNR={ch.snr_db:.1f}dB, sigma2={ch.sigma2:.4f}, h={ch.h}")

    # Test single sample
    X = np.array([[0, 1, 0, 1, 0, 0, 1, 1]])
    Y = np.array([[0, 0, 1, 1, 0, 1, 0, 1]])
    Z = ch.sample_batch(X, Y)
    print(f"X = {X[0]}")
    print(f"Y = {Y[0]}")
    print(f"Z = {Z[0].round(3)}")

    # Test leaf tensor building
    log_W = ch.build_leaf_tensors(Z[0])
    print(f"\nLeaf tensor shape: {log_W.shape}")
    print(f"log_W[0, 0, 0, 0, 0] = {log_W[0, 0, 0, 0, 0]:.4f}")  # t=0, x=0, y=0, s=0, s'=0
    print(f"log_W[0, 0, 0, 1, 0] = {log_W[0, 0, 0, 1, 0]:.4f}")  # t=0, x=0, y=0, s=1 → -inf (initial state)
    print(f"Non -inf entries per leaf: {np.sum(np.isfinite(log_W[0]))}, {np.sum(np.isfinite(log_W[1]))}")
