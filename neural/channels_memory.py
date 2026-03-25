"""
channels_memory.py — MAC Channels with Memory for neural decoder training.

Implements channels where the analytical SC decoder has complexity O(S³ N log N):
  - ISI_MAC:  Intersymbol Interference MAC with BPSK + AWGN
  - GE_MAC:   Gilbert-Elliott (2-state Markov) MAC

Our neural decoder handles these with O(md N log N) complexity because the
tree operations are completely channel-independent.
"""

import numpy as np


class ISI_MAC:
    """
    Intersymbol Interference MAC with BPSK modulation and AWGN noise.

    Channel model (2-tap ISI):
        z_t = s_x(t) + s_y(t) + alpha * [s_x(t-1) + s_y(t-1)] + w_t

    where:
        s_x(t) = 1 - 2*x_t  (BPSK)
        s_y(t) = 1 - 2*y_t  (BPSK)
        w_t ~ N(0, sigma2)
        alpha = ISI coefficient (memory strength)

    State space: (x_{t-1}, y_{t-1}) ∈ {0,1}² → S = 4 states
    Analytical SC complexity: O(S³ N log N) = O(64 N log N)
    Neural SC complexity:     O(md N log N) — independent of S

    Parameters
    ----------
    alpha  : ISI coefficient (0 = memoryless, 0.5 = strong memory)
    sigma2 : AWGN noise variance
    """
    name = "isi_mac"

    def __init__(self, alpha=0.3, sigma2=0.5):
        self.alpha = float(alpha)
        self.sigma2 = float(sigma2)

    @classmethod
    def from_snr_db(cls, snr_db, alpha=0.3):
        sigma2 = 10.0 ** (-snr_db / 10.0)
        return cls(alpha=alpha, sigma2=sigma2)

    def sample_batch(self, X, Y, rng=None):
        """
        Sample ISI MAC channel outputs.

        Parameters
        ----------
        X : (batch, N) int array — user 1 codeword bits
        Y : (batch, N) int array — user 2 codeword bits

        Returns
        -------
        Z : (batch, N) float array — channel outputs
        """
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        sx = (1 - 2 * X).astype(np.float64)
        sy = (1 - 2 * Y).astype(np.float64)

        # Current signal
        signal = sx + sy  # (batch, N)

        # ISI: add alpha * previous signal
        isi = np.zeros_like(signal)
        isi[:, 1:] = self.alpha * (sx[:, :-1] + sy[:, :-1])

        # Noise
        if rng is not None:
            noise = rng.normal(0, np.sqrt(self.sigma2), size=signal.shape)
        else:
            noise = np.random.normal(0, np.sqrt(self.sigma2), size=signal.shape)

        return signal + isi + noise

    def __repr__(self):
        return f"ISI_MAC(alpha={self.alpha}, sigma2={self.sigma2:.3f})"


class GilbertElliott_MAC:
    """
    Gilbert-Elliott (2-state Markov) MAC.

    Channel model:
        State S_t ∈ {Good, Bad}, Markov chain with transitions:
            P(Good→Bad) = p_gb,  P(Bad→Good) = p_bg

        In Good state: z_t = s_x(t) + s_y(t) + w_t,  w_t ~ N(0, sigma2_good)
        In Bad state:  z_t = s_x(t) + s_y(t) + w_t,  w_t ~ N(0, sigma2_bad)

    State space: S_t × (x_t, y_t) → effective S = 2 * 4 = 8 states
    (but Markov chain has memory → S grows with decoding horizon)

    Parameters
    ----------
    p_gb        : P(Good → Bad) transition probability
    p_bg        : P(Bad → Good) transition probability
    sigma2_good : noise variance in Good state
    sigma2_bad  : noise variance in Bad state
    """
    name = "ge_mac"

    def __init__(self, p_gb=0.1, p_bg=0.3, sigma2_good=0.1, sigma2_bad=2.0):
        self.p_gb = float(p_gb)
        self.p_bg = float(p_bg)
        self.sigma2_good = float(sigma2_good)
        self.sigma2_bad = float(sigma2_bad)

    def sample_batch(self, X, Y, rng=None):
        """
        Sample Gilbert-Elliott MAC channel outputs.

        Returns
        -------
        Z      : (batch, N) float — channel outputs
        states : (batch, N) int — channel states (0=Good, 1=Bad) for debugging
        """
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        batch, N = X.shape
        if rng is None:
            rng = np.random.default_rng()

        sx = (1 - 2 * X).astype(np.float64)
        sy = (1 - 2 * Y).astype(np.float64)

        # Generate Markov chain states
        states = np.zeros((batch, N), dtype=np.int32)
        # Start in steady state: P(Good) = p_bg / (p_gb + p_bg)
        p_good = self.p_bg / (self.p_gb + self.p_bg)
        states[:, 0] = (rng.random(batch) > p_good).astype(np.int32)

        for t in range(1, N):
            r = rng.random(batch)
            # Transition
            good_mask = states[:, t-1] == 0
            bad_mask = ~good_mask
            # Good → Bad with prob p_gb
            states[good_mask, t] = (r[good_mask] < self.p_gb).astype(np.int32)
            # Bad → Good with prob p_bg
            states[bad_mask, t] = (r[bad_mask] >= self.p_bg).astype(np.int32)

        # Generate noise based on state
        sigma = np.where(states == 0,
                         np.sqrt(self.sigma2_good),
                         np.sqrt(self.sigma2_bad))
        noise = rng.normal(0, 1, size=(batch, N)) * sigma

        Z = sx + sy + noise
        return Z, states

    def __repr__(self):
        return (f"GE_MAC(p_gb={self.p_gb}, p_bg={self.p_bg}, "
                f"σ²_good={self.sigma2_good}, σ²_bad={self.sigma2_bad})")


if __name__ == "__main__":
    # Quick test
    rng = np.random.default_rng(42)
    X = rng.integers(0, 2, (4, 8))
    Y = rng.integers(0, 2, (4, 8))

    print("=== ISI MAC ===")
    isi = ISI_MAC(alpha=0.3, sigma2=0.5)
    Z = isi.sample_batch(X, Y, rng)
    print(f"  {isi}")
    print(f"  Z shape: {Z.shape}, range: [{Z.min():.2f}, {Z.max():.2f}]")

    print("\n=== Gilbert-Elliott MAC ===")
    ge = GilbertElliott_MAC()
    Z, states = ge.sample_batch(X, Y, rng)
    print(f"  {ge}")
    print(f"  Z shape: {Z.shape}, range: [{Z.min():.2f}, {Z.max():.2f}]")
    print(f"  States: Good={np.mean(states==0):.2f}, Bad={np.mean(states==1):.2f}")
