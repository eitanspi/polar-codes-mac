"""
MAC channel abstraction for Class C decomposition.

Each MAC channel must expose two "single-user views":

  1. Stage 1 (marginal): input = raw z from the 2-user MAC, output = features
     for decoding U when V is uniformly random interference.

  2. Stage 2 (conditional): input = raw z and true/decoded X̂, output = features
     for decoding V when X is known (so interference is subtracted).

For most memoryless channels, both stages just return z in some form. The
interesting differences are:
  - Stage 1 may want LLR-like precomputed features that capture the mixture
    structure (optional — the NPD's z_encoder MLP can also learn this from raw z).
  - Stage 2 typically processes z' = z - contribution(X̂) to get a clean channel.
  - Memory channels (ISI) need a window of adjacent z values per position.

This module defines the base interface and provides concrete implementations
for GMAC, BEMAC, ABNMAC, and ISI-MAC.

Reuses:
  - `polar.channels.GaussianMAC`, `BEMAC`, etc. for the 2-user forward model
  - `polar.channels_memory.ISIMAC` for the ISI channel
"""
from __future__ import annotations
import math
import numpy as np
from abc import ABC, abstractmethod


# ─── Base interface ───────────────────────────────────────────────────────────

class MACChannelC(ABC):
    """
    Base class for a MAC channel configured for Class C (two-stage) decoding.

    Subclasses implement:
      - sample_z(x, y)                        — forward 2-user channel
      - stage1_features(z)                    — features for U decoding (phase 1)
      - stage2_features(z, x_hat)             — features for V decoding (phase 2)
      - stage1_feature_dim                    — per-position z_dim for phase 1
      - stage2_feature_dim                    — per-position z_dim for phase 2

    Optionally:
      - stage1_analytical_llr(z)              — if analytical SC reference is wanted
      - stage2_analytical_llr(z, x_hat)       — same for phase 2
    """

    name: str = "base"

    @abstractmethod
    def sample_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Forward the 2-user MAC. x, y: (B, N) binary. Returns z: (B, N) or (B, N, ?).
        """
        ...

    @abstractmethod
    def stage1_features(self, z: np.ndarray) -> np.ndarray:
        """
        Per-position features for decoding U (Stage 1, marginal channel).

        Input z:  shape from sample_z
        Returns:  (B, N) or (B, N, stage1_feature_dim)
        """
        ...

    @abstractmethod
    def stage2_features(self, z: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """
        Per-position features for decoding V (Stage 2, conditional on X known).

        Input z:     shape from sample_z
        Input x_hat: (B, N) binary, the decoded U codeword
        Returns:     (B, N) or (B, N, stage2_feature_dim)
        """
        ...

    @property
    @abstractmethod
    def stage1_feature_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def stage2_feature_dim(self) -> int:
        ...


# ─── Concrete: Gaussian MAC ───────────────────────────────────────────────────

class GaussianMAC_C(MACChannelC):
    """
    Class C view of the standard GaussianMAC: Z = (1 - 2X) + (1 - 2Y) + W, W ~ N(0, sigma2).

    Stage 1: raw z is fed directly. The z_encoder MLP learns the mixture LLR.
    Stage 2: z' = z - (1 - 2*x_hat) = (1 - 2Y) + W — clean BPSK+AWGN for V.
    """
    name = "gmac"
    stage1_feature_dim = 1
    stage2_feature_dim = 1

    def __init__(self, sigma2: float):
        self.sigma2 = float(sigma2)
        self.sigma = math.sqrt(self.sigma2)

    def sample_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bx = 1.0 - 2.0 * x.astype(np.float64)
        by = 1.0 - 2.0 * y.astype(np.float64)
        w = np.random.normal(0.0, self.sigma, x.shape)
        return (bx + by + w).astype(np.float32)

    def stage1_features(self, z: np.ndarray) -> np.ndarray:
        # raw z — z_encoder will learn the mixture LLR internally
        return z.astype(np.float32)

    def stage2_features(self, z: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        # Subtract U's contribution to get clean BPSK+AWGN for V
        bx_hat = 1.0 - 2.0 * x_hat.astype(np.float64)
        z_prime = z.astype(np.float64) - bx_hat
        return z_prime.astype(np.float32)

    # Analytical LLRs — optional, useful for SC reference and NPD debugging
    def stage1_analytical_llr(self, z: np.ndarray) -> np.ndarray:
        """LLR(x | z) for the mixture channel, treating Y as uniform random."""
        z = z.astype(np.float64)
        s2 = self.sigma2
        log_N = lambda m: -0.5 * (z - m) ** 2 / s2
        log_p0 = np.logaddexp(log_N(+2.0), log_N(0.0))  # p(z|x=0) up to const
        log_p1 = np.logaddexp(log_N(0.0), log_N(-2.0))  # p(z|x=1)
        return (log_p0 - log_p1).astype(np.float32)

    def stage2_analytical_llr(self, z: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        """LLR(y | z, x_hat) = 2 * z' / sigma2 where z' = z - (1-2*x_hat)."""
        z_prime = self.stage2_features(z, x_hat).astype(np.float64)
        return (2.0 * z_prime / self.sigma2).astype(np.float32)


# ─── Concrete: Binary Erasure MAC ─────────────────────────────────────────────
# (BEMAC is discrete but we can still use the same two-stage decomposition.
#  The z_encoder handles discrete inputs via an nn.Embedding-like path.)

class BEMAC_C(MACChannelC):
    """
    BEMAC: Z = X + Y in {0, 1, 2}. Discrete output.

    Stage 1: z has 3 possible values. z_encoder sees z as int (or float), learns mixture LLR.
    Stage 2: conditional on x_hat, Z - X is the binary value Y (no noise).
             But wait — BEMAC has no noise at all. This means once we know X,
             Y is DETERMINED by z - x_hat. No Stage 2 neural decoding is needed
             in principle. We still run the pipeline to verify correctness.
    """
    name = "bemac"
    stage1_feature_dim = 1
    stage2_feature_dim = 1

    def sample_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x + y).astype(np.float32)

    def stage1_features(self, z: np.ndarray) -> np.ndarray:
        # Center so 0, 1, 2 become -1, 0, 1 for better MLP input
        return (z.astype(np.float32) - 1.0)

    def stage2_features(self, z: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        # y_true = z - x_hat, clipped to {0, 1}. If x_hat is wrong, we still feed the result.
        y_obs = (z - x_hat.astype(np.float32))
        return y_obs - 0.5  # center around 0


# ─── Concrete: Asymmetric Binary Noisy MAC ────────────────────────────────────

class ABNMAC_C(MACChannelC):
    """
    ABNMAC: Z = (X ⊕ E_x, Y ⊕ E_y), independent binary errors per user.

    Output is a 2-tuple per position. Stage 1 feature = first component.
    Stage 2 feature = second component (already clean for V).

    Note: ABNMAC is a special case where the two users are independently
    corrupted. Stage 1 and Stage 2 are effectively independent single-user
    binary symmetric channels. Class C decomposition is natural but trivial.
    """
    name = "abnmac"
    stage1_feature_dim = 1
    stage2_feature_dim = 1

    def __init__(self, p_x: float, p_y: float):
        self.p_x = float(p_x)
        self.p_y = float(p_y)

    def sample_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ex = (np.random.rand(*x.shape) < self.p_x).astype(x.dtype)
        ey = (np.random.rand(*y.shape) < self.p_y).astype(y.dtype)
        zx = (x ^ ex)
        zy = (y ^ ey)
        # Pack as shape (B, N, 2)
        return np.stack([zx, zy], axis=-1).astype(np.float32)

    def stage1_features(self, z: np.ndarray) -> np.ndarray:
        # Only first component matters for U
        return (z[..., 0] - 0.5).astype(np.float32)

    def stage2_features(self, z: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        # Only second component matters for V; x_hat is not needed because
        # ABNMAC outputs are independent per user.
        return (z[..., 1] - 0.5).astype(np.float32)


# ─── Concrete: ISI Gaussian MAC ───────────────────────────────────────────────

class ISIMAC_C(MACChannelC):
    """
    ISI Gaussian MAC: Z[i] = (1-2X[i]) + (1-2Y[i]) + h*((1-2X[i-1]) + (1-2Y[i-1])) + W[i]

    For Stage 1 with no Y info, we use a WINDOW of z values as features, so the
    NPD can see ISI context and learn to handle the memory.

    For Stage 2 given X̂, we subtract the known U contribution (including its
    delayed tap) and feed the windowed residual.
    """
    name = "isi_mac"
    stage1_feature_dim = 3  # window: [z[i-1], z[i], z[i+1]]
    stage2_feature_dim = 3

    def __init__(self, sigma2: float, h: float = 0.5):
        self.sigma2 = float(sigma2)
        self.sigma = math.sqrt(self.sigma2)
        self.h = float(h)

    def sample_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bx = 1.0 - 2.0 * x.astype(np.float64)
        by = 1.0 - 2.0 * y.astype(np.float64)
        # bx[..., -1] pad with zeros for initial state
        bx_prev = np.concatenate([np.zeros(bx.shape[:-1] + (1,)), bx[..., :-1]], axis=-1)
        by_prev = np.concatenate([np.zeros(by.shape[:-1] + (1,)), by[..., :-1]], axis=-1)
        w = np.random.normal(0.0, self.sigma, x.shape)
        z = bx + by + self.h * (bx_prev + by_prev) + w
        return z.astype(np.float32)

    def _window(self, z: np.ndarray) -> np.ndarray:
        """Return (B, N, 3) with [z[i-1], z[i], z[i+1]] per position, zero-padded."""
        zpad_left = np.concatenate([np.zeros(z.shape[:-1] + (1,), dtype=z.dtype), z[..., :-1]], axis=-1)
        zpad_right = np.concatenate([z[..., 1:], np.zeros(z.shape[:-1] + (1,), dtype=z.dtype)], axis=-1)
        return np.stack([zpad_left, z, zpad_right], axis=-1).astype(np.float32)

    def stage1_features(self, z: np.ndarray) -> np.ndarray:
        return self._window(z)

    def stage2_features(self, z: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
        # Subtract the known U contribution (including ISI tap)
        bx = 1.0 - 2.0 * x_hat.astype(np.float64)
        bx_prev = np.concatenate([np.zeros(bx.shape[:-1] + (1,)), bx[..., :-1]], axis=-1)
        u_contribution = bx + self.h * bx_prev
        z_prime = z.astype(np.float64) - u_contribution
        return self._window(z_prime.astype(np.float32))


# ─── Registry ─────────────────────────────────────────────────────────────────

def build_channel(name: str, **kwargs) -> MACChannelC:
    """Factory function."""
    name = name.lower()
    if name == "gmac" or name == "gaussian":
        return GaussianMAC_C(sigma2=kwargs.get("sigma2", 0.2512))
    if name == "bemac":
        return BEMAC_C()
    if name == "abnmac":
        return ABNMAC_C(p_x=kwargs.get("p_x", 0.1), p_y=kwargs.get("p_y", 0.1))
    if name == "isi_mac" or name == "isi":
        return ISIMAC_C(sigma2=kwargs.get("sigma2", 0.2512), h=kwargs.get("h", 0.5))
    raise ValueError(f"Unknown channel: {name}")
