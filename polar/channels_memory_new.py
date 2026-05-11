"""
channels_memory_new.py
======================
Additional MAC channels with memory for two-user binary-input polar codes.

Implements three channels inspired by the NPD paper (Aharoni et al.):

1. ISI-MAC (already in channels_memory.py, re-exported here for convenience)
2. Trapdoor MAC — binary channel with memory, state-dependent output
3. Gilbert-Elliott MAC — bursty noise channel with good/bad states

All channels have:
  - sample_batch(X, Y) -> Z  (vectorized batch sampling)
  - build_leaf_tensors(z) -> (N, 2, 2, S, S) log-prob tensors for SC trellis decoder
  - num_states property

State convention: s = state_index (integer), deterministic transitions.
"""
import numpy as np
from polar.channels import MACChannel


class TrapdoorMAC(MACChannel):
    """
    Trapdoor MAC — binary-input, binary-output channel with memory.

    MAC model (two-user extension of the trapdoor channel):
        S[i] = X[i] XOR Y[i] XOR S[i-1]  (state update)
        Z[i] = S[i] XOR N[i]              (output = state + noise)

    where N[i] ~ Bernoulli(p_noise) is the noise.

    The key property: the output depends on the XOR of BOTH users' current
    inputs AND the previous state. This creates memory + multi-user coupling.

    With p_noise=0: Z[i] = S[i] = X[i] XOR Y[i] XOR S[i-1] (deterministic)
    With p_noise>0: noisy observations of the state.

    States: S in {0, 1}, so num_states = 2.

    Parameters
    ----------
    p_noise : float — BSC crossover probability for the output (default 0.1)
    """
    name = "trapdoor_mac"

    def __init__(self, p_noise=0.1):
        self.p_noise = float(p_noise)
        self.num_states = 2

    def sample_batch(self, X, Y):
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        if X.ndim == 1:
            X, Y = X[None, :], Y[None, :]
            squeeze = True
        else:
            squeeze = False

        batch, N = X.shape
        Z = np.zeros((batch, N), dtype=np.int32)
        S = np.zeros(batch, dtype=np.int32)  # initial state = 0

        for i in range(N):
            S = X[:, i] ^ Y[:, i] ^ S  # state update
            noise = (np.random.random(batch) < self.p_noise).astype(np.int32)
            Z[:, i] = S ^ noise

        if squeeze:
            return Z[0].astype(np.float64)
        return Z.astype(np.float64)

    def transition_prob(self, z, x, y, state=0):
        s_next = x ^ y ^ state
        if int(z) == s_next:
            return 1 - self.p_noise
        else:
            return self.p_noise

    def build_leaf_tensors(self, z_seq):
        z = np.asarray(z_seq, dtype=np.int32)
        N = len(z)
        S = self.num_states
        log_W = np.full((N, 2, 2, S, S), -np.inf, dtype=np.float64)
        lp_good = np.log(max(1 - self.p_noise, 1e-15))
        lp_bad = np.log(max(self.p_noise, 1e-15))

        for x in range(2):
            for y in range(2):
                for s in range(S):
                    s_next = x ^ y ^ s
                    for t in range(N):
                        if int(z[t]) == s_next:
                            log_W[t, x, y, s, s_next] = lp_good
                        else:
                            log_W[t, x, y, s, s_next] = lp_bad

        # Initial state = 0
        log_W[0, :, :, 1:, :] = -np.inf
        return log_W


class GilbertElliottMAC(MACChannel):
    """
    Gilbert-Elliott MAC — bursty error channel with good/bad states.

    Two-user MAC extension of the Gilbert-Elliott channel:
        State S[i] ∈ {GOOD, BAD}, Markov transitions:
            P(GOOD -> BAD)  = p_gb
            P(BAD  -> GOOD) = p_bg
        In GOOD state: Z[i] = X[i] + Y[i] + N_good[i],  N ~ N(0, sigma2_good)
        In BAD  state: Z[i] = X[i] + Y[i] + N_bad[i],   N ~ N(0, sigma2_bad)

    where X,Y are BPSK: x_bpsk = 1-2x.

    Parameters
    ----------
    p_gb : float — transition prob GOOD -> BAD (default 0.05)
    p_bg : float — transition prob BAD -> GOOD (default 0.3)
    sigma2_good : float — noise variance in GOOD state (default 0.1)
    sigma2_bad  : float — noise variance in BAD state (default 2.0)
    """
    name = "ge_mac"

    def __init__(self, p_gb=0.05, p_bg=0.3, sigma2_good=0.1, sigma2_bad=2.0):
        self.p_gb = float(p_gb)
        self.p_bg = float(p_bg)
        self.sigma2_good = float(sigma2_good)
        self.sigma2_bad = float(sigma2_bad)
        # States: 0=GOOD, 1=BAD — but for the trellis we need to track
        # both the channel state AND the previous inputs.
        # For a MAC with state: combined state = (channel_state, x_prev, y_prev)
        # But to keep it simple: state = channel_state only (2 states)
        # The channel IS memoryless conditioned on state, and state transitions
        # don't depend on inputs.
        self.num_states = 2
        self._log_norm_good = -0.5 * np.log(2 * np.pi * self.sigma2_good)
        self._log_norm_bad = -0.5 * np.log(2 * np.pi * self.sigma2_bad)

    def sample_batch(self, X, Y):
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        if X.ndim == 1:
            X, Y = X[None, :], Y[None, :]
            squeeze = True
        else:
            squeeze = False

        batch, N = X.shape
        sx = (1 - 2 * X).astype(np.float64)
        sy = (1 - 2 * Y).astype(np.float64)

        Z = np.zeros((batch, N), dtype=np.float64)
        state = np.zeros(batch, dtype=np.int32)  # start in GOOD state

        for i in range(N):
            # Sample noise based on state
            sigma = np.where(state == 0,
                             np.sqrt(self.sigma2_good),
                             np.sqrt(self.sigma2_bad))
            Z[:, i] = sx[:, i] + sy[:, i] + np.random.normal(0, 1, batch) * sigma

            # State transition
            rand = np.random.random(batch)
            new_state = state.copy()
            # GOOD -> BAD
            mask_good = (state == 0)
            new_state[mask_good & (rand < self.p_gb)] = 1
            # BAD -> GOOD
            mask_bad = (state == 1)
            new_state[mask_bad & (rand < self.p_bg)] = 0
            state = new_state

        if squeeze:
            return Z[0]
        return Z

    def transition_prob(self, z, x, y, state=0):
        mu = (1 - 2 * x) + (1 - 2 * y)
        if state == 0:
            sigma2 = self.sigma2_good
            log_norm = self._log_norm_good
        else:
            sigma2 = self.sigma2_bad
            log_norm = self._log_norm_bad
        return float(np.exp(log_norm - (z - mu) ** 2 / (2 * sigma2)))

    def build_leaf_tensors(self, z_seq):
        z = np.asarray(z_seq, dtype=np.float64)
        N = len(z)
        S = self.num_states
        # State transitions: P(s'|s) — independent of x,y
        log_trans = np.full((S, S), -np.inf, dtype=np.float64)
        log_trans[0, 0] = np.log(max(1 - self.p_gb, 1e-15))
        log_trans[0, 1] = np.log(max(self.p_gb, 1e-15))
        log_trans[1, 0] = np.log(max(self.p_bg, 1e-15))
        log_trans[1, 1] = np.log(max(1 - self.p_bg, 1e-15))

        log_W = np.full((N, 2, 2, S, S), -np.inf, dtype=np.float64)

        for x in range(2):
            for y in range(2):
                mu = (1 - 2 * x) + (1 - 2 * y)
                for s in range(S):
                    if s == 0:
                        log_lik = self._log_norm_good - (z - mu) ** 2 / (2 * self.sigma2_good)
                    else:
                        log_lik = self._log_norm_bad - (z - mu) ** 2 / (2 * self.sigma2_bad)
                    for s_next in range(S):
                        log_W[:, x, y, s, s_next] = log_lik + log_trans[s, s_next]

        # Initial state: GOOD (state=0)
        log_W[0, :, :, 1, :] = -np.inf
        return log_W


class ISIMAC2(MACChannel):
    """
    Two-tap ISI MAC — extends ISIMAC with a second tap.

    Channel model:
        Z[i] = s_x[i] + s_y[i] + h1*(s_x[i-1] + s_y[i-1]) + h2*(s_x[i-2] + s_y[i-2]) + W[i]

    where s_x = 1-2X (BPSK), W ~ N(0, sigma2).

    State: (X[i-1], Y[i-1], X[i-2], Y[i-2]) — 16 states.

    Parameters
    ----------
    sigma2 : float — noise variance
    h1 : float — first ISI tap (default 0.4)
    h2 : float — second ISI tap (default 0.2)
    """
    name = "isi2_mac"

    def __init__(self, sigma2=0.5, h1=0.4, h2=0.2):
        self.sigma2 = float(sigma2)
        self.h1 = float(h1)
        self.h2 = float(h2)
        self.num_states = 16  # (x_{-1}, y_{-1}, x_{-2}, y_{-2})
        self._log_norm = -0.5 * np.log(2 * np.pi * self.sigma2)

    @staticmethod
    def _decode_state(s):
        """State s -> (x1, y1, x2, y2) where 1=prev, 2=prev-prev."""
        x1 = (s >> 3) & 1
        y1 = (s >> 2) & 1
        x2 = (s >> 1) & 1
        y2 = s & 1
        return x1, y1, x2, y2

    @staticmethod
    def _encode_state(x1, y1, x2, y2):
        return (x1 << 3) | (y1 << 2) | (x2 << 1) | y2

    def _mu(self, x, y, state):
        x1, y1, x2, y2 = self._decode_state(state)
        return ((1-2*x) + (1-2*y)
                + self.h1 * ((1-2*x1) + (1-2*y1))
                + self.h2 * ((1-2*x2) + (1-2*y2)))

    def sample_batch(self, X, Y):
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        if X.ndim == 1:
            X, Y = X[None, :], Y[None, :]
            squeeze = True
        else:
            squeeze = False

        batch, N = X.shape
        sx = (1-2*X).astype(np.float64)
        sy = (1-2*Y).astype(np.float64)

        # Pad with initial state (0,0) -> BPSK = (+1,+1)
        sx_1 = np.concatenate([np.ones((batch, 1)), sx[:, :-1]], axis=1)
        sy_1 = np.concatenate([np.ones((batch, 1)), sy[:, :-1]], axis=1)
        sx_2 = np.concatenate([np.ones((batch, 2)), sx[:, :-2]], axis=1)
        sy_2 = np.concatenate([np.ones((batch, 2)), sy[:, :-2]], axis=1)

        mu = sx + sy + self.h1 * (sx_1 + sy_1) + self.h2 * (sx_2 + sy_2)
        Z = mu + np.random.normal(0, np.sqrt(self.sigma2), size=mu.shape)

        if squeeze:
            return Z[0]
        return Z

    def transition_prob(self, z, x, y, state=0):
        mu = self._mu(x, y, state)
        return float(np.exp(self._log_norm - (z - mu) ** 2 / (2 * self.sigma2)))

    def build_leaf_tensors(self, z_seq):
        z = np.asarray(z_seq, dtype=np.float64)
        N = len(z)
        S = self.num_states
        log_W = np.full((N, 2, 2, S, S), -np.inf, dtype=np.float64)

        for x in range(2):
            for y in range(2):
                for s in range(S):
                    x1, y1, x2, y2 = self._decode_state(s)
                    # New state: (x, y, x1, y1)
                    s_next = self._encode_state(x, y, x1, y1)
                    mu = self._mu(x, y, s)
                    log_p = self._log_norm - (z - mu) ** 2 / (2 * self.sigma2)
                    log_W[:, x, y, s, s_next] = log_p

        # Initial state: s=0 -> (x1=0,y1=0,x2=0,y2=0)
        log_W[0, :, :, 1:, :] = -np.inf
        return log_W


class MAAGNMAC(MACChannel):
    """
    Moving-Average Additive Gaussian Noise MAC (MA-AGN MAC).

    Channel model (two-user):
        Z[i] = (1-2X[i]) + (1-2Y[i]) + N[i]
        N[i] = alpha * N[i-1] + W[i]

    Where W[i] ~ N(0, sigma2 * (1 - alpha^2)) for the STATIONARY AR(1) form
    (so Var[N[i]] = sigma2 for all i) and N[0] ~ N(0, sigma2).

    The noise process is AR(1) with continuous state. Unlike the ISI-MAC or
    Gilbert-Elliott MAC, the state is a REAL number (N[i-1]), so no finite-
    state trellis applies. Hence there is NO analytical trellis SC decoder
    for this channel — the memoryless GMAC SC is the practical baseline.

    This is the "flagship" memory-channel case from Aharoni et al. 2024
    (NPD paper Sec. VI-B): only a neural decoder can learn the memory
    structure from samples.

    Parameters
    ----------
    sigma2 : float — stationary noise variance σ² (per-time-step)
    alpha  : float — AR(1) coefficient, typically in [0.2, 0.5]

    Notes
    -----
    `build_leaf_tensors` and `transition_prob` are NOT implemented because
    the state is continuous (infinite state space).
    """
    name = "maagn_mac"

    def __init__(self, sigma2=0.25, alpha=0.3):
        self.sigma2 = float(sigma2)
        self.alpha = float(alpha)
        # No discrete num_states — continuous memory.
        self.num_states = None

    @classmethod
    def from_snr_db(cls, snr_db: float, alpha: float = 0.3):
        """Create an MAAGNMAC from per-user SNR in dB (stationary noise var)."""
        sigma2 = 10.0 ** (-snr_db / 10.0)
        return cls(sigma2=sigma2, alpha=alpha)

    @property
    def snr_db(self):
        return -10.0 * np.log10(self.sigma2)

    def sample_batch(self, X, Y):
        """
        Vectorised batch sampling for MA-AGN MAC.

        X, Y : (B, N) int arrays of bits
        Returns
        -------
        Z : (B, N) float array
        """
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)
        if X.ndim == 1:
            X, Y = X[None, :], Y[None, :]
            squeeze = True
        else:
            squeeze = False

        B, N = X.shape
        sx = (1 - 2 * X).astype(np.float64)
        sy = (1 - 2 * Y).astype(np.float64)

        # Stationary AR(1) noise:
        #   N[0] ~ N(0, sigma2)
        #   N[i] = alpha * N[i-1] + W[i],  W[i] ~ N(0, sigma2 * (1 - alpha^2))
        # So that Var[N[i]] = sigma2 for all i (stationary).
        z_noise = np.zeros((B, N), dtype=np.float64)
        z_noise[:, 0] = np.random.normal(0.0, np.sqrt(self.sigma2), size=B)
        innov_std = np.sqrt(self.sigma2 * (1.0 - self.alpha ** 2))
        for i in range(1, N):
            w = np.random.normal(0.0, innov_std, size=B)
            z_noise[:, i] = self.alpha * z_noise[:, i - 1] + w

        Z = sx + sy + z_noise

        if squeeze:
            return Z[0]
        return Z


if __name__ == "__main__":
    # Quick tests
    for name, ch in [("TrapdoorMAC(0.1)", TrapdoorMAC(0.1)),
                      ("GilbertElliottMAC", GilbertElliottMAC()),
                      ("ISIMAC2", ISIMAC2())]:
        X = np.array([[0, 1, 0, 1, 0, 0, 1, 1]])
        Y = np.array([[0, 0, 1, 1, 0, 1, 0, 1]])
        Z = ch.sample_batch(X, Y)
        log_W = ch.build_leaf_tensors(Z[0])
        print(f"{name}: Z shape={Z.shape}, leaf shape={log_W.shape}, "
              f"finite entries={np.isfinite(log_W[0]).sum()}")

    # MA-AGN has no leaf tensor (continuous state)
    ch = MAAGNMAC(sigma2=0.25, alpha=0.3)
    X = np.random.randint(0, 2, size=(4, 32))
    Y = np.random.randint(0, 2, size=(4, 32))
    Z = ch.sample_batch(X, Y)
    print(f"MAAGNMAC(sigma2=0.25,alpha=0.3): Z shape={Z.shape}, "
          f"mean={Z.mean():.3f}, std={Z.std():.3f}")



class IsingMAC:
    """
    Ising MAC channel. Two states: good (S=0) and bad (S=1).
    Good state: Z = (1-2X) + (1-2Y) + W (normal GMAC)
    Bad state: Z = W (pure noise, no signal)
    State transitions: Markov with flip probability p_flip.
    """
    def __init__(self, sigma2=0.251, p_flip=0.1):
        self.sigma2 = sigma2
        self.p_flip = p_flip
        self.name = "ising_mac"
        self.num_states = 2
        self._log_norm = -0.5 * np.log(2 * np.pi * self.sigma2)

    def sample_batch(self, x, y):
        B, N = x.shape if len(x.shape) == 2 else (1, len(x))
        if len(x.shape) == 1:
            x = x.reshape(1, -1); y = y.reshape(1, -1)

        z = np.zeros((B, N), dtype=np.float64)
        sx = 1 - 2 * x.astype(np.float64)
        sy = 1 - 2 * y.astype(np.float64)
        w = np.random.normal(0, np.sqrt(self.sigma2), (B, N))

        # State sequence
        state = np.zeros((B, N), dtype=int)
        state[:, 0] = 0  # start in good state
        for i in range(1, N):
            flip = np.random.random(B) < self.p_flip
            state[:, i] = np.where(flip, 1 - state[:, i-1], state[:, i-1])

        # Channel output
        good = state == 0
        z = np.where(good, sx + sy + w, w)

        if B == 1 and len(x.shape) == 1:
            return z[0]
        return z.astype(np.float32)

    def transition_prob(self, z, x, y, state=0):
        """P(z|x,y,state) -- emission probability only (no transition)."""
        if state == 0:  # GOOD
            mu = (1 - 2 * x) + (1 - 2 * y)
        else:  # BAD
            mu = 0.0
        return float(np.exp(self._log_norm - (z - mu) ** 2 / (2 * self.sigma2)))

    def build_leaf_tensors(self, z_seq):
        """
        Build (N, 2, 2, S, S) log-prob tensors for trellis SC decoder.

        log_W[t, x, y, s, s'] = log P(z_t | x, y, s) + log P(s' | s)

        State transitions are Markov (independent of x, y):
            P(s'=s | s) = 1 - p_flip
            P(s'!=s | s) = p_flip
        """
        z = np.asarray(z_seq, dtype=np.float64)
        N = len(z)
        S = self.num_states

        # Log transition probabilities
        log_stay = np.log(max(1 - self.p_flip, 1e-15))
        log_flip = np.log(max(self.p_flip, 1e-15))

        log_W = np.full((N, 2, 2, S, S), -np.inf, dtype=np.float64)

        for x in range(2):
            for y in range(2):
                for s in range(S):
                    if s == 0:  # GOOD state: signal + noise
                        mu = (1 - 2 * x) + (1 - 2 * y)
                    else:  # BAD state: pure noise
                        mu = 0.0
                    log_lik = self._log_norm - (z - mu) ** 2 / (2 * self.sigma2)

                    for s_next in range(S):
                        log_trans = log_stay if s_next == s else log_flip
                        log_W[:, x, y, s, s_next] = log_lik + log_trans

        # Initial state: GOOD (state=0)
        log_W[0, :, :, 1, :] = -np.inf
        return log_W
