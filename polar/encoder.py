"""
encoder.py
==========
Vectorized polar encoder for MAC codes.

Implements G_N = B_N · F^{⊗n} where:
  F = [[1, 0], [1, 1]]   (Arikan kernel)
  B_N = bit-reversal permutation matrix

The encoder operates on binary vectors u^N ∈ {0,1}^N and produces
codewords x^N = u^N · G_N.

Supports:
  - Single-vector encoding (numpy, O(N log N))
  - Batch encoding (numpy vectorized, O(batch × N log N))
  - Optional TensorFlow batch encoding for GPU acceleration
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Bit-reversal permutation
# ─────────────────────────────────────────────────────────────────────────────

def bit_reversal_perm(n: int) -> np.ndarray:
    """
    Compute the bit-reversal permutation for n-bit indices.

    br[k] = reverse_bits(k, n_bits=n)

    Returns np.ndarray of shape (2^n,), dtype int32.
    """
    N = 1 << n
    br = np.arange(N, dtype=np.int32)
    for i in range(N):
        rev = 0
        x = i
        for _ in range(n):
            rev = (rev << 1) | (x & 1)
            x >>= 1
        br[i] = rev
    return br


# Precomputed cache for repeated use
_BR_CACHE: dict = {}


def _get_br(n: int) -> np.ndarray:
    if n not in _BR_CACHE:
        _BR_CACHE[n] = bit_reversal_perm(n)
    return _BR_CACHE[n]


# ─────────────────────────────────────────────────────────────────────────────
#  Single-vector encoder
# ─────────────────────────────────────────────────────────────────────────────

def polar_encode(u) -> list:
    """
    Polar-encode u^N → x^N = u^N · G_N.

    Parameters
    ----------
    u : list or 1D array of int, length N (N must be a power of 2)

    Returns
    -------
    x : list of int, length N
    """
    N = len(u)
    n = N.bit_length() - 1
    br = _get_br(n)
    x = np.array(u, dtype=np.int32)[br].copy()  # bit-reversal
    step = 1
    while step < N:
        x_r = x.reshape(-1, 2 * step)
        x_r[:, :step] ^= x_r[:, step:]
        step *= 2
    return x.tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  Batched numpy encoder
# ─────────────────────────────────────────────────────────────────────────────

def polar_encode_batch(U: np.ndarray) -> np.ndarray:
    """
    Vectorized batch polar encoder.

    Parameters
    ----------
    U : np.ndarray of shape (batch, N), dtype int — input message vectors
        N must be a power of 2.

    Returns
    -------
    X : np.ndarray of shape (batch, N), dtype int32 — encoded codewords
    """
    U = np.asarray(U, dtype=np.int32)
    if U.ndim == 1:
        U = U[np.newaxis, :]
    batch, N = U.shape
    n = N.bit_length() - 1
    assert (1 << n) == N, f"N={N} must be a power of 2"

    br = _get_br(n)
    X = U[:, br].copy()  # shape (batch, N) — bit-reversal for all vectors

    # Butterfly XOR stages — fully vectorized over batch dimension
    step = 1
    while step < N:
        # Reshape to (batch, num_blocks, 2, step)
        X_r = X.reshape(batch, N // (2 * step), 2, step)
        X_r[:, :, 0, :] ^= X_r[:, :, 1, :]  # XOR left half with right half
        step *= 2

    return X


# ─────────────────────────────────────────────────────────────────────────────
#  TensorFlow batch encoder (optional — requires TF)
# ─────────────────────────────────────────────────────────────────────────────

def polar_encode_batch_tf(U, N: int):
    """
    TensorFlow-based batched polar encoder.

    Uses tf.bitwise.bitwise_xor for GPU-accelerated encoding.
    Requires TensorFlow ≥ 2.0 and static (known at trace time) block length N.

    Parameters
    ----------
    U : tf.Tensor of shape (batch, N), dtype tf.int32
    N : int — block length (must be known statically for tf.function tracing)

    Returns
    -------
    X : tf.Tensor of shape (batch, N), dtype tf.int32
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("TensorFlow is required for polar_encode_batch_tf. "
                          "Use polar_encode_batch (numpy) instead.") from e

    n = int(np.log2(N))
    assert (1 << n) == N, f"N={N} must be a power of 2"

    br = _get_br(n).tolist()

    # Bit-reversal permutation via gather
    X = tf.gather(U, br, axis=1)  # shape (batch, N)

    # Butterfly XOR stages
    step = 1
    while step < N:
        batch_size = tf.shape(U)[0]
        num_blocks = N // (2 * step)
        X_r = tf.reshape(X, [batch_size, num_blocks, 2, step])
        X0 = X_r[:, :, 0, :]
        X1 = X_r[:, :, 1, :]
        X_new = tf.stack([tf.bitwise.bitwise_xor(X0, X1), X1], axis=2)
        X = tf.reshape(X_new, [batch_size, N])
        step *= 2

    return X


# ─────────────────────────────────────────────────────────────────────────────
#  Build full message vector from info bits + frozen bits
# ─────────────────────────────────────────────────────────────────────────────

def build_message(N: int, info_bits, info_positions) -> np.ndarray:
    """
    Place info_bits at info_positions (1-indexed) in a zero-padded vector.

    Parameters
    ----------
    N             : int — block length
    info_bits     : array-like of int, length ku
    info_positions: list[int] — 1-indexed positions, length ku

    Returns
    -------
    u : np.ndarray of shape (N,), dtype int32
    """
    u = np.zeros(N, dtype=np.int32)
    for bit, pos in zip(info_bits, info_positions):
        u[pos - 1] = int(bit)
    return u


def build_message_batch(N: int, info_batch: np.ndarray, info_positions) -> np.ndarray:
    """
    Batch version: place info_batch[:, :] at info_positions in zero-padded messages.

    Parameters
    ----------
    N             : int
    info_batch    : np.ndarray of shape (batch, ku)
    info_positions: list[int] — 1-indexed, length ku

    Returns
    -------
    U : np.ndarray of shape (batch, N)
    """
    batch = info_batch.shape[0]
    U = np.zeros((batch, N), dtype=np.int32)
    for col, pos in enumerate(info_positions):
        U[:, pos - 1] = info_batch[:, col]
    return U


# ─────────────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Encoder self-test ===")

    # Compare single encoder vs batch encoder
    rng = np.random.default_rng(123)
    for N in [4, 8, 16, 64]:
        n = N.bit_length() - 1
        U = rng.integers(0, 2, size=(20, N), dtype=np.int32)
        X_batch = polar_encode_batch(U)
        # Verify against single encoder
        ok = all(
            polar_encode(U[i].tolist()) == X_batch[i].tolist()
            for i in range(20)
        )
        print(f"  N={N:4d}: batch vs single {'OK' if ok else 'FAIL'}")

    # Test idempotence: encode twice = identity (polar code property with G_N^2 = I mod 2)
    N = 16
    u = rng.integers(0, 2, size=N, dtype=np.int32)
    x = np.array(polar_encode(u.tolist()), dtype=np.int32)
    u_recovered = np.array(polar_encode(x.tolist()), dtype=np.int32)
    print(f"  N={N}: encode(encode(u))=u: {'OK' if np.all(u == u_recovered) else 'FAIL'}")

    # Timing
    import time
    N = 512
    U_big = rng.integers(0, 2, size=(1000, N), dtype=np.int32)
    t0 = time.time()
    _ = polar_encode_batch(U_big)
    t1 = time.time()
    print(f"  N={N}, 1000 codewords: {(t1-t0)*1000:.1f}ms "
          f"({(t1-t0)/1000*1e6:.1f}µs/codeword)")
