"""
benchmark_parallel.py
=====================
Speed comparison between the sequential (NumPy) and parallel (Numba JIT)
SC decoders for Class B MAC polar codes on BE-MAC.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.design import design_bemac, make_path
from polar.channels import BEMAC
from polar.encoder import polar_encode, build_message
from polar.decoder import (
    build_log_W_leaf_batch,
    _decode_general_tensor_batch,
)
from polar.decoder_parallel import decode_parallel_batch


def benchmark():
    channel = BEMAC()
    rng = np.random.default_rng(789)
    rho = 0.7
    Ru_dir = 0.625
    Rv_dir = 0.875

    print("=" * 72)
    print("  Parallel SC Decoder Benchmark — Class B (path_i = N/2), BE-MAC")
    print("=" * 72)
    print()
    print(f"{'N':>6s}  {'#CW':>5s}  {'Sequential':>14s}  {'Parallel':>14s}  "
          f"{'Speedup':>8s}  {'Match':>5s}")
    print("-" * 72)

    for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        n = N.bit_length() - 1
        ku = max(1, min(round(rho * Ru_dir * N), N - 1))
        kv = max(1, min(round(rho * Rv_dir * N), N - 1))
        Au, Av, frozen_u, frozen_v, zu, zv = design_bemac(n, ku, kv)
        b = make_path(N, N // 2)

        num_cw = min(1000, max(20, 5000 // N))

        Z_list = []
        for _ in range(num_cw):
            u_msg = build_message(N, rng.integers(0, 2, size=ku).tolist(), Au)
            v_msg = build_message(N, rng.integers(0, 2, size=kv).tolist(), Av)
            x = np.array(polar_encode(u_msg.tolist()), dtype=np.int32)
            y = np.array(polar_encode(v_msg.tolist()), dtype=np.int32)
            Z_list.append((x + y).tolist())

        Z_arr = np.array(Z_list)
        log_W_batch = build_log_W_leaf_batch(Z_arr, channel)

        # Warmup
        _decode_general_tensor_batch(N, log_W_batch[:2], b, frozen_u, frozen_v)
        decode_parallel_batch(N, log_W_batch[:2], b, frozen_u, frozen_v)

        # Sequential (NumPy batched)
        t0 = time.perf_counter()
        u_seq, v_seq = _decode_general_tensor_batch(
            N, log_W_batch, b, frozen_u, frozen_v)
        t_seq = (time.perf_counter() - t0) / num_cw * 1000

        # Parallel (Numba JIT)
        t0 = time.perf_counter()
        u_par, v_par = decode_parallel_batch(
            N, log_W_batch, b, frozen_u, frozen_v)
        t_par = (time.perf_counter() - t0) / num_cw * 1000

        match = np.all(u_seq == u_par) and np.all(v_seq == v_par)
        speedup = t_seq / t_par if t_par > 0 else float('inf')

        print(f"{N:6d}  {num_cw:5d}  {t_seq:11.4f} ms  {t_par:11.4f} ms  "
              f"{speedup:7.2f}x  {'OK' if match else 'FAIL'}")

    print()
    print("Sequential = NumPy batched tensor decoder (_decode_general_tensor_batch)")
    print("Parallel   = Recursive level-parallel NumPy decoder (decode_parallel_batch)")


if __name__ == "__main__":
    benchmark()
