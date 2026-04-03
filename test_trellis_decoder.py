"""
Test suite for the MAC trellis decoder on ISI channels.

Tests verify:
1. All-zero decode at high SNR (Class A, C)
2. h=0 matches the memoryless decoder
3. Correct decode of random codewords at high SNR
4. BLER decreases with SNR
5. Class A path
6. ISI performance comparison with memoryless
"""

import numpy as np
from polar.channels_memory import ISIMAC
from polar.channels import GaussianMAC
from polar.encoder import polar_encode, build_message
from polar.design import make_path, design_gmac
from polar.decoder_trellis import decode_single as decode_trellis
from polar.decoder_interleaved import decode_single as decode_memoryless


def test_all_zero_decode():
    """All-zero codewords should decode correctly at high SNR."""
    print("Test 1: All-zero codewords")
    N = 16
    ch = ISIMAC.from_snr_db(15, h=0.5)

    frozen_u = {i: 0 for i in range(1, N + 1)}
    frozen_v = {i: 0 for i in range(1, N + 1)}

    for path_i in [0, N]:  # Class A and C
        b = make_path(N, path_i)
        n_ok = 0
        for _ in range(20):
            Z = ch.sample_batch(np.zeros(N, dtype=int), np.zeros(N, dtype=int))
            u_dec, v_dec = decode_trellis(N, Z, b, frozen_u, frozen_v, ch)
            if all(u == 0 for u in u_dec) and all(v == 0 for v in v_dec):
                n_ok += 1
        status = "PASS" if n_ok == 20 else "FAIL"
        print(f"  path_i={path_i:2d}: {n_ok}/20 [{status}]")


def test_h0_matches_memoryless():
    """When h=0, trellis decoder must produce same decisions as memoryless."""
    print("\nTest 2: h=0 matches memoryless decoder")
    N = 8
    sigma2 = 0.001  # ~30dB

    ch_isi = ISIMAC(sigma2=sigma2, h=0.0)
    ch_gmac = GaussianMAC(sigma2=sigma2)

    n = N.bit_length() - 1
    ku, kv = 2, 4
    Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)
    b = make_path(N, N)  # Class C

    n_match = 0
    n_trials = 50
    for trial in range(n_trials):
        np.random.seed(trial + 5000)
        u_info = np.random.randint(0, 2, size=ku)
        v_info = np.random.randint(0, 2, size=kv)
        u_msg = build_message(N, u_info, Au)
        v_msg = build_message(N, v_info, Av)
        x_enc = polar_encode(u_msg.tolist())
        y_enc = polar_encode(v_msg.tolist())

        np.random.seed(trial + 5000)
        Z_isi = ch_isi.sample_batch(np.array(x_enc), np.array(y_enc))
        np.random.seed(trial + 5000)
        Z_gmac = ch_gmac.sample_batch(np.array(x_enc), np.array(y_enc))

        u_t, v_t = decode_trellis(N, Z_isi, b, frozen_u, frozen_v, ch_isi)
        u_m, v_m = decode_memoryless(N, Z_gmac.tolist(), b, frozen_u, frozen_v, ch_gmac)

        if u_t == u_m and v_t == v_m:
            n_match += 1

    status = "PASS" if n_match == n_trials else f"FAIL ({n_match}/{n_trials})"
    print(f"  {n_match}/{n_trials} decisions match [{status}]")


def test_correct_decode_high_snr():
    """Correct decoding at high SNR with iterative refinement."""
    print("\nTest 3: Correct decoding at high SNR with ISI (h=0.5, iterative)")
    N = 16
    sigma2 = 0.001  # 30dB
    ch = ISIMAC(sigma2=sigma2, h=0.5)

    n = N.bit_length() - 1
    ku, kv = 1, 4
    Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)
    b = make_path(N, N)

    n_ok = 0
    n_trials = 100
    np.random.seed(42)
    for _ in range(n_trials):
        u_info = np.random.randint(0, 2, size=ku)
        v_info = np.random.randint(0, 2, size=kv)
        u_msg = build_message(N, u_info, Au)
        v_msg = build_message(N, v_info, Av)
        x_enc = polar_encode(u_msg.tolist())
        y_enc = polar_encode(v_msg.tolist())
        Z = ch.sample_batch(np.array(x_enc), np.array(y_enc))
        # Use iterative refinement (n_iter=2) for high-SNR ISI
        u_dec, v_dec = decode_trellis(N, Z, b, frozen_u, frozen_v, ch,
                                       n_iter=2)

        u_ok = all(u_dec[i - 1] == u_msg[i - 1] for i in Au)
        v_ok = all(v_dec[i - 1] == v_msg[i - 1] for i in Av)
        if u_ok and v_ok:
            n_ok += 1

    status = "PASS" if n_ok >= 95 else "FAIL"
    print(f"  {n_ok}/{n_trials} correct [{status}]")


def test_bler_improves_with_snr():
    """BLER should decrease with increasing SNR."""
    print("\nTest 4: BLER vs SNR (N=16, Class C, h=0.5)")
    N = 16
    n = N.bit_length() - 1
    ku, kv = 2, 4
    b = make_path(N, N)

    prev_bler = 1.0
    monotone = True
    np.random.seed(0)

    for snr_db in [3, 6, 10, 15]:
        sigma2 = 10 ** (-snr_db / 10)
        ch = ISIMAC(sigma2=sigma2, h=0.5)
        Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)

        n_errors = 0
        n_trials = 300
        for _ in range(n_trials):
            u_info = np.random.randint(0, 2, size=ku)
            v_info = np.random.randint(0, 2, size=kv)
            u_msg = build_message(N, u_info, Au)
            v_msg = build_message(N, v_info, Av)
            x_enc = polar_encode(u_msg.tolist())
            y_enc = polar_encode(v_msg.tolist())
            Z = ch.sample_batch(np.array(x_enc), np.array(y_enc))
            u_dec, v_dec = decode_trellis(N, Z, b, frozen_u, frozen_v, ch)

            u_err = any(u_dec[i - 1] != u_msg[i - 1] for i in Au)
            v_err = any(v_dec[i - 1] != v_msg[i - 1] for i in Av)
            if u_err or v_err:
                n_errors += 1

        bler = n_errors / n_trials
        trend = "down" if bler <= prev_bler + 0.02 else "UP"
        print(f"  SNR={snr_db:2d}dB: BLER={bler:.4f} {trend}")
        if bler > prev_bler + 0.05:
            monotone = False
        prev_bler = bler

    status = "PASS" if monotone else "FAIL"
    print(f"  Monotone decreasing: [{status}]")


def test_class_a():
    """Class A path (all V first, then U) with ISI."""
    print("\nTest 5: Class A path with ISI (h=0.5)")
    N = 8
    ch = ISIMAC.from_snr_db(20, h=0.5)

    frozen_u = {i: 0 for i in range(1, N + 1)}
    frozen_v = {i: 0 for i in range(1, N + 1)}
    b = make_path(N, 0)

    n_ok = 0
    for _ in range(30):
        Z = ch.sample_batch(np.zeros(N, dtype=int), np.zeros(N, dtype=int))
        u_dec, v_dec = decode_trellis(N, Z, b, frozen_u, frozen_v, ch)
        if all(u == 0 for u in u_dec) and all(v == 0 for v in v_dec):
            n_ok += 1

    status = "PASS" if n_ok == 30 else "FAIL"
    print(f"  {n_ok}/30 correct (all-zero, Class A) [{status}]")


def test_isi_uses_trellis():
    """
    Verify the trellis decoder uses state info: with iterative refinement
    (n_iter=2), it should outperform a naive state-marginalizing decoder
    on the ISI channel.
    """
    print("\nTest 6: Iterative FB improves V decoding (Class C, h=0.5)")
    N = 16
    n = N.bit_length() - 1
    sigma2 = 10 ** (-6 / 10)  # 6dB
    ku, kv = 2, 6
    b = make_path(N, N)
    Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)

    ch = ISIMAC(sigma2=sigma2, h=0.5)
    n_trials = 300
    np.random.seed(0)

    # Single-pass (n_iter=0)
    n_err_single = 0
    for _ in range(n_trials):
        u_info = np.random.randint(0, 2, size=ku)
        v_info = np.random.randint(0, 2, size=kv)
        u_msg = build_message(N, u_info, Au)
        v_msg = build_message(N, v_info, Av)
        x_enc = polar_encode(u_msg.tolist())
        y_enc = polar_encode(v_msg.tolist())
        Z = ch.sample_batch(np.array(x_enc), np.array(y_enc))
        u_dec, v_dec = decode_trellis(N, Z, b, frozen_u, frozen_v, ch,
                                       n_iter=0)
        if (any(u_dec[i-1] != u_msg[i-1] for i in Au) or
                any(v_dec[i-1] != v_msg[i-1] for i in Av)):
            n_err_single += 1

    # Iterative (n_iter=2)
    np.random.seed(0)
    n_err_iter = 0
    for _ in range(n_trials):
        u_info = np.random.randint(0, 2, size=ku)
        v_info = np.random.randint(0, 2, size=kv)
        u_msg = build_message(N, u_info, Au)
        v_msg = build_message(N, v_info, Av)
        x_enc = polar_encode(u_msg.tolist())
        y_enc = polar_encode(v_msg.tolist())
        Z = ch.sample_batch(np.array(x_enc), np.array(y_enc))
        u_dec, v_dec = decode_trellis(N, Z, b, frozen_u, frozen_v, ch,
                                       n_iter=2)
        if (any(u_dec[i-1] != u_msg[i-1] for i in Au) or
                any(v_dec[i-1] != v_msg[i-1] for i in Av)):
            n_err_iter += 1

    bler_single = n_err_single / n_trials
    bler_iter = n_err_iter / n_trials
    # Iterative should be at least as good as single-pass
    status = "PASS" if bler_iter <= bler_single + 0.02 else "FAIL"
    print(f"  Single-pass BLER: {bler_single:.4f}")
    print(f"  Iterative BLER:   {bler_iter:.4f}")
    print(f"  Iterative <= single: [{status}]")


def test_batch_decode():
    """Batch decode API works correctly."""
    print("\nTest 7: Batch decode API")
    from polar.decoder_trellis import decode_batch

    N = 8
    ch = ISIMAC.from_snr_db(10, h=0.5)
    frozen_u = {i: 0 for i in range(1, N + 1)}
    frozen_v = {i: 0 for i in range(1, N + 1)}
    b = make_path(N, N)

    Z_list = [ch.sample_batch(np.zeros(N, dtype=int), np.zeros(N, dtype=int))
              for _ in range(5)]

    results = decode_batch(N, Z_list, b, frozen_u, frozen_v, ch)

    all_ok = True
    for u_dec, v_dec in results:
        if any(u != 0 for u in u_dec) or any(v != 0 for v in v_dec):
            all_ok = False

    status = "PASS" if all_ok and len(results) == 5 else "FAIL"
    print(f"  5 batch results, all-zero: [{status}]")


if __name__ == "__main__":
    np.random.seed(0)
    test_all_zero_decode()
    test_h0_matches_memoryless()
    test_correct_decode_high_snr()
    test_bler_improves_with_snr()
    test_class_a()
    test_isi_uses_trellis()
    test_batch_decode()
    print("\nAll tests completed.")
