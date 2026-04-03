"""
test_sct_mac.py
===============
Systematic tests for the SC Trellis (SCT) decoder.
"""

import sys
import numpy as np

sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode, polar_encode_batch, build_message
from polar.design import design_gmac, make_path
from polar.decoder_trellis_sct import (
    decode_single as sct_decode_single,
    _circ_conv_trellis, _norm_prod_trellis, _logsumexp_axis,
)
from polar.decoder_interleaved import (
    decode_single as memoryless_decode_single,
    _circ_conv_batch, _norm_prod_batch,
)
from polar.efficient_decoder import build_log_W_leaf

_NEG_INF = -np.inf


def test0_circ_conv_trellis():
    """
    Test 0: Verify circ_conv_trellis matches memoryless circ_conv when
    states are trivial (S=1) or when we marginalize over states for h=0.
    """
    print("=" * 60)
    print("Test 0: circ_conv_trellis vs memoryless circ_conv")
    print("=" * 60)

    np.random.seed(42)
    L = 4
    S = 1

    # With S=1, the trellis tensors (L, 2, 2, 1, 1) should behave
    # identically to (L, 2, 2) memoryless tensors.
    A_22 = np.random.randn(L, 2, 2) - 1.0  # log-domain values
    B_22 = np.random.randn(L, 2, 2) - 1.0

    # Normalize to valid log-probs
    for i in range(L):
        t = np.logaddexp(np.logaddexp(A_22[i, 0, 0], A_22[i, 0, 1]),
                         np.logaddexp(A_22[i, 1, 0], A_22[i, 1, 1]))
        A_22[i] -= t
        t = np.logaddexp(np.logaddexp(B_22[i, 0, 0], B_22[i, 0, 1]),
                         np.logaddexp(B_22[i, 1, 0], B_22[i, 1, 1]))
        B_22[i] -= t

    # Memoryless result
    C_memoryless = _circ_conv_batch(A_22, B_22)  # (L, 2, 2)

    # Trellis version with S=1
    A_trellis = A_22[:, :, :, None, None]  # (L, 2, 2, 1, 1)
    B_trellis = B_22[:, :, :, None, None]
    C_trellis = _circ_conv_trellis(A_trellis, B_trellis, S=1)  # (L, 2, 2, 1, 1)
    C_trellis_22 = C_trellis[:, :, :, 0, 0]  # marginalize (trivial for S=1)

    diff = np.max(np.abs(C_memoryless - C_trellis_22))
    ok = diff < 1e-10
    print(f"  S=1 match: max_diff = {diff:.2e} {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  Memoryless:\n{C_memoryless[0]}")
        print(f"  Trellis:\n{C_trellis_22[0]}")

    # Now test with S=4 but uniform state distribution
    # If both tensors have uniform states, the circ_conv should still
    # produce the same BIT decisions after marginalizing states.
    S = 4
    log_uniform_s = -np.log(float(S * S))
    A_trellis4 = np.full((L, 2, 2, S, S), _NEG_INF, dtype=np.float64)
    B_trellis4 = np.full((L, 2, 2, S, S), _NEG_INF, dtype=np.float64)

    # Place memoryless values uniformly across diagonal states
    # For a memoryless channel, all states are equal, so spread uniformly
    for l in range(L):
        for a in range(2):
            for b in range(2):
                for s in range(S):
                    for sp in range(S):
                        A_trellis4[l, a, b, s, sp] = A_22[l, a, b] + log_uniform_s
                        B_trellis4[l, a, b, s, sp] = B_22[l, a, b] + log_uniform_s

    C_trellis4 = _circ_conv_trellis(A_trellis4, B_trellis4, S=S)
    # Marginalize over states
    C_marg = _logsumexp_axis(C_trellis4.reshape(L, 2, 2, -1), axis=3)

    # The marginalized result should differ from memoryless by a constant
    # (due to extra state sums), but the relative values should match.
    # Normalize both to compare:
    for l in range(L):
        t_ml = np.logaddexp(np.logaddexp(C_memoryless[l, 0, 0], C_memoryless[l, 0, 1]),
                            np.logaddexp(C_memoryless[l, 1, 0], C_memoryless[l, 1, 1]))
        C_memoryless[l] -= t_ml
        t_mg = np.logaddexp(np.logaddexp(C_marg[l, 0, 0], C_marg[l, 0, 1]),
                            np.logaddexp(C_marg[l, 1, 0], C_marg[l, 1, 1]))
        C_marg[l] -= t_mg

    diff4 = np.max(np.abs(C_memoryless - C_marg))
    ok4 = diff4 < 1e-10
    print(f"  S=4 uniform match (normalized): max_diff = {diff4:.2e} {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print(f"  Memoryless (norm):\n{C_memoryless[0]}")
        print(f"  Trellis S=4 (marg, norm):\n{C_marg[0]}")

    return ok and ok4


def test1_n2_allzero_classC():
    """
    Test 1: N=2, all-zero codewords, Class C path, high SNR.
    Simplest possible case.
    """
    print("\n" + "=" * 60)
    print("Test 1: N=2, all-zero, Class C")
    print("=" * 60)

    N = 2
    n = 1
    channel = ISIMAC(sigma2=0.001, h=0.5)

    # All-zero codewords
    X = np.array([0, 0])
    Y = np.array([0, 0])
    Z = channel.sample_batch(X, Y)

    # Class C: all U first, then all V
    b = make_path(N, N)  # [0, 0, 1, 1]

    # All frozen (simplest case)
    frozen_u = {1: 0, 2: 0}
    frozen_v = {1: 0, 2: 0}

    u_dec, v_dec = sct_decode_single(N, Z, b, frozen_u, frozen_v, channel)
    ok = (u_dec == [0, 0]) and (v_dec == [0, 0])
    print(f"  u_dec={u_dec}, v_dec={v_dec} {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"  Z={Z}")

    return ok


def test2_n2_h0_match():
    """
    Test 2: N=2, h=0 (no memory), compare SCT with memoryless decoder.
    With h=0, the ISI-MAC reduces to a memoryless Gaussian MAC.
    """
    print("\n" + "=" * 60)
    print("Test 2: N=2, h=0, match with memoryless decoder")
    print("=" * 60)

    N = 2
    n = 1
    sigma2 = 0.1

    from polar.channels import GaussianMAC
    ch_memoryless = GaussianMAC(sigma2=sigma2)
    ch_memory = ISIMAC(sigma2=sigma2, h=0.0)  # h=0 -> memoryless

    b = make_path(N, N)  # Class C

    # 1 info bit for U, 1 for V (rest frozen)
    frozen_u = {1: 0}  # position 1 frozen, position 2 info
    frozen_v = {1: 0}

    np.random.seed(123)
    n_trials = 50
    matches = 0

    for trial in range(n_trials):
        u_info = np.random.randint(0, 2)
        v_info = np.random.randint(0, 2)

        u_msg = np.array([0, u_info])
        v_msg = np.array([0, v_info])
        X = np.array(polar_encode(u_msg.tolist()))
        Y = np.array(polar_encode(v_msg.tolist()))

        # Use memoryless channel for sampling (so both decoders see same channel)
        Z_ml = ch_memoryless.sample_batch(X, Y)

        # Decode with memoryless decoder
        u_ml, v_ml = memoryless_decode_single(
            N, Z_ml, b, frozen_u, frozen_v, ch_memoryless)

        # Decode with SCT decoder (h=0)
        u_sct, v_sct = sct_decode_single(
            N, Z_ml, b, frozen_u, frozen_v, ch_memory)

        if u_ml == u_sct and v_ml == v_sct:
            matches += 1
        elif trial < 5:
            # Print diagnostics for first few mismatches
            print(f"  Trial {trial}: MISMATCH")
            print(f"    sent u={u_msg.tolist()}, v={v_msg.tolist()}")
            print(f"    Z={Z_ml}")
            print(f"    ML:  u={u_ml}, v={v_ml}")
            print(f"    SCT: u={u_sct}, v={v_sct}")

    ok = matches == n_trials
    print(f"  Matches: {matches}/{n_trials} {'PASS' if ok else 'FAIL'}")
    return ok


def test3_n4_h0_match():
    """
    Test 3: N=4, h=0, match with memoryless decoder.
    """
    print("\n" + "=" * 60)
    print("Test 3: N=4, h=0, match with memoryless decoder")
    print("=" * 60)

    N = 4
    n = 2
    sigma2 = 0.1

    from polar.channels import GaussianMAC
    ch_memoryless = GaussianMAC(sigma2=sigma2)
    ch_memory = ISIMAC(sigma2=sigma2, h=0.0)

    b = make_path(N, N)  # Class C

    # Use design_gmac for frozen sets
    ku, kv = 1, 2
    Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)

    np.random.seed(456)
    n_trials = 50
    matches = 0
    first_mismatch_printed = 0

    for trial in range(n_trials):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for pos in Au:
            u_msg[pos - 1] = np.random.randint(0, 2)
        for pos in Av:
            v_msg[pos - 1] = np.random.randint(0, 2)

        X = np.array(polar_encode(u_msg.tolist()))
        Y = np.array(polar_encode(v_msg.tolist()))
        Z_ml = ch_memoryless.sample_batch(X, Y)

        u_ml, v_ml = memoryless_decode_single(
            N, list(Z_ml), b, frozen_u, frozen_v, ch_memoryless)
        u_sct, v_sct = sct_decode_single(
            N, list(Z_ml), b, frozen_u, frozen_v, ch_memory)

        if u_ml == u_sct and v_ml == v_sct:
            matches += 1
        elif first_mismatch_printed < 3:
            first_mismatch_printed += 1
            print(f"  Trial {trial}: MISMATCH")
            print(f"    sent u={u_msg.tolist()}, v={v_msg.tolist()}")
            print(f"    Z={[f'{z:.3f}' for z in Z_ml]}")
            print(f"    ML:  u={u_ml}, v={v_ml}")
            print(f"    SCT: u={u_sct}, v={v_sct}")

    ok = matches == n_trials
    print(f"  Matches: {matches}/{n_trials} {'PASS' if ok else 'FAIL'}")
    return ok


def test4_n8_h0_match():
    """
    Test 4: N=8, h=0, match with memoryless decoder.
    """
    print("\n" + "=" * 60)
    print("Test 4: N=8, h=0, match with memoryless decoder")
    print("=" * 60)

    N = 8
    n = 3
    sigma2 = 0.001

    from polar.channels import GaussianMAC
    ch_memoryless = GaussianMAC(sigma2=sigma2)
    ch_memory = ISIMAC(sigma2=sigma2, h=0.0)

    b = make_path(N, N)  # Class C

    ku, kv = 2, 4
    Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)

    np.random.seed(789)
    n_trials = 50
    matches = 0
    first_mismatch_printed = 0

    for trial in range(n_trials):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for pos in Au:
            u_msg[pos - 1] = np.random.randint(0, 2)
        for pos in Av:
            v_msg[pos - 1] = np.random.randint(0, 2)

        X = np.array(polar_encode(u_msg.tolist()))
        Y = np.array(polar_encode(v_msg.tolist()))
        Z_ml = ch_memoryless.sample_batch(X, Y)

        u_ml, v_ml = memoryless_decode_single(
            N, list(Z_ml), b, frozen_u, frozen_v, ch_memoryless)
        u_sct, v_sct = sct_decode_single(
            N, list(Z_ml), b, frozen_u, frozen_v, ch_memory)

        if u_ml == u_sct and v_ml == v_sct:
            matches += 1
        elif first_mismatch_printed < 3:
            first_mismatch_printed += 1
            print(f"  Trial {trial}: MISMATCH")
            print(f"    sent u={u_msg.tolist()}, v={v_msg.tolist()}")
            print(f"    ML:  u={u_ml}, v={v_ml}")
            print(f"    SCT: u={u_sct}, v={v_sct}")

    ok = matches == n_trials
    print(f"  Matches: {matches}/{n_trials} {'PASS' if ok else 'FAIL'}")
    return ok


def test5_n8_h05_bler_vs_snr():
    """
    Test 5: N=8, h=0.5, show BLER decreases with SNR.
    """
    print("\n" + "=" * 60)
    print("Test 5: N=8, h=0.5, BLER vs SNR")
    print("=" * 60)

    N = 8
    n = 3

    b = make_path(N, N)
    ku, kv = 2, 4
    # Use moderate sigma2 for design
    _, _, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2=0.1)
    Au = sorted(set(range(1, N + 1)) - set(frozen_u.keys()))
    Av = sorted(set(range(1, N + 1)) - set(frozen_v.keys()))

    snr_dbs = [5, 10, 15, 20]
    blers = []
    n_trials = 200

    for snr_db in snr_dbs:
        channel = ISIMAC.from_snr_db(snr_db, h=0.5)
        errors = 0

        np.random.seed(100 + snr_db)
        for trial in range(n_trials):
            u_msg = np.zeros(N, dtype=int)
            v_msg = np.zeros(N, dtype=int)
            for pos in Au:
                u_msg[pos - 1] = np.random.randint(0, 2)
            for pos in Av:
                v_msg[pos - 1] = np.random.randint(0, 2)

            X = np.array(polar_encode(u_msg.tolist()))
            Y = np.array(polar_encode(v_msg.tolist()))
            Z = channel.sample_batch(X, Y)

            u_dec, v_dec = sct_decode_single(
                N, Z, b, frozen_u, frozen_v, channel)

            if u_dec != u_msg.tolist() or v_dec != v_msg.tolist():
                errors += 1

        bler = errors / n_trials
        blers.append(bler)
        print(f"  SNR={snr_db:2d}dB: BLER={bler:.4f} ({errors}/{n_trials})")

    # Check BLER is monotonically non-increasing (roughly)
    ok = blers[-1] <= blers[0]
    print(f"  BLER decreasing: {blers[0]:.4f} -> {blers[-1]:.4f} {'PASS' if ok else 'FAIL'}")
    return ok


def test6_n8_h05_high_snr():
    """
    Test 6: N=8, h=0.5, correct decode at high SNR (30dB).
    Expect >90% correct.
    """
    print("\n" + "=" * 60)
    print("Test 6: N=8, h=0.5, high SNR decode accuracy")
    print("=" * 60)

    N = 8
    n = 3
    snr_db = 30
    channel = ISIMAC.from_snr_db(snr_db, h=0.5)

    b = make_path(N, N)
    ku, kv = 2, 4
    _, _, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2=0.1)
    Au = sorted(set(range(1, N + 1)) - set(frozen_u.keys()))
    Av = sorted(set(range(1, N + 1)) - set(frozen_v.keys()))

    np.random.seed(999)
    n_trials = 100
    correct = 0

    for trial in range(n_trials):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for pos in Au:
            u_msg[pos - 1] = np.random.randint(0, 2)
        for pos in Av:
            v_msg[pos - 1] = np.random.randint(0, 2)

        X = np.array(polar_encode(u_msg.tolist()))
        Y = np.array(polar_encode(v_msg.tolist()))
        Z = channel.sample_batch(X, Y)

        u_dec, v_dec = sct_decode_single(
            N, Z, b, frozen_u, frozen_v, channel)

        if u_dec == u_msg.tolist() and v_dec == v_msg.tolist():
            correct += 1

    pct = correct / n_trials * 100
    ok = pct >= 90
    print(f"  Correct: {correct}/{n_trials} ({pct:.1f}%) {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = {}

    results['test0'] = test0_circ_conv_trellis()
    results['test1'] = test1_n2_allzero_classC()
    results['test2'] = test2_n2_h0_match()

    if results['test2']:
        results['test3'] = test3_n4_h0_match()
    else:
        print("\nSkipping Test 3 (Test 2 failed)")
        results['test3'] = False

    if results.get('test3', False):
        results['test4'] = test4_n8_h0_match()
    else:
        print("\nSkipping Test 4 (Test 3 failed)")
        results['test4'] = False

    results['test5'] = test5_n8_h05_bler_vs_snr()
    results['test6'] = test6_n8_h05_high_snr()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
