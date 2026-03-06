"""
Comprehensive test suite for polar codes MAC project.

Tests encoder, channels, design, decoders, and round-trip correctness.
Runtime: < 30 seconds (except optional @pytest.mark.slow tests)
"""

import pytest
import numpy as np
from polar.encoder import polar_encode, build_message
from polar.channels import BEMAC, ABNMAC
from polar.design import make_path, design_bemac, bhattacharyya_bemac
from polar.design_mc import design_bemac_mc
from polar import decoder
from polar import _decoder_base


# =============================================================================
# 1. ENCODER TESTS
# =============================================================================

class TestEncoder:
    """Tests for encoder.py"""

    def test_polar_encode_idempotence(self):
        """polar_encode(polar_encode(u)) == u for random u at various N."""
        rng = np.random.default_rng(42)
        for N in [4, 8, 16, 64]:
            for _ in range(10):
                u = rng.integers(0, 2, size=N).tolist()
                x = polar_encode(u)
                u_recovered = polar_encode(x)
                assert u_recovered == u, f"Idempotence failed at N={N}"

    def test_build_message_info_placement(self):
        """build_message places info bits at correct positions and frozen bits elsewhere."""
        N = 16
        info_bits = [1, 0, 1, 1]
        info_positions = [2, 5, 10, 15]  # 1-indexed

        u = build_message(N, info_bits, info_positions)

        # Check info bits are at correct positions
        for bit, pos in zip(info_bits, info_positions):
            assert u[pos - 1] == bit, f"Info bit mismatch at position {pos}"

        # Check frozen positions are 0
        frozen_positions = set(range(1, N + 1)) - set(info_positions)
        for pos in frozen_positions:
            assert u[pos - 1] == 0, f"Frozen position {pos} should be 0"

    def test_encoded_codeword_length(self):
        """Encoded codeword has correct length."""
        rng = np.random.default_rng(42)
        for N in [4, 8, 16, 32, 64]:
            u = rng.integers(0, 2, size=N).tolist()
            x = polar_encode(u)
            assert len(x) == N, f"Codeword length mismatch at N={N}"


# =============================================================================
# 2. CHANNEL TESTS
# =============================================================================

class TestBEMAC:
    """Tests for BEMAC channel."""

    def test_bemac_transition_prob_exact(self):
        """BEMAC: transition_prob(z, x, y) returns 1.0 when z == x+y, 0.0 otherwise."""
        be = BEMAC()

        # Test cases where z == x + y
        assert be.transition_prob(0, 0, 0) == 1.0
        assert be.transition_prob(1, 0, 1) == 1.0
        assert be.transition_prob(1, 1, 0) == 1.0
        assert be.transition_prob(2, 1, 1) == 1.0

        # Test cases where z != x + y
        assert be.transition_prob(0, 0, 1) == 0.0
        assert be.transition_prob(0, 1, 0) == 0.0
        assert be.transition_prob(0, 1, 1) == 0.0
        assert be.transition_prob(2, 0, 0) == 0.0
        assert be.transition_prob(2, 0, 1) == 0.0
        assert be.transition_prob(2, 1, 0) == 0.0

    def test_bemac_sample_batch_deterministic(self):
        """BEMAC: sample_batch returns x+y for all inputs."""
        be = BEMAC()
        X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32)
        Y = np.array([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.int32)

        Z = be.sample_batch(X, Y)

        assert Z.shape == X.shape
        expected = X + Y
        assert np.array_equal(Z, expected)

    def test_bemac_transition_probs_sum_to_one(self):
        """BEMAC: transition probs sum to 1 over z for each (x,y)."""
        be = BEMAC()

        for x in [0, 1]:
            for y in [0, 1]:
                prob_sum = sum(be.transition_prob(z, x, y)
                             for z in be.output_alphabet)
                assert abs(prob_sum - 1.0) < 1e-10


# =============================================================================
# 3. DESIGN TESTS
# =============================================================================

class TestDesign:
    """Tests for design.py"""

    def test_make_path_length(self):
        """make_path(N, path_i) returns list of length 2N."""
        for N in [4, 8, 16]:
            for path_i in [0, N // 2, N]:
                path = make_path(N, path_i)
                assert len(path) == 2 * N

    def test_make_path_structure_full_u_first(self):
        """make_path(N, N) starts with N zeros then N ones."""
        N = 16
        path = make_path(N, N)

        # First N should be all 0 (U steps)
        assert path[:N] == [0] * N
        # Last N should be all 1 (V steps)
        assert path[N:] == [1] * N

    def test_make_path_structure_full_v_first(self):
        """make_path(N, 0) starts with N ones then N zeros."""
        N = 16
        path = make_path(N, 0)

        # First N should be all 1 (V steps)
        assert path[:N] == [1] * N
        # Last N should be all 0 (U steps)
        assert path[N:] == [0] * N

    def test_design_bemac_correct_info_frozen_counts(self):
        """design_bemac returns correct number of info/frozen positions."""
        n = 4
        ku, kv = 4, 16
        N = 1 << n

        Au, Av, frozen_u, frozen_v, z_u, z_v = design_bemac(n, ku, kv)

        assert len(Au) == ku
        assert len(Av) == kv
        assert len(frozen_u) == N - ku
        assert len(frozen_v) == N - kv

    def test_design_bemac_disjoint_and_complete(self):
        """Info and frozen sets are disjoint and cover {1,...,N}."""
        n = 4
        ku, kv = 4, 16
        N = 1 << n

        Au, Av, frozen_u, frozen_v, z_u, z_v = design_bemac(n, ku, kv)

        # U sets disjoint and complete
        u_all = set(Au) | set(frozen_u.keys())
        assert u_all == set(range(1, N + 1))
        assert len(set(Au) & set(frozen_u.keys())) == 0

        # V sets disjoint and complete
        v_all = set(Av) | set(frozen_v.keys())
        assert v_all == set(range(1, N + 1))
        assert len(set(Av) & set(frozen_v.keys())) == 0


# =============================================================================
# 4. ROUND-TRIP TESTS: encode → channel → decode → verify
# =============================================================================

class TestRoundTripDecoder:
    """Round-trip tests with _decoder_base.py (recursive probability-based)."""

    def test_bemac_roundtrip_path_0n_1n(self):
        """
        For N=16,32 with BEMAC, analytical design, path_i=N (U-first):
        Encode → transmit → decode round-trip with 0 info bit errors on noiseless channel.
        """
        be = BEMAC()

        for n in [4, 5]:
            N = 1 << n
            ku = 4
            kv = N

            Au, Av, frozen_u, frozen_v, _, _ = design_bemac(n, ku, kv)
            b = make_path(N, path_i=N)

            rng = np.random.default_rng(42)

            for trial in range(50):
                # Generate random info bits
                info_u = rng.integers(0, 2, ku).tolist()
                info_v = rng.integers(0, 2, kv).tolist()

                # Build message and encode
                u = build_message(N, info_u, Au)
                v = build_message(N, info_v, Av)
                x = polar_encode(u.tolist())
                y = polar_encode(v.tolist())

                # Transmit through noiseless BEMAC
                z = be.sample_batch(np.array(x), np.array(y)).tolist()

                # Decode
                u_dec, v_dec = _decoder_base.decode_single(N, z, b, frozen_u, frozen_v, be, log_domain=True)

                # Verify decoded info bits match transmitted info bits
                assert all(u_dec[p-1] == bit for p, bit in zip(Au, info_u)), \
                    f"U info bit mismatch at N={N} trial {trial}"
                assert all(v_dec[p-1] == bit for p, bit in zip(Av, info_v)), \
                    f"V info bit mismatch at N={N} trial {trial}"

    def test_bemac_roundtrip_path_1n_0n(self):
        """
        For N=32 with BEMAC, analytical design, path_i=0 (V-first):
        Encode → transmit → decode round-trip with 0 info bit errors on noiseless channel.
        Symmetric to path_i=N with U and V swapped: call design_bemac(n, kv, ku).
        """
        be = BEMAC()

        for n in [5]:
            N = 1 << n
            kv = 4
            ku = N

            # Symmetric design: design_bemac(n, kv, ku) with swapped roles
            Av, Au, frozen_v, frozen_u, _, _ = design_bemac(n, kv, ku)
            b = make_path(N, path_i=0)

            rng = np.random.default_rng(42)

            for trial in range(50):
                # Generate random info bits
                info_u = rng.integers(0, 2, ku).tolist()
                info_v = rng.integers(0, 2, kv).tolist()

                # Build message and encode
                u = build_message(N, info_u, Au)
                v = build_message(N, info_v, Av)
                x = polar_encode(u.tolist())
                y = polar_encode(v.tolist())

                # Transmit through noiseless BEMAC
                z = be.sample_batch(np.array(x), np.array(y)).tolist()

                # Decode
                u_dec, v_dec = _decoder_base.decode_single(N, z, b, frozen_u, frozen_v, be, log_domain=True)

                # Verify decoded info bits match transmitted info bits
                assert all(u_dec[p-1] == bit for p, bit in zip(Au, info_u)), \
                    f"U info bit mismatch at N={N} trial {trial}"
                assert all(v_dec[p-1] == bit for p, bit in zip(Av, info_v)), \
                    f"V info bit mismatch at N={N} trial {trial}"



class TestRoundTripEfficientDecoder:
    """Round-trip tests with decoder.py (LLR-based)."""

    def test_bemac_roundtrip_efficient_path_0n_1n(self):
        """
        For N=16,32 with BEMAC, efficient decoder, path_i=N (U-first):
        Encode → transmit → decode round-trip with 0 info bit errors on noiseless channel.
        """
        be = BEMAC()

        for n in [4, 5]:
            N = 1 << n
            ku = 4
            kv = N

            Au, Av, frozen_u, frozen_v, _, _ = design_bemac(n, ku, kv)
            b = make_path(N, path_i=N)

            rng = np.random.default_rng(42)

            for trial in range(50):
                info_u = rng.integers(0, 2, ku).tolist()
                info_v = rng.integers(0, 2, kv).tolist()

                u = build_message(N, info_u, Au)
                v = build_message(N, info_v, Av)
                x = polar_encode(u.tolist())
                y = polar_encode(v.tolist())

                z = be.sample_batch(np.array(x), np.array(y)).tolist()

                u_dec, v_dec = decoder.decode_single(N, z, b, frozen_u, frozen_v, be)

                # Verify decoded info bits match transmitted info bits
                assert all(u_dec[p-1] == bit for p, bit in zip(Au, info_u)), \
                    f"U info bit mismatch at N={N} trial {trial}"
                assert all(v_dec[p-1] == bit for p, bit in zip(Av, info_v)), \
                    f"V info bit mismatch at N={N} trial {trial}"

    def test_bemac_roundtrip_efficient_path_1n_0n(self):
        """
        For N=32 with BEMAC, efficient decoder, path_i=0 (V-first):
        Encode → transmit → decode round-trip with 0 info bit errors on noiseless channel.
        Symmetric to path_i=N with U and V swapped: call design_bemac(n, kv, ku).
        """
        be = BEMAC()

        for n in [5]:
            N = 1 << n
            kv = 4
            ku = N

            # Symmetric design: design_bemac(n, kv, ku) with swapped roles
            Av, Au, frozen_v, frozen_u, _, _ = design_bemac(n, kv, ku)
            b = make_path(N, path_i=0)

            rng = np.random.default_rng(42)

            for trial in range(50):
                info_u = rng.integers(0, 2, ku).tolist()
                info_v = rng.integers(0, 2, kv).tolist()

                u = build_message(N, info_u, Au)
                v = build_message(N, info_v, Av)
                x = polar_encode(u.tolist())
                y = polar_encode(v.tolist())

                z = be.sample_batch(np.array(x), np.array(y)).tolist()

                u_dec, v_dec = decoder.decode_single(N, z, b, frozen_u, frozen_v, be)

                # Verify decoded info bits match transmitted info bits
                assert all(u_dec[p-1] == bit for p, bit in zip(Au, info_u)), \
                    f"U info bit mismatch at N={N} trial {trial}"
                assert all(v_dec[p-1] == bit for p, bit in zip(Av, info_v)), \
                    f"V info bit mismatch at N={N} trial {trial}"



# =============================================================================
# 5. DECODER AGREEMENT TEST
# =============================================================================

class TestDecoderAgreement:
    """Verify _decoder_base.py and decoder.py produce identical output."""

    def test_decoders_agree_bemac_n16(self):
        """For N=16 BEMAC path_i=N: both decoders agree on 20 random codewords."""
        be = BEMAC()
        N = 16
        ku = 4
        kv = 16

        Au, Av, frozen_u, frozen_v, _, _ = design_bemac(4, ku, kv)
        b = make_path(N, path_i=N)

        rng = np.random.default_rng(42)

        for trial in range(20):
            info_u = rng.integers(0, 2, ku).tolist()
            info_v = rng.integers(0, 2, kv).tolist()

            u = build_message(N, info_u, Au)
            v = build_message(N, info_v, Av)
            x = polar_encode(u.tolist())
            y = polar_encode(v.tolist())

            z = be.sample_batch(np.array(x), np.array(y)).tolist()

            u_old, v_old = _decoder_base.decode_single(N, z, b, frozen_u, frozen_v, be, log_domain=True)
            u_new, v_new = decoder.decode_single(N, z, b, frozen_u, frozen_v, be)

            assert u_old == u_new, f"Trial {trial}: U decode mismatch"
            assert v_old == v_new, f"Trial {trial}: V decode mismatch"

    def test_decoders_agree_bemac_n32(self):
        """For N=32 BEMAC path_i=N: both decoders agree on 20 random codewords."""
        be = BEMAC()
        N = 32
        ku = 8
        kv = 32

        Au, Av, frozen_u, frozen_v, _, _ = design_bemac(5, ku, kv)
        b = make_path(N, path_i=N)

        rng = np.random.default_rng(42)

        for trial in range(20):
            info_u = rng.integers(0, 2, ku).tolist()
            info_v = rng.integers(0, 2, kv).tolist()

            u = build_message(N, info_u, Au)
            v = build_message(N, info_v, Av)
            x = polar_encode(u.tolist())
            y = polar_encode(v.tolist())

            z = be.sample_batch(np.array(x), np.array(y)).tolist()

            u_old, v_old = _decoder_base.decode_single(N, z, b, frozen_u, frozen_v, be, log_domain=True)
            u_new, v_new = decoder.decode_single(N, z, b, frozen_u, frozen_v, be)

            assert u_old == u_new, f"Trial {trial}: U decode mismatch"
            assert v_old == v_new, f"Trial {trial}: V decode mismatch"


# =============================================================================
# 6. MC DESIGN TEST
# =============================================================================

class TestMCDesign:
    """Tests for design_mc.py"""

    def test_mc_design_matches_analytical_n16(self):
        """
        For N=16 BEMAC: MC design with 500 trials identifies same info set
        as analytical design for ku=4, kv=16.
        """
        # Analytical design
        Au_a, Av_a, _, _, _, _ = design_bemac(4, 4, 16)

        # MC design (faster with 500 trials for test)
        Au_m, Av_m, _, _, _, _ = design_bemac_mc(4, 4, 16, mc_trials=500, seed=42, verbose=False)

        # Should match (or be very close for MC randomness)
        assert Au_a == Au_m, f"U info set mismatch: analytical={Au_a}, mc={Au_m}"
        assert Av_a == Av_m, f"V info set mismatch: analytical={Av_a}, mc={Av_m}"


# =============================================================================
# OPTIONAL SLOW TESTS
# =============================================================================

@pytest.mark.slow
class TestSlowRoundTrips:
    """Optional slow tests for larger block lengths."""

    def test_bemac_noiseless_n256(self):
        """Optional: N=256 round-trip test (may be slow)."""
        be = BEMAC()
        N = 256
        n = 8
        ku = 64
        kv = 256

        Au, Av, frozen_u, frozen_v, _, _ = design_bemac(n, ku, kv)
        b = make_path(N, path_i=N)

        rng = np.random.default_rng(42)
        errors = 0

        for trial in range(5):  # Only 5 trials for speed
            info_u = rng.integers(0, 2, ku).tolist()
            info_v = rng.integers(0, 2, kv).tolist()

            u = build_message(N, info_u, Au)
            v = build_message(N, info_v, Av)
            x = polar_encode(u.tolist())
            y = polar_encode(v.tolist())

            z = be.sample_batch(np.array(x), np.array(y)).tolist()

            u_dec, v_dec = decoder.decode_single(N, z, b, frozen_u, frozen_v, be)

            u_ok = all(u_dec[p - 1] == bit for p, bit in zip(Au, info_u))
            v_ok = all(v_dec[p - 1] == bit for p, bit in zip(Av, info_v))

            if not (u_ok and v_ok):
                errors += 1

        assert errors == 0, f"Failed: {errors}/5 block errors at N=256"
