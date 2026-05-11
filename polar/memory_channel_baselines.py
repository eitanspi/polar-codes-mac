#!/usr/bin/env python3
"""
Generate SC BLER baselines for MAC channels with memory.

Uses the SC trellis decoder (decoder_trellis.py) for each channel.
Sweeps N=16,32,64,128,256,512,1024 and picks parameters such that
BLER decreases monotonically from ~0.1 to ~0.0001.

Channels:
  1. ISI-MAC (1 tap): Z = sx + sy + h*(sx_prev + sy_prev) + W
  2. Trapdoor MAC: Z = S XOR noise, S = X XOR Y XOR S_prev
  3. Gilbert-Elliott MAC: bursty AWGN with good/bad states
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from polar.encoder import polar_encode_batch
from polar.design import make_path
from polar.channels_memory import ISIMAC
from polar.channels_memory_new import TrapdoorMAC, GilbertElliottMAC, ISIMAC2

# Try to import trellis decoder
try:
    from polar.decoder_trellis import decode_single_trellis
    HAS_TRELLIS = True
except ImportError:
    HAS_TRELLIS = False
    print("WARNING: No trellis decoder. Using FB+memoryless SC fallback.")

# Fallback: FB equalization + memoryless SC
try:
    from polar.decoder_trellis import forward_backward_mac, decode_single_fb_sc
    HAS_FB_SC = True
except ImportError:
    HAS_FB_SC = False


def evaluate_bler_memory(channel, N, b, frozen_u, frozen_v, Au, Av,
                          n_cw=500, seed=999):
    """Evaluate BLER using the trellis or FB+SC decoder."""
    rng = np.random.default_rng(seed)
    errs = 0
    n = int(np.log2(N))

    for _ in range(n_cw):
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au:
            u[p - 1] = rng.integers(0, 2)
        for p in Av:
            v[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]

        if HAS_TRELLIS:
            u_dec, v_dec = decode_single_trellis(N, z, b, frozen_u, frozen_v, channel)
        elif HAS_FB_SC:
            u_dec, v_dec = decode_single_fb_sc(N, z, b, frozen_u, frozen_v, channel)
        else:
            # Can't decode memory channels without trellis
            return 1.0

        ue = any(u_dec[p - 1] != u[p - 1] for p in Au)
        ve = any(v_dec[p - 1] != v[p - 1] for p in Av)
        if ue or ve:
            errs += 1

    return errs / n_cw


def design_mc_memory(channel, N, n_trials=50000, seed=42):
    """
    Monte Carlo design for memory channel.
    Run genie-aided SC trellis decode, measure per-position error rates.
    """
    n = int(np.log2(N))
    b = make_path(N, N)  # Class C path
    rng = np.random.default_rng(seed)

    pe_u = np.zeros(N)
    pe_v = np.zeros(N)
    count = 0

    # Full frozen sets (all frozen) for genie-aided evaluation
    all_frozen_u = {i: 0 for i in range(1, N + 1)}
    all_frozen_v = {i: 0 for i in range(1, N + 1)}

    for trial in range(n_trials):
        u = rng.integers(0, 2, N)
        v = rng.integers(0, 2, N)
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]

        # Genie-aided: use true bits for frozen set
        frozen_u_genie = {i + 1: int(u[i]) for i in range(N)}
        frozen_v_genie = {i + 1: int(v[i]) for i in range(N)}

        if HAS_TRELLIS:
            u_dec, v_dec = decode_single_trellis(
                N, z, b, frozen_u_genie, frozen_v_genie, channel)
        elif HAS_FB_SC:
            u_dec, v_dec = decode_single_fb_sc(
                N, z, b, frozen_u_genie, frozen_v_genie, channel)
        else:
            break

        for i in range(N):
            if u_dec[i] != u[i]:
                pe_u[i] += 1
            if v_dec[i] != v[i]:
                pe_v[i] += 1
        count += 1

        if (trial + 1) % 5000 == 0:
            print(f'    {trial + 1}/{n_trials}', flush=True)

    pe_u /= max(count, 1)
    pe_v /= max(count, 1)
    return pe_u, pe_v


def select_info_positions(pe, k):
    """Select k best positions (lowest Pe) with polar tiebreak."""
    idx = np.argsort(pe)
    return sorted(idx[:k] + 1)  # 1-indexed


if __name__ == '__main__':
    print("Testing memory channel + trellis decoder availability...")
    print(f"  Trellis decoder: {HAS_TRELLIS}")
    print(f"  FB+SC decoder: {HAS_FB_SC}")

    if not HAS_TRELLIS and not HAS_FB_SC:
        print("No decoder for memory channels available. Generating designs only.")

    # Quick test at N=16
    from polar.design import make_path
    ch = ISIMAC(sigma2=0.5, h=0.5)
    N = 16
    b = make_path(N, N)  # Class C

    # Need to generate designs first
    print(f"\nGenerating ISI-MAC design for N={N}...")
    pe_u, pe_v = design_mc_memory(ch, N, n_trials=1000)
    print(f"  pe_u range: [{pe_u.min():.4f}, {pe_u.max():.4f}]")
    print(f"  pe_v range: [{pe_v.min():.4f}, {pe_v.max():.4f}]")
