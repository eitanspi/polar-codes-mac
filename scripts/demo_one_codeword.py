#!/usr/bin/env python3
"""
Demo: decode ONE codeword with the analytical SC decoder, Class B, N=1024.
Uses Monte Carlo genie-aided design (not Bhattacharyya) for correct frozen sets.
Prints everything along the way so you can see it's valid.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.encoder import polar_encode, polar_encode_batch, build_message, bit_reversal_perm
from polar.channels import BEMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_single

# ── Setup ──────────────────────────────────────────────────────────
N = 1024
n = 10
channel = BEMAC()
rng = np.random.default_rng(7)

# Class B design — use MC genie-aided design
rho = 0.886
Ru_dir = 0.625
Rv_dir = 0.875
ku = max(1, min(round(rho * Ru_dir * N), N - 1))
kv = max(1, min(round(rho * Rv_dir * N), N - 1))

design_file = os.path.join(os.path.dirname(__file__), '..', 'designs', 'bemac_B_n10.npz')
Au, Av, frozen_u, frozen_v, pe_u, pe_v, path_i = design_from_file(design_file, n, ku, kv)
b = make_path(N, path_i)

print("=" * 70)
print(f"  DEMO: One Codeword, Class B, N={N}")
print("=" * 70)

# ── Code design ────────────────────────────────────────────────────
print(f"\n── Code Design (MC genie-aided) ──")
print(f"  N = {N},  n = {n}")
print(f"  ku = {ku} info U bits  (rate Ru = {ku/N:.3f})")
print(f"  kv = {kv} info V bits  (rate Rv = {kv/N:.3f})")
print(f"  Ru + Rv = {(ku+kv)/N:.3f}")
print(f"  rho = {rho}")
print(f"  |frozen_u| = {len(frozen_u)},  |frozen_v| = {len(frozen_v)}")
print(f"  path_i = {path_i}  (Class B: 0^{path_i} 1^{N} 0^{N-path_i})")
print(f"  Path structure: {path_i} U steps, then {N} V steps, then {N-path_i} U steps")
print(f"  Design file: {os.path.basename(design_file)}")

# Show worst (highest error rate) info positions
u_info_pe = [(p, pe_u[p-1]) for p in Au]
v_info_pe = [(p, pe_v[p-1]) for p in Av]
u_info_pe.sort(key=lambda x: -x[1])
v_info_pe.sort(key=lambda x: -x[1])
print(f"\n  Worst U info channel: position {u_info_pe[0][0]}, error rate {u_info_pe[0][1]:.4f}")
print(f"  Best  U info channel: position {u_info_pe[-1][0]}, error rate {u_info_pe[-1][1]:.6f}")
print(f"  Worst V info channel: position {v_info_pe[0][0]}, error rate {v_info_pe[0][1]:.4f}")
print(f"  Best  V info channel: position {v_info_pe[-1][0]}, error rate {v_info_pe[-1][1]:.6f}")

# ── Generate random message ────────────────────────────────────────
print(f"\n── Random Message ──")
info_u = rng.integers(0, 2, size=ku).tolist()
info_v = rng.integers(0, 2, size=kv).tolist()

u = build_message(N, info_u, Au)
v = build_message(N, info_v, Av)

print(f"  info_u ({ku} bits): [{', '.join(str(b) for b in info_u[:20])}{'...' if ku > 20 else ''}]")
print(f"  info_v ({kv} bits): [{', '.join(str(b) for b in info_v[:20])}{'...' if kv > 20 else ''}]")
print(f"  u[0:20] = {u[:20].tolist()}")
print(f"  v[0:20] = {v[:20].tolist()}")
print(f"  sum(u) = {u.sum()} ones out of {N}")
print(f"  sum(v) = {v.sum()} ones out of {N}")

# ── Encode ─────────────────────────────────────────────────────────
print(f"\n── Polar Encoding ──")
x = np.array(polar_encode(u.tolist()))
y = np.array(polar_encode(v.tolist()))

print(f"  x[0:20] = {x[:20].tolist()}")
print(f"  y[0:20] = {y[:20].tolist()}")
print(f"  sum(x) = {x.sum()} ones out of {N}")
print(f"  sum(y) = {y.sum()} ones out of {N}")

# Verify: encode(encode(u)) == u (polar code property)
u_re = np.array(polar_encode(x.tolist()))
assert np.array_equal(u_re, u), "Encode is NOT involutory!"
print(f"  Verify encode(encode(u)) == u: PASS")

# ── Channel ────────────────────────────────────────────────────────
print(f"\n── BE-MAC Channel: Z = X + Y ──")
z = (x + y).tolist()
z_arr = np.array(z)

z_counts = {0: (z_arr == 0).sum(), 1: (z_arr == 1).sum(), 2: (z_arr == 2).sum()}
print(f"  z[0:20] = {z[:20]}")
print(f"  z value counts: 0 -> {z_counts[0]}, 1 -> {z_counts[1]}, 2 -> {z_counts[2]}")
print(f"  (z=0 means x=y=0, z=2 means x=y=1, z=1 is ambiguous)")

# ── Decode ─────────────────────────────────────────────────────────
print(f"\n── SC Decoding (Path B: 0^{path_i} 1^{N} 0^{N-path_i}) ──")
print(f"  Phase 1: decode U_1..U_{path_i} (left subtree)")
print(f"  Phase 2: decode V_1..V_{N} (all leaves)")
print(f"  Phase 3: decode U_{path_i+1}..U_{N} (right subtree)")
print(f"  Total: {2*N} decoding steps")

t0 = time.perf_counter()
u_dec, v_dec = decode_single(N, z, b, frozen_u, frozen_v, channel)
t1 = time.perf_counter()

u_dec = np.array(u_dec)
v_dec = np.array(v_dec)
print(f"  Decode time: {(t1-t0)*1000:.1f} ms")

# ── Check results ──────────────────────────────────────────────────
print(f"\n── Results ──")

# Extract decoded info bits (at the info positions only)
info_u_dec = [u_dec[p-1] for p in Au]
info_v_dec = [v_dec[p-1] for p in Av]

print(f"  Sent     info_u ({ku} bits): [{', '.join(str(b) for b in info_u[:30])}{'...' if ku > 30 else ''}]")
print(f"  Decoded  info_u ({ku} bits): [{', '.join(str(b) for b in info_u_dec[:30])}{'...' if ku > 30 else ''}]")
print()
print(f"  Sent     info_v ({kv} bits): [{', '.join(str(b) for b in info_v[:30])}{'...' if kv > 30 else ''}]")
print(f"  Decoded  info_v ({kv} bits): [{', '.join(str(b) for b in info_v_dec[:30])}{'...' if kv > 30 else ''}]")

# Check U info bits
u_info_ok = all(u_dec[p-1] == bit for p, bit in zip(Au, info_u))
u_errors = sum(u_dec[p-1] != bit for p, bit in zip(Au, info_u))
print(f"\n  U info bits: {'ALL CORRECT' if u_info_ok else f'{u_errors}/{ku} ERRORS'}")
if not u_info_ok:
    err_positions = [(p, bit, u_dec[p-1]) for p, bit in zip(Au, info_u) if u_dec[p-1] != bit]
    for p, sent, got in err_positions[:5]:
        print(f"    position {p}: sent {sent}, decoded {got}, channel error rate {pe_u[p-1]:.4f}")
    if len(err_positions) > 5:
        print(f"    ... ({len(err_positions)} errors total)")

# Check U frozen bits
u_frozen_ok = all(u_dec[p-1] == val for p, val in frozen_u.items())
print(f"  U frozen bits: {'ALL CORRECT' if u_frozen_ok else 'ERRORS!'}")

# Check V info bits
v_info_ok = all(v_dec[p-1] == bit for p, bit in zip(Av, info_v))
v_errors = sum(v_dec[p-1] != bit for p, bit in zip(Av, info_v))
print(f"  V info bits: {'ALL CORRECT' if v_info_ok else f'{v_errors}/{kv} ERRORS'}")
if not v_info_ok:
    err_positions = [(p, bit, v_dec[p-1]) for p, bit in zip(Av, info_v) if v_dec[p-1] != bit]
    for p, sent, got in err_positions[:5]:
        print(f"    position {p}: sent {sent}, decoded {got}, channel error rate {pe_v[p-1]:.4f}")
    if len(err_positions) > 5:
        print(f"    ... ({len(err_positions)} errors total)")

# Check V frozen bits
v_frozen_ok = all(v_dec[p-1] == val for p, val in frozen_v.items())
print(f"  V frozen bits: {'ALL CORRECT' if v_frozen_ok else 'ERRORS!'}")

# Full block check
print(f"\n  Full u vector match: {np.array_equal(u_dec, u)}")
print(f"  Full v vector match: {np.array_equal(v_dec, v)}")

# ── Verify round-trip ──────────────────────────────────────────────
print(f"\n── Round-Trip Verification ──")
x_re = np.array(polar_encode(u_dec.tolist()))
y_re = np.array(polar_encode(v_dec.tolist()))
z_re = x_re + y_re
print(f"  Re-encode decoded u,v -> x',y' -> z'")
print(f"  z' == z (channel output matches): {np.array_equal(z_re, z_arr)}")

# ── Summary ────────────────────────────────────────────────────────
all_ok = u_info_ok and v_info_ok and u_frozen_ok and v_frozen_ok
print(f"\n{'=' * 70}")
if all_ok:
    print(f"  SUCCESS: All {ku} U info bits and {kv} V info bits decoded correctly.")
    print(f"  All {len(frozen_u)} U frozen bits and {len(frozen_v)} V frozen bits correct.")
    print(f"  Rate: Ru={ku/N:.3f}, Rv={kv/N:.3f}, Ru+Rv={(ku+kv)/N:.3f}")
else:
    print(f"  BLOCK ERROR: {u_errors} U errors, {v_errors} V errors.")
    print(f"  (This is expected ~{rho*100:.0f}% of capacity — BLER is not zero.)")
print(f"  Decode time: {(t1-t0)*1000:.1f} ms for N={N}")
print(f"{'=' * 70}")
