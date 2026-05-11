#!/usr/bin/env python3
"""
generate_comprehensive_isi_table.py
====================================
Generate a comprehensive markdown table of all ISI-MAC decoder results.
"""
import json
import os
import math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def wilson_ci(k, n, z=1.96):
    p = k / n
    d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    w = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / d
    return max(0, c-w), min(1, c+w)

# Load all data
with open(os.path.join(_ROOT, 'results/reliable_evals/isi_mac_sc_10kcw.json')) as f:
    chained_sc = json.load(f)

with open(os.path.join(_ROOT, 'results/reliable_evals/isi_mac_npd_reliable.json')) as f:
    npd_data = json.load(f)

print("# Comprehensive ISI-MAC Decoder Comparison")
print(f"# Channel: ISI-MAC h=0.3, SNR=6dB, Class C corner rate")
print(f"# All reliable evaluations (>=5000 CW)")
print()

print("| N | Decoder | Config | BLER | Errors | CW | 95% CI | Params |")
print("|---|---------|--------|------|--------|----|--------|--------|")

for N in [16, 32, 64, 128, 256]:
    label_c = f'N{N}'

    # Chained trellis SC
    if label_c in chained_sc:
        r = chained_sc[label_c]
        lo, hi = wilson_ci(r['errs_total'], r['n_cw'])
        print(f"| {N} | Chained Trellis SC | 2-state BCJR | "
              f"{r['bler_total']:.4f} | {r['errs_total']} | {r['n_cw']} | "
              f"[{lo:.4f}, {hi:.4f}] | -- |")

    # Joint trellis SC (from npd_reliable or known values)
    joint_known = {16: (0.166, 1664, 10000), 32: (0.083, 825, 10000),
                   64: (0.026, 262, 10000), 128: (0.018, 180, 10000)}
    if N in joint_known:
        bler, errs, cw = joint_known[N]
        lo, hi = wilson_ci(errs, cw)
        print(f"| {N} | Joint Trellis SC | 4-state BCJR | "
              f"{bler:.4f} | {errs} | {cw} | "
              f"[{lo:.4f}, {hi:.4f}] | -- |")

    # NPD d=16
    npd_labels = {16: 'N16_d16_bigru', 32: 'N32_d16_window',
                  64: 'N64_d16_bigru'}
    if N in npd_labels and npd_labels[N] in npd_data:
        r = npd_data[npd_labels[N]]
        lo, hi = wilson_ci(r['chained_errs'], r['chained_n_cw'])
        enc = r.get('encoder', 'bigru')
        print(f"| {N} | Chained NPD | d=16 h=64 {enc} | "
              f"{r['chained_bler']:.4f} | {r['chained_errs']} | {r['chained_n_cw']} | "
              f"[{lo:.4f}, {hi:.4f}] | ~20K |")

    # NPD d=64
    npd64_labels = {128: 'N128_d64_bigru', 256: 'N256_d64_bigru'}
    if N in npd64_labels and npd64_labels[N] in npd_data:
        r = npd_data[npd64_labels[N]]
        lo, hi = wilson_ci(r['chained_errs'], r['chained_n_cw'])
        print(f"| {N} | Chained NPD | d=64 h=128 bigru | "
              f"{r['chained_bler']:.4f} | {r['chained_errs']} | {r['chained_n_cw']} | "
              f"[{lo:.4f}, {hi:.4f}] | ~114K |")

    print(f"|---|---------|--------|------|--------|----|--------|--------|")

print()
print("## Ratio Summary (NPD / Chained Trellis SC)")
print()
ratios = []
for N in [16, 32, 64, 128, 256]:
    label_c = f'N{N}'
    if label_c not in chained_sc:
        continue
    trellis_bler = chained_sc[label_c]['bler_total']

    npd_labels = {16: 'N16_d16_bigru', 32: 'N32_d16_window',
                  64: 'N64_d16_bigru', 128: 'N128_d64_bigru', 256: 'N256_d64_bigru'}
    if N in npd_labels and npd_labels[N] in npd_data:
        npd_bler = npd_data[npd_labels[N]]['chained_bler']
        ratio = npd_bler / trellis_bler
        verdict = "BEATS" if ratio < 1 else ("MATCHES" if ratio < 1.1 else f"{ratio:.2f}x gap")
        print(f"N={N}: NPD={npd_bler:.4f} / Trellis={trellis_bler:.4f} = {ratio:.2f}x -- {verdict}")
        ratios.append((N, ratio))

print()
avg = sum(r for _, r in ratios) / len(ratios)
print(f"Average ratio: {avg:.2f}x")
