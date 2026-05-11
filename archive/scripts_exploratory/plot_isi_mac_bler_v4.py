#!/usr/bin/env python3
"""
plot_isi_mac_bler_v4.py
=======================
Regenerate ISI-MAC BLER vs N figure with:
- Joint trellis SC (4-state, strongest analytical baseline)
- Chained trellis SC (2-state, apples-to-apples comparison)
- NPD d=16 h=64 (existing models, 5K CW reliable)
- NPD d=64 h=128 (N=128, 256)
- Memoryless SC (where available)

Data sources:
- Joint trellis: results/reliable_evals/isi_mac_npd_reliable.json (N=64 10K),
  class_c_npd/results/chained_trellis_sc_isi_mac.json (others)
- Chained trellis: results/reliable_evals/isi_mac_sc_10kcw.json (10K CW)
- NPD: results/reliable_evals/isi_mac_npd_reliable.json (5K CW)
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data from reliable evaluations
# Joint trellis SC (best available -- mix of sources)
joint_trellis = {
    16: (0.166, 1664, 10000),
    32: (0.083, 825, 10000),
    64: (0.026, 262, 10000),
    128: (0.018, 180, 10000),
    # 256: joint not run at 10K yet
}

# Chained trellis SC (all from isi_mac_sc_10kcw.json, 10K CW)
chained_trellis = {
    16: (0.169, 1689, 10000),
    32: (0.082, 822, 10000),
    64: (0.041, 407, 10000),
    128: (0.022, 223, 10000),
    256: (0.006, 61, 10000),
}

# NPD d=16 (from isi_mac_npd_reliable.json, 5K CW)
# N=64 updated to d=16 h=100 result (Session 11)
npd_d16 = {
    16: (0.143, 715, 5000),
    32: (0.081, 406, 5000),
    64: (0.027, 137, 5000),  # d=16 h=100 @95K iters, session 11
}

# NPD d=64 (from isi_mac_npd_reliable.json, 5K CW)
npd_d64 = {
    128: (0.030, 150, 5000),
    256: (0.011, 56, 5000),
}

# Memoryless SC (from chapter)
memoryless = {
    16: 0.185,
    32: 0.114,
    64: 0.088,
    128: 0.095,
}


def wilson_ci(k, n, z=1.96):
    p = k / n
    d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    w = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / d
    return max(0, c-w), min(1, c+w)


fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

# Joint trellis SC
Ns_j = sorted(joint_trellis.keys())
blers_j = [joint_trellis[n][0] for n in Ns_j]
ax.semilogy(Ns_j, blers_j, 's-', color='#2166ac', markersize=8, linewidth=2,
            label='Joint Trellis SC (4-state)', zorder=5)

# Chained trellis SC
Ns_c = sorted(chained_trellis.keys())
blers_c = [chained_trellis[n][0] for n in Ns_c]
ax.semilogy(Ns_c, blers_c, 'D--', color='#67a9cf', markersize=7, linewidth=1.5,
            label='Chained Trellis SC (2-state)', zorder=4)

# NPD d=16
Ns_n16 = sorted(npd_d16.keys())
blers_n16 = [npd_d16[n][0] for n in Ns_n16]
# Error bars
ci_lo_16 = []
ci_hi_16 = []
for n in Ns_n16:
    lo, hi = wilson_ci(npd_d16[n][1], npd_d16[n][2])
    ci_lo_16.append(npd_d16[n][0] - lo)
    ci_hi_16.append(hi - npd_d16[n][0])
ax.errorbar(Ns_n16, blers_n16, yerr=[ci_lo_16, ci_hi_16],
            fmt='o-', color='#b2182b', markersize=8, linewidth=2, capsize=3,
            label='NPD d=16 BiGRU (h=64/100)', zorder=6)

# NPD d=64
Ns_n64 = sorted(npd_d64.keys())
blers_n64 = [npd_d64[n][0] for n in Ns_n64]
ci_lo_64 = []
ci_hi_64 = []
for n in Ns_n64:
    lo, hi = wilson_ci(npd_d64[n][1], npd_d64[n][2])
    ci_lo_64.append(npd_d64[n][0] - lo)
    ci_hi_64.append(hi - npd_d64[n][0])
ax.errorbar(Ns_n64, blers_n64, yerr=[ci_lo_64, ci_hi_64],
            fmt='^-', color='#d6604d', markersize=9, linewidth=2, capsize=3,
            label='NPD d=64, h=128 (BiGRU)', zorder=6)

# Memoryless SC
Ns_m = sorted(memoryless.keys())
blers_m = [memoryless[n] for n in Ns_m]
ax.semilogy(Ns_m, blers_m, 'x:', color='#999999', markersize=8, linewidth=1.5,
            label='Memoryless SC (no ISI)', zorder=3)

ax.set_xlabel('Block length N', fontsize=13)
ax.set_ylabel('Chained BLER', fontsize=13)
ax.set_title('ISI-MAC (h=0.3, SNR 6 dB, Class C corner rate)', fontsize=13)
ax.set_xscale('log', base=2)
ax.set_xticks([16, 32, 64, 128, 256])
ax.set_xticklabels(['16', '32', '64', '128', '256'])
ax.set_ylim(3e-3, 0.3)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, which='both', alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()

out_dir = os.path.join(_ROOT, 'docs', 'paper_figures')
os.makedirs(out_dir, exist_ok=True)
for ext in ['pdf', 'png']:
    out_path = os.path.join(out_dir, f'fig_isi_mac_bler_v4.{ext}')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')

plt.close()
print('Done.')
