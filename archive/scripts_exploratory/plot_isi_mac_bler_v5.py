#!/usr/bin/env python3
"""
Plot ISI-MAC BLER vs N with all decoder variants.
Updated with d=16 h=100 reliable chained results.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data
N_vals = [16, 32, 64, 128, 256]

# Joint trellis SC (from Table 7)
joint_trellis = [0.166, 0.083, 0.026, 0.018, 0.006]

# Chained trellis SC (from Table 7)
chained_trellis = [0.169, 0.082, 0.041, 0.022, 0.006]

# NPD d=16 h=64 (original)
npd_d16_h64 = [0.143, 0.081, 0.046, None, None]

# NPD d=16 h=100 (Session 11-12, chained eval)
npd_d16_h100 = [None, None, 0.032, 0.081, None]

# NPD d=64 h=128
npd_d64 = [None, None, None, 0.030, 0.011]

# Memoryless SC
memoryless = [0.185, 0.114, 0.088, 0.095, None]

fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

ax.semilogy(N_vals, joint_trellis, 'k-s', label='Joint Trellis SC', ms=7, lw=2)
ax.semilogy(N_vals, chained_trellis, 'k--^', label='Chained Trellis SC', ms=7, lw=1.5)

# NPD d=16 h=64
N_d16h64 = [N_vals[i] for i in range(len(npd_d16_h64)) if npd_d16_h64[i] is not None]
v_d16h64 = [v for v in npd_d16_h64 if v is not None]
ax.semilogy(N_d16h64, v_d16h64, 'b-o', label='NPD d=16 h=64', ms=7, lw=1.5)

# NPD d=16 h=100
N_d16h100 = [N_vals[i] for i in range(len(npd_d16_h100)) if npd_d16_h100[i] is not None]
v_d16h100 = [v for v in npd_d16_h100 if v is not None]
ax.semilogy(N_d16h100, v_d16h100, 'r-D', label='NPD d=16 h=100', ms=8, lw=2)

# NPD d=64 h=128
N_d64 = [N_vals[i] for i in range(len(npd_d64)) if npd_d64[i] is not None]
v_d64 = [v for v in npd_d64 if v is not None]
ax.semilogy(N_d64, v_d64, 'g-v', label='NPD d=64 h=128', ms=8, lw=2)

# Memoryless SC
N_memless = [N_vals[i] for i in range(len(memoryless)) if memoryless[i] is not None]
v_memless = [v for v in memoryless if v is not None]
ax.semilogy(N_memless, v_memless, 'gray', ls=':', marker='x', label='Memoryless SC', ms=7, lw=1.5)

ax.set_xlabel('Block Length N', fontsize=13)
ax.set_ylabel('BLER', fontsize=13)
ax.set_title('ISI-MAC Class C: NPD vs Trellis SC\n(h=0.3, SNR=6 dB, 5K-10K CW)', fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.set_xscale('log', base=2)
ax.set_xticks(N_vals)
ax.set_xticklabels([str(n) for n in N_vals])
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0.003, 0.3)

plt.tight_layout()
out_dir = os.path.join(_ROOT, 'project_summary', 'plots')
os.makedirs(out_dir, exist_ok=True)
for ext in ['png', 'pdf']:
    path = os.path.join(out_dir, f'fig_isi_mac_bler_v5.{ext}')
    fig.savefig(path, dpi=150)
    print(f'Saved: {path}')
plt.close()
