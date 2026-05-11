#!/usr/bin/env python3
"""
Plot wall diagnostics: NPD/SC ratio vs N for all channels.
Shows where walls appear (ratio > 1.5x).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

# ISI-MAC: NPD/chained trellis ratio
N_isi = [16, 32, 64, 128, 256]
ratio_isi = [0.143/0.169, 0.081/0.082, 0.032/0.041, 0.030/0.022, 0.011/0.006]
ax.plot(N_isi, ratio_isi, 'b-o', label='ISI-MAC (NPD/Chained Trellis)', ms=7, lw=2)

# GMAC Class B: NCG/SC ratio
N_gmac_b = [32, 64, 128, 256, 512]
ratio_gmac_b = [0.050/0.045, 0.028/0.028, 0.023/0.019, 0.023/0.006, 0.012/0.001]
ax.plot(N_gmac_b, ratio_gmac_b, 'r-s', label='GMAC Class B (NCG/SC)', ms=7, lw=2)

# MA-AGN: NPD/memoryless SC ratio
N_ma = [16, 32, 64]
ratio_ma = [0.138/0.175, 0.112/0.077, 0.066/0.028]
ax.plot(N_ma, ratio_ma, 'g-^', label='MA-AGN (NPD/Memoryless SC)', ms=7, lw=2)

# GMAC Class C: NPD/SC ratio
N_gmac_c = [16, 32, 64, 256]
ratio_gmac_c = [0.107/0.162, 0.037/0.068, 0.010/0.027, 0.0003/0.002]
ax.plot(N_gmac_c, ratio_gmac_c, 'm-D', label='GMAC Class C (NPD/SC)', ms=7, lw=2)

# Reference lines
ax.axhline(y=1.0, color='black', ls='-', lw=0.5, alpha=0.5)
ax.axhline(y=1.5, color='red', ls='--', lw=1, alpha=0.3)
ax.annotate('Parity (ratio=1)', xy=(20, 1.05), fontsize=9, color='gray')
ax.annotate('Wall threshold (1.5x)', xy=(20, 1.6), fontsize=9, color='red', alpha=0.5)

# Shade the "NPD wins" region
ax.axhspan(0, 1.0, color='green', alpha=0.05)
ax.axhspan(1.5, 15, color='red', alpha=0.05)

ax.set_xlabel('Block Length N', fontsize=13)
ax.set_ylabel('NPD/Analytical Decoder BLER Ratio', fontsize=13)
ax.set_title('Neural Decoder Scaling: Where Walls Appear\n(ratio < 1 = NPD wins, ratio > 1.5 = wall)', fontsize=13)
ax.legend(fontsize=10, loc='upper left')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xticks([16, 32, 64, 128, 256, 512])
ax.set_xticklabels(['16', '32', '64', '128', '256', '512'])
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0.1, 15)

plt.tight_layout()
out_dir = os.path.join(_ROOT, 'project_summary', 'plots')
for ext in ['png', 'pdf']:
    path = os.path.join(out_dir, f'fig_wall_diagnostics.{ext}')
    fig.savefig(path, dpi=150)
    print(f'Saved: {path}')
plt.close()
