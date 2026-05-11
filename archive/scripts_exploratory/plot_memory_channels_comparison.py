#!/usr/bin/env python3
"""
Plot memory channels comparison: NPD vs baselines across ISI, MA-AGN, Ising.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: ISI-MAC
ax = axes[0]
N_vals = [16, 32, 64, 128, 256]
chained_trellis = [0.169, 0.082, 0.041, 0.022, 0.006]
npd_best = [0.143, 0.081, 0.032, 0.030, 0.011]  # mix of best configs
memoryless = [0.185, 0.114, 0.088, 0.095, None]

ax.semilogy(N_vals, chained_trellis, 'k--^', label='Chained Trellis SC', ms=6, lw=1.5)
ax.semilogy(N_vals, npd_best, 'r-o', label='Chained NPD (best)', ms=7, lw=2)
N_m = [n for n, v in zip(N_vals, memoryless) if v is not None]
v_m = [v for v in memoryless if v is not None]
ax.semilogy(N_m, v_m, 'gray', ls=':', marker='x', label='Memoryless SC', ms=6, lw=1)

ax.set_xlabel('Block Length N', fontsize=11)
ax.set_ylabel('BLER', fontsize=11)
ax.set_title('ISI-MAC (h=0.3, SNR=6dB)', fontsize=12)
ax.legend(fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xticks(N_vals)
ax.set_xticklabels([str(n) for n in N_vals])
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0.003, 0.3)

# Panel 2: MA-AGN MAC
ax = axes[1]
N_vals_ma = [16, 32, 64]
memless_ma = [0.175, 0.077, 0.028]
npd_ma = [0.138, 0.112, 0.066]

ax.semilogy(N_vals_ma, memless_ma, 'k--^', label='Memoryless SC', ms=6, lw=1.5)
ax.semilogy(N_vals_ma, npd_ma, 'r-o', label='Chained NPD (d=32 h=128)', ms=7, lw=2)

# Add region marker
ax.axhline(y=0.138, color='green', ls=':', alpha=0.5)
ax.annotate('NPD beats SC', xy=(16, 0.138), fontsize=8, color='green')
ax.annotate('NPD worse than SC', xy=(32, 0.112), fontsize=8, color='red')

ax.set_xlabel('Block Length N', fontsize=11)
ax.set_ylabel('BLER', fontsize=11)
ax.set_title('MA-AGN MAC (alpha=0.3, SNR=6dB)\nNo trellis exists', fontsize=12)
ax.legend(fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xticks(N_vals_ma)
ax.set_xticklabels([str(n) for n in N_vals_ma])
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0.01, 0.3)

# Panel 3: Ising MAC
ax = axes[2]
N_vals_is = [16, 32]
trellis_is = [0.575, 0.689]
memless_is = [0.634, 0.781]

ax.bar(np.arange(len(N_vals_is)) - 0.15, trellis_is, 0.3, label='Trellis SC (Markov FB)', color='steelblue')
ax.bar(np.arange(len(N_vals_is)) + 0.15, memless_is, 0.3, label='Memoryless SC', color='lightcoral')
ax.set_xticks(np.arange(len(N_vals_is)))
ax.set_xticklabels([f'N={n}' for n in N_vals_is])
ax.set_ylabel('BLER', fontsize=11)
ax.set_title('Ising MAC (p_flip=0.1)\nNPD training in progress', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.0)

plt.tight_layout()
out_dir = os.path.join(_ROOT, 'project_summary', 'plots')
for ext in ['png', 'pdf']:
    path = os.path.join(out_dir, f'fig_memory_channels_comparison.{ext}')
    fig.savefig(path, dpi=150)
    print(f'Saved: {path}')
plt.close()
