#!/usr/bin/env python3
"""
Plot proper per-position MI from soft SC posteriors.
Three rows: MI_U, MI_V, MI_U+MI_V. Columns: N values.
One figure per (channel, class).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, 'soft_mi_data.json')

CAPACITIES = {
    'gmac': {'I_XZ': 0.465, 'I_YZX': 0.912, 'I_XYZ': 1.376, 'label': 'GMAC (SNR=6dB)'},
    'bemac': {'I_XZ': 0.500, 'I_YZX': 1.000, 'I_XYZ': 1.500, 'label': 'BEMAC'},
    'abnmac': {'I_XZ': 0.400, 'I_YZX': 0.800, 'I_XYZ': 1.200, 'label': 'ABNMAC'},
}

CLASS_NAMES = {
    'B': 'Class B (interleaved)',
    'C': 'Class C (U first, then V)',
}

with open(DATA_PATH) as f:
    data = json.load(f)

for channel in ['gmac', 'bemac', 'abnmac']:
    for cls in ['B', 'C']:
        cap = CAPACITIES[channel]
        ch_data = data[channel][cls]
        N_values = sorted([int(k) for k in ch_data.keys()])

        fig, axes = plt.subplots(3, len(N_values), figsize=(5 * len(N_values), 9),
                                 gridspec_kw={'hspace': 0.3, 'wspace': 0.25})
        if len(N_values) == 1:
            axes = axes.reshape(3, 1)

        for col, N in enumerate(N_values):
            d = ch_data[str(N)]
            mi_u = np.array(d['mi_u'])
            mi_v = np.array(d['mi_v'])
            mi_sum = mi_u + mi_v
            pos = np.arange(1, N + 1)

            # U
            ax = axes[0, col]
            ax.scatter(pos, mi_u, c='#2166ac', s=max(3, 20 - N // 20),
                       alpha=0.7, edgecolors='none')
            ax.axhline(y=cap['I_XZ'], color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'N={N}', fontsize=11)
            if col == 0:
                ax.set_ylabel('MI_U (bits)', fontsize=10)
            n_high = np.sum(mi_u > 0.5)
            ax.text(0.97, 0.95, f'>{0.5}: {n_high}/{N}', transform=ax.transAxes,
                    fontsize=8, ha='right', va='top', color='#2166ac')
            ax.grid(True, alpha=0.15)

            # V
            ax = axes[1, col]
            ax.scatter(pos, mi_v, c='#1b7837', s=max(3, 20 - N // 20),
                       alpha=0.7, edgecolors='none')
            ax.axhline(y=cap['I_YZX'], color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_ylim(-0.05, 1.05)
            if col == 0:
                ax.set_ylabel('MI_V (bits)', fontsize=10)
            n_high = np.sum(mi_v > 0.5)
            ax.text(0.97, 0.95, f'>{0.5}: {n_high}/{N}', transform=ax.transAxes,
                    fontsize=8, ha='right', va='top', color='#1b7837')
            ax.grid(True, alpha=0.15)

            # Sum
            ax = axes[2, col]
            ax.scatter(pos, mi_sum, c='#762a83', s=max(3, 20 - N // 20),
                       alpha=0.7, edgecolors='none')
            ax.axhline(y=cap['I_XYZ'], color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_ylim(-0.1, 2.1)
            ax.set_xlabel('Position', fontsize=9)
            if col == 0:
                ax.set_ylabel('MI_U + MI_V (bits)', fontsize=10)
            ax.text(0.97, 0.95, f'avg={mi_sum.mean():.3f}',
                    transform=ax.transAxes, fontsize=8, ha='right', va='top',
                    color='#762a83')
            ax.grid(True, alpha=0.15)

        fig.suptitle(
            f'{cap["label"]} - {CLASS_NAMES[cls]}\n'
            f'Per-position MI from soft SC posteriors (genie-aided)\n'
            f'I(X;Z)={cap["I_XZ"]:.3f}  I(Y;Z|X)={cap["I_YZX"]:.3f}  '
            f'I(X,Y;Z)={cap["I_XYZ"]:.3f}  (red dashed)',
            fontsize=12, y=0.99)

        path = os.path.join(HERE, f'mi_soft_{channel}_{cls}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')

print('\nDone.')
