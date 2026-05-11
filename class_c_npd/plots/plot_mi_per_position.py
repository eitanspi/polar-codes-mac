#!/usr/bin/env python3
"""
Per-position MI plots for MAC polar codes.

For each (channel, class, N):
  - MI_U = 1 - Pe_U(i) per position i
  - MI_V = 1 - Pe_V(i) per position i
  - MI_sum = MI_U + MI_V

MI here is the genie-aided SC reliability metric: MI_i = 1 - Pe_i,
where Pe_i is the bit error probability at position i under genie-aided
(teacher-forced) SC decoding. This is the standard metric for polar code
design and shows how polarization distributes channel capacity across
positions.

Conservation: for a perfectly polarized code (N -> inf), the fraction
of positions with MI -> 1 equals the channel capacity per user.
At finite N, many positions are "in between" and the average MI_sum
exceeds I(X,Y;Z) because partially-polarized positions still contribute.

Outputs: one PNG per (channel, class) with subplots for N=32,64,128,256.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'designs')
OUT_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

# Channel capacities (bits per channel use)
CAPACITIES = {
    'gmac': {'I_XZ': 0.465, 'I_YZX': 0.912, 'I_XYZ': 1.376, 'label': 'GMAC (SNR=6dB)'},
    'bemac': {'I_XZ': 0.500, 'I_YZX': 1.000, 'I_XYZ': 1.500, 'label': 'BEMAC'},
    'abnmac': {'I_XZ': 0.400, 'I_YZX': 0.800, 'I_XYZ': 1.200, 'label': 'ABNMAC'},
}

N_VALUES = [32, 64, 128, 256]


def load_design(channel, cls, n):
    """Load design file, return (pe_u, pe_v) arrays."""
    if channel == 'gmac':
        path = os.path.join(DESIGNS_DIR, f'{channel}_{cls}_n{n}_snr6dB.npz')
    else:
        path = os.path.join(DESIGNS_DIR, f'{channel}_{cls}_n{n}.npz')
    if not os.path.exists(path):
        return None, None
    d = np.load(path)
    return d['u_error_rates'], d['v_error_rates']


def plot_channel_class(channel, cls, ax_rows, N_values):
    """Plot MI for one (channel, class) combination across N values."""
    cap = CAPACITIES[channel]

    for col, N in enumerate(N_values):
        n = int(np.log2(N))
        pe_u, pe_v = load_design(channel, cls, n)
        if pe_u is None:
            for row in range(3):
                ax_rows[row][col].text(0.5, 0.5, f'N={N}\nno data',
                                        ha='center', va='center',
                                        transform=ax_rows[row][col].transAxes,
                                        fontsize=10, color='gray')
                ax_rows[row][col].set_xlim(0, 10)
                ax_rows[row][col].set_ylim(-0.05, 1.1)
            continue

        mi_u = 1.0 - pe_u
        mi_v = 1.0 - pe_v
        mi_sum = mi_u + mi_v
        positions = np.arange(1, N + 1)

        # Count info-worthy positions (MI > 0.5)
        n_u = np.sum(mi_u > 0.5)
        n_v = np.sum(mi_v > 0.5)

        # U plot
        ax = ax_rows[0][col]
        ax.scatter(positions, mi_u, c='#2166ac', s=8, alpha=0.7, edgecolors='none')
        ax.axhline(y=cap['I_XZ'], color='red', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_ylim(-0.05, 1.1)
        ax.set_title(f'N={N}', fontsize=11)
        if col == 0:
            ax.set_ylabel('MI_U', fontsize=10)
        ax.text(0.97, 0.05, f'{n_u}/{N}', transform=ax.transAxes,
                fontsize=8, ha='right', color='#2166ac')
        ax.grid(True, alpha=0.15)

        # V plot
        ax = ax_rows[1][col]
        ax.scatter(positions, mi_v, c='#1b7837', s=8, alpha=0.7, edgecolors='none')
        ax.axhline(y=cap['I_YZX'], color='red', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_ylim(-0.05, 1.1)
        if col == 0:
            ax.set_ylabel('MI_V', fontsize=10)
        ax.text(0.97, 0.05, f'{n_v}/{N}', transform=ax.transAxes,
                fontsize=8, ha='right', color='#1b7837')
        ax.grid(True, alpha=0.15)

        # Sum plot
        ax = ax_rows[2][col]
        ax.scatter(positions, mi_sum, c='#762a83', s=8, alpha=0.7, edgecolors='none')
        ax.axhline(y=cap['I_XYZ'], color='red', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_ylim(-0.1, 2.15)
        ax.set_xlabel('Position', fontsize=9)
        if col == 0:
            ax.set_ylabel('MI_U + MI_V', fontsize=10)
        avg_sum = mi_sum.mean()
        ax.text(0.97, 0.05, f'avg={avg_sum:.2f}', transform=ax.transAxes,
                fontsize=8, ha='right', color='#762a83')
        ax.grid(True, alpha=0.15)


def make_plot(channel, cls):
    """Create one figure for (channel, class)."""
    cap = CAPACITIES[channel]
    fig, axes = plt.subplots(3, len(N_VALUES), figsize=(16, 8),
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.25})

    # Ensure axes is 2D
    if len(N_VALUES) == 1:
        axes = axes.reshape(3, 1)

    ax_rows = [axes[0], axes[1], axes[2]]
    plot_channel_class(channel, cls, ax_rows, N_VALUES)

    class_names = {'A': 'Class A (V first)', 'B': 'Class B (interleaved)',
                   'C': 'Class C (U first)'}
    fig.suptitle(f'{cap["label"]} — {class_names[cls]}\n'
                 f'Per-position MI (1 - Pe under genie-aided SC)\n'
                 f'I(X;Z)={cap["I_XZ"]:.3f}  I(Y;Z|X)={cap["I_YZX"]:.3f}  '
                 f'I(X,Y;Z)={cap["I_XYZ"]:.3f}',
                 fontsize=13, y=0.99)

    # Red dashed line legend
    axes[0, 0].plot([], [], 'r--', alpha=0.4, label='Channel capacity')
    axes[0, 0].legend(fontsize=8, loc='upper left')

    path = os.path.join(OUT_DIR, f'mi_per_pos_{channel}_{cls}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    for channel in ['gmac', 'bemac', 'abnmac']:
        for cls in ['B', 'C']:
            make_plot(channel, cls)
    print('\nAll plots saved to class_c_npd/plots/')
