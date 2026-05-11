#!/usr/bin/env python3
"""Generate Figures 5 (CRC-Aided NN-SCL) and 6 (ISI-MAC) for the paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── IEEE-style defaults ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
    'figure.dpi': 150,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    print(f'  Saved {name}.png / .pdf')


# =====================================================================
# Figure 5: CRC-Aided Neural SCL
# =====================================================================
def fig5_crc_aided():
    labels = [
        'N=32\nL=4',
        'N=64\nL=4', 'N=64\nL=8', 'N=64\nL=16',
        'N=128\nL=4', 'N=128\nL=8',
    ]
    nn_scl  = [0.023, 0.017, 0.008, 0.020, 0.014, 0.023]
    ca_scl  = [0.009, 0.002, 0.002, 0.003, 0.006, 0.001]  # 0.000 -> 0.001

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))

    bars1 = ax.bar(x - width/2, nn_scl, width, label='NN-SCL',
                   color='#2166ac', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ca_scl, width, label='NN-CA-SCL',
                   color='#b2182b', edgecolor='black', linewidth=0.5)

    ax.set_yscale('log')
    ax.set_ylabel('BLER')
    ax.set_xlabel('Configuration')
    ax.set_title('CRC-Aided Neural SCL: GMAC Class B, SNR=6dB')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, which='both')
    ax.set_ylim(5e-4, 5e-2)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.15,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.15,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    save(fig, 'fig5_crc_aided_nn_scl')
    plt.close(fig)


# =====================================================================
# Figure 6: ISI-MAC
# =====================================================================
def fig6_isi_mac():
    labels = ['N=32', 'N=64']
    nn_sc = [0.688, 0.466]
    sc    = [0.731, 0.575]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 4))

    bars1 = ax.bar(x - width/2, sc, width, label='Memoryless SC',
                   color='#bababa', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, nn_sc, width, label='Neural SC (ISI-aware)',
                   color='#2166ac', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('BLER')
    ax.set_xlabel('Block Length')
    ax.set_title(r'ISI-MAC ($h=0.3$): Neural vs Memoryless SC')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Percentage improvement annotations (arrows pointing down between bars)
    for i in range(len(labels)):
        improvement = (sc[i] - nn_sc[i]) / sc[i] * 100
        mid_x = x[i]
        arrow_top = sc[i] - 0.01
        arrow_bot = nn_sc[i] + 0.01
        # Draw arrow from SC bar down to NN bar
        ax.annotate('', xy=(mid_x, arrow_bot), xytext=(mid_x, arrow_top),
                    arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.5))
        # Label above the SC bar
        ax.text(mid_x, sc[i] + 0.06, f'$\\downarrow${improvement:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='#b2182b',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='#b2182b', alpha=0.9))

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    save(fig, 'fig6_isi_mac')
    plt.close(fig)


if __name__ == '__main__':
    print('Generating Figure 5: CRC-Aided NN-SCL...')
    fig5_crc_aided()
    print('Generating Figure 6: ISI-MAC...')
    fig6_isi_mac()
    print('Done.')
