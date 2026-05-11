#!/usr/bin/env python3
"""Generate updated ISI-MAC figure from chained NPD results (today's run)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, json

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.6,
    'lines.markersize': 8,
    'figure.dpi': 150,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    print(f'  Saved {name}.png / .pdf')


def main():
    # Chained NPD (BiGRU encoder) + trellis SC baseline from today's results
    # Source: class_c_npd/results/npd_memory_mac/isi_mac_bigru_results.json (+N16 file)
    N_arr = np.array([16, 32, 64])
    sc_trellis = np.array([0.136, 0.072, 0.028])
    npd_bigru = np.array([0.1465, 0.1115, 0.0425])
    # window encoder results too
    npd_window = np.array([0.1325, 0.078, 0.0695])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(N_arr, sc_trellis, 'o-', color='#1f77b4', label='Trellis-SC (channel-aware)')
    ax.plot(N_arr, npd_bigru, 's-', color='#d62728', label='Neural NPD (BiGRU, chained)')
    ax.plot(N_arr, npd_window, '^-', color='#2ca02c', label='Neural NPD (window w=2, chained)')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Block length N')
    ax.set_ylabel('Total BLER')
    ax.set_title('ISI-MAC ($h=0.3$, SNR=6dB): Chained NPD vs Trellis-SC')
    ax.set_xticks(N_arr)
    ax.set_xticklabels([str(n) for n in N_arr])
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')

    fig.tight_layout()
    save(fig, 'fig_isi_mac_chained_npd')
    plt.close(fig)


if __name__ == '__main__':
    main()
