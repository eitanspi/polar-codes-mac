#!/usr/bin/env python3
"""
plot_multi_snr_waterfall.py — IEEE-style waterfall plot for multi-SNR data.

Reads results/multi_snr/gmac_N128_waterfall.json and produces:
  docs/paper_figures/fig_multi_snr_waterfall.{png,pdf}
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'legend.fontsize': 9,
    'figure.dpi': 120, 'savefig.dpi': 300,
    'lines.linewidth': 1.8, 'lines.markersize': 7,
})


def load(path):
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


def main():
    d64 = load(os.path.join(BASE, 'results', 'multi_snr', 'gmac_N64_waterfall.json'))
    d128 = load(os.path.join(BASE, 'results', 'multi_snr', 'gmac_N128_waterfall.json'))
    if not d64 and not d128:
        print('  [skip] no waterfall data'); return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {64: '#1F77B4', 128: '#D62728'}
    for d, N in ((d64, 64), (d128, 128)):
        if d is None: continue
        snr_keys = sorted(int(k) for k in d['snr_results'].keys())
        sc = [d['snr_results'][str(k)]['sc_bler'] for k in snr_keys]
        nn = [d['snr_results'][str(k)]['ncg_bler'] for k in snr_keys]
        col = colors[N]
        ax.plot(snr_keys, [max(x, 1e-5) for x in sc], color=col, marker='o',
                linestyle='--', alpha=0.7, label=f'SC  N={N}')
        ax.plot(snr_keys, [max(x, 1e-5) for x in nn], color=col, marker='s',
                linestyle='-', label=f'NCG N={N} (ours)')
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BLER')
    ax.set_title('Multi-SNR waterfall — GMAC Class B (6 dB design, no retraining)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()

    base = os.path.join(BASE, 'docs', 'paper_figures', 'fig_multi_snr_waterfall')
    fig.savefig(base + '.png'); fig.savefig(base + '.pdf')
    plt.close(fig)
    print(f'  wrote {base}.png / .pdf')


if __name__ == '__main__':
    main()
