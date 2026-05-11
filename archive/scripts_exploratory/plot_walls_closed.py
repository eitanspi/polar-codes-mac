#!/usr/bin/env python3
"""plot_walls_closed.py — combined N=256 + N=512 wall-closing visualization."""

import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'legend.fontsize': 9,
    'figure.dpi': 120, 'savefig.dpi': 300,
    'lines.linewidth': 1.8, 'lines.markersize': 8,
})

SC_REFS = {256: 0.006, 512: 0.001}


def load(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)


def main():
    all_n256 = load(os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N256_all_checkpoints.json'))
    ens_n256 = load(os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N256_crc_ensemble.json'))

    all_n512 = load(os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N512_all_checkpoints.json'))
    ens_n512 = load(os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N512_crc_ensemble.json'))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, N, allc, ens in ((axes[0], 256, all_n256, ens_n256),
                              (axes[1], 512, all_n512, ens_n512)):
        bars = []
        labels = []
        for k, v in allc.items():
            if not isinstance(v, dict) or 'bler' not in v: continue
            if v['bler'] >= 0.99: continue  # incompatible
            short = k.replace('.pt', '').replace('_best', '').replace('ncg_gmac_mlp_', 'main ').replace('campaign_n' + str(N) + '_sched', 'campaign_sched').replace('campaign_n' + str(N), 'campaign').replace('n' + str(N) + '_long', 'long').replace('_', ' ')
            bars.append((short, v['bler']))

        if ens is not None:
            bars.append(('CRC-aided\nensemble (ours)', ens['ensemble_bler']))

        bars.sort(key=lambda x: x[1], reverse=True)
        xs = list(range(len(bars)))
        ys = [v for _, v in bars]
        cols = ['#999999'] * (len(bars) - 1) + ['#D62728']  # ensemble in red
        for i, (label, v) in enumerate(bars):
            if 'ensemble' in label:
                cols[i] = '#D62728'

        rects = ax.bar(xs, ys, color=cols, alpha=0.85)
        for i, (label, v) in enumerate(bars):
            ax.text(i, v * 1.08, f'{v:.3f}', ha='center', fontsize=8)

        ax.axhline(SC_REFS[N], color='black', linestyle=':', linewidth=1.3,
                   label=f'SC = {SC_REFS[N]}')
        ax.set_xticks(xs)
        ax.set_xticklabels([l for l, _ in bars], rotation=20, ha='right', fontsize=8)
        ax.set_yscale('log')
        ax.set_ylabel('BLER')
        ax.set_title(f'N={N} GMAC Class B')
        ax.grid(True, axis='y', which='both', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('Closing the N=256 and N=512 "walls" without retraining',
                 fontsize=12)
    plt.tight_layout()
    base = os.path.join(BASE, 'docs', 'paper_figures', 'fig_walls_closed')
    fig.savefig(base + '.png'); fig.savefig(base + '.pdf')
    plt.close(fig)
    print(f'  wrote {base}.png / .pdf')


if __name__ == '__main__':
    main()
