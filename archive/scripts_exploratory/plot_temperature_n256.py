#!/usr/bin/env python3
"""
plot_temperature_n256.py — IEEE-style plot for N=256 temperature sweep.

Reads results/crc_scl_expansion/gmac_N256_temperature.json and produces:
  docs/paper_figures/fig_n256_temperature.{png,pdf}
"""

import os, json
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

SC_REF_N256 = 0.006
NCG_REF_N256 = 0.023


def main():
    p = os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N256_temperature.json')
    if not os.path.exists(p):
        print(f'  [skip] {p} not found')
        return
    with open(p) as f:
        r = json.load(f)

    # 1000-cw first scan
    Ts_1k = []; blers_1k = []
    # 5000-cw extended scan
    Ts_5k = []; blers_5k = []
    for k, v in r.items():
        if not k.startswith('ncg_sc_T'): continue
        name = k.replace('ncg_sc_T', '')
        is_v2 = name.endswith('_v2')
        T = float(name.replace('_v2', ''))
        if is_v2:
            Ts_5k.append(T); blers_5k.append(max(v['bler'], 1e-5))
        else:
            Ts_1k.append(T); blers_1k.append(max(v['bler'], 1e-5))
    for xs, ys in ((Ts_1k, blers_1k), (Ts_5k, blers_5k)):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs[:] = [xs[i] for i in order]; ys[:] = [ys[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if Ts_1k:
        ax.plot(Ts_1k, blers_1k, color='#AAAAAA', marker='.', linestyle=':',
                alpha=0.7, label='NCG(T), 1000 cw (noisy)')
    if Ts_5k:
        ax.plot(Ts_5k, blers_5k, color='#D62728', marker='s', linestyle='-',
                label='NCG(T), 5000 cw (ours)')
    ax.axhline(SC_REF_N256, color='#1F77B4', linestyle=':', label='SC baseline 0.006')
    ax.axhline(NCG_REF_N256, color='#2CA02C', linestyle='--', alpha=0.5,
               label='NCG T=1 baseline 0.025')
    ax.set_yscale('log')
    ax.set_xlabel('softmax temperature T')
    ax.set_ylabel('BLER')
    ax.set_title('Temperature scaling at N=256 (GMAC Class B)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    base = os.path.join(BASE, 'docs', 'paper_figures', 'fig_n256_temperature')
    fig.savefig(base + '.png'); fig.savefig(base + '.pdf')
    plt.close(fig)
    print(f'  wrote {base}.png / .pdf')


if __name__ == '__main__':
    main()
