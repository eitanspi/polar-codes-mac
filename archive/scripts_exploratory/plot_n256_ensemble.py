#!/usr/bin/env python3
"""Plot N=256 ensemble/single-model comparison."""
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

SC_REF = 0.006


def main():
    p = os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N256_ensemble.json')
    if not os.path.exists(p):
        print(f"  [skip] {p}"); return
    with open(p) as f: d = json.load(f)

    singles = d.get('single_models', {})
    pairs = d.get('pairs', [])

    names = list(singles.keys())
    single_blers = [singles[n]['bler'] for n in names]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.arange(len(names) + len(pairs))
    all_vals = []
    labels = []

    # Singles
    for n in names:
        all_vals.append(singles[n]['bler'])
        labels.append(n.replace('.pt', '').replace('ncg_gmac_mlp_', 'main\n').replace('_best', '').replace('campaign_', 'campaign\n').replace('n256_', '').replace('_', ' '))

    # Pairs (oracle BLER)
    for pair in pairs:
        all_vals.append(pair['bler_oracle_either'])
        s1 = pair['m1'].replace('.pt','').replace('ncg_gmac_mlp_','main').replace('_best','').replace('campaign_n256_sched','campaign').replace('n256_','')
        s2 = pair['m2'].replace('.pt','').replace('ncg_gmac_mlp_','main').replace('_best','').replace('campaign_n256_sched','campaign').replace('n256_','')
        labels.append(f'oracle\n{s1}+{s2}')

    colors = ['#1F77B4'] * len(names) + ['#D62728'] * len(pairs)
    bars = ax.bar(xs, all_vals, color=colors, alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.axhline(SC_REF, color='black', linestyle=':', label=f'SC baseline = {SC_REF}')
    ax.set_yscale('log')
    ax.set_ylabel('BLER')
    ax.set_title('GMAC Class B N=256 — single models and oracle ensembles')
    ax.grid(True, axis='y', which='both', alpha=0.3)
    ax.legend(loc='upper right')

    # Annotate bars
    for i, v in enumerate(all_vals):
        ax.text(i, v * 1.05, f'{v:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    base = os.path.join(BASE, 'docs', 'paper_figures', 'fig_n256_ensemble')
    fig.savefig(base + '.png'); fig.savefig(base + '.pdf')
    plt.close(fig)
    print(f'  wrote {base}.png / .pdf')


if __name__ == '__main__':
    main()
