#!/usr/bin/env python3
"""plot_n256_wall_closed.py — single bar chart showing the N=256 progression:
  main NCG → better checkpoint → CRC-ensemble → campaign_sched + SCL L=4.
"""

import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'legend.fontsize': 9,
    'figure.dpi': 120, 'savefig.dpi': 300,
    'lines.linewidth': 1.8, 'lines.markersize': 8,
})


def main():
    # Pull data
    p_all = os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N256_all_checkpoints.json')
    with open(p_all) as f: all_ck = json.load(f)

    p_ens = os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N256_crc_ensemble.json')
    with open(p_ens) as f: ens = json.load(f)

    p_camp = os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N256_campaign_crc_scl.json')
    camp = None
    if os.path.exists(p_camp):
        with open(p_camp) as f: camp = json.load(f)

    p_hicw = os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N256_campaign_crc_scl_hicw.json')
    hicw = None
    if os.path.exists(p_hicw):
        with open(p_hicw) as f: hicw = json.load(f)

    labels = []
    values = []
    colors = []

    # 1. Main model (greedy)
    labels.append('main\nNCG (greedy)')
    values.append(all_ck['ncg_gmac_mlp_N256.pt']['bler'])
    colors.append('#777777')

    # 2. Better checkpoint (greedy)
    labels.append('campaign_sched\nNCG (greedy)')
    values.append(all_ck['campaign_n256_sched_best.pt']['bler'])
    colors.append('#1F77B4')

    # 3. CRC-aided ensemble
    labels.append('3-model CRC\nensemble (greedy)')
    values.append(ens['ensemble_bler'])
    colors.append('#2CA02C')

    # 4. campaign_sched + SCL L=4 (low-cw scan)
    if camp and 'L4' in camp:
        labels.append('campaign_sched\n+ SCL L=4\n(300 cw)')
        values.append(camp['L4']['bler_scl'])
        colors.append('#D62728')

    # 5. campaign_sched + SCL L=4 (high-cw confirm)
    if hicw and 'L4_hicw' in hicw:
        labels.append('campaign_sched\n+ SCL L=4\n(3000 cw)')
        values.append(hicw['L4_hicw']['bler_scl'])
        colors.append('#FF7F0E')

    # SC reference
    SC_REF = 0.006

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    xs = list(range(len(values)))
    bars = ax.bar(xs, values, color=colors, alpha=0.85)
    ax.axhline(SC_REF, color='black', linestyle=':', linewidth=1.5,
               label=f'Analytical SC = {SC_REF}')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel('BLER')
    ax.set_title('N=256 GMAC Class B — closing the wall without retraining')
    ax.grid(True, axis='y', which='both', alpha=0.3)
    ax.legend(loc='upper right')

    for i, v in enumerate(values):
        ax.text(i, v * 1.08, f'{v:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    base = os.path.join(BASE, 'docs', 'paper_figures', 'fig_n256_wall_closed')
    fig.savefig(base + '.png'); fig.savefig(base + '.pdf')
    plt.close(fig)
    print(f'  wrote {base}.png / .pdf')


if __name__ == '__main__':
    main()
