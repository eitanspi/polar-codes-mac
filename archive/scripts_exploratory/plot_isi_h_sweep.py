#!/usr/bin/env python3
"""Plot BLER vs h for ISI-MAC h-sweep results."""
import os, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

RESULT_PATH = os.path.join(_ROOT, 'results/snr_sweep/isi_mac_h_sweep_N32.json')
OUT_DIR = os.path.join(_ROOT, 'docs/paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    with open(RESULT_PATH) as f:
        data = json.load(f)
    h_list = data['config']['h_list']

    decoders = [
        ('chained_npd_bigru', 'Chained NPD BiGRU (trained @ h=0.3)', 'o', 'C0'),
        ('chained_trellis_sc', 'Chained trellis SC (aware of h)', 's', 'C1'),
        ('memoryless_sc', 'Memoryless SC (ignores ISI)', 'D', 'C2'),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label, marker, color in decoders:
        xs, ys, yerrs_lo, yerrs_hi = [], [], [], []
        for h in h_list:
            rec = data['results'][f'h={h}'][key]
            xs.append(h); ys.append(rec['bler'])
            yerrs_lo.append(rec['bler'] - rec['ci_lo'])
            yerrs_hi.append(rec['ci_hi'] - rec['bler'])
        ax.errorbar(xs, ys, yerr=[yerrs_lo, yerrs_hi], marker=marker,
                    label=label, color=color, capsize=3, lw=1.5, markersize=7)

    ax.set_xlabel('ISI tap coefficient h', fontsize=11)
    ax.set_ylabel('BLER', fontsize=11)
    ax.set_yscale('log')
    ax.set_title(f"ISI-MAC h-sweep (N={data['config']['N']}, SNR={data['config']['snr_db']} dB, "
                 f"ku={data['config']['ku']}, kv={data['config']['kv']}, "
                 f"{data['config']['n_cw']} codewords)",
                 fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=9)
    ax.axvline(x=0.3, color='gray', linestyle=':', alpha=0.5,
               label='_NPD training h')

    # Annotate NPD training h
    ax.annotate('NPD trained here',
                xy=(0.3, 1e-2), xytext=(0.35, 5e-2),
                fontsize=9, ha='left', color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    ax.set_xticks(h_list)
    ax.set_xticklabels([str(h) for h in h_list])

    plt.tight_layout()
    png = os.path.join(OUT_DIR, 'fig_isi_mac_h_sweep.png')
    pdf = os.path.join(OUT_DIR, 'fig_isi_mac_h_sweep.pdf')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    print(f'Saved: {png}')
    print(f'Saved: {pdf}')


if __name__ == '__main__':
    main()
