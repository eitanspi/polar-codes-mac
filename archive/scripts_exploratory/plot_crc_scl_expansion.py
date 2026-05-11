#!/usr/bin/env python3
"""
plot_crc_scl_expansion.py — IEEE-style plots for CRC-SCL expansion.

Reads JSON results from results/crc_scl_expansion/ (and the pre-existing
results/gmac_snr6dB/crc_aided_nn_scl.json) and produces:

  - fig_crc_scl_{channel}_L_sweep.{png,pdf}   BLER vs L, one line per N
  - fig_crc_scl_summary_vs_N.{png,pdf}        headline figure

All plots: serif, font 11, log-scale y, solid lines with markers, legend with
"(ours)" on neural methods.
"""

import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
})

CHAN_LABEL = {'gmac': 'Gaussian MAC (6 dB)', 'bemac': 'BE-MAC', 'abnmac': 'ABN-MAC'}
CHAN_COLOR = {'gmac': '#D62728', 'bemac': '#1F77B4', 'abnmac': '#2CA02C'}

# SC references (from project_summary/results/*.csv)
SC_REF = {
    'gmac':   {32: 0.047, 64: 0.028, 128: 0.020, 256: 0.006, 512: 0.001},
    'bemac':  {32: 0.008, 64: 0.006, 128: 0.002, 256: 8e-5, 512: 1e-5},
    'abnmac': {32: None,  64: None,  128: None},  # fill from session data
}
NCG_REF = {
    'gmac':   {32: 0.040, 64: 0.026, 128: 0.023, 256: 0.012, 512: 0.010},
    'bemac':  {32: 0.009, 64: 0.003, 128: 0.001, 256: 4e-5, 512: 1e-5},
    'abnmac': {32: None,  64: None,  128: None},
}


def load_channel_results(channel):
    """Unify pre-existing and new-session results for a channel."""
    results = {}  # N -> {L: {bler_scl, bler_crc_scl}}
    if channel == 'gmac':
        p = os.path.join(BASE, 'results', 'gmac_snr6dB', 'crc_aided_nn_scl.json')
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            for N_str, v in data.items():
                N = int(N_str)
                results.setdefault(N, {})
                for k, r in v.items():
                    if not isinstance(r, dict) or 'bler' not in r: continue
                    if k.startswith('NN_SCL_L'):
                        L = int(k.replace('NN_SCL_L',''))
                        results[N].setdefault(L, {})['bler_scl'] = r['bler']
                    elif k.startswith('NN_CA_SCL_L'):
                        L = int(k.replace('NN_CA_SCL_L',''))
                        results[N].setdefault(L, {})['bler_crc_scl'] = r['bler']

    # New session file(s): main + optional _N256 expansion
    candidates = [f'{channel}_classB_crc_scl.json',
                  f'{channel}_classB_crc_scl_N256.json',
                  f'{channel}_classB_crc_scl_N512.json',
                  f'{channel}_classB_crc_scl_N1024.json']
    for fname in candidates:
        p = os.path.join(BASE, 'results', 'crc_scl_expansion', fname)
        if not os.path.exists(p): continue
        with open(p) as f:
            data = json.load(f)
        for N_str, vN in data.items():
            if N_str == 'N' or not N_str.isdigit(): continue
            N = int(N_str)
            results.setdefault(N, {})
            for k, r in vN.items():
                if not k.startswith('L'): continue
                if not isinstance(r, dict): continue
                L = int(k[1:])
                results[N].setdefault(L, {})
                if 'bler_scl' in r:
                    results[N][L]['bler_scl'] = r['bler_scl']
                if 'bler_crc_scl' in r:
                    results[N][L]['bler_crc_scl'] = r['bler_crc_scl']
    return results


def plot_channel_L_sweep(channel, results, out_dir):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    N_markers = {32: 'o', 64: 's', 128: '^', 256: 'd'}
    N_cols    = {32: '#1F77B4', 64: '#FF7F0E', 128: '#D62728', 256: '#7F7F7F'}
    any_plotted = False
    for N in sorted(results.keys()):
        Ls = sorted(results[N].keys())
        scl_x, scl_y, crc_x, crc_y = [], [], [], []
        for L in Ls:
            r = results[N][L]
            if 'bler_scl' in r and r['bler_scl'] is not None:
                scl_x.append(L); scl_y.append(max(r['bler_scl'], 1e-5))
            if 'bler_crc_scl' in r and r['bler_crc_scl'] is not None:
                crc_x.append(L); crc_y.append(max(r['bler_crc_scl'], 1e-5))
        col = N_cols.get(N, 'k')
        mk = N_markers.get(N, 'o')
        if scl_y:
            ax.plot(scl_x, scl_y, color=col, marker=mk, linestyle='--',
                    label=f'NN-SCL N={N} (ours)', alpha=0.7)
            any_plotted = True
        if crc_y:
            ax.plot(crc_x, crc_y, color=col, marker=mk, linestyle='-',
                    label=f'NN-CA-SCL N={N} (ours)')
            any_plotted = True
        # SC horizontal line
        sc = SC_REF.get(channel, {}).get(N)
        if sc is not None:
            ax.axhline(sc, color=col, linestyle=':', alpha=0.4,
                       label=f'SC N={N}')

    if not any_plotted:
        plt.close(fig)
        return False
    ax.set_yscale('log')
    ax.set_xlabel('list size L')
    ax.set_ylabel('BLER')
    ax.set_title(f'CRC-aided Neural SCL — {CHAN_LABEL[channel]}, Class B')
    ax.set_xticks([4, 8, 16])
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', ncol=1)
    plt.tight_layout()

    base = os.path.join(out_dir, f'fig_crc_scl_{channel}_L_sweep')
    fig.savefig(base + '.png'); fig.savefig(base + '.pdf')
    plt.close(fig)
    print(f'  wrote {base}.png / .pdf')
    return True


def plot_headline_vs_N(all_results, out_dir):
    """Summary: best CRC-SCL BLER per (channel, N) vs plain SC / NCG."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for channel in ['gmac', 'bemac', 'abnmac']:
        res = all_results.get(channel, {})
        if not res: continue
        Ns = sorted(res.keys())
        sc_y = [SC_REF[channel].get(N) for N in Ns]
        ncg_y = [NCG_REF[channel].get(N) for N in Ns]
        crc_y = []
        for N in Ns:
            best = None
            for L, r in res[N].items():
                if 'bler_crc_scl' in r and r['bler_crc_scl'] is not None:
                    bv = r['bler_crc_scl']
                    if best is None or bv < best:
                        best = bv
            crc_y.append(max(best, 1e-5) if best is not None else None)

        col = CHAN_COLOR[channel]
        if any(v is not None for v in sc_y):
            xs = [N for N, v in zip(Ns, sc_y) if v is not None]
            ys = [v for v in sc_y if v is not None]
            ax.plot(xs, ys, color=col, marker='o', linestyle=':',
                    label=f'SC {CHAN_LABEL[channel]}', alpha=0.6)
        if any(v is not None for v in ncg_y):
            xs = [N for N, v in zip(Ns, ncg_y) if v is not None]
            ys = [v for v in ncg_y if v is not None]
            ax.plot(xs, ys, color=col, marker='s', linestyle='--',
                    label=f'NCG {CHAN_LABEL[channel]} (ours)', alpha=0.8)
        if any(v is not None for v in crc_y):
            xs = [N for N, v in zip(Ns, crc_y) if v is not None]
            ys = [v for v in crc_y if v is not None]
            ax.plot(xs, ys, color=col, marker='^', linestyle='-',
                    label=f'NN-CA-SCL {CHAN_LABEL[channel]} (ours)')

    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('block length N')
    ax.set_ylabel('BLER')
    ax.set_title('CRC-aided Neural SCL across channels — headline')
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(['16', '32', '64', '128', '256', '512', '1024'])
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right', ncol=1, fontsize=8)
    plt.tight_layout()

    base = os.path.join(out_dir, 'fig_crc_scl_summary_vs_N')
    fig.savefig(base + '.png'); fig.savefig(base + '.pdf')
    plt.close(fig)
    print(f'  wrote {base}.png / .pdf')


def main():
    out_dir = os.path.join(BASE, 'docs', 'paper_figures')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for channel in ['gmac', 'bemac', 'abnmac']:
        all_results[channel] = load_channel_results(channel)
        ok = plot_channel_L_sweep(channel, all_results[channel], out_dir)
        if not ok:
            print(f'  [skip] {channel} L-sweep plot (no data)')

    plot_headline_vs_N(all_results, out_dir)


if __name__ == '__main__':
    main()
