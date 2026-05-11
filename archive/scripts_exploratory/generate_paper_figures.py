#!/usr/bin/env python3
"""
generate_paper_figures.py
=========================
Generate paper-style figures from baseline and NPD evaluation results.

Produces:
  1. fig_paper_isi_mac.{png,pdf} — ISI-MAC BLER vs N
  2. fig_paper_ising_mac.{png,pdf} — Ising MAC BLER vs N
  3. fig_paper_maagn_mac.{png,pdf} — MA-AGN MAC BLER vs N
  4. fig_paper_all_memory.{png,pdf} — 3-panel summary

IEEE two-column style, log y-axis, error bars.
"""
import sys, os, json, math
import numpy as np

os.environ['OMP_NUM_THREADS'] = '4'

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

RESULTS_DIR = os.path.join(_ROOT, 'results', 'paper_style')
FIGS_DIR = os.path.join(_ROOT, 'results', 'paper_style')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.figsize': (3.5, 3.0),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def wilson_ci(n_err, n_total, z=1.96):
    if n_total == 0:
        return (0.0, 1.0)
    p_hat = n_err / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f'  WARNING: {filename} not found')
        return None
    with open(path) as f:
        return json.load(f)


def extract_bler_series(data, key_format='N{}', n_values=[16,32,64,128,256,512,1024]):
    """Extract (N_list, bler_list, ci_low_list, ci_high_list) from results dict."""
    Ns, blers, ci_lows, ci_highs = [], [], [], []
    for N in n_values:
        key = key_format.format(N)
        if key in data and 'bler_total' in data[key]:
            r = data[key]
            Ns.append(N)
            blers.append(r['bler_total'])
            ci = r.get('wilson_95_ci', wilson_ci(r.get('errs_total', 0), r.get('n_cw', 1)))
            ci_lows.append(ci[0])
            ci_highs.append(ci[1])
    return np.array(Ns), np.array(blers), np.array(ci_lows), np.array(ci_highs)


def plot_with_errorbars(ax, Ns, blers, ci_low, ci_high, label, marker, color, ls='-'):
    """Plot BLER with asymmetric error bars."""
    yerr_low = np.maximum(blers - ci_low, 1e-6)
    yerr_high = np.maximum(ci_high - blers, 1e-6)
    # Don't plot zero BLER on log scale
    mask = blers > 0
    if mask.sum() == 0:
        return
    ax.errorbar(Ns[mask], blers[mask],
                yerr=[yerr_low[mask], yerr_high[mask]],
                label=label, marker=marker, color=color, linestyle=ls,
                capsize=3, capthick=1)


def format_bler_axis(ax, title, xlim=None):
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Block length $N$')
    ax.set_ylabel('BLER')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(['16', '32', '64', '128', '256', '512', '1024'])


# ============================================================================
#  Figure 1: ISI-MAC
# ============================================================================

def fig_isi_mac():
    print('Generating ISI-MAC figure...')

    # Load baselines
    trellis = load_json('isi_mac_sc_baselines.json')
    memless = load_json('isi_mac_memoryless_sc_baselines.json')
    npd_all = load_json('npd_all_channels_5kcw.json')

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))

    # Trellis SC
    if trellis:
        Ns, blers, cl, ch = extract_bler_series(trellis)
        plot_with_errorbars(ax, Ns, blers, cl, ch,
                           'Chained trellis SC', 's', 'C0')

    # Memoryless SC
    if memless:
        Ns, blers, cl, ch = extract_bler_series(memless)
        plot_with_errorbars(ax, Ns, blers, cl, ch,
                           'Memoryless SC', 'o', 'C1', ls='--')

    # NPD models (best per N)
    if npd_all and 'isi_mac' in npd_all:
        isi_npd = npd_all['isi_mac']
        # Collect best NPD per N
        best_per_N = {}
        for key, r in isi_npd.items():
            if 'error' in r or 'bler_total' not in r:
                continue
            N = r['N']
            if N not in best_per_N or r['bler_total'] < best_per_N[N]['bler_total']:
                best_per_N[N] = r

        if best_per_N:
            Ns = sorted(best_per_N.keys())
            blers = [best_per_N[N]['bler_total'] for N in Ns]
            cis = [best_per_N[N]['wilson_95_ci'] for N in Ns]
            cl = [c[0] for c in cis]
            ch = [c[1] for c in cis]
            plot_with_errorbars(ax, np.array(Ns), np.array(blers),
                               np.array(cl), np.array(ch),
                               'NPD (best)', 'D', 'C2')

    format_bler_axis(ax, 'ISI-MAC ($h=0.3$, SNR=6 dB)')
    ax.set_ylim(bottom=1e-4, top=1.0)

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGS_DIR, f'fig_paper_isi_mac.{ext}'))
    plt.close(fig)
    print('  Saved fig_paper_isi_mac.{png,pdf}')


# ============================================================================
#  Figure 2: Ising MAC
# ============================================================================

def fig_ising_mac():
    print('Generating Ising MAC figure...')

    ising = load_json('ising_mac_baselines.json')
    npd_all = load_json('npd_all_channels_5kcw.json')

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))

    n_values = [16, 32, 64]

    # Trellis SC
    if ising and 'trellis_sc' in ising:
        Ns, blers, cl, ch = extract_bler_series(ising['trellis_sc'], n_values=n_values)
        plot_with_errorbars(ax, Ns, blers, cl, ch,
                           'Chained trellis SC', 's', 'C0')

    # Memoryless SC
    if ising and 'memoryless_sc' in ising:
        Ns, blers, cl, ch = extract_bler_series(ising['memoryless_sc'], n_values=n_values)
        plot_with_errorbars(ax, Ns, blers, cl, ch,
                           'Memoryless SC', 'o', 'C1', ls='--')

    # NPD
    if npd_all and 'ising_mac' in npd_all:
        ising_npd = npd_all['ising_mac']
        best_per_N = {}
        for key, r in ising_npd.items():
            if 'error' in r or 'bler_total' not in r:
                continue
            N = r['N']
            if N not in best_per_N or r['bler_total'] < best_per_N[N]['bler_total']:
                best_per_N[N] = r

        if best_per_N:
            Ns = sorted(best_per_N.keys())
            blers = [best_per_N[N]['bler_total'] for N in Ns]
            cis = [best_per_N[N]['wilson_95_ci'] for N in Ns]
            cl = [c[0] for c in cis]
            ch = [c[1] for c in cis]
            plot_with_errorbars(ax, np.array(Ns), np.array(blers),
                               np.array(cl), np.array(ch),
                               'NPD $d$=16 $h$=100', 'D', 'C2')

    format_bler_axis(ax, 'Ising MAC ($p_{flip}=0.1$)', xlim=(12, 80))
    ax.set_xticks([16, 32, 64])
    ax.set_xticklabels(['16', '32', '64'])
    ax.set_ylim(bottom=1e-3, top=1.0)

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGS_DIR, f'fig_paper_ising_mac.{ext}'))
    plt.close(fig)
    print('  Saved fig_paper_ising_mac.{png,pdf}')


# ============================================================================
#  Figure 3: MA-AGN MAC
# ============================================================================

def fig_maagn_mac():
    print('Generating MA-AGN MAC figure...')

    maagn = load_json('maagn_mac_baselines.json')
    npd_all = load_json('npd_all_channels_5kcw.json')

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))

    n_values = [16, 32, 64, 128]

    # Memoryless SC
    if maagn:
        Ns, blers, cl, ch = extract_bler_series(maagn, n_values=n_values)
        plot_with_errorbars(ax, Ns, blers, cl, ch,
                           'Memoryless SC', 'o', 'C1', ls='--')

    # NPD
    if npd_all and 'maagn_mac' in npd_all:
        maagn_npd = npd_all['maagn_mac']
        best_per_N = {}
        for key, r in maagn_npd.items():
            if 'error' in r or 'bler_total' not in r:
                continue
            N = r['N']
            if N not in best_per_N or r['bler_total'] < best_per_N[N]['bler_total']:
                best_per_N[N] = r

        if best_per_N:
            Ns = sorted(best_per_N.keys())
            blers = [best_per_N[N]['bler_total'] for N in Ns]
            cis = [best_per_N[N]['wilson_95_ci'] for N in Ns]
            cl = [c[0] for c in cis]
            ch = [c[1] for c in cis]
            plot_with_errorbars(ax, np.array(Ns), np.array(blers),
                               np.array(cl), np.array(ch),
                               'NPD (best)', 'D', 'C2')

    format_bler_axis(ax, r'MA-AGN MAC ($\alpha=0.3$, SNR=6 dB)', xlim=(12, 200))
    ax.set_xticks([16, 32, 64, 128])
    ax.set_xticklabels(['16', '32', '64', '128'])
    ax.set_ylim(bottom=1e-3, top=1.0)

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGS_DIR, f'fig_paper_maagn_mac.{ext}'))
    plt.close(fig)
    print('  Saved fig_paper_maagn_mac.{png,pdf}')


# ============================================================================
#  Figure 4: 3-panel summary
# ============================================================================

def fig_all_memory():
    print('Generating 3-panel summary figure...')

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.8), constrained_layout=True)

    # Panel A: ISI-MAC
    ax = axes[0]
    trellis = load_json('isi_mac_sc_baselines.json')
    memless = load_json('isi_mac_memoryless_sc_baselines.json')
    npd_all = load_json('npd_all_channels_5kcw.json')

    if trellis:
        Ns, blers, cl, ch = extract_bler_series(trellis)
        plot_with_errorbars(ax, Ns, blers, cl, ch, 'Trellis SC', 's', 'C0')
    if memless:
        Ns, blers, cl, ch = extract_bler_series(memless)
        plot_with_errorbars(ax, Ns, blers, cl, ch, 'Memless SC', 'o', 'C1', ls='--')
    if npd_all and 'isi_mac' in npd_all:
        best = {}
        for k, r in npd_all['isi_mac'].items():
            if 'error' in r or 'bler_total' not in r: continue
            N = r['N']
            if N not in best or r['bler_total'] < best[N]['bler_total']:
                best[N] = r
        if best:
            Ns = sorted(best.keys())
            b = [best[N]['bler_total'] for N in Ns]
            c = [best[N]['wilson_95_ci'] for N in Ns]
            plot_with_errorbars(ax, np.array(Ns), np.array(b),
                               np.array([x[0] for x in c]), np.array([x[1] for x in c]),
                               'NPD', 'D', 'C2')
    format_bler_axis(ax, '(a) ISI-MAC')
    ax.set_ylim(1e-4, 1.0)

    # Panel B: Ising
    ax = axes[1]
    ising = load_json('ising_mac_baselines.json')
    if ising and 'trellis_sc' in ising:
        Ns, blers, cl, ch = extract_bler_series(ising['trellis_sc'], n_values=[16,32,64])
        plot_with_errorbars(ax, Ns, blers, cl, ch, 'Trellis SC', 's', 'C0')
    if ising and 'memoryless_sc' in ising:
        Ns, blers, cl, ch = extract_bler_series(ising['memoryless_sc'], n_values=[16,32,64])
        plot_with_errorbars(ax, Ns, blers, cl, ch, 'Memless SC', 'o', 'C1', ls='--')
    if npd_all and 'ising_mac' in npd_all:
        best = {}
        for k, r in npd_all['ising_mac'].items():
            if 'error' in r or 'bler_total' not in r: continue
            N = r['N']
            if N not in best or r['bler_total'] < best[N]['bler_total']:
                best[N] = r
        if best:
            Ns = sorted(best.keys())
            b = [best[N]['bler_total'] for N in Ns]
            c = [best[N]['wilson_95_ci'] for N in Ns]
            plot_with_errorbars(ax, np.array(Ns), np.array(b),
                               np.array([x[0] for x in c]), np.array([x[1] for x in c]),
                               'NPD', 'D', 'C2')
    format_bler_axis(ax, '(b) Ising MAC', xlim=(12, 80))
    ax.set_xticks([16, 32, 64])
    ax.set_xticklabels(['16', '32', '64'])
    ax.set_ylim(1e-3, 1.0)
    ax.set_ylabel('')

    # Panel C: MA-AGN
    ax = axes[2]
    maagn = load_json('maagn_mac_baselines.json')
    if maagn:
        Ns, blers, cl, ch = extract_bler_series(maagn, n_values=[16,32,64,128])
        plot_with_errorbars(ax, Ns, blers, cl, ch, 'Memless SC', 'o', 'C1', ls='--')
    if npd_all and 'maagn_mac' in npd_all:
        best = {}
        for k, r in npd_all['maagn_mac'].items():
            if 'error' in r or 'bler_total' not in r: continue
            N = r['N']
            if N not in best or r['bler_total'] < best[N]['bler_total']:
                best[N] = r
        if best:
            Ns = sorted(best.keys())
            b = [best[N]['bler_total'] for N in Ns]
            c = [best[N]['wilson_95_ci'] for N in Ns]
            plot_with_errorbars(ax, np.array(Ns), np.array(b),
                               np.array([x[0] for x in c]), np.array([x[1] for x in c]),
                               'NPD', 'D', 'C2')
    format_bler_axis(ax, r'(c) MA-AGN MAC', xlim=(12, 200))
    ax.set_xticks([16, 32, 64, 128])
    ax.set_xticklabels(['16', '32', '64', '128'])
    ax.set_ylim(1e-3, 1.0)
    ax.set_ylabel('')

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGS_DIR, f'fig_paper_all_memory.{ext}'))
    plt.close(fig)
    print('  Saved fig_paper_all_memory.{png,pdf}')


# ============================================================================

if __name__ == '__main__':
    fig_isi_mac()
    fig_ising_mac()
    fig_maagn_mac()
    fig_all_memory()
    print('\nAll figures generated.')
