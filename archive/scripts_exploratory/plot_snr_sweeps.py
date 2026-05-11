#!/usr/bin/env python3
"""
plot_snr_sweeps.py — IEEE-style BLER-vs-SNR plots for the thesis sweeps.

Reads JSONs from results/snr_sweep/ and emits three figures to
docs/paper_figures/:
  - fig_snr_sweep_isi_mac.{png,pdf}
  - fig_snr_sweep_gmac_classB.{png,pdf}
  - fig_snr_sweep_gmac_classC.{png,pdf}
"""
from __future__ import annotations
import os
import sys
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
IN_DIR = os.path.join(_ROOT, 'results', 'snr_sweep')
OUT_DIR = os.path.join(_ROOT, 'docs', 'paper_figures')
os.makedirs(OUT_DIR, exist_ok=True)


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.dpi': 150,
})


def load_rec(name: str):
    fp = os.path.join(IN_DIR, name + '.json')
    if not os.path.exists(fp):
        return None
    with open(fp, 'r') as f:
        return json.load(f)


def rec_xy(rec):
    """Return (snrs, blers, ci_los, ci_his) as sorted arrays."""
    keys = sorted(rec['sweep'].keys(), key=lambda s: float(s))
    snrs = [float(k) for k in keys]
    blers = [rec['sweep'][k]['bler'] for k in keys]
    lo = [rec['sweep'][k]['ci_lo'] for k in keys]
    hi = [rec['sweep'][k]['ci_hi'] for k in keys]
    return np.array(snrs), np.array(blers), np.array(lo), np.array(hi)


def plot_with_ci(ax, snrs, blers, los, his, label, color, marker, linestyle='-'):
    # floor the lower CI for log-scale
    blers = np.clip(blers, 1e-5, 1.0)
    los = np.clip(los, 1e-5, 1.0)
    his = np.clip(his, 1e-5, 1.0)
    yerr = np.vstack([blers - los, his - blers])
    ax.errorbar(snrs, blers, yerr=yerr, label=label, color=color,
                marker=marker, linestyle=linestyle, capsize=2,
                elinewidth=0.8)


def save(fig, basename):
    fig.savefig(os.path.join(OUT_DIR, basename + '.png'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, basename + '.pdf'),
                bbox_inches='tight')
    print(f'  saved {basename}.png and .pdf')


# ─── Plot 1: ISI-MAC ───────────────────────────────────────────────────────

def plot_isi_mac():
    Ns = [16, 32, 64]
    colors_npd = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red
    colors_sc = ['#1f77b4', '#2ca02c', '#d62728']
    fig, ax = plt.subplots(figsize=(5.2, 3.7))

    for N, cnpd, csc in zip(Ns, colors_npd, colors_sc):
        r_npd = load_rec(f'chained_npd_isi_mac_N{N}')
        r_sc = load_rec(f'chained_trellis_sc_isi_mac_N{N}')
        if r_npd is not None:
            snrs, b, lo, hi = rec_xy(r_npd)
            plot_with_ci(ax, snrs, b, lo, hi,
                         f'NPD N={N}', cnpd, 'o', '-')
        if r_sc is not None:
            snrs, b, lo, hi = rec_xy(r_sc)
            plot_with_ci(ax, snrs, b, lo, hi,
                         f'trellis SC N={N}', csc, 's', '--')

    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BLER')
    ax.set_xlim(3.7, 8.3)
    ax.set_ylim(1e-2, 0.6)
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.set_title(r'Chained NPD vs chained trellis SC — ISI-MAC ($h=0.3$)')
    ax.legend(ncol=2, loc='lower left', frameon=True)
    fig.tight_layout()
    save(fig, 'fig_snr_sweep_isi_mac')
    plt.close(fig)


# ─── Plot 2: GMAC Class B ──────────────────────────────────────────────────

def plot_gmac_classB():
    Ns = [32, 64, 128]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    fig, ax = plt.subplots(figsize=(5.2, 3.7))

    for N, c in zip(Ns, colors):
        r_ncg = load_rec(f'ncg_gmac_classB_N{N}')
        r_sc = load_rec(f'sc_gmac_classB_N{N}')
        if r_ncg is not None:
            snrs, b, lo, hi = rec_xy(r_ncg)
            plot_with_ci(ax, snrs, b, lo, hi,
                         f'NCG N={N}', c, 'o', '-')
        if r_sc is not None:
            snrs, b, lo, hi = rec_xy(r_sc)
            plot_with_ci(ax, snrs, b, lo, hi,
                         f'SC N={N}', c, 's', '--')

    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BLER')
    ax.set_xlim(3.7, 8.3)
    ax.set_ylim(3e-3, 0.6)
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.set_title(r'NCG vs analytical SC — GMAC Class B')
    ax.legend(ncol=2, loc='lower left', frameon=True)
    fig.tight_layout()
    save(fig, 'fig_snr_sweep_gmac_classB')
    plt.close(fig)


# ─── Plot 3: GMAC Class C (corner rate) ────────────────────────────────────

def plot_gmac_classC():
    Ns = [64, 128]
    colors = ['#1f77b4', '#d62728']
    fig, ax = plt.subplots(figsize=(5.2, 3.7))

    any_plotted = False
    for N, c in zip(Ns, colors):
        r_npd = load_rec(f'chained_npd_gmac_classC_N{N}')
        r_sc = load_rec(f'sc_gmac_classC_N{N}')
        if r_npd is not None:
            snrs, b, lo, hi = rec_xy(r_npd)
            plot_with_ci(ax, snrs, b, lo, hi,
                         f'chained NPD N={N}', c, 'o', '-')
            any_plotted = True
        if r_sc is not None:
            snrs, b, lo, hi = rec_xy(r_sc)
            plot_with_ci(ax, snrs, b, lo, hi,
                         f'SC N={N}', c, 's', '--')
            any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print('  GMAC-C: no records, skipping plot')
        return
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BLER')
    ax.set_xlim(3.7, 8.3)
    ax.set_ylim(1e-2, 0.6)
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.set_title(r'Chained NPD vs analytical SC — GMAC Class C (corner rate)')
    ax.legend(ncol=2, loc='lower left', frameon=True)
    fig.tight_layout()
    save(fig, 'fig_snr_sweep_gmac_classC')
    plt.close(fig)


if __name__ == '__main__':
    print(f'Plotting from {IN_DIR}')
    print(f'Output dir: {OUT_DIR}')
    plot_isi_mac()
    plot_gmac_classB()
    plot_gmac_classC()
    print('Done.')
