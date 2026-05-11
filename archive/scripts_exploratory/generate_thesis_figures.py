#!/usr/bin/env python3
"""
generate_thesis_figures.py — Generate ALL thesis-ready figures.

IEEE style: serif 11pt, log-scale y-axis, solid lines + markers,
consistent colors across figures.

Outputs: docs/paper_figures/fig_*.{png,pdf}
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, 'docs', 'paper_figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ─── IEEE Style ───────────────────────────────────────────────────────────

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
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color scheme
C_SC = '#1f77b4'        # Blue - SC decoder
C_NCG = '#d62728'       # Red - NCG/NPD neural decoder
C_CRC_SCL = '#2ca02c'   # Green - CRC-SCL (analytical)
C_NN_SCL = '#ff7f0e'    # Orange - NN-SCL
C_NN_CRC = '#9467bd'    # Purple - NN-CRC-SCL
C_SCL = '#8c564b'       # Brown - SCL (no CRC)
C_MEMLESS = '#7f7f7f'   # Gray - Memoryless
C_TRELLIS = '#1f77b4'   # Blue - Trellis SC
C_NPD_D16 = '#d62728'   # Red - NPD d=16
C_NPD_D64 = '#ff7f0e'   # Orange - NPD d=64

MARKERS = {'SC': 's', 'NCG': 'o', 'CRC_SCL': '^', 'SCL': 'D',
           'NN_SCL': 'v', 'NN_CRC': 'P', 'MEMLESS': 'x', 'TRELLIS': 's',
           'NPD_D16': 'o', 'NPD_D64': 'D', 'NPD': 'o'}


def savefig(fig, name):
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'{name}.{ext}')
        fig.savefig(path)
        print(f'  Saved: {path}')
    plt.close(fig)


def bler_plot_setup(ax, title, xlabel='Block Length N', ylabel='BLER'):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(bottom=5e-5)


def add_line(ax, Ns, blers, label, color, marker, linestyle='-'):
    """Add a line, filtering out None and zero values."""
    valid = [(n, b) for n, b in zip(Ns, blers) if b is not None and b > 0]
    if not valid:
        return
    ns, bs = zip(*valid)
    kwargs = dict(color=color, marker=marker, linestyle=linestyle,
                  markerfacecolor='white', markeredgewidth=1.5,
                  markeredgecolor=color)
    if label is not None:
        kwargs['label'] = label
    ax.plot(ns, bs, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Data from BLER_TABLES.md and JSON files
# ═══════════════════════════════════════════════════════════════════════════

# ─── BEMAC Class B ────────────────────────────────────────────────────────

BEMAC_B = {
    'N':       [16,    32,      64,      128,     256,     512,    1024],
    'SC':      [0.011, 0.0097,  0.0032,  0.0016,  8e-5,    0.0,    1e-4],
    'NCG':     [0.011, 0.0076,  0.0032,  0.0017,  4e-5,    0.0,    1e-4],
    'SCL_L4':  [None,  0.0102,  0.0006,  0.0006,  0.0,     None,   None],
    'CRC_SCL': [None,  0.0,     0.0,     0.0,     0.0,     None,   None],
}

# ─── BEMAC Class C ────────────────────────────────────────────────────────

BEMAC_C = {
    'N':       [8,     16,    32,     64,     128,    256,    512,    1024],
    'SC':      [0.110, 0.092, 0.099,  0.055,  0.025,  0.013,  0.003,  0.00041],
    'NCG':     [0.039, 0.016, 0.006,  0.002,  0.0003, 0.0002, 0.0002, 0.0002],
    'SCL_L4':  [None,  None,  0.0076, 0.0053, 0.0010, 0.0005, None,   None],
    'CRC_SCL': [None,  None,  0.0008, 0.0007, 0.0,    0.0005, None,   None],
}

# ─── GMAC Class B ────────────────────────────────────────────────────────

GMAC_B = {
    'N':       [32,     64,     128,    256,    512],
    'SC':      [0.0450, 0.0276, 0.0187, 0.006,  0.001],
    'NCG':     [0.0503, 0.0282, 0.023,  0.023,  0.0123],
    'SCL_L4':  [0.0238, 0.0109, 0.0064, 0.0010, 0.0],
    'CRC_SCL': [0.0023, 0.0005, 0.0001, 0.0,    0.0],
}

# ─── GMAC Class C ────────────────────────────────────────────────────────

GMAC_C = {
    'N':       [16,    32,     64,     128,    256,    512,    1024],
    'SC':      [0.162, 0.0681, 0.0273, 0.0071, 0.0016, 0.1039, 0.1612],
    'NPD':     [0.107, 0.0373, 0.0100, 0.0329, 0.0003, 0.0002, 0.0],
    'NCG':     [0.119, 0.0353, 0.0133, 0.001,  None,   None,   None],
    'SCL_L4':  [None,  None,   0.0084, 0.0017, 0.0,    None,   None],
    'CRC_SCL': [None,  None,   0.0030, 0.0017, 0.0,    None,   None],
}

# ─── ABNMAC Class B ──────────────────────────────────────────────────────

ABNMAC_B = {
    'N':       [8,     16,     32,     64,     128],
    'SC':      [0.1198,0.0629, 0.0213, 0.0438, 0.0288],
    'NCG':     [0.1202,0.0570, 0.0182, 0.0416, 0.025],
    'SCL_L4':  [None,  None,   0.0104, 0.0194, 0.0077],
    'CRC_SCL': [None,  None,   0.0022, 0.0057, 0.0022],
}

# ─── ISI-MAC ─────────────────────────────────────────────────────────────

ISI_MAC = {
    'N':          [16,    32,    64,    128,   256],
    'Trellis_SC': [0.166, 0.083, 0.026, 0.018, 0.007],
    'Memless_SC': [0.185, 0.114, 0.088, 0.095, None],
    'NPD_d16_h64':  [0.143, 0.081, 0.046, None,  None],
    'NPD_d16_h100': [None,  None,  0.032, 0.081, None],
    'NPD_d64_h128': [None,  None,  None,  0.030, 0.011],
}


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: BEMAC Class B
# ═══════════════════════════════════════════════════════════════════════════

def fig_bemac_B():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    d = BEMAC_B
    add_line(ax, d['N'], d['SC'],      'SC',         C_SC,      MARKERS['SC'])
    add_line(ax, d['N'], d['NCG'],     'NCG',        C_NCG,     MARKERS['NCG'])
    add_line(ax, d['N'], d['SCL_L4'],  'SCL L=4',    C_SCL,     MARKERS['SCL'])
    add_line(ax, d['N'], d['CRC_SCL'], 'CRC-SCL L=4',C_CRC_SCL, MARKERS['CRC_SCL'])
    bler_plot_setup(ax, 'BEMAC Class B: BLER vs N')
    ax.set_ylim(1e-5, 0.5)
    # Add annotation for CRC-SCL = 0
    ax.annotate('CRC-SCL L=4: 0 errors\nat N=32,64,128,256',
                xy=(0.55, 0.02), xycoords='axes fraction',
                fontsize=8, color=C_CRC_SCL,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_CRC_SCL, alpha=0.8))
    ax.legend(loc='upper right')
    savefig(fig, 'fig_bemac_B_bler')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: BEMAC Class C
# ═══════════════════════════════════════════════════════════════════════════

def fig_bemac_C():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    d = BEMAC_C
    add_line(ax, d['N'], d['SC'],      'SC',         C_SC,      MARKERS['SC'])
    add_line(ax, d['N'], d['NCG'],     'NCG',        C_NCG,     MARKERS['NCG'])
    add_line(ax, d['N'], d['SCL_L4'],  'SCL L=4',    C_SCL,     MARKERS['SCL'])
    add_line(ax, d['N'], d['CRC_SCL'], 'CRC-SCL L=4',C_CRC_SCL, MARKERS['CRC_SCL'])
    bler_plot_setup(ax, 'BEMAC Class C: BLER vs N')
    ax.set_ylim(1e-5, 0.5)
    ax.legend(loc='upper right')
    savefig(fig, 'fig_bemac_C_bler')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: GMAC Class B
# ═══════════════════════════════════════════════════════════════════════════

def fig_gmac_B():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    d = GMAC_B
    add_line(ax, d['N'], d['SC'],      'SC',         C_SC,      MARKERS['SC'])
    add_line(ax, d['N'], d['NCG'],     'NCG',        C_NCG,     MARKERS['NCG'])
    add_line(ax, d['N'], d['SCL_L4'],  'SCL L=4',    C_SCL,     MARKERS['SCL'])
    add_line(ax, d['N'], d['CRC_SCL'], 'CRC-SCL L=4',C_CRC_SCL, MARKERS['CRC_SCL'])
    # NN-CRC-SCL data
    nn_crc = {32: 0.004, 64: 0.004, 128: 0.006, 256: 0.0133}
    nn_Ns = sorted(nn_crc.keys())
    nn_vals = [nn_crc[n] for n in nn_Ns]
    add_line(ax, nn_Ns, nn_vals, 'NN-CRC-SCL L=4', C_NN_CRC, MARKERS['NN_CRC'])
    bler_plot_setup(ax, 'GMAC Class B (6 dB): BLER vs N')
    ax.set_ylim(1e-5, 0.2)
    ax.legend(loc='upper right', fontsize=8)
    savefig(fig, 'fig_gmac_B_bler')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: GMAC Class C
# ═══════════════════════════════════════════════════════════════════════════

def fig_gmac_C():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    d = GMAC_C
    # SC: show N<=256 (N>=512 has design issues)
    add_line(ax, d['N'][:5], d['SC'][:5], 'SC', C_SC, MARKERS['SC'])
    # SC broken range (dashed)
    add_line(ax, d['N'][4:], d['SC'][4:], None, C_SC, MARKERS['SC'], linestyle='--')
    # NPD: full range including large N
    add_line(ax, d['N'], d['NPD'], 'NPD', C_NCG, MARKERS['NPD'])
    # NCG: where available
    ncg_ns = [n for n, v in zip(d['N'], d['NCG']) if v is not None]
    ncg_vs = [v for v in d['NCG'] if v is not None]
    add_line(ax, ncg_ns, ncg_vs, 'NCG', '#ff7f0e', 'D')
    add_line(ax, d['N'], d['SCL_L4'], 'SCL L=4', C_SCL, MARKERS['SCL'])
    add_line(ax, d['N'], d['CRC_SCL'], 'CRC-SCL L=4', C_CRC_SCL, MARKERS['CRC_SCL'])
    bler_plot_setup(ax, 'GMAC Class C (6 dB): BLER vs N')
    ax.set_ylim(1e-5, 0.5)
    ax.legend(loc='upper right', fontsize=8)
    savefig(fig, 'fig_gmac_C_bler')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: ABNMAC Class B
# ═══════════════════════════════════════════════════════════════════════════

def fig_abnmac_B():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    d = ABNMAC_B
    add_line(ax, d['N'], d['SC'],      'SC',         C_SC,      MARKERS['SC'])
    add_line(ax, d['N'], d['NCG'],     'NCG',        C_NCG,     MARKERS['NCG'])
    add_line(ax, d['N'], d['SCL_L4'],  'SCL L=4',    C_SCL,     MARKERS['SCL'])
    add_line(ax, d['N'], d['CRC_SCL'], 'CRC-SCL L=4',C_CRC_SCL, MARKERS['CRC_SCL'])
    # NN-CRC-SCL
    nn_crc = {32: 0.012, 64: 0.009}
    nn_Ns = sorted(nn_crc.keys())
    nn_vals = [nn_crc[n] for n in nn_Ns]
    add_line(ax, nn_Ns, nn_vals, 'NN-CRC-SCL L=4', C_NN_CRC, MARKERS['NN_CRC'])
    bler_plot_setup(ax, 'ABNMAC Class B: BLER vs N')
    ax.set_ylim(5e-4, 0.5)
    ax.legend(loc='upper right', fontsize=8)
    savefig(fig, 'fig_abnmac_B_bler')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: ISI-MAC
# ═══════════════════════════════════════════════════════════════════════════

def fig_isi_mac():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    d = ISI_MAC

    add_line(ax, d['N'], d['Trellis_SC'], 'Trellis SC (optimal)',  C_TRELLIS,  MARKERS['TRELLIS'])
    add_line(ax, d['N'], d['Memless_SC'], 'Memoryless SC',  C_MEMLESS,  MARKERS['MEMLESS'], linestyle='--')
    add_line(ax, d['N'], d['NPD_d16_h64'],  'NPD d=16 h=64',  '#2ca02c', 'v')
    add_line(ax, d['N'], d['NPD_d16_h100'], 'NPD d=16 h=100', C_NPD_D16, MARKERS['NPD_D16'])
    add_line(ax, d['N'], d['NPD_d64_h128'], 'NPD d=64 h=128', C_NPD_D64, MARKERS['NPD_D64'])

    bler_plot_setup(ax, 'ISI-MAC Class C ($h=0.3$, SNR = 6 dB): BLER vs $N$')
    ax.set_xticks(d['N'])
    ax.set_xticklabels([str(n) for n in d['N']])
    ax.set_ylim(5e-3, 0.3)
    ax.legend(loc='upper right')
    savefig(fig, 'fig_isi_mac_bler_v4')


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: All Channels Summary (multi-panel)
# ═══════════════════════════════════════════════════════════════════════════

def fig_all_channels_summary():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    # Panel 1: BEMAC Class B
    ax = axes[0, 0]
    d = BEMAC_B
    add_line(ax, d['N'], d['SC'],  'SC',  C_SC, MARKERS['SC'])
    add_line(ax, d['N'], d['NCG'], 'NCG', C_NCG, MARKERS['NCG'])
    add_line(ax, d['N'], d['CRC_SCL'], 'CRC-SCL', C_CRC_SCL, MARKERS['CRC_SCL'])
    bler_plot_setup(ax, 'BEMAC Class B')
    ax.set_ylim(1e-5, 0.5)
    ax.legend(fontsize=7, loc='upper right')

    # Panel 2: BEMAC Class C
    ax = axes[0, 1]
    d = BEMAC_C
    add_line(ax, d['N'], d['SC'],  'SC',  C_SC, MARKERS['SC'])
    add_line(ax, d['N'], d['NCG'], 'NCG', C_NCG, MARKERS['NCG'])
    add_line(ax, d['N'], d['CRC_SCL'], 'CRC-SCL', C_CRC_SCL, MARKERS['CRC_SCL'])
    bler_plot_setup(ax, 'BEMAC Class C')
    ax.set_ylim(1e-5, 0.5)
    ax.legend(fontsize=7, loc='upper right')

    # Panel 3: GMAC Class B
    ax = axes[0, 2]
    d = GMAC_B
    add_line(ax, d['N'], d['SC'],  'SC',  C_SC, MARKERS['SC'])
    add_line(ax, d['N'], d['NCG'], 'NCG', C_NCG, MARKERS['NCG'])
    add_line(ax, d['N'], d['CRC_SCL'], 'CRC-SCL', C_CRC_SCL, MARKERS['CRC_SCL'])
    nn_crc_gmac = {32: 0.004, 64: 0.004, 128: 0.006, 256: 0.0133}
    add_line(ax, list(nn_crc_gmac.keys()), list(nn_crc_gmac.values()),
             'NN-CRC-SCL', C_NN_CRC, MARKERS['NN_CRC'])
    bler_plot_setup(ax, 'GMAC Class B (6 dB)')
    ax.set_ylim(1e-5, 0.2)
    ax.legend(fontsize=7, loc='upper right')

    # Panel 4: GMAC Class C
    ax = axes[1, 0]
    d = GMAC_C
    N_good = [16, 32, 64, 128, 256]
    sc_good = [d['SC'][d['N'].index(n)] for n in N_good]
    add_line(ax, N_good, sc_good, 'SC', C_SC, MARKERS['SC'])
    npd_vals = [d['NPD'][d['N'].index(n)] for n in N_good]
    add_line(ax, N_good, npd_vals, 'NPD', C_NCG, MARKERS['NPD'])
    add_line(ax, N_good, [d['CRC_SCL'][d['N'].index(n)] for n in N_good],
             'CRC-SCL', C_CRC_SCL, MARKERS['CRC_SCL'])
    bler_plot_setup(ax, 'GMAC Class C (6 dB)')
    ax.set_ylim(1e-5, 0.5)
    ax.legend(fontsize=7, loc='upper right')

    # Panel 5: ABNMAC Class B
    ax = axes[1, 1]
    d = ABNMAC_B
    add_line(ax, d['N'], d['SC'],  'SC',  C_SC, MARKERS['SC'])
    add_line(ax, d['N'], d['NCG'], 'NCG', C_NCG, MARKERS['NCG'])
    add_line(ax, d['N'], d['CRC_SCL'], 'CRC-SCL', C_CRC_SCL, MARKERS['CRC_SCL'])
    bler_plot_setup(ax, 'ABNMAC Class B')
    ax.set_ylim(5e-4, 0.5)
    ax.legend(fontsize=7, loc='upper right')

    # Panel 6: ISI-MAC
    ax = axes[1, 2]
    d = ISI_MAC
    add_line(ax, d['N'], d['Trellis_SC'], 'Trellis SC', C_TRELLIS, MARKERS['TRELLIS'])
    add_line(ax, d['N'], d['Memless_SC'], 'Memoryless SC', C_MEMLESS, MARKERS['MEMLESS'], linestyle='--')
    add_line(ax, d['N'], d['NPD_d16_h64'], 'NPD d=16 h=64', '#2ca02c', 'v')
    add_line(ax, d['N'], d['NPD_d16_h100'], 'NPD d=16 h=100', C_NPD_D16, MARKERS['NPD_D16'])
    add_line(ax, d['N'], d['NPD_d64_h128'], 'NPD d=64 h=128', C_NPD_D64, MARKERS['NPD_D64'])
    bler_plot_setup(ax, 'ISI-MAC Class C (h=0.3, 6 dB)')
    ax.set_ylim(5e-4, 0.5)
    ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Neural Polar Decoders for MAC: BLER vs Block Length', fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, 'fig_all_channels_summary')


# ═══════════════════════════════════════════════════════════════════════════
# NN-SCL expansion data (from crc_scl_expansion)
# ═══════════════════════════════════════════════════════════════════════════

def load_nn_scl_data():
    """Load existing NN-SCL results from crc_scl_expansion/."""
    nn_scl = {}
    exp_dir = os.path.join(BASE, 'results', 'crc_scl_expansion')
    for fname in ['gmac_classB_crc_scl.json', 'bemac_classB_crc_scl.json']:
        fpath = os.path.join(exp_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                nn_scl[fname.replace('_crc_scl.json', '')] = json.load(f)
    return nn_scl


def fig_nn_scl_comparison():
    """Figure comparing NN-SCL vs analytical SCL vs SC for GMAC Class B."""
    nn_data = load_nn_scl_data()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # GMAC Class B: NN-SCL vs analytical
    ax = axes[0]
    gd = nn_data.get('gmac_classB', {})
    d = GMAC_B
    add_line(ax, d['N'], d['SC'], 'SC', C_SC, MARKERS['SC'])
    add_line(ax, d['N'], d['CRC_SCL'], 'Analytical CRC-SCL L=4', C_CRC_SCL, MARKERS['CRC_SCL'])

    # NN-SCL from expansion data
    nn_Ns, nn_scl_blers, nn_crc_blers = [], [], []
    for n_str, entry in sorted(gd.items(), key=lambda x: int(x[0])):
        N = int(n_str)
        if 'L4' in entry:
            nn_Ns.append(N)
            nn_scl_blers.append(entry['L4'].get('bler_scl', None))
            nn_crc_blers.append(entry['L4'].get('bler_crc_scl', None))

    add_line(ax, nn_Ns, nn_scl_blers, 'NN-SCL L=4', C_NN_SCL, MARKERS['NN_SCL'])
    add_line(ax, nn_Ns, nn_crc_blers, 'NN-CRC-SCL L=4', C_NN_CRC, MARKERS['NN_CRC'])

    bler_plot_setup(ax, 'GMAC Class B: NN-SCL vs Analytical SCL')
    ax.set_ylim(1e-5, 0.2)
    ax.legend(fontsize=8)

    # BEMAC Class B
    ax = axes[1]
    bd = nn_data.get('bemac_classB', {})
    d = BEMAC_B
    add_line(ax, d['N'], d['SC'], 'SC', C_SC, MARKERS['SC'])
    add_line(ax, d['N'], d['CRC_SCL'], 'Analytical CRC-SCL L=4', C_CRC_SCL, MARKERS['CRC_SCL'])

    nn_Ns, nn_scl_blers, nn_crc_blers = [], [], []
    for n_str, entry in sorted(bd.items(), key=lambda x: int(x[0])):
        N = int(n_str)
        if 'L4' in entry:
            nn_Ns.append(N)
            nn_scl_blers.append(entry['L4'].get('bler_scl', None))
            nn_crc_blers.append(entry['L4'].get('bler_crc_scl', None))

    add_line(ax, nn_Ns, nn_scl_blers, 'NN-SCL L=4', C_NN_SCL, MARKERS['NN_SCL'])
    add_line(ax, nn_Ns, nn_crc_blers, 'NN-CRC-SCL L=4', C_NN_CRC, MARKERS['NN_CRC'])

    bler_plot_setup(ax, 'BEMAC Class B: NN-SCL vs Analytical SCL')
    ax.set_ylim(1e-5, 0.5)
    ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, 'fig_nn_scl_comparison')


# ═══════════════════════════════════════════════════════════════════════════
# Figure: Multi Memory Channels (ISI + Ising + MA-AGN side by side)
# ═══════════════════════════════════════════════════════════════════════════

def fig_multi_memory_channels():
    """3-panel figure: ISI, Ising, MA-AGN memory channels."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    # ISI-MAC
    ax = axes[0]
    d = ISI_MAC
    add_line(ax, d['N'], d['Trellis_SC'], 'Trellis SC', C_TRELLIS, MARKERS['TRELLIS'])
    add_line(ax, d['N'], d['NPD_d64_h128'], 'NPD d=64 h=128', C_NPD_D64, MARKERS['NPD_D64'])
    add_line(ax, d['N'], d['NPD_d16_h100'], 'NPD d=16 h=100', C_NPD_D16, MARKERS['NPD_D16'])
    bler_plot_setup(ax, 'ISI-MAC ($h=0.3$)')
    ax.set_ylim(5e-3, 0.5)
    ax.legend(fontsize=8, loc='upper right')

    # Ising MAC
    ax = axes[1]
    ising_json = os.path.join(BASE, 'class_c_npd', 'results', 'npd_ising_mac', 'ising_mac_d16h100_results.json')
    if os.path.exists(ising_json):
        with open(ising_json) as f:
            ising = json.load(f)
        ising_N = []
        ising_npd = []
        ising_sc = []
        for Ns in sorted(ising.keys(), key=int):
            r = ising[Ns]
            ising_N.append(r['N'])
            ising_npd.append(r['chained']['bler_total'])
            ising_sc.append(r['memoryless_sc']['bler_total'])
        add_line(ax, ising_N, ising_sc, 'Memoryless SC', C_MEMLESS, MARKERS['MEMLESS'], linestyle='--')
        add_line(ax, ising_N, ising_npd, 'NPD d=16 h=100', C_NPD_D16, MARKERS['NPD_D16'])
    else:
        ax.text(0.5, 0.5, 'Data pending', transform=ax.transAxes, ha='center', fontsize=10, color='gray')
    bler_plot_setup(ax, 'Ising MAC ($p_{flip}=0.1$)')
    ax.set_ylim(5e-3, 1.0)
    ax.legend(fontsize=8, loc='upper right')

    # MA-AGN MAC
    ax = axes[2]
    maagn_json = os.path.join(BASE, 'class_c_npd', 'results', 'npd_memory_mac', 'maagn_mac_d16h100_results.json')
    if os.path.exists(maagn_json):
        with open(maagn_json) as f:
            maagn = json.load(f)
        maagn_N, maagn_npd, maagn_sc = [], [], []
        for Ns in sorted(maagn.keys(), key=int):
            r = maagn[Ns]
            maagn_N.append(r['N'])
            maagn_npd.append(r['chained']['bler_total'])
            maagn_sc.append(r['memoryless_sc']['bler_total'])
        add_line(ax, maagn_N, maagn_sc, 'Memoryless SC', C_MEMLESS, MARKERS['MEMLESS'], linestyle='--')
        add_line(ax, maagn_N, maagn_npd, 'NPD d=16 h=100', '#2ca02c', 'o')
    else:
        ax.text(0.5, 0.5, 'Training in progress', transform=ax.transAxes, ha='center', fontsize=10, color='gray')
    bler_plot_setup(ax, r'MA-AGN MAC ($\alpha=0.3$)')
    ax.set_ylim(5e-3, 1.0)
    ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Neural Polar Decoder on Memory MAC Channels (SNR = 6 dB, Class C)', fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, 'fig_multi_memory_channels')


# ═══════════════════════════════════════════════════════════════════════════
# Figure: Architecture Comparison (bar chart for N=64, N=128)
# ═══════════════════════════════════════════════════════════════════════════

def fig_architecture_comparison():
    """Horizontal bar chart comparing architectures at N=64 and N=128."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Data from consolidated results
    data_64 = [
        ('Trellis SC',     0.026, '#7f7f7f'),
        ('NPD d=16 h=100', 0.032, '#d62728'),
        ('NPD d=16 h=64',  0.046, '#2ca02c'),
    ]
    data_128 = [
        ('NPD d=64 h=128', 0.030, '#ff7f0e'),
        ('NPD d=16 h=100', 0.081, '#d62728'),
    ]

    for ax, data, target_N in [(axes[0], data_64, 64), (axes[1], data_128, 128)]:
        names = [d[0] for d in data]
        blers = [d[1] for d in data]
        colors = [d[2] for d in data]
        y_pos = range(len(names))
        ax.barh(y_pos, blers, color=colors, edgecolor='black', linewidth=0.5, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('BLER')
        ax.set_title(f'N = {target_N}')
        ax.grid(True, axis='x', alpha=0.3)
        # Add value labels
        for i, v in enumerate(blers):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)

    fig.suptitle('ISI-MAC: Architecture Comparison (5K CW)', fontsize=13)
    fig.tight_layout()
    savefig(fig, 'fig_architecture_comparison')


# ═══════════════════════════════════════════════════════════════════════════
# Figure: Wall Analysis (NPD/SC ratio vs N)
# ═══════════════════════════════════════════════════════════════════════════

def fig_wall_analysis():
    """Two panels: (1) BLER vs N, (2) NPD/SC ratio vs N."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Best NPD per N
    npd_N = [16, 32, 64, 128, 256]
    npd_bler = [0.143, 0.081, 0.032, 0.030, 0.011]

    # Trellis SC
    sc_N = [16, 32, 64]
    sc_bler = [0.166, 0.083, 0.026]

    # Left: BLER vs N
    ax = axes[0]
    ax.semilogy(npd_N, npd_bler, 'o-', color=C_NPD_D16, markersize=8, label='Best NPD')
    ax.semilogy(sc_N, sc_bler, 's--', color=C_TRELLIS, markersize=8, label='Trellis SC')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Block length $N$')
    ax.set_ylabel('BLER')
    ax.set_title('ISI-MAC: BLER vs $N$')
    ax.set_xticks(npd_N)
    ax.set_xticklabels([str(n) for n in npd_N])
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    # Right: ratio
    ratio_N = [16, 32, 64]
    ratios = [npd_bler[i] / sc_bler[i] for i in range(len(sc_N))]

    ax = axes[1]
    ax.plot(ratio_N, ratios, 'o-', color='#1f77b4', markersize=10, linewidth=2)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.fill_between(ratio_N, [1]*len(ratio_N), ratios, alpha=0.15, color='#1f77b4')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Block length $N$')
    ax.set_ylabel('NPD / Trellis SC ratio')
    ax.set_title('ISI-MAC: Wall Analysis')
    ax.set_xticks(ratio_N)
    ax.set_xticklabels([str(n) for n in ratio_N])
    ax.set_ylim(0.5, 2.0)
    ax.grid(True, alpha=0.3)

    for n, r in zip(ratio_N, ratios):
        ax.annotate(f'{r:.2f}x', (n, r), textcoords='offset points',
                    xytext=(8, 8), fontsize=10, fontweight='bold')

    fig.suptitle('ISI-MAC NPD Performance vs Block Length', fontsize=13)
    fig.tight_layout()
    savefig(fig, 'fig_wall_analysis')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Generating Thesis-Ready Figures")
    print("=" * 60)

    print("\n1. BEMAC Class B")
    fig_bemac_B()

    print("\n2. BEMAC Class C")
    fig_bemac_C()

    print("\n3. GMAC Class B")
    fig_gmac_B()

    print("\n4. GMAC Class C")
    fig_gmac_C()

    print("\n5. ABNMAC Class B")
    fig_abnmac_B()

    print("\n6. ISI-MAC")
    fig_isi_mac()

    print("\n7. All Channels Summary")
    fig_all_channels_summary()

    print("\n8. NN-SCL Comparison")
    fig_nn_scl_comparison()

    print("\n9. Multi Memory Channels")
    fig_multi_memory_channels()

    print("\n10. Architecture Comparison")
    fig_architecture_comparison()

    print("\n11. Wall Analysis")
    fig_wall_analysis()

    print(f"\n{'=' * 60}")
    print(f"  All figures saved to {FIG_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
