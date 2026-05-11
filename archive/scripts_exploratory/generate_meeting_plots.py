"""Generate BLER vs N plots for meeting documents."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'meeting')

# IEEE-style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'figure.figsize': (3.15, 2.0),  # ~8cm x 5cm
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

BLUE = '#1f77b4'
RED = '#d62728'
GRAY = '#7f7f7f'
GREEN = '#2ca02c'


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print(f'  saved {name}')


def make_bler_plot(title, series, fname, xlabel='N'):
    """series: list of (N, BLER, label, color, marker)"""
    fig, ax = plt.subplots()
    for N, bler, label, color, marker in series:
        # filter Nones
        valid = [(n, b) for n, b in zip(N, bler) if b is not None and b > 0]
        if not valid:
            continue
        ns, bs = zip(*valid)
        ax.semilogy(ns, bs, marker=marker, color=color, label=label)
    ax.set_xscale('log', base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('BLER')
    ax.set_title(title, fontsize=9)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    save(fig, fname)


def make_ratio_plot(title, N, ratio, label, fname):
    fig, ax = plt.subplots()
    valid = [(n, r) for n, r in zip(N, ratio) if r is not None]
    ns, rs = zip(*valid)
    ax.plot(ns, rs, 'o-', color=RED, label=label)
    ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('N')
    ax.set_ylabel('Neural / SC ratio')
    ax.set_title(title, fontsize=9)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    save(fig, fname)


# ---- BEMAC Class B ----
N_bb = [16, 32, 64, 128, 256, 512, 1024]
sc_bb = [0.011, 0.0097, 0.0032, 0.0016, 8e-5, None, 1e-4]
ncg_bb = [0.011, 0.0076, 0.0032, 0.0017, 4e-5, None, 1e-4]

make_bler_plot('BEMAC Class B (Symmetric)', [
    (N_bb, sc_bb, 'SC', BLUE, 's'),
    (N_bb, ncg_bb, 'NCG', RED, 'o'),
], 'plot_bemac_B.pdf')

# ---- BEMAC Class C ----
N_bc = [8, 16, 32, 64, 128, 256, 512, 1024]
sc_bc = [0.11, 0.092, 0.099, 0.055, 0.025, 0.013, 0.003, 4e-4]
ncg_bc = [0.039, 0.016, 0.006, 0.002, 3e-4, 2e-4, 2e-4, 2e-4]

make_bler_plot('BEMAC Class C (Corner Rate)', [
    (N_bc, sc_bc, 'SC', BLUE, 's'),
    (N_bc, ncg_bc, 'NCG', RED, 'o'),
], 'plot_bemac_C.pdf')

# ---- GMAC Class B ----
N_gb = [32, 64, 128, 256, 512, 1024]
sc_gb = [0.045, 0.028, 0.019, 0.006, 0.001, None]
ncg_gb = [0.050, 0.028, 0.023, 0.023, 0.012, 0.47]

make_bler_plot('GMAC Class B (Symmetric)', [
    (N_gb, sc_gb, 'SC', BLUE, 's'),
    (N_gb, ncg_gb, 'NCG', RED, 'o'),
], 'plot_gmac_B.pdf')

# ---- GMAC Class C ----
N_gc = [16, 32, 64, 128, 256, 512]
sc_gc = [0.162, 0.068, 0.027, 0.007, 0.001, 2e-4]
npd_gc = [0.107, 0.037, 0.010, 0.033, 3e-4, 2e-4]
ncg_gc = [0.119, 0.035, 0.013, 0.001, None, None]

make_bler_plot('GMAC Class C (Corner Rate)', [
    (N_gc, sc_gc, 'SC', BLUE, 's'),
    (N_gc, npd_gc, 'NPD', RED, 'o'),
    (N_gc, ncg_gc, 'NCG', GREEN, '^'),
], 'plot_gmac_C.pdf')

# ---- ABNMAC Class B ----
N_ab = [8, 16, 32, 64, 128]
sc_ab = [0.120, 0.063, 0.021, 0.044, 0.029]
ncg_ab = [0.120, 0.057, 0.018, 0.042, 0.025]

make_bler_plot('ABNMAC Class B (Symmetric)', [
    (N_ab, sc_ab, 'SC', BLUE, 's'),
    (N_ab, ncg_ab, 'NCG', RED, 'o'),
], 'plot_abnmac_B.pdf')

# ---- ISI-MAC ----
N_isi = [16, 32, 64, 128, 256, 512]
sc_isi = [0.169, 0.082, 0.041, 0.022, 0.006, 0.003]
npd_isi = [0.138, 0.057, 0.028, 0.029, 0.011, 0.108]

make_bler_plot('ISI-MAC (Chained SC vs Best NPD)', [
    (N_isi, sc_isi, 'Chained SC', BLUE, 's'),
    (N_isi, npd_isi, 'NPD (best)', RED, 'o'),
], 'plot_isi_mac.pdf')

# ---- Ising ----
N_is = [16, 32]
trellis_is = [0.575, 0.689]
memless_is = [0.634, 0.781]
npd_is = [0.365, 0.553]

make_bler_plot('Ising MAC', [
    (N_is, trellis_is, 'Trellis SC', BLUE, 's'),
    (N_is, memless_is, 'Memoryless SC', GRAY, 'D'),
    (N_is, npd_is, 'NPD', RED, 'o'),
], 'plot_ising.pdf')

# ---- MA-AGN alpha=0.3 ----
N_ma = [16, 32, 64, 128]
sc_ma03 = [0.175, 0.057, 0.025, 0.008]
npd_ma03 = [0.138, 0.053, 0.036, 0.093]

make_bler_plot(r'MA-AGN $\alpha=0.3$', [
    (N_ma, sc_ma03, 'Memoryless SC', BLUE, 's'),
    (N_ma, npd_ma03, 'NPD', RED, 'o'),
], 'plot_maagn_03.pdf')

# ---- MA-AGN alpha=0.9 ----
sc_ma09 = [0.197, 0.074, 0.043, 0.028]
npd_ma09 = [0.131, 0.056, 0.027, 0.061]

make_bler_plot(r'MA-AGN $\alpha=0.9$', [
    (N_ma, sc_ma09, 'Memoryless SC', BLUE, 's'),
    (N_ma, npd_ma09, 'NPD', RED, 'o'),
], 'plot_maagn_09.pdf')

# ---- Wall plots (ratio vs N) ----
# GMAC B ratio
ratio_gb = [ncg/sc if (ncg is not None and sc is not None) else None
            for ncg, sc in zip(ncg_gb, sc_gb)]
make_ratio_plot('GMAC B: NCG/SC Ratio vs N', N_gb, ratio_gb, 'NCG/SC', 'plot_wall_gmac.pdf')

# ISI-MAC ratio
ratio_isi = [npd/sc for npd, sc in zip(npd_isi, sc_isi)]
make_ratio_plot('ISI-MAC: NPD/SC Ratio vs N', N_isi, ratio_isi, 'NPD/Chained SC', 'plot_wall_isi.pdf')

print('All plots generated.')
