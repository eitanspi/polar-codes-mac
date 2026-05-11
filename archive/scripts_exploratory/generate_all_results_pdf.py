#!/usr/bin/env python3
"""Generate ALL_RESULTS.pdf: one page per channel/class combo with settings, table, and BLER plot."""

import torch
torch.set_num_threads(2)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# --- Style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': 'gray',
})

COLORS = {
    'SC': 'blue',
    'NCG': 'red',
    'NPD': 'green',
    'CRC-SCL L=4': 'orange',
    'NN-CRC-SCL': 'orange',
    'Memoryless SC': 'gray',
    'Trellis SC': 'cyan',
    'Joint Trellis SC': 'cyan',
    'Chained Trellis SC': 'purple',
    'SCL L=4': 'brown',
}
MARKERS = {
    'SC': 'o',
    'NCG': 's',
    'NPD': 'D',
    'CRC-SCL L=4': '^',
    'NN-CRC-SCL': '^',
    'Memoryless SC': 'v',
    'Trellis SC': '*',
    'Joint Trellis SC': '*',
    'Chained Trellis SC': 'P',
    'SCL L=4': 'X',
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'paper_figures')
OUTPUT_PDF = os.path.join(OUTPUT_DIR, 'ALL_RESULTS.pdf')


def fmt_bler(v):
    if v is None:
        return '--'
    if v == 0.0:
        return '0.0000'
    if v < 0.001:
        return f'{v:.1e}'
    return f'{v:.4f}'


def make_page(pdf, title, subtitle, equation, settings_lines, col_headers, table_data, plot_series, note=None):
    """Create one page with header, settings, table (top), and plot (bottom)."""
    fig = plt.figure(figsize=(11, 8.5))  # landscape-ish

    # --- Top half: header + settings + table ---
    ax_top = fig.add_axes([0.05, 0.50, 0.90, 0.48])
    ax_top.axis('off')

    # Header
    y = 0.98
    ax_top.text(0.5, y, title, ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax_top.transAxes)
    y -= 0.055
    ax_top.text(0.5, y, subtitle, ha='center', va='top', fontsize=11, fontstyle='italic',
                transform=ax_top.transAxes)
    y -= 0.045
    ax_top.text(0.5, y, equation, ha='center', va='top', fontsize=10, family='monospace',
                transform=ax_top.transAxes)

    # Settings box
    y -= 0.055
    for line in settings_lines:
        ax_top.text(0.05, y, line, ha='left', va='top', fontsize=8.5, transform=ax_top.transAxes)
        y -= 0.040

    # Table
    y -= 0.03
    n_cols = len(col_headers)
    n_rows = len(table_data)
    cell_text = []
    for row in table_data:
        cell_text.append([str(v) if v is not None else '--' for v in row])

    row_h = 0.040
    table_h = row_h * (n_rows + 1)
    table = ax_top.table(
        cellText=cell_text,
        colLabels=col_headers,
        loc='center',
        cellLoc='center',
        bbox=[0.02, max(0.0, y - table_h), 0.96, table_h]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5 if n_cols > 8 else 8)
    # Header row styling
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#d4e6f1')
        cell.set_text_props(fontweight='bold')
    # Alternate row shading
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            if i % 2 == 0:
                table[i, j].set_facecolor('#f9f9f9')

    # --- Bottom half: plot ---
    ax_plot = fig.add_axes([0.10, 0.07, 0.80, 0.36])

    for label, ns, blers in plot_series:
        valid = [(n, b) for n, b in zip(ns, blers) if b is not None and b > 0]
        if not valid:
            continue
        vn, vb = zip(*valid)
        color = COLORS.get(label, 'black')
        marker = MARKERS.get(label, 'o')
        ax_plot.semilogy(vn, vb, marker=marker, color=color, label=label,
                         linewidth=1.5, markersize=6, markeredgecolor='black', markeredgewidth=0.5)

    ax_plot.set_xlabel('Block length N')
    ax_plot.set_ylabel('BLER')
    if note:
        ax_plot.set_title(f'BLER vs N  --  {note}', fontsize=8, fontstyle='italic', color='#444444')
    else:
        ax_plot.set_title('BLER vs N', fontsize=11)
    ax_plot.legend(fontsize=8, loc='best', framealpha=0.9)
    ax_plot.set_xscale('log', base=2)
    # Set x-ticks to powers of 2
    all_ns = set()
    for _, ns, _ in plot_series:
        all_ns.update(ns)
    all_ns = sorted(all_ns)
    if all_ns:
        ax_plot.set_xticks(all_ns)
        ax_plot.set_xticklabels([str(n) for n in all_ns])
    ax_plot.grid(True, which='both', alpha=0.3)

    pdf.savefig(fig)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with PdfPages(OUTPUT_PDF) as pdf:

        # =====================================================================
        # 1. BEMAC Class B
        # =====================================================================
        Ns =     [16,    32,     64,     128,    256,    512,    1024]
        ku =     [8,     16,     32,     64,     128,    256,    512]
        kv =     [11,    22,     44,     89,     178,    358,    716]
        sc =     [0.011, 0.0097, 0.0032, 0.0016, 8e-5,   0.0,   1e-4]
        ncg =    [0.011, 0.0076, 0.0032, 0.0017, 4e-5,   0.0,   1e-4]
        crc_scl= [None,  0.0,    0.0,    0.0,    0.0,    None,  None]
        sc_err = ['~55/5k','146/15k','128/40k','81/50k','~4/50k','0/2k','~1/10k']

        table_data = []
        for i in range(len(Ns)):
            table_data.append([
                Ns[i], ku[i], kv[i],
                fmt_bler(sc[i]), sc_err[i],
                fmt_bler(ncg[i]),
                fmt_bler(crc_scl[i]),
            ])

        make_page(pdf,
            title='1. BEMAC Class B (non-corner, symmetric rate)',
            subtitle='Binary Erasure MAC -- Z = X + Y',
            equation='Z = X + Y,  Z in {0, 1, 2}  (deterministic)',
            settings_lines=[
                'Path: make_path(N, N//2) = [0]^{N/2} [1]^N [0]^{N/2}',
                'Capacity: I(X;Z)=0.500, I(Y;Z|X)=1.000, Symmetric point: (0.750, 0.750)',
                'Operating rate: R_U ~ 0.50, R_V ~ 0.70',
                'Architecture: NCG d=16, hidden=64, BEMAC vocab=3 embedding',
            ],
            col_headers=['N', 'ku', 'kv', 'SC BLER', 'SC errs/CW', 'NCG BLER', 'CRC-SCL L=4'],
            table_data=table_data,
            plot_series=[
                ('SC',  Ns, sc),
                ('NCG', Ns, ncg),
                ('CRC-SCL L=4', Ns, crc_scl),
            ],
            note='NCG ~ SC on Class B. CRC-SCL L=4 achieves zero errors at all tested N.',
        )

        # =====================================================================
        # 2. BEMAC Class C
        # =====================================================================
        Ns =     [8,    16,    32,    64,    128,   256,   512,   1024]
        ku =     [2,    5,     10,    19,    38,    77,    154,   307]
        kv =     [5,    10,    19,    38,    77,    154,   307,   614]
        sc =     [0.110,0.092, 0.099, 0.055, 0.025, 0.013, 0.003, 0.00041]
        ncg =    [0.039,0.016, 0.006, 0.002, 0.0003,0.0002,0.0002,0.0002]
        crc_scl= [None, None,  0.0008,0.0007,0.0,   0.0005,None,  None]
        sc_err = ['329/3k','276/3k','298/3k','165/3k','491/20k','668/50k','168/50k','26/64k']
        ncg_ratio = ['0.36x','0.18x','0.06x','0.03x','0.01x','0.01x','0.06x','0.42x']

        table_data = []
        for i in range(len(Ns)):
            table_data.append([
                Ns[i], ku[i], kv[i],
                fmt_bler(sc[i]), sc_err[i],
                fmt_bler(ncg[i]), ncg_ratio[i],
                fmt_bler(crc_scl[i]),
            ])

        make_page(pdf,
            title='2. BEMAC Class C (corner rate)',
            subtitle='Binary Erasure MAC -- Z = X + Y',
            equation='Z = X + Y,  Z in {0, 1, 2}  (deterministic)',
            settings_lines=[
                'Path: make_path(N, N) = [0]^N [1]^N (all U first, then all V)',
                'Capacity: Corner point (R_U, R_V) = (0.500, 1.000)',
                'Operating rate: R_U ~ 0.30, R_V ~ 0.60 (60% of corner capacity)',
                'Architecture: NCG d=16, hidden=64, BEMAC vocab=3 embedding',
            ],
            col_headers=['N', 'ku', 'kv', 'SC BLER', 'SC errs/CW', 'NCG BLER', 'NCG/SC', 'CRC-SCL L=4'],
            table_data=table_data,
            plot_series=[
                ('SC',  Ns, sc),
                ('NCG', Ns, ncg),
                ('CRC-SCL L=4', Ns, crc_scl),
            ],
            note='NCG beats SC by 2-100x across all N. Strongest NCG result.',
        )

        # =====================================================================
        # 3. GMAC Class B
        # =====================================================================
        Ns =     [32,    64,     128,    256,   512,    1024]
        ku_kv =  [15,    31,     62,     123,   246,    492]
        sc =     [0.0450,0.0276, 0.0187, 0.006, 0.001,  None]
        ncg =    [0.0503,0.0282, 0.023,  0.023, 0.0123, 0.47]
        crc_scl= [0.0023,0.0005, 0.0001, 0.0,   0.0,    None]
        sc_err = ['135/3k','138/5k','112/6k','30/5k','est.','--']
        ncg_ratio = ['1.12x','1.02x','1.23x','4.5x','12.3x','BROKEN']

        table_data = []
        for i in range(len(Ns)):
            table_data.append([
                Ns[i], ku_kv[i],
                fmt_bler(sc[i]), sc_err[i],
                fmt_bler(ncg[i]), ncg_ratio[i],
                fmt_bler(crc_scl[i]),
            ])

        make_page(pdf,
            title='3. GMAC Class B (non-corner, symmetric rate)',
            subtitle='Gaussian MAC -- Z = (1-2X) + (1-2Y) + W',
            equation='Z = (1-2X) + (1-2Y) + W,  W ~ N(0, sigma^2),  SNR = 6 dB',
            settings_lines=[
                'Path: make_path(N, N//2), symmetric rate ~(0.688, 0.688)',
                'Capacity: I(X;Z)=0.465, I(Y;Z|X)=0.912, I(X,Y;Z)=1.376',
                'Operating rate: R_U = R_V ~ 0.48 (~70% of symmetric capacity)',
                'Architecture: NCG d=16, hidden=64, GMAC z_encoder MLP',
            ],
            col_headers=['N', 'ku=kv', 'SC BLER', 'SC errs/CW', 'NCG BLER', 'NCG/SC', 'CRC-SCL L=4'],
            table_data=table_data,
            plot_series=[
                ('SC',  Ns, sc),
                ('NCG', Ns, ncg),
                ('CRC-SCL L=4', Ns, crc_scl),
            ],
            note='NCG does NOT beat SC on GMAC Class B. Wall at N=256. CRC-SCL L=4 is the dominant decoder.',
        )

        # =====================================================================
        # 4. GMAC Class C
        # =====================================================================
        Ns =     [16,    32,     64,     128,   256,    512,    1024]
        ku =     [4,     7,      15,     30,    59,     119,    238]
        kv =     [7,     15,     29,     58,    117,    233,    467]
        sc =     [0.162, 0.0681, 0.0273, 0.0071,0.0016, 0.1039, 0.1612]
        npd =    [0.107, 0.0373, 0.0100, 0.0329,0.0003, 0.0002, 0.0]
        ncg =    [0.119, 0.0353, 0.0133, 0.001, None,   None,   None]
        crc_scl= [None,  None,   0.0030, 0.0017,0.0,    None,   None]
        sc_err = ['1620/10k','681/10k','273/10k','71/10k','31/20k','5197/50k','8058/50k']

        table_data = []
        for i in range(len(Ns)):
            table_data.append([
                Ns[i], ku[i], kv[i],
                fmt_bler(sc[i]), sc_err[i],
                fmt_bler(npd[i]),
                fmt_bler(ncg[i]),
                fmt_bler(crc_scl[i]),
            ])

        make_page(pdf,
            title='4. GMAC Class C (corner rate)',
            subtitle='Gaussian MAC -- Z = (1-2X) + (1-2Y) + W',
            equation='Z = (1-2X) + (1-2Y) + W,  W ~ N(0, sigma^2),  SNR = 6 dB',
            settings_lines=[
                'Path: make_path(N, N), corner rate (R_U ~ 0.23, R_V ~ 0.45)',
                'Capacity: Corner point (R_U, R_V) = (0.465, 0.912)',
                'NPD/NCG use MI-designed frozen sets (co-adapted with NN decoder)',
                'Architecture: NPD d=16, hidden=64, neural fast_ce (use_analytical=False)',
            ],
            col_headers=['N', 'ku', 'kv', 'SC BLER', 'SC errs/CW', 'NPD BLER', 'NCG BLER', 'CRC-SCL L=4'],
            table_data=table_data,
            plot_series=[
                ('SC',  Ns, sc),
                ('NPD', Ns, npd),
                ('NCG', Ns, ncg),
                ('CRC-SCL L=4', Ns, crc_scl),
            ],
            note='NPD/NCG gain partly from frozen-set co-adaptation. SC at N=512,1024 has design issue (high BLER).',
        )

        # =====================================================================
        # 5. ABNMAC Class B
        # =====================================================================
        Ns =     [8,     16,     32,     64,     128]
        ku_kv =  [3,     5,      10,     22,     45]
        sc =     [0.1198,0.0629, 0.0213, 0.0438, 0.0288]
        ncg =    [0.1202,0.0570, 0.0182, 0.0416, 0.025]
        crc_scl= [None,  None,   0.0022, 0.0057, 0.0022]
        sc_err = ['1198/10k','629/10k','213/10k','219/5k','115/4k']
        ncg_ratio = ['1.00x','0.91x','0.85x','0.95x','--']

        table_data = []
        for i in range(len(Ns)):
            table_data.append([
                Ns[i], ku_kv[i],
                fmt_bler(sc[i]), sc_err[i],
                fmt_bler(ncg[i]), ncg_ratio[i],
                fmt_bler(crc_scl[i]),
            ])

        make_page(pdf,
            title='5. ABNMAC Class B (non-corner, symmetric rate)',
            subtitle='Asymmetric Binary Noise MAC',
            equation='Z = (X xor Ex, Y xor Ey),  correlated binary noise matrix',
            settings_lines=[
                'Path: make_path(N, N//2), symmetric rate ~(0.600, 0.600)',
                'Capacity: I(X;Z) ~ 0.400, I(Y;Z|X) ~ 0.800, Symmetric point: (0.600, 0.600)',
                'Operating rate: R_U ~ R_V ~ 0.30 (ku=kv)',
                'Architecture: NCG d=16, hidden=64',
            ],
            col_headers=['N', 'ku=kv', 'SC BLER', 'SC errs/CW', 'NCG BLER', 'NCG/SC', 'CRC-SCL L=4'],
            table_data=table_data,
            plot_series=[
                ('SC',  Ns, sc),
                ('NCG', Ns, ncg),
                ('CRC-SCL L=4', Ns, crc_scl),
            ],
            note='NCG gains are modest (0-15%). Non-monotonic SC BLER at N=64 > N=32 (rate-selection artifact).',
        )

        # =====================================================================
        # 6. ABNMAC Class C
        # =====================================================================
        Ns =     [16,    32,     64,     128,   256,   512,   1024]
        ku =     [3,     6,      13,     26,    51,    102,   205]
        kv =     [6,     13,     26,     51,    102,   205,   410]
        sc =     [0.062, 0.0334, 0.0478, 0.030, 0.013, 0.011, 0.0]
        crc_scl= [None,  None,   0.0247, 0.0136,None,  None,  None]
        scl4 =   [None,  None,   0.0330, 0.0144,None,  None,  None]
        sc_err = ['?/?','167/5k','239/5k','150/5k','?/?','?/?','?/?']

        table_data = []
        for i in range(len(Ns)):
            table_data.append([
                Ns[i], ku[i], kv[i],
                fmt_bler(sc[i]), sc_err[i],
                fmt_bler(scl4[i]),
                fmt_bler(crc_scl[i]),
            ])

        make_page(pdf,
            title='6. ABNMAC Class C (corner rate)',
            subtitle='Asymmetric Binary Noise MAC -- SC baseline only',
            equation='Z = (X xor Ex, Y xor Ey),  correlated binary noise matrix',
            settings_lines=[
                'Path: make_path(N, N) = [0]^N [1]^N',
                'Capacity: Corner point (R_U, R_V) = (0.400, 0.800)',
                'Operating rate: R_U ~ 0.19, R_V ~ 0.38',
                'No neural decoder trained (ABNMAC tuple output incompatible with current z_encoder)',
            ],
            col_headers=['N', 'ku', 'kv', 'SC BLER', 'SC errs/CW', 'SCL L=4', 'CRC-SCL L=4'],
            table_data=table_data,
            plot_series=[
                ('SC',  Ns, sc),
                ('SCL L=4', Ns, scl4),
                ('CRC-SCL L=4', Ns, crc_scl),
            ],
            note='No neural decoder tested. Non-monotonic SC BLER confirmed (N=64 > N=32). CRC-SCL gives 2-3x at N=64,128.',
        )

        # =====================================================================
        # 7. ISI-MAC Class C
        # =====================================================================
        Ns =        [16,    32,    64,    128,   256,   512,   1024]
        ku =        [4,     7,     15,    30,    59,    119,   238]
        kv =        [7,     15,    29,    58,    117,   233,   467]
        joint_sc =  [0.166, 0.083, 0.026, 0.018, None,  None,  None]
        chain_sc =  [0.169, 0.082, 0.041, 0.022, 0.006, 0.003, 0.007]
        memless_sc= [0.187, 0.113, 0.079, 0.101, 0.226, 0.502, 0.870]
        npd_d16 =   [0.138, 0.057, 0.028, 0.087, None,  None,  None]  # GPU curriculum best
        npd_d64 =   [None,  None,  None,  0.030, 0.011, None,  None]
        # For the table, pick best NPD
        npd_best =  [0.138, 0.057, 0.028, 0.030, 0.011, None,  None]
        npd_model = ['d16 curric','d16 curric','d16 h=100','d64 h=128','d64 h=128','--','--']

        joint_err =   ['1664/10k','825/10k','262/10k','180/10k','--','--','--']
        chain_err =   ['1689/10k','822/10k','407/10k','223/10k','61/10k','5/2k','2/300']
        memless_err = ['1866/10k','1129/10k','790/10k','1009/10k','2256/10k','5018/10k','8704/10k']
        npd_err =     ['688/5k','283/5k','139/5k','150/5k','56/5k','--','--']

        table_data = []
        for i in range(len(Ns)):
            table_data.append([
                Ns[i], ku[i], kv[i],
                fmt_bler(joint_sc[i]),
                fmt_bler(chain_sc[i]),
                fmt_bler(memless_sc[i]),
                fmt_bler(npd_best[i]),
                npd_model[i],
            ])

        make_page(pdf,
            title='7. ISI-MAC Class C (corner rate) -- Memory Channel',
            subtitle='Inter-Symbol Interference MAC',
            equation='Z[i] = (1-2X[i]) + (1-2Y[i]) + h*(1-2X[i-1]) + h*(1-2Y[i-1]) + W[i],  h=0.3',
            settings_lines=[
                'Parameters: h=0.3, SNR=6.0 dB (sigma^2=0.251)',
                'Path: make_path(N, N), corner rate',
                'Design proxy: GMAC Class C at SNR=6 dB',
                'NPD: chained decoder (Stage 1: U from z, Stage 2: V from z,u_hat)',
            ],
            col_headers=['N', 'ku', 'kv', 'Joint Trellis', 'Chained Trellis', 'Memoryless SC', 'NPD (best)', 'NPD model'],
            table_data=table_data,
            plot_series=[
                ('Joint Trellis SC', Ns, joint_sc),
                ('Chained Trellis SC', Ns, chain_sc),
                ('Memoryless SC', Ns, memless_sc),
                ('NPD', Ns, npd_best),
            ],
            note='NPD beats chained trellis SC by 19-32% at N=16,32,64. At N=64, NPD matches joint trellis SC.',
        )

        # =====================================================================
        # 8. Ising MAC Class C
        # =====================================================================
        Ns =        [16,    32]
        ku =        [4,     7]
        kv =        [7,     15]
        trellis_sc= [0.575, 0.689]
        memless_sc= [0.634, 0.781]
        npd =       [0.592, 0.770]
        trellis_err = ['2873/5k','3443/5k']
        memless_err = ['1902/3k','2342/3k']
        npd_err =     ['2960/5k','3850/5k']

        table_data = []
        for i in range(len(Ns)):
            table_data.append([
                Ns[i], ku[i], kv[i],
                fmt_bler(trellis_sc[i]), trellis_err[i],
                fmt_bler(memless_sc[i]), memless_err[i],
                fmt_bler(npd[i]), npd_err[i],
            ])

        make_page(pdf,
            title='8. Ising MAC Class C (corner rate) -- Memory Channel',
            subtitle='Ising Channel with Markov State',
            equation='Good: Z=(1-2X)+(1-2Y)+W,  Bad: Z=W (pure noise),  p_flip=0.1',
            settings_lines=[
                'Parameters: sigma^2=0.251, p_flip=0.1 (Markov state flip probability)',
                'Path: make_path(N, N), corner rate',
                'Design proxy: GMAC Class C at SNR=6 dB',
                'Extremely hard channel -- BLER > 57% at all tested N',
            ],
            col_headers=['N', 'ku', 'kv', 'Trellis SC', 'Trellis errs', 'Memless SC', 'Memless errs', 'NPD d=16', 'NPD errs'],
            table_data=table_data,
            plot_series=[
                ('Trellis SC', Ns, trellis_sc),
                ('Memoryless SC', Ns, memless_sc),
                ('NPD', Ns, npd),
            ],
            note='Ising MAC is extremely hard (BLER>57%). NPD partially learns memory (beats memoryless by 4-2%) but not trellis SC.',
        )

        # =====================================================================
        # 9. MA-AGN MAC Class C
        # =====================================================================
        Ns =        [16,    32,    64,    128]
        ku =        [4,     7,     15,    30]
        kv =        [7,     15,    29,    58]
        memless_sc= [0.175, 0.077, 0.025, None]
        npd_best =  [0.138, 0.112, 0.035, None]
        npd_model = ['d=32 h=128 BiGRU','d=32 h=128 BiGRU','d=16 h=100','--']
        npd_ratio = ['0.79x','1.46x','1.42x','--']
        memless_err = ['349/2k','153/2k','75/3k','--']
        npd_err =     ['--','567/5k','177/5k','--']

        table_data = []
        for i in range(len(Ns)):
            table_data.append([
                Ns[i], ku[i], kv[i],
                fmt_bler(memless_sc[i]), memless_err[i],
                fmt_bler(npd_best[i]), npd_model[i], npd_ratio[i],
            ])

        make_page(pdf,
            title='9. MA-AGN MAC Class C (corner rate) -- Continuous-State Memory',
            subtitle='Moving-Average Additive Gaussian Noise MAC',
            equation='Z[i] = (1-2X[i]) + (1-2Y[i]) + N[i],  N[i] = alpha*N[i-1] + W[i],  alpha=0.3',
            settings_lines=[
                'Parameters: alpha=0.3, sigma^2=0.251 (SNR=6 dB), AR(1) noise process',
                'Path: make_path(N, N), corner rate',
                'Design proxy: GMAC Class C at SNR=6 dB',
                'No analytical trellis exists (continuous state) -- only neural decoder can learn memory',
            ],
            col_headers=['N', 'ku', 'kv', 'Memless SC', 'Memless errs', 'NPD (best)', 'NPD model', 'NPD/SC'],
            table_data=table_data,
            plot_series=[
                ('Memoryless SC', Ns, memless_sc),
                ('NPD', Ns, npd_best),
            ],
            note='NPD beats memoryless SC at N=16 (21%). At N>=32, chained marginalisation hurts more than memory helps.',
        )

    print(f"PDF saved to: {OUTPUT_PDF}")
    print(f"  9 pages, one per channel/class combo")


if __name__ == '__main__':
    main()
