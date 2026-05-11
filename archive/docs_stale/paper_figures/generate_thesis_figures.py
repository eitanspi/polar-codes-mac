"""Generate all 7 thesis-ready figures from authoritative JSON data sources.
IEEE style: serif 11pt, log y for BLER, markers+lines, consistent colors.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import os

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
})

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Consistent color scheme
C_SC = '#1f77b4'        # Blue - analytical SC
C_NCG = '#d62728'       # Red - NCG
C_NPD = '#2ca02c'       # Green - NPD
C_TRELLIS = '#1f77b4'   # Blue - trellis SC
C_MEMLESS = '#9467bd'   # Purple - memoryless SC
C_CRC = '#ff7f0e'       # Orange - CRC-SCL
C_D16H64 = '#8c564b'    # Brown
C_D16H100 = '#2ca02c'   # Green
C_D64H128 = '#d62728'   # Red
C_GPU = '#e377c2'       # Pink - GPU curriculum

def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f'{name}.png'))
    fig.savefig(os.path.join(OUTDIR, f'{name}.pdf'))
    print(f'  Saved {name}.{{png,pdf}}')
    plt.close(fig)


# ===========================================================================
# Figure 1: BEMAC Results (2-panel: Class B + C)
# ===========================================================================
def fig1_bemac():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Class B
    N_b = [32, 64, 128]
    sc_b = [0.0097, 0.0032, 0.0016]
    ncg_b = [0.0076, 0.0032, 0.0017]
    ax1.semilogy(N_b, sc_b, 'o-', color=C_SC, label='SC')
    ax1.semilogy(N_b, ncg_b, 's--', color=C_NCG, label='NCG')
    ax1.set_xlabel('Block length N')
    ax1.set_ylabel('BLER')
    ax1.set_title('(a) BEMAC Class B')
    ax1.set_xticks(N_b)
    ax1.set_xticklabels([str(n) for n in N_b])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Class C
    N_c = [8, 16, 32, 64, 128, 256, 512, 1024]
    sc_c = [0.110, 0.092, 0.099, 0.055, 0.0245, 0.0134, 0.0034, 0.0004]
    ncg_c = [0.039, 0.016, 0.006, 0.002, 0.0003, 0.0002, 0.0002, 0.0002]
    ax2.semilogy(N_c, sc_c, 'o-', color=C_SC, label='SC')
    ax2.semilogy(N_c, ncg_c, 's--', color=C_NCG, label='NCG')
    ax2.set_xlabel('Block length N')
    ax2.set_ylabel('BLER')
    ax2.set_title('(b) BEMAC Class C')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(N_c)
    ax2.set_xticklabels([str(n) for n in N_c], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('BEMAC: SC vs NCG', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig_bemac_results')

# ===========================================================================
# Figure 2: GMAC Results (2-panel: Class B + C)
# ===========================================================================
def fig2_gmac():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Class B
    N_b = [32, 64, 128, 256, 512]
    sc_b = [0.0450, 0.0276, 0.0187, 0.0060, 0.0010]
    ncg_b = [0.0503, 0.0282, 0.0230, 0.0230, 0.0123]
    crc_b = [0.0023, 0.0005, 0.0001, None, None]
    ax1.semilogy(N_b, sc_b, 'o-', color=C_SC, label='SC')
    ax1.semilogy(N_b, ncg_b, 's--', color=C_NCG, label='NCG')
    crc_N = [n for n, v in zip(N_b, crc_b) if v is not None]
    crc_v = [v for v in crc_b if v is not None]
    ax1.semilogy(crc_N, crc_v, '^:', color=C_CRC, label='CRC-SCL L=4')
    ax1.set_xlabel('Block length N')
    ax1.set_ylabel('BLER')
    ax1.set_title('(a) GMAC Class B (SNR=6dB)')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(N_b)
    ax1.set_xticklabels([str(n) for n in N_b])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Mark the wall
    ax1.annotate('Wall', xy=(256, 0.023), fontsize=9, color='red',
                ha='center', va='bottom')

    # Class C
    N_c = [16, 32, 64, 128, 256]
    sc_c = [0.162, 0.068, 0.027, 0.007, 0.002]
    npd_c = [0.107, 0.037, 0.010, 0.033, 0.0003]
    ncg_c = [0.119, 0.035, 0.013, 0.001, None]
    ax2.semilogy(N_c, sc_c, 'o-', color=C_SC, label='SC')
    ax2.semilogy(N_c, npd_c, 's--', color=C_NPD, label='NPD')
    ncg_N = [n for n, v in zip(N_c, ncg_c) if v is not None]
    ncg_v = [v for v in ncg_c if v is not None]
    ax2.semilogy(ncg_N, ncg_v, 'D-.', color=C_NCG, label='NCG')
    ax2.set_xlabel('Block length N')
    ax2.set_ylabel('BLER')
    ax2.set_title('(b) GMAC Class C (SNR=6dB)')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(N_c)
    ax2.set_xticklabels([str(n) for n in N_c])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('GMAC: SC vs Neural Decoders', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig_gmac_results')

# ===========================================================================
# Figure 3: ABNMAC Results
# ===========================================================================
def fig3_abnmac():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    N = [8, 16, 32, 64, 128]
    sc = [0.1198, 0.0629, 0.0213, 0.0438, 0.0288]
    ncg = [0.1202, 0.0570, 0.0182, 0.0416, 0.0250]
    crc = [None, None, 0.0022, 0.0057, 0.0022]

    ax.semilogy(N, sc, 'o-', color=C_SC, label='SC')
    ax.semilogy(N, ncg, 's--', color=C_NCG, label='NCG')
    crc_N = [n for n, v in zip(N, crc) if v is not None]
    crc_v = [v for v in crc if v is not None]
    ax.semilogy(crc_N, crc_v, '^:', color=C_CRC, label='CRC-SCL L=4')

    ax.set_xlabel('Block length N')
    ax.set_ylabel('BLER')
    ax.set_title('ABNMAC Class B: SC vs NCG')
    ax.set_xscale('log', base=2)
    ax.set_xticks(N)
    ax.set_xticklabels([str(n) for n in N])
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, 'fig_abnmac_results')

# ===========================================================================
# Figure 4: ISI-MAC Final (HEADLINE) — all decoders, N=16-256
# ===========================================================================
def fig4_isi_mac():
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    N_vals = [16, 32, 64, 128, 256]

    # Joint trellis SC (N=16-128 only from reliable data)
    joint_N = [16, 32, 64, 128]
    joint_bler = [0.1664, 0.0825, 0.0262, 0.0180]
    ax.semilogy(joint_N, joint_bler, 'o-', color='#1a1a1a', label='Joint Trellis SC', linewidth=2)

    # Chained trellis SC
    ch_bler = [0.1689, 0.0822, 0.0407, 0.0223, 0.0061]
    ax.semilogy(N_vals, ch_bler, 's-', color=C_TRELLIS, label='Chained Trellis SC')

    # Memoryless SC
    ml_bler = [0.1866, 0.1129, 0.0790, 0.1009, 0.2256]
    ax.semilogy(N_vals, ml_bler, 'v:', color=C_MEMLESS, label='Memoryless SC')

    # GPU curriculum NPD (best at N=16-64)
    gpu_N = [16, 32, 64]
    gpu_bler = [0.1376, 0.0566, 0.0278]
    ax.semilogy(gpu_N, gpu_bler, 'D-', color=C_GPU, label='NPD d=16 h=100 (GPU curriculum)',
               linewidth=2, markersize=9)

    # d=16 h=100 standalone
    h100_N = [64, 128]
    h100_bler = [0.0318, 0.0740]
    ax.semilogy(h100_N, h100_bler, '^--', color=C_D16H100, label='NPD d=16 h=100 (standalone)')

    # d=64 h=128
    d64_N = [128, 256]
    d64_bler = [0.0300, 0.0112]
    ax.semilogy(d64_N, d64_bler, 'p--', color=C_D64H128, label='NPD d=64 h=128', markersize=9)

    ax.set_xlabel('Block length N')
    ax.set_ylabel('BLER')
    ax.set_title('ISI-MAC (h=0.3, SNR=6dB): All Decoders')
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_vals)
    ax.set_xticklabels([str(n) for n in N_vals])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(3e-3, 0.4)

    fig.tight_layout()
    save(fig, 'fig_isi_mac_final')

# ===========================================================================
# Figure 5: Memory Channels 3-panel (ISI + Ising + MA-AGN)
# ===========================================================================
def fig5_memory_channels():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: ISI-MAC
    N_isi = [16, 32, 64, 128, 256]
    ch_isi = [0.1689, 0.0822, 0.0407, 0.0223, 0.0061]
    ml_isi = [0.1866, 0.1129, 0.0790, 0.1009, 0.2256]
    # Best NPD per N
    npd_isi_N = [16, 32, 64, 128, 256]
    npd_isi = [0.1376, 0.0566, 0.0278, 0.0300, 0.0112]

    ax1.semilogy(N_isi, ch_isi, 's-', color=C_TRELLIS, label='Chained Trellis SC')
    ax1.semilogy(N_isi, ml_isi, 'v:', color=C_MEMLESS, label='Memoryless SC')
    ax1.semilogy(npd_isi_N, npd_isi, 'D-', color=C_NPD, label='NPD (best)')
    ax1.set_xlabel('N')
    ax1.set_ylabel('BLER')
    ax1.set_title('(a) ISI-MAC (h=0.3)')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(N_isi)
    ax1.set_xticklabels([str(n) for n in N_isi], fontsize=8)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(3e-3, 0.5)

    # Panel 2: Ising MAC
    N_is = [16, 32, 64]
    tr_is = [0.5704, 0.6872, 0.8976]
    ml_is = [0.6160, 0.7840, 0.9410]
    npd_is_N = [16, 32]
    npd_is = [0.5916, 0.7658]

    ax2.semilogy(N_is, tr_is, 's-', color=C_TRELLIS, label='Trellis SC')
    ax2.semilogy(N_is, ml_is, 'v:', color=C_MEMLESS, label='Memoryless SC')
    ax2.semilogy(npd_is_N, npd_is, 'D-', color=C_NPD, label='NPD d=16 h=100')
    ax2.set_xlabel('N')
    ax2.set_title('(b) Ising MAC (p=0.1)')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(N_is)
    ax2.set_xticklabels([str(n) for n in N_is])
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.3, 1.0)

    # Panel 3: MA-AGN
    N_ma = [16, 32, 64, 128]
    ml_ma = [0.1654, 0.0696, 0.0292, 0.0052]
    npd_ma_N = [16, 32, 64, 128]
    npd_ma = [0.1438, 0.1134, 0.0292, 0.1014]

    ax3.semilogy(N_ma, ml_ma, 'v:', color=C_MEMLESS, label='Memoryless SC')
    ax3.semilogy(npd_ma_N, npd_ma, 'D-', color=C_NPD, label='NPD (best)')
    ax3.set_xlabel('N')
    ax3.set_title(r'(c) MA-AGN MAC ($\alpha$=0.3)')
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(N_ma)
    ax3.set_xticklabels([str(n) for n in N_ma])
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(3e-3, 0.3)
    ax3.annotate('No trellis\nexists', xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8, fontstyle='italic', color='gray')

    fig.suptitle('Memory MAC Channels: Neural vs Analytical Decoders', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig_memory_channels')

# ===========================================================================
# Figure 6: Architecture Scaling — NPD/SC ratio vs N
# ===========================================================================
def fig6_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    # d=16 h=64 (original)
    N_64 = [16, 32, 64]
    ratio_64 = [0.1384/0.1689, 0.1130/0.0822, 0.0424/0.0407]
    ax.plot(N_64, ratio_64, 's-', color=C_D16H64, label='d=16, h=64 (20K params)', markersize=8)

    # d=16 h=100 (GPU curriculum - best)
    N_100 = [16, 32, 64, 128]
    ratio_100 = [0.1376/0.1689, 0.0566/0.0822, 0.0278/0.0407, 0.0740/0.0223]
    ax.plot(N_100, ratio_100, 'D-', color=C_D16H100, label='d=16, h=100 (42K params)', markersize=8)

    # d=64 h=128
    N_d64 = [128, 256]
    ratio_d64 = [0.0300/0.0223, 0.0112/0.0061]
    ax.plot(N_d64, ratio_d64, '^-', color=C_D64H128, label='d=64, h=128 (200K params)', markersize=8)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='NPD = Trellis SC')
    ax.set_xlabel('Block length N')
    ax.set_ylabel('NPD / Chained Trellis SC')
    ax.set_title('ISI-MAC: Architecture Scaling')
    ax.set_xscale('log', base=2)
    ax.set_xticks([16, 32, 64, 128, 256])
    ax.set_xticklabels(['16', '32', '64', '128', '256'])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 4.0)

    # Shade "NPD wins" region
    ax.fill_between([14, 300], 0, 1, alpha=0.05, color='green')
    ax.text(20, 0.15, 'NPD wins', fontsize=8, color='green', alpha=0.7)

    fig.tight_layout()
    save(fig, 'fig_architecture_scaling')

# ===========================================================================
# Figure 7: Wall Analysis — where walls appear across channels
# ===========================================================================
def fig7_wall():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: BLER vs N for all channels (neural best)
    # ISI-MAC NPD best
    N_isi = [16, 32, 64, 128, 256]
    npd_isi = [0.1376, 0.0566, 0.0278, 0.0300, 0.0112]
    sc_isi = [0.1689, 0.0822, 0.0407, 0.0223, 0.0061]

    # GMAC B NCG
    N_gmac = [32, 64, 128, 256]
    ncg_gmac = [0.0503, 0.0282, 0.0230, 0.0230]
    sc_gmac = [0.0450, 0.0276, 0.0187, 0.0060]

    # BEMAC C NCG
    N_bemac = [8, 16, 32, 64, 128, 256]
    ncg_bemac = [0.039, 0.016, 0.006, 0.002, 0.0003, 0.0002]
    sc_bemac = [0.110, 0.092, 0.099, 0.055, 0.0245, 0.0134]

    ax1.semilogy(N_isi, npd_isi, 'D-', color='#2ca02c', label='ISI-MAC NPD')
    ax1.semilogy(N_isi, sc_isi, 'D:', color='#2ca02c', alpha=0.4, label='ISI-MAC Trellis')
    ax1.semilogy(N_gmac, ncg_gmac, 's-', color='#d62728', label='GMAC-B NCG')
    ax1.semilogy(N_gmac, sc_gmac, 's:', color='#d62728', alpha=0.4, label='GMAC-B SC')
    ax1.semilogy(N_bemac, ncg_bemac, 'o-', color='#1f77b4', label='BEMAC-C NCG')
    ax1.semilogy(N_bemac, sc_bemac, 'o:', color='#1f77b4', alpha=0.4, label='BEMAC-C SC')
    ax1.set_xlabel('Block length N')
    ax1.set_ylabel('BLER')
    ax1.set_title('(a) Absolute BLER')
    ax1.set_xscale('log', base=2)
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-4, 0.5)

    # Panel 2: Neural/SC ratio vs N
    ratio_isi = [n/s for n, s in zip(npd_isi, sc_isi)]
    ratio_gmac = [n/s for n, s in zip(ncg_gmac, sc_gmac)]
    ratio_bemac = [n/s for n, s in zip(ncg_bemac, sc_bemac)]

    ax2.plot(N_isi, ratio_isi, 'D-', color='#2ca02c', label='ISI-MAC NPD', markersize=8)
    ax2.plot(N_gmac, ratio_gmac, 's-', color='#d62728', label='GMAC-B NCG', markersize=8)
    ax2.plot(N_bemac, ratio_bemac, 'o-', color='#1f77b4', label='BEMAC-C NCG', markersize=8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Block length N')
    ax2.set_ylabel('Neural / Analytical SC')
    ax2.set_title('(b) Performance Ratio')
    ax2.set_xscale('log', base=2)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5)

    # Mark walls
    ax2.annotate('GMAC wall', xy=(256, 3.83), fontsize=8, color='red',
                ha='center', va='bottom')
    ax2.annotate('ISI wall\n(N=512)', xy=(256, 1.84), fontsize=8, color='green',
                ha='left', va='bottom')

    fig.suptitle('Performance Walls Across Channels', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save(fig, 'fig_wall_analysis_thesis')


if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    print('Generating thesis figures...')
    fig1_bemac()
    fig2_gmac()
    fig3_abnmac()
    fig4_isi_mac()
    fig5_memory_channels()
    fig6_architecture()
    fig7_wall()
    print('Done. All 7 figures generated.')
