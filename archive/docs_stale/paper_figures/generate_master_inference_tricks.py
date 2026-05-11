#!/usr/bin/env python3
"""Master plot: inference-time tricks summary across channels and N."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.6,
    'lines.markersize': 8,
    'figure.dpi': 150,
})

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    print(f'  Saved {name}.png / .pdf')


def main():
    # GMAC B
    N_g = np.array([32, 64, 128, 256])
    sc_g = np.array([0.047, 0.028, 0.020, 0.005])
    ncg_g = np.array([0.040, 0.026, 0.023, 0.023])
    # CRC-SCL best achieved per N (from validated runs, 2026-04)
    # N=32 L=4: 0.009; N=64 L=8: 0.002; N=128 L=8: 0.000 (use 1e-3 cap);
    # N=256 L=4 T=1.0: 6/2000 = 0.003 CI95 [0.0014, 0.0065]  (VALIDATED)
    crc_scl_g = np.array([0.009, 0.002, 1e-3, 0.003])
    crc_scl_g_errfloor = np.array([False, False, True, False])
    # N=256 CI band for validated result
    crc_scl_g_n256_ci = (0.0014, 0.0065)

    # BEMAC B
    N_b = np.array([16, 32, 64, 128, 256, 512, 1024])
    sc_b = np.array([0.011, 0.008, 0.006, 0.002, 8e-5, 1e-5, 1e-4])
    ncg_b = np.array([0.011, 0.009, 0.003, 0.001, 4e-5, 1e-5, 1e-4])
    # CRC-SCL BEMAC N=32,64,128 all hit 0 -> floor at 1e-5
    crc_scl_b_x = np.array([32, 64, 128])
    crc_scl_b = np.array([1e-5, 1e-5, 1e-5])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # GMAC panel
    ax1.plot(N_g, sc_g, 'o--', color='#1f77b4', label='SC analytical')
    ax1.plot(N_g, ncg_g, 's-', color='#d62728', label='NCG (greedy)')
    ax1.plot(N_g, crc_scl_g, '^-', color='#2ca02c',
             label='NN-CA-SCL L=4 (validated)')
    # mark error-floor points
    ax1.plot(N_g[crc_scl_g_errfloor], crc_scl_g[crc_scl_g_errfloor],
             'v', color='#2ca02c', markersize=14, markerfacecolor='none',
             markeredgewidth=2, label='0 errors observed')
    # N=256 validated CI
    ax1.errorbar([256], [0.003],
                 yerr=[[0.003 - crc_scl_g_n256_ci[0]],
                       [crc_scl_g_n256_ci[1] - 0.003]],
                 fmt='D', color='#ff7f0e', markersize=10, capsize=4,
                 label=f'N=256 L=4: 0.003 (2000 cw, 95% CI)')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xticks(N_g)
    ax1.set_xticklabels([str(n) for n in N_g])
    ax1.set_xlabel('Block length N')
    ax1.set_ylabel('BLER')
    ax1.set_title('GMAC Class B (SNR=6 dB)')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(loc='lower left', fontsize=8)

    # BEMAC panel
    ax2.plot(N_b, sc_b, 'o--', color='#1f77b4', label='SC analytical')
    ax2.plot(N_b, ncg_b, 's-', color='#d62728', label='NCG (greedy)')
    ax2.plot(crc_scl_b_x, crc_scl_b, '^-', color='#2ca02c', label='NN-CA-SCL (0 errors)')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xticks(N_b)
    ax2.set_xticklabels([str(n) for n in N_b])
    ax2.set_xlabel('Block length N')
    ax2.set_title('BEMAC Class B')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(loc='upper right')

    fig.suptitle('Neural SC decoders with CRC-aided list decoding', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_inference_tricks_master')
    plt.close(fig)


if __name__ == '__main__':
    main()
