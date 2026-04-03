#!/usr/bin/env python3
"""Main paper figure: BEMAC + GMAC side by side."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'lines.linewidth': 1.5, 'lines.markersize': 6,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# ── Panel A: BEMAC ──
Ns = [32, 64, 128, 256]
sc    = [0.008, 0.0056, 0.002, 8e-5]
nn_sc = [0.0088, 0.003, 0.0012, 4e-5]
scl4  = [0.0037, 0.001, 0.0017]
nn_scl4 = [0.0073, 0.0007, 0.0007]

ax1.plot(Ns, sc, 'b-o', label='SC', zorder=3)
ax1.plot(Ns, nn_sc, 'r-s', label='NN-SC', zorder=3)
ax1.plot(Ns[:3], scl4, 'g-D', label='SCL $L{=}4$', zorder=3)
ax1.plot(Ns[:3], nn_scl4, '-^', color='tab:orange', label='NN-SCL $L{=}4$', zorder=3)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xlabel('Block Length $N$')
ax1.set_ylabel('BLER')
ax1.set_title('(a) BEMAC: $Z = X + Y$')
ax1.set_xticks(Ns)
ax1.set_xticklabels([str(n) for n in Ns])
ax1.grid(True, which='both', ls='--', alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_ylim(1e-5, 2e-2)

# Add annotation
ax1.annotate('NN beats SC\nat $N \\geq 64$', xy=(64, 0.003), xytext=(100, 0.008),
             arrowprops=dict(arrowstyle='->', color='red', lw=1), fontsize=8, color='red')

# ── Panel B: GMAC ──
Ns_g = [32, 64, 128, 256, 512]
sc_g    = [0.046, 0.025, 0.016, 0.005, 0.001]
nn_g    = [0.046, 0.026, 0.017, 0.015, 0.008]
scl4_g  = [0.026, 0.013, 0.008, 0.0005]
nn_ca_scl = [0.009, 0.002, 0.006]

ax2.plot(Ns_g, sc_g, 'b-o', label='SC', zorder=3)
ax2.plot(Ns_g, nn_g, 'r-s', label='NN-SC', zorder=3)
ax2.plot(Ns_g[:4], scl4_g, 'g-D', label='SCL $L{=}4$', zorder=3)
ax2.plot(Ns_g[:3], nn_ca_scl, '-v', color='darkviolet', label='NN-CA-SCL $L{=}4$', zorder=3, markersize=7)
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_xlabel('Block Length $N$')
ax2.set_ylabel('BLER')
ax2.set_title('(b) GMAC: $Z = (1{-}2X) + (1{-}2Y) + W$, SNR=6dB')
ax2.set_xticks(Ns_g)
ax2.set_xticklabels([str(n) for n in Ns_g])
ax2.grid(True, which='both', ls='--', alpha=0.3)
ax2.legend(loc='upper right')
ax2.set_ylim(1e-4, 1e-0)

# Add annotation for the gap
ax2.annotate('Gap at\n$N \\geq 256$', xy=(256, 0.015), xytext=(350, 0.04),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1), fontsize=8, color='gray')

fig.tight_layout(w_pad=2)
fig.savefig(os.path.join(OUT_DIR, 'fig_main_combined.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'fig_main_combined.pdf'), bbox_inches='tight')
print('Saved fig_main_combined.png / .pdf')
plt.close(fig)
