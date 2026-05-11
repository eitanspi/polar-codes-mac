#!/usr/bin/env python3
"""Generate Fig 7: GMAC training convergence plots for the paper.

Uses actual eval points from training logs:
  - d=16 per-level model: train_gmac_perlevel.log (N=32,64,128)
                          train_n256_long.log (N=256)
  - d=32 model:           train_gmac_d32.log (N=32,64)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── IEEE-style defaults ──────────────────────────────────────────────
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

# =====================================================================
# Data from training logs (manually extracted eval checkpoints)
# =====================================================================

# --- d=16 per-level model: train_gmac_perlevel.log ---
# N=32: 20K iters, eval every 2K
d16_n32_iters = np.array([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000])
d16_n32_bler = np.array([1.0, 0.2277, 0.0907, 0.0857, 0.0767, 0.0643, 0.0637, 0.0537, 0.0547, 0.0520, 0.0533])
d16_n32_sc = 0.046

# N=64: 60K iters, eval every 5K
d16_n64_iters = np.array([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000])
d16_n64_bler = np.array([1.0, 0.2263, 0.1257, 0.0933, 0.0827, 0.0790, 0.0710, 0.0647, 0.0643, 0.0583, 0.0533, 0.0613, 0.0547])
d16_n64_sc = 0.025

# N=128: 75K iters shown, eval every 5K
d16_n128_iters = np.array([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000,
                           55000, 60000, 65000, 70000, 75000])
d16_n128_bler = np.array([1.0, 0.2645, 0.1515, 0.1245, 0.1095, 0.1060, 0.0905, 0.0870, 0.0780,
                          0.0645, 0.0665, 0.0620, 0.0655, 0.0555, 0.0600, 0.0590])
d16_n128_sc = 0.016

# N=256: train_n256_long.log — continued training, d=16, 100K iters
d16_n256_iters = np.array([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000,
                           50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000])
d16_n256_bler = np.array([1.0, 0.0230, 0.0210, 0.0200, 0.0230, 0.0300, 0.0250, 0.0140, 0.0180,
                          0.0230, 0.0160, 0.0090, 0.0170, 0.0100, 0.0120, 0.0130, 0.0140, 0.0180,
                          0.0130, 0.0170, 0.0120])
# Note: N=256 starts from pretrained checkpoint (initial BLER=0.029), so first point is warm
# Replace iter 0 with the initial checkpoint eval
d16_n256_bler[0] = 0.0290
d16_n256_sc = 0.005

# --- d=32 model: train_gmac_d32.log ---
# N=32: 15K iters, eval every 2K
d32_n32_iters = np.array([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
d32_n32_bler = np.array([1.0, 0.1327, 0.1143, 0.0707, 0.0657, 0.0457, 0.0570, 0.0617])
d32_n32_sc = 0.046

# N=64: 30K iters, eval every 3K
d32_n64_iters = np.array([0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000])
d32_n64_bler = np.array([1.0, 0.3193, 0.1493, 0.0913, 0.0667, 0.0563, 0.0557, 0.0457, 0.0643, 0.0640, 0.0510])
d32_n64_sc = 0.025


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    print(f'  Saved {name}.png / .pdf')
    plt.close(fig)


# =====================================================================
# Figure 7: Two-panel training convergence
# =====================================================================
print('Figure 7: Training convergence')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# ── Panel (a): d=16 per-level model ─────────────────────────────────
colors = {'N=32': '#1f77b4', 'N=64': '#ff7f0e', 'N=128': '#2ca02c', 'N=256': '#d62728'}
markers = {'N=32': 'o', 'N=64': 's', 'N=128': 'D', 'N=256': '^'}

for label, iters, bler, sc_ref in [
    ('N=32',  d16_n32_iters,  d16_n32_bler,  d16_n32_sc),
    ('N=64',  d16_n64_iters,  d16_n64_bler,  d16_n64_sc),
    ('N=128', d16_n128_iters, d16_n128_bler, d16_n128_sc),
    ('N=256', d16_n256_iters, d16_n256_bler, d16_n256_sc),
]:
    c = colors[label]
    m = markers[label]
    # Filter out the initial BLER=1.0 for log-scale clarity (except N=256 which starts warm)
    if label != 'N=256':
        mask = bler < 1.0
        ax1.plot(iters[mask] / 1000, bler[mask], f'-{m}', color=c, label=label,
                 markersize=5, markevery=1, zorder=3)
    else:
        ax1.plot(iters / 1000, bler, f'-{m}', color=c, label=label,
                 markersize=5, markevery=2, zorder=3)

    # SC dashed target line
    ax1.axhline(y=sc_ref, color=c, linestyle='--', linewidth=0.8, alpha=0.5)

# Add SC reference labels at right edge
sc_labels = [('$\\mathrm{SC}_{32}$', d16_n32_sc, colors['N=32']),
             ('$\\mathrm{SC}_{64}$', d16_n64_sc, colors['N=64']),
             ('$\\mathrm{SC}_{128}$', d16_n128_sc, colors['N=128']),
             ('$\\mathrm{SC}_{256}$', d16_n256_sc, colors['N=256'])]
for lbl, yval, c in sc_labels:
    ax1.text(106, yval, lbl, fontsize=7, color=c, va='center', style='italic')

ax1.set_xlabel('Training Iterations ($\\times 10^3$)')
ax1.set_ylabel('BLER')
ax1.set_yscale('log')
ax1.set_ylim(3e-3, 0.5)
ax1.set_xlim(-2, 115)
ax1.grid(True, which='both', ls='--', alpha=0.3)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.set_title('(a) Per-level model ($d{=}16$)')

# ── Panel (b): d=32 model ───────────────────────────────────────────
for label, iters, bler, sc_ref in [
    ('N=32', d32_n32_iters, d32_n32_bler, d32_n32_sc),
    ('N=64', d32_n64_iters, d32_n64_bler, d32_n64_sc),
]:
    c = colors[label]
    m = markers[label]
    mask = bler < 1.0
    ax2.plot(iters[mask] / 1000, bler[mask], f'-{m}', color=c, label=label,
             markersize=5, zorder=3)
    ax2.axhline(y=sc_ref, color=c, linestyle='--', linewidth=0.8, alpha=0.5)

ax2.text(32.5, d32_n32_sc, '$\\mathrm{SC}_{32}$', fontsize=7, color=colors['N=32'], va='center', style='italic')
ax2.text(32.5, d32_n64_sc, '$\\mathrm{SC}_{64}$', fontsize=7, color=colors['N=64'], va='center', style='italic')

ax2.set_xlabel('Training Iterations ($\\times 10^3$)')
ax2.set_ylabel('BLER')
ax2.set_yscale('log')
ax2.set_ylim(1e-2, 0.5)
ax2.set_xlim(-1, 38)
ax2.grid(True, which='both', ls='--', alpha=0.3)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.set_title('(b) Shared model ($d{=}32$)')

fig.suptitle('GMAC Class B Training Convergence (SNR = 6.0 dB)', fontsize=13, y=1.02)
fig.tight_layout()
save(fig, 'fig7_training_convergence')

print('Done.')
