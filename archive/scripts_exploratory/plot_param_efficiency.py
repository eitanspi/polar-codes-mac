#!/usr/bin/env python3
"""
plot_param_efficiency.py
========================
Plot parameter efficiency: BLER vs model parameter count for ISI-MAC.
Shows d=16/h=100 is potentially more parameter-efficient than d=64/h=128.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data: (model_config, N, params_stage1, BLER, label)
data = [
    # d=16, h=64 models (existing)
    ('d16h64', 16, 20465, 0.143, 'N=16'),
    ('d16h64', 32, 20465, 0.081, 'N=32'),
    ('d16h64', 64, 20465, 0.046, 'N=64'),
    # d=64, h=128 models
    ('d64h128', 128, 114241, 0.030, 'N=128'),
    ('d64h128', 256, 114241, 0.011, 'N=256'),
]

# Trellis baselines for reference
trellis_chained = {16: 0.169, 32: 0.082, 64: 0.041, 128: 0.022, 256: 0.006}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: BLER vs N for different model sizes
Ns_d16 = [16, 32, 64]
bler_d16 = [0.143, 0.081, 0.046]
Ns_d64 = [128, 256]
bler_d64 = [0.030, 0.011]
Ns_trellis = sorted(trellis_chained.keys())
bler_trellis = [trellis_chained[n] for n in Ns_trellis]

ax1.semilogy(Ns_d16, bler_d16, 'o-', color='#b2182b', markersize=8, linewidth=2,
             label='NPD d=16 h=64 (20K params)')
ax1.semilogy(Ns_d64, bler_d64, '^-', color='#d6604d', markersize=9, linewidth=2,
             label='NPD d=64 h=128 (114K params)')
ax1.semilogy(Ns_trellis, bler_trellis, 's--', color='#67a9cf', markersize=7, linewidth=1.5,
             label='Chained Trellis SC (analytical)')

ax1.set_xlabel('Block length N', fontsize=12)
ax1.set_ylabel('Chained BLER', fontsize=12)
ax1.set_title('ISI-MAC BLER vs N', fontsize=12)
ax1.set_xscale('log', base=2)
ax1.set_xticks([16, 32, 64, 128, 256])
ax1.set_xticklabels(['16', '32', '64', '128', '256'])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: Parameter efficiency (params per 0.01 BLER reduction)
# Show params vs achieved BLER
configs = [
    (20465, 0.143, 'N=16\nd=16', '#b2182b'),
    (20465, 0.081, 'N=32\nd=16', '#ef8a62'),
    (20465, 0.046, 'N=64\nd=16', '#fddbc7'),
    (42461, None, 'N=64\nd=16 h=100\n(training...)', '#999999'),  # placeholder
    (114241, 0.030, 'N=128\nd=64', '#d6604d'),
    (114241, 0.011, 'N=256\nd=64', '#67001f'),
]

for params, bler, label, color in configs:
    if bler is not None:
        ax2.scatter(params, bler, s=100, color=color, zorder=5, edgecolors='black', linewidth=0.5)
        ax2.annotate(label, (params, bler), textcoords="offset points",
                     xytext=(10, 5), fontsize=8)

ax2.set_xlabel('Stage 1 parameters', fontsize=12)
ax2.set_ylabel('Stage 1 BLER', fontsize=12)
ax2.set_title('Parameter Efficiency', fontsize=12)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

out_dir = os.path.join(_ROOT, 'docs', 'paper_figures')
for ext in ['pdf', 'png']:
    out_path = os.path.join(out_dir, f'fig_isi_mac_param_efficiency.{ext}')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')

plt.close()
