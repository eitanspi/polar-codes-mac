#!/usr/bin/env python3
"""
Plot the effect of hidden width on NPD performance.
Key thesis result: h=100 matters more than d for tree operation quality.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: ISI-MAC N=64 architecture comparison
ax = axes[0]
configs = ['d=16\nh=64', 'd=16\nh=100', 'd=64\nh=128']
bler_n64 = [0.046, 0.032, None]  # d=64 not trained at N=64
params_n64 = [20000, 42000, 200000]

# ISI-MAC N=128 full comparison
bler_n128 = [None, 0.081, 0.030]  # d=16 h=64 not evaluated at N=128 (was ~0.16)
configs_128 = ['d=16\nh=64', 'd=16\nh=100', 'd=64\nh=128']

# Baselines
ax.axhline(y=0.041, color='gray', ls='--', alpha=0.5, label='Chained Trellis SC (N=64)')
ax.axhline(y=0.026, color='gray', ls=':', alpha=0.5, label='Joint Trellis SC (N=64)')

# Plot N=64 bars
x = np.arange(3)
bars = ax.bar(x[:2], [0.046, 0.032], 0.6, color=['steelblue', 'coral'],
              label='NPD Chained BLER (N=64)')
ax.bar_label(bars, fmt='%.3f', padding=3)

ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel('Chained BLER', fontsize=12)
ax.set_title('ISI-MAC N=64: Hidden Width Effect\n(h=64 vs h=100)', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.06)
ax.grid(True, alpha=0.3, axis='y')

# Annotate improvement
ax.annotate('30% improvement\n(h: 64->100)', xy=(0.5, 0.039),
            fontsize=10, ha='center', color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred'),
            xytext=(0.5, 0.048))

# Panel 2: Parameter efficiency
ax = axes[1]
# All configs with their best BLER at each N
configs_all = [
    ('d=16 h=64',  20,  [(16, 0.143), (32, 0.081), (64, 0.046)]),
    ('d=16 h=100', 42,  [(64, 0.032), (128, 0.081)]),
    ('d=64 h=128', 200, [(128, 0.030), (256, 0.011)]),
]

for name, params_k, points in configs_all:
    Ns = [p[0] for p in points]
    blers = [p[1] for p in points]
    ax.semilogy(Ns, blers, 'o-', label=f'{name} ({params_k}K params)', ms=7, lw=2)

# Baselines
N_trellis = [16, 32, 64, 128, 256]
trellis_chained = [0.169, 0.082, 0.041, 0.022, 0.006]
ax.semilogy(N_trellis, trellis_chained, 'k--^', label='Chained Trellis SC', ms=5, lw=1)

ax.set_xlabel('Block Length N', fontsize=12)
ax.set_ylabel('Chained BLER', fontsize=12)
ax.set_title('NPD Architecture Scaling\n(ISI-MAC, SNR=6 dB)', fontsize=12)
ax.legend(fontsize=9)
ax.set_xscale('log', base=2)
ax.set_xticks([16, 32, 64, 128, 256])
ax.set_xticklabels(['16', '32', '64', '128', '256'])
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0.005, 0.2)

plt.tight_layout()
out_dir = os.path.join(_ROOT, 'project_summary', 'plots')
for ext in ['png', 'pdf']:
    path = os.path.join(out_dir, f'fig_hidden_width_effect.{ext}')
    fig.savefig(path, dpi=150)
    print(f'Saved: {path}')
plt.close()
