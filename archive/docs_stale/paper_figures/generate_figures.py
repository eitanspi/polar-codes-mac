#!/usr/bin/env python3
"""Generate publication-quality figures for the polar codes MAC paper."""

import json
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
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
    'figure.dpi': 150,
})

# ── Load data ────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(OUT_DIR, '..', '..', 'results')

with open(os.path.join(DATA_ROOT, 'bemac', 'bemac_comprehensive_paper.json')) as f:
    bemac = json.load(f)

with open(os.path.join(DATA_ROOT, 'complexity_analysis.json')) as f:
    complexity = json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    print(f'  Saved {name}.png / .pdf')
    plt.close(fig)


# =====================================================================
# Figure 1: BEMAC BLER vs N (Class B)
# =====================================================================
print('Figure 1: BEMAC Class B')
cb = bemac['Class_B']
Ns_bemac = [32, 64, 128, 256, 512, 1024]

sc_vals, nn_vals, scl4_vals, nn_scl4_vals = [], [], [], []
Ns_sc, Ns_nn, Ns_scl4, Ns_nnscl4 = [], [], [], []

for n in Ns_bemac:
    d = cb[str(n)]
    if d['SC'] is not None and d['SC'] > 0:
        Ns_sc.append(n); sc_vals.append(d['SC'])
    if d['NN_SC'] is not None and d['NN_SC'] > 0:
        Ns_nn.append(n); nn_vals.append(d['NN_SC'])
    if d.get('SCL_L4') is not None and d['SCL_L4'] > 0:
        Ns_scl4.append(n); scl4_vals.append(d['SCL_L4'])
    if d.get('NN_SCL_L4') is not None and d['NN_SCL_L4'] > 0:
        Ns_nnscl4.append(n); nn_scl4_vals.append(d['NN_SCL_L4'])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Ns_sc, sc_vals, 'b-o', label='SC', zorder=3)
ax.plot(Ns_nn, nn_vals, 'r-s', label='NN-SC', zorder=3)
ax.plot(Ns_scl4, scl4_vals, 'g-D', label='SCL $L{=}4$', zorder=3)
ax.plot(Ns_nnscl4, nn_scl4_vals, '-^', color='tab:orange', label='NN-SCL $L{=}4$', zorder=3)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Block Length $N$')
ax.set_ylabel('BLER')
ax.set_title('BEMAC Class B: BLER vs Block Length')
ax.set_xticks(Ns_bemac)
ax.set_xticklabels([str(n) for n in Ns_bemac])
ax.grid(True, which='both', ls='--', alpha=0.3)
ax.legend(loc='best')
fig.tight_layout()
save(fig, 'fig1_bemac_classB')


# =====================================================================
# Figure 2: GMAC BLER vs N (Class B, SNR=6dB)
# =====================================================================
print('Figure 2: GMAC Class B')
Ns_gmac = [32, 64, 128, 256, 512]
gmac_sc   = [0.046, 0.025, 0.016, 0.005, 0.001]
gmac_nn   = [0.046, 0.026, 0.017, 0.015, 0.008]
gmac_scl4 = [0.026, 0.013, 0.008, 0.0005, None]  # N=512 is 0.0
gmac_nnscl4 = [0.022, 0.013, 0.015, 0.026, 0.045]

gmac_nn_d32 = [0.037, 0.020, 0.019]  # d=32 model

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Ns_gmac, gmac_sc, 'b-o', label='SC', zorder=3)
ax.plot(Ns_gmac, gmac_nn, 'r-s', label='NN-SC ($d{=}16$)', zorder=3)
ax.plot(Ns_gmac[:3], gmac_nn_d32, '--^', color='tab:red', label='NN-SC ($d{=}32$)', zorder=4, markersize=7)

# SCL L=4: skip N=512 (zero errors)
Ns_scl_g = [n for n, v in zip(Ns_gmac, gmac_scl4) if v is not None and v > 0]
scl_g    = [v for v in gmac_scl4 if v is not None and v > 0]
ax.plot(Ns_scl_g, scl_g, 'g-D', label='SCL $L{=}4$', zorder=3)

ax.plot(Ns_gmac, gmac_nnscl4, '-^', color='tab:orange', label='NN-SCL $L{=}4$', zorder=3)

ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Block Length $N$')
ax.set_ylabel('BLER')
ax.set_title('GMAC Class B (SNR = 6 dB): BLER vs Block Length')
ax.set_xticks(Ns_gmac)
ax.set_xticklabels([str(n) for n in Ns_gmac])
ax.grid(True, which='both', ls='--', alpha=0.3)
ax.legend(loc='best')
fig.tight_layout()
save(fig, 'fig2_gmac_classB')


# =====================================================================
# Figure 3: Complexity -- FLOPs vs N
# =====================================================================
print('Figure 3: FLOPs')
flops = complexity['flops']
Ns_f = sorted(int(k) for k in flops)
sc_flops = [flops[str(n)]['sc_flops'] for n in Ns_f]
nn_flops = [flops[str(n)]['nn_flops'] for n in Ns_f]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Ns_f, sc_flops, 'b-o', label='SC', zorder=3)
ax.plot(Ns_f, nn_flops, 'r-s', label='NN-SC', zorder=3)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Block Length $N$')
ax.set_ylabel('FLOPs')
ax.set_title('Computational Complexity: FLOPs vs Block Length')
ax.set_xticks(Ns_f)
ax.set_xticklabels([str(n) for n in Ns_f])
ax.grid(True, which='both', ls='--', alpha=0.3)
ax.legend(loc='best')

# Annotate the constant ratio
mid = 2  # N=128
ratio = nn_flops[mid] / sc_flops[mid]
ax.annotate(f'$\\approx${ratio:.0f}$\\times$',
            xy=(Ns_f[mid], nn_flops[mid]),
            xytext=(Ns_f[mid]*0.35, nn_flops[mid]*3),
            arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.2),
            fontsize=11, color='dimgray')

fig.tight_layout()
save(fig, 'fig3_flops')


# =====================================================================
# Figure 4: Complexity -- Inference Time vs N
# =====================================================================
print('Figure 4: Inference Time')
itime = complexity['inference_time']
Ns_t = sorted(int(k) for k in itime)

sc_time, nn_time = [], []
Ns_sc_t, Ns_nn_t = [], []
scl_time = []
Ns_scl_t = []

for n in Ns_t:
    d = itime[str(n)]
    if 'SC' in d:
        Ns_sc_t.append(n)
        sc_time.append(d['SC']['median_ms'])
    if 'NN_SC' in d:
        Ns_nn_t.append(n)
        nn_time.append(d['NN_SC']['median_ms'])
    if 'SCL_L4' in d:
        Ns_scl_t.append(n)
        scl_time.append(d['SCL_L4']['median_ms'])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(Ns_sc_t, sc_time, 'b-o', label='SC', zorder=3)
ax.plot(Ns_nn_t, nn_time, 'r-s', label='NN-SC', zorder=3)
ax.plot(Ns_scl_t, scl_time, 'g-D', label='SCL $L{=}4$', zorder=3)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('Block Length $N$')
ax.set_ylabel('Inference Time (ms)')
ax.set_title('Inference Latency vs Block Length')
ax.set_xticks(Ns_t)
ax.set_xticklabels([str(n) for n in Ns_t])
ax.grid(True, which='both', ls='--', alpha=0.3)
ax.legend(loc='best')
fig.tight_layout()
save(fig, 'fig4_inference_time')

print('\nAll figures generated successfully.')
