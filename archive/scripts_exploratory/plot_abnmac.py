"""
plot_abnmac.py — Generate Fig 3 (rate region) and Fig 4 (BLER vs sum rate)
for ABN-MAC, matching the style of Önay ISIT 2013 Figures 3 & 4.

Usage:
    python scripts/plot_abnmac.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 11
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ABN-MAC capacity region
I_ZX = 0.4
I_ZY = 0.4
I_ZX_given_Y = 0.8
I_ZY_given_X = 0.8
I_ZXY = 1.2


def load_results(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_points(data):
    """Extract (Ru, Rv, sum_rate, bler) from results."""
    if data is None:
        return [], [], [], []
    Ru, Rv, sr, bler = [], [], [], []
    for r in data["results"]:
        if r.get("skipped") or r["bler"] is None:
            continue
        Ru.append(r["Ru"])
        Rv.append(r["Rv"])
        sr.append(r["Ru"] + r["Rv"])
        bler.append(r["bler"])
    return Ru, Rv, sr, bler


def interpolate_bler_target(sum_rates, blers, target_bler):
    """Find sum rate where BLER = target via log-linear interpolation."""
    if len(sum_rates) < 2:
        return None
    sr = np.array(sum_rates)
    bl = np.array(blers)
    log_bl = np.log10(np.maximum(bl, 1e-10))
    log_target = np.log10(target_bler)
    for i in range(len(sr) - 1):
        if (log_bl[i] <= log_target <= log_bl[i + 1]) or \
           (log_bl[i] >= log_target >= log_bl[i + 1]):
            frac = (log_target - log_bl[i]) / (log_bl[i + 1] - log_bl[i])
            return sr[i] + frac * (sr[i + 1] - sr[i])
    return None


# ════════════════════════════════════════════════════════════════════
#  Load all results
# ════════════════════════════════════════════════════════════════════

files = {
    ("B", 1): "sim_abnmac_B_mc_L1.json",
    ("B", 32): "sim_abnmac_B_mc_L32.json",
    ("C", 1): "sim_abnmac_C_mc_L1.json",
    ("C", 32): "sim_abnmac_C_mc_L32.json",
}

data = {}
for key, fname in files.items():
    data[key] = load_results(fname)

# Paper-style markers and colors per class (matching Önay Fig 3 style)
# Paper uses: *, □, ◇, ○ for N=2^10, 2^12, 2^14, 2^16
# We only have N=1024 = 2^10
CLASS_CFG = {
    "B": {"marker": "s", "color": "blue",  "dir": (0.5, 0.7)},
    "C": {"marker": "^", "color": "green", "dir": (0.4, 0.8)},
}

# ════════════════════════════════════════════════════════════════════
#  Figure 3: Operating rates at BLER=10^-2 (rate region)
# ════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(1, 1, figsize=(5.5, 5.5))

# Capacity region boundary (like paper's red lines)
bnd_Rx = [0, I_ZX_given_Y, I_ZX, 0, 0]
bnd_Ry = [0, 0, I_ZY_given_X, I_ZY_given_X, 0]
ax3.plot(bnd_Rx, bnd_Ry, 'r-', linewidth=1.5)
# Dominant face
ax3.plot([I_ZX, I_ZX_given_Y], [I_ZY_given_X, I_ZY], 'r-', linewidth=1.5)

# Direction lines (dashed, like paper)
for cls, cfg in CLASS_CFG.items():
    Ru_d, Rv_d = cfg["dir"]
    scale = 1.1 / (Ru_d + Rv_d)  # extend past capacity
    ax3.plot([0, Ru_d * scale], [0, Rv_d * scale],
             'k--', linewidth=0.5, alpha=0.3)

# Plot operating points at BLER = 10^-2
BLER_TARGET = 1e-2
for cls in ["B", "C"]:
    cfg = CLASS_CFG[cls]
    Ru_dir, Rv_dir = cfg["dir"]
    for L in [1, 32]:
        d = data.get((cls, L))
        if d is None:
            continue
        Ru_pts, Rv_pts, sr_pts, bler_pts = get_points(d)
        if not sr_pts:
            continue
        target_sr = interpolate_bler_target(sr_pts, bler_pts, BLER_TARGET)
        if target_sr is not None:
            total_dir = Ru_dir + Rv_dir
            target_Ru = target_sr * Ru_dir / total_dir
            target_Rv = target_sr * Rv_dir / total_dir
            ls = '--' if L == 32 else '-'
            fc = cfg["color"] if L == 1 else 'white'
            ax3.plot(target_Ru, target_Rv, marker=cfg["marker"],
                     color=cfg["color"], markerfacecolor=fc,
                     markeredgecolor=cfg["color"],
                     markersize=9, markeredgewidth=1.5)

# Class labels (circled, like paper)
for cls, pos in [("B", (0.36, 0.51)), ("C", (0.24, 0.63))]:
    ax3.annotate(cls, xy=pos, fontsize=16, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='circle,pad=0.2', facecolor='white',
                           edgecolor='black', linewidth=1.5))

# Legend
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
           markersize=10, label=r'N=$2^{10}$'),
    Line2D([0], [0], color='black', linestyle='-', linewidth=1,
           label='L=1'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=1,
           label='L=32'),
]
ax3.legend(handles=legend_elements, loc='lower left', fontsize=10,
           framealpha=0.9)

ax3.set_xlabel(r'$R_x$', fontsize=14)
ax3.set_ylabel(r'$R_y$', fontsize=14)
ax3.set_title(r'Operating rates for ABN-MAC at BLER=$10^{-2}$',
              fontsize=12)
ax3.set_xlim(0, 0.9)
ax3.set_ylim(0, 0.9)
ax3.set_xticks(np.arange(0, 1.0, 0.2))
ax3.set_yticks(np.arange(0, 1.0, 0.2))
ax3.set_aspect('equal')
ax3.grid(False)

fig3.tight_layout()
fig3.savefig(os.path.join(FIGURES_DIR, "fig3_abnmac_rate_region.pdf"), dpi=150)
fig3.savefig(os.path.join(FIGURES_DIR, "fig3_abnmac_rate_region.png"), dpi=150)
print(f"Saved fig3 to figures/fig3_abnmac_rate_region.{{pdf,png}}")

# ════════════════════════════════════════════════════════════════════
#  Figure 4: BLER vs sum rate (matching paper's Fig 4 style)
# ════════════════════════════════════════════════════════════════════

fig4, ax4 = plt.subplots(1, 1, figsize=(5.5, 5))

# Paper style: solid = L=1, dashed = L=32
# Different markers per N (we only have N=1024)
for (cls, L) in [("B", 1), ("B", 32), ("C", 1), ("C", 32)]:
    d = data.get((cls, L))
    if d is None:
        continue
    _, _, sr, bler = get_points(d)
    if not sr:
        continue
    # Filter zero BLER
    sr_plot = [s for s, b in zip(sr, bler) if b > 0]
    bl_plot = [b for b in bler if b > 0]
    if not sr_plot:
        continue

    cfg = CLASS_CFG[cls]
    ls = '-' if L == 1 else '--'
    fc = cfg["color"] if L == 1 else 'white'
    label = f'Class {cls}, L={L}'

    ax4.semilogy(sr_plot, bl_plot,
                 marker=cfg["marker"], color=cfg["color"],
                 linestyle=ls, linewidth=1.5,
                 markerfacecolor=fc, markeredgecolor=cfg["color"],
                 markeredgewidth=1.5, markersize=7,
                 label=label)

ax4.set_xlabel(r'$R_x+R_y$', fontsize=14)
ax4.set_ylabel('BLER', fontsize=14)
ax4.set_title('BLER performance for ABN-MAC w.r.t. sum rate', fontsize=12)
ax4.set_xlim(0.4, 1.3)
ax4.set_ylim(1e-4, 1)
ax4.set_xticks(np.arange(0.5, 1.3, 0.1))
ax4.legend(fontsize=9, loc='lower right', framealpha=0.9)

# Minor grid like paper
ax4.grid(True, which='major', alpha=0.4, linewidth=0.5)
ax4.grid(True, which='minor', alpha=0.15, linewidth=0.3)
ax4.tick_params(which='both', direction='in', top=True, right=True)

fig4.tight_layout()
fig4.savefig(os.path.join(FIGURES_DIR, "fig4_abnmac_bler_vs_sumrate.pdf"), dpi=150)
fig4.savefig(os.path.join(FIGURES_DIR, "fig4_abnmac_bler_vs_sumrate.png"), dpi=150)
print(f"Saved fig4 to figures/fig4_abnmac_bler_vs_sumrate.{{pdf,png}}")

if os.environ.get("DISPLAY") or sys.platform == "darwin":
    try:
        plt.show(block=False)
        plt.pause(0.1)
    except Exception:
        pass
