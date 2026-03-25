"""
plot_bler_vs_N_all_classes.py — Combined BLER vs N plot for BE-MAC Classes A, B, C.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 11
import matplotlib.pyplot as plt

# Load data
classes_data = {}
for cls in ["A", "B", "C"]:
    path = os.path.join(RESULTS_DIR, f"bler_vs_N_bemac_class{cls}_L1.json")
    if os.path.exists(path):
        with open(path) as f:
            classes_data[cls] = json.load(f)
        print(f"Loaded Class {cls}: {path}")
    else:
        print(f"WARNING: Missing {path}")

# Plot styles
styles = {
    "A": {"color": "tab:red",   "marker": "o", "label": "Class A ($R_u$=0.75, $R_v$=0.75)"},
    "B": {"color": "tab:blue",  "marker": "s", "label": "Class B ($R_u$=0.625, $R_v$=0.875)"},
    "C": {"color": "tab:green", "marker": "^", "label": "Class C ($R_u$=0.5, $R_v$=1.0)"},
}

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

for cls, data in classes_data.items():
    results = data["results"]
    ns = [r["n"] for r in results]
    blers = [r["bler"] for r in results]

    # Filter out zero BLER for log scale
    ns_plot = [n for n, b in zip(ns, blers) if b > 0]
    bl_plot = [b for b in blers if b > 0]

    if not ns_plot:
        print(f"Class {cls}: all BLER = 0, skipping")
        continue

    st = styles[cls]
    ax.semilogy(ns_plot, bl_plot,
                color=st["color"], marker=st["marker"],
                markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                linewidth=1.5, label=st["label"])

rho = classes_data.get("A", classes_data.get("B", {})).get("rho", 0.5)
sum_rate = classes_data.get("A", classes_data.get("B", {})).get("sum_rate_target", 0.75)

ax.set_xlabel(r'$n = \log_2 N$', fontsize=13)
ax.set_ylabel('BLER', fontsize=13)
ax.set_title(
    f'BE-MAC, SC (L=1), Analytical Design, '
    + r'$\rho$=' + f'{rho}, '
    + r'$R_u+R_v$=' + f'{sum_rate:.2f}',
    fontsize=11)

ax.set_xticks(range(3, 11))
ax.set_xticklabels([f'{i}\n({1<<i})' for i in range(3, 11)], fontsize=9)
ax.set_ylim(1e-5, 1)
ax.grid(True, which='major', alpha=0.4, linewidth=0.5)
ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)
ax.tick_params(which='both', direction='in', top=True, right=True)
ax.legend(fontsize=10, loc='upper right')

fig.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "bler_vs_N_all_classes_L1")
fig.savefig(fig_path + ".pdf", dpi=150)
fig.savefig(fig_path + ".png", dpi=150)
print(f"Saved: {fig_path}.pdf")
print(f"Saved: {fig_path}.png")

# Print summary table
print()
print("=" * 80)
print(f"  {'Class':>5s}  {'N':>5s}  {'n':>2s}  {'ku':>4s}  {'kv':>4s}  {'Ru+Rv':>7s}  {'BLER':>10s}  {'n_cw':>7s}")
print("-" * 80)
for cls in ["A", "B", "C"]:
    if cls not in classes_data:
        continue
    for r in classes_data[cls]["results"]:
        print(f"  {cls:>5s}  {r['N']:5d}  {r['n']:2d}  {r['ku']:4d}  {r['kv']:4d}  "
              f"{r['Ru']+r['Rv']:7.4f}  {r['bler']:10.6f}  {r['n_codewords']:7d}")
    print("-" * 80)
print("=" * 80)
