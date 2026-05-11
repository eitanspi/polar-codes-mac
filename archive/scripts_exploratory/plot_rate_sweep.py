#!/usr/bin/env python3
"""
Plot the BEMAC Class C rate-BLER sweep:
  - x-axis: sum rate (R_u + R_v)
  - y-axis: BLER (log scale)
  - one curve per (decoder, N)
  - horizontal line at BLER=1e-3 to show the "real-world useful" target
  - annotate the rate at which each decoder crosses BLER=1e-3
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results/bemac/bemac_classC_rate_sweep.json"
OUT_PNG = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results/bemac/bemac_classC_rate_sweep.png"

# BEMAC capacity for context
BEMAC_CAPACITY = 1.5  # I(Z; X, Y) for symmetric BEMAC

def crossover_rate(rates, blers, target=1e-3):
    """Linear interpolation in log-space to find sum rate at BLER=target."""
    pts = [(r, b) for r, b in zip(rates, blers) if b is not None and b > 0]
    if len(pts) < 2:
        return None
    pts.sort()
    for i in range(len(pts) - 1):
        r1, b1 = pts[i]
        r2, b2 = pts[i + 1]
        if (b1 - target) * (b2 - target) <= 0:
            log_b1, log_b2 = math.log(b1), math.log(b2)
            log_t = math.log(target)
            frac = (log_t - log_b1) / (log_b2 - log_b1) if log_b2 != log_b1 else 0
            return r1 + frac * (r2 - r1)
    return None


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    colors = {'64': '#2ca02c', '128': '#1f77b4', '256': '#d62728', '512': '#9467bd'}
    markers_nn = {'64': '^', '128': 'o', '256': 's', '512': 'D'}
    markers_sc = markers_nn

    # Show only the N values that have full sweeps and are interesting for real-world scale
    show_Ns = ['128', '256', '512']

    # Plot each (N, decoder) curve
    crossovers = {}
    for N_str in show_Ns:
        if N_str not in data:
            continue
        points = data[N_str]
        if not points:
            continue
        rates = [p['sum_rate'] for p in points]
        nn_blers = [p['nn_bler'] for p in points]
        sc_blers = [p['sc_bler'] for p in points]
        col = colors.get(N_str, '#333333')

        # NN curve (solid)
        nn_pts = [(r, b) for r, b in zip(rates, nn_blers) if b is not None and b > 0]
        if nn_pts:
            ax.semilogy([r for r, _ in nn_pts], [b for _, b in nn_pts],
                        color=col, marker=markers_nn[N_str], linestyle='-',
                        lw=2, ms=8, label=f"NN-SC, N={N_str}")
        # NN zero points (BLER=0 → plot at floor)
        nn_zero = [r for r, b in zip(rates, nn_blers) if b == 0]
        if nn_zero:
            ax.scatter(nn_zero, [1e-5] * len(nn_zero), color=col, marker=markers_nn[N_str],
                       s=80, facecolors='none', edgecolors=col, linewidths=2,
                       label=f"NN-SC N={N_str} (0 errors)")

        # SC curve (dashed)
        sc_pts = [(r, b) for r, b in zip(rates, sc_blers) if b is not None and b > 0]
        if sc_pts:
            ax.semilogy([r for r, _ in sc_pts], [b for _, b in sc_pts],
                        color=col, marker=markers_sc[N_str], linestyle='--',
                        lw=2, ms=8, mfc='white', label=f"SC, N={N_str}")

        # Crossover at BLER=1e-3
        nn_cross = crossover_rate(rates, nn_blers, 1e-3)
        sc_cross = crossover_rate(rates, sc_blers, 1e-3)
        crossovers[N_str] = (nn_cross, sc_cross)

    # Horizontal target lines
    for target, label, color in [(1e-2, "1e-2", '#888'), (1e-3, "1e-3", '#444')]:
        ax.axhline(target, color=color, linestyle=':', lw=1.2)
        ax.text(0.55, target * 1.3, f"BLER = {label}", ha='left',
                va='bottom', fontsize=8, color=color)

    # Capacity vertical line
    ax.axvline(BEMAC_CAPACITY, color='black', linestyle=':', lw=1)
    ax.text(BEMAC_CAPACITY, 5e-1, " BEMAC sum capacity = 1.5",
            rotation=90, va='top', ha='left', fontsize=9, color='black')

    # Annotate crossover rates at TWO targets
    crossovers_1e2 = {}
    for N_str in show_Ns:
        if N_str not in data: continue
        points = data[N_str]
        rates = [p['sum_rate'] for p in points]
        nn_blers = [p['nn_bler'] for p in points]
        sc_blers = [p['sc_bler'] for p in points]
        crossovers_1e2[N_str] = (
            crossover_rate(rates, nn_blers, 1e-2),
            crossover_rate(rates, sc_blers, 1e-2),
        )

    txt_lines = []
    txt_lines.append("Sum rate at fixed BLER target:")
    for N_str in show_Ns:
        if N_str not in crossovers_1e2: continue
        nn_1e2, sc_1e2 = crossovers_1e2[N_str]
        nn_1e3, sc_1e3 = crossovers[N_str]
        line1 = f"  N={N_str}, BLER=1e-2: NN={nn_1e2:.2f}, SC={sc_1e2:.2f}" if nn_1e2 and sc_1e2 else f"  N={N_str}, BLER=1e-2: NN={nn_1e2}, SC={sc_1e2}"
        if nn_1e2 and sc_1e2:
            line1 += f"  (+{(nn_1e2-sc_1e2)/sc_1e2*100:.0f}%)"
        line2 = f"  N={N_str}, BLER=1e-3: NN={nn_1e3:.2f}, SC=below sweep"
        txt_lines.append(line1)
        txt_lines.append(line2)

    if txt_lines:
        ax.text(0.55, 0.06, "\n".join(["BLER=1e-3 crossover:"] + txt_lines),
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#999"),
                verticalalignment="bottom")

    ax.set_xlabel("Sum rate $R_u + R_v$ (bits/channel use)", fontsize=11)
    ax.set_ylabel("Block error rate (BLER)", fontsize=11)
    ax.set_title("BEMAC Class C — Rate vs BLER\nNeural SC vs analytical SC, fixed N", fontsize=12)
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlim(0.5, 1.55)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"Wrote {OUT_PNG}")

    # Also dump crossover info
    print("\nCrossovers at BLER = 1e-3:")
    for N_str, (nn, sc) in sorted(crossovers.items()):
        if nn and sc:
            print(f"  N={N_str}: NN reaches at R={nn:.3f}, SC reaches at R={sc:.3f}, NN advantage = +{(nn-sc)/sc*100:.0f}%")
        else:
            print(f"  N={N_str}: NN={nn}, SC={sc}")


if __name__ == "__main__":
    main()
