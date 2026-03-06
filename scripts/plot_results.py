"""
plot_results.py
===============
Plot BLER simulation results from JSON files produced by sim_bemac_class_c.py.

Usage:
    python scripts/plot_results.py results/file1.json [results/file2.json ...]

Outputs:
    results/bler_bars.pdf      – grouped bar chart: BLER vs rate pair
    results/bler_vs_rate.pdf   – line plot: BLER vs sum-rate
"""

import json
import sys
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── IEEE-style serif font settings ──────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":  "stix",
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.4,
    "lines.markersize":  5,
})

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_files(paths):
    """Load JSON result files. Returns list of (label, records)."""
    datasets = []
    for path in paths:
        with open(path) as f:
            records = json.load(f)
        label = os.path.splitext(os.path.basename(path))[0]
        datasets.append((label, records))
    return datasets


def rate_label(ru, rv):
    return f"({ru:.2f}, {rv:.2f})"


def bler_or_upper(record):
    """Return BLER value. If zero, return an upper-bound 1/n_cw (or None if n_cw=0)."""
    bler = record.get("bler", 0.0)
    n_cw = record.get("n_codewords", 0)
    if bler > 0:
        return bler, False
    if n_cw > 0:
        return 1.0 / n_cw, True   # upper bound
    return None, True


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(datasets):
    header = f"{'File':<40} {'Ru':>5} {'Rv':>5} {'L':>4} {'BLER':>12} {'n_cw':>7} {'blk_err':>8} {'time_s':>9}"
    print(header)
    print("-" * len(header))
    for label, records in datasets:
        for r in records:
            bler, is_ub = bler_or_upper(r)
            bler_str = f"<{bler:.2e}" if is_ub else f" {bler:.2e}"
            print(
                f"{label:<40} {r['Ru']:>5.2f} {r['Rv']:>5.2f} {r['L']:>4d} "
                f"{bler_str:>12} {r['n_codewords']:>7d} {r['block_errors']:>8d} "
                f"{r['time_s']:>9.1f}"
            )
    print()


# ── Plot 1: BLER vs rate pair (grouped bar chart) ────────────────────────────

def plot_bler_bars(datasets, out_path):
    all_rate_pairs = sorted(
        {(r["Ru"], r["Rv"]) for _, recs in datasets for r in recs},
        key=lambda x: (x[0] + x[1], x[0])
    )
    all_L = sorted({r["L"] for _, recs in datasets for r in recs})

    index = {}
    for label, records in datasets:
        for r in records:
            key = (label, r["Ru"], r["Rv"], r["L"])
            index[key] = bler_or_upper(r)

    n_pairs = len(all_rate_pairs)
    n_groups = len(datasets) * len(all_L)
    bar_width = 0.8 / max(n_groups, 1)
    x = np.arange(n_pairs)

    fig, ax = plt.subplots(figsize=(max(6, n_pairs * 1.4), 4.5))

    color_idx = 0
    bar_idx = 0
    legend_handles = []

    for di, (label, _) in enumerate(datasets):
        for li, L_val in enumerate(all_L):
            offsets = (bar_idx - (n_groups - 1) / 2) * bar_width
            blers = []
            uppers = []
            for rp in all_rate_pairs:
                val, is_ub = index.get((label, rp[0], rp[1], L_val), (None, True))
                blers.append(val)
                uppers.append(is_ub)

            color = COLORS[color_idx % len(COLORS)]
            L_str = f"L={L_val}"
            file_str = label if len(datasets) > 1 else ""
            bar_label = f"{file_str} {L_str}".strip()

            for xi, (bv, is_ub) in enumerate(zip(blers, uppers)):
                if bv is None:
                    continue
                hatch = "//" if is_ub else None
                bar = ax.bar(
                    x[xi] + offsets, bv,
                    width=bar_width * 0.9,
                    color=color, alpha=0.85,
                    hatch=hatch,
                    edgecolor="black", linewidth=0.5,
                    label="_nolegend_"
                )

            import matplotlib.patches as mpatches
            patch = mpatches.Patch(color=color, label=bar_label, alpha=0.85)
            legend_handles.append(patch)

            bar_idx += 1
            color_idx += 1

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([rate_label(rp[0], rp[1]) for rp in all_rate_pairs], rotation=30, ha="right")
    ax.set_xlabel(r"Rate pair $(R_u,\, R_v)$")
    ax.set_ylabel("BLER")
    ax.set_title("BLER vs. Rate Pair (BE-MAC, $N=1024$)")
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)

    ax.text(
        0.99, 0.01, "Hatched bars: $<1/n_{\\mathrm{cw}}$ (zero errors observed)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, style="italic", color="gray"
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 2: BLER vs sum-rate ──────────────────────────────────────────────────

def plot_bler_vs_rate(datasets, out_path):
    all_L = sorted({r["L"] for _, recs in datasets for r in recs})

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    color_idx = 0
    for di, (label, records) in enumerate(datasets):
        for li, L_val in enumerate(all_L):
            subset = [r for r in records if r["L"] == L_val]
            if not subset:
                continue

            subset.sort(key=lambda r: r["Ru"] + r["Rv"])
            sum_rates = [r["Ru"] + r["Rv"] for r in subset]
            blers = []
            uppers = []
            for r in subset:
                bv, is_ub = bler_or_upper(r)
                blers.append(bv)
                uppers.append(is_ub)

            color = COLORS[color_idx % len(COLORS)]
            marker = MARKERS[color_idx % len(MARKERS)]
            L_str = f"L={L_val}"
            line_label = f"{label}, {L_str}" if len(datasets) > 1 else L_str

            x_conf = [s for s, bv, ub in zip(sum_rates, blers, uppers) if bv is not None and not ub]
            y_conf = [bv for bv, ub in zip(blers, uppers) if bv is not None and not ub]
            x_ub   = [s for s, bv, ub in zip(sum_rates, blers, uppers) if bv is not None and ub]
            y_ub   = [bv for bv, ub in zip(blers, uppers) if bv is not None and ub]

            if x_conf or x_ub:
                x_all = [s for s, bv in zip(sum_rates, blers) if bv is not None]
                y_all = [bv for bv in blers if bv is not None]
                ax.semilogy(x_all, y_all, color=color, linestyle="-", marker=None, alpha=0.6)

            if x_conf:
                ax.semilogy(x_conf, y_conf, color=color, marker=marker,
                            linestyle="None", label=line_label, markeredgecolor="k",
                            markeredgewidth=0.4)
            if x_ub:
                ax.semilogy(x_ub, y_ub, color=color, marker=marker,
                            linestyle="None", markerfacecolor="white",
                            markeredgecolor=color, markeredgewidth=1.2,
                            label=f"{line_label} (UB)" if x_conf else line_label)

            color_idx += 1

    ax.set_xlabel(r"Sum rate $R_u + R_v$")
    ax.set_ylabel("BLER")
    ax.set_title(r"BLER vs. Sum Rate (BE-MAC, $N=1024$)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    ax.text(
        0.99, 0.99, "Hollow markers: $<1/n_{\\mathrm{cw}}$ (zero block errors)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7, style="italic", color="gray"
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_results.py results/file1.json [results/file2.json ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    for p in paths:
        if not os.path.isfile(p):
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    datasets = load_files(paths)

    print_summary(datasets)
    plot_bler_bars(datasets,    "results/bler_bars.pdf")
    plot_bler_vs_rate(datasets, "results/bler_vs_rate.pdf")


if __name__ == "__main__":
    main()
