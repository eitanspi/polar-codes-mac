"""
plot_results.py
===============
Publication-quality figures for polar-coded BE-MAC simulation results.

Usage:
    # Figure 1: Rate region with L=32 operating points
    python scripts/plot_results.py --rate-region -o figures/fig1_rate_region.pdf \\
        results/sim_bemac_A_mc_L32.json results/sim_bemac_B_mc_L32.json \\
        results/sim_bemac_C_L32.json

    # Figure 2: BLER vs sum-rate for Class B (L=1 vs L=32)
    python scripts/plot_results.py -o figures/fig2_bler_class_B.pdf \\
        results/sim_bemac_B_mc_L1.json results/sim_bemac_B_mc_L32.json

    # Generic BLER vs sum-rate
    python scripts/plot_results.py results/sim_bemac_C_L1.json
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── IEEE-style formatting ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":  "stix",
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.6,
    "lines.markersize":  6,
})

MARKERS_BY_N = {1024: "o", 4096: "s", 16384: "^", 65536: "D"}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f"]

# Code class directions (Ru_dir, Rv_dir) — for rate region plot
CODE_CLASS_DIRS = {
    "A": (0.75, 0.75),
    "B": (0.625, 0.875),
    "C": (0.5, 1.0),
}
CLASS_MARKERS = {"A": "o", "B": "s", "C": "^"}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_file(path):
    """Load a JSON result file. Returns the full dict with 'results' list."""
    with open(path) as f:
        data = json.load(f)
    # Attach source filename for labelling
    data["_filename"] = os.path.splitext(os.path.basename(path))[0]
    data["_path"] = path
    return data


def bler_or_upper(record):
    """Return (bler_value, is_upper_bound).  None if no data."""
    bler = record.get("bler")
    if bler is None:
        return None, True
    n_cw = record.get("n_codewords", 0)
    if bler > 0:
        return bler, False
    if n_cw > 0:
        return 1.0 / n_cw, True
    return None, True


def get_L(dataset):
    """Extract L from dataset metadata or first result."""
    if "L" in dataset:
        return dataset["L"]
    for r in dataset.get("results", []):
        if "L" in r:
            return r["L"]
    return None


def get_marker(N):
    return MARKERS_BY_N.get(N, "o")


# ── Plot 1/3: BLER vs sum-rate ───────────────────────────────────────────────

def plot_bler_vs_sumrate(datasets, out_path, title=None):
    """BLER vs sum-rate. Auto-detects L=1 vs L=32 comparison mode."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Collect all unique L values across datasets
    all_L = set()
    for ds in datasets:
        for r in ds["results"]:
            if r.get("L") is not None:
                all_L.add(r["L"])
    is_comparison = len(all_L) > 1  # Plot 3: L comparison mode

    # Build curves: one per (filename, N, L) combination
    color_idx = 0
    # Assign colors per (filename, L) combination
    file_L_colors = {}

    for ds in datasets:
        fname = ds["_filename"]
        results = [r for r in ds["results"] if r.get("bler") is not None
                   or r.get("skipped")]
        if not results:
            continue

        N_values = sorted(set(r["N"] for r in results))
        L_values = sorted(set(r.get("L", 1) for r in results))

        for L_val in L_values:
            # Assign color per (filename, L)
            key = (fname, L_val)
            if key not in file_L_colors:
                file_L_colors[key] = COLORS[color_idx % len(COLORS)]
                color_idx += 1
            color = file_L_colors[key]

            for N in N_values:
                subset = [r for r in results
                          if r["N"] == N and r.get("L", 1) == L_val
                          and r.get("bler") is not None]
                if not subset:
                    continue

                subset.sort(key=lambda r: r["Ru"] + r["Rv"])
                marker = get_marker(N)

                # Truncate: from highest sum-rate leftward, include
                # first zero-error point (as upper bound), then stop
                subset.sort(key=lambda r: r["Ru"] + r["Rv"])
                kept = []
                for r in reversed(subset):
                    bv, ub = bler_or_upper(r)
                    if bv is None:
                        continue
                    kept.append((r["Ru"] + r["Rv"], bv, ub))
                    if r["bler"] == 0:
                        break  # include this one, then stop
                kept.reverse()

                if not kept:
                    continue

                sum_rates = [k[0] for k in kept]
                blers = [k[1] for k in kept]
                is_ub = [k[2] for k in kept]

                # Line style: dashed for L=1 in comparison mode
                if is_comparison:
                    ls = "--" if L_val == 1 else "-"
                else:
                    ls = "-"

                # Label
                cls = ds.get("class", "")
                parts = []
                if cls:
                    parts.append(f"Class {cls}")
                parts.append(f"$N={N}$")
                if is_comparison or len(all_L) > 1:
                    parts.append(f"$L={L_val}$")
                label = ", ".join(parts)

                # Connecting line
                ax.semilogy(sum_rates, blers, color=color, linestyle=ls,
                            marker=None, alpha=0.5)

                # Confirmed points (bler > 0)
                x_c = [s for s, ub in zip(sum_rates, is_ub) if not ub]
                y_c = [b for b, ub in zip(blers, is_ub) if not ub]
                if x_c:
                    ax.semilogy(x_c, y_c, color=color, marker=marker,
                                linestyle="None", label=label,
                                markeredgecolor="k", markeredgewidth=0.4)

                # Upper-bound points (bler == 0 → shown as 1/n_cw)
                x_u = [s for s, ub in zip(sum_rates, is_ub) if ub]
                y_u = [b for b, ub in zip(blers, is_ub) if ub]
                if x_u:
                    ub_label = None if x_c else label
                    ax.semilogy(x_u, y_u, color=color, marker=marker,
                                linestyle="None", markerfacecolor="white",
                                markeredgecolor=color, markeredgewidth=1.2,
                                label=ub_label)

    # Capacity line — auto-detect from dataset metadata
    sum_caps = [ds.get("capacity", {}).get("I_ZXY", None) for ds in datasets]
    sum_cap = next((c for c in sum_caps if c is not None), 1.5)
    ax.axvline(x=sum_cap, color="black", linestyle=":", linewidth=1.0,
               label=f"$C_{{\\mathrm{{sum}}}}={sum_cap:.4g}$")

    ax.set_xlabel(r"Sum rate $R_u + R_v$")
    ax.set_ylabel("BLER")

    if title:
        ax.set_title(title)
    else:
        # Auto-generate title
        all_classes = set(ds.get("class", "") for ds in datasets)
        all_N_vals = set(r["N"] for ds in datasets for r in ds["results"]
                         if r.get("bler") is not None)
        cls_str = f"Class {next(iter(all_classes))}" if len(all_classes) == 1 else ""
        N_str = f"$N\\!=\\!{next(iter(all_N_vals))}$" if len(all_N_vals) == 1 else ""
        parts = ["BE-MAC"]
        if cls_str:
            parts.append(cls_str)
        if N_str:
            parts.append(N_str)
        if is_comparison:
            parts.append("SC vs SCL")
        else:
            L_str = f"$L={min(all_L)}$" if all_L else ""
            parts.append(L_str)
        ax.set_title(" — ".join(p for p in parts if p))

    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(left=1.1)
    ax.set_ylim(bottom=1e-4, top=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 2: Rate region ─────────────────────────────────────────────────────

def plot_rate_region(datasets, out_path, title=None):
    """Rate region: capacity boundary, dominant face, and L=32 operating points."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # ── Capacity region (pentagon) ──
    cap_ru = [0, 1, 1, 0.5, 0, 0]
    cap_rv = [0, 0, 0.5, 1, 1, 0]
    ax.fill(cap_ru, cap_rv, alpha=0.06, color="steelblue")
    # Non-dominant boundary edges (thin black)
    for seg in [([0, 1], [0, 0]), ([1, 1], [0, 0.5]),
                ([0, 0], [0, 1]), ([0, 0.5], [1, 1])]:
        ax.plot(seg[0], seg[1], "k-", linewidth=0.8)
    # Dominant face (red, thick)
    ax.plot([1, 0.5], [0.5, 1], "r-", linewidth=2.2,
            label=r"Dominant face ($R_u\!+\!R_v = 1.5$)", zorder=3)

    # ── Code class direction rays ──
    for cls_name, (ru_d, rv_d) in CODE_CLASS_DIRS.items():
        ax.plot([0, ru_d], [0, rv_d], "k:", linewidth=0.7, alpha=0.4)
        ax.annotate(f"Class {cls_name}", xy=(ru_d, rv_d),
                    fontsize=8, ha="center", va="bottom", color="dimgray",
                    xytext=(0, 6), textcoords="offset points")

    # ── Operating points: best (highest sum-rate) per class with BLER < 1e-3 ──
    BLER_THRESH = 1e-3
    best = {}
    for ds in datasets:
        cls = ds.get("class", "?")
        for r in ds["results"]:
            if r.get("bler") is None or r.get("skipped"):
                continue
            if r["bler"] < BLER_THRESH and r.get("n_codewords", 0) > 0:
                sr = r["Ru"] + r["Rv"]
                if cls not in best or sr > best[cls]["Ru"] + best[cls]["Rv"]:
                    best[cls] = r

    for cls in sorted(best):
        r = best[cls]
        marker = CLASS_MARKERS.get(cls, "o")
        ax.plot(r["Ru"], r["Rv"], marker=marker, color="#1f77b4",
                markeredgecolor="k", markeredgewidth=0.6,
                markersize=10, zorder=4,
                label=f"Class {cls}")
        ax.annotate(f"$({r['Ru']:.2f},\\, {r['Rv']:.2f})$",
                    xy=(r["Ru"], r["Rv"]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(6, 6), textcoords="offset points")

    ax.set_xlabel(r"$R_u$", fontsize=13)
    ax.set_ylabel(r"$R_v$", fontsize=13)
    ax.set_title(title or r"BE-MAC Rate Region — $N\!=\!1024$, SCL $L\!=\!32$",
                 fontsize=13)
    ax.set_xlim(-0.02, 1.12)
    ax.set_ylim(-0.02, 1.12)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    ax.text(0.02, 0.02,
            r"Highest achieved rate with BLER $< 10^{-3}$",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="dimgray", style="italic")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def auto_output_path(args, datasets):
    """Generate default output path from input filenames and mode."""
    if args.output:
        return args.output
    base = os.path.dirname(datasets[0]["_path"]) or "results"
    if args.rate_region:
        name = "rate_region.pdf"
    else:
        stems = [ds["_filename"] for ds in datasets]
        name = "bler_vs_sumrate_" + "_".join(stems)
        # Truncate if too long
        if len(name) > 120:
            name = name[:120]
        name += ".pdf"
    return os.path.join(base, name)


def main():
    parser = argparse.ArgumentParser(
        description="Plot polar-coded MAC simulation results (PDF output).")
    parser.add_argument("files", nargs="+", help="JSON result file(s)")
    parser.add_argument("--rate-region", action="store_true",
                        help="Generate rate region plot instead of BLER vs sum-rate")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PDF path (default: auto-generated)")
    parser.add_argument("--title", "-t", default=None,
                        help="Custom plot title")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open PDF after saving (default: just save)")
    args = parser.parse_args()

    # Validate inputs
    for p in args.files:
        if not os.path.isfile(p):
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Load datasets
    datasets = [load_file(p) for p in args.files]

    out_path = auto_output_path(args, datasets)

    if args.rate_region:
        plot_rate_region(datasets, out_path, title=args.title)
    else:
        plot_bler_vs_sumrate(datasets, out_path, title=args.title)


if __name__ == "__main__":
    main()
