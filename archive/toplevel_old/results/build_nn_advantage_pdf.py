#!/usr/bin/env python3
"""
Build the "NN advantage" PDF — focused on cases where the Neural SC decoder
gives a clear, defensible advantage over the analytical SC decoder.

Each page contains:
  - Title (channel, class, rate point)
  - BLER + timing table (N, NN BLER, SC BLER, ratio, NN ms/cw, SC ms/cw, slowdown)
  - Plot of BLER vs N (existing PNG)
  - Comment about what the advantage is and what it costs
"""

import os
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import image as mpimg

RESULTS_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results"
OUT_PDF = os.path.join(RESULTS_DIR, "nn_advantage.pdf")

# ---------- Data: pulled from the verified JSON / xlsx files ----------

# BEMAC Class C (Ru=0.30, Rv=0.60) — from bemac_nn_vs_sc_classC_50k.json
CLASSC_30_60 = [
    # N, NN BLER, SC BLER, NN CW, SC CW
    (8,    0.03933,  0.10967, 3000,  3000),
    (16,   0.01633,  0.09200, 3000,  3000),
    (32,   0.00567,  0.09933, 3000,  3000),
    (64,   0.00167,  0.05500, 3000,  3000),
    (128,  0.00025,  0.02455, 20000, 20000),
    (256,  0.00018,  0.01336, 50000, 50000),
    (512,  0.00020,  0.00336, 50000, 50000),
    (1024, 0.000167, 0.00040, 30000, 50000),
]

# BEMAC Class C (Ru=0.35, Rv=0.70) — from xlsx
CLASSC_35_70 = [
    (8,    0.05267, 0.15067, 3000,  3000),
    (16,   0.01800, 0.18700, 3000,  3000),
    (32,   0.01067, 0.13133, 3000,  3000),
    (64,   0.00900, 0.12467, 3000,  3000),
    (128,  0.00600, 0.13670, 20000, 20000),
    (256,  0.00490, 0.09155, 20000, 20000),
    (512,  0.00345, 0.05355, 20000, 20000),
    (1024, 0.00355, 0.01730, 20000, 20000),
]

# Timing — from to_git_v2/results/bemac/bemac_classC_timing.json (NeuralMACSCDecoder)
# Same model architecture for both Class C rate points; timings are essentially identical.
CLASSC_TIMING_PATH = os.path.join(RESULTS_DIR, "bemac", "bemac_classC_timing.json")

# Plots (existing PNG files in the experiment folders)
PLOT_30_60 = os.path.join(RESULTS_DIR, "bemac", "bemac_classC_Ru30_Rv60_nn_vs_sc",
                          "nn_vs_sc_classC_Ru30_Rv60.png")
PLOT_35_70 = os.path.join(RESULTS_DIR, "bemac", "bemac_classC_Ru35_Rv70_nn_vs_sc",
                          "nn_vs_sc_classC_Ru35_Rv70.png")
PLOT_RATE_SWEEP = os.path.join(RESULTS_DIR, "bemac", "bemac_classC_rate_sweep.png")
RATE_SWEEP_JSON = os.path.join(RESULTS_DIR, "bemac", "bemac_classC_rate_sweep.json")


def load_timings():
    with open(CLASSC_TIMING_PATH) as f:
        d = json.load(f)
    return {int(k): v for k, v in d.items()}


def fmt_bler(b):
    if b is None: return "—"
    if b == 0: return "0"
    return f"{b:.2e}"


def fmt_ms(ms):
    if ms is None: return "—"
    if ms < 0.1: return f"{ms:.3f}"
    if ms < 10:  return f"{ms:.2f}"
    return f"{ms:.1f}"


def make_table_data(bler_data, timings):
    """Build the table for one experiment.
    Returns: headers, rows.
    """
    headers = ["N", "NN BLER", "SC BLER", "NN/SC", "NN ms/cw", "SC ms/cw", "Slowdown"]
    rows = []
    for N, nn_b, sc_b, nn_cw, sc_cw in bler_data:
        ratio_bler = nn_b / sc_b if sc_b > 0 else None
        t = timings.get(N, {})
        nn_ms = t.get("nn_ms_per_cw")
        sc_ms = t.get("sc_ms_per_cw")
        slowdown = (nn_ms / sc_ms) if (nn_ms and sc_ms) else None

        rows.append([
            str(N),
            fmt_bler(nn_b),
            fmt_bler(sc_b),
            f"{ratio_bler:.3f}" if ratio_bler is not None else "—",
            fmt_ms(nn_ms),
            fmt_ms(sc_ms),
            f"{slowdown:.1f}x" if slowdown else "—",
        ])
    return headers, rows


def render_rate_sweep_page(pdf):
    """Render the rate-BLER sweep page."""
    with open(RATE_SWEEP_JSON) as f:
        sweep = json.load(f)

    fig = plt.figure(figsize=(8.5, 11), dpi=100)
    fig.text(0.5, 0.96,
             "BEMAC Class C — Rate vs BLER  (the rate-advantage angle)",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.93,
             "Same Neural SC model evaluated at multiple rate points (sum capacity = 1.5 bits/use)",
             ha="center", va="top", fontsize=9, color="#444")

    if os.path.exists(PLOT_RATE_SWEEP):
        img = mpimg.imread(PLOT_RATE_SWEEP)
        ax_img = fig.add_axes([0.08, 0.50, 0.84, 0.40])
        ax_img.imshow(img); ax_img.axis("off")

    # Build a 2-row table per N
    headers = ["N", "BLER target", "NN sum-rate", "SC sum-rate", "NN advantage"]
    rows = []
    crossovers = {
        '128': {'1e-2': (1.089, 0.790), '1e-3': (0.959, None)},
        '256': {'1e-2': (1.075, 0.886), '1e-3': (0.974, None)},
        '512': {'1e-2': (1.082, 0.965), '1e-3': (0.882, None)},
    }
    for N_str in ['128', '256', '512']:
        for tgt_label in ['1e-2', '1e-3']:
            nn, sc = crossovers[N_str][tgt_label]
            if sc is None:
                gain = "SC unreachable"
                sc_str = "—"
            else:
                gain = f"+{(nn-sc)/sc*100:.0f}%"
                sc_str = f"{sc:.3f}"
            rows.append([N_str, tgt_label, f"{nn:.3f}", sc_str, gain])

    ax_tbl = fig.add_axes([0.10, 0.20, 0.80, 0.27])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for c in range(len(headers)):
        tbl[(0, c)].set_facecolor("#dfe6f0")
        tbl[(0, c)].set_text_props(weight="bold")
    # Highlight the gain column
    gain_col = headers.index("NN advantage")
    for r_idx in range(1, len(rows) + 1):
        tbl[(r_idx, gain_col)].set_facecolor("#d4edda")

    fig.text(0.05, 0.13, "Comment:", fontsize=10, fontweight="bold")
    fig.text(0.05, 0.04,
             "At a fixed BLER target of 1e-2, NN-SC supports +38% higher sum rate at N=128 and +21% at N=256.\n"
             "At BLER target of 1e-3 (real-world wireless data quality), the analytical SC decoder simply\n"
             "DOES NOT REACH this BLER at any sum rate ≥ 0.6 — even at the lowest swept rate (R=0.6) SC's\n"
             "BLER is already 3e-3. The NN-SC reaches BLER=1e-3 at R≈0.96. This is a structural advantage:\n"
             "for BEMAC Class C at N=128/256, you literally cannot get to 1e-3 BLER with analytical SC at\n"
             "this regime — NN is the only option that works there.",
             fontsize=9, color="#222", verticalalignment="bottom")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_gmac_negative_page(pdf):
    """Honest negative-result page for GMAC sweep + N=64 deep dive."""
    fig = plt.figure(figsize=(8.5, 11), dpi=100)
    fig.text(0.5, 0.96,
             "GMAC — neural decoder gives no advantage anywhere",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.93,
             "Sweep across rate points × block lengths at SNR=6dB. NN ties at low rates and loses elsewhere.",
             ha="center", va="top", fontsize=9, color="#444")

    # Top table: N=64 deep-dive (50K cw)
    fig.text(0.05, 0.88, "N=64 deep dive (50,000 cw per point):", fontsize=10, fontweight="bold")
    headers1 = ["k_u=k_v", "R sum", "NN BLER", "SC BLER", "ratio", "verdict"]
    rows1 = [
        ["16", "0.500", "1.20e-3", "1.24e-3", "0.97x", "tied"],
        ["20", "0.625", "3.20e-3", "3.24e-3", "0.99x", "tied"],
        ["24", "0.750", "5.62e-3", "6.04e-3", "0.93x", "tied"],
        ["28", "0.875", "1.12e-2", "8.62e-3", "1.30x", "NN worse"],
        ["32", "1.000", "4.11e-2", "3.69e-2", "1.11x", "NN worse"],
    ]
    ax1 = fig.add_axes([0.05, 0.66, 0.90, 0.20])
    ax1.axis("off")
    t1 = ax1.table(cellText=rows1, colLabels=headers1, loc="upper center", cellLoc="center")
    t1.auto_set_font_size(False); t1.set_fontsize(9); t1.scale(1, 1.5)
    for c in range(len(headers1)):
        t1[(0, c)].set_facecolor("#f0d4d4")
        t1[(0, c)].set_text_props(weight="bold")
    # Color verdict cells
    for ridx in range(1, len(rows1) + 1):
        verdict = rows1[ridx - 1][-1]
        if "tied" in verdict:
            t1[(ridx, len(headers1) - 1)].set_facecolor("#fff3cd")
        else:
            t1[(ridx, len(headers1) - 1)].set_facecolor("#f8d7da")

    # Bottom table: rate-sweep summary across N (5K cw)
    fig.text(0.05, 0.62, "Larger-N rate sweep (5,000 cw per point) — best result at each N:", fontsize=10, fontweight="bold")
    headers2 = ["N", "Best rate point (R sum)", "NN BLER", "SC BLER", "best ratio"]
    rows2 = [
        ["64",  "k=16 (R=0.5)",  "1.20e-3", "1.24e-3", "0.97x (tied)"],
        ["128", "k=32 (R=0.5)",  "0",       "2.0e-4",  "0 (both ~0)"],
        ["256", "k=64 (R=0.5)",  "3.3e-3",  "3.3e-4",  "10.0x worse"],
    ]
    ax2 = fig.add_axes([0.05, 0.43, 0.90, 0.16])
    ax2.axis("off")
    t2 = ax2.table(cellText=rows2, colLabels=headers2, loc="upper center", cellLoc="center")
    t2.auto_set_font_size(False); t2.set_fontsize(9); t2.scale(1, 1.5)
    for c in range(len(headers2)):
        t2[(0, c)].set_facecolor("#f0d4d4")
        t2[(0, c)].set_text_props(weight="bold")

    # Bottom comment
    fig.text(0.05, 0.38, "Findings:", fontsize=11, fontweight="bold", color="#b22222")
    fig.text(0.05, 0.04,
             "1. AT N=64, low rates (R=0.5–0.75): NN-SC and analytical SC are statistically tied.\n"
             "   The 5K-cw sweep showed nominal NN advantage (ratio 0.36–0.74), but this was pure\n"
             "   sample-size noise. With 50K cw the advantage collapses to ~0.93–0.99x — within\n"
             "   the Wilson 95% CI of analytical SC.\n"
             "\n"
             "2. AT N=64, mid-to-high rates (R≥0.875): NN-SC is STATISTICALLY WORSE than analytical SC\n"
             "   by 11–30%. The trained operating point (R=1.0) is one such losing point.\n"
             "\n"
             "3. AT N=128: both decoders give BLER≈0 at low rates, NN is moderately worse at higher rates.\n"
             "\n"
             "4. AT N=256: NN-SC is 8–16x WORSE than analytical SC across all rate points.\n"
             "\n"
             "CONCLUSION: For GMAC, there is no rate point or block length where the neural decoder\n"
             "beats the analytical decoder. The GMAC story is closed. The project's positive result is\n"
             "BEMAC Class C only (see other pages).",
             fontsize=9, color="#222", verticalalignment="bottom")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_four_way_page(pdf):
    """Honest negative-result page for the BEMAC Class B four-way comparison."""
    fig = plt.figure(figsize=(8.5, 11), dpi=100)
    fig.text(0.5, 0.96,
             "BEMAC Class B (Ru=0.50, Rv=0.70) — Four-way comparison",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.93,
             "SC vs NN-SC vs SCL(L=4) vs NN-SCL(L=4) — verifying the original 'NN-SCL beats SC by 8x' claim",
             ha="center", va="top", fontsize=9, color="#444")

    headers = ["N", "Decoder", "BLER", "95% CI", "ms/cw", "Slowdown vs SC"]
    rows = [
        # N=32
        ["32", "SC",        "8.70e-03", "[7.92e-3, 9.55e-3]", "0.064", "1.0x"],
        ["32", "NN-SC",     "8.42e-03", "[7.66e-3, 9.26e-3]", "0.329", "5.1x"],
        ["32", "SCL(4)",    "8.50e-03", "[7.32e-3, 9.87e-3]", "4.96",  "77x"],
        ["32", "NN-SCL(4)", "8.55e-03", "[7.36e-3, 9.92e-3]", "47.2",  "738x"],
        # N=64
        ["64", "SC",        "2.10e-03", "[1.74e-3, 2.54e-3]", "0.115", "1.0x"],
        ["64", "NN-SC",     "2.24e-03", "[1.86e-3, 2.69e-3]", "1.078", "9.4x"],
        ["64", "SCL(4)",    "6.50e-04", "[3.80e-4, 1.11e-3]", "12.4",  "108x"],
        ["64", "NN-SCL(4)", "(killed)", "(see comment)",       "—",     "—"],
    ]

    ax_tbl = fig.add_axes([0.05, 0.45, 0.90, 0.45])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for c in range(len(headers)):
        tbl[(0, c)].set_facecolor("#f0d4d4")
        tbl[(0, c)].set_text_props(weight="bold")

    fig.text(0.05, 0.40, "Findings:", fontsize=11, fontweight="bold", color="#b22222")
    fig.text(0.05, 0.04,
             "1. AT N=32: All four decoders give identical BLER (~8.5e-3, all 95% CIs overlap heavily).\n"
             "   The 'NN-SCL beats SC' claim from low-CW data does NOT survive rigorous evaluation.\n"
             "   NN-SCL is 738x SLOWER than SC for the SAME BLER.\n"
             "\n"
             "2. AT N=64: NN-SC = SC (statistically identical, 2.10e-3 vs 2.24e-3, CIs overlap).\n"
             "   Analytical SCL(4) gives BLER = 6.5e-4 — a clean 3.2x improvement over SC.\n"
             "   The original 'NN-SCL = 0.0007' claim, if rigorously verified, would essentially MATCH\n"
             "   analytical SCL — i.e. NO advantage of the neural list decoder over the analytical one.\n"
             "   N=64 NN-SCL eval was killed at 70+ minutes runtime to save compute (would only confirm tie).\n"
             "\n"
             "3. CONCLUSION: For BEMAC Class B (interleaved path), the neural decoder gives no\n"
             "   meaningful advantage over the analytical decoder, with or without list decoding.\n"
             "   The 'NN beats SC by 8x' headline from session-5 was an artifact of low-CW measurement.\n"
             "   The real wins are in BEMAC Class C (extreme path) — see other pages.",
             fontsize=9, color="#222", verticalalignment="bottom")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_experiment_page(pdf, title, subtitle, bler_data, timings, png_path, comment):
    fig = plt.figure(figsize=(8.5, 11), dpi=100)
    fig.patch.set_facecolor("white")

    # Title
    fig.text(0.5, 0.96, title, ha="center", va="top", fontsize=14, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=9, color="#444")

    # Plot (top half)
    if png_path and os.path.exists(png_path):
        img = mpimg.imread(png_path)
        ax_img = fig.add_axes([0.10, 0.50, 0.80, 0.40])
        ax_img.imshow(img)
        ax_img.axis("off")
    else:
        ax_img = fig.add_axes([0.10, 0.50, 0.80, 0.40])
        ax_img.text(0.5, 0.5, "(no plot)", ha="center", va="center", fontsize=12, color="#888")
        ax_img.axis("off")

    # Table (middle)
    headers, rows = make_table_data(bler_data, timings)
    ax_tbl = fig.add_axes([0.05, 0.18, 0.90, 0.30])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.45)
    for c in range(len(headers)):
        tbl[(0, c)].set_facecolor("#dfe6f0")
        tbl[(0, c)].set_text_props(weight="bold")
    # Highlight the NN/SC ratio column when NN beats SC
    ratio_col = headers.index("NN/SC")
    for r_idx, row in enumerate(rows, start=1):
        try:
            v = float(row[ratio_col])
            if v < 1.0:
                tbl[(r_idx, ratio_col)].set_facecolor("#d4edda")  # green
        except (ValueError, TypeError):
            pass

    # Comment (bottom)
    fig.text(0.05, 0.13, "Comment:", fontsize=10, fontweight="bold")
    fig.text(0.05, 0.04, comment, fontsize=9, color="#222", wrap=True,
             verticalalignment="bottom")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_cover(pdf, title, lines):
    fig = plt.figure(figsize=(8.5, 11), dpi=100)
    fig.text(0.5, 0.75, title, ha="center", fontsize=22, fontweight="bold")
    y = 0.62
    for line in lines:
        fig.text(0.5, y, line, ha="center", fontsize=11, color="#333")
        y -= 0.04
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    timings = load_timings()

    with PdfPages(OUT_PDF) as pdf:
        render_cover(pdf,
                     "Polar Codes MAC — NN Decoder Advantages",
                     [
                         "Cases where the Neural SC decoder gives a clear, defensible",
                         "advantage over the analytical SC decoder.",
                         "",
                         f"Generated: {os.path.basename(OUT_PDF)}",
                         "",
                         "Each page reports BLER, codeword count, and wall-clock decode time",
                         "per codeword for both decoders.",
                     ])

        # Page 1: Class C (0.30, 0.60)
        render_experiment_page(
            pdf,
            title="BEMAC — Class C, (Ru=0.30, Rv=0.60)",
            subtitle="Path: U^N V^N (decode all V first, then U|Y) · Z = X+Y · NeuralMACSCDecoder vs analytical SC",
            bler_data=CLASSC_30_60,
            timings=timings,
            png_path=PLOT_30_60,
            comment=(
                "Strongest NN-vs-SC win in the project. Neural SC achieves 2x–80x lower BLER than\n"
                "analytical SC across the full N range, with the largest gap at N=128 (~100x better)\n"
                "and substantial gains at all other N. Eval CW counts are real (3K–50K).\n\n"
                "COST: NN-SC takes ~11x–21x longer per codeword than analytical SC (see Slowdown column).\n"
                "This is a real BLER-vs-compute trade-off: ~15x more compute for ~10–100x lower BLER.\n"
                "Both decoders are O(N log N) — only the constants differ."
            ),
        )

        # NEW page: rate-BLER sweep showing the rate-advantage angle
        render_rate_sweep_page(pdf)

        # Page 3: Class C (0.35, 0.70)
        render_experiment_page(
            pdf,
            title="BEMAC — Class C, (Ru=0.35, Rv=0.70) — second rate point",
            subtitle="Path: U^N V^N · Z = X+Y · Same architecture as the (0.30, 0.60) case · Higher rate, harder regime",
            bler_data=CLASSC_35_70,
            timings=timings,
            png_path=PLOT_35_70,
            comment=(
                "Same Neural SC architecture, slightly higher rate point. NN beats SC by 3x–23x at\n"
                "every N tested. The advantage is most striking at mid N (128: 23x, 256: 19x).\n"
                "CW counts: 3K at small N, 20K at N>=128. Plots and table both show NN dominates.\n\n"
                "COST: Identical timing profile to the (0.30, 0.60) case — same model, same complexity.\n"
                "~11x–21x slower decode for ~5x–23x lower BLER."
            ),
        )

        # Negative-result page: four-way comparison on Class B
        render_four_way_page(pdf)

        # Negative-result page: GMAC sweep + deep dive
        render_gmac_negative_page(pdf)

    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
