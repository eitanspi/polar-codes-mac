#!/usr/bin/env python3
"""
generate_tutorial_pdf.py
========================
Generates a multi-page PDF tutorial explaining MAC SC decoding from first
principles, using the Gaussian MAC channel with concrete numerical examples.

Output: docs/paper_figures/MAC_SC_TUTORIAL.pdf
"""

import torch
torch.set_num_threads(2)

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import matplotlib.patheffects as pe

# ── Project root ──────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "docs", "paper_figures", "MAC_SC_TUTORIAL.pdf")

# ── Style constants ───────────────────────────────────────────────────────
COLOR_CONV = "#2060C0"      # blue  — circ_conv
COLOR_PROD = "#208040"      # green — norm_prod
COLOR_DEC  = "#C02020"      # red   — decisions
COLOR_BG   = "#F8F6F0"
COLOR_BOX  = "#E8E4D8"
COLOR_NEURAL = "#8040C0"    # purple — neural

FONT_TITLE = {"fontsize": 14, "fontweight": "bold", "fontfamily": "serif"}
FONT_SEC   = {"fontsize": 12, "fontweight": "bold", "fontfamily": "serif"}
FONT_BODY  = {"fontsize": 10.5, "fontfamily": "serif"}
FONT_MATH  = {"fontsize": 10.5, "fontfamily": "serif", "fontstyle": "italic"}
FONT_SMALL = {"fontsize": 9.5, "fontfamily": "serif"}
FONT_NUM   = {"fontsize": 10, "fontfamily": "monospace"}


# ── Gaussian MAC helpers ─────────────────────────────────────────────────
def gmac_log_transition(z, sigma2):
    """Return 2x2 log-prob tensor: T[x,y] = log P(z | x, y) for GMAC."""
    T = np.zeros((2, 2))
    log_norm = -0.5 * np.log(2 * np.pi * sigma2)
    for x in (0, 1):
        for y in (0, 1):
            mu = (1 - 2*x) + (1 - 2*y)
            T[x, y] = log_norm - (z - mu)**2 / (2 * sigma2)
    return T


def normalize_tensor(T_log):
    """Normalize a 2x2 log-prob tensor so it sums to 1 in prob domain."""
    mx = T_log.max()
    probs = np.exp(T_log - mx)
    probs /= probs.sum()
    return probs


def circ_conv(A, B):
    """Circular convolution of two 2x2 probability tensors (linear domain)."""
    out = np.zeros((2, 2))
    for u in (0, 1):
        for v in (0, 1):
            for i in (0, 1):
                for j in (0, 1):
                    out[u, v] += A[u ^ i, v ^ j] * B[i, j]
    return out


def norm_prod(A, B):
    """Normalized elementwise product of two 2x2 probability tensors."""
    raw = A * B
    s = raw.sum()
    if s > 0:
        return raw / s
    return raw


def delta_tensor(u_hat, v_hat):
    """Delta distribution: 1 at (u_hat, v_hat), 0 elsewhere."""
    d = np.zeros((2, 2))
    d[u_hat, v_hat] = 1.0
    return d


def fmt2x2(T, decimals=4):
    """Format a 2x2 tensor as a multi-line string."""
    return (f"  [x=0,y=0]={T[0,0]:.{decimals}f}   [x=0,y=1]={T[0,1]:.{decimals}f}\n"
            f"  [x=1,y=0]={T[1,0]:.{decimals}f}   [x=1,y=1]={T[1,1]:.{decimals}f}")


def fmt2x2_uv(T, decimals=4):
    """Format a 2x2 tensor with u,v labels."""
    return (f"  [u=0,v=0]={T[0,0]:.{decimals}f}   [u=0,v=1]={T[0,1]:.{decimals}f}\n"
            f"  [u=1,v=0]={T[1,0]:.{decimals}f}   [u=1,v=1]={T[1,1]:.{decimals}f}")


# ── Concrete numerical example ───────────────────────────────────────────
SNR_DB = 6.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)

# Pick channel outputs that give a clear example
Z1_VAL = 1.3
Z2_VAL = -0.8

# True inputs for reference
TRUE_U = (0, 1)  # u1=0, u2=1
TRUE_V = (1, 0)  # v1=1, v2=0
# Encoded: x1 = u1 ^ u2 = 1, x2 = u2 = 1, y1 = v1 ^ v2 = 1, y2 = v2 = 0
# Noiseless: z1 = (1-2*1)+(1-2*1) = -2, z2 = (1-2*1)+(1-2*0) = 0

# Compute the transition tensors
Tz1_log = gmac_log_transition(Z1_VAL, SIGMA2)
Tz2_log = gmac_log_transition(Z2_VAL, SIGMA2)
Tz1 = normalize_tensor(Tz1_log)
Tz2 = normalize_tensor(Tz2_log)


# ── Helper to add a styled text page ─────────────────────────────────────
def new_page(pdf, title, page_num, total_pages=9):
    """Create a new figure with title bar and return (fig, ax)."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor(COLOR_BG)

    # Title bar
    ax.add_patch(FancyBboxPatch((0.3, 10.15), 7.9, 0.65,
                                boxstyle="round,pad=0.1",
                                facecolor="#2B4570", edgecolor="none"))
    ax.text(4.25, 10.47, title, ha="center", va="center",
            color="white", **FONT_TITLE)

    # Page number
    ax.text(8.0, 0.25, f"Page {page_num}/{total_pages}",
            ha="right", va="bottom", fontsize=8, fontfamily="serif",
            color="#888888")

    # Footer
    ax.text(0.5, 0.25, "MAC SC Decoding Tutorial",
            ha="left", va="bottom", fontsize=8, fontfamily="serif",
            color="#888888", fontstyle="italic")

    return fig, ax


def add_box(ax, x, y, w, h, text, color=COLOR_BOX, text_kw=None):
    """Add a rounded box with text."""
    kw = dict(FONT_BODY)
    if text_kw:
        kw.update(text_kw)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.08",
                                facecolor=color, edgecolor="#AAAAAA",
                                linewidth=0.5))
    ax.text(x + 0.15, y + h - 0.15, text, va="top", **kw)


def add_tensor_box(ax, x, y, T, label, color="#EEE8D5", uv=False):
    """Draw a compact 2x2 tensor display."""
    ax.add_patch(FancyBboxPatch((x, y), 3.4, 1.1,
                                boxstyle="round,pad=0.06",
                                facecolor=color, edgecolor="#999999",
                                linewidth=0.5))
    ax.text(x + 0.1, y + 0.95, label, va="top", **FONT_SEC)
    if uv:
        txt = fmt2x2_uv(T, decimals=4)
    else:
        txt = fmt2x2(T, decimals=4)
    ax.text(x + 0.1, y + 0.6, txt, va="top", **FONT_NUM)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 1: Setup
# ═════════════════════════════════════════════════════════════════════════
def page_setup(pdf):
    fig, ax = new_page(pdf, "Page 1: Problem Setup", 1)
    Y = 9.8

    # Section: Two users
    ax.text(0.5, Y, "Two Users, One Channel", **FONT_SEC)
    Y -= 0.12
    lines = [
        "User U sends bits (u1, u2). User V sends bits (v1, v2).",
        "Both encode with the N=2 polar transform before transmission.",
        "",
        "Polar encoding (N=2):",
        "  User U:  x1 = u1 XOR u2,   x2 = u2",
        "  User V:  y1 = v1 XOR v2,   y2 = v2",
        "",
        "Channel: memoryless MAC.  At each time i, both transmit simultaneously:",
        "  Zi depends on (xi, yi) only.  The receiver observes z1, z2.",
        "",
        "Goal: recover all four bits u1, u2, v1, v2 from z1, z2."
    ]
    for line in lines:
        Y -= 0.28
        ax.text(0.7, Y, line, **FONT_BODY)

    # Section: Transition tensor
    Y -= 0.4
    ax.text(0.5, Y, "The Transition Tensor", **FONT_SEC)
    Y -= 0.12
    lines2 = [
        "For each channel observation zi, define the 2x2 likelihood tensor:",
        "  Tz_i[x, y]  =  P(zi | xi=x, yi=y)",
        "",
        "This is a 2x2 matrix (x in {0,1}, y in {0,1}).",
        "It captures EVERYTHING the observation zi tells us about the inputs."
    ]
    for line in lines2:
        Y -= 0.28
        ax.text(0.7, Y, line, **FONT_BODY)

    # Section: Concrete example
    Y -= 0.4
    ax.text(0.5, Y, "Concrete Example: Gaussian MAC at SNR = 6 dB", **FONT_SEC)
    Y -= 0.12
    lines3 = [
        f"Channel: Z = (1-2X) + (1-2Y) + W,   W ~ N(0, sigma2={SIGMA2:.4f})",
        f"Received: z1 = {Z1_VAL},  z2 = {Z2_VAL}",
        "",
        "Noiseless outputs for each (x,y) pair:",
        "  (0,0) -> +2    (0,1) -> 0    (1,0) -> 0    (1,1) -> -2",
        "",
        "Compute Tz_i[x,y] = N(zi; mu(x,y), sigma2), then normalize:"
    ]
    for line in lines3:
        Y -= 0.28
        ax.text(0.7, Y, line, **FONT_BODY)

    # Show the two tensors
    Y -= 0.35
    add_tensor_box(ax, 0.7, Y - 1.1, Tz1, f"Tz1 (z1={Z1_VAL}):", color="#D6EAF8")
    add_tensor_box(ax, 4.5, Y - 1.1, Tz2, f"Tz2 (z2={Z2_VAL}):", color="#D5F5E3")

    Y -= 1.4
    ax.text(0.7, Y, f"z1={Z1_VAL} is close to 0, so Tz1 favors (0,1) and (1,0).",
            **FONT_SMALL)
    Y -= 0.25
    ax.text(0.7, Y, f"z2={Z2_VAL} is close to -2, so Tz2 favors (1,1).",
            **FONT_SMALL)

    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 2: The Two Operations
# ═════════════════════════════════════════════════════════════════════════
def page_operations(pdf):
    fig, ax = new_page(pdf, "Page 2: The Two Fundamental Operations", 2)
    Y = 9.8

    # Circ conv
    ax.text(0.5, Y, "Operation 1: Circular Convolution (circ_conv)",
            color=COLOR_CONV, **FONT_SEC)
    Y -= 0.12
    lines = [
        "Definition:   R[u, v] = SUM over (i,j) of A[u XOR i, v XOR j] * B[i, j]",
        "",
        "Meaning: If X1 = A XOR B where A ~ dist_A, B ~ dist_B independently,",
        "         then dist(X1) = circ_conv(dist_A, dist_B).",
        "",
        "In polar codes: the XOR in x1 = u1 XOR u2 creates this convolution.",
        "circ_conv MARGINALIZES over the unknown bits by summing out",
        "all possible values weighted by their probabilities.",
    ]
    for line in lines:
        Y -= 0.27
        ax.text(0.7, Y, line, **FONT_BODY)

    # Example
    Y -= 0.3
    ax.text(0.7, Y, "Example:  circ_conv(Tz1, Tz2):", color=COLOR_CONV, **FONT_SEC)
    Y -= 0.12
    CC = circ_conv(Tz1, Tz2)
    CC_norm = CC / CC.sum()
    lines_ex = [
        f"R[0,0] = Tz1[0,0]*Tz2[0,0] + Tz1[0,1]*Tz2[0,1] + Tz1[1,0]*Tz2[1,0] + Tz1[1,1]*Tz2[1,1]",
        f"       = {Tz1[0,0]:.4f}*{Tz2[0,0]:.4f} + {Tz1[0,1]:.4f}*{Tz2[0,1]:.4f} + {Tz1[1,0]:.4f}*{Tz2[1,0]:.4f} + {Tz1[1,1]:.4f}*{Tz2[1,1]:.4f}",
        f"       = {CC[0,0]:.6f}",
    ]
    for line in lines_ex:
        Y -= 0.25
        ax.text(0.7, Y, line, **FONT_SMALL)

    Y -= 0.25
    add_tensor_box(ax, 0.7, Y - 1.1, CC_norm, "circ_conv(Tz1, Tz2) normalized:",
                   color="#D6EAF8", uv=True)

    # Norm prod
    Y -= 1.4
    ax.text(0.5, Y, "Operation 2: Normalized Product (norm_prod)",
            color=COLOR_PROD, **FONT_SEC)
    Y -= 0.12
    lines2 = [
        "Definition:   R[u, v] = A[u,v] * B[u,v]  /  SUM(A * B)",
        "",
        "Meaning: If we have two INDEPENDENT observations about the SAME",
        "variable, their combined evidence is the product (Bayes' rule).",
        "",
        "In polar codes: when z1 and z2 both give evidence about (u2, v2),",
        "norm_prod combines them into a single posterior.",
    ]
    for line in lines2:
        Y -= 0.27
        ax.text(0.7, Y, line, **FONT_BODY)

    # Example
    Y -= 0.3
    NP = norm_prod(Tz1, Tz2)
    ax.text(0.7, Y, "Example:  norm_prod(Tz1, Tz2):", color=COLOR_PROD, **FONT_SEC)
    Y -= 0.25
    add_tensor_box(ax, 0.7, Y - 1.1, NP, "norm_prod(Tz1, Tz2):",
                   color="#D5F5E3")

    Y -= 1.3
    ax.text(0.7, Y,
            "Key insight: circ_conv sums out unknowns; norm_prod fuses evidence.",
            fontweight="bold", **FONT_BODY)

    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 3: Step 1 — Decode u1, v1
# ═════════════════════════════════════════════════════════════════════════
def page_step1(pdf):
    fig, ax = new_page(pdf, "Page 3: Step 1 -- Decode u1, v1", 3)
    Y = 9.8

    ax.text(0.5, Y, "Question: What is P(z1, z2 | u1, v1)?", **FONT_SEC)
    Y -= 0.12
    lines = [
        "We want to decode the FIRST pair of information bits (u1, v1).",
        "Since u2, v2 are unknown, we must marginalize them out:",
        "",
        "P(z1, z2 | u1, v1)",
        "  = SUM_{u2,v2} P(u2, v2) * P(z1 | x1, y1) * P(z2 | x2, y2)",
        "",
        "where x1 = u1 XOR u2,  x2 = u2,  y1 = v1 XOR v2,  y2 = v2.",
        "Since u2, v2 are uniform:  P(u2, v2) = 1/4.  Substituting:",
        "",
        "P(z1, z2 | u1, v1)",
        "  = (1/4) * SUM_{u2,v2} Tz1[u1 XOR u2, v1 XOR v2] * Tz2[u2, v2]",
        "  = (1/4) * circ_conv(Tz1, Tz2)[u1, v1]",
        "",
        "This is EXACTLY the circular convolution!  The XOR encoding structure",
        "turns the marginalization sum into a convolution.",
    ]
    for line in lines:
        Y -= 0.27
        ax.text(0.7, Y, line, **FONT_BODY)

    # Show computation
    Y -= 0.35
    ax.text(0.5, Y, "Numerical computation:", color=COLOR_CONV, **FONT_SEC)
    Y -= 0.12

    CC = circ_conv(Tz1, Tz2)
    CC_norm = CC / CC.sum()
    lines2 = [
        "circ_conv(Tz1, Tz2) (unnormalized, then normalized):",
        "",
        fmt2x2_uv(CC_norm, decimals=4),
    ]
    for line in lines2:
        Y -= 0.28
        ax.text(0.7, Y, line, **FONT_NUM)

    # Decision
    Y -= 0.4
    u1_hat, v1_hat = np.unravel_index(np.argmax(CC_norm), (2, 2))
    ax.text(0.5, Y, "Decision:", color=COLOR_DEC, **FONT_SEC)
    Y -= 0.12
    lines3 = [
        f"argmax  =>  u1_hat = {u1_hat},  v1_hat = {v1_hat}   (Prob: {CC_norm[u1_hat, v1_hat]:.4f})",
    ]
    for line in lines3:
        Y -= 0.27
        ax.text(0.7, Y, line, color=COLOR_DEC, fontweight="bold", **FONT_BODY)

    # Visual
    Y -= 0.45
    ax.text(0.5, Y, "Intuition:", **FONT_SEC)
    Y -= 0.12
    lines4 = [
        "The convolution asks: \"For each possible (u1, v1), how well do ALL",
        "possible encodings (u1 XOR u2, v1 XOR v2) explain the observations?\"",
        "",
        "It weights each (u2, v2) by what z2 says about (u2, v2), then checks",
        "if z1 is consistent with the implied (x1, y1) = (u1^u2, v1^v2).",
        "",
        "This is CalcLeft at the root of the SC decoding tree.",
    ]
    for line in lines4:
        Y -= 0.26
        ax.text(0.7, Y, line, **FONT_BODY)

    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 4: Step 2 — Decode u2, v2
# ═════════════════════════════════════════════════════════════════════════
def page_step2(pdf):
    fig, ax = new_page(pdf, "Page 4: Step 2 -- Decode u2, v2", 4)
    Y = 9.8

    CC_norm = circ_conv(Tz1, Tz2)
    CC_norm = CC_norm / CC_norm.sum()
    u1_hat, v1_hat = np.unravel_index(np.argmax(CC_norm), (2, 2))

    ax.text(0.5, Y, "Question: What is P(z1, z2 | u2, v2, u1_hat, v1_hat)?",
            **FONT_SEC)
    Y -= 0.12
    lines = [
        f"We decided u1_hat={u1_hat}, v1_hat={v1_hat}. Now decode (u2, v2).",
        "Given the decision, the encoded bits are determined:",
        f"  x1 = u1_hat XOR u2 = {u1_hat} XOR u2,   y1 = v1_hat XOR v2 = {v1_hat} XOR v2",
        "  x2 = u2,   y2 = v2",
        "",
        "So: P(z1, z2 | u2, v2, decision)",
        f"  = Tz1[{u1_hat} XOR u2, {v1_hat} XOR v2] * Tz2[u2, v2]",
    ]
    for line in lines:
        Y -= 0.27
        ax.text(0.7, Y, line, **FONT_BODY)

    # Sub-step A
    Y -= 0.4
    ax.text(0.5, Y, "Sub-step A: What z1 says about (u2,v2)",
            color=COLOR_CONV, **FONT_SEC)
    Y -= 0.12
    delta = delta_tensor(u1_hat, v1_hat)
    step_a = circ_conv(delta, Tz1)
    lines_a = [
        f"circ_conv(delta_(u1={u1_hat},v1={v1_hat}), Tz1)",
        "The delta selects one shifted version of Tz1:",
        f"  Result[u2, v2] = Tz1[{u1_hat} XOR u2, {v1_hat} XOR v2]",
    ]
    for line in lines_a:
        Y -= 0.26
        ax.text(0.7, Y, line, **FONT_BODY)

    Y -= 0.25
    add_tensor_box(ax, 0.7, Y - 1.1, step_a,
                   "Step A result (z1 evidence about u2,v2):",
                   color="#D6EAF8", uv=True)

    # Sub-step B
    Y -= 1.45
    ax.text(0.5, Y, "Sub-step B: Combine z1 and z2 evidence",
            color=COLOR_PROD, **FONT_SEC)
    Y -= 0.12
    step_b = norm_prod(step_a, Tz2)
    lines_b = [
        "norm_prod(Step_A_result, Tz2)",
        "z1 (via Step A) and z2 give INDEPENDENT evidence about (u2, v2).",
        "norm_prod combines them via Bayes' rule (multiply and normalize).",
    ]
    for line in lines_b:
        Y -= 0.26
        ax.text(0.7, Y, line, **FONT_BODY)

    Y -= 0.25
    add_tensor_box(ax, 0.7, Y - 1.1, step_b,
                   "Step B result (combined posterior for u2,v2):",
                   color="#D5F5E3", uv=True)

    # Decision
    Y -= 1.4
    u2_hat, v2_hat = np.unravel_index(np.argmax(step_b), (2, 2))
    ax.text(0.5, Y, "Decision:", color=COLOR_DEC, **FONT_SEC)
    Y -= 0.12
    lines_d = [
        f"argmax  =>  u2_hat = {u2_hat},  v2_hat = {v2_hat}   (Prob: {step_b[u2_hat, v2_hat]:.4f})",
        "",
        "This is CalcRight at the root: first circ_conv with delta (pass decision",
        "through the XOR), then norm_prod to combine with the other observation.",
    ]
    for line in lines_d:
        Y -= 0.27
        ax.text(0.7, Y, line, **FONT_BODY)

    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 5: The Tree Connection
# ═════════════════════════════════════════════════════════════════════════
def page_tree(pdf):
    fig, ax = new_page(pdf, "Page 5: The Tree Connection", 5)
    Y = 9.8

    ax.text(0.5, Y, "From N=2 to General N: Recursive Structure", **FONT_SEC)
    Y -= 0.12
    lines = [
        "For N=2, we had exactly two steps:",
        "  Step 1 (left child):  CalcLeft  = circ_conv(Tz1, Tz2)",
        "  Step 2 (right child): CalcRight = norm_prod(circ_conv(delta, Tz1), Tz2)",
        "For general N, the SAME two operations apply recursively at every",
        "node in a binary tree.  The tree has depth n = log2(N).",
    ]
    for line in lines:
        Y -= 0.27
        ax.text(0.7, Y, line, **FONT_BODY)

    # Draw N=4 tree
    Y -= 0.4
    ax.text(0.5, Y, "Example: N=4 Decoding Tree", **FONT_SEC)

    # Tree coordinates
    tree_cx = 4.25
    tree_top = Y - 0.5
    tree_h = 3.0
    level_y = [tree_top, tree_top - tree_h*0.4, tree_top - tree_h*0.8]

    # Root
    root_x, root_y = tree_cx, level_y[0]
    # Level 1
    l1_left_x, l1_right_x = tree_cx - 2.0, tree_cx + 2.0
    l1_y = level_y[1]
    # Level 2 (leaves)
    leaf_xs = [tree_cx - 3.0, tree_cx - 1.0, tree_cx + 1.0, tree_cx + 3.0]
    leaf_y = level_y[2]

    # Draw edges
    for (x1, y1, x2, y2, label, col) in [
        (root_x, root_y, l1_left_x, l1_y, "CalcLeft", COLOR_CONV),
        (root_x, root_y, l1_right_x, l1_y, "CalcRight", COLOR_PROD),
        (l1_left_x, l1_y, leaf_xs[0], leaf_y, "CalcLeft", COLOR_CONV),
        (l1_left_x, l1_y, leaf_xs[1], leaf_y, "CalcRight", COLOR_PROD),
        (l1_right_x, l1_y, leaf_xs[2], leaf_y, "CalcLeft", COLOR_CONV),
        (l1_right_x, l1_y, leaf_xs[3], leaf_y, "CalcRight", COLOR_PROD),
    ]:
        ax.annotate("", xy=(x2, y2 + 0.2), xytext=(x1, y1 - 0.15),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        mid_x = (x1 + x2) / 2 + (0.15 if x2 > x1 else -0.15)
        mid_y = (y1 + y2) / 2 + 0.1
        ax.text(mid_x, mid_y, label, fontsize=7, color=col,
                ha="center", fontfamily="serif", fontweight="bold",
                rotation=0)

    # Draw nodes
    for (x, y, label, desc) in [
        (root_x, root_y, "Root", "[Tz1,Tz2,Tz3,Tz4]"),
        (l1_left_x, l1_y, "V=1", "(u1,v1), (u3,v3)"),
        (l1_right_x, l1_y, "V=2", "(u2,v2), (u4,v4)"),
    ]:
        ax.add_patch(FancyBboxPatch((x - 0.6, y - 0.15), 1.2, 0.35,
                                    boxstyle="round,pad=0.05",
                                    facecolor="white", edgecolor="#333"))
        ax.text(x, y + 0.02, label, ha="center", va="center",
                fontsize=9, fontfamily="serif", fontweight="bold")
        ax.text(x, y - 0.28, desc, ha="center", va="top",
                fontsize=7, fontfamily="serif", color="#666")

    for i, (x, label) in enumerate(zip(leaf_xs,
                                        ["(u1,v1)", "(u3,v3)", "(u2,v2)", "(u4,v4)"])):
        ax.add_patch(FancyBboxPatch((x - 0.5, leaf_y - 0.1), 1.0, 0.3,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#FFF3CD", edgecolor="#333"))
        ax.text(x, leaf_y + 0.02, f"Leaf {i+1}", ha="center", va="center",
                fontsize=8, fontfamily="serif")
        ax.text(x, leaf_y - 0.2, label, ha="center", va="top",
                fontsize=7, fontfamily="serif", color="#666")

    # Decode order
    Y = leaf_y - 0.5
    ax.text(0.7, Y, "Decoding order: Leaf 1, Leaf 2, Leaf 3, Leaf 4",
            fontweight="bold", **FONT_BODY)
    Y -= 0.22
    ax.text(0.7, Y, "(Due to bit-reversal, the order maps to SC positions)",
            **FONT_SMALL)

    # Explanation
    Y -= 0.35
    ax.text(0.5, Y, "At Every Node:", **FONT_SEC)
    Y -= 0.12
    lines2 = [
        "  - Going LEFT:  CalcLeft marginalizes (circ_conv) over unknowns.",
        "  - Going RIGHT: CalcRight combines (norm_prod) the right half of the",
        "    parent with the circ_conv of the left child's decision.",
        "  - Going UP: CalcParent reconstructs the parent tensor from children.",
        "",
        "The tree has n = log2(N) levels.  Each leaf visit costs O(1) operations",
        "at each level => total decoding cost is O(N log N).",
    ]
    for line in lines2:
        Y -= 0.25
        ax.text(0.7, Y, line, **FONT_BODY)

    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 6: CalcLeft and CalcRight Formal Definitions
# ═════════════════════════════════════════════════════════════════════════
def page_formal(pdf):
    fig, ax = new_page(pdf, "Page 6: CalcLeft, CalcRight, CalcParent", 6)
    Y = 9.8

    ax.text(0.5, Y, "Formal Definitions (Ren et al., 2025)", **FONT_SEC)
    Y -= 0.15
    lines = [
        "Each edge in the tree carries an array of 2x2 tensors.",
        "At vertex beta, the parent edge has 2L tensors, children have L each.",
        "We write parent = [P_1, ..., P_L, P_{L+1}, ..., P_{2L}]",
        "         first half = [P_1, ..., P_L],  second half = [P_{L+1}, ..., P_{2L}]"
    ]
    for line in lines:
        Y -= 0.28
        ax.text(0.7, Y, line, **FONT_BODY)

    # CalcLeft
    Y -= 0.45
    box_h = 1.35
    ax.add_patch(FancyBboxPatch((0.4, Y - box_h), 7.6, box_h,
                                boxstyle="round,pad=0.1",
                                facecolor="#D6EAF8", edgecolor=COLOR_CONV,
                                linewidth=1.5))
    ax.text(0.6, Y - 0.05, "CalcLeft(parent, right_child):",
            color=COLOR_CONV, **FONT_SEC)
    calc_left_lines = [
        "  For each position t = 1..L:",
        "    temp[t] = norm_prod( parent[L+t],  right_child[t] )",
        "    left_child[t] = circ_conv( parent[t],  temp[t] )",
        "  Intuition: fold in sibling info (norm_prod), then marginalize (circ_conv).",
    ]
    for i, line in enumerate(calc_left_lines):
        ax.text(0.6, Y - 0.32 - i*0.22, line, **FONT_BODY)

    # CalcRight
    Y -= box_h + 0.25
    ax.add_patch(FancyBboxPatch((0.4, Y - box_h), 7.6, box_h,
                                boxstyle="round,pad=0.1",
                                facecolor="#D5F5E3", edgecolor=COLOR_PROD,
                                linewidth=1.5))
    ax.text(0.6, Y - 0.05, "CalcRight(parent, left_child):",
            color=COLOR_PROD, **FONT_SEC)
    calc_right_lines = [
        "  For each position t = 1..L:",
        "    temp[t] = circ_conv( left_child[t],  parent[t] )",
        "    right_child[t] = norm_prod( parent[L+t],  temp[t] )",
        "  Intuition: pass decision through XOR (circ_conv), combine (norm_prod).",
    ]
    for i, line in enumerate(calc_right_lines):
        ax.text(0.6, Y - 0.32 - i*0.22, line, **FONT_BODY)

    # CalcParent
    Y -= box_h + 0.25
    box_h2 = 1.15
    ax.add_patch(FancyBboxPatch((0.4, Y - box_h2), 7.6, box_h2,
                                boxstyle="round,pad=0.1",
                                facecolor="#FCF3CF", edgecolor="#B7950B",
                                linewidth=1.5))
    ax.text(0.6, Y - 0.05, "CalcParent(left_child, right_child):",
            color="#7D6608", **FONT_SEC)
    calc_parent_lines = [
        "  For each position t = 1..L:",
        "    parent[t]   = circ_conv( left_child[t],  right_child[t] )",
        "    parent[L+t] = right_child[t]   (just copy)",
        "  Used when navigating UP the tree (between distant leaves).",
    ]
    for i, line in enumerate(calc_parent_lines):
        ax.text(0.6, Y - 0.32 - i*0.22, line, **FONT_BODY)

    # N=2 connection
    Y -= box_h2 + 0.25
    ax.text(0.5, Y, "Connection to N=2:", **FONT_SEC)
    Y -= 0.12
    lines2 = [
        "At N=2: parent = [Tz1, Tz2], L=1, right_child = [uniform (undecided)].",
        "  CalcLeft: temp = norm_prod(Tz2, uniform) = Tz2.",
        "            left = circ_conv(Tz1, Tz2).  (Exactly Step 1!)",
        "  CalcRight: after deciding, left_child = [delta_{u1,v1}].",
        "            temp = circ_conv(delta, Tz1) = shifted Tz1.",
        "            right = norm_prod(Tz2, temp).  (Exactly Step 2!)",
    ]
    for line in lines2:
        Y -= 0.25
        ax.text(0.7, Y, line, **FONT_BODY)

    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 7: Why CalcLeft has norm_prod too
# ═════════════════════════════════════════════════════════════════════════
def page_why_normprod(pdf):
    fig, ax = new_page(pdf, "Page 7: Why CalcLeft Needs norm_prod", 7)
    Y = 9.8

    ax.text(0.5, Y, "The Subtlety at N > 2", **FONT_SEC)
    Y -= 0.12
    lines = [
        "At N=2, CalcLeft was just circ_conv(Tz1, Tz2). But for N=4 and larger,",
        "CalcLeft is circ_conv(first_half, norm_prod(second_half, right_child)).",
        "Why the extra norm_prod?  Let us trace through N=4.",
    ]
    for line in lines:
        Y -= 0.26
        ax.text(0.7, Y, line, **FONT_BODY)

    # N=4 example
    Y -= 0.35
    ax.text(0.5, Y, "N=4 Example: Decoding the First Leaf", **FONT_SEC)
    Y -= 0.12
    lines2 = [
        "Root edge carries [Tz1, Tz2, Tz3, Tz4] (4 tensors, after bit-reversal).",
        "Vertex 1 (root): first_half = [Tz1, Tz2], second_half = [Tz3, Tz4]",
        "The right child (edge 3) initially carries [uniform, uniform].",
        "",
        "CalcLeft at root:",
        "  temp[1] = norm_prod(Tz3, uniform) = Tz3",
        "  temp[2] = norm_prod(Tz4, uniform) = Tz4",
        "  left[1] = circ_conv(Tz1, Tz3),  left[2] = circ_conv(Tz2, Tz4)",
        "",
        "Left edge (edge 2) carries [circ_conv(Tz1,Tz3), circ_conv(Tz2,Tz4)].",
        "We descend again: CalcLeft at vertex 2 gives us the first leaf.",
    ]
    for line in lines2:
        Y -= 0.26
        ax.text(0.7, Y, line, **FONT_BODY)

    # Why norm_prod matters
    Y -= 0.3
    ax.text(0.5, Y, "When norm_prod Actually Matters", **FONT_SEC)
    Y -= 0.12
    lines3 = [
        "After decoding leaf 1 and leaf 2 (at vertex 2), we go BACK UP",
        "(CalcParent) and then down to vertex 3 to decode leaves 3 and 4.",
        "",
        "When we CalcLeft at the root AGAIN, the right child is NO LONGER",
        "uniform -- it has been updated by previous decisions.",
        "",
        "In the computational graph approach, the sibling edge may already",
        "contain information from partial decoding.  The norm_prod ensures",
        "this information is properly incorporated.",
        "",
        "Concretely: if the right child says 'v2 is probably 1', then",
        "norm_prod(Tz4, right_child) gives a BETTER estimate for position 4",
        "than Tz4 alone, giving a more accurate posterior for the left child.",
    ]
    for line in lines3:
        Y -= 0.24
        ax.text(0.7, Y, line, **FONT_BODY)

    Y -= 0.3
    ax.text(0.5, Y, "Summary", **FONT_SEC)
    Y -= 0.1
    lines4 = [
        "  norm_prod in CalcLeft = \"use what we already know about the sibling\"",
        "  circ_conv in CalcLeft = \"marginalize over the encoding XOR\"",
        "  circ_conv in CalcRight = \"pass the decision through the XOR\"",
        "  norm_prod in CalcRight = \"combine with direct evidence\"",
    ]
    for line in lines4:
        Y -= 0.24
        ax.text(0.7, Y, line, fontweight="bold", **FONT_BODY)

    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 8: The Neural Replacement
# ═════════════════════════════════════════════════════════════════════════
def page_neural(pdf):
    fig, ax = new_page(pdf, "Page 8: The Neural Replacement (NCG Decoder)", 8)
    Y = 9.8

    ax.text(0.5, Y, "From Analytical to Neural", **FONT_SEC)
    Y -= 0.12
    lines = [
        "The analytical SC decoder requires EXACT knowledge of P(z|x,y).",
        "What if the channel is too complex for closed-form Tz?",
        "  - Memory channels: P(zi | xi, yi, z_{i-1}, ...)",
        "  - Non-Gaussian noise / unknown channel parameters",
        "Solution: replace each analytical operation with a neural network.",
    ]
    for line in lines:
        Y -= 0.27
        ax.text(0.7, Y, line, **FONT_BODY)

    # Analytical vs Neural table
    Y -= 0.4
    ax.text(0.5, Y, "Analytical vs Neural: Side by Side", **FONT_SEC)

    # Table
    Y -= 0.25
    table_data = [
        ("Component", "Analytical", "Neural (NCG)"),
        ("Leaf input", "Tz = 2x2 tensor from\nP(z|x,y)", "e = d-dim embedding from\nz_encoder(z)  (MLP)"),
        ("CalcLeft", "circ_conv(parent,\n  norm_prod(parent, sib))",
         "MLP(concat(parent, sib))\n  -> d-dim output"),
        ("CalcRight", "norm_prod(parent,\n  circ_conv(left, parent))",
         "MLP(concat(parent,\n  left, decision)) -> d-dim"),
        ("CalcParent", "circ_conv(left, right)\n(exact inverse)",
         "MLP(left, right) with\ngated residual connection"),
        ("Decision", "argmax of 2x2 tensor",
         "Linear(d -> 4) then\nargmax over 4 classes"),
    ]

    col_x = [0.5, 2.5, 5.5]
    col_w = [1.8, 2.8, 2.8]
    row_h = 0.6
    header_y = Y

    for row_i, row in enumerate(table_data):
        ry = header_y - row_i * row_h
        bg = "#E8E4D8" if row_i == 0 else ("white" if row_i % 2 == 1 else "#F5F5F0")
        for col_i, (cell, cx, cw) in enumerate(zip(row, col_x, col_w)):
            ax.add_patch(FancyBboxPatch((cx, ry - row_h + 0.05), cw, row_h - 0.05,
                                        boxstyle="square,pad=0",
                                        facecolor=bg,
                                        edgecolor="#CCCCCC",
                                        linewidth=0.5))
            font = FONT_SEC if row_i == 0 else FONT_SMALL
            color = "black"
            if row_i > 0 and col_i == 1:
                color = "#333333"
            if row_i > 0 and col_i == 2:
                color = COLOR_NEURAL
            ax.text(cx + 0.1, ry - 0.1, cell, va="top",
                    color=color, **font)

    # Key insight
    Y = header_y - len(table_data) * row_h - 0.25
    ax.text(0.5, Y, "Key Insight", **FONT_SEC)
    Y -= 0.12
    lines2 = [
        "The neural decoder preserves the TREE STRUCTURE of SC decoding.",
        "Only the operations at each node are replaced by learned MLPs.",
        "Training: given (channel outputs, true bits), minimize cross-entropy",
        "loss at each leaf decision.  The MLPs learn to approximate the",
        "analytical operations from data alone.",
        "Advantage: works for ANY channel, including channels where Tz cannot",
        "be computed.  The z_encoder learns the right representation of z.",
        "Cost: d-dimensional embeddings instead of 2x2 tensors.  Typically",
        "d = 16 or 32.  The O(N log N) tree structure is preserved.",
    ]
    for line in lines2:
        Y -= 0.26
        ax.text(0.7, Y, line, **FONT_BODY)

    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  PAGE 9: Summary
# ═════════════════════════════════════════════════════════════════════════
def page_summary(pdf):
    fig, ax = new_page(pdf, "Page 9: Summary", 9)
    Y = 9.8

    ax.text(0.5, Y, "The Complete Picture", **FONT_SEC)
    Y -= 0.15

    # Two primitives box
    Y -= 0.12
    box_h = 2.2
    ax.add_patch(FancyBboxPatch((0.4, Y - box_h), 7.6, box_h,
                                boxstyle="round,pad=0.1",
                                facecolor="white", edgecolor="#333",
                                linewidth=1.0))
    ax.text(0.6, Y - 0.05, "The Two Primitives", **FONT_SEC)
    prim_lines = [
        ("1. circ_conv(A, B)[u,v] = SUM_{i,j} A[u^i, v^j] * B[i,j]",
         COLOR_CONV),
        ("   \"Distribution of XOR of two independent variables\"",
         COLOR_CONV),
        ("   Used for: marginalizing unknowns, passing decisions, going up.",
         COLOR_CONV),
        ("", "black"),
        ("2. norm_prod(A, B)[u,v] = A[u,v] * B[u,v] / SUM(A*B)",
         COLOR_PROD),
        ("   \"Combine two independent pieces of evidence (Bayes' rule)\"",
         COLOR_PROD),
        ("   Used for: combining sibling info, fusing direct evidence.",
         COLOR_PROD),
    ]
    for i, (line, col) in enumerate(prim_lines):
        ax.text(0.7, Y - 0.35 - i*0.24, line, color=col, **FONT_BODY)

    # Tree recursion
    Y -= box_h + 0.3
    ax.text(0.5, Y, "The Recursive Tree", **FONT_SEC)
    Y -= 0.12
    tree_lines = [
        "For N codeword positions (n = log2(N) levels):",
        "  - The root edge carries N tensors (one per channel observation).",
        "  - Each tree level splits the tensor array in half.",
        "  - CalcLeft goes down-left (marginalize), CalcRight goes down-right",
        "    (condition on decision), CalcParent goes back up.",
        "  - At each leaf: make a hard decision (argmax) for one (u,v) pair.",
        "  Total: N leaves, each requiring O(n) operations => O(N log N) total.",
    ]
    for line in tree_lines:
        Y -= 0.26
        ax.text(0.7, Y, line, **FONT_BODY)

    # Neural extension
    Y -= 0.4
    ax.text(0.5, Y, "The Neural Extension (NCG)", color=COLOR_NEURAL, **FONT_SEC)
    Y -= 0.12
    neural_lines = [
        "Replace 2x2 tensors with d-dimensional learned embeddings.",
        "Replace circ_conv, norm_prod with learned MLPs.",
        "Preserve the tree structure and O(N log N) complexity.",
        "Works for any channel -- no need for analytical P(z|x,y).",
    ]
    for line in neural_lines:
        Y -= 0.26
        ax.text(0.7, Y, line, color=COLOR_NEURAL, **FONT_BODY)

    # Final note
    Y -= 0.4
    ax.text(0.5, Y, "Why This Matters", **FONT_SEC)
    Y -= 0.12
    final_lines = [
        "The polar code transform (XOR butterfly + bit reversal) creates a",
        "specific algebraic structure that decomposes multi-user decoding into",
        "a tree of simple 2x2 tensor operations.  The two operations --",
        "circular convolution and normalized product -- are not arbitrary choices.",
        "They arise uniquely from:",
        "  (a) the XOR structure of polar encoding => circ_conv",
        "  (b) the independence of channel observations => norm_prod",
        "Understanding this makes the SC decoder a natural consequence of",
        "the code structure and Bayesian inference, not a black box.",
    ]
    for line in final_lines:
        Y -= 0.26
        ax.text(0.7, Y, line, **FONT_BODY)

    pdf.savefig(fig)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print(f"Generating tutorial PDF...")
    print(f"  GMAC SNR = {SNR_DB} dB, sigma2 = {SIGMA2:.6f}")
    print(f"  z1 = {Z1_VAL}, z2 = {Z2_VAL}")
    print(f"  Tz1:\n{fmt2x2(Tz1)}")
    print(f"  Tz2:\n{fmt2x2(Tz2)}")

    CC = circ_conv(Tz1, Tz2)
    CC_norm = CC / CC.sum()
    u1h, v1h = np.unravel_index(np.argmax(CC_norm), (2, 2))
    print(f"\n  circ_conv(Tz1,Tz2) normalized:\n{fmt2x2_uv(CC_norm)}")
    print(f"  Decision: u1={u1h}, v1={v1h}")

    delta_d = delta_tensor(u1h, v1h)
    step_a = circ_conv(delta_d, Tz1)
    step_b = norm_prod(step_a, Tz2)
    u2h, v2h = np.unravel_index(np.argmax(step_b), (2, 2))
    print(f"\n  Step A (circ_conv(delta, Tz1)):\n{fmt2x2_uv(step_a)}")
    print(f"  Step B (norm_prod(A, Tz2)):\n{fmt2x2_uv(step_b)}")
    print(f"  Decision: u2={u2h}, v2={v2h}")

    with PdfPages(OUT_PATH) as pdf:
        page_setup(pdf)
        page_operations(pdf)
        page_step1(pdf)
        page_step2(pdf)
        page_tree(pdf)
        page_formal(pdf)
        page_why_normprod(pdf)
        page_neural(pdf)
        page_summary(pdf)

    print(f"\nPDF saved to: {OUT_PATH}")
    print(f"  9 pages generated.")


if __name__ == "__main__":
    main()
