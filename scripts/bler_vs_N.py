"""
bler_vs_N.py — BLER vs block length analysis for ABN-MAC, Class B, SC (L=1).

Fixed rate point: rho=0.5, direction (0.5, 0.7) → Ru≈0.25N, Rv≈0.35N
Sum rate ≈ 0.6 (50% of capacity 1.2).

Generates MC designs + runs simulations + plots + writes LaTeX PDF.

Usage:
    python scripts/bler_vs_N.py
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.encoder import polar_encode, polar_encode_batch, build_message, build_message_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.decoder import decode_batch, build_log_W_leaf

# ════════════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════════════

N_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024]
RHO = 0.5
RU_DIR, RV_DIR = 0.5, 0.7
PATH_I_FRAC = 0.5   # Class B
DESIGN_TRIALS = 5000
TARGET_CW = 50000    # codewords per N (more for small N since they're fast)
MAX_TIME_PER_N = 600  # 10 min max per N
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DESIGNS_DIR = os.path.join(SCRIPT_DIR, "..", "designs")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(DESIGNS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg=""):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════
#  MC Design (inline, single-process for small N)
# ════════════════════════════════════════════════════════════════════

def genie_design_inline(n, channel, path_i, n_trials, seed):
    """Quick MC genie design — single process, good for small N."""
    from polar.decoder import _CompGraph, _norm_prod_single, _NEG_INF, _LOG_HALF, _LOG_QUARTER

    N = 1 << n
    b = make_path(N, path_i)
    rng = np.random.default_rng(seed)

    u_err_counts = np.zeros(N, dtype=np.float64)
    v_err_counts = np.zeros(N, dtype=np.float64)

    for trial in range(n_trials):
        u_true = rng.integers(0, 2, size=N).tolist()
        v_true = rng.integers(0, 2, size=N).tolist()
        x = polar_encode(u_true)
        y = polar_encode(v_true)
        z = channel.sample_batch(np.array(x), np.array(y)).tolist()

        log_W = build_log_W_leaf(z, channel)
        graph = _CompGraph(n, log_W)
        u_hat, v_hat = {}, {}
        i_u, i_v = 0, 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u
            else:
                i_v += 1; i_t = i_v

            leaf_edge = i_t + N - 1
            target_vertex = leaf_edge >> 1
            graph.step_to(target_vertex)
            temp = graph.edge_data[leaf_edge][0].copy()

            if leaf_edge & 1 == 0:
                graph.calc_left(target_vertex)
            else:
                graph.calc_right(target_vertex)

            top_down = graph.edge_data[leaf_edge][0]
            combined = _norm_prod_single(top_down, temp)

            if gamma == 0:
                p0 = np.logaddexp(combined[0, 0], combined[0, 1])
                p1 = np.logaddexp(combined[1, 0], combined[1, 1])
                decoded = 1 if p1 > p0 else 0
                true_val = u_true[i_t - 1]
                if decoded != true_val:
                    u_err_counts[i_t - 1] += 1
                u_hat[i_t] = true_val
            else:
                p0 = np.logaddexp(combined[0, 0], combined[1, 0])
                p1 = np.logaddexp(combined[0, 1], combined[1, 1])
                decoded = 1 if p1 > p0 else 0
                true_val = v_true[i_t - 1]
                if decoded != true_val:
                    v_err_counts[i_t - 1] += 1
                v_hat[i_t] = true_val

            new_leaf = np.full((2, 2), _NEG_INF, dtype=np.float64)
            u_val = u_hat.get(i_t)
            v_val = v_hat.get(i_t)
            if u_val is not None and v_val is not None:
                new_leaf[u_val, v_val] = 0.0
            elif u_val is not None:
                new_leaf[u_val, 0] = _LOG_HALF
                new_leaf[u_val, 1] = _LOG_HALF
            elif v_val is not None:
                new_leaf[0, v_val] = _LOG_HALF
                new_leaf[1, v_val] = _LOG_HALF
            else:
                new_leaf[:, :] = _LOG_QUARTER
            graph.edge_data[leaf_edge][0] = new_leaf

    pe_u = u_err_counts / n_trials
    pe_v = v_err_counts / n_trials
    sorted_u = np.argsort(pe_u)
    sorted_v = np.argsort(pe_v)
    return pe_u, pe_v, sorted_u, sorted_v


def get_or_make_design(n, channel, path_i):
    """Load existing design or generate one."""
    N = 1 << n
    design_path = os.path.join(DESIGNS_DIR, f"abnmac_B_n{n}.npz")

    if os.path.exists(design_path):
        d = np.load(design_path)
        nt = int(d['n_trials']) if 'n_trials' in d else 0
        if nt >= 1000:
            log(f"  Loaded existing design: {nt} trials")
            if 'u_error_rates' in d:
                sorted_u = np.argsort(d['u_error_rates'])
                sorted_v = np.argsort(d['v_error_rates'])
            else:
                sorted_u = d['sorted_u']
                sorted_v = d['sorted_v']
            return sorted_u, sorted_v

    # Generate new design
    trials = min(DESIGN_TRIALS, max(2000, DESIGN_TRIALS))
    log(f"  Generating MC design ({trials} trials)...")
    t0 = time.time()
    pe_u, pe_v, sorted_u, sorted_v = genie_design_inline(
        n, channel, path_i, trials, SEED)
    elapsed = time.time() - t0
    log(f"  Design done in {elapsed:.1f}s")

    np.savez(design_path,
             u_error_rates=pe_u, v_error_rates=pe_v,
             path_i=np.array(path_i), n_trials=np.array(trials),
             seed=np.array(SEED))
    return sorted_u, sorted_v


def select_info_frozen(N, sorted_indices, k):
    A_0idx = sorted(sorted_indices[:k].tolist())
    A = [i + 1 for i in A_0idx]
    all_pos = set(range(1, N + 1))
    frozen = {pos: 0 for pos in sorted(all_pos - set(A))}
    return A, frozen


# ════════════════════════════════════════════════════════════════════
#  Simulation
# ════════════════════════════════════════════════════════════════════

def sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v, b, batch_size, rng, channel):
    info_u = rng.integers(0, 2, size=(batch_size, ku))
    info_v = rng.integers(0, 2, size=(batch_size, kv))
    U = build_message_batch(N, info_u, Au)
    V = build_message_batch(N, info_v, Av)
    X = polar_encode_batch(U)
    Y = polar_encode_batch(V)
    Z = channel.sample_batch(X, Y)

    results = decode_batch(N, Z.tolist(), b, frozen_u, frozen_v,
                           channel, vectorized=True)

    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])

    block_errors = 0
    u_bit_errors = 0
    v_bit_errors = 0
    for i, (u_dec, v_dec) in enumerate(results):
        ue = int(np.sum(np.array(u_dec)[u_info_idx] != info_u[i]))
        ve = int(np.sum(np.array(v_dec)[v_info_idx] != info_v[i]))
        u_bit_errors += ue
        v_bit_errors += ve
        if ue > 0 or ve > 0:
            block_errors += 1
    return block_errors, u_bit_errors, v_bit_errors


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════

def main():
    channel = ABNMAC()
    I_ZX, I_ZY_X, I_ZXY = channel.capacity()

    log(f"BLER vs N — ABN-MAC Class B, SC (L=1), rho={RHO}")
    log(f"  Direction: ({RU_DIR}, {RV_DIR}), sum rate = {RHO*(RU_DIR+RV_DIR):.3f}")
    log(f"  Capacity: I(Z;X,Y)={I_ZXY:.4f}")
    log(f"  N values: {N_VALUES}")
    log()

    all_results = []

    for N in N_VALUES:
        n = N.bit_length() - 1
        path_i = round(PATH_I_FRAC * N)
        b = make_path(N, path_i)

        ku = max(1, round(RHO * RU_DIR * N))
        kv = max(1, round(RHO * RV_DIR * N))
        ku = min(ku, N - 1)
        kv = min(kv, N - 1)
        Ru = ku / N
        Rv = kv / N

        log(f"N={N:>5d} (n={n:>2d}): ku={ku}, kv={kv}, Ru={Ru:.4f}, Rv={Rv:.4f}, "
            f"Ru+Rv={Ru+Rv:.4f}")

        # Design
        sorted_u, sorted_v = get_or_make_design(n, channel, path_i)
        Au, frozen_u = select_info_frozen(N, sorted_u, ku)
        Av, frozen_v = select_info_frozen(N, sorted_v, kv)

        # Benchmark
        rng = np.random.default_rng(SEED)
        bs_test = min(20, TARGET_CW)
        t0 = time.perf_counter()
        sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v, b, bs_test, rng, channel)
        ms_per_cw = (time.perf_counter() - t0) / bs_test * 1000

        # Compute how many codewords we can do
        n_cw = min(TARGET_CW, max(5000, int(MAX_TIME_PER_N / (ms_per_cw / 1000))))
        batch_size = min(500, n_cw)

        log(f"  {ms_per_cw:.1f} ms/cw, running {n_cw} codewords...")

        block_errors = 0
        u_bit_errors = 0
        v_bit_errors = 0
        cw_done = 0
        t0 = time.time()

        while cw_done < n_cw:
            bs = min(batch_size, n_cw - cw_done)
            rng = np.random.default_rng(SEED + cw_done + n * 100000)
            be, ube, vbe = sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v,
                                     b, bs, rng, channel)
            block_errors += be
            u_bit_errors += ube
            v_bit_errors += vbe
            cw_done += bs

        elapsed = time.time() - t0
        bler = block_errors / n_cw
        ber_u = u_bit_errors / max(1, n_cw * ku)
        ber_v = v_bit_errors / max(1, n_cw * kv)

        log(f"  BLER={bler:.6f}  ({block_errors}/{n_cw})  {elapsed:.1f}s")

        all_results.append({
            "N": N, "n": n, "ku": ku, "kv": kv,
            "Ru": round(Ru, 6), "Rv": round(Rv, 6),
            "bler": bler, "ber_u": ber_u, "ber_v": ber_v,
            "block_errors": block_errors, "n_codewords": n_cw,
            "time_s": round(elapsed, 2),
        })
        log()

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "bler_vs_N_classB_L1.json")
    output = {
        "description": "BLER vs N — ABN-MAC Class B, SC (L=1)",
        "channel": "ABN-MAC",
        "class": "B", "L": 1,
        "rho": RHO,
        "Ru_dir": RU_DIR, "Rv_dir": RV_DIR,
        "sum_rate_target": RHO * (RU_DIR + RV_DIR),
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Results saved to {json_path}")

    # ── Plot ──
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.size'] = 11
    import matplotlib.pyplot as plt

    Ns = [r["N"] for r in all_results]
    ns = [r["n"] for r in all_results]
    blers = [r["bler"] for r in all_results]

    # Filter out zero BLER
    ns_plot = [n for n, b in zip(ns, blers) if b > 0]
    bl_plot = [b for b in blers if b > 0]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.semilogy(ns_plot, bl_plot, 'b-s', markersize=8,
                markerfacecolor='blue', markeredgecolor='black',
                markeredgewidth=0.5, linewidth=1.5)

    ax.set_xlabel(r'$n = \log_2 N$', fontsize=13)
    ax.set_ylabel('BLER', fontsize=13)
    ax.set_title(
        r'ABN-MAC Class B, SC (L=1), $\rho$=' + f'{RHO}'
        + r', $R_x+R_y$=' + f'{RHO*(RU_DIR+RV_DIR):.2f}',
        fontsize=11)
    ax.set_xticks(range(3, 11))
    ax.set_xticklabels([f'{i}\n({1<<i})' for i in range(3, 11)], fontsize=9)
    ax.set_ylim(1e-4, 1)
    ax.grid(True, which='major', alpha=0.4, linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)
    ax.tick_params(which='both', direction='in', top=True, right=True)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "bler_vs_N_classB_L1")
    fig.savefig(fig_path + ".pdf", dpi=150)
    fig.savefig(fig_path + ".png", dpi=150)
    log(f"Figure saved to {fig_path}.{{pdf,png}}")

    # ── LaTeX PDF ──
    tex_path = os.path.join(FIGURES_DIR, "bler_vs_N_classB_L1.tex")
    with open(tex_path, "w") as f:
        f.write(r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\begin{document}

\begin{center}
{\Large \textbf{BLER vs Block Length --- ABN-MAC, Class B, SC (L=1)}}
\end{center}

\vspace{0.5em}

\noindent
Channel: Additive Binary Noise MAC (ABN-MAC), $Z = (X \oplus E_x, Y \oplus E_y)$,
$p_{E_x,E_y} = \begin{pmatrix} 0.1286 & 0.0175 \\ 0.0175 & 0.8364 \end{pmatrix}$.\\
Capacity: $I(Z;X,Y) \approx 1.200$ bits.\\
Code class B: direction $(R_u, R_v) = (0.5, 0.7)$, path $0^{N/2} 1^N 0^{N/2}$.\\
Rate scaling: $\rho = """ + f"{RHO}" + r"""$, target sum rate $R_x + R_y = """ + f"{RHO*(RU_DIR+RV_DIR):.2f}" + r"""$.\\
Decoder: Successive Cancellation (L=1).\\
Design: Monte Carlo genie-aided.

\vspace{1em}

\begin{center}
\begin{tabular}{rrrrrrr}
\toprule
$N$ & $n$ & $k_u$ & $k_v$ & $R_x + R_y$ & BLER & Codewords \\
\midrule
""")
        for r in all_results:
            bler_str = f"{r['bler']:.6f}" if r['bler'] > 0 else "$< 2 \\times 10^{-5}$"
            f.write(f"{r['N']:>5d} & {r['n']:>2d} & {r['ku']:>4d} & {r['kv']:>4d} "
                    f"& {r['Ru']+r['Rv']:.4f} & {bler_str} & {r['n_codewords']:>6d} \\\\\n")

        f.write(r"""\bottomrule
\end{tabular}
\end{center}

\vspace{1em}

\begin{center}
\includegraphics[width=0.85\textwidth]{bler_vs_N_classB_L1.pdf}
\end{center}

\end{document}
""")
    log(f"LaTeX saved to {tex_path}")

    # Compile PDF
    import subprocess
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", FIGURES_DIR, tex_path],
            capture_output=True, timeout=30)
        if result.returncode == 0:
            log(f"PDF compiled: {os.path.join(FIGURES_DIR, 'bler_vs_N_classB_L1.pdf')}")
        else:
            log(f"pdflatex failed (return code {result.returncode}), .tex file still available")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log("pdflatex not found or timed out — .tex file saved for manual compilation")

    # Print summary table
    log()
    log("=" * 70)
    log(f"  {'N':>5s}  {'n':>2s}  {'ku':>4s}  {'kv':>4s}  {'Ru+Rv':>7s}  {'BLER':>10s}  {'n_cw':>7s}")
    log("-" * 70)
    for r in all_results:
        log(f"  {r['N']:5d}  {r['n']:2d}  {r['ku']:4d}  {r['kv']:4d}  "
            f"{r['Ru']+r['Rv']:7.4f}  {r['bler']:10.6f}  {r['n_codewords']:7d}")
    log("=" * 70)


if __name__ == "__main__":
    main()
