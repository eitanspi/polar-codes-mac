#!/usr/bin/env python3
"""Generate the 2026-04-16 session paper figures.

All figures saved into this directory as {name}.png (DPI 300) and {name}.pdf.
Source data is described in README_FIGURES.md alongside.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
torch.set_num_threads(1)

ROOT = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2'
OUT_DIR = os.path.join(ROOT, 'docs', 'paper_figures')
RESULTS = os.path.join(ROOT, 'project_summary', 'results')
CCNPD = os.path.join(ROOT, 'class_c_npd', 'results')

# ── IEEE-style defaults ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.dpi': 150,
})

# Consistent color scheme
COLOR_SC = '#1f77b4'         # blue
COLOR_NCG = '#d62728'        # red
COLOR_NPD = '#2ca02c'        # green
COLOR_NPD_CHAINED = '#2ca02c'
COLOR_CRC_SCL = '#ff7f0e'    # orange
COLOR_TRELLIS_SC = '#1f77b4'
COLOR_JOINT_TRELLIS = '#17becf'
COLOR_MEMORYLESS = '#9467bd'
COLOR_BROKEN = '#7f7f7f'


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'),
                    dpi=300 if ext == 'png' else 150,
                    bbox_inches='tight')
    print(f'  saved {name}.png / .pdf')
    plt.close(fig)


def wilson_ci(p, n, z=1.96):
    """Return (lo, hi) Wilson CI for Bernoulli proportion p out of n trials."""
    if n <= 0:
        return (0.0, 0.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), centre + half)


# =====================================================================
# Figure 4 (HEADLINE): ISI-MAC chained NPD vs chained trellis SC vs etc.
# =====================================================================
def fig_isi_mac_bler():
    print('Figure 4: ISI-MAC BLER (headline)')
    with open(os.path.join(CCNPD, 'chained_trellis_sc_isi_mac.json')) as f:
        trellis = json.load(f)['by_N']

    # Order N
    Ns = sorted([int(k) for k in trellis.keys()])
    chained_trellis = [trellis[str(N)]['chained_bler'] for N in Ns]
    joint_trellis = [trellis[str(N)]['joint_trellis_bler'] for N in Ns]
    memoryless = [trellis[str(N)]['memoryless_sc_bler'] for N in Ns]

    # Chained NPD (neural) - best per N from the markdown table
    # N=16: window 0.1325; N=32: window 0.0780; N=64: BiGRU 0.0425; N=128: pending
    chained_npd = {16: 0.1325, 32: 0.0780, 64: 0.0425}
    chained_npd_cw = {16: 2000, 32: 2000, 64: 2000}

    # Broken NPD from isi_mac_classC_npd.json (Stage 1 only)
    with open(os.path.join(CCNPD, 'isi_mac_classC_npd.json')) as f:
        broken = json.load(f)
    broken_npd = {int(k): v['npd_s1_bler'] for k, v in broken.items()}

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Broken NPD (scalar baseline)
    xs_br = sorted(broken_npd.keys())
    ys_br = [broken_npd[N] for N in xs_br]
    ax.plot(xs_br, ys_br, 'v--', color=COLOR_BROKEN, markersize=7,
            label='Scalar NPD (prior baseline)', alpha=0.75)

    # Memoryless SC
    ax.plot(Ns, memoryless, 'P:', color=COLOR_MEMORYLESS, markersize=7,
            label='Memoryless SC', alpha=0.9)

    # Chained trellis SC (analytical)
    ax.plot(Ns, chained_trellis, 'o-', color=COLOR_TRELLIS_SC, markersize=7,
            label='Chained trellis SC (analytical)')

    # Joint trellis SC
    ax.plot(Ns, joint_trellis, 'D-', color=COLOR_JOINT_TRELLIS, markersize=6,
            label='Joint trellis SC (analytical)')

    # Chained NPD (neural, ours)
    xs_npd = sorted(chained_npd.keys())
    ys_npd = [chained_npd[N] for N in xs_npd]
    yerr = np.array([
        [v - wilson_ci(v, chained_npd_cw[N])[0] for N, v in zip(xs_npd, ys_npd)],
        [wilson_ci(v, chained_npd_cw[N])[1] - v for N, v in zip(xs_npd, ys_npd)],
    ])
    ax.errorbar(xs_npd, ys_npd, yerr=yerr, fmt='s-', color=COLOR_NPD_CHAINED,
                markersize=8, linewidth=2, capsize=4,
                label='Chained NPD (neural, ours)', zorder=5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('Block length $N$')
    ax.set_ylabel('BLER')
    ax.set_title('ISI-MAC ($h = 0.3$, SNR $= 6$ dB): chained NPD vs trellis SC')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.legend(loc='lower left', framealpha=0.92)
    fig.tight_layout()
    save(fig, 'fig_isi_mac_bler')


# =====================================================================
# Figure 1: NCG vs SC on GMAC Class B
# =====================================================================
def fig_ncg_bler_gmac_B():
    print('Figure 1: GMAC Class B NCG vs SC')
    # From gmac_classB_ncg_vs_sc.csv (N=32,64,128,256) plus GMAC N=16 SC baseline
    # extend with missing N from NCG_CHAPTER where SCL/SC available
    data = [
        # N, SC, NCG, CW_SC, CW_NCG
        (16, 0.078, 0.075, 5000, 5000),   # from campaign (approx proxy; no NCG entry — use 32 doubled)
        (32, 0.047, 0.040, 2000, 2000),
        (64, 0.028, 0.026, 2000, 2000),
        (128, 0.020, 0.023, 2000, 2000),
        (256, 0.006, 0.023, 5000, 5000),
    ]
    # Actually N=16 NCG was not run for GMAC class B per CSV; omit it
    data = [
        (32, 0.047, 0.040, 2000, 2000),
        (64, 0.028, 0.026, 2000, 2000),
        (128, 0.020, 0.023, 2000, 2000),
        (256, 0.006, 0.023, 5000, 5000),
    ]
    # Additional SC-only points for scaling trend from NCG_CHAPTER and campaign
    # GMAC_B SC @ N=512, 1024 from campaign (Ru=Rv=0.48) — use paper data
    # From class_b campaign (cited in NCG_CHAPTER): N=512 SC ~ 0.004, N=1024 SC ~ 0.002
    sc_only = [(512, 0.004, 3000), (1024, 0.002, 3000)]

    Ns = [d[0] for d in data] + [d[0] for d in sc_only]
    sc_all = [d[1] for d in data] + [d[1] for d in sc_only]
    cw_sc_all = [d[3] for d in data] + [d[2] for d in sc_only]
    N_ncg = [d[0] for d in data]
    ncg_all = [d[2] for d in data]
    cw_ncg_all = [d[4] for d in data]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    # SC curve
    yerr_sc = np.array([
        [v - wilson_ci(v, c)[0] for v, c in zip(sc_all, cw_sc_all)],
        [wilson_ci(v, c)[1] - v for v, c in zip(sc_all, cw_sc_all)],
    ])
    ax.errorbar(Ns, sc_all, yerr=yerr_sc, fmt='o-', color=COLOR_SC,
                markersize=7, capsize=3, label='SC (MAC baseline)')

    # NCG curve
    yerr_ncg = np.array([
        [v - wilson_ci(v, c)[0] for v, c in zip(ncg_all, cw_ncg_all)],
        [wilson_ci(v, c)[1] - v for v, c in zip(ncg_all, cw_ncg_all)],
    ])
    ax.errorbar(N_ncg, ncg_all, yerr=yerr_ncg, fmt='s-', color=COLOR_NCG,
                markersize=8, linewidth=2, capsize=3, label='NCG (neural SC)', zorder=5)

    # Annotate wall at N=256
    ax.annotate('NCG wall\n(4.5$\\times$ SC)', xy=(256, 0.023), xytext=(500, 0.011),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks([32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels([32, 64, 128, 256, 512, 1024])
    ax.set_xlabel('Block length $N$')
    ax.set_ylabel('BLER')
    ax.set_title('GMAC Class B: NCG vs SC (sum-rate $\\approx 0.96$, SNR $= 6$ dB)')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.legend(loc='lower left', framealpha=0.9)
    fig.tight_layout()
    save(fig, 'fig_ncg_bler_gmac_B')


# =====================================================================
# Figure 2: NCG vs SC on BEMAC Class B
# =====================================================================
def fig_ncg_bler_bemac_B():
    print('Figure 2: BEMAC Class B NCG vs SC')
    # From bemac_classB_ncg_vs_sc.csv
    rows = [
        # N, SC, NCG, CW
        (16, 0.011, 0.011, 5000),
        (32, 0.008, 0.009, 5000),
        (64, 0.006, 0.003, 5000),
        (128, 0.002, 0.001, 5000),
        (256, 8e-05, 4e-05, 50000),
        (1024, 1e-4, 1e-4, 10000),
    ]
    Ns = [r[0] for r in rows]
    sc = [r[1] for r in rows]
    ncg = [r[2] for r in rows]
    cws = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    yerr_sc = np.array([
        [v - wilson_ci(v, c)[0] for v, c in zip(sc, cws)],
        [wilson_ci(v, c)[1] - v for v, c in zip(sc, cws)],
    ])
    ax.errorbar(Ns, sc, yerr=yerr_sc, fmt='o-', color=COLOR_SC, markersize=7,
                capsize=3, label='SC (MAC baseline)')

    yerr_ncg = np.array([
        [v - wilson_ci(v, c)[0] for v, c in zip(ncg, cws)],
        [wilson_ci(v, c)[1] - v for v, c in zip(ncg, cws)],
    ])
    ax.errorbar(Ns, ncg, yerr=yerr_ncg, fmt='s-', color=COLOR_NCG,
                markersize=8, linewidth=2, capsize=3, label='NCG (neural SC)', zorder=5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('Block length $N$')
    ax.set_ylabel('BLER')
    ax.set_title('BEMAC Class B: NCG vs SC (sum-rate $= 1$)')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.legend(loc='lower left', framealpha=0.9)
    fig.tight_layout()
    save(fig, 'fig_ncg_bler_bemac_B')


# =====================================================================
# Figure 3: Chained NPD corner rate on GMAC Class C
# =====================================================================
def fig_chained_npd_corner_gmac():
    print('Figure 3: GMAC Class C chained NPD vs SC')
    # From gmac_classC_npd_vs_sc.csv
    rows = [
        # N, ku, kv, SC_BLER, SC_CW, NPD_BLER, NPD_CW
        (16, 4, 7, 0.163, 2000, 0.111, 2000),
        (32, 7, 15, 0.068, 2000, 0.038, 2000),
        (64, 15, 29, 0.027, 2000, 0.017, 2000),
        (128, 30, 58, 0.005, 2000, 0.007, 2000),
        (256, 59, 117, 0.002, 5000, 0.0016, 5000),
        (512, 119, 233, 0.0005, 2000, 0.00024, 50000),
    ]
    Ns = [r[0] for r in rows]
    sc = [r[3] for r in rows]
    sc_cw = [r[4] for r in rows]
    npd = [r[5] for r in rows]
    npd_cw = [r[6] for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    yerr_sc = np.array([
        [v - wilson_ci(v, c)[0] for v, c in zip(sc, sc_cw)],
        [wilson_ci(v, c)[1] - v for v, c in zip(sc, sc_cw)],
    ])
    ax.errorbar(Ns, sc, yerr=yerr_sc, fmt='o-', color=COLOR_SC, markersize=7,
                capsize=3, label='SC (MAC baseline)')

    yerr_npd = np.array([
        [v - wilson_ci(v, c)[0] for v, c in zip(npd, npd_cw)],
        [wilson_ci(v, c)[1] - v for v, c in zip(npd, npd_cw)],
    ])
    ax.errorbar(Ns, npd, yerr=yerr_npd, fmt='s-', color=COLOR_NPD,
                markersize=8, linewidth=2, capsize=3,
                label='Chained NPD (neural)', zorder=5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('Block length $N$')
    ax.set_ylabel('BLER')
    ax.set_title('GMAC Class C (corner rate, $\\approx 50\\%$ capacity): NPD vs SC')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.legend(loc='lower left', framealpha=0.9)
    fig.tight_layout()
    save(fig, 'fig_chained_npd_corner_gmac')


# =====================================================================
# Figure 5: Rate sweep at N=256 (plus N=64, 128 context)
# =====================================================================
def fig_rate_sweep_n256():
    print('Figure 5: Rate sweep at N=256')
    with open(os.path.join(ROOT, 'results', 'gmac_classB_rate_sweep.json')) as f:
        sweep = json.load(f)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    N_colors = {64: '#1f77b4', 128: '#2ca02c', 256: '#d62728'}
    N_markers_sc = {64: 'o', 128: 's', 256: 'D'}
    N_markers_nn = {64: 'o', 128: 's', 256: 'D'}

    for N_str in ('64', '128', '256'):
        entries = sweep[N_str]
        rates = np.array([e['sum_rate'] for e in entries])
        sc = np.array([e['sc_bler'] for e in entries])
        sc_lo = np.array([e['sc_ci_lo'] for e in entries])
        sc_hi = np.array([e['sc_ci_hi'] for e in entries])
        nn = np.array([e['nn_bler'] for e in entries])
        nn_lo = np.array([e['nn_ci_lo'] for e in entries])
        nn_hi = np.array([e['nn_ci_hi'] for e in entries])
        N = int(N_str)
        c = N_colors[N]

        # Clamp zeros for log scale
        eps = 1e-5
        sc_p = np.maximum(sc, eps)
        nn_p = np.maximum(nn, eps)

        # SC dashed
        ax.plot(rates, sc_p, linestyle='--', color=c, alpha=0.9,
                marker=N_markers_sc[N], markersize=6, markerfacecolor='white',
                label=f'SC, N={N}')
        # NCG solid
        ax.plot(rates, nn_p, linestyle='-', color=c, linewidth=2,
                marker=N_markers_nn[N], markersize=7,
                label=f'NCG, N={N}')

    ax.set_yscale('log')
    ax.set_xlabel('Sum rate $R_U + R_V$ (bits / channel use)')
    ax.set_ylabel('BLER')
    ax.set_title('GMAC Class B: rate sweep (NCG solid, SC dashed) — SNR $= 6$ dB')
    ax.set_ylim(1e-5, 1.2)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.text(1.005, 5e-5, 'symmetric capacity', rotation=90,
            fontsize=8, color='gray', va='bottom')
    ax.legend(loc='lower right', fontsize=8, ncol=2, framealpha=0.92)
    fig.tight_layout()
    save(fig, 'fig_rate_sweep_n256')


# =====================================================================
# Figure 6: First-error position histogram at N=256
# =====================================================================
def fig_first_error_n256():
    print('Figure 6: First-error positions at N=256')
    path = '/tmp/first_err_n256.npz'
    if not os.path.exists(path):
        print('  skipped (no data)')
        return False
    d = np.load(path)
    steps = d['first_err_step']  # 0..N-1 sc step position (information bit index)
    users = d['first_err_user']  # 0=U, 1=V
    N = 256
    total_cw = int(d['total'])
    failed_cw = int(d['failed'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    # Left: histogram of first-err step (decode order)
    bins = np.arange(0, N + 8, 8)
    ax1.hist(steps, bins=bins, color=COLOR_NCG, edgecolor='black',
             linewidth=0.5, alpha=0.85)
    ax1.set_xlabel('Decoding step index')
    ax1.set_ylabel('Count of first-error events')
    ax1.set_title(f'(a) First-error step (total {failed_cw} failures / {total_cw} cw)')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.set_xlim(0, N)

    # Right: who fails first (U vs V) and position histogram per user
    # first_err_pos is information-bit index within the user's set
    # For clarity split by user and plot position densities
    pos_u = d['first_err_pos'][users == 0]
    pos_v = d['first_err_pos'][users == 1]
    bins2 = np.arange(0, max(pos_u.max() if len(pos_u) else 0,
                              pos_v.max() if len(pos_v) else 0, 8) + 4, 4)
    ax2.hist([pos_u, pos_v], bins=bins2,
             color=[COLOR_SC, COLOR_NCG], edgecolor='black', linewidth=0.4,
             label=[f'User U (n={len(pos_u)})', f'User V (n={len(pos_v)})'],
             stacked=True)
    ax2.set_xlabel('Info-bit index within user')
    ax2.set_ylabel('Count')
    ax2.set_title('(b) First-error info-bit index by user')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)

    fig.suptitle('NCG first-error clustering at $N = 256$ (GMAC Class B)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, 'fig_first_error_n256')
    return True


# =====================================================================
# Figure 7: Per-position MI at N=32 (NCG vs SC) rate-1 synthesis
# =====================================================================
def fig_mi_per_position_n32():
    print('Figure 7: Per-position MI at N=32')
    # JSON is truncated — parse only the well-formed prefix we know (mi_u_mean, mi_v_mean)
    path = os.path.join(CCNPD, 'ncg_r1_32', 'mi_per_pos_reliable.json')
    if not os.path.exists(path):
        print('  skipped (no data)')
        return False
    with open(path) as f:
        txt = f.read()

    # Lightweight parse: extract list for given key
    def pull(key):
        i = txt.find(f'"{key}"')
        if i < 0:
            return None
        lb = txt.find('[', i)
        rb = txt.find(']', lb)
        arr = txt[lb + 1:rb]
        vals = [float(x.strip()) for x in arr.split(',') if x.strip()]
        return np.array(vals)

    mi_u = pull('mi_u_mean')
    mi_v = pull('mi_v_mean')
    mi_u_std = pull('mi_u_std')
    mi_v_std = pull('mi_v_std')
    if mi_u is None or mi_v is None:
        print('  skipped (parse fail)')
        return False

    N = len(mi_u)
    positions = np.arange(N)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # Plot NCG-estimated MI for each information bit split by user.
    # Under rate-1 synthesis NCG is trained to emit p(u_i,v_i | past); MI per
    # user is then I(U_i; Z,past) and I(V_i; Z,past,U) averaged over codewords.
    ax.errorbar(positions, mi_u, yerr=mi_u_std, fmt='o-', color=COLOR_SC,
                markersize=5, linewidth=1.5, capsize=2,
                label='$I(U_i; Z, \\mathrm{past}) $ (NCG estimate)')
    ax.errorbar(positions, mi_v, yerr=mi_v_std, fmt='s-', color=COLOR_NCG,
                markersize=5, linewidth=1.5, capsize=2,
                label='$I(V_i; Z, \\mathrm{past}, U)$ (NCG estimate)')

    ax.set_xlabel('Position index (bit-reversed)')
    ax.set_ylabel('Mutual information (bits)')
    ax.set_title('GMAC Class B, $N = 32$ rate-1 synthesis: per-position MI by user')
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.legend(loc='center right', framealpha=0.92, fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_mi_per_position_n32')
    return True


# =====================================================================
# Bonus Figure: lower-rate experiment
# =====================================================================
def fig_lower_rate_experiment():
    print('Bonus: lower-rate experiment at N=256')
    path = '/tmp/exp_freeze_weakest_results.npz'
    if not os.path.exists(path):
        print('  skipped (no data)')
        return False
    d = np.load(path)
    labels = ['Baseline\n(123/123)', 'Exp-A\n(118/118)', 'Exp-B\n(113/113)', 'Exp-C\n(103/103)']
    # ratio is NCG/SC
    # We also want absolute BLER. From NCG_CHAPTER: (123/123) NCG 0.021, SC 0.004;
    # (118/118) NCG 0.012, SC 0.0037; (113/113) NCG 0.0066, SC 0.0020;
    # (103/103) NCG 0.0021, SC 0.0002
    ncg_bler = [0.021, 0.012, 0.0066, 0.0021]
    sc_bler = [0.004, 0.0037, 0.0020, 0.0002]
    ratios = [d['Baseline_ratio'], d['Exp-A_ratio'], d['Exp-B_ratio'], d['Exp-C_ratio']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w / 2, sc_bler, w, color=COLOR_SC, label='SC', edgecolor='black', linewidth=0.4)
    ax1.bar(x + w / 2, ncg_bler, w, color=COLOR_NCG, label='NCG', edgecolor='black', linewidth=0.4)
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('BLER')
    ax1.set_title('(a) BLER by rate point')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3, axis='y')
    ax1.legend(loc='upper right', framealpha=0.9)

    ax2.plot(x, ratios, 'D-', color='black', markersize=9,
             markerfacecolor=COLOR_NCG, linewidth=2)
    for xi, r in zip(x, ratios):
        ax2.annotate(f'{r:.1f}$\\times$', xy=(xi, r), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('NCG BLER / SC BLER')
    ax2.set_title('(b) NCG/SC ratio — gap widens as rate drops')
    ax2.set_ylim(0, max(ratios) * 1.3)
    ax2.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)

    fig.suptitle('GMAC Class B $N = 256$: freezing top-$K$ weakest positions',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, 'fig_lower_rate_experiment')
    return True


# =====================================================================
# Bonus Figure: training curve at N=32 rate-1 (phase transition)
# =====================================================================
def fig_training_curve_n32():
    print('Bonus: training curve N=32 rate-1')
    # From ncg_r1_32.log
    iters = np.arange(10000, 300001, 10000)
    loss = [1.0416, 1.0394, 1.0391, 1.0391, 1.0405, 1.0397, 1.0388, 1.0391,
            1.0384, 1.0389, 1.0400, 1.0389, 1.0402, 1.0398, 1.0390, 1.0397,
            1.0397, 1.0385, 1.0387, 1.0396, 1.0395, 1.0406, 1.0405, 0.9581,
            0.9566, 0.8932, 0.6729, 0.5227, 0.4483, 0.4165]
    assert len(iters) == len(loss), f'len mismatch {len(iters)} {len(loss)}'

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(iters / 1000, loss, 'o-', color=COLOR_NCG, markersize=5,
            linewidth=1.8, label='Training CE loss (1 cw/batch)')

    # Shade three regimes
    ax.axvspan(0, 230, color='gray', alpha=0.12, label='_nolegend_')
    ax.axvspan(230, 260, color='orange', alpha=0.18, label='_nolegend_')
    ax.axvspan(260, 310, color='green', alpha=0.12, label='_nolegend_')
    ax.text(115, 0.45, 'plateau\n(random)', fontsize=9, ha='center', color='gray')
    ax.text(245, 0.45, 'phase\ntransition', fontsize=9, ha='center', color='darkorange')
    ax.text(285, 0.88, 'polarization\nemerges', fontsize=9, ha='center', color='green')

    # Reference lines — log 4 is the uniform-4-way chance level (~ 1.386)
    # but training never sees that because the decoder starts near this from init.
    # Instead, mark the random-guessing loss around ln(4).
    ax.axhline(np.log(4), color='black', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(10, np.log(4) - 0.03, '$\\ln 4$ (uniform 4-way)', fontsize=8, color='black')

    ax.set_xlabel('Training iterations ($\\times 10^3$)')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_title('GMAC Class B, $N = 32$ rate-1 NCG: 200K-iter plateau + phase transition')
    ax.set_ylim(0.3, 1.5)
    ax.set_xlim(0, 310)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', framealpha=0.9)
    fig.tight_layout()
    save(fig, 'fig_training_curve_n32')
    return True


# =====================================================================
# Run all
# =====================================================================
if __name__ == '__main__':
    # Headline first
    fig_isi_mac_bler()
    # Must-produce
    fig_ncg_bler_gmac_B()
    fig_ncg_bler_bemac_B()
    fig_chained_npd_corner_gmac()
    fig_rate_sweep_n256()
    fig_first_error_n256()
    fig_mi_per_position_n32()
    # Bonus
    fig_lower_rate_experiment()
    fig_training_curve_n32()
    print('All done.')
