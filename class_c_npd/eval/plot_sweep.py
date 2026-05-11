"""
Plot the GMAC Class C sweep results: BLER vs N curves for SC and NPD.

Reads:
  - results/gmac_sc_reference_50pct.json (SC baseline)
  - results/curriculum_sweep_results.json (NPD results from sweep)

Produces:
  - results/plot_bler_vs_N.png    BLER curves on log scale
  - results/plot_ratio_vs_N.png   NPD/SC ratio with target line
  - results/sweep_summary.md      Markdown table of results
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results')


def load_results():
    sc_path = os.path.join(RESULTS_DIR, 'gmac_sc_reference_50pct.json')
    npd_path = os.path.join(RESULTS_DIR, 'curriculum_sweep_results.json')

    with open(sc_path) as f:
        sc_data = json.load(f)

    npd_data = {}
    if os.path.exists(npd_path):
        with open(npd_path) as f:
            npd_data = json.load(f)

    return sc_data, npd_data


def make_plots():
    sc_data, npd_data = load_results()

    # Extract data
    Ns_sc = sorted(int(k) for k in sc_data['results'].keys())
    bler_sc = [sc_data['results'][str(N)]['sc_bler'] for N in Ns_sc]

    Ns_npd = sorted(int(k) for k in npd_data.keys()) if npd_data else []
    bler_npd = [npd_data[str(N)]['chained']['bler'] for N in Ns_npd]
    ci_low = [npd_data[str(N)]['chained']['ci_low'] for N in Ns_npd]
    ci_high = [npd_data[str(N)]['chained']['ci_high'] for N in Ns_npd]

    # Try to import matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping plots')
        return

    # ─── Plot 1: BLER vs N ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(Ns_sc, bler_sc, 'k-o', label='Analytical SC', linewidth=2, markersize=8)
    if Ns_npd:
        ax.semilogy(Ns_npd, bler_npd, 'r--s', label='NPD Class C (chained)', linewidth=2, markersize=8)
        # CI as filled region
        ax.fill_between(Ns_npd, ci_low, ci_high, alpha=0.2, color='red', label='95% Wilson CI')
    ax.set_xlabel('N (block length)')
    ax.set_ylabel('BLER (log scale)')
    ax.set_title('GMAC Class C: NPD vs SC at 50% per-user capacity, SNR=6 dB')
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ns_sc)
    ax.set_xticklabels([str(N) for N in Ns_sc])
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, 'plot_bler_vs_N.png')
    plt.savefig(out1, dpi=120)
    plt.close()
    print(f'  Saved {out1}')

    # ─── Plot 2: Ratio vs N ─────────────────────────────────────────────────
    if Ns_npd:
        fig, ax = plt.subplots(figsize=(8, 5))
        ratios = [bler_npd[i] / bler_sc[Ns_sc.index(Ns_npd[i])] for i in range(len(Ns_npd))]
        ax.plot(Ns_npd, ratios, 'b-o', linewidth=2, markersize=10, label='NPD / SC')
        ax.axhline(1.0, color='k', linestyle='-', alpha=0.5, label='SC (=1.0x)')
        ax.axhline(1.5, color='r', linestyle='--', alpha=0.5, label='1.5x SC (target)')
        ax.set_xlabel('N (block length)')
        ax.set_ylabel('NPD BLER / SC BLER')
        ax.set_title('Ratio of NPD chained BLER to analytical SC BLER')
        ax.set_xscale('log', base=2)
        ax.set_xticks(Ns_npd)
        ax.set_xticklabels([str(N) for N in Ns_npd])
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='best')
        ax.set_ylim([0, max(3, max(ratios) * 1.2)])
        plt.tight_layout()
        out2 = os.path.join(RESULTS_DIR, 'plot_ratio_vs_N.png')
        plt.savefig(out2, dpi=120)
        plt.close()
        print(f'  Saved {out2}')


def make_summary_table():
    sc_data, npd_data = load_results()

    lines = ['# GMAC Class C NPD Sweep Summary', '']
    lines.append('Channel: Gaussian MAC, SNR = 6 dB')
    lines.append('Path: Class C (path_i = N), 50% of per-user capacity')
    lines.append('')
    lines.append('| N | ku | kv | SC BLER | NPD BLER | 95% CI | Ratio | Pass? |')
    lines.append('|---|---|---|---|---|---|---|---|')

    Ns_sc = sorted(int(k) for k in sc_data['results'].keys())
    for N in Ns_sc:
        sc = sc_data['results'][str(N)]
        sc_bler = sc['sc_bler']
        ku, kv = sc['ku'], sc['kv']

        if str(N) in npd_data:
            npd = npd_data[str(N)]
            chained = npd['chained']
            bler = chained['bler']
            ci = f'[{chained["ci_low"]:.4f}, {chained["ci_high"]:.4f}]'
            ratio = chained['ratio_to_sc']
            passed = 'PASS' if chained['pass'] else 'FAIL'
            lines.append(f'| {N} | {ku} | {kv} | {sc_bler:.4f} | {bler:.4f} | {ci} | {ratio:.2f}x | {passed} |')
        else:
            lines.append(f'| {N} | {ku} | {kv} | {sc_bler:.4f} | (pending) | - | - | - |')

    out = os.path.join(RESULTS_DIR, 'sweep_summary.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved {out}')


def main():
    print('Generating sweep plots and summary...')
    make_plots()
    make_summary_table()
    print('Done.')


if __name__ == '__main__':
    main()
