#!/usr/bin/env python3
"""
plot_bemac_comprehensive.py — Comprehensive BEMAC comparison plot.

Combines all BEMAC Class B (Ru~0.50, Rv~0.70) results:
  - SC decoder
  - SCL (L=32) decoder
  - NN-SC (neural greedy, L=1)
  - NN-SCL (neural list, L=4)

Saves to: results/bemac/bemac_classB_Ru50_Rv70_nn_scl/
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(BASE, 'results', 'bemac', 'bemac_classB_Ru50_Rv70_nn_scl')

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ─── Collect all data ──────────────────────────────────────────────

    # SC and NN-SC from bemac_nn_vs_sc_complete.json
    nn_vs_sc_path = os.path.join(BASE, 'results', 'bemac',
                                  'bemac_classB_Ru50_Rv70_nn_vs_sc',
                                  'bemac_nn_vs_sc_complete.json')
    with open(nn_vs_sc_path) as f:
        nn_vs_sc = json.load(f)

    # SCL-32 from bemac_classB_scl32.json
    scl32_path = os.path.join(BASE, 'results', 'bemac',
                               'bemac_classB_Ru50_Rv70_scl32',
                               'bemac_classB_scl32.json')
    with open(scl32_path) as f:
        scl32 = json.load(f)

    # NN-SCL from bemac_nn_scl_results.json
    nn_scl_path = os.path.join(RESULTS_DIR, 'bemac_nn_scl_results.json')
    with open(nn_scl_path) as f:
        nn_scl = json.load(f)

    # ─── Build data arrays ──────────────────────────────────────────────

    # SC decoder
    sc_data = {}
    for k, v in nn_vs_sc.items():
        N = int(k)
        bler = v.get('sc_bler', 0)
        if bler and bler > 0:
            sc_data[N] = bler

    # NN-SC decoder
    nn_sc_data = {}
    for k, v in nn_vs_sc.items():
        N = int(k)
        bler = v.get('nn_bler', 0)
        if bler and bler > 0:
            nn_sc_data[N] = bler

    # SCL-32 decoder
    scl32_data = {}
    for k, v in scl32.items():
        N = int(k)
        bler = v.get('scl32_bler', 0)
        if bler and bler > 0:
            scl32_data[N] = bler

    # NN-SCL (L=4) decoder
    nn_scl4_data = {}
    for k, v in nn_scl.items():
        N = int(k)
        # Check both nn_scl4_bler and nn_sc_bler (for L=1 entries)
        bler = v.get('nn_scl4_bler', 0)
        if bler and bler > 0:
            nn_scl4_data[N] = bler

    print("SC data:", sc_data)
    print("NN-SC data:", nn_sc_data)
    print("SCL-32 data:", scl32_data)
    print("NN-SCL L=4 data:", nn_scl4_data)

    # ─── Plot ──────────────────────────────────────────────────────────

    fig, ax = plt.subplots(1, 1, figsize=(11, 7.5))

    # SC
    if sc_data:
        Ns = sorted(sc_data.keys())
        vals = [sc_data[N] for N in Ns]
        ax.semilogy(Ns, vals, 'b-o', markersize=9, linewidth=2.2,
                    label='SC Decoder', zorder=5)

    # SCL-32
    if scl32_data:
        Ns = sorted(scl32_data.keys())
        vals = [scl32_data[N] for N in Ns]
        ax.semilogy(Ns, vals, 'g-^', markersize=9, linewidth=2.2,
                    label='SCL Decoder (L=32)', zorder=5)

    # NN-SC
    if nn_sc_data:
        Ns = sorted(nn_sc_data.keys())
        vals = [nn_sc_data[N] for N in Ns]
        ax.semilogy(Ns, vals, 'r--s', markersize=9, linewidth=2.2,
                    label='Neural SC (L=1)', zorder=5)

    # NN-SCL L=4
    if nn_scl4_data:
        Ns = sorted(nn_scl4_data.keys())
        vals = [nn_scl4_data[N] for N in Ns]
        ax.semilogy(Ns, vals, 'm-D', markersize=9, linewidth=2.2,
                    label='Neural SCL (L=4)', zorder=6)

    # All N values for x-axis ticks
    all_Ns = sorted(set(list(sc_data.keys()) + list(nn_sc_data.keys()) +
                        list(scl32_data.keys()) + list(nn_scl4_data.keys())))

    ax.set_xscale('log', base=2)
    ax.set_xticks(all_Ns)
    ax.set_xticklabels([str(n) for n in all_Ns])
    ax.set_xlabel('Block Length N', fontsize=14)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=14)
    ax.set_title(
        'BEMAC Class B: Comprehensive Decoder Comparison\n'
        r'$R_u \approx 0.50,\; R_v \approx 0.70$',
        fontsize=13
    )
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.tick_params(labelsize=11)

    # Add annotation for key finding
    if 64 in nn_scl4_data and 64 in sc_data:
        ratio = nn_scl4_data[64] / sc_data[64]
        ax.annotate(f'NN-SCL: {ratio:.1f}x SC at N=64',
                    xy=(64, nn_scl4_data[64]),
                    xytext=(100, nn_scl4_data[64] * 0.25),
                    fontsize=10, color='purple', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
                    ha='center')

    plt.tight_layout()

    # Save
    fig.savefig(os.path.join(RESULTS_DIR, 'bemac_comprehensive_comparison.png'), dpi=200)
    fig.savefig(os.path.join(RESULTS_DIR, 'bemac_comprehensive_comparison.pdf'))
    print(f"\nPlot saved to {RESULTS_DIR}")
    plt.close()

    # ─── Save combined data as JSON ───────────────────────────────────

    combined = {
        'description': 'BEMAC Class B (Ru~0.50, Rv~0.70) comprehensive comparison',
        'sc': {str(k): v for k, v in sc_data.items()},
        'scl32': {str(k): v for k, v in scl32_data.items()},
        'nn_sc': {str(k): v for k, v in nn_sc_data.items()},
        'nn_scl4': {str(k): v for k, v in nn_scl4_data.items()},
    }
    with open(os.path.join(RESULTS_DIR, 'bemac_comprehensive_data.json'), 'w') as f:
        json.dump(combined, f, indent=2)

    # ─── Summary table ────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print(f"BEMAC Class B Comprehensive Comparison (Ru~0.50, Rv~0.70)")
    print(f"{'='*80}")
    print(f"{'N':>6}  {'SC':>10}  {'SCL-32':>10}  {'NN-SC':>10}  {'NN-SCL4':>10}  {'NN-SCL4/SC':>12}")
    print(f"{'-'*70}")
    for N in sorted(all_Ns):
        sc_s = f"{sc_data[N]:.6f}" if N in sc_data else "N/A"
        scl_s = f"{scl32_data[N]:.6f}" if N in scl32_data else "N/A"
        nn_s = f"{nn_sc_data[N]:.6f}" if N in nn_sc_data else "N/A"
        nscl_s = f"{nn_scl4_data[N]:.6f}" if N in nn_scl4_data else "N/A"
        if N in nn_scl4_data and N in sc_data and sc_data[N] > 0:
            ratio = f"{nn_scl4_data[N]/sc_data[N]:.2f}x"
        else:
            ratio = "N/A"
        print(f"{N:>6}  {sc_s:>10}  {scl_s:>10}  {nn_s:>10}  {nscl_s:>10}  {ratio:>12}")


    # ─── Save XLSX ──────────────────────────────────────────────────

    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "BEMAC Comprehensive"
        ws.append(['BEMAC Class B Comprehensive Decoder Comparison (Ru~0.50, Rv~0.70)'])
        ws.append([])
        ws.append(['N', 'SC BLER', 'SCL-32 BLER', 'NN-SC BLER', 'NN-SCL(L=4) BLER',
                    'NN-SCL4/SC Ratio'])
        for N in sorted(all_Ns):
            row = [N,
                   sc_data.get(N),
                   scl32_data.get(N),
                   nn_sc_data.get(N),
                   nn_scl4_data.get(N)]
            if N in nn_scl4_data and N in sc_data and sc_data[N] > 0:
                row.append(nn_scl4_data[N] / sc_data[N])
            else:
                row.append(None)
            ws.append(row)
        wb.save(os.path.join(RESULTS_DIR, 'bemac_comprehensive_comparison.xlsx'))
        print(f"\nXLSX saved.")
    except ImportError:
        print("openpyxl not available, XLSX not saved.")


if __name__ == '__main__':
    main()
