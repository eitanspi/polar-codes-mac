"""
eval_bemac_classB_scl32.py
==========================
BEMAC Class B (Ru50/Rv70) — SCL-32 evaluation at N=16,32,64,128.
Uses v3 optimized SCL decoder.
5K CW per N. Saves to bemac_classB_Ru50_Rv70_scl32/.
"""

import sys
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v3')

import os, math, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polar.design import make_path
from polar.channels import BEMAC
from polar.eval import MACEval


DESIGNS_DIR = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs'
RESULTS_DIR = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results/bemac/bemac_classB_Ru50_Rv70_scl32'


def load_design(N):
    """Load BEMAC Class B design for given N."""
    n = int(math.log2(N))
    path = os.path.join(DESIGNS_DIR, f'bemac_B_n{n}.npz')
    assert os.path.exists(path), f"Design file not found: {path}"
    d = np.load(path)
    eu = d['u_error_rates']
    ev = d['v_error_rates']
    su = np.argsort(eu)
    sv = np.argsort(ev)

    ku = round(0.50 * N)
    kv = round(0.70 * N)
    path_i = N // 2  # Class B: symmetric path

    Au = sorted([int(i + 1) for i in su[:ku]])
    Av = sorted([int(i + 1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    frozen_u = {p: 0 for p in sorted(all_pos - set(Au))}
    frozen_v = {p: 0 for p in sorted(all_pos - set(Av))}

    return ku, kv, Au, Av, frozen_u, frozen_v, path_i


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    channel = BEMAC()

    Ns =       [16,   32,   64,   128]
    n_cws =    [5000, 5000, 5000, 5000]
    batch_sz = [100,  50,   25,   10]

    # Existing NN vs SC results for comparison
    existing_nn_sc = {
        16:  {'nn_bler': 0.0114, 'sc_bler': 0.0106},
        32:  {'nn_bler': 0.0088, 'sc_bler': 0.008},
        64:  {'nn_bler': 0.003,  'sc_bler': 0.0056},
        128: {'nn_bler': 0.0012, 'sc_bler': 0.002},
    }

    results = {}

    for N, n_cw, bs in zip(Ns, n_cws, batch_sz):
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N} — SCL-32", flush=True)
        print(f"{'=' * 60}", flush=True)

        ku, kv, Au, Av, frozen_u, frozen_v, path_i = load_design(N)
        b = make_path(N, path_i)

        print(f"  ku={ku}, kv={kv}, path_i={path_i}, n_cw={n_cw}", flush=True)

        ev = MACEval(channel, decoder_type='scl', L=32, backend='interleaved')
        t0 = time.time()
        _, _, bler = ev.run(N, b, Au, Av, frozen_u, frozen_v,
                            n_codewords=n_cw, batch_size=bs, verbose=True)
        elapsed = time.time() - t0
        print(f"  SCL-32: BLER={bler:.6f}, {elapsed:.1f}s ({n_cw} codewords)", flush=True)

        sc_bler = existing_nn_sc.get(N, {}).get('sc_bler', None)
        nn_bler = existing_nn_sc.get(N, {}).get('nn_bler', None)

        results[N] = {
            'N': N, 'ku': ku, 'kv': kv,
            'scl32_bler': bler,
            'sc_bler': sc_bler,
            'nn_bler': nn_bler,
            'n_cw': n_cw,
            'Ru': ku / N, 'Rv': kv / N,
            'time_s': round(elapsed, 1),
        }

        # Save intermediate
        with open(os.path.join(RESULTS_DIR, 'bemac_classB_scl32.json'), 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Summary
    print(f"\n\n{'=' * 70}", flush=True)
    print(f"SUMMARY: BEMAC Class B (Ru~0.50, Rv~0.70) — SCL-32 vs SC vs NN", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'N':>6} {'ku':>4} {'kv':>4} {'SCL-32':>12} {'SC':>12} {'NN':>12} {'CW':>6}", flush=True)
    print(f"{'-' * 60}", flush=True)
    for N in sorted(results.keys()):
        r = results[N]
        scl_s = f"{r['scl32_bler']:.6f}"
        sc_s = f"{r['sc_bler']:.6f}" if r['sc_bler'] is not None else "N/A"
        nn_s = f"{r['nn_bler']:.6f}" if r['nn_bler'] is not None else "N/A"
        print(f"{N:>6} {r['ku']:>4} {r['kv']:>4} {scl_s:>12} {sc_s:>12} {nn_s:>12} {r['n_cw']:>6}", flush=True)

    # Save XLSX
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['BE-MAC — Class B, (Ru=0.50, Rv=0.70), SCL-32 vs SC vs NN'])
    ws.append(['Capacity: I(Z;X)=0.5000 | I(Z;Y|X)=1.0000 | Frozen sets: 50K MC trials'])
    ws.append([])
    ws.append(['N', 'Ru', 'Rv', 'Sum Rate', 'SCL-32 BLER', 'SC BLER', 'NN BLER', 'Codewords'])
    for N in sorted(results.keys()):
        r = results[N]
        ws.append([N, r['Ru'], r['Rv'], r['Ru'] + r['Rv'],
                   r['scl32_bler'], r['sc_bler'], r['nn_bler'], r['n_cw']])
    ws.append([])
    ws.append(['Capacity', 0.5, 1.0, 1.5])
    wb.save(os.path.join(RESULTS_DIR, 'bemac_classB_Ru50_Rv70_scl32.xlsx'))
    print(f"\nXLSX saved.", flush=True)

    # Plot
    plot_Ns = sorted(results.keys())

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # SCL-32
    scl_Ns = [N for N in plot_Ns if results[N]['scl32_bler'] > 0]
    scl_vals = [results[N]['scl32_bler'] for N in scl_Ns]
    if scl_Ns:
        ax.semilogy(scl_Ns, scl_vals, 'g-^', markersize=8, linewidth=2,
                    label='SCL-32 Decoder', zorder=5)

    # SC
    sc_Ns = [N for N in plot_Ns if results[N]['sc_bler'] and results[N]['sc_bler'] > 0]
    sc_vals = [results[N]['sc_bler'] for N in sc_Ns]
    if sc_Ns:
        ax.semilogy(sc_Ns, sc_vals, 'b-o', markersize=8, linewidth=2,
                    label='SC Decoder', zorder=5)

    # NN
    nn_Ns = [N for N in plot_Ns if results[N]['nn_bler'] and results[N]['nn_bler'] > 0]
    nn_vals = [results[N]['nn_bler'] for N in nn_Ns]
    if nn_Ns:
        ax.semilogy(nn_Ns, nn_vals, 'r--s', markersize=8, linewidth=2,
                    label='Pure Neural NCG Decoder', zorder=5)

    ax.set_xscale('log', base=2)
    ax.set_xticks(plot_Ns)
    ax.set_xticklabels([str(n) for n in plot_Ns])
    ax.set_xlabel('Block Length N', fontsize=13)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=13)
    ax.set_title(
        'BEMAC Class B: SCL-32 vs SC vs Neural Decoder\n'
        r'$R_u \approx 0.50,\; R_v \approx 0.70$',
        fontsize=12
    )
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    fig.savefig(os.path.join(RESULTS_DIR, 'bemac_classB_Ru50_Rv70_scl32.png'), dpi=200)
    fig.savefig(os.path.join(RESULTS_DIR, 'bemac_classB_Ru50_Rv70_scl32.pdf'))
    print(f"Plot saved.", flush=True)
    plt.close()


if __name__ == '__main__':
    main()
