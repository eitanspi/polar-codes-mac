"""
eval_bemac_classA_nn.py
=======================
BEMAC Class A (Ru62/Rv62) — Pure Neural decoder evaluation.
Class A: path_i=0, symmetric rates ku=kv=round(0.62*N).
Models trained on Class B — test generalization to Class A.
"""

import sys
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v3')
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/nn_mac')

import os, math, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.design import make_path
from polar.channels import BEMAC
from polar.eval import MACEval
from ncg_pure_neural import PureNeuralCompGraphDecoder


DESIGNS_DIR = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs'
MODELS_DIR = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/nn_mac/saved_models'
RESULTS_DIR = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results/bemac/bemac_classA_Ru62_Rv62_nn_vs_sc'


def load_design(N):
    """Load BEMAC Class A design for given N."""
    n = int(math.log2(N))
    path = os.path.join(DESIGNS_DIR, f'bemac_A_n{n}.npz')
    assert os.path.exists(path), f"Design file not found: {path}"
    d = np.load(path)
    eu = d['u_error_rates']
    ev = d['v_error_rates']
    su = np.argsort(eu)
    sv = np.argsort(ev)

    ku = round(0.62 * N)
    kv = round(0.62 * N)
    path_i = int(d['path_i'])  # 3N/8 for Class A

    Au = sorted([int(i + 1) for i in su[:ku]])
    Av = sorted([int(i + 1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    frozen_u = {p: 0 for p in sorted(all_pos - set(Au))}
    frozen_v = {p: 0 for p in sorted(all_pos - set(Av))}

    return ku, kv, Au, Av, frozen_u, frozen_v, path_i


def eval_nn(N, b, Au, Av, frozen_u, frozen_v, n_cw, batch_size):
    """Evaluate pure neural decoder BLER."""
    model_path = os.path.join(MODELS_DIR, f'ncg_pure_neural_N{N}.pt')
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    ku = len(Au)
    kv = len(Av)
    rng = np.random.default_rng(42)
    errs = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        while total < n_cw:
            bs = min(batch_size, n_cw - total)
            uf = np.zeros((bs, N), dtype=int)
            vf = np.zeros((bs, N), dtype=int)
            for p in Au:
                uf[:, p - 1] = rng.integers(0, 2, bs)
            for p in Av:
                vf[:, p - 1] = rng.integers(0, 2, bs)
            x = polar_encode_batch(uf)
            y = polar_encode_batch(vf)
            z = torch.from_numpy(x + y).long()

            _, _, uh, vh, _ = model(z, b, frozen_u, frozen_v)

            for i in range(bs):
                e = any(int(uh[p][i].item()) != uf[i, p - 1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p - 1] for p in Av if p in vh)
                if e:
                    errs += 1
            total += bs
            if total % 500 == 0:
                print(f"    NN progress: {total}/{n_cw}, errs={errs}", flush=True)

    elapsed = time.time() - t0
    bler = errs / total
    print(f"  NN:  {errs}/{total} errors, BLER={bler:.6f}, {elapsed:.1f}s", flush=True)
    return bler, errs, total


def eval_sc(N, b, Au, Av, frozen_u, frozen_v, n_cw, batch_size):
    """Evaluate analytical SC decoder BLER."""
    sc = MACEval(BEMAC(), backend='interleaved')
    t0 = time.time()
    _, _, bler = sc.run(N, b, Au, Av, frozen_u, frozen_v,
                        n_codewords=n_cw, batch_size=batch_size, verbose=False)
    elapsed = time.time() - t0
    print(f"  SC:  BLER={bler:.6f}, {elapsed:.1f}s ({n_cw} codewords)", flush=True)
    return bler


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Configuration: N values and codeword counts
    # Class A path_i=0: all V first, then all U
    Ns =       [16,   32,   64,   128,  256,  512,  1024]
    nn_cws =   [5000, 5000, 5000, 5000, 5000, 5000, 5000]
    sc_cws =   [5000, 5000, 5000, 5000, 2000, 1000, 500]
    nn_batch = [25,   25,   25,   25,   25,   25,   25]
    sc_batch = [100,  100,  100,  100,  25,   10,   5]

    results = {}

    for N, nn_cw, sc_cw, nn_bs, sc_bs in zip(Ns, nn_cws, sc_cws, nn_batch, sc_batch):
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N}", flush=True)
        print(f"{'=' * 60}", flush=True)

        ku, kv, Au, Av, frozen_u, frozen_v, path_i = load_design(N)
        b = make_path(N, path_i)  # Class A: path_i = 3N/8

        print(f"  ku={ku}, kv={kv}, path_i={path_i}, nn_cw={nn_cw}, sc_cw={sc_cw}", flush=True)

        # Check if model exists
        model_path = os.path.join(MODELS_DIR, f'ncg_pure_neural_N{N}.pt')
        if not os.path.exists(model_path):
            print(f"  SKIPPING N={N}: model not found at {model_path}", flush=True)
            continue

        nn_bler, nn_errs, nn_total = eval_nn(N, b, Au, Av, frozen_u, frozen_v, nn_cw, nn_bs)
        sc_bler = eval_sc(N, b, Au, Av, frozen_u, frozen_v, sc_cw, sc_bs)

        if sc_bler > 0:
            ratio = nn_bler / sc_bler
        elif nn_bler == 0:
            ratio = 0.0
        else:
            ratio = -1.0  # NN has errors, SC has zero: use -1 as sentinel
        print(f"  Ratio (NN/SC) = {ratio:.3f}", flush=True)

        results[N] = {
            'N': N,
            'nn_bler': nn_bler,
            'nn_errs': int(nn_errs),
            'nn_total': int(nn_total),
            'sc_bler': sc_bler,
            'sc_total': sc_cw,
            'ratio': round(ratio, 3),
            'ku': ku, 'kv': kv, 'path_i': path_i
        }

        # Save intermediate results
        json_results = {str(k): v for k, v in results.items()}
        with open(os.path.join(RESULTS_DIR, 'bemac_nn_vs_sc_classA.json'), 'w') as f:
            json.dump(json_results, f, indent=2)

    # Print summary
    print(f"\n\n{'=' * 70}", flush=True)
    print(f"SUMMARY: Pure Neural vs Analytical SC — BEMAC Class A (Ru=Rv=0.62)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"{'N':>6} {'ku':>4} {'kv':>4} {'NN BLER':>12} {'SC BLER':>12} {'Ratio':>8}", flush=True)
    print(f"{'-' * 50}", flush=True)
    for N in sorted(results.keys()):
        r = results[N]
        nn_s = f"{r['nn_bler']:.6f}" if r['nn_bler'] > 0 else "<2e-4"
        sc_s = f"{r['sc_bler']:.6f}" if r['sc_bler'] > 0 else "<2e-4"
        ratio_s = f"{r['ratio']:.3f}" if r['ratio'] != float('inf') else "N/A"
        print(f"{N:>6} {r['ku']:>4} {r['kv']:>4} {nn_s:>12} {sc_s:>12} {ratio_s:>8}", flush=True)

    # Save final JSON
    json_results = {str(k): v for k, v in results.items()}
    with open(os.path.join(RESULTS_DIR, 'bemac_nn_vs_sc_classA.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/bemac_nn_vs_sc_classA.json", flush=True)

    # Plot
    all_Ns = sorted(results.keys())
    plot_nn, plot_sc = [], []
    plot_Ns_nn, plot_Ns_sc = [], []

    for N in all_Ns:
        r = results[N]
        if r['sc_bler'] > 0:
            plot_Ns_sc.append(N)
            plot_sc.append(r['sc_bler'])
        if r['nn_bler'] > 0:
            plot_Ns_nn.append(N)
            plot_nn.append(r['nn_bler'])

    if plot_Ns_nn or plot_Ns_sc:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        if plot_Ns_sc:
            ax.semilogy(plot_Ns_sc, plot_sc, 'b-o', markersize=8, linewidth=2,
                        label='Analytical SC Decoder', zorder=5)
        if plot_Ns_nn:
            ax.semilogy(plot_Ns_nn, plot_nn, 'r--s', markersize=8, linewidth=2,
                        label='Pure Neural NCG Decoder', zorder=5)

        ax.set_xscale('log', base=2)
        ax.set_xticks(all_Ns)
        ax.set_xticklabels([str(n) for n in all_Ns])
        ax.set_xlabel('Block Length N', fontsize=13)
        ax.set_ylabel('Block Error Rate (BLER)', fontsize=13)
        ax.set_title(
            'BEMAC Class A: Pure Neural Decoder vs Analytical SC\n'
            r'$R_u = R_v \approx 0.62,\;$ Path $b = U^{3N/8}\, V^N\, U^{5N/8}$',
            fontsize=12
        )
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, which='both', alpha=0.3)
        ax.tick_params(labelsize=11)
        plt.tight_layout()

        fig.savefig(os.path.join(RESULTS_DIR, 'nn_vs_sc_classA_Ru62_Rv62.png'), dpi=200)
        fig.savefig(os.path.join(RESULTS_DIR, 'nn_vs_sc_classA_Ru62_Rv62.pdf'))
        print(f"Plot saved.", flush=True)
        plt.close()


if __name__ == '__main__':
    main()
