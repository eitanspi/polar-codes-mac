"""
eval_bemac_classC_nn_50k.py
===========================
BEMAC Class C (Ru30/Rv60) — Extended CW evaluation at N=256,512,1024.
Uses nn_decoder_C model (NeuralMACSCDecoder) for NN.
Runs SC with 50K CW for better confidence. NN with feasible counts.

Class C: path_i=N, rho=0.6, Ru_dir=0.5, Rv_dir=1.0.
"""

import sys
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git')

import os, math, json, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polar.channels import BEMAC
from polar.design import design_bemac, make_path
from polar.encoder import polar_encode_batch, build_message_batch
from polar.nn_decoder import NeuralMACSCDecoder, evaluate_nn_decoder
from polar.decoder import decode_batch


MODEL_DIR = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git/models/nn_decoder_C'
RESULTS_DIR = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results/bemac/bemac_classC_Ru30_Rv60_nn_vs_sc'

# Class C config
RHO = 0.6
RU_DIR = 0.5
RV_DIR = 1.0
PATH_I_FRAC = 1.0


def load_nn_model():
    model = NeuralMACSCDecoder(embed_dim=16, hidden_dim=128, n_layers=2, vocab_size=3)
    fpath = os.path.join(MODEL_DIR, 'nn_decoder_weights.pt')
    state = torch.load(fpath, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_design(N, rho=RHO):
    n = int(math.log2(N))
    ku = max(1, min(round(rho * RU_DIR * N), N - 1))
    kv = max(1, min(round(rho * RV_DIR * N), N - 1))
    path_i = round(PATH_I_FRAC * N)
    Au, Av, fu, fv, z_u, z_v = design_bemac(n, ku, kv)
    b = make_path(N, path_i)
    return ku, kv, Au, Av, fu, fv, b, path_i


def eval_nn(model, N, b, fu, fv, channel, n_cw, batch_size):
    t0 = time.time()
    bler, ber_u, ber_v = evaluate_nn_decoder(
        model, N, b, fu, fv, channel,
        n_codewords=n_cw, batch_size=batch_size)
    elapsed = time.time() - t0
    print(f"  NN:  BLER={bler:.6f}, {elapsed:.1f}s ({n_cw} codewords)", flush=True)
    return bler


def eval_sc(N, b, Au, Av, fu, fv, channel, n_cw, batch_size):
    rng = np.random.default_rng(42)
    ku, kv = len(Au), len(Av)
    errs = 0
    total = 0
    t0 = time.time()

    while total < n_cw:
        bs = min(batch_size, n_cw - total)
        info_u = rng.integers(0, 2, size=(bs, ku))
        info_v = rng.integers(0, 2, size=(bs, kv))
        U = build_message_batch(N, info_u, Au)
        V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U)
        Y = polar_encode_batch(V)
        Z = (X + Y).astype(np.int32)

        results = decode_batch(N, Z.tolist(), b, fu, fv, channel, vectorized=True)
        for i, (ud, vd) in enumerate(results):
            u_err = any(ud[p - 1] != info_u[i, j] for j, p in enumerate(Au))
            v_err = any(vd[p - 1] != info_v[i, j] for j, p in enumerate(Av))
            if u_err or v_err:
                errs += 1

        total += bs
        if total % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    SC progress: {total}/{n_cw}, errs={errs}, {elapsed:.1f}s", flush=True)

    elapsed = time.time() - t0
    bler = errs / total
    print(f"  SC:  BLER={bler:.6f} ({errs}/{total}), {elapsed:.1f}s", flush=True)
    return bler


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    channel = BEMAC()
    model = load_nn_model()
    print(f"Model loaded from {MODEL_DIR}", flush=True)

    # N values with CW counts — balanced for speed vs confidence
    # NN is slow (~50s/1K at N=256, ~100s/1K at N=512, ~500s/1K at N=1024)
    # SC is fast (~2s/1K at N=256, ~8s/1K at N=512, ~17s/1K at N=1024)
    configs = [
        (256,  50000, 50000, 50, 100),   # NN: ~42min, SC: ~2min
        (512,  50000, 50000, 25, 50),    # NN: ~83min, SC: ~7min
        (1024, 30000, 50000, 10, 25),    # NN: ~4h, SC: ~14min
    ]

    # Existing results for smaller N
    existing = {
        8:    {'nn_bler': 0.03933, 'sc_bler': 0.10967, 'nn_cw': 3000, 'sc_cw': 3000},
        16:   {'nn_bler': 0.01633, 'sc_bler': 0.09200, 'nn_cw': 3000, 'sc_cw': 3000},
        32:   {'nn_bler': 0.00567, 'sc_bler': 0.09933, 'nn_cw': 3000, 'sc_cw': 3000},
        64:   {'nn_bler': 0.00167, 'sc_bler': 0.05500, 'nn_cw': 3000, 'sc_cw': 3000},
        128:  {'nn_bler': 0.00025, 'sc_bler': 0.02455, 'nn_cw': 20000, 'sc_cw': 20000},
    }

    results = {}

    for N, nn_cw, sc_cw, nn_bs, sc_bs in configs:
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N}", flush=True)
        print(f"{'=' * 60}", flush=True)

        ku, kv, Au, Av, fu, fv, b, path_i = get_design(N)
        print(f"  ku={ku}, kv={kv}, Ru={ku/N:.4f}, Rv={kv/N:.4f}, path_i={path_i}", flush=True)
        print(f"  NN: {nn_cw} CW, batch={nn_bs}", flush=True)
        print(f"  SC: {sc_cw} CW, batch={sc_bs}", flush=True)

        nn_bler = eval_nn(model, N, b, fu, fv, channel, nn_cw, nn_bs)
        sc_bler = eval_sc(N, b, Au, Av, fu, fv, channel, sc_cw, sc_bs)

        ratio = nn_bler / sc_bler if sc_bler > 0 else (0.0 if nn_bler == 0 else -1.0)
        print(f"  Ratio (NN/SC) = {ratio:.3f}", flush=True)

        results[N] = {
            'N': N, 'ku': ku, 'kv': kv,
            'nn_bler': nn_bler, 'sc_bler': sc_bler,
            'nn_cw': nn_cw, 'sc_cw': sc_cw,
            'ratio': round(ratio, 3),
            'Ru': ku / N, 'Rv': kv / N,
        }

        # Save intermediate
        with open(os.path.join(RESULTS_DIR, 'bemac_nn_vs_sc_classC_50k.json'), 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Merge with existing
    all_results = {}
    for N, d in existing.items():
        ku_n = max(1, min(round(RHO * RU_DIR * N), N - 1))
        kv_n = max(1, min(round(RHO * RV_DIR * N), N - 1))
        ratio = d['nn_bler'] / d['sc_bler'] if d['sc_bler'] > 0 else 0.0
        all_results[N] = {
            'N': N, 'ku': ku_n, 'kv': kv_n,
            'nn_bler': d['nn_bler'], 'sc_bler': d['sc_bler'],
            'nn_cw': d['nn_cw'], 'sc_cw': d['sc_cw'],
            'ratio': round(ratio, 2),
            'Ru': ku_n / N, 'Rv': kv_n / N,
        }
    all_results.update(results)

    # Summary
    print(f"\n\n{'=' * 80}", flush=True)
    print(f"SUMMARY: NN vs SC — BEMAC Class C (Ru~0.30, Rv~0.60)", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"{'N':>6} {'ku':>4} {'kv':>4} {'NN BLER':>12} {'SC BLER':>12} {'Ratio':>8} {'NN CW':>8} {'SC CW':>8}", flush=True)
    print(f"{'-' * 70}", flush=True)
    for N in sorted(all_results.keys()):
        r = all_results[N]
        print(f"{N:>6} {r['ku']:>4} {r['kv']:>4} {r['nn_bler']:>12.6f} {r['sc_bler']:>12.6f} {r['ratio']:>8.3f} {r['nn_cw']:>8} {r['sc_cw']:>8}", flush=True)

    # Save final JSON
    with open(os.path.join(RESULTS_DIR, 'bemac_nn_vs_sc_classC_50k.json'), 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Save XLSX
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['BE-MAC — Class C, (Ru=0.30, Rv=0.60), Neural SC vs SC (L=1) — Extended CW'])
    ws.append(['Capacity: I(Z;X)=0.5000 | I(Z;Y|X)=1.0000 | I(Z;X,Y)=1.5000 | Path: U^N V^N'])
    ws.append([])
    ws.append(['N', 'Ru', 'Rv', 'Sum Rate', 'NN BLER', 'SC BLER', 'NN/SC', 'NN CW', 'SC CW'])
    for N in sorted(all_results.keys()):
        r = all_results[N]
        ws.append([N, r['Ru'], r['Rv'], r['Ru'] + r['Rv'],
                   r['nn_bler'], r['sc_bler'], r['ratio'], r['nn_cw'], r['sc_cw']])
    ws.append([])
    ws.append(['Capacity', 0.5, 1.0, 1.5])
    wb.save(os.path.join(RESULTS_DIR, 'nn_vs_sc_classC_Ru30_Rv60.xlsx'))
    print(f"\nXLSX saved to {RESULTS_DIR}/nn_vs_sc_classC_Ru30_Rv60.xlsx", flush=True)

    # Plot
    plot_Ns = sorted(all_results.keys())
    nn_blers = [all_results[N]['nn_bler'] for N in plot_Ns]
    sc_blers = [all_results[N]['sc_bler'] for N in plot_Ns]

    nn_Ns = [N for N, b in zip(plot_Ns, nn_blers) if b > 0]
    nn_vals = [b for b in nn_blers if b > 0]
    sc_Ns = [N for N, b in zip(plot_Ns, sc_blers) if b > 0]
    sc_vals = [b for b in sc_blers if b > 0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    if sc_Ns:
        ax.semilogy(sc_Ns, sc_vals, 'b-o', markersize=8, linewidth=2,
                    label='Analytical SC Decoder', zorder=5)
    if nn_Ns:
        ax.semilogy(nn_Ns, nn_vals, 'r--s', markersize=8, linewidth=2,
                    label='Neural SC Decoder', zorder=5)

    ax.set_xscale('log', base=2)
    ax.set_xticks(plot_Ns)
    ax.set_xticklabels([str(n) for n in plot_Ns])
    ax.set_xlabel('Block Length N', fontsize=13)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=13)
    ax.set_title(
        'BEMAC Class C: Neural Decoder vs Analytical SC\n'
        r'$R_u \approx 0.30,\; R_v \approx 0.60,\;$ Path $b = V^N\, U^N$',
        fontsize=12
    )
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()

    fig.savefig(os.path.join(RESULTS_DIR, 'nn_vs_sc_classC_Ru30_Rv60.png'), dpi=200)
    fig.savefig(os.path.join(RESULTS_DIR, 'nn_vs_sc_classC_Ru30_Rv60.pdf'))
    print(f"Plot saved.", flush=True)
    plt.close()


if __name__ == '__main__':
    main()
