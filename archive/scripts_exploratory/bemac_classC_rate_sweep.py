#!/usr/bin/env python3
"""
BEMAC Class C rate-BLER sweep at fixed N.

For a fixed block length N, sweep the rate parameter rho ∈ [0.4 .. 0.9]
and run both NN-SC and analytical SC. Each rho corresponds to:
   ku = round(rho * 0.5 * N), kv = round(rho * 1.0 * N)
which is the (Ru, Rv) used by the Class C trained model.

Output: a rate-BLER curve. The interesting question is the crossover
point where NN-SC achieves BLER ≤ 1e-3 but analytical SC does not.

Uses the OLD nn_decoder_C model from to_git/.
"""

import sys
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git')

import os, math, json, time
import numpy as np
import torch

from polar.channels import BEMAC
from polar.design import design_bemac, make_path
from polar.encoder import polar_encode_batch, build_message_batch
from polar.nn_decoder import NeuralMACSCDecoder, evaluate_nn_decoder
from polar.decoder import decode_batch


MODEL_DIR = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git/models/nn_decoder_C'
PATH_I_FRAC = 1.0  # Class C
RU_DIR = 0.5
RV_DIR = 1.0


def load_nn_model():
    model = NeuralMACSCDecoder(embed_dim=16, hidden_dim=128, n_layers=2, vocab_size=3)
    fpath = os.path.join(MODEL_DIR, 'nn_decoder_weights.pt')
    state = torch.load(fpath, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def get_design(N, rho):
    n = int(math.log2(N))
    ku = max(1, min(round(rho * RU_DIR * N), N - 1))
    kv = max(1, min(round(rho * RV_DIR * N), N - 1))
    path_i = round(PATH_I_FRAC * N)
    Au, Av, fu, fv, _, _ = design_bemac(n, ku, kv)
    b = make_path(N, path_i)
    return ku, kv, Au, Av, fu, fv, b


def eval_nn(model, N, b, fu, fv, channel, n_cw, batch_size):
    bler, _, _ = evaluate_nn_decoder(model, N, b, fu, fv, channel,
                                      n_codewords=n_cw, batch_size=batch_size)
    return bler


def eval_sc(N, b, Au, Av, fu, fv, channel, n_cw, batch_size, seed=42):
    rng = np.random.default_rng(seed)
    ku, kv = len(Au), len(Av)
    errs, total = 0, 0
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
    return errs / total


def main():
    channel = BEMAC()
    model = load_nn_model()

    rhos = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    plan = [
        # (N, n_cw, batch_nn, batch_sc)
        (64,  5000, 100, 100),
        (512, 3000, 10,  25),
    ]

    # Load existing results to extend instead of overwriting
    out_path = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results/bemac/bemac_classC_rate_sweep.json'
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
    else:
        out = {}

    for N, n_cw, bs_nn, bs_sc in plan:
        print(f"\n{'='*72}\n  N = {N}, {n_cw} cw per point\n{'='*72}", flush=True)
        out[str(N)] = []
        for rho in rhos:
            ku, kv, Au, Av, fu, fv, b = get_design(N, rho)
            sum_rate = (ku + kv) / N
            print(f"\n  rho={rho:.2f}, ku={ku}, kv={kv}, R_sum={sum_rate:.3f}", flush=True)

            t0 = time.perf_counter()
            try:
                nn_bler = eval_nn(model, N, b, fu, fv, channel, n_cw, bs_nn)
            except Exception as e:
                print(f"    NN ERROR: {e}", flush=True)
                nn_bler = None
            t_nn = time.perf_counter() - t0

            t0 = time.perf_counter()
            try:
                sc_bler = eval_sc(N, b, Au, Av, fu, fv, channel, n_cw, bs_sc)
            except Exception as e:
                print(f"    SC ERROR: {e}", flush=True)
                sc_bler = None
            t_sc = time.perf_counter() - t0

            ratio = nn_bler / sc_bler if (nn_bler and sc_bler and sc_bler > 0) else None
            print(f"    NN: {nn_bler:.4e}  SC: {sc_bler:.4e}  ratio: {ratio:.3f}  [{t_nn:.0f}s NN, {t_sc:.0f}s SC]"
                  if ratio is not None else f"    NN: {nn_bler}  SC: {sc_bler}",
                  flush=True)

            out[str(N)].append({
                'N': N, 'rho': rho, 'ku': ku, 'kv': kv,
                'Ru': ku/N, 'Rv': kv/N, 'sum_rate': sum_rate,
                'nn_bler': nn_bler, 'sc_bler': sc_bler,
                'ratio': ratio,
                'n_cw': n_cw,
                'time_nn_sec': t_nn, 'time_sc_sec': t_sc,
            })

            # Save after each point
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(out, f, indent=2)

    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
