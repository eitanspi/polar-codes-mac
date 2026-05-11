#!/usr/bin/env python3
"""
Correct timing benchmark for BEMAC Class C (Ru~0.30, Rv~0.60).
Uses the actual model that produced the published Class C results:
  to_git/models/nn_decoder_C/nn_decoder_weights.pt
  with NeuralMACSCDecoder(embed_dim=16, hidden_dim=128, n_layers=2, vocab_size=3)

For each N, runs a small batch (100 cw) of NN and SC, measures wall time per cw.
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


def get_design(N):
    n = int(math.log2(N))
    ku = max(1, min(round(RHO * RU_DIR * N), N - 1))
    kv = max(1, min(round(RHO * RV_DIR * N), N - 1))
    path_i = round(PATH_I_FRAC * N)
    Au, Av, fu, fv, _, _ = design_bemac(n, ku, kv)
    b = make_path(N, path_i)
    return ku, kv, Au, Av, fu, fv, b, path_i


def bench_nn_classC(model, N, b, fu, fv, channel, n_cw, batch_size):
    t0 = time.perf_counter()
    bler, _, _ = evaluate_nn_decoder(model, N, b, fu, fv, channel,
                                      n_codewords=n_cw, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    return bler, elapsed / n_cw


def bench_sc_classC(N, b, Au, Av, fu, fv, channel, n_cw, batch_size):
    rng = np.random.default_rng(42)
    ku, kv = len(Au), len(Av)
    errs = 0
    total = 0
    t_total = 0.0
    while total < n_cw:
        bs = min(batch_size, n_cw - total)
        info_u = rng.integers(0, 2, size=(bs, ku))
        info_v = rng.integers(0, 2, size=(bs, kv))
        U = build_message_batch(N, info_u, Au)
        V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U)
        Y = polar_encode_batch(V)
        Z = (X + Y).astype(np.int32)

        t0 = time.perf_counter()
        results = decode_batch(N, Z.tolist(), b, fu, fv, channel, vectorized=True)
        t_total += time.perf_counter() - t0

        for i, (ud, vd) in enumerate(results):
            u_err = any(ud[p - 1] != info_u[i, j] for j, p in enumerate(Au))
            v_err = any(vd[p - 1] != info_v[i, j] for j, p in enumerate(Av))
            if u_err or v_err:
                errs += 1
        total += bs
    return errs / total, t_total / total


def main():
    channel = BEMAC()
    model = load_nn_model()
    print(f"Model loaded from {MODEL_DIR}", flush=True)

    # N → (n_cw, batch_size_nn, batch_size_sc)
    plan = [
        (8,    200, 100, 100),
        (16,   200, 100, 100),
        (32,   200, 100, 100),
        (64,   200, 100, 100),
        (128,  200, 50,  100),
        (256,  100, 25,  50),
        (512,  50,  10,  25),
        (1024, 30,  5,   10),
    ]

    out = {}
    print(f"\n{'N':>6} {'NN ms/cw':>12} {'SC ms/cw':>12} {'NN/SC':>10} {'NN BLER':>12} {'SC BLER':>12}", flush=True)
    print("-" * 72, flush=True)
    for N, n_cw, bs_nn, bs_sc in plan:
        try:
            ku, kv, Au, Av, fu, fv, b, path_i = get_design(N)
        except Exception as e:
            print(f"  N={N}: design error: {e}", flush=True)
            continue

        try:
            nn_bler, nn_t = bench_nn_classC(model, N, b, fu, fv, channel, n_cw, bs_nn)
        except Exception as e:
            print(f"  N={N}: NN error: {e}", flush=True)
            nn_bler, nn_t = None, None

        try:
            sc_bler, sc_t = bench_sc_classC(N, b, Au, Av, fu, fv, channel, n_cw, bs_sc)
        except Exception as e:
            print(f"  N={N}: SC error: {e}", flush=True)
            sc_bler, sc_t = None, None

        nn_ms = nn_t * 1000 if nn_t else None
        sc_ms = sc_t * 1000 if sc_t else None
        ratio = nn_ms / sc_ms if (nn_ms and sc_ms) else None
        nn_bler_str = f"{nn_bler:.4f}" if nn_bler is not None else "?"
        sc_bler_str = f"{sc_bler:.4f}" if sc_bler is not None else "?"
        nn_ms_str = f"{nn_ms:.3f}" if nn_ms else "?"
        sc_ms_str = f"{sc_ms:.3f}" if sc_ms else "?"
        ratio_str = f"{ratio:.2f}x" if ratio else "?"
        print(f"  {N:>4} {nn_ms_str:>12} {sc_ms_str:>12} {ratio_str:>10} {nn_bler_str:>12} {sc_bler_str:>12}", flush=True)

        out[str(N)] = {
            'N': N, 'ku': ku, 'kv': kv, 'path_i': path_i,
            'nn_bler_smoke': nn_bler,
            'sc_bler_smoke': sc_bler,
            'nn_ms_per_cw': nn_ms,
            'sc_ms_per_cw': sc_ms,
            'nn_over_sc_time': ratio,
            'n_cw_used': n_cw,
        }

    out_path = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results/bemac/bemac_classC_timing.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == '__main__':
    main()
