#!/usr/bin/env python3
"""
n256_perbit_analysis.py

For the N=256 paired validation set, recompute per-info-bit error rates
under both NN and SC. If the neural decoder's excess errors are
concentrated on specific bit positions (in the polar tree), that
points to a structural weakness at particular tree depths. If they are
uniform, the NN is just noisier everywhere.

This script regenerates the same test set as validate_gmac_fresh.py and
analyze_n256_failures.py (seed=42, N=256, 10000 cw). For each info
position p in Au (and p in Av) it counts how many codewords had a bit
error at that position for the NN and for SC.

Output
------
results/n256_perbit_counts.json  — per-position error counts
docs/n256_perbit.md              — short markdown summary
"""
import os
import sys
import json
import time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder_interleaved import decode_single
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 256
n_log = 8
ku = kv = 123
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
SEED = 42
N_CW = 10000
CKPT = 'saved_models/ncg_gmac_mlp_N256.pt'


def load_nn():
    sd = torch.load(CKPT, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)
    model_sd = model.state_dict()
    for k, v in sd.items():
        nk = k
        if nk.startswith('tree.'):
            nk = nk[5:]
        elif nk.startswith('z_enc.'):
            nk = 'z_encoder.' + nk[6:]
        if 'embedding_z' in nk:
            continue
        if nk in model_sd and model_sd[nk].shape == v.shape:
            model_sd[nk] = v
    model.load_state_dict(model_sd)
    model.eval()
    return model


def main():
    os.chdir(ROOT)

    Au, Av, frozen_u, frozen_v, _, _, path_i = design_from_file(
        f'designs/gmac_B_n{n_log}_snr6dB.npz', n_log, ku=ku, kv=kv
    )
    b = make_path(N, path_i)
    channel = GaussianMAC(sigma2=SIGMA2)

    rng = np.random.default_rng(SEED)
    U_info = rng.integers(0, 2, size=(N_CW, ku), dtype=np.int32)
    V_info = rng.integers(0, 2, size=(N_CW, kv), dtype=np.int32)
    U_msg = build_message_batch(N, U_info, Au)
    V_msg = build_message_batch(N, V_info, Av)
    X = polar_encode_batch(U_msg)
    Y = polar_encode_batch(V_msg)
    np.random.seed(SEED + 7919)
    Z = channel.sample_batch(X, Y).astype(np.float64)

    # Per-position error counters (0-indexed over positions 1..N; we only
    # care about info positions but we keep arrays of size N for ease)
    nn_u_err = np.zeros(N, dtype=np.int64)
    nn_v_err = np.zeros(N, dtype=np.int64)
    sc_u_err = np.zeros(N, dtype=np.int64)
    sc_v_err = np.zeros(N, dtype=np.int64)

    print(f"Running NN on {N_CW} codewords...", flush=True)
    model = load_nn()
    BS = 64
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N_CW, BS):
            end = min(start + BS, N_CW)
            z_t = torch.from_numpy(Z[start:end]).float()
            _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u, frozen_v)
            bs = end - start
            u_dec = np.zeros((bs, N), dtype=np.int32)
            v_dec = np.zeros((bs, N), dtype=np.int32)
            for pos, val in u_hat.items():
                u_dec[:, pos - 1] = val.round().int().cpu().numpy()
            for pos, val in v_hat.items():
                v_dec[:, pos - 1] = val.round().int().cpu().numpy()
            nn_u_err += (u_dec != U_msg[start:end]).sum(axis=0)
            nn_v_err += (v_dec != V_msg[start:end]).sum(axis=0)
    print(f"  NN done in {time.time()-t0:.1f}s", flush=True)

    print(f"Running SC on {N_CW} codewords...", flush=True)
    t0 = time.time()
    for i in range(N_CW):
        z_list = Z[i].tolist()
        u_dec, v_dec = decode_single(
            N, z_list, b, frozen_u, frozen_v, channel, log_domain=False
        )
        u_dec = np.asarray(u_dec, dtype=np.int32)
        v_dec = np.asarray(v_dec, dtype=np.int32)
        sc_u_err[u_dec != U_msg[i]] += 1
        sc_v_err[v_dec != V_msg[i]] += 1
        if (i + 1) % 1000 == 0:
            print(f"  SC {i+1}/{N_CW}", flush=True)
    print(f"  SC done in {time.time()-t0:.1f}s", flush=True)

    # Extract per-info-position rates (we don't care about frozen positions —
    # those are always 0 and both decoders know)
    Au_sorted = sorted(Au)
    Av_sorted = sorted(Av)

    report = {
        'config': {
            'N': N, 'ku': ku, 'kv': kv, 'n_cw': N_CW,
            'seed': SEED, 'SNR_dB': SNR_DB, 'checkpoint': CKPT,
        },
        'u_info_positions': Au_sorted,
        'v_info_positions': Av_sorted,
        'nn_u_per_position': [int(nn_u_err[p - 1]) for p in Au_sorted],
        'nn_v_per_position': [int(nn_v_err[p - 1]) for p in Av_sorted],
        'sc_u_per_position': [int(sc_u_err[p - 1]) for p in Au_sorted],
        'sc_v_per_position': [int(sc_v_err[p - 1]) for p in Av_sorted],
    }

    os.makedirs('results', exist_ok=True)
    with open('results/n256_perbit_counts.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\nSaved results/n256_perbit_counts.json")

    # ─── analysis ─────────────────────────────────────────────────────────
    nn_u_counts = np.array(report['nn_u_per_position'])
    nn_v_counts = np.array(report['nn_v_per_position'])
    sc_u_counts = np.array(report['sc_u_per_position'])
    sc_v_counts = np.array(report['sc_v_per_position'])

    nn_total = nn_u_counts.sum() + nn_v_counts.sum()
    sc_total = sc_u_counts.sum() + sc_v_counts.sum()
    print(f"\nTotal bit errors:  NN={nn_total}  SC={sc_total}  "
          f"ratio={nn_total/max(1,sc_total):.2f}x")
    print(f"Max per-position NN errs (U): {int(nn_u_counts.max())} "
          f"at info-pos {Au_sorted[int(nn_u_counts.argmax())]}")
    print(f"Max per-position SC errs (U): {int(sc_u_counts.max())} "
          f"at info-pos {Au_sorted[int(sc_u_counts.argmax())]}")

    # Top-10 positions by NN error count
    top_idx = np.argsort(nn_u_counts)[::-1][:10]
    print("\nTop-10 NN U-positions by error count:")
    print(f"  {'pos':<6}{'NN':<8}{'SC':<8}{'NN/SC':<8}")
    for idx in top_idx:
        p = Au_sorted[idx]
        nn_c = nn_u_counts[idx]
        sc_c = sc_u_counts[idx]
        r = nn_c / max(1, sc_c)
        print(f"  {p:<6}{nn_c:<8}{sc_c:<8}{r:<8.2f}")

    # Concentration: what fraction of NN errors comes from the top-10 positions?
    sorted_nn_u = np.sort(nn_u_counts)[::-1]
    frac_top10_u = sorted_nn_u[:10].sum() / max(1, nn_u_counts.sum())
    sorted_nn_v = np.sort(nn_v_counts)[::-1]
    frac_top10_v = sorted_nn_v[:10].sum() / max(1, nn_v_counts.sum())
    print(f"\nNN U-bit errors concentration: top-10 positions = "
          f"{frac_top10_u*100:.1f}% of all NN U bit-errors")
    print(f"NN V-bit errors concentration: top-10 positions = "
          f"{frac_top10_v*100:.1f}% of all NN V bit-errors")

    sorted_sc_u = np.sort(sc_u_counts)[::-1]
    frac_top10_u_sc = sorted_sc_u[:10].sum() / max(1, sc_u_counts.sum())
    print(f"SC U-bit errors concentration: top-10 positions = "
          f"{frac_top10_u_sc*100:.1f}% of all SC U bit-errors")


if __name__ == '__main__':
    main()
