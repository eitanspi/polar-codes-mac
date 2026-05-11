#!/usr/bin/env python3
"""
analyze_n256_failures.py

On the N=256 paired validation set, identify the codewords where the
neural decoder fails but the analytical SC decoder succeeds (the "NN only"
failures). For each such codeword, compute features of the channel
realization:

    - noise norm            ||W||^2 / N
    - BPSK-distance         min over the 3 valid sum symbols {-2, 0, +2}
                            of ||Z - s||^2 / N (how far the channel
                            output sits from the nearest valid centre)
    - ML-margin             difference between the ML sum symbol and the
                            second-best, averaged over the block

and compare those distributions to:

    (a) codewords both decoders got right (the "easy" baseline)
    (b) codewords SC also got wrong (the "genuinely hard" channels)

Goal: determine whether the 5x gap at N=256 is explained by the NN
choking on high-noise channel realizations, or whether it fails on
codewords that look ordinary by noise metrics — which would indicate
a systematic / structural issue with the trained tree ops at this depth.

Outputs
-------
results/n256_failure_analysis.json   — quantitative summary
docs/n256_failure_analysis.md        — short markdown write-up
"""
import os
import sys
import json
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

# ─── config — match validate_gmac_fresh.py exactly ──────────────────────────

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

    # ─── generate the same test set as validate_gmac_fresh.py ──────────────
    rng = np.random.default_rng(SEED)
    U_info = rng.integers(0, 2, size=(N_CW, ku), dtype=np.int32)
    V_info = rng.integers(0, 2, size=(N_CW, kv), dtype=np.int32)
    U_msg = build_message_batch(N, U_info, Au)
    V_msg = build_message_batch(N, V_info, Av)
    X = polar_encode_batch(U_msg)
    Y = polar_encode_batch(V_msg)

    np.random.seed(SEED + 7919)
    Z = channel.sample_batch(X, Y).astype(np.float64)

    # ─── noiseless sum signal S = (1-2X) + (1-2Y), the channel mean ─────────
    S = (1 - 2 * X).astype(np.float64) + (1 - 2 * Y).astype(np.float64)
    W = Z - S  # the realized noise (Gaussian, var SIGMA2)

    # per-codeword noise energy and a normalised noise norm per symbol
    noise_energy = (W ** 2).sum(axis=1)  # (N_CW,)
    noise_per_symbol = noise_energy / N  # ~ chi^2/N, mean = SIGMA2

    # per-codeword BPSK-distance: how far the channel output sits from the
    # nearest valid centre in {-2, 0, +2} (on each symbol, averaged)
    centres = np.array([-2.0, 0.0, 2.0])
    dists = np.abs(Z[..., None] - centres)  # (N_CW, N, 3)
    nearest = dists.min(axis=-1)  # (N_CW, N)
    bpsk_dist_per_symbol = (nearest ** 2).mean(axis=1)

    # ─── run NN decoder on all 10k codewords (batched) ─────────────────────
    print(f"Running NN on {N_CW} codewords...", flush=True)
    model = load_nn()
    BS = 64
    nn_err = np.zeros(N_CW, dtype=np.int32)
    import time
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
            for i in range(bs):
                u_bad = any(u_dec[i, p - 1] != U_msg[start + i, p - 1] for p in Au)
                v_bad = any(v_dec[i, p - 1] != V_msg[start + i, p - 1] for p in Av)
                if u_bad or v_bad:
                    nn_err[start + i] = 1
    print(f"  NN done in {time.time()-t0:.1f}s  "
          f"BLER={nn_err.mean():.4f}", flush=True)

    # ─── run SC decoder (sequential, one at a time) ────────────────────────
    print(f"Running SC on {N_CW} codewords...", flush=True)
    sc_err = np.zeros(N_CW, dtype=np.int32)
    t0 = time.time()
    for i in range(N_CW):
        z_list = Z[i].tolist()
        u_dec, v_dec = decode_single(
            N, z_list, b, frozen_u, frozen_v, channel, log_domain=False
        )
        u_dec = np.asarray(u_dec, dtype=np.int32)
        v_dec = np.asarray(v_dec, dtype=np.int32)
        u_bad = any(u_dec[p - 1] != U_msg[i, p - 1] for p in Au)
        v_bad = any(v_dec[p - 1] != V_msg[i, p - 1] for p in Av)
        if u_bad or v_bad:
            sc_err[i] = 1
        if (i + 1) % 500 == 0:
            print(f"  SC {i+1}/{N_CW}  BLER={sc_err[:i+1].mean():.4f}", flush=True)
    print(f"  SC done in {time.time()-t0:.1f}s  "
          f"BLER={sc_err.mean():.4f}", flush=True)

    # ─── partition codewords by outcome ────────────────────────────────────
    both_ok = (nn_err == 0) & (sc_err == 0)
    both_err = (nn_err == 1) & (sc_err == 1)
    nn_only = (nn_err == 1) & (sc_err == 0)  # the interesting group
    sc_only = (nn_err == 0) & (sc_err == 1)

    groups = {
        'both_ok': both_ok,
        'both_err': both_err,
        'nn_only': nn_only,   # NN fails, SC succeeds
        'sc_only': sc_only,   # SC fails, NN succeeds
    }

    def summary(x):
        return {
            'n': int(x.size),
            'mean': float(np.mean(x)),
            'median': float(np.median(x)),
            'std': float(np.std(x)),
            'p10': float(np.percentile(x, 10)),
            'p90': float(np.percentile(x, 90)),
            'min': float(np.min(x)),
            'max': float(np.max(x)),
        } if len(x) > 0 else {'n': 0}

    report = {
        'config': {
            'N': N, 'ku': ku, 'kv': kv, 'SNR_dB': SNR_DB,
            'sigma2': SIGMA2, 'seed': SEED, 'n_cw': N_CW,
            'checkpoint': CKPT,
        },
        'overall': {
            'nn_bler': float(nn_err.mean()),
            'sc_bler': float(sc_err.mean()),
            'both_ok_count': int(both_ok.sum()),
            'both_err_count': int(both_err.sum()),
            'nn_only_count': int(nn_only.sum()),
            'sc_only_count': int(sc_only.sum()),
            'expected_noise_per_symbol': SIGMA2,
        },
        'noise_per_symbol': {
            name: summary(noise_per_symbol[mask])
            for name, mask in groups.items()
        },
        'bpsk_dist_per_symbol': {
            name: summary(bpsk_dist_per_symbol[mask])
            for name, mask in groups.items()
        },
    }

    os.makedirs('results', exist_ok=True)
    with open('results/n256_failure_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\nSaved results/n256_failure_analysis.json", flush=True)

    # ─── pretty console output ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("N=256 failure-mode analysis")
    print("=" * 60)
    print(f"Total codewords: {N_CW}")
    print(f"Both OK:  {both_ok.sum()}  ({both_ok.mean():.4f})")
    print(f"Both err: {both_err.sum()}  ({both_err.mean():.4f})")
    print(f"NN only:  {nn_only.sum()}  ({nn_only.mean():.4f})  ← NN fails, SC ok")
    print(f"SC only:  {sc_only.sum()}  ({sc_only.mean():.4f})  ← SC fails, NN ok")
    print()
    print(f"Expected noise/symbol (σ²): {SIGMA2:.4f}")
    print()
    print("noise_per_symbol (||W||²/N)  mean  median  std      p10    p90")
    for name, mask in groups.items():
        x = noise_per_symbol[mask]
        if len(x) > 0:
            print(f"  {name:10s}  n={len(x):5d}  "
                  f"{x.mean():.4f}  {np.median(x):.4f}  "
                  f"{x.std():.4f}  {np.percentile(x,10):.4f}  "
                  f"{np.percentile(x,90):.4f}")
    print()
    print("bpsk_dist_per_symbol (nearest centre)  mean  median  std")
    for name, mask in groups.items():
        x = bpsk_dist_per_symbol[mask]
        if len(x) > 0:
            print(f"  {name:10s}  n={len(x):5d}  "
                  f"{x.mean():.4f}  {np.median(x):.4f}  "
                  f"{x.std():.4f}")

    # ─── t-like comparison between groups ─────────────────────────────────
    # Compare nn_only vs both_ok on the two metrics
    if nn_only.sum() > 0 and both_ok.sum() > 0:
        a = noise_per_symbol[nn_only]
        b_ = noise_per_symbol[both_ok]
        z = (a.mean() - b_.mean()) / (a.std() / np.sqrt(len(a)) + 1e-12)
        print(f"\nnoise_per_symbol, nn_only vs both_ok:")
        print(f"  delta mean = {a.mean() - b_.mean():+.4f}  "
              f"(nn_only is {'harder' if a.mean() > b_.mean() else 'easier'})")
        print(f"  z-like     = {z:.2f}")

    if nn_only.sum() > 0 and both_err.sum() > 0:
        a = noise_per_symbol[nn_only]
        b_ = noise_per_symbol[both_err]
        print(f"\nnoise_per_symbol, nn_only vs both_err:")
        print(f"  delta mean = {a.mean() - b_.mean():+.4f}  "
              f"(nn_only is {'harder' if a.mean() > b_.mean() else 'easier'} "
              f"than both_err)")

    # Save error-index arrays too — useful for downstream analysis
    np.savez(
        'results/n256_failure_indices.npz',
        nn_err=nn_err,
        sc_err=sc_err,
        noise_per_symbol=noise_per_symbol,
        bpsk_dist_per_symbol=bpsk_dist_per_symbol,
    )
    print("\nSaved results/n256_failure_indices.npz")


if __name__ == '__main__':
    main()
