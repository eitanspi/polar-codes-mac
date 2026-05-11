#!/usr/bin/env python3
"""
validate_gmac_fresh.py

Fair, reproducible head-to-head: Neural SC decoder vs analytical SC decoder
on the GMAC (Class B symmetric, SNR=6 dB), at N=32, 64, 128, 256.

The NN and SC are evaluated on the *exact same* codewords (same info bits,
same channel noise) so BLER differences are meaningful.

Outputs
-------
results/validate_gmac_fresh.json   — per-N: NN BLER, SC BLER, n_cw, elapsed
docs/validate_gmac_fresh.md         — short markdown table

Run
---
    python scripts/validate_gmac_fresh.py
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder_interleaved import decode_single
from polar.decoder_scl import decode_batch_list
from neural.ncg_gmac import GmacNeuralCompGraphDecoder


# ───────────────────────────── checkpoint loading ─────────────────────────────

def load_ncg_gmac_mlp(ckpt_path, d=16, hidden=64, n_layers=2):
    """
    Load ncg_gmac_mlp_N*.pt checkpoints.

    These files have two key prefixes:
      - 'tree.<...>'     → maps to GmacNeuralCompGraphDecoder internal ops
      - 'z_encoder.<...>' or 'z_enc.<...>' → the channel z-encoder MLP

    N=32's checkpoint uses 'z_enc.*'; N>=64 use 'z_encoder.*'. We handle both.
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']

    model = GmacNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)
    model_sd = model.state_dict()

    loaded, skipped = 0, 0
    for k, v in sd.items():
        new_k = k
        if new_k.startswith('tree.'):
            new_k = new_k[len('tree.'):]
        elif new_k.startswith('z_enc.'):
            new_k = 'z_encoder.' + new_k[len('z_enc.'):]
        # 'z_encoder.*' is already correct

        if 'embedding_z' in new_k:
            skipped += 1
            continue
        if new_k in model_sd and model_sd[new_k].shape == v.shape:
            model_sd[new_k] = v
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_sd)
    model.eval()
    return model, loaded, skipped


# ───────────────────────────── test-set generation ────────────────────────────

def generate_test_set(N, Au, Av, channel, n_codewords, seed):
    """
    Generate a fixed, reproducible test set.

    Returns:
        U_msg (n_cw, N)  — U info-bit full message vectors (frozen = 0)
        V_msg (n_cw, N)  — V info-bit full message vectors
        X     (n_cw, N)  — polar-encoded U
        Y     (n_cw, N)  — polar-encoded V
        Z     (n_cw, N)  — GMAC channel output (float)
    """
    rng = np.random.default_rng(seed)
    ku, kv = len(Au), len(Av)
    U_info = rng.integers(0, 2, size=(n_codewords, ku), dtype=np.int32)
    V_info = rng.integers(0, 2, size=(n_codewords, kv), dtype=np.int32)
    U_msg = build_message_batch(N, U_info, Au)
    V_msg = build_message_batch(N, V_info, Av)
    X = polar_encode_batch(U_msg)
    Y = polar_encode_batch(V_msg)

    # GaussianMAC.sample_batch uses np.random — seed globally for reproducibility
    np.random.seed(seed + 7919)
    Z = channel.sample_batch(X, Y)
    return U_msg, V_msg, X, Y, Z


# ───────────────────────────── evaluators ─────────────────────────────────────

def evaluate_nn(model, Z, b, frozen_u, frozen_v, Au, Av, U_msg, V_msg,
                batch_size=64, verbose=False):
    """Return per-codeword error flags (1 if block error, else 0), and elapsed."""
    n_cw, N = Z.shape
    err = np.zeros(n_cw, dtype=np.int32)
    t0 = time.time()
    for start in range(0, n_cw, batch_size):
        end = min(start + batch_size, n_cw)
        z_batch = torch.from_numpy(Z[start:end]).float()
        with torch.no_grad():
            _, _, u_hat, v_hat, _ = model(z_batch, b, frozen_u, frozen_v)
        # u_hat / v_hat are dicts {1-indexed pos: (batch,) tensor}
        bs = end - start
        u_dec = np.zeros((bs, N), dtype=np.int32)
        v_dec = np.zeros((bs, N), dtype=np.int32)
        for pos, val in u_hat.items():
            u_dec[:, pos - 1] = val.round().int().cpu().numpy()
        for pos, val in v_hat.items():
            v_dec[:, pos - 1] = val.round().int().cpu().numpy()

        for i in range(bs):
            u_errs = any(
                u_dec[i, p - 1] != U_msg[start + i, p - 1] for p in Au
            )
            v_errs = any(
                v_dec[i, p - 1] != V_msg[start + i, p - 1] for p in Av
            )
            if u_errs or v_errs:
                err[start + i] = 1

        if verbose and (start // batch_size) % 5 == 0:
            partial = err[:end].sum() / end
            print(f"      NN {end}/{n_cw}  BLER={partial:.4f}", flush=True)
    return err, time.time() - t0


def evaluate_sc(Z, b, frozen_u, frozen_v, channel, Au, Av, U_msg, V_msg,
                verbose=False):
    """Analytical SC decoder (interleaved backend). One codeword at a time."""
    n_cw, N = Z.shape
    err = np.zeros(n_cw, dtype=np.int32)
    t0 = time.time()
    # log_domain=False is the default for GMAC
    for i in range(n_cw):
        z_list = Z[i].tolist()
        u_dec, v_dec = decode_single(
            N, z_list, b, frozen_u, frozen_v, channel, log_domain=False
        )
        u_dec = np.asarray(u_dec, dtype=np.int32)
        v_dec = np.asarray(v_dec, dtype=np.int32)
        u_errs = any(u_dec[p - 1] != U_msg[i, p - 1] for p in Au)
        v_errs = any(v_dec[p - 1] != V_msg[i, p - 1] for p in Av)
        if u_errs or v_errs:
            err[i] = 1
        if verbose and (i + 1) % 50 == 0:
            partial = err[:i + 1].sum() / (i + 1)
            print(f"      SC {i + 1}/{n_cw}  BLER={partial:.4f}", flush=True)
    return err, time.time() - t0


def evaluate_scl(Z, b, frozen_u, frozen_v, channel, Au, Av, U_msg, V_msg,
                 L=4, verbose=False):
    """
    Analytical SCL decoder (list size L). Uses the batched vectorised
    implementation from polar.decoder_scl.

    SCL for this MAC decoder requires log_domain=True.
    """
    n_cw, N = Z.shape
    err = np.zeros(n_cw, dtype=np.int32)
    t0 = time.time()

    # decode_batch_list takes a list of z-vectors (one per codeword)
    Z_list = [Z[i].tolist() for i in range(n_cw)]

    if verbose:
        print(f"      SCL(L={L}) running on {n_cw} codewords ...", flush=True)

    results = decode_batch_list(
        N, Z_list, b, frozen_u, frozen_v, channel,
        log_domain=True, L=L, vectorized=True,
    )

    for i, (u_dec, v_dec) in enumerate(results):
        u_dec = np.asarray(u_dec, dtype=np.int32)
        v_dec = np.asarray(v_dec, dtype=np.int32)
        u_errs = any(u_dec[p - 1] != U_msg[i, p - 1] for p in Au)
        v_errs = any(v_dec[p - 1] != V_msg[i, p - 1] for p in Av)
        if u_errs or v_errs:
            err[i] = 1

    if verbose:
        bler = err.sum() / n_cw
        print(f"      SCL(L={L}) DONE  BLER={bler:.4f}  "
              f"({err.sum()}/{n_cw}) in {time.time()-t0:.1f}s", flush=True)
    return err, time.time() - t0


# ───────────────────────────── main ───────────────────────────────────────────

CASES = [
    # (N, n=log2 N, ku, kv, ckpt, n_cw_target)
    (32,  5,  15,  15, 'saved_models/ncg_gmac_mlp_N32.pt',  5000),
    (64,  6,  31,  31, 'saved_models/ncg_gmac_mlp_N64.pt',  5000),
    (128, 7,  62,  62, 'saved_models/ncg_gmac_mlp_N128.pt', 5000),
    (256, 8, 123, 123, 'saved_models/ncg_gmac_mlp_N256.pt', 10000),
]

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Use 200 codewords per N (smoke test)')
    parser.add_argument('--only', type=int, default=None,
                        help='Only run a specific N (one of 32/64/128/256)')
    parser.add_argument('--n-cw', type=int, default=None,
                        help='Override codeword count for all selected Ns')
    parser.add_argument('--include-scl', action='store_true',
                        help='Also evaluate analytical SCL (L=4) on the '
                             'same paired codewords')
    parser.add_argument('--scl-L', type=int, default=4,
                        help='SCL list size (default 4)')
    args = parser.parse_args()

    os.chdir(ROOT)

    # Merge into existing JSON if present — lets `--only N` update one row
    # without destroying previously-computed rows for other Ns.
    out_json = 'results/validate_gmac_fresh.json'
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                results = json.load(f)
            if 'per_N' not in results:
                results['per_N'] = {}
        except (ValueError, OSError):
            results = {'per_N': {}}
    else:
        results = {'per_N': {}}

    results['meta'] = {
        'snr_dB': SNR_DB,
        'sigma2': SIGMA2,
        'seed': SEED,
        'class': 'B (symmetric, Ru≈Rv≈0.48)',
        'channel': 'GaussianMAC',
        'decoder_neural': 'GmacNeuralCompGraphDecoder d=16 hidden=64',
        'decoder_analytical': 'decoder_interleaved.decode_single (log_domain=False)',
    }

    for N, n, ku, kv, ckpt, n_cw in CASES:
        if args.only is not None and args.only != N:
            continue
        if args.quick:
            n_cw = 200
        if args.n_cw is not None:
            n_cw = args.n_cw

        print(f"\n=== N={N}  (ku={ku}, kv={kv})  n_cw={n_cw} ===", flush=True)

        design_file = f'designs/gmac_B_n{n}_snr6dB.npz'
        Au, Av, frozen_u, frozen_v, _, _, path_i = design_from_file(
            design_file, n, ku=ku, kv=kv
        )
        b = make_path(N, path_i)
        channel = GaussianMAC(sigma2=SIGMA2)
        print(f"  design: path_i={path_i}  |Au|={len(Au)}  |Av|={len(Av)}")

        print(f"  loading NN checkpoint: {ckpt}")
        model, loaded, skipped = load_ncg_gmac_mlp(ckpt, d=16, hidden=64)
        print(f"    loaded {loaded} params, skipped {skipped}")

        print(f"  generating test set (seed={SEED}) ...")
        U_msg, V_msg, X, Y, Z = generate_test_set(
            N, Au, Av, channel, n_cw, seed=SEED
        )

        print(f"  evaluating NN  ...")
        nn_err, nn_time = evaluate_nn(
            model, Z, b, frozen_u, frozen_v, Au, Av, U_msg, V_msg,
            batch_size=64, verbose=True
        )
        nn_bler = float(nn_err.mean())
        print(f"    NN BLER = {nn_bler:.4f}  ({nn_err.sum()}/{n_cw}, {nn_time:.1f}s)")

        print(f"  evaluating SC  ...")
        sc_err, sc_time = evaluate_sc(
            Z, b, frozen_u, frozen_v, channel, Au, Av, U_msg, V_msg,
            verbose=True
        )
        sc_bler = float(sc_err.mean())
        print(f"    SC BLER = {sc_bler:.4f}  ({sc_err.sum()}/{n_cw}, {sc_time:.1f}s)")

        scl_err = None
        scl_bler = None
        scl_time = None
        if args.include_scl:
            print(f"  evaluating SCL (L={args.scl_L}) ...")
            scl_err, scl_time = evaluate_scl(
                Z, b, frozen_u, frozen_v, channel, Au, Av, U_msg, V_msg,
                L=args.scl_L, verbose=True,
            )
            scl_bler = float(scl_err.mean())
            print(f"    SCL BLER = {scl_bler:.4f}  ({scl_err.sum()}/{n_cw}, "
                  f"{scl_time:.1f}s)")

        # Agreement: how many codewords do NN and SC decode identically?
        same = int((nn_err == sc_err).sum())
        both_err = int(((nn_err == 1) & (sc_err == 1)).sum())
        nn_only = int(((nn_err == 1) & (sc_err == 0)).sum())
        sc_only = int(((nn_err == 0) & (sc_err == 1)).sum())

        per_n = {
            'N': N,
            'ku': ku,
            'kv': kv,
            'n_cw': int(n_cw),
            'path_i': int(path_i),
            'nn_bler': nn_bler,
            'sc_bler': sc_bler,
            'nn_errors': int(nn_err.sum()),
            'sc_errors': int(sc_err.sum()),
            'ratio_nn_over_sc': (nn_bler / sc_bler) if sc_bler > 0 else None,
            'agreement_count': same,
            'both_error': both_err,
            'nn_only_error': nn_only,
            'sc_only_error': sc_only,
            'nn_time_s': round(nn_time, 2),
            'sc_time_s': round(sc_time, 2),
        }
        if scl_err is not None:
            # vs-SCL head-to-head
            nn_only_vs_scl = int(((nn_err == 1) & (scl_err == 0)).sum())
            scl_only_vs_nn = int(((nn_err == 0) & (scl_err == 1)).sum())
            sc_vs_scl_sc_only = int(((sc_err == 1) & (scl_err == 0)).sum())
            sc_vs_scl_scl_only = int(((sc_err == 0) & (scl_err == 1)).sum())
            per_n.update({
                'scl_L': int(args.scl_L),
                'scl_bler': scl_bler,
                'scl_errors': int(scl_err.sum()),
                'scl_time_s': round(scl_time, 2),
                'ratio_nn_over_scl': (
                    (nn_bler / scl_bler) if scl_bler > 0 else None
                ),
                'nn_only_vs_scl': nn_only_vs_scl,
                'scl_only_vs_nn': scl_only_vs_nn,
                'sc_only_vs_scl': sc_vs_scl_sc_only,
                'scl_only_vs_sc': sc_vs_scl_scl_only,
            })

        results['per_N'][str(N)] = per_n
        print(f"  agreement NN↔SC: {same}/{n_cw}  "
              f"both_err={both_err}  nn_only={nn_only}  sc_only={sc_only}")
        if scl_err is not None:
            print(f"  vs SCL:  nn_only_vs_scl={per_n['nn_only_vs_scl']}  "
                  f"scl_only_vs_nn={per_n['scl_only_vs_nn']}")

    # Save JSON
    os.makedirs('results', exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    # Write markdown table fragment. Full report lives in docs/validate_gmac_fresh.md
    # and is maintained by hand; this file is the auto-generated table.
    os.makedirs('docs', exist_ok=True)
    out_md = 'docs/validate_gmac_fresh_autotable.md'
    has_scl = any('scl_bler' in results['per_N'].get(k, {})
                  for k in ['32', '64', '128', '256'])
    with open(out_md, 'w') as f:
        f.write("<!-- Auto-generated by scripts/validate_gmac_fresh.py. "
                "Do not edit by hand. -->\n\n")
        f.write("# Fresh GMAC Paired Validation — raw table\n\n")
        f.write(f"**Seed:** {SEED}  **SNR:** {SNR_DB} dB  "
                f"**Class B symmetric** (Ru ≈ Rv ≈ 0.48)\n\n")
        if has_scl:
            f.write("| N   | n_cw  | NN BLER | SC BLER | SCL BLER | NN/SC | "
                    "NN/SCL | NN errs | SC errs | SCL errs | NN only vs SC | "
                    "SC only vs NN | NN only vs SCL | SCL only vs NN |\n")
            f.write("|-----|-------|---------|---------|----------|-------|"
                    "--------|---------|---------|----------|---------------|"
                    "---------------|----------------|----------------|\n")
        else:
            f.write("| N   | n_cw  | NN BLER | SC BLER | NN/SC | NN errs | "
                    "SC errs | NN only | SC only | both |\n")
            f.write("|-----|-------|---------|---------|-------|---------|"
                    "---------|---------|---------|------|\n")
        for key in ['32', '64', '128', '256']:
            if key not in results['per_N']:
                continue
            r = results['per_N'][key]
            ratio_sc = (f"{r['ratio_nn_over_sc']:.2f}x"
                        if r.get('ratio_nn_over_sc') else '—')
            if has_scl and 'scl_bler' in r:
                ratio_scl = (f"{r['ratio_nn_over_scl']:.2f}x"
                             if r.get('ratio_nn_over_scl') else '—')
                f.write(
                    f"| {r['N']} | {r['n_cw']} | {r['nn_bler']:.4f} | "
                    f"{r['sc_bler']:.4f} | {r['scl_bler']:.4f} | "
                    f"{ratio_sc} | {ratio_scl} | "
                    f"{r['nn_errors']} | {r['sc_errors']} | "
                    f"{r['scl_errors']} | {r['nn_only_error']} | "
                    f"{r['sc_only_error']} | {r['nn_only_vs_scl']} | "
                    f"{r['scl_only_vs_nn']} |\n"
                )
            elif has_scl:
                # This N didn't have SCL — leave SCL cells blank
                f.write(
                    f"| {r['N']} | {r['n_cw']} | {r['nn_bler']:.4f} | "
                    f"{r['sc_bler']:.4f} | — | {ratio_sc} | — | "
                    f"{r['nn_errors']} | {r['sc_errors']} | — | "
                    f"{r['nn_only_error']} | {r['sc_only_error']} | — | — |\n"
                )
            else:
                f.write(
                    f"| {r['N']} | {r['n_cw']} | {r['nn_bler']:.4f} | "
                    f"{r['sc_bler']:.4f} | {ratio_sc} | {r['nn_errors']} | "
                    f"{r['sc_errors']} | {r['nn_only_error']} | "
                    f"{r['sc_only_error']} | {r['both_error']} |\n"
                )
        f.write("\n**Checkpoints:** `ncg_gmac_mlp_N{32,64,128,256}.pt` "
                "(d=16, hidden=64).\n")
    print(f"Saved {out_md}")

    print("\n=== SUMMARY ===")
    for key in ['32', '64', '128', '256']:
        if key not in results['per_N']:
            continue
        r = results['per_N'][key]
        ratio = f"{r['ratio_nn_over_sc']:.2f}x" if r['ratio_nn_over_sc'] else '—'
        print(f"  N={r['N']}: NN={r['nn_bler']:.4f}  SC={r['sc_bler']:.4f}  "
              f"({ratio})")


if __name__ == '__main__':
    main()
