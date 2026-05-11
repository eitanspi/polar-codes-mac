#!/usr/bin/env python3
"""
TASK 3: CG Reference Validation at N=256.
Load the production CG decoder and evaluate under same conditions.
"""

import sys, os, math, time, json
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path

print("Starting CG decoder N=256 evaluation...", flush=True)

# ─── Config ─────────────────────────────────────────────────────────────────
N = 256; n = 8
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
DESIGN_FILE = 'designs/gmac_B_n8_snr6dB.npz'

Au, Av, frozen_u, frozen_v, pe_u, pe_v, path_i = design_from_file(DESIGN_FILE, n, ku=123, kv=123)
b = make_path(N, path_i)
channel = GaussianMAC(sigma2=SIGMA2)

print(f"Design: N={N}, path_i={path_i}, ku={len(Au)}, kv={len(Av)}", flush=True)

# ─── Try to load CG models ─────────────────────────────────────────────────
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

cg_checkpoints = [
    ('ncg_gmac_mlp_N256', 'saved_models/ncg_gmac_mlp_N256.pt'),
    ('campaign_n256_sched_best', 'saved_models/campaign_n256_sched_best.pt'),
    ('n256_long_best', 'saved_models/n256_long_best.pt'),
    ('ncg_pure_neural_N256', 'saved_models/ncg_pure_neural_N256.pt'),
]

results = {}

for name, ckpt_path in cg_checkpoints:
    print(f"\n--- Trying {name} ---", flush=True)

    # Try different d values
    for d_val in [16, 32]:
        try:
            cg_model = GmacNeuralCompGraphDecoder(d=d_val, hidden=64, n_layers=2)
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            if isinstance(state, dict) and 'model_state_dict' in state:
                cg_model.load_state_dict(state['model_state_dict'])
            else:
                cg_model.load_state_dict(state)
            cg_model.eval()
            n_params = cg_model.count_parameters()
            print(f"  Loaded {name} with d={d_val}, params={n_params}", flush=True)

            # Evaluate
            N_CW = 100
            bler_count = 0
            t0 = time.time()

            for i in range(N_CW):
                rng = np.random.default_rng(999 + i)
                U = np.zeros((1, N), dtype=np.int32)
                V = np.zeros((1, N), dtype=np.int32)
                for pos in Au: U[0, pos-1] = rng.integers(0, 2)
                for pos in Av: V[0, pos-1] = rng.integers(0, 2)
                X = polar_encode_batch(U)
                Y = polar_encode_batch(V)
                Z = channel.sample_batch(X, Y)

                z_t = torch.from_numpy(Z).float()
                with torch.no_grad():
                    all_logits, all_targets, u_hat, v_hat, _ = cg_model(
                        z_t, b, frozen_u, frozen_v)

                # Extract decoded bits
                u_dec = np.zeros((1, N), dtype=np.int32)
                v_dec = np.zeros((1, N), dtype=np.int32)
                for pos, val in u_hat.items():
                    u_dec[0, pos-1] = int(val[0].round().item())
                for pos, val in v_hat.items():
                    v_dec[0, pos-1] = int(val[0].round().item())

                if (u_dec != U).any() or (v_dec != V).any():
                    bler_count += 1

                if (i+1) % 25 == 0:
                    print(f"    {i+1}/{N_CW} done, BLER so far={bler_count/(i+1):.4f}", flush=True)

            elapsed = time.time() - t0
            bler = bler_count / N_CW
            print(f"  Result: BLER={bler:.4f} ({bler_count}/{N_CW}) in {elapsed:.1f}s", flush=True)
            results[name] = {
                'd': d_val, 'params': n_params, 'bler': bler,
                'bler_count': bler_count, 'n_cw': N_CW, 'time': elapsed
            }
            break  # success, don't try other d values

        except Exception as e:
            print(f"  d={d_val} failed: {e}", flush=True)
            continue

# Also try with hidden=128 for d=16
for name, ckpt_path in cg_checkpoints:
    if name in results:
        continue
    print(f"\n--- Retry {name} with hidden=128 ---", flush=True)
    for d_val, h_val in [(16, 128), (16, 32), (32, 128), (32, 32)]:
        try:
            cg_model = GmacNeuralCompGraphDecoder(d=d_val, hidden=h_val, n_layers=2)
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            if isinstance(state, dict) and 'model_state_dict' in state:
                cg_model.load_state_dict(state['model_state_dict'])
            else:
                cg_model.load_state_dict(state)
            cg_model.eval()
            n_params = cg_model.count_parameters()
            print(f"  Loaded {name} with d={d_val}, hidden={h_val}, params={n_params}", flush=True)

            # Quick eval
            N_CW = 100
            bler_count = 0
            t0 = time.time()
            for i in range(N_CW):
                rng = np.random.default_rng(999 + i)
                U = np.zeros((1, N), dtype=np.int32)
                V = np.zeros((1, N), dtype=np.int32)
                for pos in Au: U[0, pos-1] = rng.integers(0, 2)
                for pos in Av: V[0, pos-1] = rng.integers(0, 2)
                X = polar_encode_batch(U)
                Y = polar_encode_batch(V)
                Z = channel.sample_batch(X, Y)
                z_t = torch.from_numpy(Z).float()
                with torch.no_grad():
                    _, _, u_hat, v_hat, _ = cg_model(z_t, b, frozen_u, frozen_v)
                u_dec = np.zeros((1, N), dtype=np.int32)
                v_dec = np.zeros((1, N), dtype=np.int32)
                for pos, val in u_hat.items():
                    u_dec[0, pos-1] = int(val[0].round().item())
                for pos, val in v_hat.items():
                    v_dec[0, pos-1] = int(val[0].round().item())
                if (u_dec != U).any() or (v_dec != V).any():
                    bler_count += 1
                if (i+1) % 50 == 0:
                    print(f"    {i+1}/{N_CW} BLER={bler_count/(i+1):.4f}", flush=True)
            elapsed = time.time() - t0
            bler = bler_count / N_CW
            print(f"  Result: BLER={bler:.4f} ({bler_count}/{N_CW}) in {elapsed:.1f}s", flush=True)
            results[name] = {
                'd': d_val, 'hidden': h_val, 'params': n_params, 'bler': bler,
                'bler_count': bler_count, 'n_cw': N_CW, 'time': elapsed
            }
            break
        except Exception as e:
            print(f"  d={d_val} h={h_val} failed: {e}", flush=True)

print("\n\n=== SUMMARY ===", flush=True)
for name, r in results.items():
    print(f"  {name}: BLER={r['bler']:.4f} d={r['d']} params={r['params']}", flush=True)

with open('results/task3_cg_evaluation.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nResults saved to results/task3_cg_evaluation.json", flush=True)
print("TASK 3 COMPLETE", flush=True)
