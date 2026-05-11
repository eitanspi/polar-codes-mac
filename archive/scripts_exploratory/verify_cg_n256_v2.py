#!/usr/bin/env python3
"""
TASK 3: CG Reference Validation at N=256 — fixed checkpoint loading.
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
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

print("Starting CG decoder N=256 evaluation (v2)...", flush=True)

N = 256; n = 8
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
DESIGN_FILE = 'designs/gmac_B_n8_snr6dB.npz'
Au, Av, frozen_u, frozen_v, pe_u, pe_v, path_i = design_from_file(DESIGN_FILE, n, ku=123, kv=123)
b = make_path(N, path_i)
channel = GaussianMAC(sigma2=SIGMA2)
print(f"Design: N={N}, path_i={path_i}, ku={len(Au)}, kv={len(Av)}", flush=True)

def load_cg_model(ckpt_path, d=16, hidden=64):
    """Load CG model handling the tree.* prefix wrapper format."""
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'model_state_dict' in state:
        sd = state['model_state_dict']
    else:
        sd = state

    # Check if keys have tree. prefix
    has_tree_prefix = any(k.startswith('tree.') for k in sd.keys())
    has_z_encoder = any(k.startswith('z_encoder.') for k in sd.keys())
    has_embedding_z = any('embedding_z' in k for k in sd.keys())

    # Build the CG model
    model = GmacNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=2)

    if has_tree_prefix:
        # Strip tree. prefix for CG model params
        cg_sd = {}
        z_enc_sd = {}
        for k, v in sd.items():
            if k.startswith('tree.'):
                new_k = k[5:]  # strip 'tree.'
                if 'embedding_z' in new_k:
                    continue  # skip discrete embedding
                cg_sd[new_k] = v
            elif k.startswith('z_encoder.'):
                z_enc_sd[k] = v

        # Load CG model (skip z_encoder, load separately)
        model_sd = model.state_dict()
        for k, v in cg_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v

        # Load z_encoder
        for k, v in z_enc_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v

        model.load_state_dict(model_sd)
    else:
        model.load_state_dict(sd)

    model.eval()
    return model

def evaluate_cg(model, name, n_cw=100):
    print(f"\n  Evaluating {name} on {n_cw} codewords...", flush=True)
    bler_count = 0
    t0 = time.time()
    for i in range(n_cw):
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
            _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u, frozen_v)
        u_dec = np.zeros((1, N), dtype=np.int32)
        v_dec = np.zeros((1, N), dtype=np.int32)
        for pos, val in u_hat.items():
            u_dec[0, pos-1] = int(val[0].round().item())
        for pos, val in v_hat.items():
            v_dec[0, pos-1] = int(val[0].round().item())
        if (u_dec != U).any() or (v_dec != V).any():
            bler_count += 1
        if (i+1) % 25 == 0:
            print(f"    {i+1}/{n_cw} BLER={bler_count/(i+1):.4f}", flush=True)
    elapsed = time.time() - t0
    bler = bler_count / n_cw
    print(f"  RESULT: BLER={bler:.4f} ({bler_count}/{n_cw}) in {elapsed:.1f}s", flush=True)
    return {'bler': bler, 'bler_count': bler_count, 'n_cw': n_cw, 'time_s': elapsed}

results = {}

# Load and eval ncg_gmac_mlp_N256
checkpoints = [
    ('ncg_gmac_mlp_N256', 'saved_models/ncg_gmac_mlp_N256.pt', 16, 64),
    ('campaign_n256_sched_best', 'saved_models/campaign_n256_sched_best.pt', 16, 64),
    ('n256_long_best', 'saved_models/n256_long_best.pt', 16, 64),
]

for name, path, d, h in checkpoints:
    print(f"\n--- {name} ---", flush=True)
    try:
        model = load_cg_model(path, d=d, hidden=h)
        n_params = model.count_parameters()
        print(f"  Loaded: d={d}, hidden={h}, params={n_params}", flush=True)
        r = evaluate_cg(model, name)
        r['d'] = d; r['hidden'] = h; r['params'] = n_params
        results[name] = r
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

# Also check ncg_gmac_mlp_N128 for comparison
for alt_name, alt_path, alt_N in [
    ('ncg_gmac_mlp_N128', 'saved_models/ncg_gmac_mlp_N128.pt', 128),
]:
    print(f"\n--- {alt_name} (N={alt_N}, different from target N=256) ---", flush=True)
    print(f"  Skipping: different N, not directly comparable.", flush=True)

print("\n\n=== CG SUMMARY ===", flush=True)
for name, r in results.items():
    print(f"  {name}: BLER={r['bler']:.4f} params={r['params']}", flush=True)

with open('results/task3_cg_evaluation.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nTASK 3 COMPLETE", flush=True)
