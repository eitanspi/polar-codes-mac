#!/usr/bin/env python3
"""
nn_vs_sc_true_bits.py — Compare NN decoder vs SC decoder against TRUE transmitted bits.

For N=256, Class B (path_i=128), SNR=6dB, runs 500 codewords and tracks:
- Overall BLER for NN and SC
- Per-position error rates grouped by quartiles
- Top 10 worst positions for each decoder
- Average errors per codeword
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder_interleaved import decode_single
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# ─── Config ──────────────────────────────────────────────────────────────────

N = 256
n = int(math.log2(N))
PATH_I = N // 2  # Class B: path_i = 128
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
KU = 123
KV = 123
NUM_CW = 500
NN_BATCH = 8  # batch size for NN decoder

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

# ─── Model ───────────────────────────────────────────────────────────────────

class SimpleMLP_Gmac(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2, z_hidden=32):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

    def forward(self, z, b, fu, fv, u_true=None, v_true=None):
        n = int(math.log2(z.shape[1]))
        br = torch.from_numpy(bit_reversal_perm(n)).long()
        root = self.z_encoder(z.unsqueeze(-1))[:, br]
        return self.tree(z=None, b=b, frozen_u=fu, frozen_v=fv,
                        u_true=u_true, v_true=v_true, root_emb=root)

# ─── Load design ─────────────────────────────────────────────────────────────

print(f"Config: N={N}, Class B (path_i={PATH_I}), SNR={SNR_DB}dB, sigma2={SIGMA2:.6f}")
print(f"  KU={KU}, KV={KV}, num_codewords={NUM_CW}")

mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{int(SNR_DB)}dB.npz')
Au, Av, frozen_u, frozen_v, pe_u, pe_v, path_i_loaded = design_from_file(mc_path, n, KU, KV)
b = make_path(N, PATH_I)

print(f"  Design loaded from {mc_path}")
print(f"  |Au|={len(Au)}, |Av|={len(Av)}, |frozen_u|={len(frozen_u)}, |frozen_v|={len(frozen_v)}")

# ─── Load NN model ───────────────────────────────────────────────────────────

model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
model_path = os.path.join(MODELS_DIR, 'ncg_gmac_mlp_N256.pt')
sd = torch.load(model_path, map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()
print(f"  NN model loaded from {model_path} ({model.tree.count_parameters()} tree params)")

# ─── Channel ─────────────────────────────────────────────────────────────────

channel = GaussianMAC(sigma2=SIGMA2)

# ─── Generate all codewords ──────────────────────────────────────────────────

print(f"\nGenerating {NUM_CW} codewords...")
rng = np.random.default_rng(42)

all_u = np.zeros((NUM_CW, N), dtype=np.int32)
all_v = np.zeros((NUM_CW, N), dtype=np.int32)
for p in Au:
    all_u[:, p - 1] = rng.integers(0, 2, NUM_CW)
for p in Av:
    all_v[:, p - 1] = rng.integers(0, 2, NUM_CW)

all_x = polar_encode_batch(all_u)
all_y = polar_encode_batch(all_v)
all_z = channel.sample_batch(all_x, all_y)

print(f"  Generated. z shape: {all_z.shape}, z range: [{all_z.min():.2f}, {all_z.max():.2f}]")

# ─── Per-position error tracking ─────────────────────────────────────────────

# Track errors at each 1-indexed info position for U and V
nn_u_errors = np.zeros(N + 1, dtype=np.int32)  # 1-indexed
nn_v_errors = np.zeros(N + 1, dtype=np.int32)
sc_u_errors = np.zeros(N + 1, dtype=np.int32)
sc_v_errors = np.zeros(N + 1, dtype=np.int32)

nn_block_errors = 0
sc_block_errors = 0
nn_total_bit_errors = 0
sc_total_bit_errors = 0

# ─── NN Decoding ─────────────────────────────────────────────────────────────

print(f"\nRunning NN decoder (batch={NN_BATCH})...")
t0 = time.time()

with torch.no_grad():
    for start in range(0, NUM_CW, NN_BATCH):
        end = min(start + NN_BATCH, NUM_CW)
        actual = end - start

        z_batch = torch.from_numpy(all_z[start:end]).float()
        _, _, uh, vh, _ = model(z_batch, b, frozen_u, frozen_v)

        for i in range(actual):
            cw_idx = start + i
            block_err = False

            for p in Au:
                if p in uh:
                    nn_bit = int(uh[p][i].item())
                    true_bit = all_u[cw_idx, p - 1]
                    if nn_bit != true_bit:
                        nn_u_errors[p] += 1
                        nn_total_bit_errors += 1
                        block_err = True

            for p in Av:
                if p in vh:
                    nn_bit = int(vh[p][i].item())
                    true_bit = all_v[cw_idx, p - 1]
                    if nn_bit != true_bit:
                        nn_v_errors[p] += 1
                        nn_total_bit_errors += 1
                        block_err = True

            if block_err:
                nn_block_errors += 1

        if (end) % 100 == 0 or end == NUM_CW:
            elapsed = time.time() - t0
            print(f"  NN: {end}/{NUM_CW} done ({elapsed:.1f}s)")

nn_time = time.time() - t0
print(f"  NN decoding complete: {nn_time:.1f}s ({nn_time/NUM_CW*1000:.1f}ms/cw)")

# ─── SC Decoding ─────────────────────────────────────────────────────────────

print(f"\nRunning SC decoder (single-threaded)...")
t0 = time.time()

for cw_idx in range(NUM_CW):
    z_list = all_z[cw_idx].tolist()
    u_dec, v_dec = decode_single(N, z_list, b, frozen_u, frozen_v, channel)

    block_err = False
    for p in Au:
        sc_bit = u_dec[p - 1]
        true_bit = all_u[cw_idx, p - 1]
        if sc_bit != true_bit:
            sc_u_errors[p] += 1
            sc_total_bit_errors += 1
            block_err = True

    for p in Av:
        sc_bit = v_dec[p - 1]
        true_bit = all_v[cw_idx, p - 1]
        if sc_bit != true_bit:
            sc_v_errors[p] += 1
            sc_total_bit_errors += 1
            block_err = True

    if block_err:
        sc_block_errors += 1

    if (cw_idx + 1) % 100 == 0 or cw_idx + 1 == NUM_CW:
        elapsed = time.time() - t0
        print(f"  SC: {cw_idx + 1}/{NUM_CW} done ({elapsed:.1f}s)")

sc_time = time.time() - t0
print(f"  SC decoding complete: {sc_time:.1f}s ({sc_time/NUM_CW*1000:.1f}ms/cw)")

# ─── Results ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RESULTS: NN vs SC decoder (compared against TRUE transmitted bits)")
print("=" * 70)

nn_bler = nn_block_errors / NUM_CW
sc_bler = sc_block_errors / NUM_CW
print(f"\n  Overall BLER:")
print(f"    NN:  {nn_block_errors}/{NUM_CW} = {nn_bler:.4f}")
print(f"    SC:  {sc_block_errors}/{NUM_CW} = {sc_bler:.4f}")
print(f"    Ratio NN/SC: {nn_bler/sc_bler:.3f}" if sc_bler > 0 else "    SC BLER = 0")

total_info_bits = NUM_CW * (KU + KV)
print(f"\n  Average bit errors per codeword:")
print(f"    NN:  {nn_total_bit_errors/NUM_CW:.2f} errors/cw (BER={nn_total_bit_errors/total_info_bits:.6f})")
print(f"    SC:  {sc_total_bit_errors/NUM_CW:.2f} errors/cw (BER={sc_total_bit_errors/total_info_bits:.6f})")

# ─── Per-position analysis ───────────────────────────────────────────────────

# Combine U and V info positions into a single ordered list by decoding order
# For analysis, we group ALL info positions (Au + Av) by quartile of their index

all_info_pos_u = sorted(Au)
all_info_pos_v = sorted(Av)

def quartile_analysis(errors_u, errors_v, info_u, info_v, name):
    """Group info positions by quartile and compute error rates."""
    # Collect (position, error_count, user) for all info positions
    all_errs = []
    for p in info_u:
        all_errs.append((p, errors_u[p], 'U'))
    for p in info_v:
        all_errs.append((p, errors_v[p], 'V'))

    # Sort by position
    all_errs.sort(key=lambda x: x[0])
    n_info = len(all_errs)
    q_size = n_info // 4

    print(f"\n  {name} per-position error rate by quartile (of {n_info} info positions):")
    for qi, qname in enumerate(["Q1 (first quarter)", "Q2 (second quarter)",
                                 "Q3 (third quarter)", "Q4 (last quarter)"]):
        start = qi * q_size
        end = (qi + 1) * q_size if qi < 3 else n_info
        q_errs = [e[1] for e in all_errs[start:end]]
        q_total = sum(q_errs)
        q_positions = end - start
        avg_rate = q_total / (q_positions * NUM_CW)
        pos_range = f"pos {all_errs[start][0]}-{all_errs[end-1][0]}"
        print(f"    {qname}: {q_total:5d} errors across {q_positions} positions "
              f"(avg rate={avg_rate:.4f}) [{pos_range}]")

    # Also by user
    u_total = sum(errors_u[p] for p in info_u)
    v_total = sum(errors_v[p] for p in info_v)
    print(f"    User U total: {u_total} errors across {len(info_u)} info positions "
          f"(avg rate={u_total/(len(info_u)*NUM_CW):.4f})")
    print(f"    User V total: {v_total} errors across {len(info_v)} info positions "
          f"(avg rate={v_total/(len(info_v)*NUM_CW):.4f})")

quartile_analysis(nn_u_errors, nn_v_errors, Au, Av, "NN")
quartile_analysis(sc_u_errors, sc_v_errors, Au, Av, "SC")

# ─── Top 10 worst positions ──────────────────────────────────────────────────

def top_worst(errors_u, errors_v, info_u, info_v, name):
    all_errs = []
    for p in info_u:
        all_errs.append((p, errors_u[p], 'U'))
    for p in info_v:
        all_errs.append((p, errors_v[p], 'V'))
    all_errs.sort(key=lambda x: -x[1])

    print(f"\n  {name} Top 10 highest error positions:")
    for rank, (pos, count, user) in enumerate(all_errs[:10], 1):
        rate = count / NUM_CW
        print(f"    #{rank}: pos={pos:3d} (user {user}) — {count}/{NUM_CW} = {rate:.4f}")

top_worst(nn_u_errors, nn_v_errors, Au, Av, "NN")
top_worst(sc_u_errors, sc_v_errors, Au, Av, "SC")

# ─── Position-by-position comparison ────────────────────────────────────────

print(f"\n  Positions where NN is BETTER than SC (lower error count):")
nn_better = []
sc_better = []
for p in Au:
    diff = nn_u_errors[p] - sc_u_errors[p]
    if diff < 0:
        nn_better.append((p, 'U', -diff, nn_u_errors[p], sc_u_errors[p]))
    elif diff > 0:
        sc_better.append((p, 'U', diff, nn_u_errors[p], sc_u_errors[p]))
for p in Av:
    diff = nn_v_errors[p] - sc_v_errors[p]
    if diff < 0:
        nn_better.append((p, 'V', -diff, nn_v_errors[p], sc_v_errors[p]))
    elif diff > 0:
        sc_better.append((p, 'V', diff, nn_v_errors[p], sc_v_errors[p]))

nn_better.sort(key=lambda x: -x[2])
sc_better.sort(key=lambda x: -x[2])

print(f"    NN better at {len(nn_better)} positions, SC better at {len(sc_better)} positions")
if nn_better:
    print(f"    Top 5 NN advantages:")
    for pos, user, diff, nn_e, sc_e in nn_better[:5]:
        print(f"      pos={pos:3d} ({user}): NN={nn_e}, SC={sc_e} (NN saves {diff} errors)")
if sc_better:
    print(f"    Top 5 SC advantages:")
    for pos, user, diff, nn_e, sc_e in sc_better[:5]:
        print(f"      pos={pos:3d} ({user}): NN={nn_e}, SC={sc_e} (SC saves {diff} errors)")

print(f"\n  Timing: NN={nn_time:.1f}s, SC={sc_time:.1f}s")
print(f"  Done.")
