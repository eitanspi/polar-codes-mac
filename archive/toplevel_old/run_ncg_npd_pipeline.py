#!/usr/bin/env python3
"""
NCG NPD-style pipeline for GMAC Class B at N=32.

1. Train NCG at rate 1 (all positions info) for 30K iterations
2. Measure BCE-based MI per position
3. Pick top ku=kv=15 positions (same rate as SC design)
4. Eval the rate-1 model with those positions (no retraining)
5. Eval with SC-designed positions for comparison
6. SC analytical BLER for reference
7. Also eval existing rate-0.48 model with both position sets
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from polar.encoder import polar_encode_batch, build_message_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path, design_gmac
from polar.design_mc import design_from_file
from polar.eval import MACEval
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

# ── Config ──────────────────────────────────────────────────────────────────
N = 32
n = 5
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
PATH_I = 16  # Class B
KU, KV = 15, 15
BATCH = 8
LR = 5e-5
WEIGHT_DECAY = 1e-5
TRAIN_ITERS = 30000
CKPT_EVERY = 5000
MI_CODEWORDS = 500
EVAL_CODEWORDS = 2000
D = 16
HIDDEN = 64
N_LAYERS = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channel = GaussianMAC(sigma2=SIGMA2)
b = make_path(N, PATH_I)

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), 'designs')

print(f"=== NCG NPD Pipeline: GMAC Class B, N={N}, SNR={SNR_DB}dB ===")
print(f"    sigma2={SIGMA2:.6f}, path_i={PATH_I}")
print(f"    device={device}")
print()


# ── Helper: generate training data ─────────────────────────────────────────
def generate_batch(batch_size):
    """Generate a batch of (z, u_true, v_true) for rate-1 training."""
    rng = np.random.default_rng()
    U = rng.integers(0, 2, size=(batch_size, N), dtype=np.int32)
    V = rng.integers(0, 2, size=(batch_size, N), dtype=np.int32)
    X = polar_encode_batch(U)
    Y = polar_encode_batch(V)
    Z = np.zeros((batch_size, N), dtype=np.float64)
    for k in range(batch_size):
        Z[k] = channel.sample_batch(X[k], Y[k])
    z_t = torch.tensor(Z, dtype=torch.float32, device=device)
    u_t = torch.tensor(U, dtype=torch.float32, device=device)
    v_t = torch.tensor(V, dtype=torch.float32, device=device)
    return z_t, u_t, v_t


# ── Helper: NCG eval (BLER) ────────────────────────────────────────────────
@torch.no_grad()
def eval_ncg_bler(model, frozen_u_set, frozen_v_set, n_codewords=1000, batch_size=50):
    """
    Evaluate NCG model BLER.
    frozen_u_set, frozen_v_set: sets of 0-indexed frozen positions.
    """
    model.eval()

    # Convert 0-indexed frozen sets to 1-indexed frozen dicts for the model
    fu_dict = {pos + 1: 0 for pos in sorted(frozen_u_set)}
    fv_dict = {pos + 1: 0 for pos in sorted(frozen_v_set)}

    # Info positions (0-indexed)
    info_u = sorted(set(range(N)) - frozen_u_set)
    info_v = sorted(set(range(N)) - frozen_v_set)
    ku = len(info_u)
    kv = len(info_v)

    block_errors = 0
    n_done = 0

    while n_done < n_codewords:
        bs = min(batch_size, n_codewords - n_done)
        rng = np.random.default_rng()

        # Build messages
        U_msg = np.zeros((bs, N), dtype=np.int32)
        V_msg = np.zeros((bs, N), dtype=np.int32)
        U_info = rng.integers(0, 2, size=(bs, ku), dtype=np.int32)
        V_info = rng.integers(0, 2, size=(bs, kv), dtype=np.int32)
        for i, pos in enumerate(info_u):
            U_msg[:, pos] = U_info[:, i]
        for i, pos in enumerate(info_v):
            V_msg[:, pos] = V_info[:, i]

        X = polar_encode_batch(U_msg)
        Y = polar_encode_batch(V_msg)
        Z = np.zeros((bs, N), dtype=np.float64)
        for k in range(bs):
            Z[k] = channel.sample_batch(X[k], Y[k])

        z_t = torch.tensor(Z, dtype=torch.float32, device=device)
        _, _, u_hat, v_hat, _ = model(z_t, b, fu_dict, fv_dict)

        for k in range(bs):
            err = False
            for pos in info_u:
                if int(u_hat[pos + 1][k].item() + 0.5) != U_msg[k, pos]:
                    err = True
                    break
            if not err:
                for pos in info_v:
                    if int(v_hat[pos + 1][k].item() + 0.5) != V_msg[k, pos]:
                        err = True
                        break
            if err:
                block_errors += 1
        n_done += bs

    model.train()
    return block_errors / n_codewords


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Train NCG at rate 1 for 30K iterations
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Train NCG at rate 1 (fu={}, fv={}) for 30K iterations")
print("=" * 70)

model = GmacNeuralCompGraphDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS).to(device)
print(f"Model parameters: {model.count_parameters()}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
fu_rate1 = {}  # empty = all info
fv_rate1 = {}

losses = []
ckpt_losses = {}
t0 = time.time()

for it in range(1, TRAIN_ITERS + 1):
    z_t, u_t, v_t = generate_batch(BATCH)

    all_logits, all_targets, _, _, _ = model(z_t, b, fu_rate1, fv_rate1, u_t, v_t)

    if len(all_logits) == 0:
        continue

    logits_cat = torch.stack(all_logits, dim=0)  # (2N, B, 4)
    targets_cat = torch.stack(all_targets, dim=0)  # (2N, B)

    loss = F.cross_entropy(logits_cat.reshape(-1, 4), targets_cat.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())

    if it % 1000 == 0:
        avg = np.mean(losses[-1000:])
        elapsed = time.time() - t0
        print(f"  iter {it:6d}/{TRAIN_ITERS}  loss={avg:.4f}  ({elapsed:.1f}s)")

    if it % CKPT_EVERY == 0:
        avg = np.mean(losses[-CKPT_EVERY:])
        ckpt_losses[it] = avg
        ckpt_path = os.path.join(SAVE_DIR, f'ncg_gmac_rate1_N{N}_{it//1000}k.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  ** Checkpoint {it}: avg_loss={avg:.4f}, saved to {ckpt_path}")

print(f"\nTraining complete in {time.time()-t0:.1f}s")
print("Checkpoint losses:")
for it_k, l in ckpt_losses.items():
    print(f"  {it_k:6d} iters: loss={l:.4f}")

# Save final
final_path = os.path.join(SAVE_DIR, f'ncg_gmac_rate1_N{N}_final.pt')
torch.save(model.state_dict(), final_path)
print(f"Final model saved to {final_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Measure BCE-based MI per position
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Measure BCE-based MI per position (500 codewords)")
print("=" * 70)

model.eval()
u_bce = np.zeros(N)
v_bce = np.zeros(N)
count = np.zeros(N)

n_done = 0
mi_batch = 50

with torch.no_grad():
    while n_done < MI_CODEWORDS:
        bs = min(mi_batch, MI_CODEWORDS - n_done)
        z_t, u_t, v_t = generate_batch(bs)

        all_logits, all_targets, _, _, _ = model(z_t, b, fu_rate1, fv_rate1, u_t, v_t)

        # Track which position each logit corresponds to
        i_u, i_v = 0, 0
        for step_idx in range(2 * N):
            gamma = b[step_idx]
            if gamma == 0:
                i_u += 1; i_t = i_u
            else:
                i_v += 1; i_t = i_v

            pos_0idx = i_t - 1  # convert to 0-indexed
            logits = all_logits[step_idx]  # (B, 4)

            probs = F.softmax(logits, dim=-1)  # (B, 4) — classes (u=0,v=0), (u=0,v=1), (u=1,v=0), (u=1,v=1)

            # Marginalise
            u_p1 = probs[:, 2] + probs[:, 3]  # P(u=1)
            v_p1 = probs[:, 1] + probs[:, 3]  # P(v=1)

            # BCE against true bits
            u_true_bits = u_t[:, pos_0idx]
            v_true_bits = v_t[:, pos_0idx]

            u_bce_val = F.binary_cross_entropy(u_p1, u_true_bits, reduction='mean').item()
            v_bce_val = F.binary_cross_entropy(v_p1, v_true_bits, reduction='mean').item()

            u_bce[pos_0idx] += u_bce_val * bs
            v_bce[pos_0idx] += v_bce_val * bs
            count[pos_0idx] += bs

        n_done += bs

u_bce /= count
v_bce /= count

log2 = math.log(2)
u_mi = (log2 - u_bce) / log2
v_mi = (log2 - v_bce) / log2

print(f"\nU MI per position (0-indexed):")
for i in range(N):
    print(f"  pos {i:2d}: MI_u={u_mi[i]:.4f}  MI_v={v_mi[i]:.4f}")

u_mi_avg = np.mean(u_mi)
v_mi_avg = np.mean(v_mi)
print(f"\nAverage MI:  U={u_mi_avg:.4f}  V={v_mi_avg:.4f}")
print(f"Expected capacity: U~0.465, V~0.912")

u_high = np.sum(u_mi > 0.5)
v_high = np.sum(v_mi > 0.5)
print(f"Positions with MI > 0.5:  U={u_high}  V={v_high}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: Pick top ku=kv=15 positions based on MI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Select top 15 positions for U and V based on MI")
print("=" * 70)

ncg_info_u = sorted(np.argsort(u_mi)[::-1][:KU].tolist())  # top KU by MI (0-indexed)
ncg_info_v = sorted(np.argsort(v_mi)[::-1][:KV].tolist())
ncg_frozen_u = set(range(N)) - set(ncg_info_u)
ncg_frozen_v = set(range(N)) - set(ncg_info_v)

print(f"NCG info_u (0-idx): {ncg_info_u}")
print(f"NCG info_v (0-idx): {ncg_info_v}")
print(f"NCG frozen_u count: {len(ncg_frozen_u)}")
print(f"NCG frozen_v count: {len(ncg_frozen_v)}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 & 5: Get SC design positions
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Load SC design positions")
print("=" * 70)

mc_path = os.path.join(DESIGNS_DIR, 'gmac_B_n5_snr6dB.npz')
Au_sc, Av_sc, fu_sc, fv_sc, pe_u_sc, pe_v_sc, _ = design_from_file(mc_path, n, KU, KV)
sc_info_u_0idx = sorted([p - 1 for p in Au_sc])  # convert to 0-indexed
sc_info_v_0idx = sorted([p - 1 for p in Av_sc])
sc_frozen_u = set(range(N)) - set(sc_info_u_0idx)
sc_frozen_v = set(range(N)) - set(sc_info_v_0idx)

print(f"SC info_u (0-idx): {sc_info_u_0idx}")
print(f"SC info_v (0-idx): {sc_info_v_0idx}")
print(f"Overlap with NCG: U={len(set(ncg_info_u) & set(sc_info_u_0idx))}/{KU}, V={len(set(ncg_info_v) & set(sc_info_v_0idx))}/{KV}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: Eval rate-1 model with NCG positions and SC positions
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Eval rate-1 NCG model")
print("=" * 70)

print(f"\n  Evaluating NCG model + NCG positions ({EVAL_CODEWORDS} codewords)...")
bler_ncg_ncg = eval_ncg_bler(model, ncg_frozen_u, ncg_frozen_v, n_codewords=EVAL_CODEWORDS)
print(f"  BLER (NCG model + NCG positions): {bler_ncg_ncg:.4f}")

print(f"\n  Evaluating NCG model + SC positions ({EVAL_CODEWORDS} codewords)...")
bler_ncg_sc = eval_ncg_bler(model, sc_frozen_u, sc_frozen_v, n_codewords=EVAL_CODEWORDS)
print(f"  BLER (NCG model + SC positions):  {bler_ncg_sc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: SC analytical BLER reference
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: SC BLER reference (interleaved decoder)")
print("=" * 70)

sc_eval = MACEval(channel, log_domain=True, backend='interleaved')
ber_u_sc, ber_v_sc, bler_sc = sc_eval.run(
    N, b, Au_sc, Av_sc, fu_sc, fv_sc,
    n_codewords=EVAL_CODEWORDS, batch_size=25, verbose=True
)
print(f"  SC BLER: {bler_sc:.4f}  (ber_u={ber_u_sc:.4e}, ber_v={ber_v_sc:.4e})")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: Eval existing rate-0.48 model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Eval existing rate-0.48 model (ncg_gmac_mlp_N32.pt)")
print("=" * 70)

existing_path = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N32.pt')
if os.path.exists(existing_path):
    model_existing = GmacNeuralCompGraphDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS).to(device)
    sd = torch.load(existing_path, map_location=device, weights_only=True)
    # Remove 'tree.' prefix and skip embedding_z
    sd_clean = {k.replace('tree.', ''): v for k, v in sd.items() if 'embedding_z' not in k}
    # Try loading
    try:
        model_existing.load_state_dict(sd_clean, strict=False)
        print(f"  Loaded existing model from {existing_path}")
    except Exception as e:
        print(f"  Warning loading: {e}")
        # Try to load what we can
        model_state = model_existing.state_dict()
        loaded = 0
        for k, v in sd_clean.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model_existing.load_state_dict(model_state)
        print(f"  Partially loaded {loaded} params")

    print(f"\n  Evaluating existing model + NCG positions ({EVAL_CODEWORDS} codewords)...")
    bler_exist_ncg = eval_ncg_bler(model_existing, ncg_frozen_u, ncg_frozen_v, n_codewords=EVAL_CODEWORDS)
    print(f"  BLER (existing model + NCG positions): {bler_exist_ncg:.4f}")

    print(f"\n  Evaluating existing model + SC positions ({EVAL_CODEWORDS} codewords)...")
    bler_exist_sc = eval_ncg_bler(model_existing, sc_frozen_u, sc_frozen_v, n_codewords=EVAL_CODEWORDS)
    print(f"  BLER (existing model + SC positions):  {bler_exist_sc:.4f}")
else:
    print(f"  Existing model not found at {existing_path}")
    bler_exist_ncg = None
    bler_exist_sc = None


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Config: GMAC Class B, N={N}, SNR={SNR_DB}dB, ku=kv={KU}")
print()
print("Training checkpoints:")
for it_k, l in ckpt_losses.items():
    print(f"  {it_k:6d}: loss={l:.4f}")
print()
print(f"MI analysis: avg_U={u_mi_avg:.4f} (cap~0.465), avg_V={v_mi_avg:.4f} (cap~0.912)")
print(f"  Positions MI>0.5: U={u_high}/{N}, V={v_high}/{N}")
print()
print(f"NCG info_u: {ncg_info_u}")
print(f"SC  info_u: {sc_info_u_0idx}")
print(f"NCG info_v: {ncg_info_v}")
print(f"SC  info_v: {sc_info_v_0idx}")
print(f"Overlap: U={len(set(ncg_info_u) & set(sc_info_u_0idx))}/{KU}, V={len(set(ncg_info_v) & set(sc_info_v_0idx))}/{KV}")
print()
print("BLER Results:")
print(f"  Rate-1 NCG + NCG positions: {bler_ncg_ncg:.4f}")
print(f"  Rate-1 NCG + SC positions:  {bler_ncg_sc:.4f}")
print(f"  SC decoder (reference):     {bler_sc:.4f}")
if bler_exist_ncg is not None:
    print(f"  Existing NCG + NCG positions: {bler_exist_ncg:.4f}")
    print(f"  Existing NCG + SC positions:  {bler_exist_sc:.4f}")
print()
print("Done!")
