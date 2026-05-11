"""
Minimal N=4 teacher-forcing demo.

Goal: show what the decoder actually sees at each leaf under fast_ce /
teacher forcing — the true u and v at that leaf, the logits the model
produces, and the 4-class target that the loss is taken against.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)

from polar.encoder import polar_encode_batch
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 4
B = 1  # batch size 1 for readability
d = 8

# ── Step 1: pick true u and v (all info bits, no frozen) ─────────────────
u_true_np = np.array([[1, 0, 1, 1]], dtype=np.int32)
v_true_np = np.array([[0, 1, 1, 0]], dtype=np.int32)

# Polar-encode both users
X = polar_encode_batch(u_true_np)  # (1, 4)
Y = polar_encode_batch(v_true_np)

print("=" * 60)
print(f"N = {N}  (batch size {B})")
print("=" * 60)
print(f"true u      : {u_true_np[0].tolist()}")
print(f"true v      : {v_true_np[0].tolist()}")
print(f"X = enc(u)  : {X[0].tolist()}")
print(f"Y = enc(v)  : {Y[0].tolist()}")

# ── Step 2: simulate GMAC channel output ─────────────────────────────────
# Z[i] = (1-2X[i]) + (1-2Y[i]) + noise
sigma = 0.3
bpsk_x = 1 - 2 * X.astype(np.float32)
bpsk_y = 1 - 2 * Y.astype(np.float32)
noise = sigma * np.random.randn(*X.shape).astype(np.float32)
Z = bpsk_x + bpsk_y + noise
print(f"Z (GMAC out): {[f'{zi:+.3f}' for zi in Z[0]]}")

# ── Step 3: build an UNTRAINED model, set up class-A path ────────────────
model = GmacNeuralCompGraphDecoder(d=d, hidden=32, n_layers=2)
model.eval()

# Path "b": 0 means a U-leaf, 1 means a V-leaf.
# Class-A = all U first, then all V
b_path = [0] * N + [1] * N

# No frozen positions: every leaf is an info leaf
frozen_u, frozen_v = {}, {}

z_t = torch.from_numpy(Z).float()
u_t = torch.from_numpy(u_true_np).float()
v_t = torch.from_numpy(v_true_np).float()

# ── Step 4: run forward in teacher-forced mode ───────────────────────────
# The forward() method, when given u_true and v_true, uses the TRUE bits
# as the committed values at each leaf — that is exactly the teacher
# forcing used in training.
with torch.no_grad():
    all_logits, all_targets, u_hat, v_hat, _ = model(
        z_t, b_path, frozen_u, frozen_v, u_true=u_t, v_true=v_t
    )

# ── Step 5: pretty-print what happened at each leaf ──────────────────────
print()
print("Leaf-by-leaf trace under teacher forcing:")
print("-" * 60)
print(f"{'step':<6}{'user':<6}{'i':<4}{'true u':<8}{'true v':<8}"
      f"{'target (u*2+v)':<16}{'logits (argmax)':<16}")
print("-" * 60)

u_idx = 0
v_idx = 0
leaf_counter = 0
for step, gamma in enumerate(b_path):
    if gamma == 0:
        i_t = u_idx + 1  # 1-indexed
        u_idx += 1
        user = "U"
    else:
        i_t = v_idx + 1
        v_idx += 1
        user = "V"

    # Position in the original (pre-path) bit index: both users share the
    # same code position i_t (1-indexed from 1..N)
    tu = int(u_true_np[0, i_t - 1])
    tv = int(v_true_np[0, i_t - 1])
    target_cls = tu * 2 + tv

    # logits / targets lists were appended in step order for info leaves
    logits = all_logits[leaf_counter][0].numpy()
    pred = int(logits.argmax())
    leaf_counter += 1

    logits_str = "[" + ", ".join(f"{l:+.2f}" for l in logits) + f"] -> {pred}"
    print(f"{step:<6}{user:<6}{i_t:<4}{tu:<8}{tv:<8}"
          f"{target_cls:<16}{logits_str}")

print("-" * 60)
print()
print("Loss computation:")
targets = torch.stack(all_targets)  # (num_info_leaves, B)
logits_stack = torch.stack(all_logits)  # (num_info_leaves, B, 4)
ce = torch.nn.functional.cross_entropy(
    logits_stack.view(-1, 4), targets.view(-1), reduction='none'
)
for k in range(len(all_logits)):
    print(f"  leaf {k}:  target class = {int(targets[k,0])},  "
          f"CE = {ce[k].item():.4f}")
print(f"  -> total loss (mean) = {ce.mean().item():.4f}")
