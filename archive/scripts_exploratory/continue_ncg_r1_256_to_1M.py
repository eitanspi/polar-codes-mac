#!/usr/bin/env python3
"""
Continue NCG rate-1 N=256 training from iter300000 to 1M.
Waits for PID 92903 to finish first, then loads the checkpoint and continues.

Saves checkpoints every 50K iters (since we already have every-10K up to 300K).
"""
import sys, os, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn.functional as F
from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

# Wait for PID 92903 to finish
TARGET_PID = 92903
print(f"Waiting for PID {TARGET_PID} to finish...", flush=True)
while True:
    try:
        os.kill(TARGET_PID, 0)  # check if alive
        time.sleep(60)
    except OSError:
        print(f"PID {TARGET_PID} finished. Starting continuation.", flush=True)
        break

N = 256
n = 8
SIGMA2 = 10 ** (-6.0 / 10)
BATCH = 8
START_ITER = 300000
END_ITER = 1000000
SAVE_EVERY = 50000

CKPT_DIR = 'class_c_npd/results/ncg_r1_256'
CKPT_IN = os.path.join(CKPT_DIR, f'iter{START_ITER}.pt')

# Wait for checkpoint file to exist (might take a few seconds after process ends)
for _ in range(30):
    if os.path.exists(CKPT_IN):
        break
    time.sleep(10)

if not os.path.exists(CKPT_IN):
    print(f"ERROR: {CKPT_IN} not found. Aborting.", flush=True)
    sys.exit(1)

ch = GaussianMAC(sigma2=SIGMA2)
b = make_path(N, N // 2)

model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)
model.load_state_dict(torch.load(CKPT_IN, map_location='cpu', weights_only=True))
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
rng = np.random.default_rng(2026)

print(f"Loaded {CKPT_IN}. Training {START_ITER+1} -> {END_ITER}.", flush=True)
t0 = time.time()

for it in range(START_ITER + 1, END_ITER + 1):
    u = rng.integers(0, 2, (BATCH, N)).astype(int)
    v = rng.integers(0, 2, (BATCH, N)).astype(int)
    x = polar_encode_batch(u)
    y = polar_encode_batch(v)
    z = torch.from_numpy(ch.sample_batch(x, y)).float()
    logits, targets, _, _, _ = model(
        z, b, {}, {},
        u_true=torch.from_numpy(u).float(),
        v_true=torch.from_numpy(v).float())
    if logits:
        loss = F.cross_entropy(
            torch.stack(logits).reshape(-1, 4),
            torch.stack(targets).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if it % SAVE_EVERY == 0:
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f'iter{it}.pt'))
        elapsed = (time.time() - t0) / 60
        print(f'[{it}/{END_ITER}] loss={loss.item():.4f} {elapsed:.1f}min', flush=True)

elapsed = (time.time() - t0) / 60
torch.save(model.state_dict(), os.path.join(CKPT_DIR, f'iter{END_ITER}.pt'))
print(f'Done. {elapsed:.1f}min total.', flush=True)
