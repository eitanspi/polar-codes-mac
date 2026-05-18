"""Quick local CPU smoke test for NCG ISI-MAC decoder.

Trains tiny NCG model on ISI-MAC h=0.3 SNR=6dB at N=16 and measures BLER.
Goal: validate the decoder runs end-to-end, see learning curve and final BLER.
"""
import sys, os, time
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")

import numpy as np
import torch
import torch.nn.functional as F

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode_batch
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_isi_mac import ISIMACNeuralDecoder

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cpu")

N, ku, kv, n = 16, 4, 7, 4
SIGMA2 = 10**(-0.6)
ch = ISIMAC(sigma2=SIGMA2, h=0.3)
b = make_path(N, N)  # Class C corner-rate

# Frozen sets — use GMAC proxy design for this smoke test
gmac_file = f"/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs/gmac_C_n{n}_snr6dB.npz"
Au, Av, fu_set, fv_set, _, _, _ = design_from_file(gmac_file, n, ku, kv)
fu = {int(p): 0 for p in fu_set}
fv = {int(p): 0 for p in fv_set}
print(f"Au={sorted(int(p) for p in Au)}  Av={sorted(int(p) for p in Av)}")

# Smallish model
model = ISIMACNeuralDecoder(d=16, hidden=64, n_layers=2, z_hidden=32, z_encoder_type='window').to(device)
print(f"Model params: {model.count_parameters():,}")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training: rate-1, teacher-forced (use true u,v)
def gen_batch(B):
    rng = np.random.default_rng()
    u = rng.integers(0, 2, (B, N)).astype(np.int32)
    v = rng.integers(0, 2, (B, N)).astype(np.int32)
    x = polar_encode_batch(u); y = polar_encode_batch(v)
    z = ch.sample_batch(x, y).astype(np.float32)
    return (torch.from_numpy(z).to(device),
            torch.from_numpy(u).to(device).long(),
            torch.from_numpy(v).to(device).long())

print("Training (rate-1, teacher-forced, 10000 iters, distill alpha 0.5)...", flush=True)
t0 = time.time()
N_ITERS = 10000
BATCH = 16
DISTILL_ALPHA = 0.5
losses, ce_losses, distill_losses = [], [], []
for it in range(1, N_ITERS + 1):
    z_t, u_t, v_t = gen_batch(BATCH)
    # rate-1 → no frozen positions
    root = model.encode_z(z_t)
    all_logits, all_targets, _, _, distill = model.tree(
        z=None, b=b, frozen_u={}, frozen_v={},
        u_true=u_t, v_true=v_t, root_emb=root, distill_alpha=DISTILL_ALPHA)
    if not all_logits:
        continue
    logits = torch.stack(all_logits, dim=1)  # (B, 2N, 4)
    targets = torch.stack(all_targets, dim=1)  # (B, 2N)
    ce = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
    loss = ce + DISTILL_ALPHA * distill
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item()); ce_losses.append(ce.item()); distill_losses.append(float(distill))
    if it % 500 == 0:
        elapsed = (time.time() - t0) / 60
        print(f"  iter {it}/{N_ITERS}  ce={np.mean(ce_losses[-200:]):.4f}  distill={np.mean(distill_losses[-200:]):.4f}  elapsed={elapsed:.1f}min", flush=True)

print(f"Training done in {(time.time()-t0)/60:.1f}min")

# Eval at target rate (Class C with GMAC frozen)
print("\nEval at target rate (5K CW)...")
model.eval()
n_cw = 5000
errs = 0
rng = np.random.default_rng(42)
t0 = time.time()
with torch.no_grad():
    for cw in range(n_cw):
        u_arr = np.zeros(N, dtype=np.int32); v_arr = np.zeros(N, dtype=np.int32)
        for p in Au: u_arr[int(p)-1] = rng.integers(0, 2)
        for p in Av: v_arr[int(p)-1] = rng.integers(0, 2)
        x = polar_encode_batch(u_arr.reshape(1,-1))[0]
        y = polar_encode_batch(v_arr.reshape(1,-1))[0]
        z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
        z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
        _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u=fu, frozen_v=fv)
        # u_hat is dict {1..N: tensor(B,)}
        ue = any(int(u_hat[int(p)].item()) != int(u_arr[int(p)-1]) for p in Au)
        ve = any(int(v_hat[int(p)].item()) != int(v_arr[int(p)-1]) for p in Av)
        if ue or ve: errs += 1
        if (cw+1) % 1000 == 0:
            print(f"  {cw+1}/{n_cw} errs={errs}  ({(time.time()-t0)/60:.1f}min)")

bler = errs / n_cw
print(f"\n=== Results ===")
print(f"N={N}, ku={ku}, kv={kv}, Class C corner-rate, h=0.3, SNR=6 dB")
print(f"NCG BLER = {bler:.4f}  ({errs}/{n_cw})")
print(f"Reference (chained NPD high-CW): 0.1665")
print(f"Reference (joint trellis SCT):   0.1501")
