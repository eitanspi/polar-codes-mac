"""Step 1: Stabilize NCG at N=16 on ISI-MAC h=0.3 SNR=6dB, Class C corner-rate.

Recipe based on observed instability:
- No self-distillation (caused explosion)
- Gradient clipping (max norm 1.0)
- Adam lr=1e-3, batch=32
- 50K iters at N=16 (~10 min CPU)
- Standard CE loss on 4-way logits

Target: match chained NPD's BLER ≈ 0.165 at N=16 within ~20%.
"""
import sys, os, time, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np
import torch
import torch.nn.functional as F

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode_batch
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_isi_mac import ISIMACNeuralDecoder

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cpu")

N, ku, kv, n = 16, 4, 7, 4
SIGMA2 = 10**(-0.6)
ch = ISIMAC(sigma2=SIGMA2, h=0.3)
b = make_path(N, N)

gmac_file = f"/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs/gmac_C_n{n}_snr6dB.npz"
Au, Av, fu_set, fv_set, _, _, _ = design_from_file(gmac_file, n, ku, kv)
Au = [int(p) for p in Au]; Av = [int(p) for p in Av]
fu = {int(p): 0 for p in fu_set}; fv = {int(p): 0 for p in fv_set}
print(f"Au={sorted(Au)}  Av={sorted(Av)}", flush=True)

model = ISIMACNeuralDecoder(d=16, hidden=64, n_layers=2, z_hidden=32, z_encoder_type='window').to(device)
print(f"Params: {model.count_parameters():,}", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def gen_batch(B):
    rng = np.random.default_rng()
    u = rng.integers(0, 2, (B, N)).astype(np.int32)
    v = rng.integers(0, 2, (B, N)).astype(np.int32)
    x = polar_encode_batch(u); y = polar_encode_batch(v)
    z = ch.sample_batch(x, y).astype(np.float32)
    return (torch.from_numpy(z), torch.from_numpy(u).long(), torch.from_numpy(v).long())

N_ITERS = 50_000
BATCH = 32
GRAD_CLIP = 1.0

print(f"Training (rate-1, teacher-forced, {N_ITERS} iters, batch={BATCH}, grad_clip={GRAD_CLIP}, no distill)...", flush=True)
t0 = time.time(); ce_losses = []
best_loss = float('inf')
for it in range(1, N_ITERS + 1):
    z_t, u_t, v_t = gen_batch(BATCH)
    # rate-1, teacher-forced
    root = model.encode_z(z_t)
    all_logits, all_targets, _, _, _ = model.tree(
        z=None, b=b, frozen_u={}, frozen_v={},
        u_true=u_t, v_true=v_t, root_emb=root, distill_alpha=0.0)
    logits = torch.stack(all_logits, dim=1).reshape(-1, 4)
    targets = torch.stack(all_targets, dim=1).reshape(-1)
    ce = F.cross_entropy(logits, targets)
    opt.zero_grad(); ce.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    ce_losses.append(ce.item())
    if it % 1000 == 0:
        avg = float(np.mean(ce_losses[-500:]))
        print(f"  iter {it}/{N_ITERS}  ce={avg:.4f}  best={best_loss:.4f}  elapsed={(time.time()-t0)/60:.1f}min", flush=True)
        if avg < best_loss: best_loss = avg

print(f"Training done in {(time.time()-t0)/60:.1f}min, final avg ce={float(np.mean(ce_losses[-500:])):.4f}", flush=True)

# Eval at target rate
print("\nEval at target rate (5K CW)...", flush=True)
model.eval()
n_cw = 5000
errs = 0
rng = np.random.default_rng(42)
t0 = time.time()
with torch.no_grad():
    for cw in range(n_cw):
        u_arr = np.zeros(N, dtype=np.int32); v_arr = np.zeros(N, dtype=np.int32)
        for p in Au: u_arr[p-1] = rng.integers(0, 2)
        for p in Av: v_arr[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u_arr.reshape(1,-1))[0]
        y = polar_encode_batch(v_arr.reshape(1,-1))[0]
        z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
        z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
        _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u=fu, frozen_v=fv)
        ue = any(int(u_hat[p].item()) != int(u_arr[p-1]) for p in Au)
        ve = any(int(v_hat[p].item()) != int(v_arr[p-1]) for p in Av)
        if ue or ve: errs += 1
        if (cw+1) % 1000 == 0:
            print(f"  {cw+1}/{n_cw} errs={errs} ({(time.time()-t0)/60:.1f}min)", flush=True)

bler = errs / n_cw
print(f"\n=== Step 1 results ===", flush=True)
print(f"N={N}: NCG BLER = {bler:.4f}  ({errs}/{n_cw})", flush=True)
print(f"Reference chained NPD: 0.1665", flush=True)
print(f"Reference joint trellis SCT: 0.1501", flush=True)

# Save model
os.makedirs("scripts/local_analysis/ncg_models", exist_ok=True)
torch.save({"state_dict": model.state_dict(),
            "config": dict(d=16, hidden=64, n_layers=2, z_hidden=32, z_encoder_type='window'),
            "N": N, "ku": ku, "kv": kv, "n": n,
            "bler": bler, "errs": errs, "n_cw": n_cw,
            "iters": N_ITERS},
           "scripts/local_analysis/ncg_models/ncg_isi_N16_step1.pt")
print("Saved checkpoint.", flush=True)
