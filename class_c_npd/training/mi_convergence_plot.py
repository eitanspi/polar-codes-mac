#!/usr/bin/env python3
"""Train Stage 1 NPD from scratch at rate 1, log per-position MI every 500 iters."""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import torch
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from class_c_npd.models.npd_single_user import NPDSingleUser, npd_encode
from class_c_npd.training.train_stage import generate_stage1_batch
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design

N = 32
n = 5
BATCH = 64
ITERS = 30000
EVAL_EVERY = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

ch = build_channel("gmac", sigma2=10**(-6.0/10))
Au, Av, fu, fv, pe_u, pe_v = load_class_c_design("gmac", n, snr_db=6.0, ku=None, kv=None)
br = np.array(bit_reversal_perm(n))
print(f"N={N}, ku={len(Au)}, kv={len(Av)}", flush=True)


@torch.no_grad()
def measure_mi_per_pos(model, ch, N, device, n_samples=2000, batch=100):
    nn = int(math.log2(N))
    br_l = np.array(bit_reversal_perm(nn))
    rng2 = np.random.default_rng(789)
    leaf_bce = np.zeros(N)
    count = 0
    model.eval()
    while count < n_samples:
        actual = min(batch, n_samples - count)
        # Rate 1: all info
        u = rng2.integers(0, 2, (actual, N)).astype(int)
        x = polar_encode_batch(u)
        v = rng2.integers(0, 2, (actual, N)).astype(int)
        y = polar_encode_batch(v)
        z = ch.sample_z(x, y)
        features = ch.stage1_features(z)
        ft = torch.from_numpy(features.astype(np.float32)).to(device)
        if ft.dim() == 2:
            ft = ft.unsqueeze(-1)
        emb = model.encode_channel(ft)
        # NPD codeword
        x_npd = npd_encode(u)
        cw = torch.from_numpy(x_npd).long().to(device)
        B, N_, d = emb.shape
        # Run analytical tree to leaves
        E_scalar = emb[:, :, 0]
        V = [cw]
        E = [E_scalar]
        for depth in range(nn):
            Vo, Ve, Eo, Ee = [], [], [], []
            for vc, ec in zip(V, E):
                Vo.append(vc[:, 0::2]); Ve.append(vc[:, 1::2])
                Eo.append(ec[:, 0::2]); Ee.append(ec[:, 1::2])
            Vo = torch.cat(Vo, 1); Ve = torch.cat(Ve, 1)
            Eo = torch.cat(Eo, 1); Ee = torch.cat(Ee, 1)
            vt = Vo ^ Ve; vb = Ve
            nc = 2 ** depth; cs = (N_ // 2) // nc
            vtc = torch.split(vt, cs, 1); vbc = torch.split(vb, cs, 1)
            Vn = []
            for a, b in zip(vtc, vbc):
                Vn += [a, b]
            Vl = torch.cat(Vn[0::2], 1)
            et = model.analytical_checknode(Eo, Ee)
            eb = model.analytical_bitnode(Eo, Ee, Vl)
            etc = torch.split(et, cs, 1); ebc = torch.split(eb, cs, 1)
            En = []
            for a, b in zip(etc, ebc):
                En += [a, b]
            V = Vn; E = En
        e_leaf = torch.cat(E, 1)
        v_leaf = torch.cat(V, 1)
        bce = F.binary_cross_entropy_with_logits(
            e_leaf, v_leaf.float(), reduction="none")
        leaf_bce += bce.sum(0).detach().cpu().numpy()
        count += actual
    model.train()
    avg_bce = leaf_bce / count
    bce_nat = np.zeros(N)
    for t in range(N):
        bce_nat[br_l[t]] = avg_bce[t]
    mi_nat = np.clip((np.log(2) - bce_nat) / np.log(2), 0, 1)
    return mi_nat


# Model
model = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=1,
                       use_analytical_training=True).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
rng = np.random.default_rng(42)

results = {"N": N, "iters": [], "mi_per_pos": [], "mi_avg": [], "loss": []}
t0 = time.time()

model.train()
for it in range(1, ITERS + 1):
    # Rate 1 Stage 1 training
    u = rng.integers(0, 2, (BATCH, N)).astype(int)
    x = polar_encode_batch(u)
    v = rng.integers(0, 2, (BATCH, N)).astype(int)
    y = polar_encode_batch(v)
    z = ch.sample_z(x, y)
    features = ch.stage1_features(z)
    ft = torch.from_numpy(features.astype(np.float32)).to(device)
    if ft.dim() == 2:
        ft = ft.unsqueeze(-1)
    x_npd = npd_encode(u)
    cw = torch.from_numpy(x_npd).long().to(device)
    emb = model.encode_channel(ft)
    loss = model.fast_ce(emb, cw)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if it % EVAL_EVERY == 0 or it == 1:
        mi = measure_mi_per_pos(model, ch, N, device, n_samples=2000)
        results["iters"].append(it)
        results["mi_per_pos"].append(mi.tolist())
        results["mi_avg"].append(float(mi.mean()))
        results["loss"].append(loss.item())
        elapsed = (time.time() - t0) / 60
        print(f"[{it:>5}/{ITERS}] loss={loss.item():.4f} "
              f"MI_avg={mi.mean():.4f} MI_min={mi.min():.4f} "
              f"MI_max={mi.max():.4f} {elapsed:.1f}min", flush=True)

with open("class_c_npd/results/mi_convergence_N32.json", "w") as f:
    json.dump(results, f)
print("Done.", flush=True)
