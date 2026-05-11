#!/usr/bin/env python3
"""
Clean experiment: Does fast_ce pretraining help sequential decoding?
Three experiments at N=32, d=16, hidden=64, n_layers=2, SNR=6dB, Class B.

A) fast_ce only (50K iters)
B) Sequential from scratch (50K iters)
C) Hybrid: 15K fast_ce + 35K sequential
"""

import sys, os, math, time, copy
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file

# ── Hyperparameters ──
D = 16; HIDDEN = 64; N_LAYERS = 2
N = 32; n = 5
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
SC_BLER = 0.046  # reference SC BLER at this operating point
TOTAL_ITERS = 50_000; EVAL_EVERY = 5_000
N_EVAL_CW = 2000
SEED = 42

# ── Channel & design ──
channel = GaussianMAC(sigma2=SIGMA2)
mc_path = 'designs/gmac_B_n5_snr6dB.npz'
Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, 15, 15)
br_np = bit_reversal_perm(n)
br = torch.from_numpy(br_np).long()
fu_nat = {p-1 for p in range(1, N+1) if p not in Au}
fv_nat = {p-1 for p in range(1, N+1) if p not in Av}

print(f"N={N}, |Au|={len(Au)}, |Av|={len(Av)}, SNR={SNR_DB}dB, sigma2={SIGMA2:.4f}")
print(f"fu_nat ({len(fu_nat)} frozen): {sorted(fu_nat)}")
print(f"fv_nat ({len(fv_nat)} frozen): {sorted(fv_nat)}")

# ── Model ──
def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.checknode = _make_mlp(2*d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2*d, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def bitnode(self, e_odd, e_even, uv_left):
        u_left = uv_left // 2; v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([e_odd[:,:,:h]*u_sign, e_odd[:,:,h:]*v_sign], dim=-1)
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce_leaf_only(self, emb, joint_cw):
        B, N_, d = emb.shape; n_ = int(math.log2(N_))
        E_chunks = [emb]; J_chunks = [joint_cw]
        for depth in range(n_):
            E_odds, E_evens, J_odds, J_evens = [], [], [], []
            for e, j in zip(E_chunks, J_chunks):
                M = e.shape[1]
                E_odds.append(e.reshape(B, M//2, 2, d)[:,:,0,:])
                E_evens.append(e.reshape(B, M//2, 2, d)[:,:,1,:])
                J_odds.append(j.reshape(B, M//2, 2)[:,:,0])
                J_evens.append(j.reshape(B, M//2, 2)[:,:,1])
            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)
            J_odd = torch.cat(J_odds, 1); J_even = torch.cat(J_evens, 1)
            u_o = J_odd//2; v_o = J_odd%2; u_e = J_even//2; v_e = J_even%2
            J_left = (u_o^u_e)*2 + (v_o^v_e); J_right = J_even
            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)
            nc = 2**depth; cs = (N_//2)//nc
            el = torch.split(e_left, cs, 1); er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left, cs, 1); jr = torch.split(J_right, cs, 1)
            E_chunks = []; J_chunks = []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]; J_chunks += [c, dd]
        e_all = torch.cat(E_chunks, 1); j_all = torch.cat(J_chunks, 1)
        logits = self.emb2logits(e_all)
        return F.cross_entropy(logits.reshape(-1, 4), j_all.reshape(-1))

    def sequential_forward(self, emb, br_np, fu_nat, fv_nat, u_true_nat, v_true_nat):
        """Sequential tree walk with teacher forcing. Returns loss."""
        B = emb.shape[0]; N_ = emb.shape[1]
        br_t = torch.from_numpy(br_np).long()
        leaf_idx = [0]
        all_logits = []; all_targets = []

        def _decode(eb):
            bs = eb.shape[1]
            if bs == 1:
                logits = self.emb2logits(eb[:, 0, :])
                idx = leaf_idx[0]; leaf_idx[0] += 1
                nat_idx = int(br_t[idx])
                uf = nat_idx in fu_nat; vf = nat_idx in fv_nat
                if not (uf and vf):
                    all_logits.append(logits)
                    target = u_true_nat[:, nat_idx]*2 + v_true_nat[:, nat_idx]
                    all_targets.append(target)
                # Teacher forcing: use true bits
                dec = u_true_nat[:, nat_idx] * 2 + v_true_nat[:, nat_idx]
                return dec.unsqueeze(1)
            half = bs // 2
            e_odd = eb[:, 0::2, :]; e_even = eb[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            cw_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, cw_left)
            cw_right = _decode(e_right)
            u_l = cw_left//2; v_l = cw_left%2; u_r = cw_right//2; v_r = cw_right%2
            cw_odd = (u_l^u_r)*2 + (v_l^v_r); cw_even = cw_right
            result = torch.zeros(B, bs, dtype=torch.long)
            result[:, 0::2] = cw_odd; result[:, 1::2] = cw_even
            return result

        _decode(emb)
        if all_logits:
            loss = F.cross_entropy(torch.stack(all_logits, 1).reshape(-1, 4),
                                   torch.stack(all_targets, 1).reshape(-1))
        else:
            loss = torch.tensor(0.0)
        return loss

    def sc_decode_correct(self, emb, fu_nat, fv_nat, br_np):
        """Correct sequential decode (inference, no teacher forcing)."""
        B = emb.shape[0]; N_ = emb.shape[1]
        br_t = torch.from_numpy(br_np).long()
        u_hat = torch.zeros(B, N_, dtype=torch.long)
        v_hat = torch.zeros(B, N_, dtype=torch.long)
        leaf_idx = [0]

        def _decode(eb):
            bs = eb.shape[1]
            if bs == 1:
                logits = self.emb2logits(eb[:, 0, :])
                idx = leaf_idx[0]; leaf_idx[0] += 1
                nat_idx = int(br_t[idx])
                uf = nat_idx in fu_nat; vf = nat_idx in fv_nat
                if uf and vf:
                    dec = torch.zeros(B, dtype=torch.long)
                elif uf:
                    dec = (logits[:, 1] > logits[:, 0]).long()
                elif vf:
                    dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else:
                    dec = logits.argmax(dim=-1)
                u_hat[:, nat_idx] = dec // 2
                v_hat[:, nat_idx] = dec % 2
                return dec.unsqueeze(1)
            half = bs // 2
            e_odd = eb[:, 0::2, :]; e_even = eb[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            cw_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, cw_left)
            cw_right = _decode(e_right)
            u_l = cw_left//2; v_l = cw_left%2; u_r = cw_right//2; v_r = cw_right%2
            cw_odd = (u_l^u_r)*2 + (v_l^v_r); cw_even = cw_right
            result = torch.zeros(B, bs, dtype=torch.long)
            result[:, 0::2] = cw_odd; result[:, 1::2] = cw_even
            return result

        with torch.no_grad():
            _decode(emb)
        return u_hat, v_hat


# ── Data generation ──
def make_batch(rng, batch_size):
    """Generate a batch of encoded codewords and channel outputs."""
    uf = np.zeros((batch_size, N), dtype=int)
    vf = np.zeros((batch_size, N), dtype=int)
    for p in Au:
        uf[:, p-1] = rng.integers(0, 2, batch_size)
    for p in Av:
        vf[:, p-1] = rng.integers(0, 2, batch_size)
    xf = polar_encode_batch(uf)
    yf = polar_encode_batch(vf)
    zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
    # joint codeword for fast_ce (in codeword domain, bit-reversed)
    joint_cw = torch.from_numpy(xf * 2 + yf).long()
    # natural-order info bits for sequential
    u_nat = torch.from_numpy(uf).long()
    v_nat = torch.from_numpy(vf).long()
    return zf, joint_cw, u_nat, v_nat


# ── Evaluation (same for all experiments) ──
def evaluate(model, n_cw=N_EVAL_CW):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(32, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au:
                uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av:
                vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode_correct(emb, fu_nat, fv_nat, br_np)
            for i in range(actual):
                ue = any(u_dec[i, p-1].item() != uf[i, p-1] for p in Au)
                ve = any(v_dec[i, p-1].item() != vf[i, p-1] for p in Av)
                if ue or ve:
                    errs += 1
            total += actual
    model.train()
    return errs / total


# ── Training loops ──
def train_fast_ce(model, opt, n_iters, rng, label="fast_ce", eval_every=EVAL_EVERY,
                  start_iter=0, results=None):
    """Train with fast_ce leaf-only loss. batch=128."""
    if results is None:
        results = []
    model.train()
    t0 = time.time()
    for it in range(1, n_iters + 1):
        global_it = start_iter + it
        zf, joint_cw, _, _ = make_batch(rng, 128)
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        # joint_cw is already in codeword domain; bit-reverse for tree order
        joint_cw_br = joint_cw[:, br_np]
        loss = model.fast_ce_leaf_only(emb, joint_cw_br)
        opt.zero_grad(); loss.backward(); opt.step()
        if global_it % eval_every == 0:
            bler = evaluate(model)
            elapsed = (time.time() - t0) / 60
            ratio = bler / SC_BLER if bler > 0 else float('inf')
            print(f"  [{global_it:>5d}] {label} BLER={bler:.4f} ({ratio:.1f}x SC) "
                  f"loss={loss.item():.4f} {elapsed:.1f}min")
            results.append((global_it, bler, elapsed, label))
    return results


def train_sequential(model, opt, n_iters, rng, label="sequential",
                     eval_every=EVAL_EVERY, start_iter=0, results=None):
    """Train with sequential teacher-forced loss. batch=32."""
    if results is None:
        results = []
    model.train()
    t0 = time.time()
    for it in range(1, n_iters + 1):
        global_it = start_iter + it
        zf, _, u_nat, v_nat = make_batch(rng, 32)
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        loss = model.sequential_forward(emb, br_np, fu_nat, fv_nat, u_nat, v_nat)
        opt.zero_grad(); loss.backward(); opt.step()
        if global_it % eval_every == 0:
            bler = evaluate(model)
            elapsed = (time.time() - t0) / 60
            ratio = bler / SC_BLER if bler > 0 else float('inf')
            print(f"  [{global_it:>5d}] {label} BLER={bler:.4f} ({ratio:.1f}x SC) "
                  f"loss={loss.item():.4f} {elapsed:.1f}min")
            results.append((global_it, bler, elapsed, label))
    return results


# ══════════════════════════════════════════════════════════
#  Run all three experiments
# ══════════════════════════════════════════════════════════

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Experiment A: fast_ce only ──
print("\n" + "="*60)
print("=== Experiment A: fast_ce only (50K iters) ===")
print("="*60)
model_a = Decoder()
opt_a = torch.optim.Adam(model_a.parameters(), lr=3e-4)
rng_a = np.random.default_rng(SEED)
t_start_a = time.time()
results_a = train_fast_ce(model_a, opt_a, TOTAL_ITERS, rng_a, label="fast_ce")
wall_a = (time.time() - t_start_a) / 60
print(f"  Total wall time: {wall_a:.1f} min")

# ── Experiment B: Sequential from scratch ──
print("\n" + "="*60)
print("=== Experiment B: Sequential from scratch (50K iters) ===")
print("="*60)
torch.manual_seed(SEED)
model_b = Decoder()
opt_b = torch.optim.Adam(model_b.parameters(), lr=3e-4)
rng_b = np.random.default_rng(SEED)
t_start_b = time.time()
results_b = train_sequential(model_b, opt_b, TOTAL_ITERS, rng_b, label="sequential")
wall_b = (time.time() - t_start_b) / 60
print(f"  Total wall time: {wall_b:.1f} min")

# ── Experiment C: Hybrid (15K fast_ce + 35K sequential) ──
print("\n" + "="*60)
print("=== Experiment C: Hybrid (15K fast_ce + 35K sequential) ===")
print("="*60)
torch.manual_seed(SEED)
model_c = Decoder()
opt_c = torch.optim.Adam(model_c.parameters(), lr=3e-4)
rng_c = np.random.default_rng(SEED)
t_start_c = time.time()

# Phase 1: fast_ce, 15K iters
print("  --- Phase 1: fast_ce (15K) ---")
results_c = train_fast_ce(model_c, opt_c, 15_000, rng_c, label="fast_ce",
                          start_iter=0, results=[])

# Phase 2: sequential, 35K iters, lower LR
print("  --- Phase 2: sequential (35K), lr=1e-4 ---")
opt_c2 = torch.optim.Adam(model_c.parameters(), lr=1e-4)
results_c = train_sequential(model_c, opt_c2, 35_000, rng_c, label="sequential",
                             eval_every=EVAL_EVERY, start_iter=15_000,
                             results=results_c)
wall_c = (time.time() - t_start_c) / 60
print(f"  Total wall time: {wall_c:.1f} min")

# ══════════════════════════════════════════════════════════
#  Final comparison
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("=== FINAL COMPARISON ===")
print("="*60)

best_a = min(results_a, key=lambda x: x[1])
best_b = min(results_b, key=lambda x: x[1])
best_c = min(results_c, key=lambda x: x[1])

print(f"{'':20s} {'Best BLER':>10s} {'vs SC':>8s} {'Wall time':>10s} {'@ iter':>8s}")
print(f"{'fast_ce only:':20s} {best_a[1]:10.4f} {best_a[1]/SC_BLER:7.1f}x {wall_a:9.1f}m  @{best_a[0]:>5d}")
print(f"{'Sequential:':20s} {best_b[1]:10.4f} {best_b[1]/SC_BLER:7.1f}x {wall_b:9.1f}m  @{best_b[0]:>5d}")
print(f"{'Hybrid:':20s} {best_c[1]:10.4f} {best_c[1]/SC_BLER:7.1f}x {wall_c:9.1f}m  @{best_c[0]:>5d}")
print(f"{'SC reference:':20s} {SC_BLER:10.4f} {'1.0':>7s}x {'--':>10s}")

# Also print full curves for reference
print("\n--- Full curves ---")
for name, res in [("A (fast_ce)", results_a), ("B (sequential)", results_b), ("C (hybrid)", results_c)]:
    print(f"\n  {name}:")
    for it, bler, t, lbl in res:
        print(f"    [{it:>5d}] {lbl:12s} BLER={bler:.4f} ({bler/SC_BLER:.1f}x SC) {t:.1f}min")

print("\nDone.")
