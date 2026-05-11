#!/usr/bin/env python3
"""Test Ideas 5 & 6: Loss placement variants and equivariant BitNode."""
import sys, os, math, time
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file

SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)

# ── Standard Decoder (for Ideas 5 and 6-baseline) ──────────────────────────
class Decoder(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2):
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
            J_left = (u_o^u_e)*2+(v_o^v_e); J_right = J_even
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

    def fast_ce_all_depths(self, emb, joint_cw):
        B, N_, d = emb.shape; n_ = int(math.log2(N_))
        all_losses = []
        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 4), joint_cw.reshape(-1)))
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
            J_left = (u_o^u_e)*2+(v_o^v_e); J_right = J_even
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
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4), j_all.reshape(-1)))
        return torch.stack(all_losses).mean()

    def fast_ce_weighted(self, emb, joint_cw):
        B, N_, d = emb.shape; n_ = int(math.log2(N_))
        all_losses = []; weights = []
        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 4), joint_cw.reshape(-1)))
        weights.append(0.1)
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
            J_left = (u_o^u_e)*2+(v_o^v_e); J_right = J_even
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
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4), j_all.reshape(-1)))
            w = 2.0 ** depth
            weights.append(w)
        losses_t = torch.stack(all_losses)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        weights_t = weights_t / weights_t.sum()
        return (losses_t * weights_t).sum()

    def sc_decode_correct(self, emb, fu_nat, fv_nat, br_np):
        B = emb.shape[0]; N_ = emb.shape[1]
        br_t = torch.from_numpy(br_np).long()
        u_hat = torch.zeros(B, N_, dtype=torch.long); v_hat = torch.zeros(B, N_, dtype=torch.long)
        leaf_idx = [0]
        def _decode(eb):
            bs = eb.shape[1]
            if bs == 1:
                logits = self.emb2logits(eb[:, 0, :])
                idx = leaf_idx[0]; leaf_idx[0] += 1
                nat_idx = int(br_t[idx])
                uf = nat_idx in fu_nat; vf = nat_idx in fv_nat
                if uf and vf: dec = torch.zeros(B, dtype=torch.long)
                elif uf: dec = (logits[:, 1] > logits[:, 0]).long()
                elif vf: dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else: dec = logits.argmax(dim=-1)
                u_hat[:, nat_idx] = dec // 2; v_hat[:, nat_idx] = dec % 2
                return dec.unsqueeze(1)
            half = bs // 2
            e_odd = eb[:, 0::2, :]; e_even = eb[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            cw_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, cw_left)
            cw_right = _decode(e_right)
            u_l = cw_left//2; v_l = cw_left%2; u_r = cw_right//2; v_r = cw_right%2
            cw_odd = (u_l^u_r)*2+(v_l^v_r); cw_even = cw_right
            result = torch.zeros(B, bs, dtype=torch.long)
            result[:, 0::2] = cw_odd; result[:, 1::2] = cw_even
            return result
        with torch.no_grad(): _decode(emb)
        return u_hat, v_hat


# ── Equivariant Decoder (Idea 6B: separate u/v pathways) ───────────────────
class DecoderEquivariant(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2):
        super().__init__()
        self.d = d
        h = d // 2
        self.z_encoder = nn.Sequential(nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.checknode = _make_mlp(2*d, hidden, d, n_layers)
        self.bitnode_u = _make_mlp(2*h, hidden, h, n_layers)
        self.bitnode_v = _make_mlp(2*h, hidden, h, n_layers)
        self.bitnode_interact = _make_mlp(d, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def bitnode(self, e_odd, e_even, uv_left):
        u_left = uv_left // 2; v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_u = e_odd[:,:,:h] * u_sign
        e_v = e_odd[:,:,h:] * v_sign
        e_u_proc = self.bitnode_u(torch.cat([e_u, e_even[:,:,:h]], -1))
        e_v_proc = self.bitnode_v(torch.cat([e_v, e_even[:,:,h:]], -1))
        combined = torch.cat([e_u_proc, e_v_proc], -1)
        out = self.bitnode_interact(combined)
        return out + torch.cat([e_u, e_v], -1) + e_even

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
            J_left = (u_o^u_e)*2+(v_o^v_e); J_right = J_even
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

    def sc_decode_correct(self, emb, fu_nat, fv_nat, br_np):
        B = emb.shape[0]; N_ = emb.shape[1]
        br_t = torch.from_numpy(br_np).long()
        u_hat = torch.zeros(B, N_, dtype=torch.long); v_hat = torch.zeros(B, N_, dtype=torch.long)
        leaf_idx = [0]
        def _decode(eb):
            bs = eb.shape[1]
            if bs == 1:
                logits = self.emb2logits(eb[:, 0, :])
                idx = leaf_idx[0]; leaf_idx[0] += 1
                nat_idx = int(br_t[idx])
                uf = nat_idx in fu_nat; vf = nat_idx in fv_nat
                if uf and vf: dec = torch.zeros(B, dtype=torch.long)
                elif uf: dec = (logits[:, 1] > logits[:, 0]).long()
                elif vf: dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else: dec = logits.argmax(dim=-1)
                u_hat[:, nat_idx] = dec // 2; v_hat[:, nat_idx] = dec % 2
                return dec.unsqueeze(1)
            half = bs // 2
            e_odd = eb[:, 0::2, :]; e_even = eb[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            cw_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, cw_left)
            cw_right = _decode(e_right)
            u_l = cw_left//2; v_l = cw_left%2; u_r = cw_right//2; v_r = cw_right%2
            cw_odd = (u_l^u_r)*2+(v_l^v_r); cw_even = cw_right
            result = torch.zeros(B, bs, dtype=torch.long)
            result[:, 0::2] = cw_odd; result[:, 1::2] = cw_even
            return result
        with torch.no_grad(): _decode(emb)
        return u_hat, v_hat


# ── Interaction-term Decoder (Idea 6C: 3d input) ───────────────────────────
class DecoderInteraction(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.checknode = _make_mlp(2*d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(3*d, hidden, d, n_layers)  # 3d input
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def bitnode(self, e_odd, e_even, uv_left):
        u_left = uv_left // 2; v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([e_odd[:,:,:h]*u_sign, e_odd[:,:,h:]*v_sign], dim=-1)
        interaction = e_signed * e_even
        inp = torch.cat([e_signed, e_even, interaction], dim=-1)
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
            J_left = (u_o^u_e)*2+(v_o^v_e); J_right = J_even
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

    def sc_decode_correct(self, emb, fu_nat, fv_nat, br_np):
        B = emb.shape[0]; N_ = emb.shape[1]
        br_t = torch.from_numpy(br_np).long()
        u_hat = torch.zeros(B, N_, dtype=torch.long); v_hat = torch.zeros(B, N_, dtype=torch.long)
        leaf_idx = [0]
        def _decode(eb):
            bs = eb.shape[1]
            if bs == 1:
                logits = self.emb2logits(eb[:, 0, :])
                idx = leaf_idx[0]; leaf_idx[0] += 1
                nat_idx = int(br_t[idx])
                uf = nat_idx in fu_nat; vf = nat_idx in fv_nat
                if uf and vf: dec = torch.zeros(B, dtype=torch.long)
                elif uf: dec = (logits[:, 1] > logits[:, 0]).long()
                elif vf: dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else: dec = logits.argmax(dim=-1)
                u_hat[:, nat_idx] = dec // 2; v_hat[:, nat_idx] = dec % 2
                return dec.unsqueeze(1)
            half = bs // 2
            e_odd = eb[:, 0::2, :]; e_even = eb[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            cw_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, cw_left)
            cw_right = _decode(e_right)
            u_l = cw_left//2; v_l = cw_left%2; u_r = cw_right//2; v_r = cw_right%2
            cw_odd = (u_l^u_r)*2+(v_l^v_r); cw_even = cw_right
            result = torch.zeros(B, bs, dtype=torch.long)
            result[:, 0::2] = cw_odd; result[:, 1::2] = cw_even
            return result
        with torch.no_grad(): _decode(emb)
        return u_hat, v_hat


# ── Setup ───────────────────────────────────────────────────────────────────
N = 32; n = 5; BATCH = 128
channel = GaussianMAC(sigma2=SIGMA2)
mc_path = f'designs/gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz'
Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, 15, 15)
br_np = bit_reversal_perm(n); br = torch.from_numpy(br_np).long()
fu_nat = {p-1 for p in range(1, N+1) if p not in Au}
fv_nat = {p-1 for p in range(1, N+1) if p not in Av}

SC_BLER = 0.046

def evaluate(model, n_cw=2000):
    model.eval(); errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(32, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode_correct(emb, fu_nat, fv_nat, br_np)
            for i in range(actual):
                ue = any(u_dec[i, p-1].item() != uf[i, p-1] for p in Au)
                ve = any(v_dec[i, p-1].item() != vf[i, p-1] for p in Av)
                if ue or ve: errs += 1
            total += actual
    model.train()
    return errs / total

def train_model(model, loss_fn_name, iters, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng()
    model.train()
    for it in range(1, iters+1):
        uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        joint_cw = torch.from_numpy(xf*2+yf).long()[:, br]
        if loss_fn_name == 'leaf_only':
            loss = model.fast_ce_leaf_only(emb, joint_cw)
        elif loss_fn_name == 'all_depths':
            loss = model.fast_ce_all_depths(emb, joint_cw)
        elif loss_fn_name == 'weighted':
            loss = model.fast_ce_weighted(emb, joint_cw)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 10000 == 0:
            print(f'    [{it}/{iters}] loss={loss.item():.4f}', flush=True)

ITERS = 50000

# ── IDEA 5: Loss placement ─────────────────────────────────────────────────
print("=" * 60)
print("IDEA 5: Loss placement comparison")
print("=" * 60)

results_5 = {}
label_map = {'leaf_only': 'A', 'all_depths': 'B', 'weighted': 'C'}
for name, loss_fn in [('leaf_only', 'leaf_only'), ('all_depths', 'all_depths'), ('weighted', 'weighted')]:
    print(f"\n--- 5{label_map[name]}] {name}, {ITERS//1000}K iters ---")
    t0 = time.time()
    model = Decoder(d=16, hidden=64, n_layers=2)
    train_model(model, loss_fn, ITERS)
    bler = evaluate(model)
    elapsed = time.time() - t0
    ratio = bler / SC_BLER
    results_5[name] = (bler, ratio)
    print(f"  BLER={bler:.4f} ({ratio:.1f}x SC), time={elapsed:.0f}s")

# ── IDEA 6: Equivariant BitNode ────────────────────────────────────────────
print("\n" + "=" * 60)
print("IDEA 6: Equivariant BitNode comparison")
print("=" * 60)

results_6 = {}

# 6A: Standard (baseline, same as Idea 5A leaf_only)
print(f"\n--- 6A] Standard sign-based (baseline), {ITERS//1000}K iters ---")
t0 = time.time()
model_6a = Decoder(d=16, hidden=64, n_layers=2)
train_model(model_6a, 'leaf_only', ITERS)
bler = evaluate(model_6a)
elapsed = time.time() - t0
ratio = bler / SC_BLER
results_6['standard'] = (bler, ratio)
print(f"  BLER={bler:.4f} ({ratio:.1f}x SC), time={elapsed:.0f}s")

# 6B: Equivariant separate pathways
print(f"\n--- 6B] Separate u/v pathways, {ITERS//1000}K iters ---")
t0 = time.time()
model_6b = DecoderEquivariant(d=16, hidden=64, n_layers=2)
train_model(model_6b, 'leaf_only', ITERS)
bler = evaluate(model_6b)
elapsed = time.time() - t0
ratio = bler / SC_BLER
results_6['equivariant'] = (bler, ratio)
print(f"  BLER={bler:.4f} ({ratio:.1f}x SC), time={elapsed:.0f}s")

# 6C: Interaction term (3d input)
print(f"\n--- 6C] Interaction term (3d input), {ITERS//1000}K iters ---")
t0 = time.time()
model_6c = DecoderInteraction(d=16, hidden=64, n_layers=2)
train_model(model_6c, 'leaf_only', ITERS)
bler = evaluate(model_6c)
elapsed = time.time() - t0
ratio = bler / SC_BLER
results_6['interaction'] = (bler, ratio)
print(f"  BLER={bler:.4f} ({ratio:.1f}x SC), time={elapsed:.0f}s")

# ── Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"\nSC reference BLER = {SC_BLER}")

print(f"\n=== IDEA 5: Loss placement ===")
for label, key in [('A) leaf_only', 'leaf_only'), ('B) all_depths', 'all_depths'), ('C) weighted', 'weighted')]:
    b, r = results_5[key]
    print(f"{label}, 50K: BLER={b:.4f} ({r:.1f}x SC)")

print(f"\n=== IDEA 6: Equivariant BitNode ===")
for label, key in [('A) Standard sign-based (baseline)', 'standard'),
                    ('B) Separate u/v pathways', 'equivariant'),
                    ('C) Interaction term (3d input)', 'interaction')]:
    b, r = results_6[key]
    print(f"{label}: BLER={b:.4f} ({r:.1f}x SC)")
