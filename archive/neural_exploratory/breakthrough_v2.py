#!/usr/bin/env python3
"""
breakthrough_v2.py — Fixed experiments with proper fast_ce/SC consistency.

Key insight: fast_ce and SC decode MUST use the exact same neural network
functions. The previous attempt used calc_left_nn(e_odd, e_even, zeros) for
fast_ce but calc_left_nn(parent[:l], parent[l:], right) for sequential.
These are completely different input patterns.

Solution: Use proper NPD-style architecture where CheckNode and BitNode
are the fundamental operations used in BOTH fast_ce and SC decode.
"""
import sys, os, math, time, traceback, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file

D = 16
HIDDEN = 64
N_LAYERS = 2
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
BATCH = 128
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'breakthrough_agent.log')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        return design_from_file(mc_path, n, ku, kv)
    from polar.design import design_gmac
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv, None, None, None


def gen_batch(Au, Av, N, channel, rng, batch=BATCH):
    uf = np.zeros((batch, N), dtype=int)
    vf = np.zeros((batch, N), dtype=int)
    for p in Au: uf[:, p - 1] = rng.integers(0, 2, batch)
    for p in Av: vf[:, p - 1] = rng.integers(0, 2, batch)
    xf = polar_encode_batch(uf)
    yf = polar_encode_batch(vf)
    zf = channel.sample_batch(xf, yf)
    return uf, vf, xf, yf, zf


# ══════════════════════════════════════════════════════════════════════════════
#  Consistent NPD-style MAC decoder (fast_ce + SC decode use same ops)
# ══════════════════════════════════════════════════════════════════════════════

class ConsistentMACDecoder(nn.Module):
    """4-class MAC decoder where fast_ce and SC decode use the exact same
    CheckNode and BitNode operations.

    Architecture:
    - z_encoder: Linear(1,32)->ELU->Linear(32,d)
    - checknode: MLP(2d -> hidden -> d) -- like NPD single-user
    - bitnode: MLP(2d -> hidden -> d) with joint-class conditioning
    - emb2logits: MLP(d -> hidden -> 4)
    """

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d)
        )
        # CheckNode: (e_odd, e_even) -> e_left
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        # BitNode: (e_odd * joint_sign, e_even) -> e_right + residual
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        # Decision head: embedding -> 4-class logits
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def bitnode(self, e_odd, e_even, uv_left):
        """BitNode with joint-class sign conditioning.
        uv_left: (B, M) integer 0-3, encoding (u,v) as u*2+v.

        We use sign conditioning: split d into 2 halves.
        First half: e_odd * u_sign, Second half: e_odd * v_sign.
        """
        u_left = uv_left // 2
        v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)  # 0->+1, 1->-1
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([e_odd[:, :, :h] * u_sign,
                              e_odd[:, :, h:] * v_sign], dim=-1)
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce(self, emb, joint_cw):
        """Parallel fast_ce for 4-class MAC.
        emb: (B, N, d), joint_cw: (B, N) in {0,1,2,3}."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 4),
                                          joint_cw.reshape(-1)))

        E_chunks = [emb]
        J_chunks = [joint_cw]

        for depth in range(n):
            E_odds, E_evens, J_odds, J_evens = [], [], [], []
            for e, j in zip(E_chunks, J_chunks):
                M = e.shape[1]
                E_odds.append(e.reshape(B, M // 2, 2, d)[:, :, 0, :])
                E_evens.append(e.reshape(B, M // 2, 2, d)[:, :, 1, :])
                J_odds.append(j.reshape(B, M // 2, 2)[:, :, 0])
                J_evens.append(j.reshape(B, M // 2, 2)[:, :, 1])

            E_odd = torch.cat(E_odds, 1)
            E_even = torch.cat(E_evens, 1)
            J_odd = torch.cat(J_odds, 1)
            J_even = torch.cat(J_evens, 1)

            u_o, v_o = J_odd // 2, J_odd % 2
            u_e, v_e = J_even // 2, J_even % 2
            J_left = (u_o ^ u_e) * 2 + (v_o ^ v_e)
            J_right = J_even

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left, cs, 1)
            jr = torch.split(J_right, cs, 1)

            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

            e_all = torch.cat(E_chunks, 1)
            j_all = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_all)
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4),
                                              j_all.reshape(-1)))

        return torch.stack(all_losses).mean()

    def fast_ce_scheduled_sampling(self, emb, joint_cw, p_sample=0.0):
        """Fast_ce with scheduled sampling: with probability p_sample,
        use model's own predictions instead of true values for conditioning."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 4),
                                          joint_cw.reshape(-1)))

        E_chunks = [emb]
        J_chunks = [joint_cw]  # true joint codeword

        for depth in range(n):
            E_odds, E_evens, J_odds, J_evens = [], [], [], []
            for e, j in zip(E_chunks, J_chunks):
                M = e.shape[1]
                E_odds.append(e.reshape(B, M // 2, 2, d)[:, :, 0, :])
                E_evens.append(e.reshape(B, M // 2, 2, d)[:, :, 1, :])
                J_odds.append(j.reshape(B, M // 2, 2)[:, :, 0])
                J_evens.append(j.reshape(B, M // 2, 2)[:, :, 1])

            E_odd = torch.cat(E_odds, 1)
            E_even = torch.cat(E_evens, 1)
            J_odd = torch.cat(J_odds, 1)
            J_even = torch.cat(J_evens, 1)

            u_o, v_o = J_odd // 2, J_odd % 2
            u_e, v_e = J_even // 2, J_even % 2
            J_left_true = (u_o ^ u_e) * 2 + (v_o ^ v_e)
            J_right = J_even

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))

            # Scheduled sampling: use model's predictions for J_left
            if p_sample > 0 and self.training:
                with torch.no_grad():
                    left_logits = self.emb2logits(e_left)
                    J_left_pred = left_logits.argmax(dim=-1)
                mask = (torch.rand(B, J_left_true.shape[1]) < p_sample).to(J_left_true.device)
                J_left = torch.where(mask, J_left_pred, J_left_true)
            else:
                J_left = J_left_true

            e_right = self.bitnode(E_odd, E_even, J_left)

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            # Targets are always true values
            jl = torch.split(J_left_true, cs, 1)
            jr = torch.split(J_right, cs, 1)

            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

            e_all = torch.cat(E_chunks, 1)
            j_all = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_all)
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4),
                                              j_all.reshape(-1)))

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def sc_decode(self, emb, frozen_u, frozen_v):
        """Sequential SC decode using the SAME checknode/bitnode."""
        B, N = emb.shape[0], emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(emb_block):
            block_size = emb_block.shape[1]
            if block_size == 1:
                logits = self.emb2logits(emb_block[:, 0, :])
                idx = leaf_idx[0]
                leaf_idx[0] += 1

                u_frozen = idx in frozen_u
                v_frozen = idx in frozen_v

                if u_frozen and v_frozen:
                    dec = torch.zeros(B, dtype=torch.long)
                elif u_frozen:
                    dec = (logits[:, 1] > logits[:, 0]).long()
                elif v_frozen:
                    dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else:
                    dec = logits.argmax(dim=-1)

                u_hat[:, idx] = dec // 2
                v_hat[:, idx] = dec % 2
                return dec.unsqueeze(1)

            e_odd = emb_block[:, 0::2, :]
            e_even = emb_block[:, 1::2, :]

            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            uv_left = _decode(e_left)

            e_right = self.bitnode(e_odd, e_even, uv_left)
            uv_right = _decode(e_right)

            # Reconstruct parent codeword
            u_l, v_l = uv_left // 2, uv_left % 2
            u_r, v_r = uv_right // 2, uv_right % 2
            joint = torch.zeros(B, block_size, dtype=torch.long)
            joint[:, 0::2] = (u_l ^ u_r) * 2 + (v_l ^ v_r)
            joint[:, 1::2] = u_r * 2 + v_r
            return joint

        _decode(emb)
        return u_hat, v_hat


def evaluate_bler(model, channel, N, Au, Av, fu, fv, n_cw=500):
    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}
    n = int(math.log2(N))
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    errs = 0
    rng = np.random.default_rng(999)
    model.eval()
    with torch.no_grad():
        for _ in range(n_cw):
            uf = np.zeros((1, N), dtype=int)
            vf = np.zeros((1, N), dtype=int)
            for p in Au: uf[0, p - 1] = rng.integers(0, 2)
            for p in Av: vf[0, p - 1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode(emb, fu_set, fv_set)
            ue = any(u_dec[0, p - 1].item() != uf[0, p - 1] for p in Au)
            ve = any(v_dec[0, p - 1].item() != vf[0, p - 1] for p in Av)
            if ue or ve:
                errs += 1
    model.train()
    return errs / n_cw


def evaluate_bler_batch(model, channel, N, Au, Av, fu, fv, n_cw=1000,
                        batch_size=50):
    """Batched BLER evaluation for speed."""
    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}
    n = int(math.log2(N))
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    errs = 0
    rng = np.random.default_rng(999)
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, n_cw, batch_size):
            bs = min(batch_size, n_cw - batch_start)
            uf = np.zeros((bs, N), dtype=int)
            vf = np.zeros((bs, N), dtype=int)
            for p in Au: uf[:, p - 1] = rng.integers(0, 2, bs)
            for p in Av: vf[:, p - 1] = rng.integers(0, 2, bs)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode(emb, fu_set, fv_set)
            for i in range(bs):
                ue = any(u_dec[i, p - 1].item() != uf[i, p - 1] for p in Au)
                ve = any(v_dec[i, p - 1].item() != vf[i, p - 1] for p in Av)
                if ue or ve:
                    errs += 1
    model.train()
    return errs / n_cw


# ══════════════════════════════════════════════════════════════════════════════
#  Binary Decomposition Decoder (separate u and v|u)
# ══════════════════════════════════════════════════════════════════════════════

class BinaryDecompDecoder(nn.Module):
    """Two-stage: decode u (binary NPD), then v|u (binary NPD conditioned on u).
    Both use fast_ce for O(log N) training."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d)
        )

        # U decoder
        self.u_checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.u_bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.u_emb2llr = _make_mlp(d, hidden, 1, n_layers)

        # V|U decoder
        self.v_checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.v_bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.v_emb2llr = _make_mlp(d, hidden, 1, n_layers)

        # U conditioning for V decoder
        self.u_cond = nn.Sequential(
            nn.Linear(d + 1, hidden), nn.ELU(), nn.Linear(hidden, d)
        )

    def u_bitnode(self, e_odd, e_even, u_hard):
        if u_hard.dim() == 2:
            u_hard = u_hard.unsqueeze(-1)
        u_sign = 2.0 * u_hard.float() - 1.0
        u_sign = u_sign.expand_as(e_odd)
        e_signed = e_odd * u_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.u_bitnode_mlp(inp) + e_signed + e_even

    def v_bitnode(self, e_odd, e_even, v_hard):
        if v_hard.dim() == 2:
            v_hard = v_hard.unsqueeze(-1)
        v_sign = 2.0 * v_hard.float() - 1.0
        v_sign = v_sign.expand_as(e_odd)
        e_signed = e_odd * v_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.v_bitnode_mlp(inp) + e_signed + e_even

    def fast_ce_u(self, emb, u_cw):
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        pred = self.u_emb2llr(emb).squeeze(-1)
        all_losses.append(F.binary_cross_entropy_with_logits(pred, u_cw.float()))

        V, E = [u_cw], [emb]
        for depth in range(n):
            V_odds, V_evens, E_odds, E_evens = [], [], [], []
            for v, e in zip(V, E):
                V_odds.append(v[:, 0::2]); V_evens.append(v[:, 1::2])
                E_odds.append(e[:, 0::2, :]); E_evens.append(e[:, 1::2, :])
            V_odd = torch.cat(V_odds, 1); V_even = torch.cat(V_evens, 1)
            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)

            v_xor = V_odd ^ V_even; v_id = V_even
            e_left = self.u_checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.u_bitnode(E_odd, E_even, v_xor)

            nc = 2 ** depth; cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1); er = torch.split(e_right, cs, 1)
            vl = torch.split(v_xor, cs, 1); vr = torch.split(v_id, cs, 1)
            V_new, E_new = [], []
            for a, b, c, dd in zip(el, er, vl, vr):
                E_new += [a, b]; V_new += [c, dd]

            e_all = torch.cat(E_new, 1); v_all = torch.cat(V_new, 1)
            pred = self.u_emb2llr(e_all).squeeze(-1)
            all_losses.append(F.binary_cross_entropy_with_logits(pred, v_all.float()))
            V = V_new; E = E_new

        return torch.stack(all_losses).mean()

    def fast_ce_v(self, emb, v_cw, u_cw):
        """Fast CE for v|u. Condition embedding on true u codeword."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        # Condition on u
        emb_v = self.u_cond(torch.cat([emb, u_cw.float().unsqueeze(-1)], -1))

        pred = self.v_emb2llr(emb_v).squeeze(-1)
        all_losses.append(F.binary_cross_entropy_with_logits(pred, v_cw.float()))

        V, E, U = [v_cw], [emb_v], [u_cw]
        for depth in range(n):
            V_odds, V_evens, E_odds, E_evens, U_odds, U_evens = [], [], [], [], [], []
            for v, e, u in zip(V, E, U):
                V_odds.append(v[:, 0::2]); V_evens.append(v[:, 1::2])
                E_odds.append(e[:, 0::2, :]); E_evens.append(e[:, 1::2, :])
                U_odds.append(u[:, 0::2]); U_evens.append(u[:, 1::2])
            V_odd = torch.cat(V_odds, 1); V_even = torch.cat(V_evens, 1)
            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)
            U_odd = torch.cat(U_odds, 1); U_even = torch.cat(U_evens, 1)

            v_xor = V_odd ^ V_even; v_id = V_even
            u_xor = U_odd ^ U_even; u_id = U_even

            e_left = self.v_checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.v_bitnode(E_odd, E_even, v_xor)

            nc = 2 ** depth; cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1); er = torch.split(e_right, cs, 1)
            vl = torch.split(v_xor, cs, 1); vr = torch.split(v_id, cs, 1)
            ul = torch.split(u_xor, cs, 1); ur = torch.split(u_id, cs, 1)
            V_new, E_new, U_new = [], [], []
            for a, b, c, dd, uu1, uu2 in zip(el, er, vl, vr, ul, ur):
                E_new += [a, b]; V_new += [c, dd]; U_new += [uu1, uu2]

            e_all = torch.cat(E_new, 1); v_all = torch.cat(V_new, 1)
            pred = self.v_emb2llr(e_all).squeeze(-1)
            all_losses.append(F.binary_cross_entropy_with_logits(pred, v_all.float()))
            V = V_new; E = E_new; U = U_new

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def sc_decode(self, emb, frozen_u, frozen_v):
        B, N = emb.shape[0], emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)

        # Decode u
        u_idx = [0]
        def _decode_u(e_block):
            block_size = e_block.shape[1]
            if block_size == 1:
                llr = self.u_emb2llr(e_block[:, 0, :]).squeeze(-1)
                idx = u_idx[0]; u_idx[0] += 1
                dec = torch.zeros(B, dtype=torch.long) if idx in frozen_u \
                      else (llr > 0).long()
                u_hat[:, idx] = dec
                return dec.unsqueeze(1)
            e_odd = e_block[:, 0::2, :]; e_even = e_block[:, 1::2, :]
            e_left = self.u_checknode(torch.cat([e_odd, e_even], -1))
            u1_cw = _decode_u(e_left)
            e_right = self.u_bitnode(e_odd, e_even, u1_cw)
            u2_cw = _decode_u(e_right)
            x = torch.zeros(B, block_size, dtype=torch.long)
            x[:, 0::2] = u1_cw ^ u2_cw; x[:, 1::2] = u2_cw
            return x

        u_cw = _decode_u(emb)

        # Decode v|u
        emb_v = self.u_cond(torch.cat([emb, u_cw.float().unsqueeze(-1)], -1))
        v_idx = [0]
        def _decode_v(e_block):
            block_size = e_block.shape[1]
            if block_size == 1:
                llr = self.v_emb2llr(e_block[:, 0, :]).squeeze(-1)
                idx = v_idx[0]; v_idx[0] += 1
                dec = torch.zeros(B, dtype=torch.long) if idx in frozen_v \
                      else (llr > 0).long()
                v_hat[:, idx] = dec
                return dec.unsqueeze(1)
            e_odd = e_block[:, 0::2, :]; e_even = e_block[:, 1::2, :]
            e_left = self.v_checknode(torch.cat([e_odd, e_even], -1))
            v1_cw = _decode_v(e_left)
            e_right = self.v_bitnode(e_odd, e_even, v1_cw)
            v2_cw = _decode_v(e_right)
            x = torch.zeros(B, block_size, dtype=torch.long)
            x[:, 0::2] = v1_cw ^ v2_cw; x[:, 1::2] = v2_cw
            return x

        _decode_v(emb_v)
        return u_hat, v_hat


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

def run_exp_joint_fastce():
    """Joint 4-class fast_ce baseline with consistent architecture."""
    log("=" * 70)
    log("EXP A: Joint 4-class fast_ce (consistent arch)")
    log("=" * 70)

    N = 32; ku, kv = 15, 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    br = torch.from_numpy(bit_reversal_perm(int(math.log2(N)))).long()

    model = ConsistentMACDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    log(f"Params: {model.count_parameters():,}")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, 20001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng)
        zf_t = torch.from_numpy(zf).float()
        emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br]
        joint = torch.from_numpy(xf * 2 + yf).long()[:, br]

        loss = model.fast_ce(emb, joint)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            bler = evaluate_bler(model, channel, N, Au, Av, fu, fv, 500)
            log(f"  [{it}/20000] loss={loss.item():.4f} BLER={bler:.4f} ({time.time()-t0:.0f}s)")

    bler = evaluate_bler_batch(model, channel, N, Au, Av, fu, fv, 2000)
    log(f"EXP A RESULT: BLER={bler:.4f} (SC=0.046)")
    return model, bler


def run_exp_scheduled_sampling():
    """Joint 4-class fast_ce with scheduled sampling."""
    log("=" * 70)
    log("EXP B: Scheduled sampling fast_ce")
    log("=" * 70)

    N = 32; ku, kv = 15, 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    br = torch.from_numpy(bit_reversal_perm(int(math.log2(N)))).long()

    model = ConsistentMACDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, 30001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng)
        zf_t = torch.from_numpy(zf).float()
        emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br]
        joint = torch.from_numpy(xf * 2 + yf).long()[:, br]

        # Ramp up scheduled sampling
        p_sample = min(0.5, it / 30000 * 0.5)
        loss = model.fast_ce_scheduled_sampling(emb, joint, p_sample)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            bler = evaluate_bler(model, channel, N, Au, Av, fu, fv, 500)
            log(f"  [{it}/30000] loss={loss.item():.4f} p={p_sample:.3f} "
                f"BLER={bler:.4f} ({time.time()-t0:.0f}s)")

    bler = evaluate_bler_batch(model, channel, N, Au, Av, fu, fv, 2000)
    log(f"EXP B RESULT: BLER={bler:.4f} (SC=0.046)")
    return model, bler


def run_exp_binary_decomp():
    """Binary decomposition: u then v|u, both fast_ce."""
    log("=" * 70)
    log("EXP C: Binary decomposition (u then v|u)")
    log("=" * 70)

    N = 32; ku, kv = 15, 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    br = torch.from_numpy(bit_reversal_perm(int(math.log2(N)))).long()
    fu_set = {p - 1 for p in fu}; fv_set = {p - 1 for p in fv}

    model = BinaryDecompDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Params: {params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, 20001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng)
        zf_t = torch.from_numpy(zf).float()
        emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br]
        u_cw = torch.from_numpy(xf).long()[:, br]
        v_cw = torch.from_numpy(yf).long()[:, br]

        loss_u = model.fast_ce_u(emb, u_cw)
        loss_v = model.fast_ce_v(emb, v_cw, u_cw)
        loss = loss_u + loss_v

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            errs = 0; n_test = 500
            model.eval()
            rng_t = np.random.default_rng(999)
            with torch.no_grad():
                for _ in range(n_test):
                    uf_t = np.zeros((1, N), dtype=int)
                    vf_t = np.zeros((1, N), dtype=int)
                    for p in Au: uf_t[0, p-1] = rng_t.integers(0, 2)
                    for p in Av: vf_t[0, p-1] = rng_t.integers(0, 2)
                    xf_t = polar_encode_batch(uf_t)
                    yf_t = polar_encode_batch(vf_t)
                    zf_t2 = torch.from_numpy(channel.sample_batch(xf_t, yf_t)).float()
                    emb_t = model.z_encoder(zf_t2.unsqueeze(-1))[:, br]
                    u_dec, v_dec = model.sc_decode(emb_t, fu_set, fv_set)
                    ue = any(u_dec[0, p-1].item() != uf_t[0, p-1] for p in Au)
                    ve = any(v_dec[0, p-1].item() != vf_t[0, p-1] for p in Av)
                    if ue or ve: errs += 1
            model.train()
            bler = errs / n_test
            log(f"  [{it}/20000] loss_u={loss_u.item():.4f} loss_v={loss_v.item():.4f} "
                f"BLER={bler:.4f} ({time.time()-t0:.0f}s)")

    # Final eval
    errs = 0; n_test = 2000
    model.eval()
    rng_t = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(n_test):
            uf_t = np.zeros((1, N), dtype=int)
            vf_t = np.zeros((1, N), dtype=int)
            for p in Au: uf_t[0, p-1] = rng_t.integers(0, 2)
            for p in Av: vf_t[0, p-1] = rng_t.integers(0, 2)
            xf_t = polar_encode_batch(uf_t)
            yf_t = polar_encode_batch(vf_t)
            zf_t2 = torch.from_numpy(channel.sample_batch(xf_t, yf_t)).float()
            emb_t = model.z_encoder(zf_t2.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode(emb_t, fu_set, fv_set)
            ue = any(u_dec[0, p-1].item() != uf_t[0, p-1] for p in Au)
            ve = any(v_dec[0, p-1].item() != vf_t[0, p-1] for p in Av)
            if ue or ve: errs += 1
    bler = errs / n_test
    log(f"EXP C RESULT: BLER={bler:.4f} (SC=0.046)")
    return model, bler


def run_exp_binary_decomp_v2():
    """Binary decomposition v2: condition on u at every tree level, not just root."""
    log("=" * 70)
    log("EXP C2: Binary decomp with tree-level u conditioning")
    log("=" * 70)

    N = 32; ku, kv = 15, 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    br = torch.from_numpy(bit_reversal_perm(int(math.log2(N)))).long()
    fu_set = {p - 1 for p in fu}; fv_set = {p - 1 for p in fv}

    class BinaryDecompV2(nn.Module):
        """V2: v decoder conditions on u at every level via element-wise product."""
        def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
            super().__init__()
            self.d = d
            self.z_encoder = nn.Sequential(nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d))
            self.u_checknode = _make_mlp(2*d, hidden, d, n_layers)
            self.u_bitnode_mlp = _make_mlp(2*d, hidden, d, n_layers)
            self.u_emb2llr = _make_mlp(d, hidden, 1, n_layers)
            self.v_checknode = _make_mlp(2*d, hidden, d, n_layers)
            self.v_bitnode_mlp = _make_mlp(2*d, hidden, d, n_layers)
            self.v_emb2llr = _make_mlp(d, hidden, 1, n_layers)
            # Per-level conditioning: u_cw -> sign modulation
            self.u_to_gate = nn.Sequential(nn.Linear(1, d), nn.Tanh())

        def u_bitnode(self, e_odd, e_even, u_hard):
            if u_hard.dim() == 2: u_hard = u_hard.unsqueeze(-1)
            u_sign = (2.0 * u_hard.float() - 1.0).expand_as(e_odd)
            e_signed = e_odd * u_sign
            inp = torch.cat([e_signed, e_even], dim=-1)
            return self.u_bitnode_mlp(inp) + e_signed + e_even

        def v_bitnode(self, e_odd, e_even, v_hard):
            if v_hard.dim() == 2: v_hard = v_hard.unsqueeze(-1)
            v_sign = (2.0 * v_hard.float() - 1.0).expand_as(e_odd)
            e_signed = e_odd * v_sign
            inp = torch.cat([e_signed, e_even], dim=-1)
            return self.v_bitnode_mlp(inp) + e_signed + e_even

        def condition_on_u(self, emb, u_cw):
            """Modulate v embeddings based on u codeword."""
            gate = self.u_to_gate(u_cw.float().unsqueeze(-1))  # (B, M, d)
            return emb * (1 + gate)  # multiplicative modulation

        def fast_ce_u(self, emb, u_cw):
            B, N, d = emb.shape; n = int(math.log2(N))
            all_losses = []
            pred = self.u_emb2llr(emb).squeeze(-1)
            all_losses.append(F.binary_cross_entropy_with_logits(pred, u_cw.float()))

            V, E = [u_cw], [emb]
            for depth in range(n):
                VO, VE, EO, EE = [], [], [], []
                for v, e in zip(V, E):
                    VO.append(v[:, 0::2]); VE.append(v[:, 1::2])
                    EO.append(e[:, 0::2, :]); EE.append(e[:, 1::2, :])
                Vo = torch.cat(VO, 1); Ve = torch.cat(VE, 1)
                Eo = torch.cat(EO, 1); Ee = torch.cat(EE, 1)
                vx = Vo ^ Ve; vi = Ve
                el = self.u_checknode(torch.cat([Eo, Ee], -1))
                er = self.u_bitnode(Eo, Ee, vx)
                nc = 2**depth; cs = (N//2)//nc
                els = torch.split(el, cs, 1); ers = torch.split(er, cs, 1)
                vls = torch.split(vx, cs, 1); vrs = torch.split(vi, cs, 1)
                Vn, En = [], []
                for a, b, c, dd in zip(els, ers, vls, vrs):
                    En += [a, b]; Vn += [c, dd]
                ea = torch.cat(En, 1); va = torch.cat(Vn, 1)
                pred = self.u_emb2llr(ea).squeeze(-1)
                all_losses.append(F.binary_cross_entropy_with_logits(pred, va.float()))
                V = Vn; E = En
            return torch.stack(all_losses).mean()

        def fast_ce_v(self, emb, v_cw, u_cw):
            """V decoder with per-level u conditioning."""
            B, N, d = emb.shape; n = int(math.log2(N))
            all_losses = []
            emb_v = self.condition_on_u(emb, u_cw)
            pred = self.v_emb2llr(emb_v).squeeze(-1)
            all_losses.append(F.binary_cross_entropy_with_logits(pred, v_cw.float()))

            V, E, U = [v_cw], [emb_v], [u_cw]
            for depth in range(n):
                VO, VE, EO, EE, UO, UE = [], [], [], [], [], []
                for v, e, u in zip(V, E, U):
                    VO.append(v[:, 0::2]); VE.append(v[:, 1::2])
                    EO.append(e[:, 0::2, :]); EE.append(e[:, 1::2, :])
                    UO.append(u[:, 0::2]); UE.append(u[:, 1::2])
                Vo = torch.cat(VO, 1); Ve = torch.cat(VE, 1)
                Eo = torch.cat(EO, 1); Ee = torch.cat(EE, 1)
                Uo = torch.cat(UO, 1); Ue = torch.cat(UE, 1)
                vx = Vo ^ Ve; vi = Ve
                ux = Uo ^ Ue; ui = Ue
                el = self.v_checknode(torch.cat([Eo, Ee], -1))
                er = self.v_bitnode(Eo, Ee, vx)
                # Condition on u at this level
                nc = 2**depth; cs = (N//2)//nc
                els = torch.split(el, cs, 1); ers = torch.split(er, cs, 1)
                vls = torch.split(vx, cs, 1); vrs = torch.split(vi, cs, 1)
                uls = torch.split(ux, cs, 1); urs = torch.split(ui, cs, 1)
                Vn, En, Un = [], [], []
                for a, b, c, dd, uu1, uu2 in zip(els, ers, vls, vrs, uls, urs):
                    # Condition each embedding chunk on corresponding u chunk
                    a_cond = self.condition_on_u(a, uu1)
                    b_cond = self.condition_on_u(b, uu2)
                    En += [a_cond, b_cond]; Vn += [c, dd]; Un += [uu1, uu2]
                ea = torch.cat(En, 1); va = torch.cat(Vn, 1)
                pred = self.v_emb2llr(ea).squeeze(-1)
                all_losses.append(F.binary_cross_entropy_with_logits(pred, va.float()))
                V = Vn; E = En; U = Un
            return torch.stack(all_losses).mean()

        @torch.no_grad()
        def sc_decode(self, emb, frozen_u, frozen_v):
            B, N = emb.shape[0], emb.shape[1]
            u_hat = torch.zeros(B, N, dtype=torch.long)
            v_hat = torch.zeros(B, N, dtype=torch.long)
            u_idx = [0]
            def _du(e):
                bs = e.shape[1]
                if bs == 1:
                    llr = self.u_emb2llr(e[:, 0, :]).squeeze(-1)
                    idx = u_idx[0]; u_idx[0] += 1
                    dec = torch.zeros(B, dtype=torch.long) if idx in frozen_u else (llr > 0).long()
                    u_hat[:, idx] = dec
                    return dec.unsqueeze(1)
                eo = e[:, 0::2, :]; ee = e[:, 1::2, :]
                el = self.u_checknode(torch.cat([eo, ee], -1))
                u1 = _du(el)
                er = self.u_bitnode(eo, ee, u1)
                u2 = _du(er)
                x = torch.zeros(B, bs, dtype=torch.long)
                x[:, 0::2] = u1 ^ u2; x[:, 1::2] = u2
                return x
            u_cw = _du(emb)

            emb_v = self.condition_on_u(emb, u_cw)
            v_idx = [0]
            def _dv(e):
                bs = e.shape[1]
                if bs == 1:
                    llr = self.v_emb2llr(e[:, 0, :]).squeeze(-1)
                    idx = v_idx[0]; v_idx[0] += 1
                    dec = torch.zeros(B, dtype=torch.long) if idx in frozen_v else (llr > 0).long()
                    v_hat[:, idx] = dec
                    return dec.unsqueeze(1)
                eo = e[:, 0::2, :]; ee = e[:, 1::2, :]
                el = self.v_checknode(torch.cat([eo, ee], -1))
                v1 = _dv(el)
                er = self.v_bitnode(eo, ee, v1)
                v2 = _dv(er)
                x = torch.zeros(B, bs, dtype=torch.long)
                x[:, 0::2] = v1 ^ v2; x[:, 1::2] = v2
                return x
            _dv(emb_v)
            return u_hat, v_hat

    model = BinaryDecompV2(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Params: {params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, 20001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng)
        zf_t = torch.from_numpy(zf).float()
        emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br]
        u_cw = torch.from_numpy(xf).long()[:, br]
        v_cw = torch.from_numpy(yf).long()[:, br]

        loss_u = model.fast_ce_u(emb, u_cw)
        loss_v = model.fast_ce_v(emb, v_cw, u_cw)
        loss = loss_u + loss_v

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            errs = 0; nt = 500
            model.eval()
            rng_t = np.random.default_rng(999)
            with torch.no_grad():
                for _ in range(nt):
                    uf_t = np.zeros((1, N), dtype=int)
                    vf_t = np.zeros((1, N), dtype=int)
                    for p in Au: uf_t[0, p-1] = rng_t.integers(0, 2)
                    for p in Av: vf_t[0, p-1] = rng_t.integers(0, 2)
                    xf_t = polar_encode_batch(uf_t)
                    yf_t = polar_encode_batch(vf_t)
                    zf_t2 = torch.from_numpy(channel.sample_batch(xf_t, yf_t)).float()
                    emb_t = model.z_encoder(zf_t2.unsqueeze(-1))[:, br]
                    u_dec, v_dec = model.sc_decode(emb_t, fu_set, fv_set)
                    ue = any(u_dec[0, p-1].item() != uf_t[0, p-1] for p in Au)
                    ve = any(v_dec[0, p-1].item() != vf_t[0, p-1] for p in Av)
                    if ue or ve: errs += 1
            model.train()
            bler = errs / nt
            log(f"  [{it}/20000] loss={loss.item():.4f} BLER={bler:.4f} ({time.time()-t0:.0f}s)")

    # Final
    errs = 0; nt = 2000
    model.eval()
    rng_t = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(nt):
            uf_t = np.zeros((1, N), dtype=int)
            vf_t = np.zeros((1, N), dtype=int)
            for p in Au: uf_t[0, p-1] = rng_t.integers(0, 2)
            for p in Av: vf_t[0, p-1] = rng_t.integers(0, 2)
            xf_t = polar_encode_batch(uf_t)
            yf_t = polar_encode_batch(vf_t)
            zf_t2 = torch.from_numpy(channel.sample_batch(xf_t, yf_t)).float()
            emb_t = model.z_encoder(zf_t2.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode(emb_t, fu_set, fv_set)
            ue = any(u_dec[0, p-1].item() != uf_t[0, p-1] for p in Au)
            ve = any(v_dec[0, p-1].item() != vf_t[0, p-1] for p in Av)
            if ue or ve: errs += 1
    bler = errs / nt
    log(f"EXP C2 RESULT: BLER={bler:.4f} (SC=0.046)")
    return model, bler


def scale_test(model_class_or_model, N_test, is_binary=False):
    """Test a trained model at larger N."""
    n = int(math.log2(N_test))
    # Need appropriate ku, kv for N_test
    if N_test == 64:
        ku, kv = 31, 31
    elif N_test == 128:
        ku, kv = 63, 63
    elif N_test == 256:
        ku, kv = 127, 127
    else:
        return None

    channel = GaussianMAC(sigma2=SIGMA2)
    try:
        result = load_design(N_test, ku, kv)
    except:
        log(f"  No design for N={N_test}")
        return None
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    errs = 0; nt = 500
    model_class_or_model.eval()
    rng_t = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(nt):
            uf_t = np.zeros((1, N_test), dtype=int)
            vf_t = np.zeros((1, N_test), dtype=int)
            for p in Au: uf_t[0, p-1] = rng_t.integers(0, 2)
            for p in Av: vf_t[0, p-1] = rng_t.integers(0, 2)
            xf_t = polar_encode_batch(uf_t)
            yf_t = polar_encode_batch(vf_t)
            zf_t2 = torch.from_numpy(channel.sample_batch(xf_t, yf_t)).float()
            emb_t = model_class_or_model.z_encoder(zf_t2.unsqueeze(-1))[:, br]
            u_dec, v_dec = model_class_or_model.sc_decode(emb_t, fu_set, fv_set)
            ue = any(u_dec[0, p-1].item() != uf_t[0, p-1] for p in Au)
            ve = any(v_dec[0, p-1].item() != vf_t[0, p-1] for p in Av)
            if ue or ve: errs += 1
    bler = errs / nt
    return bler


if __name__ == '__main__':
    log("\n" + "=" * 70)
    log("BREAKTHROUGH V2 EXPERIMENTS STARTED")
    log("=" * 70)

    results = {}

    # Exp A: Joint 4-class fast_ce baseline
    try:
        model_a, bler_a = run_exp_joint_fastce()
        results['joint_fastce'] = bler_a
    except Exception as e:
        log(f"EXP A FAILED: {e}"); traceback.print_exc()

    # Exp B: Scheduled sampling
    try:
        model_b, bler_b = run_exp_scheduled_sampling()
        results['scheduled_sampling'] = bler_b
    except Exception as e:
        log(f"EXP B FAILED: {e}"); traceback.print_exc()

    # Exp C: Binary decomposition
    try:
        model_c, bler_c = run_exp_binary_decomp()
        results['binary_decomp'] = bler_c
    except Exception as e:
        log(f"EXP C FAILED: {e}"); traceback.print_exc()

    # Exp C2: Binary decomp with tree-level conditioning
    try:
        model_c2, bler_c2 = run_exp_binary_decomp_v2()
        results['binary_decomp_v2'] = bler_c2
    except Exception as e:
        log(f"EXP C2 FAILED: {e}"); traceback.print_exc()

    # Scale promising results to N=64
    best_name = min(results, key=results.get) if results else None
    if best_name and results[best_name] < 0.20:
        log(f"\nBest approach: {best_name} BLER={results[best_name]:.4f}")
        log("Testing at N=64...")
        model_best = {'joint_fastce': model_a, 'scheduled_sampling': model_b,
                      'binary_decomp': model_c, 'binary_decomp_v2': model_c2}.get(best_name)
        if model_best:
            bler64 = scale_test(model_best, 64)
            if bler64 is not None:
                log(f"N=64 BLER={bler64:.4f}")
                results[f'{best_name}_N64'] = bler64

    log("\n" + "=" * 70)
    log("ALL RESULTS:")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        log(f"  {k}: BLER={v:.4f}")
    log(f"  SC baseline: BLER=0.046")
    log("=" * 70)
