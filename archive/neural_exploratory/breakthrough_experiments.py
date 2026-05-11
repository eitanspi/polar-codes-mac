#!/usr/bin/env python3
"""
breakthrough_experiments.py — Systematic experiments to close the fast_ce/sequential gap
for 4-class MAC polar code neural decoder.

Experiments:
1. Hybrid: fast_ce pretrain -> sequential fine-tune
2. Parallel scheduled sampling
3. Gradient detaching in sequential decode
4. Binary decomposition (u then v|u)
5. Multi-pass refinement
"""
import sys, os, math, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file

# ── Config ──────────────────────────────────────────────────────────────────
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


# ── Data generation ─────────────────────────────────────────────────────────

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


# ── NeuralCalcParent ────────────────────────────────────────────────────────

class NeuralCalcParent(nn.Module):
    def __init__(self, d, hidden, n_layers=2):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.ELU(),
            nn.Linear(hidden, d), nn.Sigmoid()
        )
        self.candidate_net = _make_mlp(2 * d, hidden, d, n_layers)

    def forward(self, left_emb, right_emb):
        concat = torch.cat([left_emb, right_emb], dim=-1)
        gate = self.gate_net(concat)
        candidate = self.candidate_net(concat)
        residual = (left_emb + right_emb) / 2.0
        return gate * candidate + (1 - gate) * residual


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Hybrid fast_ce pretrain -> sequential fine-tune
# ══════════════════════════════════════════════════════════════════════════════

class HybridMACDecoder(nn.Module):
    """MAC decoder that supports both fast_ce and sequential tree walk."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d

        # Channel encoder
        self.z_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d)
        )

        # Tree operations — shared between fast_ce and sequential
        self.calc_left_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_parent_nn = NeuralCalcParent(d, hidden, n_layers)
        self.parent_second_nn = nn.Sequential(nn.Linear(d, d))

        # Decision head
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)
        self.logits2emb = _make_mlp(4, hidden, d, n_layers)

        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── fast_ce: parallel training (maps to NPD-style tree) ─────────────
    # The fast_ce uses the NPD even/odd tree, which corresponds to the
    # ncg tree operations: calc_left_nn takes (parent_first, parent_second, right)
    # In NPD-style: odd positions are "left" in the tree, even are "right"
    # CheckNode: (e_odd, e_even) -> e_left  maps to calc_left_nn(parent[:half], parent[half:], right)
    # But in fast_ce, parent = full embedding, and right = e_even embedding
    # We adapt: calc_left_nn(e_odd, e_even, zeros) and calc_right_nn(e_odd, e_even, left_cw)

    def checknode(self, e_odd, e_even):
        """CheckNode = calc_left_nn(e_odd, e_even, zeros)"""
        zeros = torch.zeros_like(e_odd)
        inp = torch.cat([e_odd, e_even, zeros], dim=-1)
        return self.calc_left_nn(inp)

    def bitnode(self, e_odd, e_even, uv_left):
        """BitNode = calc_right_nn(e_odd, e_even, left_emb)
        uv_left: (B, M) integer 0-3 joint decisions."""
        # Convert decisions to embeddings
        B, M = uv_left.shape
        d = self.d
        device = e_odd.device
        lp = torch.full((B * M, 4), -30.0, device=device)
        idx = uv_left.reshape(-1).long()
        lp[torch.arange(B * M, device=device), idx] = 0.0
        left_emb = self.logits2emb(lp).reshape(B, M, d)

        inp = torch.cat([e_odd, e_even, left_emb], dim=-1)
        return self.calc_right_nn(inp)

    def fast_ce(self, emb, joint_cw):
        """Parallel teacher-forced training for 4-class MAC.
        emb: (B, N, d), joint_cw: (B, N) in {0,1,2,3}."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        # Depth 0: predict from raw embeddings
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

            # XOR for left, identity for right
            u_o, v_o = J_odd // 2, J_odd % 2
            u_e, v_e = J_even // 2, J_even % 2
            J_left = (u_o ^ u_e) * 2 + (v_o ^ v_e)
            J_right = J_even

            e_left = self.checknode(E_odd, E_even)
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

            E_chunks_split = list(torch.split(e_all, cs, 1)) if cs > 0 else E_chunks
            # Keep the already-split chunks
            pass

        return torch.stack(all_losses).mean()

    def sc_decode_npd(self, emb, frozen_u, frozen_v):
        """Sequential SC decode using NPD tree structure (even/odd).
        frozen_u, frozen_v: sets of 0-indexed positions."""
        B, N = emb.shape[0], emb.shape[1]
        d = self.d
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

            e_left = self.checknode(e_odd, e_even)
            uv_left = _decode(e_left)

            e_right = self.bitnode(e_odd, e_even, uv_left)
            uv_right = _decode(e_right)

            # Return joint codeword
            half = block_size // 2
            u_l, v_l = uv_left // 2, uv_left % 2
            u_r, v_r = uv_right // 2, uv_right % 2
            # Reconstruct: odd = xor, even = identity
            joint = torch.zeros(B, block_size, dtype=torch.long)
            joint[:, 0::2] = (u_l ^ u_r) * 2 + (v_l ^ v_r)
            joint[:, 1::2] = u_r * 2 + v_r
            return joint

        with torch.no_grad():
            _decode(emb)
        return u_hat, v_hat

    # ── Sequential tree walk (ncg-style) for fine-tuning ─────────────────

    def _neural_calc_left(self, beta, edge_data):
        parent = edge_data[beta]
        right = edge_data[2 * beta + 1]
        l = right.shape[1]
        p_first = parent[:, :l]
        p_second = parent[:, l:]
        inp = torch.cat([p_first, p_second, right], dim=-1)
        edge_data[2 * beta] = self.calc_left_nn(inp)

    def _neural_calc_right(self, beta, edge_data):
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]
        p_first = parent[:, :l]
        p_second = parent[:, l:]
        inp = torch.cat([p_first, p_second, left], dim=-1)
        edge_data[2 * beta + 1] = self.calc_right_nn(inp)

    def _calc_parent(self, beta, edge_data):
        left = edge_data[2 * beta]
        right = edge_data[2 * beta + 1]
        parent_first = self.calc_parent_nn(left, right)
        parent_second = self.parent_second_nn(right)
        edge_data[beta] = torch.cat([parent_first, parent_second], dim=1)

    def _step_one(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0:
                self._neural_calc_left(current, edge_data)
            else:
                self._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self._calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: {current}->{beta}")

    @staticmethod
    def _get_path(current, target):
        if current == target:
            return []
        path_up, path_down = [], []
        c, t = current, target
        while c != t:
            if c > t:
                c >>= 1
                path_up.append(c)
            else:
                path_down.append(t)
                t >>= 1
        path_down.reverse()
        return path_up + path_down

    def _step_to(self, current, target, edge_data):
        if current == target:
            return current
        for beta in self._get_path(current, target):
            current = self._step_one(current, beta, edge_data)
        return current

    def _make_leaf_emb(self, u_val, v_val, batch, device):
        lp = torch.full((batch, 4), -30.0, device=device)
        LH = math.log(0.5)
        if u_val is not None and v_val is not None:
            idx = (u_val.long() * 2 + v_val.long()).unsqueeze(1)
            lp.scatter_(1, idx, 0.0)
        elif u_val is not None:
            lp.scatter_(1, (u_val.long() * 2).unsqueeze(1), LH)
            lp.scatter_(1, (u_val.long() * 2 + 1).unsqueeze(1), LH)
        elif v_val is not None:
            lp.scatter_(1, v_val.long().unsqueeze(1), LH)
            lp.scatter_(1, (v_val.long() + 2).unsqueeze(1), LH)
        else:
            lp.fill_(math.log(0.25))
        return self.logits2emb(lp)

    def forward_sequential(self, z, b, frozen_u, frozen_v,
                           u_true=None, v_true=None):
        """Sequential tree walk decode (ncg-style). For fine-tuning."""
        B, N = z.shape
        device = z.device
        d = self.d
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.z_encoder(z.unsqueeze(-1))[:, br]

        teacher = u_true is not None and v_true is not None

        edge_data = [None] * (2 * N)
        edge_data[1] = root
        no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0)
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

        dec_head = 1
        u_hat, v_hat = {}, {}
        all_logits, all_targets = [], []
        i_u, i_v = 0, 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = frozen_v

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1
            dec_head = self._step_to(dec_head, target_vtx, edge_data)

            temp = edge_data[leaf_edge][:, 0].clone()
            if leaf_edge & 1 == 0:
                self._neural_calc_left(target_vtx, edge_data)
            else:
                self._neural_calc_right(target_vtx, edge_data)
            top_down = edge_data[leaf_edge][:, 0]
            combined = top_down + temp
            logits = self.emb2logits(combined)

            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.float32,
                                 device=device)
            else:
                all_logits.append(logits)
                if teacher:
                    target = (u_true[:, i_t - 1] * 2 + v_true[:, i_t - 1]).long()
                    all_targets.append(target)
                    bit = u_true[:, i_t - 1] if gamma == 0 else v_true[:, i_t - 1]
                else:
                    with torch.no_grad():
                        if gamma == 0:
                            p0 = torch.logsumexp(logits[:, :2], dim=1)
                            p1 = torch.logsumexp(logits[:, 2:], dim=1)
                        else:
                            p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                            p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                        bit = (p1 > p0).float()

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            new_emb = self._make_leaf_emb(u_hat.get(i_t), v_hat.get(i_t),
                                          B, device)
            edge_data[leaf_edge] = new_emb.unsqueeze(1)

        return all_logits, all_targets, u_hat, v_hat


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_bler_npd(model, channel, N, Au, Av, fu, fv, n_cw=500):
    """Evaluate BLER using NPD-style SC decode."""
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
            u_dec, v_dec = model.sc_decode_npd(emb, fu_set, fv_set)
            ue = any(u_dec[0, p - 1].item() != uf[0, p - 1] for p in Au)
            ve = any(v_dec[0, p - 1].item() != vf[0, p - 1] for p in Av)
            if ue or ve:
                errs += 1
    model.train()
    return errs / n_cw


def evaluate_bler_sequential(model, channel, N, Au, Av, fu, fv, b,
                             frozen_u, frozen_v, n_cw=500):
    """Evaluate BLER using sequential tree walk decode."""
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
            _, _, u_hat, v_hat = model.forward_sequential(
                zf, b, frozen_u, frozen_v)
            ue = any(u_hat[i_t].item() != uf[0, i_t - 1]
                     for i_t in range(1, N + 1) if i_t in Au)
            ve = any(v_hat[i_t].item() != vf[0, i_t - 1]
                     for i_t in range(1, N + 1) if i_t in Av)
            if ue or ve:
                errs += 1
    model.train()
    return errs / n_cw


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 5: Binary Decomposition (u then v|u)
# ══════════════════════════════════════════════════════════════════════════════

class BinaryDecompDecoder(nn.Module):
    """Decompose 4-class into two binary problems: first u, then v|u.
    Both use NPD-style fast_ce with shared z_encoder."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d

        # Shared channel encoder
        self.z_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d)
        )

        # U decoder (binary)
        self.u_checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.u_bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.u_emb2llr = _make_mlp(d, hidden, 1, n_layers)

        # V|U decoder (binary, conditioned on u)
        self.v_checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.v_bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.v_emb2llr = _make_mlp(d, hidden, 1, n_layers)

        # Conditioning: merge u decision into v embeddings
        self.u_cond = nn.Sequential(nn.Linear(d + 1, hidden), nn.ELU(),
                                    nn.Linear(hidden, d))

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
        """Fast CE for u decoder (standard NPD)."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        pred = self.u_emb2llr(emb).squeeze(-1)
        all_losses.append(F.binary_cross_entropy_with_logits(
            pred, u_cw.float(), reduction='mean'))

        V = [u_cw]
        E = [emb]

        for depth in range(n):
            V_odds, V_evens, E_odds, E_evens = [], [], [], []
            for v, e in zip(V, E):
                V_odds.append(v[:, 0::2]); V_evens.append(v[:, 1::2])
                E_odds.append(e[:, 0::2, :]); E_evens.append(e[:, 1::2, :])

            V_odd = torch.cat(V_odds, 1); V_even = torch.cat(V_evens, 1)
            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)

            v_xor = V_odd ^ V_even
            v_id = V_even

            e_left = self.u_checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.u_bitnode(E_odd, E_even, v_xor)

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            vl = torch.split(v_xor, cs, 1)
            vr = torch.split(v_id, cs, 1)

            V_new, E_new = [], []
            for a, b, c, dd in zip(el, er, vl, vr):
                E_new += [a, b]; V_new += [c, dd]

            e_all = torch.cat(E_new, 1)
            v_all = torch.cat(V_new, 1)
            pred = self.u_emb2llr(e_all).squeeze(-1)
            all_losses.append(F.binary_cross_entropy_with_logits(
                pred, v_all.float(), reduction='mean'))

            V = V_new; E = E_new

        return torch.stack(all_losses).mean()

    def fast_ce_v(self, emb, v_cw, u_cw):
        """Fast CE for v|u decoder. Conditions on u decisions."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        # Condition embedding on u_cw
        u_cond_input = torch.cat([emb, u_cw.float().unsqueeze(-1)], dim=-1)
        emb_v = self.u_cond(u_cond_input)

        pred = self.v_emb2llr(emb_v).squeeze(-1)
        all_losses.append(F.binary_cross_entropy_with_logits(
            pred, v_cw.float(), reduction='mean'))

        V = [v_cw]
        E = [emb_v]
        U = [u_cw]

        for depth in range(n):
            V_odds, V_evens, E_odds, E_evens = [], [], [], []
            U_odds, U_evens = [], []
            for v, e, u in zip(V, E, U):
                V_odds.append(v[:, 0::2]); V_evens.append(v[:, 1::2])
                E_odds.append(e[:, 0::2, :]); E_evens.append(e[:, 1::2, :])
                U_odds.append(u[:, 0::2]); U_evens.append(u[:, 1::2])

            V_odd = torch.cat(V_odds, 1); V_even = torch.cat(V_evens, 1)
            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)
            U_odd = torch.cat(U_odds, 1); U_even = torch.cat(U_evens, 1)

            v_xor = V_odd ^ V_even
            v_id = V_even
            u_xor = U_odd ^ U_even
            u_id = U_even

            e_left = self.v_checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.v_bitnode(E_odd, E_even, v_xor)

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            vl = torch.split(v_xor, cs, 1)
            vr = torch.split(v_id, cs, 1)
            ul = torch.split(u_xor, cs, 1)
            ur = torch.split(u_id, cs, 1)

            V_new, E_new, U_new = [], [], []
            for a, b, c, dd, uu1, uu2 in zip(el, er, vl, vr, ul, ur):
                E_new += [a, b]; V_new += [c, dd]; U_new += [uu1, uu2]

            e_all = torch.cat(E_new, 1)
            v_all = torch.cat(V_new, 1)
            pred = self.v_emb2llr(e_all).squeeze(-1)
            all_losses.append(F.binary_cross_entropy_with_logits(
                pred, v_all.float(), reduction='mean'))

            V = V_new; E = E_new; U = U_new

        return torch.stack(all_losses).mean()

    def sc_decode(self, emb, frozen_u, frozen_v):
        """Sequential decode: first u, then v|u."""
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
                if idx in frozen_u:
                    dec = torch.zeros(B, dtype=torch.long)
                else:
                    dec = (llr > 0).long()
                u_hat[:, idx] = dec
                return dec.unsqueeze(1)

            e_odd = e_block[:, 0::2, :]
            e_even = e_block[:, 1::2, :]
            e_left = self.u_checknode(torch.cat([e_odd, e_even], -1))
            u1_cw = _decode_u(e_left)
            e_right = self.u_bitnode(e_odd, e_even, u1_cw)
            u2_cw = _decode_u(e_right)
            x = torch.zeros(B, block_size, dtype=torch.long)
            x[:, 0::2] = u1_cw ^ u2_cw
            x[:, 1::2] = u2_cw
            return x

        with torch.no_grad():
            u_cw = _decode_u(emb)

        # Condition on decoded u and decode v
        u_cond_input = torch.cat([emb, u_cw.float().unsqueeze(-1)], dim=-1)
        emb_v = self.u_cond(u_cond_input)

        v_idx = [0]
        def _decode_v(e_block):
            block_size = e_block.shape[1]
            if block_size == 1:
                llr = self.v_emb2llr(e_block[:, 0, :]).squeeze(-1)
                idx = v_idx[0]; v_idx[0] += 1
                if idx in frozen_v:
                    dec = torch.zeros(B, dtype=torch.long)
                else:
                    dec = (llr > 0).long()
                v_hat[:, idx] = dec
                return dec.unsqueeze(1)

            e_odd = e_block[:, 0::2, :]
            e_even = e_block[:, 1::2, :]
            e_left = self.v_checknode(torch.cat([e_odd, e_even], -1))
            v1_cw = _decode_v(e_left)
            e_right = self.v_bitnode(e_odd, e_even, v1_cw)
            v2_cw = _decode_v(e_right)
            x = torch.zeros(B, block_size, dtype=torch.long)
            x[:, 0::2] = v1_cw ^ v2_cw
            x[:, 1::2] = v2_cw
            return x

        with torch.no_grad():
            _decode_v(emb_v)

        return u_hat, v_hat


# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Gradient Detaching
# ══════════════════════════════════════════════════════════════════════════════

class DetachedSequentialDecoder(nn.Module):
    """Sequential tree walk with gradient detaching every K steps."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d)
        )
        self.calc_left_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_parent_nn = NeuralCalcParent(d, hidden, n_layers)
        self.parent_second_nn = nn.Sequential(nn.Linear(d, d))
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)
        self.logits2emb = _make_mlp(4, hidden, d, n_layers)
        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def _neural_calc_left(self, beta, edge_data):
        parent = edge_data[beta]
        right = edge_data[2 * beta + 1]
        l = right.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], right], dim=-1)
        edge_data[2 * beta] = self.calc_left_nn(inp)

    def _neural_calc_right(self, beta, edge_data):
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], left], dim=-1)
        edge_data[2 * beta + 1] = self.calc_right_nn(inp)

    def _calc_parent(self, beta, edge_data):
        left = edge_data[2 * beta]
        right = edge_data[2 * beta + 1]
        parent_first = self.calc_parent_nn(left, right)
        parent_second = self.parent_second_nn(right)
        edge_data[beta] = torch.cat([parent_first, parent_second], dim=1)

    def _step_one(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0:
                self._neural_calc_left(current, edge_data)
            else:
                self._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self._calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: {current}->{beta}")

    @staticmethod
    def _get_path(current, target):
        if current == target:
            return []
        path_up, path_down = [], []
        c, t = current, target
        while c != t:
            if c > t:
                c >>= 1
                path_up.append(c)
            else:
                path_down.append(t)
                t >>= 1
        path_down.reverse()
        return path_up + path_down

    def _step_to(self, current, target, edge_data):
        if current == target:
            return current
        for beta in self._get_path(current, target):
            current = self._step_one(current, beta, edge_data)
        return current

    def _make_leaf_emb(self, u_val, v_val, batch, device):
        lp = torch.full((batch, 4), -30.0, device=device)
        LH = math.log(0.5)
        if u_val is not None and v_val is not None:
            idx = (u_val.long() * 2 + v_val.long()).unsqueeze(1)
            lp.scatter_(1, idx, 0.0)
        elif u_val is not None:
            lp.scatter_(1, (u_val.long() * 2).unsqueeze(1), LH)
            lp.scatter_(1, (u_val.long() * 2 + 1).unsqueeze(1), LH)
        elif v_val is not None:
            lp.scatter_(1, v_val.long().unsqueeze(1), LH)
            lp.scatter_(1, (v_val.long() + 2).unsqueeze(1), LH)
        else:
            lp.fill_(math.log(0.25))
        return self.logits2emb(lp)

    def forward(self, z, b, frozen_u, frozen_v, u_true=None, v_true=None,
                detach_every=None):
        """Sequential decode with optional gradient detaching."""
        B, N = z.shape
        device = z.device
        d = self.d
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.z_encoder(z.unsqueeze(-1))[:, br]

        teacher = u_true is not None and v_true is not None

        edge_data = [None] * (2 * N)
        edge_data[1] = root
        no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0)
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

        dec_head = 1
        u_hat, v_hat = {}, {}
        all_logits, all_targets = [], []
        i_u, i_v = 0, 0

        for step in range(2 * N):
            # Detach gradients periodically
            if detach_every and step > 0 and step % detach_every == 0:
                for idx in range(1, 2 * N):
                    if edge_data[idx] is not None:
                        edge_data[idx] = edge_data[idx].detach()

            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = frozen_v

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1
            dec_head = self._step_to(dec_head, target_vtx, edge_data)

            temp = edge_data[leaf_edge][:, 0].clone()
            if leaf_edge & 1 == 0:
                self._neural_calc_left(target_vtx, edge_data)
            else:
                self._neural_calc_right(target_vtx, edge_data)
            top_down = edge_data[leaf_edge][:, 0]
            combined = top_down + temp
            logits = self.emb2logits(combined)

            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.float32,
                                 device=device)
            else:
                all_logits.append(logits)
                if teacher:
                    target = (u_true[:, i_t - 1] * 2 + v_true[:, i_t - 1]).long()
                    all_targets.append(target)
                    bit = u_true[:, i_t - 1] if gamma == 0 else v_true[:, i_t - 1]
                else:
                    with torch.no_grad():
                        if gamma == 0:
                            p0 = torch.logsumexp(logits[:, :2], dim=1)
                            p1 = torch.logsumexp(logits[:, 2:], dim=1)
                        else:
                            p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                            p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                        bit = (p1 > p0).float()

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            new_emb = self._make_leaf_emb(u_hat.get(i_t), v_hat.get(i_t),
                                          B, device)
            edge_data[leaf_edge] = new_emb.unsqueeze(1)

        return all_logits, all_targets, u_hat, v_hat


def evaluate_bler_detached(model, channel, N, Au, Av, fu, fv, b,
                           frozen_u, frozen_v, n_cw=500):
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
            _, _, u_hat, v_hat = model(zf, b, frozen_u, frozen_v)
            ue = any(u_hat[i_t].item() != uf[0, i_t - 1]
                     for i_t in range(1, N + 1) if i_t in Au)
            ve = any(v_hat[i_t].item() != vf[0, i_t - 1]
                     for i_t in range(1, N + 1) if i_t in Av)
            if ue or ve:
                errs += 1
    model.train()
    return errs / n_cw


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN: Run experiments
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment_1_hybrid():
    """Hybrid: fast_ce pretrain -> sequential fine-tune."""
    log("=" * 70)
    log("EXPERIMENT 1: Hybrid fast_ce pretrain -> sequential fine-tune")
    log("=" * 70)

    N = 32; n = 5; ku = 15; kv = 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    b = make_path(N, N // 2)
    frozen_u = {i: 0 for i in range(1, N + 1) if i not in Au}
    frozen_v = {i: 0 for i in range(1, N + 1) if i not in Av}
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    model = HybridMACDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    log(f"Model params: {model.count_parameters():,}")

    # Phase 1: fast_ce pretrain
    log("--- Phase 1: fast_ce pretrain (10K iters) ---")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, 10001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng)
        zf_t = torch.from_numpy(zf).float()
        emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br]
        joint = torch.from_numpy(xf * 2 + yf).long()[:, br]

        loss = model.fast_ce(emb, joint)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 2000 == 0:
            bler = evaluate_bler_npd(model, channel, N, Au, Av, fu, fv, 500)
            log(f"  fast_ce [{it}/10000] loss={loss.item():.4f} NPD-BLER={bler:.4f} "
                f"({time.time()-t0:.0f}s)")

    bler_after_fastce = evaluate_bler_npd(model, channel, N, Au, Av, fu, fv, 1000)
    log(f"After fast_ce: NPD-BLER={bler_after_fastce:.4f}")

    # Also check sequential BLER
    bler_seq = evaluate_bler_sequential(model, channel, N, Au, Av, fu, fv, b,
                                        frozen_u, frozen_v, 500)
    log(f"After fast_ce: Sequential-BLER={bler_seq:.4f}")

    # Phase 2: sequential fine-tune
    log("--- Phase 2: Sequential fine-tune (15K iters) ---")
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4)
    rng2 = np.random.default_rng(123)

    for it in range(1, 15001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng2, batch=32)
        zf_t = torch.from_numpy(zf).float()
        u_t = torch.from_numpy(uf).float()
        v_t = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _ = model.forward_sequential(
            zf_t, b, frozen_u, frozen_v, u_t, v_t)

        if len(all_logits) > 0:
            loss = F.cross_entropy(torch.cat(all_logits, 0),
                                   torch.cat(all_targets, 0))
            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
        else:
            loss = torch.tensor(0.0)

        if it % 3000 == 0:
            bler = evaluate_bler_sequential(model, channel, N, Au, Av, fu, fv,
                                            b, frozen_u, frozen_v, 500)
            log(f"  seq [{it}/15000] loss={loss.item():.4f} BLER={bler:.4f} "
                f"({time.time()-t0:.0f}s)")

    bler_final = evaluate_bler_sequential(model, channel, N, Au, Av, fu, fv,
                                          b, frozen_u, frozen_v, 1000)
    log(f"EXPERIMENT 1 RESULT: BLER={bler_final:.4f} (SC=0.046)")

    # Also train from scratch baseline
    log("--- Baseline: Sequential from scratch (15K iters) ---")
    model_scratch = HybridMACDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    opt3 = torch.optim.Adam(model_scratch.parameters(), lr=1e-4)
    rng3 = np.random.default_rng(123)

    for it in range(1, 15001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng3, batch=32)
        zf_t = torch.from_numpy(zf).float()
        u_t = torch.from_numpy(uf).float()
        v_t = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _ = model_scratch.forward_sequential(
            zf_t, b, frozen_u, frozen_v, u_t, v_t)

        if len(all_logits) > 0:
            loss = F.cross_entropy(torch.cat(all_logits, 0),
                                   torch.cat(all_targets, 0))
            opt3.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_scratch.parameters(), 1.0)
            opt3.step()

        if it % 3000 == 0:
            bler = evaluate_bler_sequential(model_scratch, channel, N, Au, Av,
                                            fu, fv, b, frozen_u, frozen_v, 500)
            log(f"  scratch [{it}/15000] loss={loss.item():.4f} BLER={bler:.4f} "
                f"({time.time()-t0:.0f}s)")

    bler_scratch = evaluate_bler_sequential(model_scratch, channel, N, Au, Av,
                                            fu, fv, b, frozen_u, frozen_v, 1000)
    log(f"BASELINE RESULT: Scratch BLER={bler_scratch:.4f} (SC=0.046)")
    log(f"COMPARISON: Hybrid={bler_final:.4f} vs Scratch={bler_scratch:.4f}")

    return bler_final, bler_scratch


def run_experiment_3_detached():
    """Gradient detaching in sequential decode."""
    log("=" * 70)
    log("EXPERIMENT 3: Gradient detaching (K=log(N) steps)")
    log("=" * 70)

    N = 32; n = 5; ku = 15; kv = 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    b = make_path(N, N // 2)
    frozen_u = {i: 0 for i in range(1, N + 1) if i not in Au}
    frozen_v = {i: 0 for i in range(1, N + 1) if i not in Av}

    for K in [5, 8, 16]:
        log(f"--- Detach every K={K} steps ---")
        model = DetachedSequentialDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        rng = np.random.default_rng(42)
        t0 = time.time()

        for it in range(1, 15001):
            uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng, batch=32)
            zf_t = torch.from_numpy(zf).float()
            u_t = torch.from_numpy(uf).float()
            v_t = torch.from_numpy(vf).float()

            all_logits, all_targets, _, _ = model(
                zf_t, b, frozen_u, frozen_v, u_t, v_t, detach_every=K)

            if len(all_logits) > 0:
                loss = F.cross_entropy(torch.cat(all_logits, 0),
                                       torch.cat(all_targets, 0))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                loss = torch.tensor(0.0)

            if it % 5000 == 0:
                bler = evaluate_bler_detached(model, channel, N, Au, Av, fu, fv,
                                              b, frozen_u, frozen_v, 500)
                log(f"  K={K} [{it}/15000] loss={loss.item():.4f} BLER={bler:.4f} "
                    f"({time.time()-t0:.0f}s)")

        bler = evaluate_bler_detached(model, channel, N, Au, Av, fu, fv,
                                      b, frozen_u, frozen_v, 1000)
        log(f"  K={K} FINAL: BLER={bler:.4f} (SC=0.046)")


def run_experiment_5_binary_decomp():
    """Binary decomposition: separate u and v|u decoders."""
    log("=" * 70)
    log("EXPERIMENT 5: Binary decomposition (u then v|u)")
    log("=" * 70)

    N = 32; n = 5; ku = 15; kv = 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}

    model = BinaryDecompDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Binary decomp params: {params:,}")

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

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            # Evaluate
            errs = 0; n_test = 500
            model.eval()
            rng_t = np.random.default_rng(999)
            with torch.no_grad():
                for _ in range(n_test):
                    uf_t = np.zeros((1, N), dtype=int)
                    vf_t = np.zeros((1, N), dtype=int)
                    for p in Au: uf_t[0, p - 1] = rng_t.integers(0, 2)
                    for p in Av: vf_t[0, p - 1] = rng_t.integers(0, 2)
                    xf_t = polar_encode_batch(uf_t)
                    yf_t = polar_encode_batch(vf_t)
                    zf_t2 = torch.from_numpy(channel.sample_batch(xf_t, yf_t)).float()
                    emb_t = model.z_encoder(zf_t2.unsqueeze(-1))[:, br]
                    u_dec, v_dec = model.sc_decode(emb_t, fu_set, fv_set)
                    ue = any(u_dec[0, p - 1].item() != uf_t[0, p - 1] for p in Au)
                    ve = any(v_dec[0, p - 1].item() != vf_t[0, p - 1] for p in Av)
                    if ue or ve:
                        errs += 1
            model.train()
            bler = errs / n_test
            log(f"  [{it}/20000] loss={loss.item():.4f} BLER={bler:.4f} "
                f"({time.time()-t0:.0f}s)")

    # Final eval
    errs = 0; n_test = 1000
    model.eval()
    rng_t = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(n_test):
            uf_t = np.zeros((1, N), dtype=int)
            vf_t = np.zeros((1, N), dtype=int)
            for p in Au: uf_t[0, p - 1] = rng_t.integers(0, 2)
            for p in Av: vf_t[0, p - 1] = rng_t.integers(0, 2)
            xf_t = polar_encode_batch(uf_t)
            yf_t = polar_encode_batch(vf_t)
            zf_t2 = torch.from_numpy(channel.sample_batch(xf_t, yf_t)).float()
            emb_t = model.z_encoder(zf_t2.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode(emb_t, fu_set, fv_set)
            ue = any(u_dec[0, p - 1].item() != uf_t[0, p - 1] for p in Au)
            ve = any(v_dec[0, p - 1].item() != vf_t[0, p - 1] for p in Av)
            if ue or ve:
                errs += 1
    bler = errs / n_test
    log(f"EXPERIMENT 5 RESULT: BLER={bler:.4f} (SC=0.046)")
    return bler


def run_experiment_2_scheduled_sampling():
    """Parallel scheduled sampling: two-pass fast_ce."""
    log("=" * 70)
    log("EXPERIMENT 2: Parallel scheduled sampling")
    log("=" * 70)

    N = 32; n = 5; ku = 15; kv = 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}

    model = HybridMACDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, 20001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng)
        zf_t = torch.from_numpy(zf).float()
        emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br]
        joint = torch.from_numpy(xf * 2 + yf).long()[:, br]

        # Scheduled sampling probability (increases over training)
        p_sample = min(0.5, it / 20000)

        # Pass 1: standard fast_ce to get predictions
        B, NN, d = emb.shape
        nn_log = int(math.log2(NN))

        # Get model predictions at each level
        with torch.no_grad():
            pred_logits = model.emb2logits(emb)
            pred_joint = pred_logits.argmax(dim=-1)  # (B, N)

        # Pass 2: fast_ce but with scheduled sampling
        # Replace some true values with model predictions
        mask = torch.rand(B, NN) < p_sample
        joint_ss = joint.clone()
        joint_ss[mask] = pred_joint[mask]

        loss = model.fast_ce(emb, joint_ss)

        # Also add standard fast_ce loss for stability
        loss_std = model.fast_ce(emb, joint)
        loss_total = 0.5 * loss + 0.5 * loss_std

        opt.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            bler = evaluate_bler_npd(model, channel, N, Au, Av, fu, fv, 500)
            log(f"  [{it}/20000] loss={loss_total.item():.4f} p={p_sample:.2f} "
                f"BLER={bler:.4f} ({time.time()-t0:.0f}s)")

    bler = evaluate_bler_npd(model, channel, N, Au, Av, fu, fv, 1000)
    log(f"EXPERIMENT 2 RESULT: BLER={bler:.4f} (SC=0.046)")
    return bler


def run_experiment_hybrid_v2_longer():
    """Extended hybrid with more fast_ce pretraining and longer fine-tuning."""
    log("=" * 70)
    log("EXPERIMENT 1b: Extended hybrid (20K fast_ce + 30K sequential)")
    log("=" * 70)

    N = 32; n = 5; ku = 15; kv = 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    b = make_path(N, N // 2)
    frozen_u = {i: 0 for i in range(1, N + 1) if i not in Au}
    frozen_v = {i: 0 for i in range(1, N + 1) if i not in Av}
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    model = HybridMACDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)

    # Phase 1: fast_ce pretrain (20K)
    log("--- Phase 1: fast_ce pretrain (20K iters) ---")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, 20001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng)
        zf_t = torch.from_numpy(zf).float()
        emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br]
        joint = torch.from_numpy(xf * 2 + yf).long()[:, br]

        loss = model.fast_ce(emb, joint)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            bler = evaluate_bler_npd(model, channel, N, Au, Av, fu, fv, 500)
            log(f"  fast_ce [{it}/20000] loss={loss.item():.4f} NPD-BLER={bler:.4f}")

    # Phase 2: sequential fine-tune with lower LR and cosine schedule
    log("--- Phase 2: Sequential fine-tune (30K iters, cosine LR) ---")
    opt2 = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, 30000, eta_min=1e-6)
    rng2 = np.random.default_rng(123)

    best_bler = 1.0
    for it in range(1, 30001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng2, batch=32)
        zf_t = torch.from_numpy(zf).float()
        u_t = torch.from_numpy(uf).float()
        v_t = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _ = model.forward_sequential(
            zf_t, b, frozen_u, frozen_v, u_t, v_t)

        if len(all_logits) > 0:
            loss = F.cross_entropy(torch.cat(all_logits, 0),
                                   torch.cat(all_targets, 0))
            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            scheduler.step()

        if it % 5000 == 0:
            bler = evaluate_bler_sequential(model, channel, N, Au, Av, fu, fv,
                                            b, frozen_u, frozen_v, 500)
            if bler < best_bler:
                best_bler = bler
            log(f"  seq [{it}/30000] loss={loss.item():.4f} BLER={bler:.4f} "
                f"best={best_bler:.4f} ({time.time()-t0:.0f}s)")

    bler_final = evaluate_bler_sequential(model, channel, N, Au, Av, fu, fv,
                                          b, frozen_u, frozen_v, 1000)
    log(f"EXPERIMENT 1b RESULT: BLER={bler_final:.4f} best={best_bler:.4f} (SC=0.046)")
    return bler_final


def run_experiment_hybrid_detached():
    """Hybrid: fast_ce pretrain -> detached sequential fine-tune."""
    log("=" * 70)
    log("EXPERIMENT 1c: Hybrid fast_ce + detached sequential")
    log("=" * 70)

    N = 32; n = 5; ku = 15; kv = 15
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    b = make_path(N, N // 2)
    frozen_u = {i: 0 for i in range(1, N + 1) if i not in Au}
    frozen_v = {i: 0 for i in range(1, N + 1) if i not in Av}
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    model = DetachedSequentialDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)

    # Phase 1: We can't use fast_ce with this model directly, so let's use
    # a HybridMACDecoder for fast_ce, then transfer weights.
    # Actually, let's just do detached sequential from scratch with different K values.
    # But also try: pretrain with small K (more detaching = faster), then fine-tune with K=None (full gradients)

    log("--- Phase 1: Detached sequential K=8 (10K iters) ---")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, 10001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng, batch=32)
        zf_t = torch.from_numpy(zf).float()
        u_t = torch.from_numpy(uf).float()
        v_t = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _ = model(
            zf_t, b, frozen_u, frozen_v, u_t, v_t, detach_every=8)

        if len(all_logits) > 0:
            loss = F.cross_entropy(torch.cat(all_logits, 0),
                                   torch.cat(all_targets, 0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if it % 5000 == 0:
            bler = evaluate_bler_detached(model, channel, N, Au, Av, fu, fv,
                                          b, frozen_u, frozen_v, 500)
            log(f"  K=8 [{it}/10000] loss={loss.item():.4f} BLER={bler:.4f}")

    log("--- Phase 2: Full gradients fine-tune (10K iters) ---")
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4)

    for it in range(1, 10001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng, batch=32)
        zf_t = torch.from_numpy(zf).float()
        u_t = torch.from_numpy(uf).float()
        v_t = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _ = model(
            zf_t, b, frozen_u, frozen_v, u_t, v_t, detach_every=None)

        if len(all_logits) > 0:
            loss = F.cross_entropy(torch.cat(all_logits, 0),
                                   torch.cat(all_targets, 0))
            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()

        if it % 5000 == 0:
            bler = evaluate_bler_detached(model, channel, N, Au, Av, fu, fv,
                                          b, frozen_u, frozen_v, 500)
            log(f"  full [{it}/10000] loss={loss.item():.4f} BLER={bler:.4f}")

    bler = evaluate_bler_detached(model, channel, N, Au, Av, fu, fv,
                                  b, frozen_u, frozen_v, 1000)
    log(f"EXPERIMENT 1c RESULT: BLER={bler:.4f} (SC=0.046)")
    return bler


if __name__ == '__main__':
    log("=" * 70)
    log("BREAKTHROUGH EXPERIMENTS STARTED")
    log("=" * 70)

    results = {}

    try:
        # Experiment 1: Hybrid
        bler_hybrid, bler_scratch = run_experiment_1_hybrid()
        results['hybrid'] = bler_hybrid
        results['scratch_baseline'] = bler_scratch
    except Exception as e:
        log(f"EXPERIMENT 1 FAILED: {e}")
        traceback.print_exc()

    try:
        # Experiment 3: Gradient detaching
        run_experiment_3_detached()
    except Exception as e:
        log(f"EXPERIMENT 3 FAILED: {e}")
        traceback.print_exc()

    try:
        # Experiment 5: Binary decomposition
        bler_decomp = run_experiment_5_binary_decomp()
        results['binary_decomp'] = bler_decomp
    except Exception as e:
        log(f"EXPERIMENT 5 FAILED: {e}")
        traceback.print_exc()

    try:
        # Experiment 2: Scheduled sampling
        bler_ss = run_experiment_2_scheduled_sampling()
        results['scheduled_sampling'] = bler_ss
    except Exception as e:
        log(f"EXPERIMENT 2 FAILED: {e}")
        traceback.print_exc()

    # If any approach shows promise, run extended versions
    if results.get('hybrid', 1.0) < 0.20:
        try:
            bler_ext = run_experiment_hybrid_v2_longer()
            results['hybrid_extended'] = bler_ext
        except Exception as e:
            log(f"Extended hybrid FAILED: {e}")

    try:
        bler_hd = run_experiment_hybrid_detached()
        results['hybrid_detached'] = bler_hd
    except Exception as e:
        log(f"Hybrid detached FAILED: {e}")

    log("=" * 70)
    log("ALL RESULTS:")
    for k, v in results.items():
        log(f"  {k}: BLER={v:.4f}")
    log(f"  SC baseline: BLER=0.046")
    log("=" * 70)
