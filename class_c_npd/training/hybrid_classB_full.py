#!/usr/bin/env python3
"""
Full hybrid Class B decoder: fast_ce + supervised CalcParent + MI tracking.

Phase 1: Train checknode/bitnode/z_encoder/emb2logits/logits2emb with fast_ce
Phase 2: Freeze Phase 1, train neural CalcParent on (left, right) -> parent pairs
Eval:    CG tree walk inference with neural CalcLeft/CalcRight/CalcParent

MI metric: MI_i = (log(4) - CE_i) / log(4) per position (4-class, max=1.0)
Logged every eval step for plotting.

Usage:
    python hybrid_classB_full.py          # runs on GPU if available
    python hybrid_classB_full.py --cpu    # force CPU
"""
import sys, os, math, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file

# ─── Config ────────────────────────────────────────────────────────────────
D = 16
HIDDEN = 64
N_LAYERS = 2
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'designs')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ─── Circular convolution in torch (log-domain) ───────────────────────────

def _circ_conv_torch(A, B):
    """Circular convolution of (..., 2, 2) log-prob tensors."""
    A00 = A[..., 0, 0]; A01 = A[..., 0, 1]
    A10 = A[..., 1, 0]; A11 = A[..., 1, 1]
    B00 = B[..., 0, 0]; B01 = B[..., 0, 1]
    B10 = B[..., 1, 0]; B11 = B[..., 1, 1]
    out = torch.empty_like(A)
    out[..., 0, 0] = torch.logaddexp(
        torch.logaddexp(A00+B00, A01+B01), torch.logaddexp(A10+B10, A11+B11))
    out[..., 0, 1] = torch.logaddexp(
        torch.logaddexp(A01+B00, A00+B01), torch.logaddexp(A11+B10, A10+B11))
    out[..., 1, 0] = torch.logaddexp(
        torch.logaddexp(A10+B00, A11+B01), torch.logaddexp(A00+B10, A01+B11))
    out[..., 1, 1] = torch.logaddexp(
        torch.logaddexp(A11+B00, A10+B01), torch.logaddexp(A01+B10, A00+B11))
    return out


# ─── Neural CalcParent ─────────────────────────────────────────────────────

class NeuralCalcParent(nn.Module):
    """
    Learned CalcParent: (left_emb, right_emb) -> parent_emb.
    Trained supervisedly to invert CalcLeft/CalcRight.
    parent = [calc_parent_first(left, right); calc_parent_second(right)]
    """
    def __init__(self, d, hidden, n_layers=2):
        super().__init__()
        self.d = d
        # First half: combines left and right info
        self.first_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        # Second half: projects right
        self.second_mlp = nn.Sequential(nn.Linear(d, d))

    def forward(self, left_emb, right_emb):
        """
        left_emb:  (B, L, d) — left child embedding
        right_emb: (B, L, d) — right child embedding
        Returns:   (B, 2L, d) — parent embedding
        """
        first = self.first_mlp(torch.cat([left_emb, right_emb], dim=-1))
        second = self.second_mlp(right_emb)
        return torch.cat([first, second], dim=1)


# ─── Model ──────────────────────────────────────────────────────────────────

class HybridMACDecoderV2(nn.Module):
    """
    4-class MAC decoder with:
    - fast_ce training for checknode/bitnode (Phase 1)
    - Supervised training for CalcParent (Phase 2)
    - Full CG tree walk inference
    """

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d

        # Phase 1: trained with fast_ce
        self.z_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)
        self.logits2emb = _make_mlp(4, hidden, d, n_layers)

        # Phase 2: trained supervisedly
        self.calc_parent = NeuralCalcParent(d, hidden, n_layers)

    def bitnode(self, e_odd, e_even, uv_left):
        """BitNode with 4-class sign-flip residual."""
        u_left = uv_left // 2
        v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([
            e_odd[:, :, :h] * u_sign,
            e_odd[:, :, h:] * v_sign], dim=-1)
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def count_parameters(self, phase=None):
        if phase == 1:
            exclude = set(self.calc_parent.parameters())
            return sum(p.numel() for p in self.parameters()
                       if p.requires_grad and p not in exclude)
        elif phase == 2:
            return sum(p.numel() for p in self.calc_parent.parameters()
                       if p.requires_grad)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Training: joint fast_ce + CalcParent ──────────────────────────────

    def fast_ce(self, emb, joint_cw):
        """
        Parallel teacher-forced training with three joint losses:
        1. fast_ce: 4-class CE at every depth (trains checknode/bitnode/emb2logits)
        2. CalcParent reconstruction: CalcParent(left, right) ≈ parent
           (trains CalcParent AND encourages checknode/bitnode to be invertible)
        3. logits2emb autoencoder: one_hot -> logits2emb -> emb2logits round-trip

        All O(log N) gradient depth. Returns (ce_loss, cp_loss, ae_loss).
        """
        B, N, d = emb.shape
        n = int(math.log2(N))
        ce_losses = []
        cp_losses = []
        ae_targets = []

        logits = self.emb2logits(emb)
        ce_losses.append(F.cross_entropy(
            logits.reshape(-1, 4), joint_cw.reshape(-1), reduction='mean'))
        ae_targets.append(joint_cw)

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

            E_odd = torch.cat(E_odds, 1)   # = parent[:l]
            E_even = torch.cat(E_evens, 1)  # = parent[l:]
            J_odd = torch.cat(J_odds, 1)
            J_even = torch.cat(J_evens, 1)

            u_o = J_odd // 2; v_o = J_odd % 2
            u_e = J_even // 2; v_e = J_even % 2
            J_left = (u_o ^ u_e) * 2 + (v_o ^ v_e)

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)

            # CalcParent reconstruction: (left, right) -> parent
            parent_target = torch.cat([E_odd, E_even], dim=1)  # (B, 2*M/2, d)
            parent_pred = self.calc_parent(e_left, e_right)    # (B, 2*M/2, d)
            cp_losses.append(F.mse_loss(parent_pred, parent_target))

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left, cs, 1)
            jr = torch.split(J_even, cs, 1)

            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

            e_all = torch.cat(E_chunks, 1)
            j_all = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_all)
            ce_losses.append(F.cross_entropy(
                logits.reshape(-1, 4), j_all.reshape(-1), reduction='mean'))
            ae_targets.append(j_all)

        ce_loss = torch.stack(ce_losses).mean()
        cp_loss = torch.stack(cp_losses).mean()

        # Autoencoder loss for logits2emb
        ae_losses = []
        for j_batch in ae_targets:
            one_hot = F.one_hot(j_batch, 4).float()
            log_one_hot = torch.where(one_hot > 0.5,
                                      torch.zeros_like(one_hot),
                                      torch.full_like(one_hot, -30.0))
            recon = self.emb2logits(self.logits2emb(log_one_hot))
            ae_losses.append(F.cross_entropy(
                recon.reshape(-1, 4), j_batch.reshape(-1), reduction='mean'))
        ae_loss = torch.stack(ae_losses).mean()

        return ce_loss, cp_loss, ae_loss

    # ── Phase 2: generate CalcParent training data ─────────────────────────

    @torch.no_grad()
    def generate_calc_parent_data(self, channel, N, batch, device, rng):
        """
        Run fast_ce forward pass, collect (left, right, parent) triples
        at every depth level.

        Returns list of (parent_emb, left_emb, right_emb) tensors.
        """
        n = int(math.log2(N))
        br = torch.from_numpy(np.array(bit_reversal_perm(n))).long().to(device)

        uf = rng.integers(0, 2, (batch, N)).astype(int)
        vf = rng.integers(0, 2, (batch, N)).astype(int)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float().to(device)

        emb = self.z_encoder(zf.unsqueeze(-1))[:, br]
        joint_cw = torch.from_numpy(xf * 2 + yf).long().to(device)[:, br]

        B, _, d = emb.shape
        pairs = []

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

            E_odd = torch.cat(E_odds, 1)  # = parent[:l]
            E_even = torch.cat(E_evens, 1)  # = parent[l:]
            J_odd = torch.cat(J_odds, 1)
            J_even = torch.cat(J_evens, 1)

            u_o = J_odd // 2; v_o = J_odd % 2
            u_e = J_even // 2; v_e = J_even % 2
            J_left = (u_o ^ u_e) * 2 + (v_o ^ v_e)

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)

            # parent_emb = cat(E_odd, E_even) = original parent before split
            parent_emb = torch.cat([E_odd, E_even], dim=1)

            pairs.append((parent_emb.detach(), e_left.detach(), e_right.detach(),
                          J_left.detach()))

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left, cs, 1)
            jr = torch.split(J_even, cs, 1)

            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

        return pairs

    # ── Phase 2: CalcParent training step ──────────────────────────────────

    def calc_parent_loss(self, pairs, joint_cw_pairs=None):
        """
        Combined loss for CalcParent:
        1. MSE: CalcParent(left, right) ≈ parent
        2. Round-trip: checknode(CalcParent_first, CalcParent_second) ≈ left
        3. Round-trip: bitnode(CalcParent_first, CalcParent_second, cw) ≈ right
        """
        mse_total = 0.0
        rt_total = 0.0
        count = 0
        for i, (parent_emb, left_emb, right_emb) in enumerate(pairs):
            pred_parent = self.calc_parent(left_emb, right_emb)
            l = left_emb.shape[1]

            # MSE loss
            mse_total += F.mse_loss(pred_parent, parent_emb)

            # Round-trip through checknode: should recover left
            pred_left = self.checknode(
                torch.cat([pred_parent[:, :l], pred_parent[:, l:]], dim=-1))
            rt_total += F.mse_loss(pred_left, left_emb)

            # Round-trip through bitnode: should recover right
            if joint_cw_pairs is not None:
                j_left = joint_cw_pairs[i]
                pred_right = self.bitnode(
                    pred_parent[:, :l], pred_parent[:, l:], j_left)
                rt_total += F.mse_loss(pred_right, right_emb)

            count += 1
        return (mse_total + rt_total) / count

    # ── MI measurement ─────────────────────────────────────────────────────

    @torch.no_grad()
    def measure_mi(self, channel, N, device, n_samples=5000, batch=100):
        """
        Measure per-position 4-class MI under teacher forcing.
        MI_i = (log(4) - CE_i) / log(4), in [0, 1].
        """
        n = int(math.log2(N))
        br = torch.from_numpy(np.array(bit_reversal_perm(n))).long().to(device)
        rng = np.random.default_rng(789)

        leaf_ce = np.zeros(N)
        count = 0
        self.eval()

        while count < n_samples:
            actual = min(batch, n_samples - count)
            uf = rng.integers(0, 2, (actual, N)).astype(int)
            vf = rng.integers(0, 2, (actual, N)).astype(int)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float().to(device)

            emb = self.z_encoder(zf.unsqueeze(-1))[:, br]
            joint_cw = torch.from_numpy(xf * 2 + yf).long().to(device)[:, br]

            B, N_, d = emb.shape
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

                u_o = J_odd // 2; v_o = J_odd % 2
                u_e = J_even // 2; v_e = J_even % 2
                J_left = (u_o ^ u_e) * 2 + (v_o ^ v_e)

                e_left = self.checknode(torch.cat([E_odd, E_even], -1))
                e_right = self.bitnode(E_odd, E_even, J_left)

                nc = 2 ** depth
                cs = (N_ // 2) // nc
                el = torch.split(e_left, cs, 1)
                er = torch.split(e_right, cs, 1)
                jl = torch.split(J_left, cs, 1)
                jr = torch.split(J_even, cs, 1)

                E_chunks, J_chunks = [], []
                for a, b, c, dd in zip(el, er, jl, jr):
                    E_chunks += [a, b]
                    J_chunks += [c, dd]

            # At leaves: measure per-position CE
            e_leaf = torch.cat(E_chunks, 1)
            j_leaf = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_leaf)
            ce = F.cross_entropy(logits.reshape(-1, 4), j_leaf.reshape(-1),
                                 reduction='none').reshape(B, N_)
            leaf_ce += ce.sum(0).cpu().numpy()
            count += actual

        self.train()
        avg_ce = leaf_ce / count

        # Map back to natural order
        ce_nat = np.zeros(N)
        for t in range(N):
            ce_nat[br[t]] = avg_ce[t]

        # MI: (log(4) - CE) / log(4), clamped to [0, 1]
        mi_nat = np.clip((np.log(4) - ce_nat) / np.log(4), 0, 1)

        return float(np.mean(mi_nat)), float(np.min(mi_nat)), mi_nat.tolist()

    # ── Inference: CG tree walk ────────────────────────────────────────────

    @torch.no_grad()
    def cg_decode(self, z, b, frozen_u, frozen_v, use_analytical_cp=True):
        """Full CG tree walk decode. use_analytical_cp: analytical vs neural CalcParent."""
        B, N = z.shape
        n = int(math.log2(N))
        d = self.d
        device = z.device
        LH = math.log(0.5)

        br = torch.from_numpy(np.array(bit_reversal_perm(n))).long().to(device)
        root_emb = self.z_encoder(z.unsqueeze(-1))[:, br]

        edge_emb = [None] * (2 * N)
        edge_emb[1] = root_emb

        uniform_lp = torch.full((1, 1, 4), math.log(0.25), device=device)
        no_info = self.logits2emb(uniform_lp)
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_emb[beta] = no_info.expand(B, size, d).clone()

        dec_head = 1
        u_hat, v_hat = {}, {}
        i_u, i_v = 0, 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = frozen_v

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1

            dec_head = self._cg_step_to(dec_head, target_vtx, edge_emb, use_analytical_cp)
            temp = edge_emb[leaf_edge][:, 0].clone()

            if leaf_edge & 1 == 0:
                self._neural_calc_left(target_vtx, edge_emb)
            else:
                self._neural_calc_right(target_vtx, edge_emb)
            top_down = edge_emb[leaf_edge][:, 0]

            combined = top_down + temp
            logits = self.emb2logits(combined)

            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.long, device=device)
            else:
                if gamma == 0:
                    p0 = torch.logsumexp(logits[:, :2], dim=1)
                    p1 = torch.logsumexp(logits[:, 2:], dim=1)
                else:
                    p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                    p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                bit = (p1 > p0).long()

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            new_lp = torch.full((B, 4), -30.0, device=device)
            u_val = u_hat.get(i_t)
            v_val = v_hat.get(i_t)
            if u_val is not None and v_val is not None:
                new_lp.scatter_(1, (u_val * 2 + v_val).unsqueeze(1), 0.0)
            elif u_val is not None:
                new_lp.scatter_(1, (u_val * 2).unsqueeze(1), LH)
                new_lp.scatter_(1, (u_val * 2 + 1).unsqueeze(1), LH)
            elif v_val is not None:
                new_lp.scatter_(1, v_val.unsqueeze(1), LH)
                new_lp.scatter_(1, (v_val + 2).unsqueeze(1), LH)
            else:
                new_lp.fill_(math.log(0.25))

            edge_emb[leaf_edge] = self.logits2emb(new_lp).unsqueeze(1)

        u_dec = torch.zeros(B, N, dtype=torch.long, device=device)
        v_dec = torch.zeros(B, N, dtype=torch.long, device=device)
        for k in range(1, N + 1):
            if k in u_hat: u_dec[:, k - 1] = u_hat[k]
            if k in v_hat: v_dec[:, k - 1] = v_hat[k]
        return u_dec, v_dec

    # ── CG tree walk helpers ───────────────────────────────────────────────

    def _neural_calc_left(self, beta, edge_emb):
        parent = edge_emb[beta]
        right = edge_emb[2 * beta + 1]
        l = right.shape[1]
        edge_emb[2 * beta] = self.checknode(
            torch.cat([parent[:, :l], parent[:, l:]], dim=-1))

    def _neural_calc_right(self, beta, edge_emb):
        parent = edge_emb[beta]
        left = edge_emb[2 * beta]
        l = left.shape[1]
        left_logits = self.emb2logits(left)
        left_cw = left_logits.argmax(dim=-1)
        edge_emb[2 * beta + 1] = self.bitnode(
            parent[:, :l], parent[:, l:], left_cw)

    def _neural_calc_parent(self, beta, edge_emb):
        """Neural CalcParent."""
        left = edge_emb[2 * beta]
        right = edge_emb[2 * beta + 1]
        edge_emb[beta] = self.calc_parent(left, right)

    def _analytical_calc_parent(self, beta, edge_emb):
        """Analytical CalcParent: emb -> prob -> circconv -> prob -> emb."""
        left = edge_emb[2 * beta]
        right = edge_emb[2 * beta + 1]
        l = left.shape[1]
        left_lp = F.log_softmax(self.emb2logits(left), dim=-1).reshape(-1, l, 2, 2)
        right_lp = F.log_softmax(self.emb2logits(right), dim=-1).reshape(-1, l, 2, 2)
        parent_first = _circ_conv_torch(left_lp, right_lp)
        parent_lp = torch.cat([parent_first, right_lp], dim=1).reshape(-1, 2*l, 4)
        edge_emb[beta] = self.logits2emb(parent_lp)

    def _cg_step_one(self, current, beta, edge_emb, use_analytical_cp=True):
        if current == beta >> 1:
            if beta & 1 == 0:
                self._neural_calc_left(current, edge_emb)
            else:
                self._neural_calc_right(current, edge_emb)
            return beta
        elif beta == current >> 1:
            if use_analytical_cp:
                self._analytical_calc_parent(current, edge_emb)
            else:
                self._neural_calc_parent(current, edge_emb)
            return beta
        else:
            raise ValueError(f"Invalid step: {current} -> {beta}")

    @staticmethod
    def _cg_get_path(current, target):
        if current == target:
            return []
        path_up, path_down = [], []
        c, t = current, target
        while c != t:
            if c > t:
                c >>= 1; path_up.append(c)
            else:
                path_down.append(t); t >>= 1
        path_down.reverse()
        return path_up + path_down

    def _cg_step_to(self, current, target, edge_emb, use_analytical_cp=True):
        if current == target:
            return current
        for beta in self._cg_get_path(current, target):
            current = self._cg_step_one(current, beta, edge_emb, use_analytical_cp)
        return current


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_cg(model, channel, N, b, frozen_u, frozen_v, Au, Av, n_cw,
                device, batch_size=50):
    rng = np.random.default_rng(999)
    errs = 0
    model.eval()

    for start in range(0, n_cw, batch_size):
        bs = min(batch_size, n_cw - start)
        uf = np.zeros((bs, N), dtype=int)
        vf = np.zeros((bs, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, bs)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, bs)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float().to(device)

        u_dec, v_dec = model.cg_decode(zf, b, frozen_u, frozen_v)

        for i in range(bs):
            ue = any(u_dec[i, p - 1].item() != uf[i, p - 1] for p in Au)
            ve = any(v_dec[i, p - 1].item() != vf[i, p - 1] for p in Av)
            if ue or ve: errs += 1

    model.train()
    return errs / n_cw


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        return design_from_file(mc_path, n, ku, kv)
    from polar.design import design_gmac
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv, None, None, None


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--phase1-iters', type=int, default=100000)
    parser.add_argument('--phase2-iters', type=int, default=50000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--eval-cw', type=int, default=500)
    parser.add_argument('--ku', type=int, default=None)
    parser.add_argument('--kv', type=int, default=None)
    args = parser.parse_args()

    N = args.N
    n = int(math.log2(N))
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # Path and design
    path_i = N // 2
    b = make_path(N, path_i)
    channel = GaussianMAC(sigma2=SIGMA2)

    # Standard rates from PROJECT_RESULTS.md (Class B, GMAC SNR=6dB)
    standard_rates = {
        16: (8, 8), 32: (15, 15), 64: (31, 31), 128: (62, 62),
        256: (123, 123), 512: (246, 246), 1024: (492, 492),
    }
    if args.ku is not None and args.kv is not None:
        ku, kv = args.ku, args.kv
    elif N in standard_rates:
        ku, kv = standard_rates[N]
    else:
        ku = N // 2
        kv = N // 2

    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    frozen_u = {p: 0 for p in fu}
    frozen_v = {p: 0 for p in fv}

    print(f'N={N}, ku={len(Au)}, kv={len(Av)}, path_i={path_i}')

    sc_bler_ref = {32: 0.047, 64: 0.028, 128: 0.020, 256: 0.006}
    sc_bler = sc_bler_ref.get(N, None)
    if sc_bler:
        print(f'SC BLER reference: {sc_bler}')

    model = HybridMACDecoderV2(d=D, hidden=HIDDEN, n_layers=N_LAYERS).to(device)
    print(f'Total params: {model.count_parameters():,}')

    results = {
        'config': {
            'N': N, 'ku': len(Au), 'kv': len(Av), 'path_i': path_i,
            'd': D, 'hidden': HIDDEN, 'snr_db': SNR_DB,
            'iters': args.phase1_iters, 'batch': args.batch,
            'device': str(device), 'sc_bler': sc_bler,
        },
        'training': [],
        'final': {},
    }

    br = torch.from_numpy(np.array(bit_reversal_perm(n))).long().to(device)
    rng = np.random.default_rng(42)

    # --- Joint training: fast_ce + CalcParent + logits2emb ---
    total_iters = args.phase1_iters
    print(f'\n{"="*60}')
    print(f'JOINT TRAINING ({total_iters} iters)')
    print(f'fast_ce + CalcParent reconstruction + logits2emb AE')
    print(f'{"="*60}')

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    t0 = time.time()

    for it in range(1, total_iters + 1):
        uf = rng.integers(0, 2, (args.batch, N)).astype(int)
        vf = rng.integers(0, 2, (args.batch, N)).astype(int)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float().to(device)
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        joint_cw = torch.from_numpy(xf * 2 + yf).long().to(device)[:, br]

        ce_loss, cp_loss, ae_loss = model.fast_ce(emb, joint_cw)
        loss = ce_loss + 1.0 * cp_loss + 0.1 * ae_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % args.eval_every == 0 or it == 1:
            elapsed = time.time() - t0
            mi_avg, mi_min, mi_per_pos = model.measure_mi(
                channel, N, device, n_samples=2000, batch=100)
            bler_cg = evaluate_cg(
                model, channel, N, b, frozen_u, frozen_v, Au, Av,
                args.eval_cw, device)

            entry = {
                'iter': it, 'ce_loss': ce_loss.item(),
                'cp_loss': cp_loss.item(), 'ae_loss': ae_loss.item(),
                'mi_avg': mi_avg, 'mi_min': mi_min,
                'mi_per_pos': mi_per_pos, 'bler_cg': bler_cg,
                'elapsed_min': elapsed / 60,
            }
            results['training'].append(entry)

            sc_str = f' (SC={sc_bler})' if sc_bler else ''
            r = f' {bler_cg/sc_bler:.1f}x' if sc_bler and bler_cg < 1 else ''
            print(f'[{it:>6}/{total_iters}] '
                  f'ce={ce_loss.item():.4f} cp={cp_loss.item():.4f} '
                  f'ae={ae_loss.item():.4f} | '
                  f'MI={mi_avg:.4f}/{mi_min:.4f} | '
                  f'BLER={bler_cg:.4f}{sc_str}{r} | '
                  f'{elapsed/60:.1f}min', flush=True)

    train_time = time.time() - t0
    print(f'\nTraining done: {train_time/60:.1f} min')

    ckpt_path = os.path.join(RESULTS_DIR, f'hybrid_v2_N{N}.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f'Saved: {ckpt_path}')

    # --- Final evaluation ---
    print(f'\n{"="*60}')
    print(f'FINAL EVALUATION')
    print(f'{"="*60}')

    bler_final = evaluate_cg(
        model, channel, N, b, frozen_u, frozen_v, Au, Av, 2000, device)
    mi_avg, mi_min, mi_per_pos = model.measure_mi(
        channel, N, device, n_samples=5000, batch=100)

    results['final'] = {
        'bler_cg': bler_final, 'mi_avg': mi_avg, 'mi_min': mi_min,
        'mi_per_pos': mi_per_pos, 'train_time_min': train_time / 60,
    }

    sc_str = f' (SC={sc_bler})' if sc_bler else ''
    ratio = f' ({bler_final/sc_bler:.2f}x SC)' if sc_bler and sc_bler > 0 else ''
    print(f'\nClass B CG decode:  BLER={bler_final:.4f}{sc_str}{ratio}')
    print(f'MI:                 avg={mi_avg:.4f}  min={mi_min:.4f}')
    print(f'Training time:      {train_time/60:.1f} min')

    json_path = os.path.join(RESULTS_DIR, f'hybrid_v2_N{N}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {json_path}')


if __name__ == '__main__':
    main()
