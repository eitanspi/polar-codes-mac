#!/usr/bin/env python3
"""
POC: Hybrid Class B decoder — fast_ce training + CG tree walk inference.

Architecture:
  - 4-class joint (u,v) output per tree position
  - Trained with fast_ce (O(log N) gradient depth, parallel top-down)
  - Inference via full CG tree walk (handles CalcParent for non-corner paths)
  - Neural CalcLeft/CalcRight (checknode/bitnode from fast_ce training)
  - Analytical CalcParent (emb -> prob -> circconv -> prob -> emb)
  - logits2emb trained as autoencoder alongside fast_ce

Tests Class B (path_i = N/2) at N=32 on GMAC SNR=6dB.
SC baseline BLER ~ 0.046.
"""
import sys, os, math, time
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


def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ─── Circular convolution in torch (log-domain) ───────────────────────────

def _circ_conv_torch(A, B):
    """
    Circular convolution of (*, 2, 2) log-prob tensors.
    out[a,b] = logaddexp_{a',b'} A[a^a', b^b'] + B[a', b']
    """
    A00 = A[..., 0, 0]; A01 = A[..., 0, 1]
    A10 = A[..., 1, 0]; A11 = A[..., 1, 1]
    B00 = B[..., 0, 0]; B01 = B[..., 0, 1]
    B10 = B[..., 1, 0]; B11 = B[..., 1, 1]

    out = torch.empty_like(A)
    out[..., 0, 0] = torch.logaddexp(
        torch.logaddexp(A00 + B00, A01 + B01),
        torch.logaddexp(A10 + B10, A11 + B11))
    out[..., 0, 1] = torch.logaddexp(
        torch.logaddexp(A01 + B00, A00 + B01),
        torch.logaddexp(A11 + B10, A10 + B11))
    out[..., 1, 0] = torch.logaddexp(
        torch.logaddexp(A10 + B00, A11 + B01),
        torch.logaddexp(A00 + B10, A01 + B11))
    out[..., 1, 1] = torch.logaddexp(
        torch.logaddexp(A11 + B00, A10 + B01),
        torch.logaddexp(A01 + B10, A00 + B11))
    return out


# ─── Model ──────────────────────────────────────────────────────────────────

class HybridMACDecoder(nn.Module):
    """
    4-class MAC decoder: fast_ce training + CG tree walk inference.

    Training: parallel fast_ce with 4-class cross-entropy at each depth.
    Inference: full CG tree walk supporting CalcParent for non-corner paths.
      - CalcLeft/CalcRight: neural (checknode/bitnode)
      - CalcParent: analytical (emb <-> prob conversion)
    """

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d

        # Channel encoder
        self.z_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))

        # Tree operations (from fast_ce)
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)    # CalcLeft
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)  # CalcRight core

        # Decision head + re-embedding
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)
        self.logits2emb = _make_mlp(4, hidden, d, n_layers)  # for CalcParent + leaf re-embedding

    def bitnode(self, e_odd, e_even, uv_left):
        """
        BitNode with 4-class sign-flip residual.
        uv_left: (B, M) integer 0-3 = u*2 + v.
        First half of d flipped by u_sign, second half by v_sign.
        """
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Training: fast_ce ──────────────────────────────────────────────────

    def fast_ce(self, emb, joint_cw):
        """
        Parallel teacher-forced 4-class cross-entropy over all tree depths.

        emb:      (B, N, d)  channel embeddings (bit-reversed)
        joint_cw: (B, N) int — codeword classes 0-3 = xu*2 + xv (bit-reversed)

        Returns: (fast_ce_loss, ae_loss) tuple.
        """
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        # Depth 0: predict from raw embeddings
        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(
            logits.reshape(-1, 4), joint_cw.reshape(-1), reduction='mean'))

        E_chunks = [emb]
        J_chunks = [joint_cw]

        # Collect all embeddings for autoencoder loss
        all_embs = [emb]
        all_targets = [joint_cw]

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

            # Codeword relationship: left = odd XOR even, right = even
            u_o = J_odd // 2; v_o = J_odd % 2
            u_e = J_even // 2; v_e = J_even % 2
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
            all_losses.append(F.cross_entropy(
                logits.reshape(-1, 4), j_all.reshape(-1), reduction='mean'))

            all_embs.append(e_all)
            all_targets.append(j_all)

        fast_ce_loss = torch.stack(all_losses).mean()

        # Autoencoder loss for logits2emb: one_hot -> logits2emb -> emb2logits
        # should recover the original class. Trains logits2emb end-to-end.
        ae_losses = []
        for e_batch, j_batch in zip(all_embs, all_targets):
            one_hot = F.one_hot(j_batch, 4).float()
            log_one_hot = torch.where(one_hot > 0.5,
                                      torch.zeros_like(one_hot),
                                      torch.full_like(one_hot, -30.0))
            recon = self.emb2logits(self.logits2emb(log_one_hot))
            ae_losses.append(F.cross_entropy(
                recon.reshape(-1, 4), j_batch.reshape(-1), reduction='mean'))
        ae_loss = torch.stack(ae_losses).mean()

        return fast_ce_loss, ae_loss

    # ── Inference: CG tree walk ────────────────────────────────────────────

    @torch.no_grad()
    def cg_decode(self, z, b, frozen_u, frozen_v):
        """
        Full CG tree walk decode for any monotone chain path.

        z:        (B, N) float — channel output
        b:        list of 2N ints — path (0=U, 1=V)
        frozen_u: dict {1-indexed: 0} — frozen U positions
        frozen_v: dict {1-indexed: 0} — frozen V positions

        Returns: u_dec (B, N), v_dec (B, N) int tensors (0-indexed).
        """
        B, N = z.shape
        n = int(math.log2(N))
        d = self.d
        device = z.device
        LH = math.log(0.5)

        br = torch.from_numpy(np.array(bit_reversal_perm(n))).long().to(device)

        # Root: channel embeddings, bit-reversed
        root_emb = self.z_encoder(z.unsqueeze(-1))[:, br]  # (B, N, d)
        edge_emb = [None] * (2 * N)
        edge_emb[1] = root_emb

        # No-info embedding: logits2emb(uniform)
        uniform_lp = torch.full((1, 1, 4), math.log(0.25), device=device)
        no_info = self.logits2emb(uniform_lp)  # (1, 1, d)
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_emb[beta] = no_info.expand(B, size, d).clone()

        dec_head = 1
        u_hat = {}  # 1-indexed -> (B,) tensor
        v_hat = {}
        i_u, i_v = 0, 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = frozen_v

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1

            # Navigate to target vertex
            dec_head = self._cg_step_to(dec_head, target_vtx, edge_emb)

            # Save leaf's bottom-up state
            temp = edge_emb[leaf_edge][:, 0].clone()  # (B, d)

            # CalcLeft or CalcRight to get top-down at leaf
            if leaf_edge & 1 == 0:
                self._neural_calc_left(target_vtx, edge_emb)
            else:
                self._neural_calc_right(target_vtx, edge_emb)
            top_down = edge_emb[leaf_edge][:, 0]  # (B, d)

            # Combine top-down + bottom-up
            combined = top_down + temp
            logits = self.emb2logits(combined)  # (B, 4)

            # Hard decision
            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.long, device=device)
            else:
                if gamma == 0:  # U decision: marginalize over V
                    p0 = torch.logsumexp(logits[:, :2], dim=1)   # P(u=0)
                    p1 = torch.logsumexp(logits[:, 2:], dim=1)   # P(u=1)
                else:  # V decision: marginalize over U
                    p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)  # P(v=0)
                    p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)  # P(v=1)
                bit = (p1 > p0).long()

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            # Update leaf with partial decision via logits2emb
            new_lp = torch.full((B, 4), -30.0, device=device)
            u_val = u_hat.get(i_t)
            v_val = v_hat.get(i_t)

            if u_val is not None and v_val is not None:
                idx = u_val * 2 + v_val  # (B,)
                new_lp.scatter_(1, idx.unsqueeze(1), 0.0)
            elif u_val is not None:
                new_lp.scatter_(1, (u_val * 2).unsqueeze(1), LH)
                new_lp.scatter_(1, (u_val * 2 + 1).unsqueeze(1), LH)
            elif v_val is not None:
                new_lp.scatter_(1, v_val.unsqueeze(1), LH)
                new_lp.scatter_(1, (v_val + 2).unsqueeze(1), LH)
            else:
                new_lp.fill_(math.log(0.25))

            edge_emb[leaf_edge] = self.logits2emb(new_lp).unsqueeze(1)

        # Collect results
        u_dec = torch.zeros(B, N, dtype=torch.long, device=device)
        v_dec = torch.zeros(B, N, dtype=torch.long, device=device)
        for k in range(1, N + 1):
            if k in u_hat:
                u_dec[:, k - 1] = u_hat[k]
            if k in v_hat:
                v_dec[:, k - 1] = v_hat[k]

        return u_dec, v_dec

    # ── CG tree walk navigation ────────────────────────────────────────────

    def _neural_calc_left(self, beta, edge_emb):
        """CalcLeft: checknode(parent_first, parent_second) -> left child."""
        parent = edge_emb[beta]
        right = edge_emb[2 * beta + 1]
        l = right.shape[1]
        p_first = parent[:, :l]
        p_second = parent[:, l:]
        edge_emb[2 * beta] = self.checknode(
            torch.cat([p_first, p_second], dim=-1))

    def _neural_calc_right(self, beta, edge_emb):
        """CalcRight: bitnode(parent_first, parent_second, left_cw) -> right child."""
        parent = edge_emb[beta]
        left = edge_emb[2 * beta]
        l = left.shape[1]
        p_first = parent[:, :l]
        p_second = parent[:, l:]
        # Extract left child's codeword class from its embedding
        left_logits = self.emb2logits(left)  # (B, l, 4)
        left_cw = left_logits.argmax(dim=-1)  # (B, l) in {0,1,2,3}
        edge_emb[2 * beta + 1] = self.bitnode(p_first, p_second, left_cw)

    def _analytical_calc_parent(self, beta, edge_emb):
        """CalcParent: emb -> prob -> analytical circconv -> prob -> emb."""
        left = edge_emb[2 * beta]       # (B, l, d)
        right = edge_emb[2 * beta + 1]  # (B, l, d)
        l = left.shape[1]

        # Convert to log-prob (2x2)
        left_lp = F.log_softmax(self.emb2logits(left), dim=-1)
        right_lp = F.log_softmax(self.emb2logits(right), dim=-1)
        left_22 = left_lp.reshape(-1, l, 2, 2)
        right_22 = right_lp.reshape(-1, l, 2, 2)

        # Analytical circular convolution
        parent_first_22 = _circ_conv_torch(left_22, right_22)

        # parent = [circconv(left, right) ; right]
        parent_22 = torch.cat([parent_first_22, right_22], dim=1)  # (B, 2l, 2, 2)
        parent_flat = parent_22.reshape(-1, 2 * l, 4)

        # Convert back to embedding
        edge_emb[beta] = self.logits2emb(parent_flat)

    def _cg_step_one(self, current, beta, edge_emb):
        """Execute one navigation step."""
        if current == beta >> 1:
            # Going DOWN: current is parent of beta
            if beta & 1 == 0:
                self._neural_calc_left(current, edge_emb)
            else:
                self._neural_calc_right(current, edge_emb)
            return beta
        elif beta == current >> 1:
            # Going UP: beta is parent of current
            self._analytical_calc_parent(current, edge_emb)
            return beta
        else:
            raise ValueError(f"Invalid step: current={current}, target={beta}")

    @staticmethod
    def _cg_get_path(current, target):
        """Compute traversal path from current to target vertex."""
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

    def _cg_step_to(self, current, target, edge_emb):
        """Navigate from current vertex to target vertex."""
        if current == target:
            return current
        for beta in self._cg_get_path(current, target):
            current = self._cg_step_one(current, beta, edge_emb)
        return current

    # ── Also provide top-down recursive decode (corner paths only) ─────────

    @torch.no_grad()
    def recursive_decode(self, emb, frozen_u, frozen_v):
        """
        Top-down recursive SC decode. Only works for corner paths (Class C).
        frozen_u, frozen_v: sets of 0-indexed frozen positions.
        """
        B = emb.shape[0]
        N = emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(emb_block):
            bs = emb_block.shape[1]
            if bs == 1:
                logits = self.emb2logits(emb_block[:, 0, :])
                idx = leaf_idx[0]
                leaf_idx[0] += 1

                u_frz = idx in frozen_u
                v_frz = idx in frozen_v

                if u_frz and v_frz:
                    dec = torch.zeros(B, dtype=torch.long)
                elif u_frz:
                    dec = (logits[:, 1] > logits[:, 0]).long()
                elif v_frz:
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
            u_l = uv_left // 2; v_l = uv_left % 2
            u_r = uv_right // 2; v_r = uv_right % 2
            cw = torch.zeros(B, bs, dtype=torch.long)
            cw[:, 0::2] = (u_l ^ u_r) * 2 + (v_l ^ v_r)
            cw[:, 1::2] = uv_right
            return cw

        _decode(emb)
        return u_hat, v_hat


# ─── Training + Evaluation ─────────────────────────────────────────────────

def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        return design_from_file(mc_path, n, ku, kv)
    from polar.design import design_gmac
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv, None, None, None


def evaluate_cg(model, channel, N, b, frozen_u, frozen_v, Au, Av, n_cw,
                device='cpu'):
    """Evaluate using CG tree walk decode (works for any path)."""
    rng = np.random.default_rng(999)
    errs = 0
    model.eval()
    batch_size = min(50, n_cw)

    for start in range(0, n_cw, batch_size):
        bs = min(batch_size, n_cw - start)
        uf = np.zeros((bs, N), dtype=int)
        vf = np.zeros((bs, N), dtype=int)
        for p in Au:
            uf[:, p - 1] = rng.integers(0, 2, bs)
        for p in Av:
            vf[:, p - 1] = rng.integers(0, 2, bs)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float().to(device)

        u_dec, v_dec = model.cg_decode(zf, b, frozen_u, frozen_v)

        for i in range(bs):
            ue = any(u_dec[i, p - 1].item() != uf[i, p - 1] for p in Au)
            ve = any(v_dec[i, p - 1].item() != vf[i, p - 1] for p in Av)
            if ue or ve:
                errs += 1

    model.train()
    return errs / n_cw


def evaluate_recursive(model, channel, N, Au, Av, fu_set, fv_set, n_cw,
                       device='cpu'):
    """Evaluate using top-down recursive decode (for Class C sanity check)."""
    n = int(math.log2(N))
    br = torch.from_numpy(np.array(bit_reversal_perm(n))).long().to(device)
    rng = np.random.default_rng(999)
    errs = 0
    model.eval()

    for _ in range(n_cw):
        uf = np.zeros((1, N), dtype=int)
        vf = np.zeros((1, N), dtype=int)
        for p in Au:
            uf[0, p - 1] = rng.integers(0, 2)
        for p in Av:
            vf[0, p - 1] = rng.integers(0, 2)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float().to(device)
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        u_dec, v_dec = model.recursive_decode(emb, fu_set, fv_set)
        ue = any(u_dec[0, p - 1].item() != uf[0, p - 1] for p in Au)
        ve = any(v_dec[0, p - 1].item() != vf[0, p - 1] for p in Av)
        if ue or ve:
            errs += 1

    model.train()
    return errs / n_cw


def main():
    N = 32
    BATCH = 128
    ITERS = 50000
    n = int(math.log2(N))

    # Class B path
    path_i = N // 2
    b = make_path(N, path_i)
    print(f'Class B path (N={N}, path_i={path_i}): '
          f'{sum(1 for x in b if x==0)} U-steps, {sum(1 for x in b if x==1)} V-steps')

    channel = GaussianMAC(sigma2=SIGMA2)

    # Load design
    ku, kv = 15, 15
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    frozen_u = {p: 0 for p in fu}
    frozen_v = {p: 0 for p in fv}
    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}

    print(f'N={N}, ku={len(Au)}, kv={len(Av)}')
    print(f'SC BLER baseline: ~0.046')

    # Model
    model = HybridMACDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    print(f'Model: d={D}, hidden={HIDDEN}, params={model.count_parameters():,}')

    # Training
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    br = torch.from_numpy(np.array(bit_reversal_perm(n))).long()
    t0 = time.time()

    print(f'\n--- Training fast_ce (rate 1, all positions, {ITERS} iters) ---')

    for it in range(1, ITERS + 1):
        # Rate 1: all positions are info
        uf = rng.integers(0, 2, (BATCH, N)).astype(int)
        vf = rng.integers(0, 2, (BATCH, N)).astype(int)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        joint_cw = torch.from_numpy(xf * 2 + yf).long()[:, br]

        fast_ce_loss, ae_loss = model.fast_ce(emb, joint_cw)
        loss = fast_ce_loss + 0.1 * ae_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0 or it == 1:
            elapsed = time.time() - t0

            # CG decode on Class B
            bler_cg = evaluate_cg(
                model, channel, N, b, frozen_u, frozen_v, Au, Av, 500)

            # Recursive decode (sanity: should match the old poc_joint_fastce)
            bler_rec = evaluate_recursive(
                model, channel, N, Au, Av, fu_set, fv_set, 500)

            print(f'[{it:>5}/{ITERS}] ce={fast_ce_loss.item():.4f} '
                  f'ae={ae_loss.item():.4f} | '
                  f'CG_BLER={bler_cg:.4f}  Rec_BLER={bler_rec:.4f}  '
                  f'(SC=0.046) {elapsed/60:.1f}min', flush=True)

    # Final evaluation
    print(f'\n--- Final evaluation (2000 cw) ---')
    bler_cg = evaluate_cg(
        model, channel, N, b, frozen_u, frozen_v, Au, Av, 2000)
    bler_rec = evaluate_recursive(
        model, channel, N, Au, Av, fu_set, fv_set, 2000)
    print(f'Class B CG decode:    BLER={bler_cg:.4f}')
    print(f'Recursive decode:     BLER={bler_rec:.4f}')
    print(f'SC baseline:          BLER=0.046')

    # Also test CG decode on Class C path (sanity: should work since no CalcParent)
    b_classC = make_path(N, N)
    bler_cg_c = evaluate_cg(
        model, channel, N, b_classC, frozen_u, frozen_v, Au, Av, 500)
    print(f'Class C CG decode:    BLER={bler_cg_c:.4f} (sanity check, same frozen set)')

    # Diagnostic: how often does CalcParent fire?
    n_up = 0
    n_down = 0
    dec_head = 1
    i_u, i_v = 0, 0
    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u
        else:
            i_v += 1; i_t = i_v
        leaf_edge = i_t + N - 1
        target_vtx = leaf_edge >> 1
        path = HybridMACDecoder._cg_get_path(dec_head, target_vtx)
        for beta in path:
            if beta == dec_head >> 1:
                n_up += 1
            else:
                n_down += 1
            # Simulate navigation
            if dec_head == beta >> 1:
                dec_head = beta
            elif beta == dec_head >> 1:
                dec_head = beta
        # Account for leaf CalcLeft/CalcRight
        n_down += 1  # the explicit CalcLeft/CalcRight at the leaf
        dec_head = target_vtx

    print(f'\nTree walk stats: {n_up} CalcParent (up), {n_down} CalcLeft/Right (down)')


if __name__ == '__main__':
    main()
