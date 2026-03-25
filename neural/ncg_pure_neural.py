"""
ncg_pure_neural.py — Neural Computational Graph SC Decoder with PURE NEURAL CalcParent.

Uses a learned gated residual module for CalcParent:

    NeuralCalcParent: R^d x R^d -> R^d  (O(md) complexity, channel-independent)

The second-half of parent (parent[l:]) is kept as the right child embedding,
which mirrors the polar code structure (not channel-dependent).

Supports knowledge distillation from the analytical CalcParent teacher
during training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from polar.encoder import bit_reversal_perm


# ─── MLP helper ──────────────────────────────────────────────────────────────

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ─── Differentiable circular convolution (teacher only) ───────────────────────

def circ_conv_torch(A, B):
    """Circular convolution on (*, 2, 2) log-prob tensors (used by teacher)."""
    A00, A01, A10, A11 = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    B00, B01, B10, B11 = B[..., 0, 0], B[..., 0, 1], B[..., 1, 0], B[..., 1, 1]

    out = torch.empty_like(A)
    out[..., 0, 0] = torch.logsumexp(torch.stack([A00+B00, A01+B01, A10+B10, A11+B11], -1), -1)
    out[..., 0, 1] = torch.logsumexp(torch.stack([A01+B00, A00+B01, A11+B10, A10+B11], -1), -1)
    out[..., 1, 0] = torch.logsumexp(torch.stack([A10+B00, A11+B01, A00+B10, A01+B11], -1), -1)
    out[..., 1, 1] = torch.logsumexp(torch.stack([A11+B00, A10+B01, A01+B10, A00+B11], -1), -1)
    return out


# ─── Neural CalcParent Module ────────────────────────────────────────────────

class NeuralCalcParent(nn.Module):
    """
    Gated residual for calcParent first half.

    Given left_emb (B, l, d) and right_emb (B, l, d):
        concat = cat(left, right)   # (B, l, 2d)
        gate = sigmoid(Wg @ concat + bg)       # what to keep from residual
        candidate = ELU(Wc @ concat + bc)      # new candidate content
        parent_first = gate * candidate + (1-gate) * (left + right) / 2

    The residual (left+right)/2 provides a stable starting point.
    """
    def __init__(self, d, hidden, n_layers=2):
        super().__init__()
        self.d = d
        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d, hidden),
            nn.ELU(),
            nn.Linear(hidden, d),
            nn.Sigmoid()
        )
        # Candidate network (deeper)
        self.candidate_net = _make_mlp(2 * d, hidden, d, n_layers)

    def forward(self, left_emb, right_emb):
        """
        Args:
            left_emb:  (B, l, d)
            right_emb: (B, l, d)
        Returns:
            parent_first: (B, l, d)
        """
        concat = torch.cat([left_emb, right_emb], dim=-1)  # (B, l, 2d)
        gate = self.gate_net(concat)                         # (B, l, d)
        candidate = self.candidate_net(concat)               # (B, l, d)
        residual = (left_emb + right_emb) / 2.0
        return gate * candidate + (1 - gate) * residual


# ─── Pure Neural Computational Graph Decoder ──────────────────────────────────

class PureNeuralCompGraphDecoder(nn.Module):
    """
    Neural SC Decoder with PURE NEURAL CalcParent.

    CalcParent is a learned gated MLP instead of analytical circ_conv.

    Supports knowledge distillation: when distill_alpha > 0, also computes
    teacher (analytical) CalcParent output and returns MSE for distillation loss.
    """

    def __init__(self, d=16, hidden=64, n_layers=2, vocab_size=3,
                 use_combine_nn=False, tau=1.0):
        super().__init__()
        self.d = d
        self.use_combine_nn = use_combine_nn
        self.tau = tau

        # Channel embedding
        self.embedding_z = nn.Embedding(vocab_size, d)

        # Tree operations (weight-shared)
        self.calc_left_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_nn = _make_mlp(3 * d, hidden, d, n_layers)

        # PURE NEURAL CalcParent
        self.calc_parent_nn = NeuralCalcParent(d, hidden, n_layers)

        # Second-half transform for parent[l:] = f(right)
        # Simple linear projection (could also just copy right, but let's
        # give it a small transform to allow adaptation)
        self.parent_second_nn = nn.Sequential(
            nn.Linear(d, d),
        )

        # Decision head
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

        # Re-embedding (for teacher distillation and leaf embedding)
        self.logits2emb = _make_mlp(4, hidden, d, n_layers)

        # Combine
        if use_combine_nn:
            self.combine_nn = _make_mlp(2 * d, hidden, d, n_layers)

        # Learnable "no info" embedding
        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Core tree operations ─────────────────────────────────────────────

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

    def _pure_neural_calc_parent(self, beta, edge_data):
        """Pure neural CalcParent: no analytical circ_conv."""
        left = edge_data[2 * beta]
        right = edge_data[2 * beta + 1]

        # First half: neural gated residual
        parent_first = self.calc_parent_nn(left, right)

        # Second half: transform of right
        parent_second = self.parent_second_nn(right)

        edge_data[beta] = torch.cat([parent_first, parent_second], dim=1)

    def _teacher_calc_parent(self, beta, edge_data):
        """
        Teacher (analytical) CalcParent.
        Returns the teacher embedding WITHOUT modifying edge_data.
        """
        left = edge_data[2 * beta]
        right = edge_data[2 * beta + 1]
        l = left.shape[1]

        left_lp = F.log_softmax(self.emb2logits(left) / self.tau, dim=-1)
        right_lp = F.log_softmax(self.emb2logits(right) / self.tau, dim=-1)

        left_22 = left_lp.reshape(-1, l, 2, 2)
        right_22 = right_lp.reshape(-1, l, 2, 2)

        parent_first = circ_conv_torch(left_22, right_22)
        parent_first_emb = self.logits2emb(parent_first.reshape(-1, l, 4))
        parent_second_emb = self.logits2emb(right_lp)

        return torch.cat([parent_first_emb, parent_second_emb], dim=1)

    # ── Navigation ───────────────────────────────────────────────────────

    def _step_one(self, current, beta, edge_data, distill_pairs):
        if current == beta >> 1:
            # Going DOWN
            if beta & 1 == 0:
                self._neural_calc_left(current, edge_data)
            else:
                self._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            # Going UP → Neural CalcParent
            if distill_pairs is not None:
                teacher_emb = self._teacher_calc_parent(current, edge_data)
            self._pure_neural_calc_parent(current, edge_data)
            if distill_pairs is not None:
                student_emb = edge_data[current]
                distill_pairs.append((student_emb, teacher_emb.detach()))
            return beta
        else:
            raise ValueError(f"Invalid step: current={current}, target={beta}")

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

    def _step_to(self, current, target, edge_data, distill_pairs):
        if current == target:
            return current
        for beta in self._get_path(current, target):
            current = self._step_one(current, beta, edge_data, distill_pairs)
        return current

    # ── Leaf decision helpers ────────────────────────────────────────────

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

    # ── Main forward pass ────────────────────────────────────────────────

    def forward(self, z, b, frozen_u, frozen_v, u_true=None, v_true=None,
                root_emb=None, distill_alpha=0.0):
        """
        Forward pass.

        Extra parameter:
            distill_alpha: if > 0, collect teacher-student pairs for distillation.

        Returns
        -------
        all_logits, all_targets, u_hat, v_hat, distill_loss
        """
        if root_emb is not None:
            B, N, d = root_emb.shape
            root = root_emb
            device = root.device
        else:
            B, N = z.shape
            device = z.device
            d = self.d
            br = torch.from_numpy(bit_reversal_perm(N.bit_length() - 1)).long().to(device)
            root = self.embedding_z(z)[:, br]

        n = N.bit_length() - 1
        assert (1 << n) == N
        teacher = u_true is not None and v_true is not None

        edge_data = [None] * (2 * N)
        edge_data[1] = root

        no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0)
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

        # Collect distillation pairs
        distill_pairs = [] if distill_alpha > 0 else None

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

            dec_head = self._step_to(dec_head, target_vtx, edge_data, distill_pairs)

            temp = edge_data[leaf_edge][:, 0].clone()

            if leaf_edge & 1 == 0:
                self._neural_calc_left(target_vtx, edge_data)
            else:
                self._neural_calc_right(target_vtx, edge_data)
            top_down = edge_data[leaf_edge][:, 0]

            if self.use_combine_nn:
                combined = self.combine_nn(torch.cat([top_down, temp], dim=-1))
            else:
                combined = top_down + temp
            logits = self.emb2logits(combined)

            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.float32, device=device)
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

            new_emb = self._make_leaf_emb(
                u_hat.get(i_t), v_hat.get(i_t), B, device)
            edge_data[leaf_edge] = new_emb.unsqueeze(1)

        # Compute distillation loss
        distill_loss = torch.tensor(0.0, device=device)
        if distill_pairs:
            mse_losses = []
            for student, teacher_tgt in distill_pairs:
                mse_losses.append(F.mse_loss(student, teacher_tgt))
            distill_loss = torch.stack(mse_losses).mean()

        return all_logits, all_targets, u_hat, v_hat, distill_loss
