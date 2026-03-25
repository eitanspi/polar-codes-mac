"""
ncg_gmac.py — Pure Neural Computational Graph SC Decoder for Gaussian MAC.

Identical to PureNeuralCompGraphDecoder (ncg_pure_neural.py) except:
  - EmbeddingZ (nn.Embedding for discrete {0,1,2}) is replaced with
    z_encoder: a small MLP that maps continuous z (float) to R^d.
  - forward() takes z as (batch, N) float tensor instead of long tensor.

All tree operations (CalcLeft, CalcRight, CalcParent, Emb2Logits, Logits2Emb)
are architecturally identical and their weights can be transferred from a
BEMAC-trained model — they are channel-independent by design.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from polar.encoder import bit_reversal_perm


# ─── MLP helper (same as ncg_pure_neural.py) ─────────────────────────────────

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ─── Neural CalcParent (same as ncg_pure_neural.py) ──────────────────────────

class NeuralCalcParent(nn.Module):
    def __init__(self, d, hidden, n_layers=2):
        super().__init__()
        self.d = d
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


# ─── GMAC Pure Neural Computational Graph Decoder ────────────────────────────

class GmacNeuralCompGraphDecoder(nn.Module):
    """
    Neural SC Decoder for Gaussian MAC with continuous channel output.

    Replaces nn.Embedding(3, d) with z_encoder: Linear(1, 32) -> ELU -> Linear(32, d).
    Everything else is architecturally identical to PureNeuralCompGraphDecoder.
    """

    def __init__(self, d=16, hidden=64, n_layers=2, z_hidden=32,
                 use_combine_nn=False, tau=1.0):
        super().__init__()
        self.d = d
        self.use_combine_nn = use_combine_nn
        self.tau = tau

        # Continuous channel embedding (replaces nn.Embedding(3, d))
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden),
            nn.ELU(),
            nn.Linear(z_hidden, d),
        )

        # Tree operations (weight-shared) — identical to BEMAC model
        self.calc_left_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_parent_nn = NeuralCalcParent(d, hidden, n_layers)
        self.parent_second_nn = nn.Sequential(nn.Linear(d, d))

        # Decision head
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)
        self.logits2emb = _make_mlp(4, hidden, d, n_layers)

        # Combine
        if use_combine_nn:
            self.combine_nn = _make_mlp(2 * d, hidden, d, n_layers)

        # Learnable "no info" embedding
        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Core tree operations (identical to ncg_pure_neural.py) ────────────

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
        left = edge_data[2 * beta]
        right = edge_data[2 * beta + 1]
        parent_first = self.calc_parent_nn(left, right)
        parent_second = self.parent_second_nn(right)
        edge_data[beta] = torch.cat([parent_first, parent_second], dim=1)

    # ── Navigation (identical) ────────────────────────────────────────────

    def _step_one(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0:
                self._neural_calc_left(current, edge_data)
            else:
                self._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self._pure_neural_calc_parent(current, edge_data)
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

    def _step_to(self, current, target, edge_data):
        if current == target:
            return current
        for beta in self._get_path(current, target):
            current = self._step_one(current, beta, edge_data)
        return current

    # ── Leaf decision helpers (identical) ─────────────────────────────────

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

    # ── Main forward pass ─────────────────────────────────────────────────

    def forward(self, z, b, frozen_u, frozen_v, u_true=None, v_true=None):
        """
        Forward pass for Gaussian MAC.

        Parameters
        ----------
        z : (B, N) float tensor — continuous channel output
        b : list of 2N ints — path (0=user U, 1=user V)
        frozen_u, frozen_v : dict {1-indexed pos: 0} — frozen positions
        u_true, v_true : (B, N) float tensors — ground truth (for training)

        Returns
        -------
        all_logits, all_targets, u_hat, v_hat, dummy_loss
        """
        B, N = z.shape
        device = z.device
        d = self.d

        # Continuous z embedding + bit-reversal
        br = torch.from_numpy(bit_reversal_perm(N.bit_length() - 1 if isinstance(N, int) else int(math.log2(N)))).long().to(device)
        root = self.z_encoder(z.unsqueeze(-1))[:, br]  # (B, N, d)

        n = int(math.log2(N))
        assert (1 << n) == N
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

        dummy_loss = torch.tensor(0.0, device=device)
        return all_logits, all_targets, u_hat, v_hat, dummy_loss
