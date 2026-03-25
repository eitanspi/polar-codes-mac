"""
neural_comp_graph.py — Neural Computational Graph SC Decoder for MAC Polar Codes.

Implements the O(N log N) computational graph from Ren et al. (2025) with
neural replacements for the tree operations:

  - NeuralCalcLeft:  MLP replaces analytical calcLeft  (circ_conv + norm_prod)
  - NeuralCalcRight: MLP replaces analytical calcRight (norm_prod + circ_conv)
  - Soft-Bit Bridge for calcParent: emb→prob (shared Emb2Logits) →
        analytical circ_conv (differentiable) → prob→emb (Logits2Emb)

Complexity: O(N log N · md) where m=hidden, d=embedding dim.
Node operations are O(md), independent of channel memory/alphabet size.

Public API:
    model = NeuralCompGraphDecoder(d=16, hidden=64)
    logits, targets, u_hat, v_hat = model(z, b, frozen_u, frozen_v,
                                           u_true=u, v_true=v)
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


# ─── Differentiable circular convolution (for Soft-Bit Bridge) ───────────────

def circ_conv_torch(A, B):
    """
    Circular convolution ⊛ on (*, 2, 2) log-prob tensors.
    out[a,b] = logsumexp_{a',b'} ( A[a^a', b^b'] + B[a', b'] )
    Fully differentiable via torch.logsumexp.
    """
    A00, A01, A10, A11 = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    B00, B01, B10, B11 = B[..., 0, 0], B[..., 0, 1], B[..., 1, 0], B[..., 1, 1]

    out = torch.empty_like(A)
    out[..., 0, 0] = torch.logsumexp(torch.stack([A00+B00, A01+B01, A10+B10, A11+B11], -1), -1)
    out[..., 0, 1] = torch.logsumexp(torch.stack([A01+B00, A00+B01, A11+B10, A10+B11], -1), -1)
    out[..., 1, 0] = torch.logsumexp(torch.stack([A10+B00, A11+B01, A00+B10, A01+B11], -1), -1)
    out[..., 1, 1] = torch.logsumexp(torch.stack([A11+B00, A10+B01, A01+B10, A00+B11], -1), -1)
    return out


# ─── Neural Computational Graph Decoder ──────────────────────────────────────

class NeuralCompGraphDecoder(nn.Module):
    """
    Neural SC Decoder using the 2025 computational graph skeleton.

    Parameters
    ----------
    d         : embedding dimension
    hidden    : MLP hidden layer width
    n_layers  : number of hidden layers per MLP
    vocab_size: channel output alphabet size (3 for BEMAC: {0,1,2})
    """

    def __init__(self, d=16, hidden=64, n_layers=2, vocab_size=3,
                 use_combine_nn=False, tau=1.0):
        super().__init__()
        self.d = d
        self.use_combine_nn = use_combine_nn
        self.tau = tau  # temperature for Soft-Bit Bridge log_softmax

        # Channel embedding: z ∈ {0..vocab_size-1} → R^d
        self.embedding_z = nn.Embedding(vocab_size, d)

        # Tree operations (weight-shared across all levels)
        self.calc_left_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_nn = _make_mlp(3 * d, hidden, d, n_layers)

        # Decision head: R^d → R^4 joint logits P(u,v)
        # Shared for leaf decisions AND Soft-Bit Bridge (calcParent input)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

        # Soft-Bit Bridge: R^4 log-probs → R^d embedding
        self.logits2emb = _make_mlp(4, hidden, d, n_layers)

        # Combine top-down + bottom-up at leaf (MLP or additive)
        if use_combine_nn:
            self.combine_nn = _make_mlp(2 * d, hidden, d, n_layers)

        # Learnable "no info" embedding for leaf initialization
        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Core tree operations ─────────────────────────────────────────────

    def _neural_calc_left(self, beta, edge_data):
        """Neural calcLeft at vertex β: parent + right → left."""
        parent = edge_data[beta]             # (B, 2l, d)
        right = edge_data[2 * beta + 1]      # (B, l, d)
        l = right.shape[1]
        p_first = parent[:, :l]              # (B, l, d)
        p_second = parent[:, l:]             # (B, l, d)
        inp = torch.cat([p_first, p_second, right], dim=-1)  # (B, l, 3d)
        edge_data[2 * beta] = self.calc_left_nn(inp)          # (B, l, d)

    def _neural_calc_right(self, beta, edge_data):
        """Neural calcRight at vertex β: parent + left → right."""
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]
        p_first = parent[:, :l]
        p_second = parent[:, l:]
        inp = torch.cat([p_first, p_second, left], dim=-1)
        edge_data[2 * beta + 1] = self.calc_right_nn(inp)

    def _soft_bit_calc_parent(self, beta, edge_data):
        """
        Soft-Bit Bridge calcParent at vertex β.
        1. Emb → log-probs (shared emb2logits + log_softmax)
        2. Analytical circ_conv (differentiable)
        3. Log-probs → Emb (logits2emb)
        """
        left = edge_data[2 * beta]           # (B, l, d)
        right = edge_data[2 * beta + 1]      # (B, l, d)
        l = left.shape[1]

        # Embedding → normalized log-probabilities (temperature-scaled)
        left_lp = F.log_softmax(self.emb2logits(left) / self.tau, dim=-1)    # (B, l, 4)
        right_lp = F.log_softmax(self.emb2logits(right) / self.tau, dim=-1)  # (B, l, 4)

        # Reshape to (B, l, 2, 2) for circ_conv
        left_22 = left_lp.reshape(-1, l, 2, 2)
        right_22 = right_lp.reshape(-1, l, 2, 2)

        # Analytical calcParent: parent[:l] = circ_conv(left, right), parent[l:] = right
        parent_first = circ_conv_torch(left_22, right_22)  # (B, l, 2, 2)

        # Re-embed
        parent_first_emb = self.logits2emb(parent_first.reshape(-1, l, 4))
        parent_second_emb = self.logits2emb(right_lp)  # right_lp already (B, l, 4)

        edge_data[beta] = torch.cat([parent_first_emb, parent_second_emb], dim=1)

    # ── Navigation (same as analytical decoder) ──────────────────────────

    def _step_one(self, current, beta, edge_data):
        if current == beta >> 1:
            # Going DOWN
            if beta & 1 == 0:
                self._neural_calc_left(current, edge_data)
            else:
                self._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            # Going UP → Soft-Bit Bridge
            self._soft_bit_calc_parent(current, edge_data)
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

    # ── Leaf decision helpers ────────────────────────────────────────────

    def _make_leaf_emb(self, u_val, v_val, batch, device):
        """
        Create partially deterministic leaf embedding (eq 16).
        u_val, v_val: (batch,) tensor or None.
        """
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

        return self.logits2emb(lp)  # (batch, d)

    # ── Main forward pass ────────────────────────────────────────────────

    def forward(self, z, b, frozen_u, frozen_v, u_true=None, v_true=None,
                root_emb=None):
        """
        Forward pass through the computational graph.

        Parameters
        ----------
        z        : (B, N) long — channel outputs (discrete), or None if root_emb given
        b        : list[int], len 2N — path vector (0=U, 1=V)
        frozen_u : dict {1-indexed pos: value}
        frozen_v : dict {1-indexed pos: value}
        u_true   : (B, N) float or None — true info bits user 1 (teacher forcing)
        v_true   : (B, N) float or None — true info bits user 2
        root_emb : (B, N, d) float or None — pre-computed root embeddings
                   (already bit-reversed). If given, z and embedding_z are bypassed.

        Returns
        -------
        all_logits  : list of (B, 4) tensors at info positions
        all_targets : list of (B,) long tensors (if teacher forcing)
        u_hat       : dict {1-indexed pos: (B,) float tensor}
        v_hat       : dict {1-indexed pos: (B,) float tensor}
        """
        if root_emb is not None:
            B, N, d = root_emb.shape
            root = root_emb
            device = root.device
        else:
            B, N = z.shape
            device = z.device
            d = self.d
            br = torch.from_numpy(bit_reversal_perm(n := N.bit_length() - 1)).long().to(device)
            root = self.embedding_z(z)[:, br]   # (B, N, d) bit-reversed

        n = N.bit_length() - 1
        assert (1 << n) == N
        d = self.d
        teacher = u_true is not None and v_true is not None

        edge_data = [None] * (2 * N)
        edge_data[1] = root

        # Internal edges: "no info"
        no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

        # ── Decode along path ──
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

            # Navigate
            dec_head = self._step_to(dec_head, target_vtx, edge_data)

            # Save bottom-up (getAsParent) at leaf
            temp = edge_data[leaf_edge][:, 0].clone()   # (B, d)

            # Compute top-down (getAsChild) at leaf
            if leaf_edge & 1 == 0:
                self._neural_calc_left(target_vtx, edge_data)
            else:
                self._neural_calc_right(target_vtx, edge_data)
            top_down = edge_data[leaf_edge][:, 0]       # (B, d)

            # Combine top-down + bottom-up → decision
            if self.use_combine_nn:
                combined = self.combine_nn(torch.cat([top_down, temp], dim=-1))
            else:
                combined = top_down + temp               # additive combination
            logits = self.emb2logits(combined)           # (B, 4)

            # Decision
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

            # Record
            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            # Update leaf to partially deterministic
            new_emb = self._make_leaf_emb(
                u_hat.get(i_t), v_hat.get(i_t), B, device)
            edge_data[leaf_edge] = new_emb.unsqueeze(1)  # (B, 1, d)

        return all_logits, all_targets, u_hat, v_hat
