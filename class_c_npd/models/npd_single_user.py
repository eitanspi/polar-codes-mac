"""
Single-user Neural Polar Decoder (NPD) — clean implementation.

Architecture:
  - z_encoder: maps scalar channel output(s) to a d-dim embedding per position
  - CheckNode (f-node): MLP(e_odd, e_even) -> d-dim
  - BitNode (g-node):   MLP(e_odd * u_sign, e_even) + e_odd * u_sign + e_even
                        residual architecture with sign-flip symmetry
  - emb2llr: MLP(d) -> 1 (binary LLR per position)

Key conventions (matching Aharoni et al. and the project's npd_pytorch.py):
  - NPD tree traversal uses even/odd recursive split (no bit-reversal inside)
  - Codeword is computed by npd_encode(u) = F^⊗n u (no B_N)
  - polar_encode_batch(u) == npd_encode(u)[bit_reversal_perm(n)]
  - sign convention: u_sign = 2u - 1, so 0 -> -1, 1 -> +1
  - fast_ce training: predicts codeword bit at every tree level, not just leaves
  - Targets are codeword bits (NOT message bits)

This implementation is channel-agnostic: the z_encoder accepts any fixed-dim
input, so you can feed it raw z, precomputed LLRs, or any channel-specific
features. For channels with memory, you can feed a window of z values around
each position by adjusting z_dim.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── Utilities ────────────────────────────────────────────────────────────────

def _make_mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    """MLP with ELU activations. n_layers counts hidden layers (+1 output)."""
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def npd_encode(u: np.ndarray) -> np.ndarray:
    """
    Recursive polar encode in NPD tree order (no bit-reversal).

    Relation to standard encoder:
        polar_encode_batch(u) == npd_encode(u)[..., bit_reversal_perm(n)]

    Input  u: numpy array (..., N)
    Output x: numpy array (..., N), same shape
    """
    N = u.shape[-1]
    if N == 1:
        return u.copy()
    u_odd = u[..., 0::2]
    u_even = u[..., 1::2]
    x_left = npd_encode(u_odd ^ u_even)   # top branch
    x_right = npd_encode(u_even)           # bottom branch
    x = np.empty_like(u)
    x[..., 0::2] = x_left
    x[..., 1::2] = x_right
    return x


def npd_encode_torch(u: torch.Tensor) -> torch.Tensor:
    """Recursive polar encode for torch tensors (vectorized, same logic)."""
    N = u.shape[-1]
    if N == 1:
        return u.clone()
    u_odd = u[..., 0::2]
    u_even = u[..., 1::2]
    x_left = npd_encode_torch(u_odd ^ u_even)
    x_right = npd_encode_torch(u_even)
    x = torch.zeros_like(u)
    x[..., 0::2] = x_left
    x[..., 1::2] = x_right
    return x


# ─── Single-user NPD model ───────────────────────────────────────────────────

class NPDSingleUser(nn.Module):
    """
    Single-user Neural Polar Decoder with fast_ce training and sequential decode.

    KEY DESIGN (matching Aharoni et al. NeuralPolarDecoder):
    - fast_ce training uses ANALYTICAL checknode/bitnode (exact LLR f/g functions)
    - Sequential decode uses NEURAL checknode/bitnode (learned MLPs)
    - The z_encoder learns to produce LLR-like scalar embeddings that work
      with the analytical tree ops during training
    - At inference, the neural tree ops approximate the analytical ones

    When use_analytical_training=True (default, matching the paper):
    - Training: z_encoder maps z -> scalar LLR-like embedding
    - Training tree ops: exact f(L1,L2) and g(L1,L2,u) on scalars
    - Decode tree ops: neural MLPs on d-dim embeddings (learned separately)

    When use_analytical_training=False (our earlier broken approach):
    - Everything is neural, including training tree ops

    Args:
        d: embedding dimension
        hidden: MLP hidden width for tree ops and z_encoder
        n_layers: number of hidden layers per MLP
        z_dim: dimensionality of channel features per position
        use_analytical_training: if True, use analytical f/g during fast_ce
    """

    def __init__(self, d: int = 16, hidden: int = 64, n_layers: int = 2,
                 z_dim: int = 1, use_analytical_training: bool = True):
        super().__init__()
        self.d = d
        self.z_dim = z_dim
        self.use_analytical_training = use_analytical_training

        # Channel feature encoder: maps z_dim-dim per-position features to d-dim embedding
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, d),
        )

        # Neural CheckNode f: (e_odd, e_even) -> d (used at decode time)
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)

        # Neural BitNode g: (e_odd * u_sign, e_even) -> d, with residual (used at decode time)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)

        # Leaf decision: d -> 1 logit (BCE convention: logit > 0 means bit = 1)
        self.emb2llr = _make_mlp(d, hidden, 1, n_layers)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Analytical tree ops (for fast_ce training, matching NPD paper) ───────

    @staticmethod
    def analytical_checknode(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        """
        Exact LLR f-function: f(L1, L2) = log[(1 + e^{L1+L2}) / (e^L1 + e^L2)]
        Equivalent to: -2 * atanh(tanh(L1/2) * tanh(L2/2))

        Operates on the LAST dimension (scalar LLR per position).
        Uses the logaddexp form for numerical stability.
        """
        # f(a, b) = logaddexp(a+b, 0) - logaddexp(a, b)
        result = torch.logaddexp(e1 + e2, torch.zeros_like(e1)) - torch.logaddexp(e1, e2)
        # Handle NaN from inf-inf (when both inputs are ±inf)
        nan_mask = torch.isnan(result)
        if nan_mask.any():
            fallback = torch.sign(e1) * torch.sign(e2) * torch.minimum(e1.abs(), e2.abs())
            result = torch.where(nan_mask, fallback, result)
        return result

    @staticmethod
    def analytical_bitnode(e1: torch.Tensor, e2: torch.Tensor,
                           u_hard: torch.Tensor) -> torch.Tensor:
        """
        Exact LLR g-function: g(L1, L2, u) = L2 + (1 - 2u) * L1

        u_hard: 0 or 1 bits.
        """
        if u_hard.dim() > e1.dim():
            u_hard = u_hard.squeeze(-1)
        u_sign = 1.0 - 2.0 * u_hard.float()
        return e2 + u_sign * e1

    # ── Core operations ──────────────────────────────────────────────────────

    def encode_channel(self, z_features: torch.Tensor) -> torch.Tensor:
        """
        Map per-position channel features to embeddings.

        z_features: (B, N, z_dim) or (B, N) — if 2D, last dim added
        returns:    (B, N, d)
        """
        if z_features.dim() == 2:
            z_features = z_features.unsqueeze(-1)
        return self.z_encoder(z_features)

    def bitnode(self, e_odd: torch.Tensor, e_even: torch.Tensor,
                u_hard: torch.Tensor) -> torch.Tensor:
        """
        BitNode g-operation with sign-flip + residual.

        e_odd, e_even: (B, M, d) — children embeddings
        u_hard:        (B, M) or (B, M, 1) — decoded left-child bits (0 or 1)
        returns:       (B, M, d) — right-child embedding
        """
        if u_hard.dim() == 2:
            u_hard = u_hard.unsqueeze(-1)
        u_sign = 2.0 * u_hard.float() - 1.0  # 0 -> -1, 1 -> +1
        u_sign = u_sign.expand_as(e_odd)
        e_signed = e_odd * u_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    # ── Training: fast_ce parallel forward pass ──────────────────────────────

    def fast_ce(self, emb: torch.Tensor, x_cw: torch.Tensor) -> torch.Tensor:
        """
        Parallel teacher-forced cross-entropy over all tree depths.

        emb:  (B, N, d)   channel embeddings (from encode_channel)
        x_cw: (B, N)      true codeword bits in NPD tree order

        When use_analytical_training=True (default, matching NPD paper):
          The tree ops use ANALYTICAL f/g on SCALAR LLR-like values.
          The z_encoder output is squeezed to (B, N, 1) and treated as a scalar.
          emb2llr during training is identity (the embedding IS the LLR).

        When use_analytical_training=False:
          All neural, same as before.

        Returns the mean of per-depth BCE losses. O(log N) gradient depth.
        """
        if x_cw.dim() == 3:
            x_cw = x_cw.squeeze(-1)

        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        if self.use_analytical_training:
            # Analytical mode: squeeze to scalar LLR per position
            # The z_encoder should learn to output a scalar-like embedding
            # We use only the FIRST dimension as the LLR signal
            E_scalar = emb[:, :, 0]  # (B, N) — first dim = LLR

            # Depth 0: predict from raw LLR
            all_losses.append(F.binary_cross_entropy_with_logits(
                E_scalar, x_cw.float(), reduction='mean'))

            V = [x_cw]
            E = [E_scalar]

            for depth in range(n):
                V_odds, V_evens, E_odds, E_evens = [], [], [], []
                for v_chunk, e_chunk in zip(V, E):
                    V_odds.append(v_chunk[:, 0::2])
                    V_evens.append(v_chunk[:, 1::2])
                    E_odds.append(e_chunk[:, 0::2])
                    E_evens.append(e_chunk[:, 1::2])

                V_odd = torch.cat(V_odds, dim=1)
                V_even = torch.cat(V_evens, dim=1)
                E_odd = torch.cat(E_odds, dim=1)
                E_even = torch.cat(E_evens, dim=1)

                v_top = V_odd ^ V_even
                v_bot = V_even

                num_chunks = 2 ** depth
                chunk_size = (N // 2) // num_chunks
                v_top_chunks = torch.split(v_top, chunk_size, dim=1)
                v_bot_chunks = torch.split(v_bot, chunk_size, dim=1)
                V_new = []
                for vt, vb in zip(v_top_chunks, v_bot_chunks):
                    V_new.append(vt); V_new.append(vb)
                V_left = torch.cat(V_new[0::2], dim=1)

                # ANALYTICAL tree ops on scalar LLRs
                e_top = self.analytical_checknode(E_odd, E_even)
                e_bot = self.analytical_bitnode(E_odd, E_even, V_left)

                e_top_chunks = torch.split(e_top, chunk_size, dim=1)
                e_bot_chunks = torch.split(e_bot, chunk_size, dim=1)
                E_new = []
                for et, eb in zip(e_top_chunks, e_bot_chunks):
                    E_new.append(et); E_new.append(eb)

                e_all = torch.cat(E_new, dim=1)
                v_all = torch.cat(V_new, dim=1)
                # Loss: the scalar embeddings ARE LLRs, so BCE directly
                all_losses.append(F.binary_cross_entropy_with_logits(
                    e_all, v_all.float(), reduction='mean'))

                V = V_new
                E = E_new

        else:
            # Full neural mode (original, before we found the paper's approach)
            logit = self.emb2llr(emb).squeeze(-1)
            all_losses.append(F.binary_cross_entropy_with_logits(
                logit, x_cw.float(), reduction='mean'))

            V = [x_cw]
            E = [emb]

            for depth in range(n):
                V_odds, V_evens, E_odds, E_evens = [], [], [], []
                for v_chunk, e_chunk in zip(V, E):
                    V_odds.append(v_chunk[:, 0::2])
                    V_evens.append(v_chunk[:, 1::2])
                    E_odds.append(e_chunk[:, 0::2, :])
                    E_evens.append(e_chunk[:, 1::2, :])

                V_odd = torch.cat(V_odds, dim=1)
                V_even = torch.cat(V_evens, dim=1)
                E_odd = torch.cat(E_odds, dim=1)
                E_even = torch.cat(E_evens, dim=1)

                v_top = V_odd ^ V_even; v_bot = V_even
                num_chunks = 2 ** depth
                chunk_size = (N // 2) // num_chunks
                v_top_chunks = torch.split(v_top, chunk_size, dim=1)
                v_bot_chunks = torch.split(v_bot, chunk_size, dim=1)
                V_new = []
                for vt, vb in zip(v_top_chunks, v_bot_chunks):
                    V_new.append(vt); V_new.append(vb)
                V_left = torch.cat(V_new[0::2], dim=1)

                e_top = self.checknode(torch.cat([E_odd, E_even], dim=-1))
                e_bot = self.bitnode(E_odd, E_even, V_left)

                e_top_chunks = torch.split(e_top, chunk_size, dim=1)
                e_bot_chunks = torch.split(e_bot, chunk_size, dim=1)
                E_new = []
                for et, eb in zip(e_top_chunks, e_bot_chunks):
                    E_new.append(et); E_new.append(eb)

                e_all = torch.cat(E_new, dim=1)
                v_all = torch.cat(V_new, dim=1)
                logit = self.emb2llr(e_all).squeeze(-1)
                all_losses.append(F.binary_cross_entropy_with_logits(
                    logit, v_all.float(), reduction='mean'))

                V = V_new; E = E_new

        return torch.stack(all_losses).mean()

    # ── Inference: sequential SC decode ──────────────────────────────────────

    @torch.no_grad()
    def decode(self, emb: torch.Tensor, frozen_set: set[int]) -> torch.Tensor:
        """
        Sequential SC decode. Matches fast_ce tree structure exactly.

        emb:         (B, N, d) channel embeddings
        frozen_set:  set of 0-indexed frozen positions in NATURAL message order.

        IMPORTANT: The NPD tree traversal visits positions in BIT-REVERSED
        order of the natural message positions. Leaf index i corresponds to
        natural message position br[i]. We handle this mapping inside the
        decoder so the caller can pass frozen sets in natural message order.

        Returns u_hat: (B, N) decoded message bits in natural order.
        """
        from polar.encoder import bit_reversal_perm
        import math as _math
        B, N, _ = emb.shape
        n = int(_math.log2(N))
        br = bit_reversal_perm(n)  # list: tree-pos -> natural-pos
        u_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(e_block: torch.Tensor) -> torch.Tensor:
            """
            Recursive decode. Returns the CODEWORD of this subtree
            (for feeding upward into the parent's BitNode).
            """
            block_size = e_block.shape[1]

            if block_size == 1:
                # Leaf: decide single message bit
                logit = self.emb2llr(e_block[:, 0, :]).squeeze(-1)  # (B,)
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                nat_idx = int(br[idx])  # natural-order message position

                if nat_idx in frozen_set:
                    dec = torch.zeros(B, dtype=torch.long)
                else:
                    dec = (logit > 0).long()  # BCE sigmoid convention

                u_hat[:, nat_idx] = dec
                return dec.unsqueeze(1)  # (B, 1) — at leaf, codeword == message

            # Non-leaf: split, recurse left (top), then right (bot)
            e_odd = e_block[:, 0::2, :]
            e_even = e_block[:, 1::2, :]

            e_top = self.checknode(torch.cat([e_odd, e_even], dim=-1))
            cw_top = _decode(e_top)  # top-subtree codeword

            e_bot = self.bitnode(e_odd, e_even, cw_top)
            cw_bot = _decode(e_bot)  # bot-subtree codeword

            # Reconstruct parent codeword via f2 + interleave
            # f2(cw_top, cw_bot) = (cw_top XOR cw_bot, cw_bot)
            # interleave: x[0::2] = top XOR bot, x[1::2] = bot
            cw = torch.zeros(B, block_size, dtype=torch.long)
            cw[:, 0::2] = cw_top ^ cw_bot
            cw[:, 1::2] = cw_bot
            return cw

        _decode(emb)
        return u_hat

    # ── Convenience: full pipeline from raw z ────────────────────────────────

    def forward(self, z_features: torch.Tensor, x_cw_true: torch.Tensor = None,
                frozen_set: set[int] = None) -> torch.Tensor:
        """
        Unified forward pass.

        If x_cw_true is provided: training mode, returns fast_ce loss.
        Else if frozen_set is provided: inference mode, returns u_hat.
        Else: returns only the embeddings.
        """
        emb = self.encode_channel(z_features)
        if x_cw_true is not None:
            return self.fast_ce(emb, x_cw_true)
        if frozen_set is not None:
            return self.decode(emb, frozen_set)
        return emb
