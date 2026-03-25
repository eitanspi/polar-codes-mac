"""
ncg_memory.py — NCG Decoder with Sequence Encoder for MAC Channels with Memory.

Architecture:
  1. Sequence Encoder: z ∈ R^N (continuous) → root embeddings ∈ R^(N, d)
     - Bidirectional GRU captures temporal dependencies (channel memory)
     - Replaces the discrete EmbeddingZ lookup table
  2. Tree Operations: Unchanged from neural_comp_graph.py
     - NeuralCalcLeft, NeuralCalcRight, CalcParent
     - Completely independent of channel memory/state space
  3. Complexity: O(md N log N) — independent of channel state space size S

The sequence encoder is the ONLY channel-dependent component. Everything
inside the tree is channel-agnostic.
"""

import math
import torch
import torch.nn as nn

from polar.encoder import bit_reversal_perm
from neural.neural_comp_graph import NeuralCompGraphDecoder, _make_mlp


class SequenceEncoder(nn.Module):
    """
    Encode continuous channel output sequence z ∈ R^N into root embeddings.

    Uses a bidirectional GRU to capture temporal dependencies (channel memory),
    followed by a projection to the embedding dimension d.

    The GRU processes z_1, ..., z_N and outputs h_1, ..., h_N where each h_t
    captures information from the entire sequence (bidirectional).

    Parameters
    ----------
    d       : output embedding dimension (must match NCG decoder)
    gru_dim : GRU hidden dimension per direction (total = 2*gru_dim)
    n_gru   : number of GRU layers
    """

    def __init__(self, d=16, gru_dim=32, n_gru=1):
        super().__init__()
        self.input_proj = nn.Linear(1, gru_dim)  # scalar z → gru_dim
        self.gru = nn.GRU(gru_dim, gru_dim, num_layers=n_gru,
                          batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(2 * gru_dim, d)  # bidirectional → d
        self.ln = nn.LayerNorm(d)

    def forward(self, z):
        """
        z: (B, N) float — continuous channel outputs
        Returns: (B, N, d) — root embeddings (NOT yet bit-reversed)
        """
        x = z.unsqueeze(-1)              # (B, N, 1)
        x = torch.relu(self.input_proj(x))  # (B, N, gru_dim)
        x, _ = self.gru(x)               # (B, N, 2*gru_dim)
        x = self.output_proj(x)          # (B, N, d)
        return self.ln(x)


class NCGMemoryDecoder(nn.Module):
    """
    NCG Decoder for MAC channels with memory.

    Combines:
      - SequenceEncoder (channel-dependent, captures memory)
      - NeuralCompGraphDecoder tree operations (channel-independent)

    The tree ops are loaded from a pre-trained BEMAC model, then the
    sequence encoder is trained while optionally fine-tuning the tree ops.

    Parameters
    ----------
    d        : embedding dimension
    hidden   : MLP hidden width for tree operations
    n_layers : MLP depth
    gru_dim  : GRU hidden dimension for sequence encoder
    n_gru    : number of GRU layers
    tau      : temperature for CalcParent
    """

    def __init__(self, d=16, hidden=64, n_layers=2, gru_dim=32, n_gru=1,
                 tau=1.0):
        super().__init__()
        self.d = d

        # Channel-dependent: sequence encoder
        self.seq_encoder = SequenceEncoder(d=d, gru_dim=gru_dim, n_gru=n_gru)

        # Channel-independent: tree operations (from NCG decoder)
        self.tree = NeuralCompGraphDecoder(
            d=d, hidden=hidden, n_layers=n_layers, tau=tau)

    def load_tree_weights(self, checkpoint_path):
        """Load pre-trained tree operation weights from a BEMAC model."""
        state = torch.load(checkpoint_path, weights_only=True)
        # Load only tree operation weights (not embedding_z)
        tree_state = self.tree.state_dict()
        loaded = 0
        for key in tree_state:
            if key in state and key != 'embedding_z.emb.weight':
                if tree_state[key].shape == state[key].shape:
                    tree_state[key] = state[key]
                    loaded += 1
        self.tree.load_state_dict(tree_state)
        return loaded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, z_continuous, b, frozen_u, frozen_v,
                u_true=None, v_true=None):
        """
        Forward pass for channels with memory.

        Parameters
        ----------
        z_continuous : (B, N) float — continuous channel outputs
        b            : list[int], len 2N — path vector
        frozen_u/v   : dict {1-indexed pos: value}
        u_true/v_true: (B, N) float or None — teacher forcing

        Returns
        -------
        Same as NeuralCompGraphDecoder.forward
        """
        B, N = z_continuous.shape
        n = N.bit_length() - 1
        device = z_continuous.device

        # Sequence encoder: continuous z → root embeddings
        root_emb = self.seq_encoder(z_continuous)  # (B, N, d)

        # Bit-reverse for tree structure
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root_emb = root_emb[:, br]  # (B, N, d) bit-reversed

        # Forward through tree (channel-independent operations)
        return self.tree(
            z=None, b=b, frozen_u=frozen_u, frozen_v=frozen_v,
            u_true=u_true, v_true=v_true, root_emb=root_emb)

    def freeze_tree(self):
        """Freeze tree operations (train only sequence encoder)."""
        for param in self.tree.parameters():
            param.requires_grad = False

    def unfreeze_tree(self):
        """Unfreeze tree operations for end-to-end fine-tuning."""
        for param in self.tree.parameters():
            param.requires_grad = True
