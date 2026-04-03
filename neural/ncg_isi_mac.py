"""
ncg_isi_mac.py — Neural SC Decoder for ISI-MAC (channel with memory).

Extends the GMAC neural decoder with a temporal z_encoder that takes
(z[i], z[i-1]) as input, capturing inter-symbol interference.

Three z_encoder options:
  a) Sliding window: z_encoder takes (z[i], z[i-1]) as input → d-dimensional embedding
  b) 1D convolution: conv1d over the z sequence
  c) RNN/LSTM: sequential encoding of z

We implement option (a) first as the simplest approach.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from polar.encoder import bit_reversal_perm
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder, _make_mlp


class ISIMACNeuralDecoder(nn.Module):
    """
    Neural SC Decoder for ISI-MAC with temporal z_encoder.

    z_encoder takes (z[i], z[i-1]) as a 2-dimensional input and produces
    a d-dimensional embedding. For i=0, z[i-1] is set to 0 (known initial state).
    """

    def __init__(self, d=16, hidden=64, n_layers=2, z_hidden=32,
                 z_encoder_type='window'):
        super().__init__()
        self.d = d
        self.z_encoder_type = z_encoder_type

        if z_encoder_type == 'window':
            # Sliding window: input is (z[i], z[i-1]) → 2 floats
            self.z_encoder = nn.Sequential(
                nn.Linear(2, z_hidden),
                nn.ELU(),
                nn.Linear(z_hidden, d),
            )
        elif z_encoder_type == 'conv':
            # 1D convolution with kernel size 2
            self.z_conv = nn.Conv1d(1, d, kernel_size=2, padding=1)
            self.z_proj = nn.Linear(d, d)
        elif z_encoder_type == 'gru':
            # GRU encoder
            self.z_gru = nn.GRU(1, d, batch_first=True)
        else:
            raise ValueError(f"Unknown z_encoder_type: {z_encoder_type}")

        # Tree decoder (weight-shared, channel-independent)
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

    def encode_z(self, z):
        """
        Encode channel outputs with temporal context.

        Parameters
        ----------
        z : (B, N) float tensor — ISI-MAC channel outputs

        Returns
        -------
        root : (B, N, d) tensor — position embeddings for the tree root
        """
        B, N = z.shape
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(z.device)

        if self.z_encoder_type == 'window':
            # Create (z[i], z[i-1]) pairs
            z_prev = torch.cat([torch.zeros(B, 1, device=z.device), z[:, :-1]], dim=1)
            z_pairs = torch.stack([z, z_prev], dim=-1)  # (B, N, 2)
            embeddings = self.z_encoder(z_pairs)  # (B, N, d)

        elif self.z_encoder_type == 'conv':
            z_in = z.unsqueeze(1)  # (B, 1, N)
            z_conv = self.z_conv(z_in)[:, :, :N]  # (B, d, N) — trim padding
            embeddings = self.z_proj(z_conv.transpose(1, 2))  # (B, N, d)

        elif self.z_encoder_type == 'gru':
            z_in = z.unsqueeze(-1)  # (B, N, 1)
            embeddings, _ = self.z_gru(z_in)  # (B, N, d)

        # Apply bit-reversal permutation (matches encoder's bit reversal)
        root = embeddings[:, br]  # (B, N, d)
        return root

    def forward(self, z, b, frozen_u, frozen_v, u_true=None, v_true=None):
        """
        Forward pass for ISI-MAC.

        Parameters
        ----------
        z : (B, N) float tensor — ISI-MAC channel outputs
        b : list of 2N ints — path vector
        frozen_u, frozen_v : dict — frozen positions
        u_true, v_true : optional ground truth for training

        Returns
        -------
        all_logits, all_targets, u_hat, v_hat, dummy_loss
        """
        root = self.encode_z(z)
        return self.tree(z=None, b=b, frozen_u=frozen_u, frozen_v=frozen_v,
                         u_true=u_true, v_true=v_true, root_emb=root)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
