"""
npd_memory_mac.py
=================
Neural Polar Decoder for memory MAC channels (Class C corner-rate path).

Architecture for chained 2-user memory MAC NPD
----------------------------------------------
The corner-rate Class C path decodes U first, then V|U. For each stage we train
a separate NPDSingleUser-style tree, but the CHANNEL EMBEDDING E^W takes the
FULL z-sequence into account via a sliding-window MLP or a bidirectional GRU.
This is the direct adaptation of Algorithm 3 of Aharoni/Huleihel/Pfister/
Permuter (NPD for channels with memory) to the MAC corner-rate chain.

Stage 1 (U decode on marginal + memory channel):
   emb_i = f_theta_1( [z_{i-W..i+W}] )                 (sliding window), OR
   emb_i = BiGRU_theta_1(z_1^N)_i                      (bidirectional RNN)
   Tree operations (F, G, H) consume these per-position embeddings.

Stage 2 (V decode given decoded U hat + memory):
   Input at position i is [z_{i-W..i+W}, u_hat_{i-W..i+W}]
   (concatenate the decoded U sequence alongside z).
   emb_i = f_theta_2([z window, u hat window])

KEY differences from the broken `class_c_npd.models.npd_single_user.NPDSingleUser`:

1. The z_encoder is SEQUENCE-LEVEL, not per-position. Even with a fixed
   3-window MLP it *was* per-position but applied to pre-built windows. Here
   we make it an explicit nn.Module that takes the full (B, N) z-sequence
   and returns (B, N, d) embeddings. A BiGRU variant is included.

2. Training uses FULL-NEURAL tree operations (`use_analytical_training=False`).
   The broken config tried to force scalar LLR on the squashed embedding, losing
   channel context. Here the d-dim embedding carries state information.

3. Stage 2 concatenates U hat into the encoder input. During teacher-forced
   training we use the true U (teacher forcing); at inference we use the Stage 1
   decoder output.

Author: session agent, 2026-04-16.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ─── Memory Z encoders ────────────────────────────────────────────────────────

class MemoryZEncoderWindow(nn.Module):
    """
    Sliding-window channel feature encoder.

    At position i, concatenate z[i-W..i+W] (total 2W+1 values) and pass through a
    2-layer MLP. Out-of-bounds positions are zero-padded.

    Args:
        window_size: W, so total context length = 2W + 1. Default 1 → [z_{i-1}, z_i, z_{i+1}].
        d: output embedding dim
        hidden: MLP hidden width
        extra_dim: additional per-position input dims (e.g. u_hat side info for Stage 2)
    """

    def __init__(self, window_size: int = 1, d: int = 16, hidden: int = 64,
                 n_layers: int = 2, extra_dim: int = 0):
        super().__init__()
        self.W = int(window_size)
        self.d = d
        self.extra_dim = extra_dim
        # Window has (2W+1) z values plus the same window of extra side info
        in_dim = (2 * self.W + 1) * (1 + extra_dim)
        self.mlp = _make_mlp(in_dim, hidden, d, n_layers)

    def _window(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, N) -> (B, N, 2W+1) via zero-padding."""
        B, N = z.shape
        pad = self.W
        # pad front/back with zeros
        z_pad = F.pad(z, (pad, pad), mode='constant', value=0.0)
        # unfold: length 2W+1 windows at every position
        windows = z_pad.unfold(dimension=1, size=2 * self.W + 1, step=1)  # (B, N, 2W+1)
        return windows

    def forward(self, z: torch.Tensor, side: torch.Tensor = None) -> torch.Tensor:
        """
        z    : (B, N) float — raw channel output
        side : (B, N, extra_dim) float — per-position side info (e.g. u_hat as ±1)

        Returns: (B, N, d) embeddings.
        """
        B, N = z.shape
        win_z = self._window(z)  # (B, N, K) with K = 2W+1
        if self.extra_dim > 0:
            assert side is not None and side.shape == (B, N, self.extra_dim), \
                f'side must be (B,N,{self.extra_dim}), got {None if side is None else side.shape}'
            # window each side channel
            side_windows = []
            for k in range(self.extra_dim):
                side_windows.append(self._window(side[..., k]))  # (B,N,K)
            win_side = torch.stack(side_windows, dim=-1)  # (B,N,K,extra_dim)
            win_side = win_side.flatten(start_dim=2)  # (B,N,K*extra_dim)
            features = torch.cat([win_z, win_side], dim=-1)  # (B,N,K*(1+extra_dim))
        else:
            features = win_z
        return self.mlp(features)


class MemoryZEncoderBiGRU(nn.Module):
    """
    Bidirectional GRU channel feature encoder.

    Processes the full z-sequence (plus optional side info per-position) and
    produces a d-dim embedding per position. This is likely closer to the
    NPD paper's recurrent E^W for memory channels.

    Args:
        d: output embedding dim. Uses hidden=d//2 per direction.
        extra_dim: additional per-position input dims.
        num_layers: GRU stacked layers.
    """

    def __init__(self, d: int = 16, extra_dim: int = 0, num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        assert d % 2 == 0, 'BiGRU needs d even (d//2 per direction)'
        self.d = d
        self.extra_dim = extra_dim
        in_size = 1 + extra_dim
        hidden = d // 2
        self.gru = nn.GRU(input_size=in_size, hidden_size=hidden,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout)

    def forward(self, z: torch.Tensor, side: torch.Tensor = None) -> torch.Tensor:
        B, N = z.shape
        x = z.unsqueeze(-1)  # (B, N, 1)
        if self.extra_dim > 0:
            assert side is not None and side.shape == (B, N, self.extra_dim)
            x = torch.cat([x, side], dim=-1)  # (B, N, 1 + extra_dim)
        out, _ = self.gru(x)  # (B, N, d)
        return out


# ─── NPD tree (channel-independent) ──────────────────────────────────────────

class NPDTree(nn.Module):
    """
    Neural polar decoder tree. Channel-independent: consumes per-position
    embeddings (B, N, d) and supports fast_ce training + sequential decode.

    This is the NEURAL tree (use_analytical_training=False in the baseline
    NPDSingleUser). We avoid the analytical scalar-LLR path entirely because
    memory channels' effective marginal is NOT a simple scalar LLR.

    Args:
        d, hidden, n_layers: MLP architecture for f/g/h
    """

    def __init__(self, d: int = 16, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.d = d
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.emb2llr = _make_mlp(d, hidden, 1, n_layers)

    def bitnode(self, e_odd: torch.Tensor, e_even: torch.Tensor,
                u_hard: torch.Tensor) -> torch.Tensor:
        if u_hard.dim() == 2:
            u_hard = u_hard.unsqueeze(-1)
        u_sign = 2.0 * u_hard.float() - 1.0
        u_sign = u_sign.expand_as(e_odd)
        e_signed = e_odd * u_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce(self, emb: torch.Tensor, x_cw: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced BCE across all depths of the NPD tree.

        emb  : (B, N, d) channel embeddings (already in NPD tree order)
        x_cw : (B, N)    true codeword bits in NPD tree order

        Returns mean of per-depth BCE losses (neural mode).
        """
        if x_cw.dim() == 3:
            x_cw = x_cw.squeeze(-1)

        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        # Depth 0: direct leaf-from-root probe
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

            V = V_new
            E = E_new

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def decode(self, emb: torch.Tensor, frozen_set: set) -> torch.Tensor:
        """
        Sequential SC decode; frozen_set in 0-indexed NATURAL message order.

        emb         : (B, N, d) channel embeddings in NPD tree order
        returns u_hat : (B, N) in natural order
        """
        from polar.encoder import bit_reversal_perm
        import math as _math
        B, N, _ = emb.shape
        n = int(_math.log2(N))
        br = bit_reversal_perm(n)
        dev = emb.device
        u_hat = torch.zeros(B, N, dtype=torch.long, device=dev)
        leaf_idx = [0]

        def _decode(e_block: torch.Tensor) -> torch.Tensor:
            block_size = e_block.shape[1]
            if block_size == 1:
                logit = self.emb2llr(e_block[:, 0, :]).squeeze(-1)
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                nat_idx = int(br[idx])
                if nat_idx in frozen_set:
                    dec = torch.zeros(B, dtype=torch.long, device=dev)
                else:
                    dec = (logit > 0).long()
                u_hat[:, nat_idx] = dec
                return dec.unsqueeze(1)

            e_odd = e_block[:, 0::2, :]
            e_even = e_block[:, 1::2, :]

            e_top = self.checknode(torch.cat([e_odd, e_even], dim=-1))
            cw_top = _decode(e_top)

            e_bot = self.bitnode(e_odd, e_even, cw_top)
            cw_bot = _decode(e_bot)

            cw = torch.zeros(B, block_size, dtype=torch.long, device=dev)
            cw[:, 0::2] = cw_top ^ cw_bot
            cw[:, 1::2] = cw_bot
            return cw

        _decode(emb)
        return u_hat


# ─── Single-stage memory NPD (Stage 1 or Stage 2) ────────────────────────────

class MemoryStageNPD(nn.Module):
    """
    One stage of the chained memory MAC NPD. Owns its own MemoryZEncoder and
    NPDTree.

    Args:
        d, hidden, n_layers: tree architecture
        encoder_type: 'window' or 'bigru'
        window_size: W for 'window' encoder
        extra_dim: additional side info channels per position
          - Stage 1: extra_dim=0 (only raw z)
          - Stage 2: extra_dim=1 (the decoded U as ±1)
    """

    def __init__(self, d: int = 16, hidden: int = 64, n_layers: int = 2,
                 encoder_type: str = 'window', window_size: int = 1,
                 extra_dim: int = 0, gru_layers: int = 1):
        super().__init__()
        self.d = d
        self.encoder_type = encoder_type
        self.extra_dim = extra_dim
        if encoder_type == 'window':
            self.z_encoder = MemoryZEncoderWindow(
                window_size=window_size, d=d, hidden=hidden,
                n_layers=n_layers, extra_dim=extra_dim,
            )
        elif encoder_type == 'bigru':
            self.z_encoder = MemoryZEncoderBiGRU(
                d=d, extra_dim=extra_dim, num_layers=gru_layers,
            )
        else:
            raise ValueError(f'unknown encoder_type {encoder_type!r}')
        self.tree = NPDTree(d=d, hidden=hidden, n_layers=n_layers)

    def encode_channel(self, z: torch.Tensor, side: torch.Tensor = None) -> torch.Tensor:
        """
        z    : (B, N)
        side : (B, N, extra_dim) or None

        Returns: (B, N, d) embeddings in NATURAL (position) order.
        """
        return self.z_encoder(z, side=side)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Full chained NPD ────────────────────────────────────────────────────────

class ChainedNPD_MAC(nn.Module):
    """
    Full chained NPD for two-user memory MAC, corner-rate Class C path.

    Stage 1 decodes U from z alone (V treated as random).
    Stage 2 decodes V from (z, u_hat).

    Args:
        d, hidden, n_layers: tree architecture (shared between stages, but each
          has its own parameters)
        encoder_type: 'window' or 'bigru'
        window_size: W for 'window' encoder
        gru_layers: for 'bigru'
    """

    def __init__(self, d: int = 16, hidden: int = 64, n_layers: int = 2,
                 encoder_type: str = 'window', window_size: int = 1,
                 gru_layers: int = 1):
        super().__init__()
        self.stage1 = MemoryStageNPD(
            d=d, hidden=hidden, n_layers=n_layers,
            encoder_type=encoder_type, window_size=window_size,
            extra_dim=0, gru_layers=gru_layers,
        )
        self.stage2 = MemoryStageNPD(
            d=d, hidden=hidden, n_layers=n_layers,
            encoder_type=encoder_type, window_size=window_size,
            extra_dim=1, gru_layers=gru_layers,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
