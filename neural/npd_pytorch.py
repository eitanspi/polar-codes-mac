#!/usr/bin/env python3
"""
npd_pytorch.py — Faithful PyTorch port of the NPD (Neural Polar Decoder).

Ported from NPDforCourse TensorFlow implementation (Aharoni et al.).
Key conventions maintained:
  - NO bit-reversal (NPD uses even/odd split, not bit-reversal)
  - Codeword x = interleave(f2(u1_cw, u2_cw)) where f2(a,b) = (a^b, b)
  - fast_ce target is the codeword x from the encoder
  - V_left in fast_ce = u1hardprev in decode (consistency guaranteed)
  - BitNode residual: out = MLP(e1*u_sign, e2) + e1*u_sign + e2
  - u_sign = 2*u - 1  (0→-1, 1→+1)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ─── NPD-compatible encoder (no bit-reversal) ───────────────────────────────

def npd_encode(u):
    """Recursive polar encode matching NPD tree structure. No bit-reversal.
    u: numpy array (..., N). Returns codeword x of same shape."""
    N = u.shape[-1]
    if N == 1:
        return u.copy()
    u_odd = u[..., 0::2]
    u_even = u[..., 1::2]
    x_left = npd_encode(u_odd ^ u_even)   # top branch codeword
    x_right = npd_encode(u_even)           # bottom branch codeword
    # Interleave: x[0::2] = x_left, x[1::2] = x_right
    x = np.empty_like(u)
    x[..., 0::2] = x_left
    x[..., 1::2] = x_right
    return x


# ─── Single-user NPD decoder ────────────────────────────────────────────────

class NPDSingleUser(nn.Module):
    """Single-user NPD with fast_ce training and sequential decode.
    Faithfully matches NPDforCourse conventions."""

    def __init__(self, d=16, hidden=64, n_layers=2, z_dim=1):
        super().__init__()
        self.d = d

        # Channel encoder Ey: maps channel output to embedding
        self.ey = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ELU(), nn.Linear(hidden, d),
        )

        # CheckNode (f-node): (e_odd, e_even) → embedding
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)

        # BitNode (g-node): (e_odd * u_sign, e_even) → embedding + residual
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)

        # Embedding to LLR (binary decision)
        self.emb2llr = _make_mlp(d, hidden, 1, n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def bitnode(self, e_odd, e_even, u_hard):
        """BitNode with NPD residual. u_hard: (batch, M) or (batch, M, 1) bits."""
        if u_hard.dim() == 2:
            u_hard = u_hard.unsqueeze(-1)
        u_sign = 2.0 * u_hard.float() - 1.0  # 0→-1, 1→+1 (NPD convention)
        u_sign = u_sign.expand_as(e_odd)
        e_signed = e_odd * u_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce(self, emb, x_cw):
        """
        Parallel teacher-forced training. O(log N) sequential gradient depth.

        Args:
            emb: (B, N, d) channel embeddings (from ey)
            x_cw: (B, N) or (B, N, 1) codeword bits (from npd_encode)

        Returns:
            mean loss across all tree depths
        """
        if x_cw.dim() == 3:
            x_cw = x_cw.squeeze(-1)

        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        # Depth 0: predict codeword from raw embeddings
        pred = self.emb2llr(emb).squeeze(-1)  # (B, N)
        loss = F.binary_cross_entropy_with_logits(pred, x_cw.float(), reduction='mean')
        all_losses.append(loss)

        # V = target bits, E = embeddings (lists of chunks)
        V = [x_cw]
        E = [emb]

        for depth in range(n):
            # Split each chunk into odd/even
            V_odds, V_evens = [], []
            E_odds, E_evens = [], []
            for v_chunk, e_chunk in zip(V, E):
                V_odds.append(v_chunk[:, 0::2])
                V_evens.append(v_chunk[:, 1::2])
                E_odds.append(e_chunk[:, 0::2, :])
                E_evens.append(e_chunk[:, 1::2, :])

            V_odd = torch.cat(V_odds, dim=1)  # (B, N/2)
            V_even = torch.cat(V_evens, dim=1)
            E_odd = torch.cat(E_odds, dim=1)  # (B, N/2, d)
            E_even = torch.cat(E_evens, dim=1)

            # Target bits for next level
            v_xor = V_odd ^ V_even   # u1 targets (top branch)
            v_id = V_even            # u2 targets (bottom branch)

            # Split into chunks and interleave [xor_0, id_0, xor_1, id_1, ...]
            num_chunks = 2 ** depth
            chunk_size = (N // 2) // num_chunks
            v_xor_chunks = torch.split(v_xor, chunk_size, dim=1)
            v_id_chunks = torch.split(v_id, chunk_size, dim=1)
            V_new = []
            for vx, vi in zip(v_xor_chunks, v_id_chunks):
                V_new.append(vx)  # u1 chunk
                V_new.append(vi)  # u2 chunk

            # V_left = all u1 targets (even-indexed chunks)
            V_left = torch.cat(V_new[0::2], dim=1)  # = v_xor = u1 targets

            # Compute embeddings
            inp = torch.cat([E_odd, E_even], dim=-1)
            e_left = self.checknode(inp)      # top branch embedding
            e_right = self.bitnode(E_odd, E_even, V_left)  # bottom branch (conditioned on u1)

            # Split and interleave embeddings
            e_left_chunks = torch.split(e_left, chunk_size, dim=1)
            e_right_chunks = torch.split(e_right, chunk_size, dim=1)
            E_new = []
            for el, er in zip(e_left_chunks, e_right_chunks):
                E_new.append(el)
                E_new.append(er)

            # Compute loss at this depth
            e_all = torch.cat(E_new, dim=1)  # (B, N/2, d) at depth+1
            v_all = torch.cat(V_new, dim=1)  # (B, N/2) at depth+1
            pred = self.emb2llr(e_all).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, v_all.float(), reduction='mean')
            all_losses.append(loss)

            V = V_new
            E = E_new

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def decode(self, emb, frozen_set):
        """
        Sequential SC decode. Matches fast_ce conventions exactly.

        Args:
            emb: (B, N, d) channel embeddings
            frozen_set: set of 0-indexed frozen positions

        Returns:
            u_hat: (B, N) decoded message bits
        """
        B = emb.shape[0]
        N = emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(e_block):
            """Recursive decode. Returns CODEWORD of this subtree."""
            block_size = e_block.shape[1]

            if block_size == 1:
                llr = self.emb2llr(e_block[:, 0, :]).squeeze(-1)  # (B,)
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                if idx in frozen_set:
                    dec = torch.zeros(B, dtype=torch.long)
                else:
                    dec = (llr > 0).long()  # positive logit → bit 1 (BCE convention)
                u_hat[:, idx] = dec
                # At leaf: codeword = message bit
                return dec.unsqueeze(1)  # (B, 1)

            e_odd = e_block[:, 0::2, :]   # (B, M/2, d)
            e_even = e_block[:, 1::2, :]  # (B, M/2, d)

            # Top branch: CheckNode
            inp = torch.cat([e_odd, e_even], dim=-1)
            e_top = self.checknode(inp)
            u1_cw = _decode(e_top)  # left child CODEWORD (B, M/2)

            # Bottom branch: BitNode conditioned on u1_cw
            e_bot = self.bitnode(e_odd, e_even, u1_cw)
            u2_cw = _decode(e_bot)  # right child CODEWORD (B, M/2)

            # Reconstruct parent codeword via f2 + interleave
            # f2(u1_cw, u2_cw) = (u1_cw ^ u2_cw, u2_cw)
            # interleave: x[0::2] = u1_cw ^ u2_cw, x[1::2] = u2_cw
            x = torch.zeros(B, block_size, dtype=torch.long)
            x[:, 0::2] = u1_cw ^ u2_cw
            x[:, 1::2] = u2_cw
            return x

        _decode(emb)
        return u_hat


# ─── Quick self-test ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Verify encoder consistency with standard polar encoder
    from polar.encoder import polar_encode_batch, bit_reversal_perm
    N = 16
    u = np.array([[0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1]])
    x_std = polar_encode_batch(u)
    x_npd = npd_encode(u)
    br = bit_reversal_perm(4)
    assert np.all(x_npd == x_std[0, br].reshape(1, -1)), "Encoder mismatch!"
    print(f"Encoder verified: npd_encode == standard_encode[bit_reversal]")

    # Verify fast_ce + decode consistency
    torch.manual_seed(42)
    model = NPDSingleUser(d=8, hidden=32, n_layers=2, z_dim=1)

    # Generate test data
    N = 8; k = 4; sigma = 0.3
    rng = np.random.default_rng(42)
    u = np.zeros((1, N), dtype=int)
    for p in range(k, N): u[:, p] = rng.integers(0, 2, 1)
    x = npd_encode(u)
    z = (1.0 - 2.0*x.astype(float)) + rng.normal(0, sigma, (1, N))

    emb = model.ey(torch.from_numpy(z).float().unsqueeze(-1))
    loss = model.fast_ce(emb, torch.from_numpy(x).long())
    print(f"fast_ce loss: {loss.item():.4f}")

    frozen_set = set(range(k))
    u_dec = model.decode(emb, frozen_set)
    print(f"u_true: {u[0]}")
    print(f"u_dec:  {u_dec[0].numpy()}")
    print(f"Decode runs OK (untrained, expected errors)")

    # Train briefly and verify learning
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(500):
        u_batch = np.zeros((32, N), dtype=int)
        for p in range(k, N): u_batch[:, p] = rng.integers(0, 2, 32)
        x_batch = npd_encode(u_batch)
        z_batch = (1.0 - 2.0*x_batch.astype(float)) + rng.normal(0, sigma, (32, N))
        emb_b = model.ey(torch.from_numpy(z_batch).float().unsqueeze(-1))
        loss = model.fast_ce(emb_b, torch.from_numpy(x_batch).long())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Evaluate
    errs = 0; total = 200
    model.eval()
    trng = np.random.default_rng(999)
    for _ in range(total):
        u1 = np.zeros((1, N), dtype=int)
        for p in range(k, N): u1[:, p] = trng.integers(0, 2, 1)
        x1 = npd_encode(u1)
        z1 = (1.0-2.0*x1) + trng.normal(0, sigma, (1, N))
        emb1 = model.ey(torch.from_numpy(z1).float().unsqueeze(-1))
        u_d = model.decode(emb1, frozen_set)
        if any(u_d[0, p].item() != u1[0, p] for p in range(k, N)):
            errs += 1
    print(f"\nAfter 500 iters: BLER={errs/total:.4f} (N={N}, rate={1-k/N:.2f}, sigma={sigma})")
    print("PASS" if errs/total < 0.5 else "FAIL — decode still broken")
