#!/usr/bin/env python3
"""
Debug script: find exactly where fast_ce parallel and sc_decode sequential diverge.

FINDING: The root cause is a DOMAIN MISMATCH in the bitnode's uv_left argument.

- fast_ce passes J_left = XOR of adjacent codeword pairs = sub-CODEWORD values
- sc_decode passes decoded leaf values from recursion = sub-MESSAGE values
- These are related by: J_left = polar_encode(leaf_values) per sub-block
- They differ whenever the sub-encoding is non-trivial

This means the bitnode is TRAINED with codeword-domain signs but EVALUATED
with message-domain signs. The model learns a function that only makes sense
in the codeword domain, causing ~25% (random) accuracy in sequential mode.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm


def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class TinyDecoder(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.d = d
        self.checknode = _make_mlp(2 * d, hidden, d)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d)
        self.emb2logits = _make_mlp(d, hidden, 4)

    def bitnode(self, e_odd, e_even, uv_left):
        u_left = uv_left // 2
        v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([e_odd[:, :, :h] * u_sign, e_odd[:, :, h:] * v_sign], dim=-1)
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce_leaf_embs(self, emb, joint_cw):
        """Parallel fast_ce returning final leaf embeddings and bitnode inputs at each depth."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        bitnode_inputs = []

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

            bitnode_inputs.append(('parallel', depth, J_left.clone()))

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left, cs, 1)
            jr = torch.split(J_even, cs, 1)

            E_chunks = []
            J_chunks = []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

        e_all = torch.cat(E_chunks, 1)
        j_all = torch.cat(J_chunks, 1)
        return e_all, j_all, bitnode_inputs

    def sc_decode_leaf_embs(self, emb, true_leaf_targets):
        """Sequential SC decode with teacher forcing, recording bitnode inputs."""
        B = emb.shape[0]
        leaf_idx = [0]
        leaf_embs = []
        bitnode_inputs = []
        depth_counter = [0]

        def _decode(emb_block, depth=0):
            block_size = emb_block.shape[1]
            if block_size == 1:
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                leaf_embs.append((idx, emb_block[:, 0, :].clone()))
                dec = true_leaf_targets[:, idx].clone()
                return dec.unsqueeze(1)

            e_odd = emb_block[:, 0::2, :]
            e_even = emb_block[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            uv_left = _decode(e_left, depth + 1)

            bitnode_inputs.append(('sequential', depth, uv_left.clone()))

            e_right = self.bitnode(e_odd, e_even, uv_left)
            uv_right = _decode(e_right, depth + 1)
            return torch.cat([uv_left, uv_right], 1)

        with torch.no_grad():
            _decode(emb)
        return leaf_embs, bitnode_inputs


def main():
    D = 4; HIDDEN = 8; N = 4
    torch.manual_seed(42)

    model = TinyDecoder(d=D, hidden=HIDDEN)
    model.eval()

    # Use an actual polar-encoded codeword for a concrete example
    u_msg = np.array([1, 1, 0, 1])
    v_msg = np.array([0, 1, 1, 0])
    x = np.array(polar_encode(u_msg))
    y = np.array(polar_encode(v_msg))
    br = bit_reversal_perm(2)

    x_br = x[br]
    y_br = y[br]
    joint_cw = torch.tensor([x_br * 2 + y_br]).long()

    print("=" * 70)
    print("SETUP: N=4 polar code with 2-user MAC")
    print("=" * 70)
    print(f"u = {u_msg}, x = polar_encode(u) = {x}")
    print(f"v = {v_msg}, y = polar_encode(v) = {y}")
    print(f"bit-reversal: {br}")
    print(f"joint_cw (bit-reversed, x*2+y) = {joint_cw[0].tolist()}")

    # Use random embeddings (doesn't matter for the structural bug)
    emb = torch.randn(1, N, D)

    # ─── Run parallel ─────────────────────────────────────────────────────
    with torch.no_grad():
        p_embs, p_targets, p_bn_inputs = model.fast_ce_leaf_embs(emb, joint_cw)

    print(f"\nParallel leaf targets: {p_targets[0].tolist()}")

    # ─── Compute sequential leaf targets ──────────────────────────────────
    # These are the message-domain values that the sequential decoder would
    # see at each leaf position.
    # For N=4: leaf targets = [u0, u1, u2, u3] for user 1, [v0, v1, v2, v3] for user 2
    # In joint form: leaf_i = u_i * 2 + v_i
    seq_targets = torch.tensor([u_msg * 2 + v_msg]).long()
    print(f"Sequential leaf targets (message domain): {seq_targets[0].tolist()}")

    # ─── Run sequential ───────────────────────────────────────────────────
    with torch.no_grad():
        s_leaf_data, s_bn_inputs = model.sc_decode_leaf_embs(emb, seq_targets)

    # ─── Compare bitnode inputs ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BITNODE uv_left COMPARISON (the root cause)")
    print("=" * 70)

    print("\nParallel bitnode inputs (J_left = sub-CODEWORD values):")
    for mode, depth, vals in p_bn_inputs:
        print(f"  depth {depth}: uv_left = {vals[0].tolist()}")

    print("\nSequential bitnode inputs (decoded leaf values = sub-MESSAGE values):")
    for mode, depth, vals in s_bn_inputs:
        print(f"  depth {depth}: uv_left = {vals[0].tolist()}")

    # Show the relationship
    print("\n" + "=" * 70)
    print("THE MISMATCH")
    print("=" * 70)

    # At the root level (depth 0):
    p_root_uv = p_bn_inputs[0][2]  # J_left at depth 0
    s_root_uv = s_bn_inputs[0][2]  # sequential decoded values from left subtree
    print(f"\nRoot-level bitnode (depth 0):")
    print(f"  Parallel  uv_left = {p_root_uv[0].tolist()} (sub-codeword)")
    print(f"  Sequential uv_left = {s_root_uv[0].tolist()} (sub-message)")
    print(f"  Same? {torch.equal(p_root_uv, s_root_uv)}")

    if not torch.equal(p_root_uv, s_root_uv):
        # Verify the relationship: parallel = polar_encode(sequential)
        s_u = (s_root_uv[0] // 2).numpy()
        s_v = (s_root_uv[0] % 2).numpy()
        encoded_u = np.array(polar_encode(s_u))
        encoded_v = np.array(polar_encode(s_v))
        reconstructed = encoded_u * 2 + encoded_v
        print(f"\n  Verification: polar_encode(sequential_u) * 2 + polar_encode(sequential_v)")
        print(f"    sequential u-part: {s_u}, encoded: {encoded_u}")
        print(f"    sequential v-part: {s_v}, encoded: {encoded_v}")
        print(f"    reconstructed: {reconstructed}")
        print(f"    parallel value:  {p_root_uv[0].numpy()}")
        print(f"    Match? {np.array_equal(reconstructed, p_root_uv[0].numpy())}")

    # ─── Compare leaf embeddings ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("LEAF EMBEDDING COMPARISON")
    print("=" * 70)
    all_match = True
    for idx, s_emb in s_leaf_data:
        p_emb = p_embs[0, idx, :]
        match = torch.allclose(p_emb, s_emb[0], atol=1e-5)
        if not match:
            all_match = False
            diff = (p_emb - s_emb[0]).abs().max().item()
            print(f"  Leaf {idx}: MISMATCH (max diff = {diff:.6f})")
        else:
            print(f"  Leaf {idx}: match")

    # ─── Test with N=8 ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TESTING N=8")
    print("=" * 70)

    N8 = 8
    torch.manual_seed(42)
    model8 = TinyDecoder(d=D, hidden=HIDDEN)
    model8.eval()

    u8 = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    v8 = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    x8 = np.array(polar_encode(u8))
    y8 = np.array(polar_encode(v8))
    br8 = bit_reversal_perm(3)
    joint8 = torch.tensor([x8[br8] * 2 + y8[br8]]).long()
    emb8 = torch.randn(1, N8, D)

    with torch.no_grad():
        p_embs8, p_tgt8, p_bn8 = model8.fast_ce_leaf_embs(emb8, joint8)

    seq_tgt8 = torch.tensor([u8 * 2 + v8]).long()
    with torch.no_grad():
        s_leaf8, s_bn8 = model8.sc_decode_leaf_embs(emb8, seq_tgt8)

    print("\nBitnode uv_left at each depth:")
    print("  PARALLEL:")
    for _, depth, vals in p_bn8:
        print(f"    depth {depth}: {vals[0].tolist()}")
    print("  SEQUENTIAL:")
    for _, depth, vals in s_bn8:
        print(f"    depth {depth}: {vals[0].tolist()}")

    mismatches = 0
    for (_, pd, pv), (_, sd, sv) in zip(p_bn8, s_bn8):
        if not torch.equal(pv, sv):
            mismatches += 1
    print(f"\n  Bitnode input mismatches: {mismatches}/{len(p_bn8)}")

    n_leaf_mismatch = 0
    for idx, s_emb in s_leaf8:
        p_emb = p_embs8[0, idx, :]
        if not torch.allclose(p_emb, s_emb[0], atol=1e-5):
            n_leaf_mismatch += 1
    print(f"  Leaf embedding mismatches: {n_leaf_mismatch}/{N8}")

    # ─── Root cause summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROOT CAUSE SUMMARY")
    print("=" * 70)
    print("""
BUG: Domain mismatch in bitnode's uv_left argument.

PARALLEL (fast_ce):
  At each depth, J_left = XOR(J_odd, J_even) in the CODEWORD domain.
  This is the sub-CODEWORD of the left subtree.
  bitnode receives these codeword-domain values as uv_left.

SEQUENTIAL (sc_decode):
  At each depth, _decode(e_left) returns decoded LEAF values.
  These are MESSAGE-domain values (after recursive decoding).
  bitnode receives these message-domain values as uv_left.

RELATIONSHIP:
  J_left = polar_encode(leaf_values) for each sub-block.
  They are the SAME only when the polar encoding is trivial (identity),
  which happens only at the leaf level (size 1). At higher levels,
  the encoding mixes bits, so codeword != message.

CONSEQUENCE:
  The bitnode sign-flipping uses (1 - 2*bit). A wrong domain means
  wrong signs on the first half of the embedding. The model learns
  to rely on codeword-domain signs during training but gets message-domain
  signs at inference. Since the domains differ ~50% of the time,
  this explains the ~25% accuracy (random guessing on 4 classes).

FIX OPTIONS:
  1. Fix fast_ce: pass message-domain values to bitnode (requires
     recursive computation, losing the parallelism advantage).
  2. Fix sc_decode: re-encode decoded leaves before passing to bitnode
     (apply polar_encode to the decoded sub-messages at each level).
  3. Redesign: use codeword-domain values everywhere. In sc_decode,
     the return from _decode should be the sub-codeword, not sub-message.
     This means _decode returns polar_encode(decoded_leaves) at each level.

Option 3 is cleanest: make _decode return XOR-combined (codeword-domain)
values that match what fast_ce produces. Concretely, for a size-2 subtree
with leaves [a, b], _decode should return [a^b, b] instead of [a, b].
""")


if __name__ == '__main__':
    main()
