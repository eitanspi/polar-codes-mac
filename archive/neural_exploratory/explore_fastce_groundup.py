#!/usr/bin/env python3
"""
Ground-up exploration of fast_ce structure at N=4 and N=8.

CRITICAL FINDINGS:
1. fast_ce leaf targets != message targets (only 25% match at N=4)
2. At N=4, J_left depends on right-half message bits
3. At N=8, x[2k]^x[2k+1] depends only on left-half message bits (BUT this
   is just the codeword XOR, not the full J_left after bit-reversal)

The root cause: bit-reversal SCRAMBLES which codeword positions correspond
to which message positions. The fast_ce interleaved decomposition operates
on BIT-REVERSED codeword positions, not the natural message order.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm


def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class TracingDecoder(nn.Module):
    def __init__(self, d=4, hidden=8):
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

    def fast_ce_full_trace(self, emb, joint_cw):
        B, N, d = emb.shape
        n = int(math.log2(N))
        bn_inputs = {}

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
            J_right = J_even

            bn_inputs[depth] = J_left.clone()

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = list(torch.split(e_left, cs, 1))
            er = list(torch.split(e_right, cs, 1))
            jl = list(torch.split(J_left, cs, 1))
            jr = list(torch.split(J_right, cs, 1))

            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

        e_all = torch.cat(E_chunks, 1)
        j_all = torch.cat(J_chunks, 1)
        return e_all, j_all, bn_inputs

    def sc_decode_cw_domain(self, emb, true_cw_targets):
        """
        Sequential SC decode where leaf targets are in CODEWORD domain
        (matching fast_ce leaf targets), and the return value at each level
        reconstructs the parent's codeword via butterfly.
        """
        B = emb.shape[0]
        N = emb.shape[1]
        leaf_idx = [0]
        leaf_embs = {}
        bn_inputs = []

        def _decode(emb_block, depth=0):
            block_size = emb_block.shape[1]
            if block_size == 1:
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                leaf_embs[idx] = emb_block[:, 0, :].clone()
                dec = true_cw_targets[:, idx:idx+1].clone()
                return dec

            e_odd = emb_block[:, 0::2, :]
            e_even = emb_block[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            cw_left = _decode(e_left, depth + 1)

            bn_inputs.append((depth, cw_left.clone()))

            e_right = self.bitnode(e_odd, e_even, cw_left)
            cw_right = _decode(e_right, depth + 1)

            u_l = cw_left // 2;  v_l = cw_left % 2
            u_r = cw_right // 2; v_r = cw_right % 2
            cw_odd = (u_l ^ u_r) * 2 + (v_l ^ v_r)
            cw_even = cw_right
            result = torch.zeros(B, block_size, dtype=torch.long)
            result[:, 0::2] = cw_odd
            result[:, 1::2] = cw_even
            return result

        with torch.no_grad():
            full_cw = _decode(emb)
        return leaf_embs, bn_inputs, full_cw


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Polar Transform Structure
# ═══════════════════════════════════════════════════════════════════════

def part1():
    print("=" * 70)
    print("PART 1: Polar Transform and Bit-Reversal at N=4")
    print("=" * 70)

    br = bit_reversal_perm(2)
    u = np.array([1, 1, 0, 1])
    x = np.array(polar_encode(u))
    print(f"  br = {br}")
    print(f"  u = {u}, x = polar_encode(u) = {x}")

    # Show G_4
    F_k = np.array([[1,0],[1,1]])
    F2 = np.kron(F_k, F_k)
    B = np.zeros((4,4), dtype=int)
    for i in range(4): B[i, br[i]] = 1
    G = (B @ F2) % 2
    print(f"  G_4 (rows are basis vectors):")
    for i in range(4):
        print(f"    u[{i}] contributes to x: {G[i]}")

    # Verify
    all_ok = all(np.array_equal((np.array([(v>>i)&1 for i in range(4)]) @ G) % 2,
                                 np.array(polar_encode([(v>>i)&1 for i in range(4)])))
                 for v in range(16))
    print(f"  All 16 messages verified: {all_ok}")
    return br


# ═══════════════════════════════════════════════════════════════════════
# PART 2: fast_ce leaf targets vs message targets
# ═══════════════════════════════════════════════════════════════════════

def compute_fast_ce_leaves(J, N):
    """Pure-numpy fast_ce leaf target computation."""
    n = int(math.log2(N))
    chunks = [J.copy()]

    for depth in range(n):
        new_chunks = []
        for c in chunks:
            M = len(c)
            c_odd = c[0::2]
            c_even = c[1::2]
            c_left = ((c_odd//2) ^ (c_even//2)) * 2 + ((c_odd%2) ^ (c_even%2))
            c_right = c_even
            new_chunks.append(c_left)
            new_chunks.append(c_right)
        chunks = new_chunks

    return np.concatenate(chunks)


def part2(br):
    print("\n" + "=" * 70)
    print("PART 2: fast_ce leaf targets vs message targets")
    print("=" * 70)

    N = 4
    u = np.array([1, 1, 0, 1])
    v = np.array([0, 1, 1, 0])
    x = np.array(polar_encode(u))
    y = np.array(polar_encode(v))
    J = x[br] * 2 + y[br]
    msg = u * 2 + v

    leaves = compute_fast_ce_leaves(J, N)
    print(f"  u={u}, v={v}, x={x}, y={y}")
    print(f"  J (bit-reversed cw) = {J}")
    print(f"  fast_ce leaves = {leaves}")
    print(f"  msg targets    = {msg}")
    print(f"  Match: {np.array_equal(leaves, msg)}")

    # Exhaustive test
    n_match = sum(1 for uv in range(16) for vv in range(16)
                  if np.array_equal(
                      compute_fast_ce_leaves(
                          np.array(polar_encode([(uv>>i)&1 for i in range(N)]))[br] * 2 +
                          np.array(polar_encode([(vv>>i)&1 for i in range(N)]))[br],
                          N),
                      np.array([(uv>>i)&1 for i in range(N)]) * 2 +
                      np.array([(vv>>i)&1 for i in range(N)])))
    print(f"\n  Exhaustive N=4: {n_match}/256 match ({n_match/256*100:.0f}%)")

    # KEY: What IS the mapping from fast_ce leaves to message?
    # fast_ce undoes the encoding on x[br] and y[br] separately.
    # Undoing encoding = applying encoding again (G^2 = I over GF(2)).
    # So fast_ce_leaves_x = polar_encode(x[br]) and same for y.
    # But x[br] is the bit-reversed codeword.
    # polar_encode(x[br]) = (x[br])[br] then butterfly = x[br^2] then butterfly
    # Since br is an involution (br[br[i]] = i), x[br][br] = x.
    # So polar_encode(x[br]) applies butterfly to x, which gives...

    # Actually let me just check: are fast_ce leaves = polar_encode(x[br])?
    # Note: polar_encode includes bit-reversal + butterfly.
    x_br = x[br]
    encoded_xbr = np.array(polar_encode(x_br))
    print(f"\n  x[br] = {x_br}")
    print(f"  polar_encode(x[br]) = {encoded_xbr}")
    print(f"  u = {u}")
    print(f"  polar_encode(x[br]) == u? {np.array_equal(encoded_xbr, u)}")

    # So fast_ce does NOT just apply polar_encode to x[br].
    # The fast_ce decomposition is a DIFFERENT operation from polar_encode.
    # fast_ce does: at each level, split into odd/even, XOR for left, pass-through for right.
    # This is the BUTTERFLY INVERSION (without bit-reversal).

    # Let me verify: does applying JUST the butterfly (no bit-reversal) to x[br] give u?
    # Butterfly inversion = butterfly (it's self-inverse)
    def butterfly_only(w):
        w = w.copy()
        step = 1
        while step < len(w):
            w_r = w.reshape(-1, 2*step)
            w_r[:, :step] ^= w_r[:, step:]
            step *= 2
        return w

    xbr_butterfly = butterfly_only(x_br)
    print(f"\n  butterfly_only(x[br]) = {xbr_butterfly}")
    print(f"  u = {u}")
    print(f"  Match? {np.array_equal(xbr_butterfly, u)}")

    # If they match, fast_ce is applying the butterfly inversion to x[br],
    # which recovers u (since encoding = bit-reverse then butterfly, so
    # inverse = butterfly then bit-reverse, but starting from x[br] already
    # has the bit-reversal baked in).

    # But the fast_ce decomposition splits into INTERLEAVED odd/even,
    # not CONTIGUOUS halves. This is DIFFERENT from the butterfly!

    # The butterfly at step=1: XOR pairs (0,1), (2,3), etc.
    # fast_ce at depth 0: XOR odd/even = pairs (0,1), (2,3), etc.
    # These are the SAME!

    # The butterfly at step=2: XOR (0,2), (1,3) [left half ^= right half]
    # fast_ce at depth 1: within each chunk, XOR odd/even again.
    # For the LEFT chunk [Jl[0], Jl[1]]: XOR positions 0 and 1.
    # For the RIGHT chunk [Jr[0], Jr[1]]: XOR positions 0 and 1.

    # The butterfly's step=2 operates on the FULL array: pos 0 ^= pos 2, pos 1 ^= pos 3.
    # fast_ce depth 1 operates WITHIN each chunk.
    # These are DIFFERENT operations!

    # fast_ce is a RECURSIVE decomposition, not a flat butterfly.
    # Let me verify by computing fast_ce on just x (single user) for x[br]:

    fast_ce_x_leaves = compute_fast_ce_leaves(x_br * 2, N) // 2  # extract u-part
    print(f"\n  fast_ce applied to x[br] (u-component only):")
    print(f"  fast_ce leaves = {fast_ce_x_leaves}")
    print(f"  u = {u}")
    print(f"  butterfly(x[br]) = {xbr_butterfly}")

    # Are fast_ce leaves the same as butterfly(x[br])?
    print(f"  fast_ce == butterfly? {np.array_equal(fast_ce_x_leaves, xbr_butterfly)}")

    # Let me check ALL 16 single-user cases
    n_match_butterfly = 0
    n_match_msg = 0
    for uval in range(16):
        u_t = np.array([(uval>>i)&1 for i in range(N)])
        x_t = np.array(polar_encode(u_t))
        xbr_t = x_t[br]
        fc_leaves = compute_fast_ce_leaves(xbr_t, N)  # single-user, no *2
        bfly = butterfly_only(xbr_t)
        if np.array_equal(fc_leaves, u_t):
            n_match_msg += 1
        if np.array_equal(fc_leaves, bfly):
            n_match_butterfly += 1

    print(f"\n  Single-user N=4, all 16 messages:")
    print(f"    fast_ce == u (message): {n_match_msg}/16")
    print(f"    fast_ce == butterfly(x[br]): {n_match_butterfly}/16")


# ═══════════════════════════════════════════════════════════════════════
# PART 3: What IS the fast_ce decomposition?
# ═══════════════════════════════════════════════════════════════════════

def part3(br):
    print("\n" + "=" * 70)
    print("PART 3: What is the fast_ce decomposition exactly?")
    print("=" * 70)

    N = 4
    # fast_ce at each depth: split chunk into odd/even, left=odd^even, right=even
    # This is the inverse of: merge left,right -> result[odd]=left^right, result[even]=right
    # Which is: result[2i] = left[i] ^ right[i], result[2i+1] = right[i]
    # This is ONE BUTTERFLY STAGE with INTERLEAVED addressing.

    # Standard butterfly (step s): for blocks of size 2s,
    #   x[0..s-1] ^= x[s..2s-1]  (contiguous halves)

    # fast_ce butterfly: for each chunk, odd ^= even (interleaved halves)
    # This is the SAME as standard butterfly with step=1 applied to each chunk.
    # Then the chunks are split into left and right sub-chunks.

    # So the FULL fast_ce decomposition for N=4 on vector w:
    # Level 0: one chunk [w0,w1,w2,w3]
    #   odd=[w0,w2], even=[w1,w3]
    #   left = [w0^w1, w2^w3], right = [w1, w3]
    # Level 1:
    #   Chunk 0 (left): [w0^w1, w2^w3]
    #     odd=[w0^w1], even=[w2^w3]
    #     left = [(w0^w1)^(w2^w3)], right = [w2^w3]
    #   Chunk 1 (right): [w1, w3]
    #     odd=[w1], even=[w3]
    #     left = [w1^w3], right = [w3]
    # Leaves: [(w0^w1)^(w2^w3), w2^w3, w1^w3, w3]
    #        = [w0^w2^w3^... hmm let me just compute]

    # For w = x[br] = [x0, x2, x1, x3] (since br = [0,2,1,3]):
    # w0=x0, w1=x2, w2=x1, w3=x3
    # Leaves:
    #   L0 = (w0^w1)^(w2^w3) = (x0^x2)^(x1^x3) = x0^x1^x2^x3
    #   L1 = w2^w3 = x1^x3
    #   L2 = w1^w3 = x2^x3
    #   L3 = w3 = x3

    print(f"  fast_ce leaves for w = [w0,w1,w2,w3]:")
    print(f"    L0 = w0^w1^w2^w3")
    print(f"    L1 = w2^w3")
    print(f"    L2 = w1^w3")
    print(f"    L3 = w3")

    print(f"\n  With w = x[br] = [x[0],x[2],x[1],x[3]]:")
    print(f"    L0 = x[0]^x[2]^x[1]^x[3] = x0^x1^x2^x3")
    print(f"    L1 = x[1]^x[3]")
    print(f"    L2 = x[2]^x[3]")
    print(f"    L3 = x[3]")

    # Now, x = u * G_4 (mod 2). From Part 1:
    # G_4 = [[1,0,0,0],[1,0,1,0],[1,1,0,0],[1,1,1,1]]
    # x0 = u0 + u1 + u2 + u3
    # x1 = u2 + u3
    # x2 = u1 + u3
    # x3 = u3

    print(f"\n  Codeword in terms of message:")
    print(f"    x0 = u0+u1+u2+u3, x1 = u2+u3, x2 = u1+u3, x3 = u3")

    print(f"\n  fast_ce leaves in terms of message:")
    # L0 = x0+x1+x2+x3 = (u0+u1+u2+u3)+(u2+u3)+(u1+u3)+(u3) = u0+u3
    # Wait, mod 2: u0+u1+u2+u3+u2+u3+u1+u3+u3 = u0 + 2u1 + 2u2 + 4u3 = u0 mod 2
    # L0 = u0
    # L1 = x1+x3 = (u2+u3)+u3 = u2
    # L2 = x2+x3 = (u1+u3)+u3 = u1
    # L3 = x3 = u3

    print(f"    L0 = x0^x1^x2^x3 = u0")
    print(f"    L1 = x1^x3 = u2")
    print(f"    L2 = x2^x3 = u1")
    print(f"    L3 = x3 = u3")
    print(f"\n    So fast_ce leaves = [u0, u2, u1, u3] = u[br]!")

    # Verify
    u = np.array([1, 1, 0, 1])
    x = np.array(polar_encode(u))
    xbr = x[br]
    fc = compute_fast_ce_leaves(xbr, N)
    ubr = u[br]
    print(f"\n  Verification: u={u}, u[br]={ubr}")
    print(f"  fast_ce(x[br]) = {fc}")
    print(f"  u[br] = {ubr}")
    print(f"  Match: {np.array_equal(fc, ubr)}")

    # Exhaustive verification (single user)
    n_match = sum(1 for uval in range(16)
                  if np.array_equal(
                      compute_fast_ce_leaves(
                          np.array(polar_encode([(uval>>i)&1 for i in range(N)]))[br], N),
                      np.array([(uval>>i)&1 for i in range(N)])[br]))
    print(f"  Exhaustive single-user: fast_ce(x[br]) == u[br] for {n_match}/16")

    # For 2-user case: fast_ce leaves should be u[br]*2 + v[br]
    n_match2 = 0
    for uval in range(16):
        for vval in range(16):
            u_t = np.array([(uval>>i)&1 for i in range(N)])
            v_t = np.array([(vval>>i)&1 for i in range(N)])
            x_t = np.array(polar_encode(u_t))
            y_t = np.array(polar_encode(v_t))
            J = x_t[br] * 2 + y_t[br]
            fc = compute_fast_ce_leaves(J, N)
            expected = u_t[br] * 2 + v_t[br]
            if np.array_equal(fc, expected):
                n_match2 += 1
    print(f"  Exhaustive 2-user: fast_ce(x[br]*2+y[br]) == u[br]*2+v[br] for {n_match2}/256")

    print(f"\n  *** BREAKTHROUGH: fast_ce leaves = u[br]*2 + v[br] ***")
    print(f"  The fast_ce decomposition recovers the BIT-REVERSED MESSAGE!")
    print(f"  Not the natural-order message u*2+v.")

    return


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Verify for N=8 and N=16
# ═══════════════════════════════════════════════════════════════════════

def part4():
    print("\n" + "=" * 70)
    print("PART 4: Verify fast_ce_leaves = u[br]*2+v[br] for N=8,16")
    print("=" * 70)

    rng = np.random.default_rng(42)

    for N in [8, 16, 32]:
        n = int(math.log2(N))
        br = bit_reversal_perm(n)
        n_pass = 0
        n_tests = 1000
        for _ in range(n_tests):
            u = rng.integers(0, 2, N)
            v = rng.integers(0, 2, N)
            x = np.array(polar_encode(u))
            y = np.array(polar_encode(v))
            J = x[br] * 2 + y[br]
            fc = compute_fast_ce_leaves(J, N)
            expected = u[br] * 2 + v[br]
            if np.array_equal(fc, expected):
                n_pass += 1
        print(f"  N={N:3d}: fast_ce_leaves == u[br]*2+v[br] for {n_pass}/{n_tests}")


# ═══════════════════════════════════════════════════════════════════════
# PART 5: What does this mean for the bitnode inputs?
# ═══════════════════════════════════════════════════════════════════════

def part5(br):
    print("\n" + "=" * 70)
    print("PART 5: Bitnode inputs in fast_ce (intermediate codeword values)")
    print("=" * 70)

    N = 4
    u = np.array([1, 1, 0, 1])
    v = np.array([0, 1, 1, 0])
    x = np.array(polar_encode(u))
    y = np.array(polar_encode(v))
    J = x[br] * 2 + y[br]

    print(f"  u={u}, v={v}")
    print(f"  x={x}, y={y}")
    print(f"  J (input) = {J}")

    # Depth 0: J_left (bitnode input)
    Jo = J[0::2]; Je = J[1::2]
    Jl = ((Jo//2)^(Je//2))*2 + ((Jo%2)^(Je%2))
    print(f"\n  Depth 0: J_left = {Jl}")
    print(f"    These are the intermediate values after undoing one butterfly stage.")

    # What are the J_left values in terms of u[br]?
    # We showed fast_ce_leaves = u[br]. So the intermediate values are
    # partial polar encodings of u[br].
    # At depth 0: J_left[i] = the XOR of J[2i] and J[2i+1]
    # J is the codeword of u[br], and J_left is one butterfly step into
    # the inversion.

    # Key: the fast_ce's depth-0 J_left = intermediate values that the
    # sequential decoder's left subtree should produce.
    # In the sequential decoder, the left subtree decodes to message values
    # and we need to reconstruct the intermediate codeword values.

    # Since fast_ce_leaves = u[br], and the sequential decoder gets message
    # values u (not u[br]), we need a mapping.

    # At the leaves, fast_ce expects u[br[i]]*2+v[br[i]] as the target.
    # The sequential decoder knows u[i]*2+v[i] (message domain).
    # To convert: fast_ce_leaf[i] = u[br[i]]*2+v[br[i]] = msg[br[i]].

    # So the fast_ce leaf at position i has target msg[br[i]], NOT msg[i]!

    # This means the fix needs to:
    # 1. At each leaf, the model predicts msg[br[i]] (bit-reversed message)
    # 2. Store u_hat[br[i]] and v_hat[br[i]] from the prediction
    # 3. Return the prediction for codeword reconstruction

    # OR: train with codeword-domain targets everywhere and convert at the end.

    print(f"\n  fast_ce leaf targets: {compute_fast_ce_leaves(J, N)}")
    msg = u * 2 + v
    msg_br = msg[br]
    print(f"  msg = {msg}, msg[br] = {msg_br}")
    print(f"  fast_ce_leaves == msg[br]: {np.array_equal(compute_fast_ce_leaves(J, N), msg_br)}")

    # Now the critical question: can the sequential decoder's leaf values
    # be converted to match fast_ce's intermediate values?

    # In the sequential decoder, leaf i decides msg[i] (natural order).
    # But fast_ce's leaf i has target msg[br[i]].
    # If we change the sequential decoder to decide msg[br[i]] at leaf i,
    # then the bitnode inputs will be in the right domain.

    # BUT: the order of leaf processing in the sequential decoder is
    # 0, 1, 2, 3 (depth-first left-right). This is the SAME order as
    # fast_ce's leaves. So we just need the leaf targets to match.

    # If we teacher-force with msg[br] instead of msg, and then reconstruct
    # the codeword at each level using the butterfly, everything should match.

    print(f"\n  === THE CORRECT FIX ===")
    print(f"  1. In sc_decode, use msg[br] as leaf targets (not msg)")
    print(f"  2. After decoding, permute back: u_hat = u_hat_br[inv_br]")
    print(f"  3. The codeword reconstruction (butterfly per level) will then")
    print(f"     match fast_ce's intermediate values.")

    # Let me verify this with the model
    print(f"\n  Verifying with model...")

    torch.manual_seed(42)
    D = 4
    model = TracingDecoder(d=D, hidden=8)
    model.eval()

    joint_cw = torch.tensor([J]).long()
    joint_msg_br = torch.tensor([msg_br]).long()
    emb = torch.randn(1, N, D)

    with torch.no_grad():
        par_embs, par_tgt, par_bn = model.fast_ce_full_trace(emb, joint_cw)
        seq_embs, seq_bn, seq_cw = model.sc_decode_cw_domain(emb, joint_msg_br)

    print(f"\n  Parallel leaf targets: {par_tgt[0].tolist()}")
    print(f"  msg[br] targets:       {msg_br.tolist()}")
    print(f"  Match: {torch.equal(par_tgt, joint_msg_br)}")

    print(f"\n  Leaf embedding comparison (parallel vs seq with msg[br] targets):")
    all_match = True
    for i in range(N):
        p = par_embs[0, i]
        s = seq_embs[i][0]
        diff = (p - s).abs().max().item()
        status = "MATCH" if diff < 1e-5 else "MISMATCH"
        if diff >= 1e-5:
            all_match = False
        print(f"    Leaf {i}: {status} (max diff = {diff:.8f})")

    print(f"\n  ALL MATCH: {all_match}")

    # Check bitnode inputs
    print(f"\n  Bitnode inputs:")
    print(f"    Parallel:")
    for d in sorted(par_bn.keys()):
        print(f"      depth {d}: {par_bn[d][0].tolist()}")
    print(f"    Sequential (msg[br] targets):")
    for d, vals in seq_bn:
        print(f"      depth {d}: {vals[0].tolist()}")

    return all_match


# ═══════════════════════════════════════════════════════════════════════
# PART 6: Exhaustive verification of the correct fix
# ═══════════════════════════════════════════════════════════════════════

def part6():
    print("\n" + "=" * 70)
    print("PART 6: Exhaustive verification at N=4, N=8, N=16")
    print("=" * 70)

    rng = np.random.default_rng(42)

    for N in [4, 8, 16]:
        n = int(math.log2(N))
        br = bit_reversal_perm(n)
        D = 4

        torch.manual_seed(42)
        model = TracingDecoder(d=D, hidden=8)
        model.eval()

        n_tests = 50 if N <= 8 else 20
        n_pass = 0
        max_diff_all = 0.0

        for _ in range(n_tests):
            u = rng.integers(0, 2, N)
            v = rng.integers(0, 2, N)
            x = np.array(polar_encode(u))
            y = np.array(polar_encode(v))
            msg = u * 2 + v
            msg_br = msg[br]

            joint_cw = torch.tensor([x[br] * 2 + y[br]]).long()
            joint_msg_br = torch.tensor([msg_br]).long()
            emb = torch.randn(1, N, D)

            with torch.no_grad():
                par_embs, par_tgt, _ = model.fast_ce_full_trace(emb, joint_cw)
                seq_embs, _, _ = model.sc_decode_cw_domain(emb, joint_msg_br)

            ok = True
            for i in range(N):
                diff = (par_embs[0, i] - seq_embs[i][0]).abs().max().item()
                max_diff_all = max(max_diff_all, diff)
                if diff > 1e-4:
                    ok = False
            if ok:
                n_pass += 1

        print(f"  N={N:3d}: {n_pass}/{n_tests} pass, max diff = {max_diff_all:.2e}")


# ═══════════════════════════════════════════════════════════════════════
# PART 7: What the fix means for the real decoder
# ═══════════════════════════════════════════════════════════════════════

def part7(br):
    print("\n" + "=" * 70)
    print("PART 7: Implications for the real decoder")
    print("=" * 70)

    N = 4
    print(f"""
  THE BUG AND THE FIX:

  BUG: fast_ce trains with leaf targets = u[br]*2+v[br] (bit-reversed message),
       but sc_decode decides msg[i] = u[i]*2+v[i] (natural-order message).
       The bitnode conditioning uses wrong-domain values.

  FIX (Option A - fix sc_decode):
    1. At leaf i, the model predicts joint_br = u[br[i]]*2+v[br[i]]
    2. Store: u_hat[br[i]] = joint_br // 2, v_hat[br[i]] = joint_br % 2
    3. Return the raw prediction for codeword reconstruction up the tree
    4. After all leaves decoded, u_hat and v_hat are in natural order

  FIX (Option B - fix fast_ce):
    1. Change fast_ce to use msg-domain targets instead of cw-domain
    2. This means NOT passing x[br]*2+y[br] but instead u*2+v
    3. But this breaks the parallel decomposition structure

  FIX (Option C - change leaf target mapping):
    1. fast_ce already works correctly in its own domain
    2. Just change sc_decode's leaf handling:
       - Frozen bits: check if br[i] (not i) is frozen
       - Hard decision: store result at position br[i]
    3. Bitnode conditioning: convert leaf decisions to codeword domain
       using the butterfly reconstruction (as proposed)

  Option A is cleanest. The decoder naturally operates in the bit-reversed
  message domain. After decoding, apply inverse bit-reversal to get u_hat, v_hat.
""")

    # Verify frozen bit handling
    print(f"  Frozen bit example:")
    print(f"  br = {br}")
    print(f"  If position 0 is frozen (u[0]=0), in fast_ce this means")
    print(f"  the leaf at SC-tree position br^{{-1}}[0] has u-component frozen.")
    inv_br = np.argsort(br)
    print(f"  inv_br = {inv_br}")
    print(f"  br^{{-1}}[0] = {inv_br[0]} -> leaf {inv_br[0]} has frozen u-component")
    print(f"  br^{{-1}}[1] = {inv_br[1]} -> leaf {inv_br[1]} would be for msg position 1")

    # Actually, the leaf ORDER in both fast_ce and sc_decode is the same.
    # Leaf i in sc_decode corresponds to leaf i in fast_ce.
    # fast_ce leaf i has target msg[br[i]].
    # So for frozen bit at message position j, the affected leaf is br^{-1}[j].

    # In the current sc_decode, frozen_u is a set of message positions.
    # With the fix, leaf i needs to check if msg position br[i] is frozen.
    # So: frozen check at leaf i should be "br[i] in frozen_u" (not "i in frozen_u").

    print(f"\n  With the fix, frozen check at leaf i:")
    print(f"  u frozen at leaf i iff br[i] in frozen_u (not i in frozen_u)")
    print(f"  v frozen at leaf i iff br[i] in frozen_v")


# ═══════════════════════════════════════════════════════════════════════
# PART 8: Sign symmetry under wrong predictions (with correct fix)
# ═══════════════════════════════════════════════════════════════════════

def part8():
    print("\n" + "=" * 70)
    print("PART 8: Sign symmetry under wrong predictions")
    print("=" * 70)

    print(f"""
  With the correct fix (leaf i targets msg[br[i]]):

  During teacher-forced training (fast_ce):
    - All bitnode inputs are ground-truth codeword-domain values
    - Model learns the correct sign-flipping pattern

  During sequential inference:
    - Leaf i makes a prediction (possibly wrong)
    - This prediction is in the bit-reversed message domain
    - The butterfly reconstruction converts it to codeword domain
    - If the prediction is wrong, the wrong value propagates UP the tree
    - This is standard SC error propagation - unavoidable

  Key insight: the sign symmetry property is PRESERVED.
  If leaf i decides value d (correct or not), the bitnode sees a
  consistent codeword-domain value. The model was trained to handle
  all possible values at each bitnode position.

  The only issue is that wrong predictions cause cascading errors
  (as in any SC decoder). This is NOT a domain mismatch bug.
""")


# ═══════════════════════════════════════════════════════════════════════
# PART 9: Verify at N=8 with full model
# ═══════════════════════════════════════════════════════════════════════

def part9():
    print("\n" + "=" * 70)
    print("PART 9: Full model verification at N=8")
    print("=" * 70)

    N = 8; D = 4; n = 3
    br = bit_reversal_perm(n)

    torch.manual_seed(42)
    model = TracingDecoder(d=D, hidden=8)
    model.eval()

    u = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    v = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    x = np.array(polar_encode(u))
    y = np.array(polar_encode(v))
    msg = u * 2 + v
    msg_br = msg[br]

    print(f"  u = {u}")
    print(f"  v = {v}")
    print(f"  msg = {msg}")
    print(f"  br = {br}")
    print(f"  msg[br] = {msg_br}")

    joint_cw = torch.tensor([x[br] * 2 + y[br]]).long()
    joint_msg_br = torch.tensor([msg_br]).long()
    emb = torch.randn(1, N, D)

    with torch.no_grad():
        par_embs, par_tgt, par_bn = model.fast_ce_full_trace(emb, joint_cw)
        seq_embs, seq_bn, seq_cw = model.sc_decode_cw_domain(emb, joint_msg_br)

    print(f"\n  Parallel leaf targets: {par_tgt[0].tolist()}")
    print(f"  msg[br] = {msg_br.tolist()}")
    print(f"  Match: {torch.equal(par_tgt, joint_msg_br)}")

    print(f"\n  Leaf embeddings:")
    all_match = True
    for i in range(N):
        p = par_embs[0, i]
        s = seq_embs[i][0]
        diff = (p - s).abs().max().item()
        status = "MATCH" if diff < 1e-4 else "MISMATCH"
        if diff >= 1e-4:
            all_match = False
        print(f"    Leaf {i}: {status} (diff={diff:.2e})")
    print(f"  All match: {all_match}")

    print(f"\n  Bitnode inputs:")
    print(f"    Parallel: {', '.join(str(par_bn[d][0].tolist()) for d in sorted(par_bn.keys()))}")
    seq_by_depth = {}
    for d, v in seq_bn:
        if d not in seq_by_depth:
            seq_by_depth[d] = []
        seq_by_depth[d].append(v[0].tolist())
    for d in sorted(seq_by_depth.keys()):
        print(f"    Seq depth {d}: {seq_by_depth[d]}")


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def summary():
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  FINDING: fast_ce leaf targets = u[br] * 2 + v[br] (bit-reversed message)
           NOT u * 2 + v (natural-order message)

  This means:
  1. emb2logits at leaf position i is trained to predict msg[br[i]]
  2. The bitnode at depth d gets intermediate codeword-domain values
     that are consistent with the bit-reversed message domain
  3. The proposed fix (simple butterfly reconstruction) is WRONG because
     it operates on natural-order message values, not bit-reversed ones

  CORRECT FIX for sc_decode:
  - Leaf i should predict msg[br[i]] = u[br[i]]*2 + v[br[i]]
  - Frozen check: br[i] in frozen_u (not i in frozen_u)
  - Store: u_hat[br[i]] = prediction // 2, v_hat[br[i]] = prediction % 2
  - Return raw prediction for butterfly reconstruction up the tree
  - After butterfly reconstruction at each level, the return value
    matches fast_ce's intermediate codeword-domain values exactly

  VERIFIED: All leaf embeddings match between parallel fast_ce and
  sequential sc_decode when using msg[br] as leaf targets, for
  N = 4, 8, 16 across random messages.
""")


if __name__ == '__main__':
    br = part1()
    part2(br)
    part3(br)
    part4()
    all_match = part5(br)
    part6()
    part7(br)
    part8()
    part9()
    summary()
