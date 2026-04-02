"""
poc_snapshot_pure.py
====================
Snapshot-only training POC for neural MAC polar decoder.

Key idea: ALL neural components operate on logits2emb(log(prob)) embeddings,
so there is no mismatch between snapshot training and inference.

The interface between tree operations is ALWAYS a (4,) probability vector.
Each neural operation:
  1. Takes probability vector inputs
  2. Internally converts to embeddings via logits2emb(log(prob))
  3. Processes via MLP
  4. Outputs probability vector via softmax(emb2logits(output_emb))

This ensures composability: snapshot training matches inference exactly.
"""

import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from polar.channels import GaussianMAC
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.decoder import build_log_W_leaf, _norm_prod_single
from polar.design import design_gmac, make_path

torch.manual_seed(42)
np.random.seed(42)

D_EMB = 32
D_HIDDEN = 128
EPS = 1e-10


# =============================================================================
#  Model: prob-in, prob-out operations
# =============================================================================

class SnapshotModel(nn.Module):
    """
    Neural polar code operations. All operate on probability vectors.

    CalcLeft:  (parent_top_prob[4], parent_bot_prob[4], right_prob[4]) -> output_prob[4]
    CalcRight: (parent_top_prob[4], parent_bot_prob[4], left_prob[4])  -> output_prob[4]
    CalcParent: (left_prob[4], right_prob[4]) -> parent_top_prob[4]
      (parent_bot = right, as in the analytical decoder)

    All go through: log -> logits2emb -> MLP -> emb2logits -> softmax
    """

    def __init__(self, d=D_EMB, h=D_HIDDEN):
        super().__init__()
        self.d = d

        # Shared encoder/decoder for probability vectors
        self.logits2emb = nn.Sequential(
            nn.Linear(4, h), nn.GELU(),
            nn.Linear(h, d),
        )

        self.emb2logits = nn.Sequential(
            nn.Linear(d, h), nn.GELU(),
            nn.Linear(h, 4),
        )

        # CalcLeft: takes 3 embeddings -> 1 embedding
        self.calc_left_net = nn.Sequential(
            nn.Linear(3 * d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, d),
        )

        # CalcRight: takes 3 embeddings -> 1 embedding
        self.calc_right_net = nn.Sequential(
            nn.Linear(3 * d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, d),
        )

        # CalcParent: takes 2 embeddings -> 1 embedding
        self.calc_parent_net = nn.Sequential(
            nn.Linear(2 * d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, d),
        )

    def _to_emb(self, prob):
        """prob: (..., 4) -> (..., d) via log -> linear."""
        log_p = torch.log(prob.clamp(min=EPS))
        return self.logits2emb(log_p)

    def _to_prob(self, emb):
        """emb: (..., d) -> (..., 4) via linear -> softmax."""
        logits = self.emb2logits(emb)
        return F.softmax(logits, dim=-1)

    def calc_left(self, parent_top, parent_bot, right):
        """All inputs: (..., 4) prob vectors. Output: (..., 4) prob vector."""
        e1 = self._to_emb(parent_top)
        e2 = self._to_emb(parent_bot)
        e3 = self._to_emb(right)
        out_emb = self.calc_left_net(torch.cat([e1, e2, e3], dim=-1))
        return self._to_prob(out_emb)

    def calc_right(self, parent_top, parent_bot, left):
        """All inputs: (..., 4) prob vectors. Output: (..., 4) prob vector."""
        e1 = self._to_emb(parent_top)
        e2 = self._to_emb(parent_bot)
        e3 = self._to_emb(left)
        out_emb = self.calc_right_net(torch.cat([e1, e2, e3], dim=-1))
        return self._to_prob(out_emb)

    def calc_parent(self, left, right):
        """left, right: (..., 4) prob vectors. Output: (..., 4) parent_top prob."""
        e1 = self._to_emb(left)
        e2 = self._to_emb(right)
        out_emb = self.calc_parent_net(torch.cat([e1, e2], dim=-1))
        return self._to_prob(out_emb)


# =============================================================================
#  Snapshot generation
# =============================================================================

def prob_from_log(log_tensor):
    """Convert (2,2) log-prob tensor to (4,) prob vector, normalized."""
    flat = log_tensor.ravel()
    p = np.exp(flat - np.max(flat))
    s = p.sum()
    return p / s if s > 0 else np.ones(4) / 4.0


def generate_snapshots(n, N, num_cw, sigma2, ku, kv, path_i):
    """
    Run analytical tensor SC decoder and record CalcLeft/CalcRight snapshots.

    For CalcLeft at vertex beta with l = right.shape[0]:
      For each i in [0, l):
        temp_i = norm_prod(parent[l+i], right[i])
        output_i = circ_conv(parent[i], temp_i)
        Snapshot: (parent[i], parent[l+i], right[i]) -> output_i

    For CalcRight at vertex beta with l = left.shape[0]:
      For each i in [0, l):
        temp_i = circ_conv(left[i], parent[i])
        output_i = norm_prod(parent[l+i], temp_i)
        Snapshot: (parent[i], parent[l+i], left[i]) -> output_i

    Also record CalcParent snapshots for training CalcParent:
      parent_top[i] = circ_conv(left[i], right[i])
      Snapshot: (left[i], right[i]) -> parent_top[i]
    """
    channel = GaussianMAC(sigma2=sigma2)
    Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)
    b = make_path(N, path_i)

    left_snaps = []
    right_snaps = []
    parent_snaps = []

    from polar.decoder import (_CompGraph, _norm_prod_single, _LOG_HALF, _LOG_QUARTER,
                                _circ_conv_batch, _norm_prod_batch)

    _NEG_INF = -np.inf

    class InstrumentedCompGraph(_CompGraph):
        def calc_left(self, beta):
            parent = self.edge_data[beta]
            right = self.edge_data[2 * beta + 1]
            l = right.shape[0]
            for i in range(l):
                temp_i = _norm_prod_single(parent[l + i], right[i])
                out_i = _circ_conv_batch(parent[i:i+1], temp_i[np.newaxis])[0]
                left_snaps.append((
                    prob_from_log(parent[i]),
                    prob_from_log(parent[l + i]),
                    prob_from_log(right[i]),
                    prob_from_log(out_i),
                ))
            super().calc_left(beta)

        def calc_right(self, beta):
            parent = self.edge_data[beta]
            left = self.edge_data[2 * beta]
            l = left.shape[0]
            for i in range(l):
                temp_i = _circ_conv_batch(left[i:i+1], parent[i:i+1])[0]
                out_i = _norm_prod_single(parent[l + i], temp_i)
                right_snaps.append((
                    prob_from_log(parent[i]),
                    prob_from_log(parent[l + i]),
                    prob_from_log(left[i]),
                    prob_from_log(out_i),
                ))
            super().calc_right(beta)

        def calc_parent(self, beta):
            left = self.edge_data[2 * beta]
            right = self.edge_data[2 * beta + 1]
            l = left.shape[0]
            # parent[:l] = circ_conv(left, right)
            parent_top = _circ_conv_batch(left, right)
            for i in range(l):
                parent_snaps.append((
                    prob_from_log(left[i]),
                    prob_from_log(right[i]),
                    prob_from_log(parent_top[i]),
                ))
            super().calc_parent(beta)

    for trial in range(num_cw):
        u_msg = np.zeros(N, dtype=np.int32)
        v_msg = np.zeros(N, dtype=np.int32)
        for i in range(1, N + 1):
            if i not in frozen_u:
                u_msg[i - 1] = np.random.randint(0, 2)
            if i not in frozen_v:
                v_msg[i - 1] = np.random.randint(0, 2)

        x_enc = polar_encode_batch(u_msg.reshape(1, -1))[0]
        y_enc = polar_encode_batch(v_msg.reshape(1, -1))[0]
        z = channel.sample_batch(x_enc, y_enc)
        log_W = build_log_W_leaf(z, channel)

        graph = InstrumentedCompGraph(n, log_W)
        u_hat = {}
        v_hat = {}
        i_u = 0
        i_v = 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; frozen_dict = frozen_u
            else:
                i_v += 1; i_t = i_v; frozen_dict = frozen_v

            leaf_edge = i_t + N - 1
            target_vertex = leaf_edge >> 1
            graph.step_to(target_vertex)

            temp = graph.edge_data[leaf_edge][0].copy()
            if leaf_edge & 1 == 0:
                graph.calc_left(target_vertex)
            else:
                graph.calc_right(target_vertex)

            top_down = graph.edge_data[leaf_edge][0]
            combined = _norm_prod_single(top_down, temp)

            if i_t in frozen_dict:
                bit = frozen_dict[i_t]
            else:
                if gamma == 0:
                    p0 = np.logaddexp(combined[0, 0], combined[0, 1])
                    p1 = np.logaddexp(combined[1, 0], combined[1, 1])
                    bit = 1 if p1 > p0 else 0
                else:
                    p0 = np.logaddexp(combined[0, 0], combined[1, 0])
                    p1 = np.logaddexp(combined[0, 1], combined[1, 1])
                    bit = 1 if p1 > p0 else 0

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            new_leaf = np.full((2, 2), _NEG_INF, dtype=np.float64)
            u_val = u_hat.get(i_t)
            v_val = v_hat.get(i_t)
            if u_val is not None and v_val is not None:
                new_leaf[u_val, v_val] = 0.0
            elif u_val is not None:
                new_leaf[u_val, 0] = _LOG_HALF
                new_leaf[u_val, 1] = _LOG_HALF
            elif v_val is not None:
                new_leaf[0, v_val] = _LOG_HALF
                new_leaf[1, v_val] = _LOG_HALF
            else:
                new_leaf[:, :] = _LOG_QUARTER
            graph.edge_data[leaf_edge][0] = new_leaf

    print(f"  Generated {len(left_snaps)} L, {len(right_snaps)} R, "
          f"{len(parent_snaps)} P snapshots")
    return left_snaps, right_snaps, parent_snaps


# =============================================================================
#  Training
# =============================================================================

def train_snapshot(model, left_snaps, right_snaps, parent_snaps,
                   n_iters=10000, batch_size=256, lr=1e-3):
    """Train all NNs jointly on snapshot data using soft CE loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def to_tensors(snaps, n_in):
        """Convert list of tuples to tensors."""
        arrays = [np.array([s[i] for s in snaps]) for i in range(n_in + 1)]
        return [torch.tensor(a, dtype=torch.float32) for a in arrays]

    left_t = to_tensors(left_snaps, 3)   # [in1, in2, in3, out]
    right_t = to_tensors(right_snaps, 3)
    parent_t = to_tensors(parent_snaps, 2)  # [in1, in2, out]

    n_left = left_t[0].shape[0]
    n_right = right_t[0].shape[0]
    n_parent = parent_t[0].shape[0]

    print(f"\n  Training: {n_left} L + {n_right} R + {n_parent} P snapshots")
    print(f"  Iters: {n_iters}, Batch: {batch_size}, LR: {lr}")

    losses = []
    t0 = time.time()

    for it in range(n_iters):
        bs = batch_size // 3

        # Sample batches
        idx_l = torch.randint(0, n_left, (bs,))
        idx_r = torch.randint(0, n_right, (bs,))
        idx_p = torch.randint(0, n_parent, (bs,))

        # CalcLeft
        l_pred = model.calc_left(left_t[0][idx_l], left_t[1][idx_l], left_t[2][idx_l])
        l_target = left_t[3][idx_l]
        loss_l = -torch.sum(l_target * torch.log(l_pred.clamp(min=EPS)), dim=-1).mean()

        # CalcRight
        r_pred = model.calc_right(right_t[0][idx_r], right_t[1][idx_r], right_t[2][idx_r])
        r_target = right_t[3][idx_r]
        loss_r = -torch.sum(r_target * torch.log(r_pred.clamp(min=EPS)), dim=-1).mean()

        # CalcParent
        p_pred = model.calc_parent(parent_t[0][idx_p], parent_t[1][idx_p])
        p_target = parent_t[2][idx_p]
        loss_p = -torch.sum(p_target * torch.log(p_pred.clamp(min=EPS)), dim=-1).mean()

        loss = loss_l + loss_r + loss_p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if it % 2000 == 0 or it == n_iters - 1:
            elapsed = time.time() - t0
            print(f"    iter {it:5d}  loss={loss.item():.4f}  "
                  f"(L={loss_l.item():.4f} R={loss_r.item():.4f} P={loss_p.item():.4f})  "
                  f"[{elapsed:.1f}s]")

    return losses


# =============================================================================
#  Neural SC decode — prob-vector interface
# =============================================================================

def neural_sc_decode(model, log_W, n, N, b, frozen_u, frozen_v):
    """
    Full SC decode using trained neural components.

    Edge data stored as numpy (L, 4) probability vectors.
    Neural operations take prob vectors in, return prob vectors out.
    """
    br = bit_reversal_perm(n)

    # Build root: bit-reversed, normalized probability vectors
    log_W_br = log_W[br].copy()
    for t in range(N):
        total = np.logaddexp(
            np.logaddexp(log_W_br[t, 0, 0], log_W_br[t, 0, 1]),
            np.logaddexp(log_W_br[t, 1, 0], log_W_br[t, 1, 1])
        )
        if np.isfinite(total):
            log_W_br[t] -= total

    # Convert to prob vectors (N, 4)
    root_probs = np.exp(log_W_br.reshape(N, 4))
    root_probs = root_probs / (root_probs.sum(axis=1, keepdims=True) + EPS)

    model.eval()
    with torch.no_grad():
        # Edge data: edge_data[e] = numpy array (size, 4) of prob vectors
        edge_data = [None] * (2 * N)
        edge_data[1] = root_probs  # (N, 4)

        # Initialize other edges with uniform
        uniform = np.ones(4) / 4.0
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_data[beta] = np.tile(uniform, (size, 1))

        def do_calc_left(beta):
            parent = edge_data[beta]  # (2l, 4)
            right = edge_data[2 * beta + 1]  # (l, 4)
            l = right.shape[0]
            pt = torch.tensor(parent[:l], dtype=torch.float32)
            pb = torch.tensor(parent[l:], dtype=torch.float32)
            rt = torch.tensor(right, dtype=torch.float32)
            result = model.calc_left(pt, pb, rt)
            edge_data[2 * beta] = result.numpy()

        def do_calc_right(beta):
            parent = edge_data[beta]
            left = edge_data[2 * beta]
            l = left.shape[0]
            pt = torch.tensor(parent[:l], dtype=torch.float32)
            pb = torch.tensor(parent[l:], dtype=torch.float32)
            lt = torch.tensor(left, dtype=torch.float32)
            result = model.calc_right(pt, pb, lt)
            edge_data[2 * beta + 1] = result.numpy()

        def do_calc_parent(beta):
            left = edge_data[2 * beta]   # (l, 4)
            right = edge_data[2 * beta + 1]  # (l, 4)
            lt = torch.tensor(left, dtype=torch.float32)
            rt = torch.tensor(right, dtype=torch.float32)
            parent_top = model.calc_parent(lt, rt).numpy()
            # parent_bot = right (as in analytical decoder)
            edge_data[beta] = np.concatenate([parent_top, right], axis=0)

        # Navigation
        dec_head = 1

        def get_path(current, target):
            if current == target:
                return []
            path_up = []
            path_down = []
            c, t = current, target
            while c != t:
                if c > t:
                    c = c >> 1
                    path_up.append(c)
                else:
                    path_down.append(t)
                    t = t >> 1
            path_down.reverse()
            return path_up + path_down

        def step_one(beta):
            nonlocal dec_head
            if beta == dec_head >> 1:
                do_calc_parent(dec_head)
                dec_head = beta
            elif beta >> 1 == dec_head:
                if beta & 1 == 0:
                    do_calc_left(dec_head)
                else:
                    do_calc_right(dec_head)
                dec_head = beta

        def step_to(target):
            nonlocal dec_head
            if dec_head == target:
                return
            path = get_path(dec_head, target)
            for beta in path:
                step_one(beta)

        # Decode
        u_hat = {}
        v_hat = {}
        i_u = 0
        i_v = 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; frozen_dict = frozen_u
            else:
                i_v += 1; i_t = i_v; frozen_dict = frozen_v

            leaf_edge = i_t + N - 1
            target_vertex = leaf_edge >> 1

            step_to(target_vertex)

            # Save leaf data (bottom-up info = getAsParent)
            temp = edge_data[leaf_edge][0].copy()  # (4,)

            # Compute top-down message
            if leaf_edge & 1 == 0:
                do_calc_left(target_vertex)
            else:
                do_calc_right(target_vertex)

            top_down = edge_data[leaf_edge][0]  # (4,)

            # Combine: normalized product (elementwise multiply + renormalize)
            combined = top_down * temp
            s = combined.sum()
            if s > EPS:
                combined = combined / s
            else:
                combined = np.ones(4) / 4.0

            # Decide
            if i_t in frozen_dict:
                bit = frozen_dict[i_t]
            else:
                # combined is [p(0,0), p(0,1), p(1,0), p(1,1)]
                if gamma == 0:  # deciding U (x)
                    p0 = combined[0] + combined[1]  # x=0
                    p1 = combined[2] + combined[3]  # x=1
                    bit = 1 if p1 > p0 else 0
                else:  # deciding V (y)
                    p0 = combined[0] + combined[2]  # y=0
                    p1 = combined[1] + combined[3]  # y=1
                    bit = 1 if p1 > p0 else 0

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            # Set leaf to partially deterministic prob
            u_val = u_hat.get(i_t)
            v_val = v_hat.get(i_t)
            new_prob = np.ones(4) * 0.25
            if u_val is not None and v_val is not None:
                new_prob = np.zeros(4)
                new_prob[u_val * 2 + v_val] = 1.0
            elif u_val is not None:
                new_prob = np.zeros(4)
                new_prob[u_val * 2 + 0] = 0.5
                new_prob[u_val * 2 + 1] = 0.5
            elif v_val is not None:
                new_prob = np.zeros(4)
                new_prob[0 * 2 + v_val] = 0.5
                new_prob[1 * 2 + v_val] = 0.5
            edge_data[leaf_edge] = new_prob.reshape(1, 4)

    u_dec = [u_hat.get(k, 0) for k in range(1, N + 1)]
    v_dec = [v_hat.get(k, 0) for k in range(1, N + 1)]
    return u_dec, v_dec


# =============================================================================
#  Evaluation
# =============================================================================

def eval_bler(model, n, N, num_cw, sigma2, ku, kv, path_i, label=""):
    """Evaluate BLER using neural decoder."""
    channel = GaussianMAC(sigma2=sigma2)
    Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)
    b = make_path(N, path_i)

    block_errors = 0
    info_u_pos = set(range(1, N + 1)) - set(frozen_u.keys())
    info_v_pos = set(range(1, N + 1)) - set(frozen_v.keys())
    u_bit_errors = 0
    v_bit_errors = 0

    for trial in range(num_cw):
        u_msg = np.zeros(N, dtype=np.int32)
        v_msg = np.zeros(N, dtype=np.int32)
        for i in range(1, N + 1):
            if i not in frozen_u:
                u_msg[i - 1] = np.random.randint(0, 2)
            if i not in frozen_v:
                v_msg[i - 1] = np.random.randint(0, 2)

        x_enc = polar_encode_batch(u_msg.reshape(1, -1))[0]
        y_enc = polar_encode_batch(v_msg.reshape(1, -1))[0]
        z = channel.sample_batch(x_enc, y_enc)
        log_W = build_log_W_leaf(z, channel)

        u_dec, v_dec = neural_sc_decode(model, log_W, n, N, b, frozen_u, frozen_v)

        u_err = sum(1 for i in info_u_pos if u_dec[i - 1] != u_msg[i - 1])
        v_err = sum(1 for i in info_v_pos if v_dec[i - 1] != v_msg[i - 1])
        u_bit_errors += u_err
        v_bit_errors += v_err

        if u_err > 0 or v_err > 0:
            block_errors += 1

    bler = block_errors / num_cw
    total_u = num_cw * len(info_u_pos)
    total_v = num_cw * len(info_v_pos)
    print(f"  {label}N={N}: BLER={bler:.4f} ({block_errors}/{num_cw})  "
          f"U-BER={u_bit_errors/max(1,total_u):.4f}  "
          f"V-BER={v_bit_errors/max(1,total_v):.4f}  "
          f"ku={len(info_u_pos)} kv={len(info_v_pos)}")
    return bler


def eval_bler_analytical(n, N, num_cw, sigma2, ku, kv, path_i, label=""):
    """Evaluate BLER using analytical SC decoder (reference)."""
    from polar.decoder import decode_single
    channel = GaussianMAC(sigma2=sigma2)
    Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)
    b = make_path(N, path_i)

    block_errors = 0
    info_u_pos = set(range(1, N + 1)) - set(frozen_u.keys())
    info_v_pos = set(range(1, N + 1)) - set(frozen_v.keys())

    for trial in range(num_cw):
        u_msg = np.zeros(N, dtype=np.int32)
        v_msg = np.zeros(N, dtype=np.int32)
        for i in range(1, N + 1):
            if i not in frozen_u:
                u_msg[i - 1] = np.random.randint(0, 2)
            if i not in frozen_v:
                v_msg[i - 1] = np.random.randint(0, 2)

        x_enc = polar_encode_batch(u_msg.reshape(1, -1))[0]
        y_enc = polar_encode_batch(v_msg.reshape(1, -1))[0]
        z = channel.sample_batch(x_enc, y_enc)

        u_dec, v_dec = decode_single(N, z.tolist(), b, frozen_u, frozen_v, channel)

        u_err = sum(1 for i in info_u_pos if u_dec[i - 1] != u_msg[i - 1])
        v_err = sum(1 for i in info_v_pos if v_dec[i - 1] != v_msg[i - 1])
        if u_err > 0 or v_err > 0:
            block_errors += 1

    bler = block_errors / num_cw
    print(f"  {label}N={N}: BLER={bler:.4f} ({block_errors}/{num_cw})  "
          f"ku={len(info_u_pos)} kv={len(info_v_pos)}")
    return bler


# =============================================================================
#  Sanity check: verify snapshot accuracy
# =============================================================================

def verify_snapshots(left_snaps, right_snaps, parent_snaps, n_show=5):
    """Print a few snapshots to verify they make sense."""
    print(f"\n  Sample CalcLeft snapshots (first {n_show}):")
    for i in range(min(n_show, len(left_snaps))):
        s = left_snaps[i]
        print(f"    in1={s[0][:2]}.. in2={s[1][:2]}.. in3={s[2][:2]}.. -> out={s[3][:2]}..")

    print(f"\n  Entropy of outputs:")
    for name, snaps in [("Left", left_snaps), ("Right", right_snaps), ("Parent", parent_snaps)]:
        outs = np.array([s[-1] for s in snaps])
        ent = -np.sum(outs * np.log(outs + EPS), axis=1)
        print(f"    {name}: mean_entropy={ent.mean():.4f}, "
              f"min={ent.min():.4f}, max={ent.max():.4f}")


# =============================================================================
#  Main
# =============================================================================

def main():
    print("=" * 70)
    print("POC: Snapshot-Only Neural MAC Polar Decoder (v2 - prob interface)")
    print("=" * 70)

    # Parameters
    SNR_dB = 6.0
    sigma2 = 10 ** (-SNR_dB / 10)
    path_i_N8 = 4  # Class B
    n_train = 3
    N_train = 8
    ku_8 = 3
    kv_8 = 4

    print(f"\nConfig: SNR={SNR_dB}dB, sigma2={sigma2:.4f}")
    print(f"  N={N_train}, n={n_train}, ku={ku_8}, kv={kv_8}, path_i={path_i_N8}")

    # Step 1: Analytical SC reference
    print("\n--- Step 1: Analytical SC reference ---")
    np.random.seed(123)
    ref_bler_8 = eval_bler_analytical(n_train, N_train, 1000, sigma2, ku_8, kv_8,
                                       path_i_N8, label="[SC ref] ")

    # Step 2: Generate snapshots
    print("\n--- Step 2: Generate snapshots ---")
    np.random.seed(42)
    t0 = time.time()
    left_snaps, right_snaps, parent_snaps = generate_snapshots(
        n_train, N_train, 2000, sigma2, ku_8, kv_8, path_i_N8
    )
    print(f"  Snapshot generation took {time.time()-t0:.1f}s")

    verify_snapshots(left_snaps, right_snaps, parent_snaps)

    # Step 3: Train
    print("\n--- Step 3: Train snapshot model ---")
    model = SnapshotModel(d=D_EMB, h=D_HIDDEN)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params}")

    t0 = time.time()
    losses = train_snapshot(model, left_snaps, right_snaps, parent_snaps,
                            n_iters=15000, batch_size=384, lr=5e-4)
    train_time = time.time() - t0
    print(f"  Training took {train_time:.1f}s")
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Step 4: Evaluate at N=8
    print("\n--- Step 4: Neural decode at N=8 ---")
    np.random.seed(123)
    neural_bler_8 = eval_bler(model, n_train, N_train, 1000, sigma2, ku_8, kv_8,
                               path_i_N8, label="[Neural] ")

    # Step 5: Evaluate at N=16 WITHOUT retraining
    print("\n--- Step 5: Neural decode at N=16 (zero-shot) ---")
    n_16 = 4
    N_16 = 16
    ku_16 = 6
    kv_16 = 8
    path_i_16 = 8

    np.random.seed(456)
    ref_bler_16 = eval_bler_analytical(n_16, N_16, 500, sigma2, ku_16, kv_16,
                                        path_i_16, label="[SC ref] ")
    np.random.seed(456)
    neural_bler_16 = eval_bler(model, n_16, N_16, 500, sigma2, ku_16, kv_16,
                                path_i_16, label="[Neural] ")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  N=8:  SC BLER={ref_bler_8:.4f}  Neural BLER={neural_bler_8:.4f}")
    print(f"  N=16: SC BLER={ref_bler_16:.4f}  Neural BLER={neural_bler_16:.4f}")
    print(f"  Training: {len(left_snaps)+len(right_snaps)+len(parent_snaps)} snapshots, "
          f"{train_time:.1f}s")
    print(f"  Model: {total_params} params, d={D_EMB}, hidden={D_HIDDEN}")
    print("=" * 70)


if __name__ == "__main__":
    main()
