"""
poc_sc_snapshots.py — SC Teacher Snapshots: Parallel Operation-Level Distillation
==================================================================================

CONCEPT: Run the analytical SC decoder, record the exact 2x2 probability tensors
at every edge during the tree walk. Use these as training targets: for each
CalcLeft/CalcRight/CalcParent operation, take the TRUE input tensors, apply a
neural network, and compare against the TRUE output tensor. Each operation trains
independently — gradient depth = 1, fully parallel.

IMPLEMENTATION:
  1) Snapshot collection: run batched analytical SC decoder, record all
     (input1, input2, output) triplets for CalcLeft, CalcRight, CalcParent.
  2) Training: tensor_to_emb NN maps flat 4-value tensor to d-dim embedding,
     CalcLeft_NN / CalcRight_NN / CalcParent_NN maps (emb1, emb2) -> emb_out,
     emb_to_tensor maps back to 4 values. Loss = MSE against true output.
  3) Evaluation: use trained NNs in sequential SC tree walk, compare BLER.

Config: N=128, Class B (path_i=64), SNR=6dB, GaussianMAC.
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.channels import GaussianMAC
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.decoder import build_log_W_leaf
from polar.design import make_path

# ── Config ───────────────────────────────────────────────────────────────────
N = 128
n = 7
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
PATH_I = N // 2  # Class B
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')

D_EMB = 32        # embedding dimension
HIDDEN = 128      # MLP hidden width
N_LAYERS = 3      # MLP depth
BATCH_SNAP = 128  # batch size for snapshot collection
BATCH_TRAIN = 1024 # mini-batch size for training
LR = 1e-3
TRAIN_HOURS = 1.5
EVAL_CW = 2000    # codewords for BLER evaluation

DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'

# Clamp -inf to this finite floor for NN inputs/outputs
LOG_FLOOR = -20.0

print(f"Device: {DEVICE}")

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# ── Tensor ops (analytical, for snapshot collection) ─────────────────────────

_NEG_INF = -np.inf
_LOG_HALF = np.log(0.5)
_LOG_QUARTER = np.log(0.25)


def _circ_conv_batch_np(A, B):
    """Circular convolution for (batch, L, 2, 2) arrays."""
    A00 = A[:, :, 0, 0]; A01 = A[:, :, 0, 1]
    A10 = A[:, :, 1, 0]; A11 = A[:, :, 1, 1]
    B00 = B[:, :, 0, 0]; B01 = B[:, :, 0, 1]
    B10 = B[:, :, 1, 0]; B11 = B[:, :, 1, 1]
    out = np.empty_like(A)
    out[:, :, 0, 0] = np.logaddexp(np.logaddexp(A00+B00, A01+B01),
                                    np.logaddexp(A10+B10, A11+B11))
    out[:, :, 0, 1] = np.logaddexp(np.logaddexp(A01+B00, A00+B01),
                                    np.logaddexp(A11+B10, A10+B11))
    out[:, :, 1, 0] = np.logaddexp(np.logaddexp(A10+B00, A11+B01),
                                    np.logaddexp(A00+B10, A01+B11))
    out[:, :, 1, 1] = np.logaddexp(np.logaddexp(A11+B00, A10+B01),
                                    np.logaddexp(A01+B10, A00+B11))
    return out


def _norm_prod_batch_np(A, B):
    """Normalized product for (batch, L, 2, 2) arrays."""
    raw = A + B
    total = np.logaddexp(
        np.logaddexp(raw[:, :, 0, 0], raw[:, :, 0, 1]),
        np.logaddexp(raw[:, :, 1, 0], raw[:, :, 1, 1])
    )
    finite = np.isfinite(total)
    result = raw.copy()
    result[finite] -= total[finite, None, None]
    return result


def _norm_prod_single_batch_np(A, B):
    """Normalized product for (batch, 2, 2) arrays."""
    raw = A + B
    total = np.logaddexp(
        np.logaddexp(raw[:, 0, 0], raw[:, 0, 1]),
        np.logaddexp(raw[:, 1, 0], raw[:, 1, 1])
    )
    finite = np.isfinite(total)
    result = raw.copy()
    result[finite] -= total[finite, None, None]
    return result


# ── Instrumented batched computational graph ─────────────────────────────────

class InstrumentedCompGraphBatched:
    """
    Batched computational graph that records all operation snapshots.

    Each snapshot is (op_type, input1, input2, output) where tensors are
    flattened to (batch, L, 4) for the batch of L-sized tensor arrays.
    """

    def __init__(self, n, log_W_batch):
        self.n = n
        self.N = 1 << n
        N_ = self.N
        self.batch = log_W_batch.shape[0]

        self.edge_data = [None] * (2 * N_)

        br = bit_reversal_perm(n)
        root = log_W_batch[:, br].copy()
        totals = np.logaddexp(
            np.logaddexp(root[:, :, 0, 0], root[:, :, 0, 1]),
            np.logaddexp(root[:, :, 1, 0], root[:, :, 1, 1])
        )
        finite = np.isfinite(totals)
        root[finite] -= totals[finite, None, None]
        self.edge_data[1] = root

        for beta in range(2, 2 * N_):
            level = beta.bit_length() - 1
            size = N_ >> level
            self.edge_data[beta] = np.full(
                (self.batch, size, 2, 2), _LOG_QUARTER, dtype=np.float64)

        self.dec_head = 1

        # Snapshot storage: list of (op_type, in1_flat, in2_flat, out_flat)
        # Each flat tensor: (batch * L, 4)
        self.snapshots_left = []   # CalcLeft snapshots
        self.snapshots_right = []  # CalcRight snapshots
        self.snapshots_parent = [] # CalcParent snapshots

    def calc_left(self, beta):
        """calcLeft with snapshot recording."""
        parent = self.edge_data[beta]
        right = self.edge_data[2 * beta + 1]
        l = right.shape[1]

        # Record inputs before computation
        in1 = parent[:, :l].copy()   # parent[:l] (batch, l, 2, 2)
        in2_raw = parent[:, l:].copy()  # parent[l:] (batch, l, 2, 2)
        in3 = right.copy()           # right (batch, l, 2, 2)

        # Compute: left = circ_conv(parent[:l], norm_prod(parent[l:], right))
        temp = _norm_prod_batch_np(parent[:, l:], right)
        result = _circ_conv_batch_np(parent[:, :l], temp)
        self.edge_data[2 * beta] = result

        # Record snapshot: inputs are (parent[:l], norm_prod(parent[l:], right))
        # But for the neural network, we want to learn the full CalcLeft op:
        # input1 = parent[:l], input2 = norm_prod(parent[l:], right), output = result
        # The norm_prod is an intermediate step. Let's record the actual
        # primitive operations:
        # Step 1: norm_prod(parent[l:], right) -> temp
        # Step 2: circ_conv(parent[:l], temp) -> result
        #
        # Actually, CalcLeft IS circ_conv(parent[:l], norm_prod(parent[l:], right))
        # The NN should learn the COMPOSITE operation from (parent_top, parent_bot, right) -> left
        # But with 3 inputs that's more complex. Instead, let's record both sub-ops:

        # Record as: CalcLeft takes (parent_top_half, parent_bot_half, right) -> output
        # We flatten each into (batch*l, 4)
        b = self.batch
        self.snapshots_left.append((
            in1.reshape(-1, 4).copy(),       # parent[:l]
            in2_raw.reshape(-1, 4).copy(),   # parent[l:]
            in3.reshape(-1, 4).copy(),       # right
            result.reshape(-1, 4).copy()     # output
        ))

    def calc_right(self, beta):
        """calcRight with snapshot recording."""
        parent = self.edge_data[beta]
        left = self.edge_data[2 * beta]
        l = left.shape[1]

        in1 = parent[:, :l].copy()   # parent[:l]
        in2 = parent[:, l:].copy()   # parent[l:]
        in3 = left.copy()            # left

        # Compute: right = norm_prod(parent[l:], circ_conv(left, parent[:l]))
        temp = _circ_conv_batch_np(left, parent[:, :l])
        result = _norm_prod_batch_np(parent[:, l:], temp)
        self.edge_data[2 * beta + 1] = result

        b = self.batch
        self.snapshots_right.append((
            in1.reshape(-1, 4).copy(),   # parent[:l]
            in2.reshape(-1, 4).copy(),   # parent[l:]
            in3.reshape(-1, 4).copy(),   # left
            result.reshape(-1, 4).copy() # output
        ))

    def calc_parent(self, beta):
        """calcParent with snapshot recording."""
        left = self.edge_data[2 * beta]
        right = self.edge_data[2 * beta + 1]
        l = left.shape[0] if left.ndim == 3 else left.shape[1]

        in1 = left.copy()
        in2 = right.copy()

        # Compute: parent[:l] = circ_conv(left, right), parent[l:] = right
        parent_left = _circ_conv_batch_np(left, right)
        self.edge_data[beta] = np.concatenate([parent_left, right], axis=1)

        # Record the circ_conv part only (the copy part is trivial)
        self.snapshots_parent.append((
            in1.reshape(-1, 4).copy(),
            in2.reshape(-1, 4).copy(),
            parent_left.reshape(-1, 4).copy()  # output of circ_conv
        ))

    def step_to(self, target):
        current = self.dec_head
        if current == target:
            return
        path = self._get_path(current, target)
        for beta in path:
            self._step_one(beta)
        self.dec_head = target

    def _step_one(self, beta):
        current = self.dec_head
        if current == beta >> 1:
            if beta & 1 == 0:
                self.calc_left(current)
            else:
                self.calc_right(current)
            self.dec_head = beta
        elif beta == current >> 1:
            self.calc_parent(current)
            self.dec_head = beta

    def _get_path(self, current, target):
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


def collect_snapshots(channel, N, b, frozen_u, frozen_v, batch_size, rng,
                      Au, Av):
    """
    Run the analytical SC decoder on a batch of random codewords,
    recording all operation snapshots.

    Returns: snapshots_left, snapshots_right, snapshots_parent, bler
    """
    n = N.bit_length() - 1

    # Generate random codewords
    u = np.zeros((batch_size, N), dtype=np.int32)
    v = np.zeros((batch_size, N), dtype=np.int32)
    for p in Au:
        u[:, p - 1] = rng.integers(0, 2, batch_size)
    for p in Av:
        v[:, p - 1] = rng.integers(0, 2, batch_size)
    x = polar_encode_batch(u)
    y = polar_encode_batch(v)
    Z_batch = channel.sample_batch(x, y)

    # Build leaf tensors
    log_W_batch = np.stack(
        [build_log_W_leaf(Z_batch[i], channel) for i in range(batch_size)])

    # Create instrumented graph
    graph = InstrumentedCompGraphBatched(n, log_W_batch)

    # Run SC decoder (same as decode_vectorized)
    u_hat = {}
    v_hat = {}
    i_u = 0
    i_v = 0

    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1
            i_t = i_u
            frozen_dict = frozen_u
        else:
            i_v += 1
            i_t = i_v
            frozen_dict = frozen_v

        leaf_edge = i_t + N - 1
        target_vertex = leaf_edge >> 1

        graph.step_to(target_vertex)

        temp = graph.edge_data[leaf_edge][:, 0].copy()

        if leaf_edge & 1 == 0:
            graph.calc_left(target_vertex)
        else:
            graph.calc_right(target_vertex)

        top_down = graph.edge_data[leaf_edge][:, 0]
        combined = _norm_prod_single_batch_np(top_down, temp)

        if i_t in frozen_dict:
            bit = np.full(batch_size, frozen_dict[i_t], dtype=np.int32)
        else:
            if gamma == 0:
                p0 = np.logaddexp(combined[:, 0, 0], combined[:, 0, 1])
                p1 = np.logaddexp(combined[:, 1, 0], combined[:, 1, 1])
            else:
                p0 = np.logaddexp(combined[:, 0, 0], combined[:, 1, 0])
                p1 = np.logaddexp(combined[:, 0, 1], combined[:, 1, 1])
            bit = (p1 > p0).astype(np.int32)

        if gamma == 0:
            u_hat[i_t] = bit
        else:
            v_hat[i_t] = bit

        new_leaf = np.full((batch_size, 2, 2), _NEG_INF, dtype=np.float64)
        u_val = u_hat.get(i_t)
        v_val = v_hat.get(i_t)

        if u_val is not None and v_val is not None:
            new_leaf[np.arange(batch_size), u_val, v_val] = 0.0
        elif u_val is not None:
            new_leaf[np.arange(batch_size), u_val, 0] = _LOG_HALF
            new_leaf[np.arange(batch_size), u_val, 1] = _LOG_HALF
        elif v_val is not None:
            new_leaf[np.arange(batch_size), 0, v_val] = _LOG_HALF
            new_leaf[np.arange(batch_size), 1, v_val] = _LOG_HALF
        else:
            new_leaf[:, :, :] = _LOG_QUARTER

        graph.edge_data[leaf_edge] = new_leaf[:, None, :, :]

    # Compute BLER
    u_dec = np.zeros((batch_size, N), dtype=np.int32)
    v_dec = np.zeros((batch_size, N), dtype=np.int32)
    for k in range(1, N + 1):
        if k in u_hat:
            u_dec[:, k - 1] = u_hat[k]
        if k in v_hat:
            v_dec[:, k - 1] = v_hat[k]

    errs = 0
    for i in range(batch_size):
        if (any(u_dec[i, p - 1] != u[i, p - 1] for p in Au) or
                any(v_dec[i, p - 1] != v[i, p - 1] for p in Av)):
            errs += 1

    return graph.snapshots_left, graph.snapshots_right, graph.snapshots_parent, errs


# ── Neural network modules ──────────────────────────────────────────────────

def _make_mlp(in_dim, hidden, out_dim, n_layers=3):
    layers = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class CalcLeftNN(nn.Module):
    """Neural CalcLeft: (parent_top, parent_bot, right) -> left_output.

    Each input is a flat 4-value log-prob tensor. The NN operates on
    embeddings of each tensor.
    """
    def __init__(self, d_emb=32, hidden=128, n_layers=3):
        super().__init__()
        self.tensor_to_emb = _make_mlp(4, hidden, d_emb, n_layers)
        self.combine = _make_mlp(3 * d_emb, hidden, d_emb, n_layers)
        self.emb_to_tensor = _make_mlp(d_emb, hidden, 4, n_layers)

    def forward(self, parent_top, parent_bot, right):
        """
        parent_top, parent_bot, right: (batch, 4) log-prob tensors
        Returns: (batch, 4) predicted output tensor
        """
        e1 = self.tensor_to_emb(parent_top)
        e2 = self.tensor_to_emb(parent_bot)
        e3 = self.tensor_to_emb(right)
        combined = torch.cat([e1, e2, e3], dim=-1)
        e_out = self.combine(combined)
        return self.emb_to_tensor(e_out)


class CalcRightNN(nn.Module):
    """Neural CalcRight: (parent_top, parent_bot, left) -> right_output."""
    def __init__(self, d_emb=32, hidden=128, n_layers=3):
        super().__init__()
        self.tensor_to_emb = _make_mlp(4, hidden, d_emb, n_layers)
        self.combine = _make_mlp(3 * d_emb, hidden, d_emb, n_layers)
        self.emb_to_tensor = _make_mlp(d_emb, hidden, 4, n_layers)

    def forward(self, parent_top, parent_bot, left):
        e1 = self.tensor_to_emb(parent_top)
        e2 = self.tensor_to_emb(parent_bot)
        e3 = self.tensor_to_emb(left)
        combined = torch.cat([e1, e2, e3], dim=-1)
        e_out = self.combine(combined)
        return self.emb_to_tensor(e_out)


class CalcParentNN(nn.Module):
    """Neural CalcParent: (left, right) -> parent_top (circ_conv part only)."""
    def __init__(self, d_emb=32, hidden=128, n_layers=3):
        super().__init__()
        self.tensor_to_emb = _make_mlp(4, hidden, d_emb, n_layers)
        self.combine = _make_mlp(2 * d_emb, hidden, d_emb, n_layers)
        self.emb_to_tensor = _make_mlp(d_emb, hidden, 4, n_layers)

    def forward(self, left, right):
        e1 = self.tensor_to_emb(left)
        e2 = self.tensor_to_emb(right)
        combined = torch.cat([e1, e2], dim=-1)
        e_out = self.combine(combined)
        return self.emb_to_tensor(e_out)


class SnapshotDecoder(nn.Module):
    """Container for all three operation NNs."""
    def __init__(self, d_emb=32, hidden=128, n_layers=3):
        super().__init__()
        self.calc_left_nn = CalcLeftNN(d_emb, hidden, n_layers)
        self.calc_right_nn = CalcRightNN(d_emb, hidden, n_layers)
        self.calc_parent_nn = CalcParentNN(d_emb, hidden, n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training ─────────────────────────────────────────────────────────────────

def _clamp_log(arr):
    """Clamp -inf values to LOG_FLOOR for NN consumption."""
    return np.clip(arr, LOG_FLOOR, 0.0)


def train_on_snapshots(model, snapshots_left, snapshots_right, snapshots_parent,
                       optimizer, device):
    """
    Train the model on a collection of operation snapshots.
    All log-prob values are clamped from -inf to LOG_FLOOR.
    Returns dict of losses.
    """
    model.train()
    losses = {}

    # Train CalcLeft
    if snapshots_left:
        all_in1 = _clamp_log(np.concatenate([s[0] for s in snapshots_left], axis=0))
        all_in2 = _clamp_log(np.concatenate([s[1] for s in snapshots_left], axis=0))
        all_in3 = _clamp_log(np.concatenate([s[2] for s in snapshots_left], axis=0))
        all_out = _clamp_log(np.concatenate([s[3] for s in snapshots_left], axis=0))

        in1_t = torch.from_numpy(all_in1).float().to(device)
        in2_t = torch.from_numpy(all_in2).float().to(device)
        in3_t = torch.from_numpy(all_in3).float().to(device)
        out_t = torch.from_numpy(all_out).float().to(device)

        # Mini-batch if too large
        n_total = in1_t.shape[0]
        idx = torch.randperm(n_total, device=device)[:BATCH_TRAIN]
        pred = model.calc_left_nn(in1_t[idx], in2_t[idx], in3_t[idx])
        loss_left = F.mse_loss(pred, out_t[idx])

        optimizer.zero_grad()
        loss_left.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses['left'] = loss_left.item()

    # Train CalcRight
    if snapshots_right:
        all_in1 = _clamp_log(np.concatenate([s[0] for s in snapshots_right], axis=0))
        all_in2 = _clamp_log(np.concatenate([s[1] for s in snapshots_right], axis=0))
        all_in3 = _clamp_log(np.concatenate([s[2] for s in snapshots_right], axis=0))
        all_out = _clamp_log(np.concatenate([s[3] for s in snapshots_right], axis=0))

        in1_t = torch.from_numpy(all_in1).float().to(device)
        in2_t = torch.from_numpy(all_in2).float().to(device)
        in3_t = torch.from_numpy(all_in3).float().to(device)
        out_t = torch.from_numpy(all_out).float().to(device)

        n_total = in1_t.shape[0]
        idx = torch.randperm(n_total, device=device)[:BATCH_TRAIN]
        pred = model.calc_right_nn(in1_t[idx], in2_t[idx], in3_t[idx])
        loss_right = F.mse_loss(pred, out_t[idx])

        optimizer.zero_grad()
        loss_right.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses['right'] = loss_right.item()

    # Train CalcParent
    if snapshots_parent:
        all_in1 = _clamp_log(np.concatenate([s[0] for s in snapshots_parent], axis=0))
        all_in2 = _clamp_log(np.concatenate([s[1] for s in snapshots_parent], axis=0))
        all_out = _clamp_log(np.concatenate([s[2] for s in snapshots_parent], axis=0))

        in1_t = torch.from_numpy(all_in1).float().to(device)
        in2_t = torch.from_numpy(all_in2).float().to(device)
        out_t = torch.from_numpy(all_out).float().to(device)

        n_total = in1_t.shape[0]
        idx = torch.randperm(n_total, device=device)[:BATCH_TRAIN]
        pred = model.calc_parent_nn(in1_t[idx], in2_t[idx])
        loss_parent = F.mse_loss(pred, out_t[idx])

        optimizer.zero_grad()
        loss_parent.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses['parent'] = loss_parent.item()

    return losses


# ── Neural SC Decoder (evaluation) ──────────────────────────────────────────

def neural_sc_decode(model, log_W_batch, b, frozen_u, frozen_v, device):
    """
    SC decode using neural CalcLeft/CalcRight/CalcParent instead of analytical.

    Parameters
    ----------
    model      : SnapshotDecoder
    log_W_batch: (batch, N, 2, 2) ndarray
    b          : path vector length 2N
    frozen_u   : dict {1-indexed: value}
    frozen_v   : dict {1-indexed: value}
    device     : torch device

    Returns
    -------
    u_dec, v_dec : (batch, N) ndarrays
    """
    model.eval()
    batch = log_W_batch.shape[0]
    N_ = log_W_batch.shape[1]
    n_ = N_.bit_length() - 1

    # Initialize edge data (same as analytical)
    edge_data = [None] * (2 * N_)

    br = bit_reversal_perm(n_)
    root = log_W_batch[:, br].copy()
    totals = np.logaddexp(
        np.logaddexp(root[:, :, 0, 0], root[:, :, 0, 1]),
        np.logaddexp(root[:, :, 1, 0], root[:, :, 1, 1])
    )
    finite = np.isfinite(totals)
    root[finite] -= totals[finite, None, None]
    edge_data[1] = root

    for beta in range(2, 2 * N_):
        level = beta.bit_length() - 1
        size = N_ >> level
        edge_data[beta] = np.full(
            (batch, size, 2, 2), _LOG_QUARTER, dtype=np.float64)

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

    def neural_calc_left(beta):
        nonlocal edge_data
        parent = edge_data[beta]
        right = edge_data[2 * beta + 1]
        l = right.shape[1]

        # Flatten each tensor for the NN
        in1 = parent[:, :l].reshape(-1, 4)  # parent_top
        in2 = parent[:, l:].reshape(-1, 4)  # parent_bot
        in3 = right.reshape(-1, 4)          # right

        with torch.no_grad():
            t1 = torch.from_numpy(in1).float().to(device)
            t2 = torch.from_numpy(in2).float().to(device)
            t3 = torch.from_numpy(in3).float().to(device)
            pred = model.calc_left_nn(t1, t2, t3)
            result = pred.cpu().numpy().astype(np.float64)

        edge_data[2 * beta] = result.reshape(batch, l, 2, 2)

    def neural_calc_right(beta):
        nonlocal edge_data
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]

        in1 = parent[:, :l].reshape(-1, 4)
        in2 = parent[:, l:].reshape(-1, 4)
        in3 = left.reshape(-1, 4)

        with torch.no_grad():
            t1 = torch.from_numpy(in1).float().to(device)
            t2 = torch.from_numpy(in2).float().to(device)
            t3 = torch.from_numpy(in3).float().to(device)
            pred = model.calc_right_nn(t1, t2, t3)
            result = pred.cpu().numpy().astype(np.float64)

        edge_data[2 * beta + 1] = result.reshape(batch, l, 2, 2)

    def neural_calc_parent(beta):
        nonlocal edge_data
        left = edge_data[2 * beta]
        right = edge_data[2 * beta + 1]
        l = left.shape[1]

        in1 = left.reshape(-1, 4)
        in2 = right.reshape(-1, 4)

        with torch.no_grad():
            t1 = torch.from_numpy(in1).float().to(device)
            t2 = torch.from_numpy(in2).float().to(device)
            pred = model.calc_parent_nn(t1, t2)
            parent_left = pred.cpu().numpy().astype(np.float64).reshape(batch, l, 2, 2)

        edge_data[beta] = np.concatenate([parent_left, right], axis=1)

    def step_one(beta):
        nonlocal dec_head
        if dec_head == beta >> 1:
            if beta & 1 == 0:
                neural_calc_left(dec_head)
            else:
                neural_calc_right(dec_head)
            dec_head = beta
        elif beta == dec_head >> 1:
            neural_calc_parent(dec_head)
            dec_head = beta

    def step_to(target):
        nonlocal dec_head
        if dec_head == target:
            return
        path = get_path(dec_head, target)
        for beta in path:
            step_one(beta)
        dec_head = target

    # SC decoding loop
    u_hat = {}
    v_hat = {}
    i_u = 0
    i_v = 0

    for step in range(2 * N_):
        gamma = b[step]
        if gamma == 0:
            i_u += 1
            i_t = i_u
            frozen_dict = frozen_u
        else:
            i_v += 1
            i_t = i_v
            frozen_dict = frozen_v

        leaf_edge = i_t + N_ - 1
        target_vertex = leaf_edge >> 1
        step_to(target_vertex)

        temp = edge_data[leaf_edge][:, 0].copy()

        if leaf_edge & 1 == 0:
            neural_calc_left(target_vertex)
        else:
            neural_calc_right(target_vertex)

        top_down = edge_data[leaf_edge][:, 0]
        combined = _norm_prod_single_batch_np(top_down, temp)

        if i_t in frozen_dict:
            bit = np.full(batch, frozen_dict[i_t], dtype=np.int32)
        else:
            if gamma == 0:
                p0 = np.logaddexp(combined[:, 0, 0], combined[:, 0, 1])
                p1 = np.logaddexp(combined[:, 1, 0], combined[:, 1, 1])
            else:
                p0 = np.logaddexp(combined[:, 0, 0], combined[:, 1, 0])
                p1 = np.logaddexp(combined[:, 0, 1], combined[:, 1, 1])
            bit = (p1 > p0).astype(np.int32)

        if gamma == 0:
            u_hat[i_t] = bit
        else:
            v_hat[i_t] = bit

        new_leaf = np.full((batch, 2, 2), _NEG_INF, dtype=np.float64)
        u_val = u_hat.get(i_t)
        v_val = v_hat.get(i_t)

        if u_val is not None and v_val is not None:
            new_leaf[np.arange(batch), u_val, v_val] = 0.0
        elif u_val is not None:
            new_leaf[np.arange(batch), u_val, 0] = _LOG_HALF
            new_leaf[np.arange(batch), u_val, 1] = _LOG_HALF
        elif v_val is not None:
            new_leaf[np.arange(batch), 0, v_val] = _LOG_HALF
            new_leaf[np.arange(batch), 1, v_val] = _LOG_HALF
        else:
            new_leaf[:, :, :] = _LOG_QUARTER

        edge_data[leaf_edge] = new_leaf[:, None, :, :]

    u_dec = np.zeros((batch, N_), dtype=np.int32)
    v_dec = np.zeros((batch, N_), dtype=np.int32)
    for k in range(1, N_ + 1):
        if k in u_hat:
            u_dec[:, k - 1] = u_hat[k]
        if k in v_hat:
            v_dec[:, k - 1] = v_hat[k]

    return u_dec, v_dec


def evaluate_bler(model, channel, N_, b, Au, Av, frozen_u, frozen_v,
                  n_cw, batch_size, device, use_neural=True):
    """Evaluate BLER using either neural or analytical decoder."""
    if model is not None:
        model.eval()
    errs = 0
    total = 0
    rng = np.random.default_rng(42)

    while total < n_cw:
        actual = min(batch_size, n_cw - total)

        u = np.zeros((actual, N_), dtype=np.int32)
        v = np.zeros((actual, N_), dtype=np.int32)
        for p in Au:
            u[:, p - 1] = rng.integers(0, 2, actual)
        for p in Av:
            v[:, p - 1] = rng.integers(0, 2, actual)
        x = polar_encode_batch(u)
        y = polar_encode_batch(v)
        Z_batch = channel.sample_batch(x, y)

        log_W_batch = np.stack(
            [build_log_W_leaf(Z_batch[i], channel) for i in range(actual)])

        if use_neural:
            u_dec, v_dec = neural_sc_decode(
                model, log_W_batch, b, frozen_u, frozen_v, device)
        else:
            # Analytical decoder
            from polar.decoder_interleaved import decode_vectorized
            u_dec, v_dec = decode_vectorized(
                N_, Z_batch, b, frozen_u, frozen_v, channel)

        for i in range(actual):
            if (any(u_dec[i, p - 1] != u[i, p - 1] for p in Au) or
                    any(v_dec[i, p - 1] != v[i, p - 1] for p in Av)):
                errs += 1
        total += actual

    return errs / max(total, 1)


# ── Load design ──────────────────────────────────────────────────────────────

def load_design(N_, ku, kv):
    """Load GMAC Class B design from npz file."""
    n_ = int(math.log2(N_))
    dp = os.path.join(DESIGNS_DIR, f'gmac_B_n{n_}_snr{SNR_DB:.0f}dB.npz')
    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i + 1) for i in su[:ku]])
    Av = sorted([int(i + 1) for i in sv[:kv]])
    all_pos = set(range(1, N_ + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SC Teacher Snapshots — Parallel Operation-Level Distillation POC")
    print("=" * 70)
    print(f"N={N}, Class B (path_i={PATH_I}), SNR={SNR_DB}dB, sigma2={SIGMA2:.6f}")
    print(f"d_emb={D_EMB}, hidden={HIDDEN}, n_layers={N_LAYERS}")
    print(f"batch_snap={BATCH_SNAP}, batch_train={BATCH_TRAIN}, lr={LR}")
    print(f"Train time target: {TRAIN_HOURS}h")
    print()

    # Channel
    channel = GaussianMAC(sigma2=SIGMA2)

    # Design: pick rates that give ~2-5% SC BLER for a meaningful comparison
    ku, kv = 64, 64
    Au, Av, frozen_u, frozen_v = load_design(N, ku, kv)
    b = make_path(N, PATH_I)

    print(f"Code design: ku={ku} (R_u={ku/N:.3f}), kv={kv} (R_v={kv/N:.3f})")
    print(f"Total rate: {(ku+kv)/N:.3f}")
    print(f"|Au|={len(Au)}, |Av|={len(Av)}")
    print(f"|frozen_u|={len(frozen_u)}, |frozen_v|={len(frozen_v)}")
    print()

    # Evaluate analytical SC BLER first (smaller count for speed)
    sc_eval_cw = 1000
    print("Evaluating analytical SC decoder BLER...")
    sc_bler = evaluate_bler(None, channel, N, b, Au, Av, frozen_u, frozen_v,
                            sc_eval_cw, 200, DEVICE, use_neural=False)
    print(f"Analytical SC BLER: {sc_bler:.4f} ({int(sc_bler * sc_eval_cw)}/{sc_eval_cw})")
    print()

    # Create model
    model = SnapshotDecoder(d_emb=D_EMB, hidden=HIDDEN, n_layers=N_LAYERS).to(DEVICE)
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=2, eta_min=LR * 0.01)

    # Training loop
    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    rng = np.random.default_rng()
    t_start = time.time()
    t_end = t_start + TRAIN_HOURS * 3600

    iteration = 0
    best_loss = float('inf')
    best_state = None
    running_losses = {'left': [], 'right': [], 'parent': []}

    while time.time() < t_end:
        iteration += 1

        # Collect fresh snapshots
        snap_left, snap_right, snap_parent, snap_errs = collect_snapshots(
            channel, N, b, frozen_u, frozen_v, BATCH_SNAP, rng, Au, Av)

        # Train on these snapshots (multiple passes per snapshot collection)
        n_passes = 10
        for _ in range(n_passes):
            losses = train_on_snapshots(
                model, snap_left, snap_right, snap_parent, optimizer, DEVICE)
            scheduler.step()

        # Track losses
        for k in ['left', 'right', 'parent']:
            if k in losses:
                running_losses[k].append(losses[k])

        # Logging
        if iteration % 10 == 1 or iteration <= 5:
            elapsed = time.time() - t_start
            remaining = t_end - time.time()
            avg_losses = {}
            for k in ['left', 'right', 'parent']:
                recent = running_losses[k][-20:]
                if recent:
                    avg_losses[k] = np.mean(recent)
            loss_str = "  ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
            print(f"[{elapsed/60:.1f}m / {remaining/60:.0f}m left]  "
                  f"iter={iteration}  {loss_str}  "
                  f"snap_bler={snap_errs/BATCH_SNAP:.3f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

            # Track best
            total_loss = sum(avg_losses.values())
            if total_loss < best_loss:
                best_loss = total_loss
                best_state = {k: v.clone().cpu()
                              for k, v in model.state_dict().items()}

        # Periodic evaluation
        if iteration % 100 == 0:
            elapsed = time.time() - t_start
            print(f"\n--- Eval at iter {iteration} ({elapsed/60:.1f}m) ---")
            nn_bler = evaluate_bler(
                model, channel, N, b, Au, Av, frozen_u, frozen_v,
                min(EVAL_CW, 2000), 32, DEVICE, use_neural=True)
            print(f"Neural SC BLER: {nn_bler:.4f}  (SC={sc_bler:.4f})")
            print()
            model.train()

    elapsed_total = time.time() - t_start
    print(f"\nTraining complete: {iteration} iterations in {elapsed_total/3600:.2f}h")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
        print(f"Loaded best model (loss={best_loss:.4f})")

    # Final evaluation
    print()
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    model.eval()
    nn_bler = evaluate_bler(
        model, channel, N, b, Au, Av, frozen_u, frozen_v,
        EVAL_CW, 32, DEVICE, use_neural=True)

    print(f"\nAnalytical SC BLER: {sc_bler:.4f}")
    print(f"Neural SC BLER:    {nn_bler:.4f}")
    if sc_bler > 0:
        ratio = nn_bler / sc_bler
        print(f"Ratio (neural/SC): {ratio:.2f}")
    print(f"\nTotal training time: {elapsed_total/3600:.2f}h")
    print(f"Model parameters: {n_params:,}")

    # Save model
    save_path = os.path.join(os.path.dirname(__file__),
                              'saved_models', 'poc_sc_snapshots.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'N': N, 'n': n, 'snr_db': SNR_DB, 'sigma2': SIGMA2,
            'path_i': PATH_I, 'ku': ku, 'kv': kv,
            'd_emb': D_EMB, 'hidden': HIDDEN, 'n_layers': N_LAYERS,
        },
        'sc_bler': sc_bler,
        'nn_bler': nn_bler,
        'n_params': n_params,
        'train_time_h': elapsed_total / 3600,
        'n_iters': iteration,
    }, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == '__main__':
    main()
