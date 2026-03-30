#!/usr/bin/env python3
"""
poc_snapshot_finetune_n256.py — Snapshot fine-tuning at N=256.

Start from the best freeze-extend model (BLER=0.034), then fine-tune ALL
level-specific MLPs using SC snapshot supervision:
  - Run analytical SC decoder, record true 2x2 tensors at every edge
  - For each CalcLeft/CalcRight call: feed TRUE input tensors through the NN,
    compare output against TRUE output tensor
  - Gradient depth = 1 per operation, fully parallel within one codeword
  - Loss = MSE between NN output embedding and true output embedding

Key difference from failed POC 3: we START from a working model (0.034 BLER),
not from scratch. The snapshot training should polish existing operations.
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import build_log_W_leaf
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder, _make_mlp, NeuralCalcParent

D = 16
HIDDEN = 64
N_LAYERS = 2
Z_HIDDEN = 32

N_VAL = 256
KU = 123
KV = 123
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
SC_BLER = 0.005

BATCH_SNAP = 32   # codewords per snapshot batch
LR = 5e-5
TOTAL_ITERS = 2000
EVAL_EVERY = 200
EVAL_CW = 1000

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')


# ─── Analytical SC operations (for computing true tensors) ─────────────────

def _circ_conv(A, B):
    """2x2 circular convolution in log domain."""
    out = np.empty_like(A)
    out[..., 0, 0] = np.logaddexp(
        np.logaddexp(A[..., 0, 0] + B[..., 0, 0], A[..., 0, 1] + B[..., 0, 1]),
        np.logaddexp(A[..., 1, 0] + B[..., 1, 0], A[..., 1, 1] + B[..., 1, 1]))
    out[..., 0, 1] = np.logaddexp(
        np.logaddexp(A[..., 0, 1] + B[..., 0, 0], A[..., 0, 0] + B[..., 0, 1]),
        np.logaddexp(A[..., 1, 1] + B[..., 1, 0], A[..., 1, 0] + B[..., 1, 1]))
    out[..., 1, 0] = np.logaddexp(
        np.logaddexp(A[..., 1, 0] + B[..., 0, 0], A[..., 1, 1] + B[..., 0, 1]),
        np.logaddexp(A[..., 0, 0] + B[..., 1, 0], A[..., 0, 1] + B[..., 1, 1]))
    out[..., 1, 1] = np.logaddexp(
        np.logaddexp(A[..., 1, 1] + B[..., 0, 0], A[..., 1, 0] + B[..., 0, 1]),
        np.logaddexp(A[..., 0, 1] + B[..., 1, 0], A[..., 0, 0] + B[..., 1, 1]))
    return out


def _norm_prod(A, B):
    """Normalized element-wise product in log domain."""
    C = A + B
    log_sum = np.logaddexp(np.logaddexp(C[..., 0, 0], C[..., 0, 1]),
                           np.logaddexp(C[..., 1, 0], C[..., 1, 1]))
    return C - log_sum[..., None, None]


# ─── Snapshot collector ────────────────────────────────────────────────────

def collect_snapshots(channel, N, b, frozen_u, frozen_v, batch_size, rng):
    """
    Run analytical SC decoder with genie, record all edge tensors.
    Returns list of (op_type, input1_tensor, input2_tensor, output_tensor, level).
    Tensors are (batch, 2, 2) log-probability arrays.
    """
    n = int(np.log2(N))

    # Generate data
    u = np.zeros((batch_size, N), dtype=int)
    v = np.zeros((batch_size, N), dtype=int)
    Au = [p for p in range(1, N + 1) if p not in frozen_u]
    Av = [p for p in range(1, N + 1) if p not in frozen_v]
    for p in Au:
        u[:, p - 1] = rng.integers(0, 2, batch_size)
    for p in Av:
        v[:, p - 1] = rng.integers(0, 2, batch_size)

    x = polar_encode_batch(u)
    y = polar_encode_batch(v)
    Z = channel.sample_batch(x, y)

    # Build leaf tensors
    log_W = np.stack([build_log_W_leaf(Z[i], channel) for i in range(batch_size)])
    # log_W shape: (batch, N, 2, 2)

    # Bit-reverse
    br = bit_reversal_perm(n)
    log_W_br = log_W[:, br]

    # Initialize edge data
    edge_data = [None] * (2 * N)
    edge_data[1] = log_W_br  # root: (batch, N, 2, 2)

    NEG_INF = -1e30
    LOG_QUARTER = np.log(0.25)

    for beta in range(2, 2 * N):
        level = beta.bit_length() - 1
        size = N >> level
        edge_data[beta] = np.full((batch_size, size, 2, 2), LOG_QUARTER)

    snapshots = []
    dec_head = 1
    i_u, i_v = 0, 0

    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u; fdict = frozen_u
        else:
            i_v += 1; i_t = i_v; fdict = frozen_v

        leaf_edge = i_t + N - 1
        target_vtx = leaf_edge >> 1

        # Navigate to target
        path = []
        c, t = dec_head, target_vtx
        while c != t:
            if c > t:
                c >>= 1
                path.append(('up', c))
            else:
                path.append(('down', t))
                t >>= 1
        path_down = [p for p in path if p[0] == 'down']
        path_down.reverse()
        full_path = [p for p in path if p[0] == 'up'] + path_down

        for direction, beta in full_path:
            if direction == 'down':
                parent = beta >> 1
                if beta & 1 == 0:  # CalcLeft
                    p_edge = edge_data[parent]
                    r_edge = edge_data[2 * parent + 1]
                    l = r_edge.shape[1]
                    result = _circ_conv(p_edge[:, :l], r_edge)
                    snapshots.append(('left', p_edge[:, :l].copy(), r_edge.copy(),
                                     result.copy(), parent.bit_length() - 1))
                    edge_data[2 * parent] = result
                else:  # CalcRight
                    p_edge = edge_data[parent]
                    l_edge = edge_data[2 * parent]
                    l = l_edge.shape[1]
                    result = _norm_prod(p_edge[:, :l], l_edge)
                    snapshots.append(('right', p_edge[:, :l].copy(), l_edge.copy(),
                                     result.copy(), parent.bit_length() - 1))
                    edge_data[2 * parent + 1] = result
                dec_head = beta
            else:  # up = CalcParent
                current = dec_head
                l_edge = edge_data[2 * current]
                r_edge = edge_data[2 * current + 1]
                result_first = _circ_conv(l_edge, r_edge)
                # parent = first_half concat second_half, but for simplicity record first half
                snapshots.append(('parent', l_edge.copy(), r_edge.copy(),
                                 result_first.copy(), current.bit_length() - 1))
                # Reconstruct parent edge
                p_second = r_edge  # second half = right child
                edge_data[current] = np.concatenate([result_first, p_second], axis=1)
                dec_head = beta

        # Leaf operation
        temp = edge_data[leaf_edge][:, 0].copy()
        parent = target_vtx
        if leaf_edge & 1 == 0:
            p_edge = edge_data[parent]
            r_edge = edge_data[2 * parent + 1]
            result = _circ_conv(p_edge[:, :1], r_edge[:, :1] if r_edge.shape[1] >= 1 else r_edge)
            edge_data[2 * parent] = result
        else:
            p_edge = edge_data[parent]
            l_edge = edge_data[2 * parent]
            result = _norm_prod(p_edge[:, :1], l_edge[:, :1] if l_edge.shape[1] >= 1 else l_edge)
            edge_data[2 * parent + 1] = result

        # Decision
        top_down = edge_data[leaf_edge][:, 0]
        combined = _norm_prod(
            top_down.reshape(-1, 1, 2, 2),
            temp.reshape(-1, 1, 2, 2)
        ).reshape(-1, 2, 2)

        if i_t in fdict:
            u_bit = np.zeros(batch_size, dtype=int)
            v_bit = np.zeros(batch_size, dtype=int)
        else:
            if gamma == 0:
                u_bit = u[:, i_t - 1]
                v_bit = v[:, i_t - 1] if i_t in {p for p in range(1, N+1) if p not in frozen_v} else np.zeros(batch_size, dtype=int)
            else:
                u_bit = u[:, i_t - 1] if i_t in {p for p in range(1, N+1) if p not in frozen_u} else np.zeros(batch_size, dtype=int)
                v_bit = v[:, i_t - 1]
            # Use true bits (genie)
            u_bit = u[:, i_t - 1]
            v_bit = v[:, i_t - 1]

        # Set leaf to partially deterministic
        leaf_tensor = np.full((batch_size, 1, 2, 2), np.log(0.25))
        for b_idx in range(batch_size):
            if i_t in frozen_u and i_t in frozen_v:
                leaf_tensor[b_idx, 0, 0, 0] = 0.0
                leaf_tensor[b_idx, 0, 0, 1] = -1e30
                leaf_tensor[b_idx, 0, 1, 0] = -1e30
                leaf_tensor[b_idx, 0, 1, 1] = -1e30
            elif i_t in frozen_u:
                leaf_tensor[b_idx, 0, 0, v_bit[b_idx]] = 0.0
                leaf_tensor[b_idx, 0, 0, 1 - v_bit[b_idx]] = -1e30
                leaf_tensor[b_idx, 0, 1, 0] = -1e30
                leaf_tensor[b_idx, 0, 1, 1] = -1e30
            elif i_t in frozen_v:
                leaf_tensor[b_idx, 0, u_bit[b_idx], 0] = 0.0
                leaf_tensor[b_idx, 0, 1 - u_bit[b_idx], 0] = -1e30
                leaf_tensor[b_idx, 0, 0, 1] = -1e30
                leaf_tensor[b_idx, 0, 1, 1] = -1e30
            else:
                leaf_tensor[b_idx, 0, u_bit[b_idx], v_bit[b_idx]] = 0.0
                leaf_tensor[b_idx, 0, 1 - u_bit[b_idx], :] = -1e30
                leaf_tensor[b_idx, 0, :, 1 - v_bit[b_idx]] = -1e30
        edge_data[leaf_edge] = leaf_tensor

    return snapshots, Z


# ─── Model (reuse freeze-extend structure but train all levels) ────────────

class SnapshotFinetuneModel(nn.Module):
    """Model with per-level CalcLeft/CalcRight, trained with snapshot supervision."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_hidden=Z_HIDDEN, max_levels=8):
        super().__init__()
        self.d = d

        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

        # Tensor <-> Embedding conversion
        self.tensor_to_emb = nn.Sequential(
            nn.Linear(4, hidden), nn.ELU(), nn.Linear(hidden, d),
        )
        self.emb_to_tensor = nn.Sequential(
            nn.Linear(d, hidden), nn.ELU(), nn.Linear(hidden, 4),
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def snapshot_loss(self, snapshots):
        """
        Compute loss from snapshot triplets.
        Each snapshot: (op_type, input1_tensor, input2_tensor, output_tensor, level)
        Tensors are numpy (batch, M, 2, 2).

        All tensor values are clamped to [-30, 0] to avoid NaN from -1e30 values.
        Only uses snapshots where inputs are in reasonable range (skip delta-function edges).
        """
        total_loss = 0.0
        count = 0
        CLAMP_MIN = -30.0

        for op_type, inp1, inp2, target, level in snapshots:
            # Don't skip — clamp everything to [-30, 0] instead

            B = inp1.shape[0]
            M = inp1.shape[1]

            # Clamp all tensors to [-30, 0]
            inp1_c = np.clip(inp1, CLAMP_MIN, 0.0)
            inp2_c = np.clip(inp2, CLAMP_MIN, 0.0)
            target_c = np.clip(target, CLAMP_MIN, 0.0)

            inp1_flat = torch.from_numpy(inp1_c.reshape(B * M, 4)).float()
            inp2_flat = torch.from_numpy(inp2_c.reshape(B * M, 4)).float()
            target_flat = torch.from_numpy(target_c.reshape(B * M, 4)).float()

            emb1 = self.tensor_to_emb(inp1_flat)
            emb2 = self.tensor_to_emb(inp2_flat)

            if op_type == 'left':
                inp = torch.cat([emb1, emb1, emb2], dim=-1)
                out_emb = self.tree.calc_left_nn(inp)
            elif op_type == 'right':
                inp = torch.cat([emb1, emb1, emb2], dim=-1)
                out_emb = self.tree.calc_right_nn(inp)
            else:  # parent
                out_emb = self.tree.calc_parent_nn(emb1.unsqueeze(1), emb2.unsqueeze(1)).squeeze(1)

            pred_tensor = self.emb_to_tensor(out_emb)
            loss = F.mse_loss(pred_tensor, target_flat)
            total_loss += loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, requires_grad=True)
        return total_loss / count


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    return design_from_file(mc_path, n, ku, kv)


def evaluate_sequential(model, channel, N, b, Au, Av, fu, fv, n_cw):
    """Evaluate using standard sequential tree walk with the NN."""
    model.eval()
    n = int(math.log2(N))
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    errs = 0; total = 0
    rng = np.random.default_rng(999)

    with torch.no_grad():
        for _ in range(n_cw):
            uf = np.zeros((1, N), dtype=int); vf = np.zeros((1, N), dtype=int)
            for p in Au: uf[0, p-1] = rng.integers(0, 2)
            for p in Av: vf[0, p-1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            # Use the tree model's forward for sequential decode
            root = model.z_encoder(zf.unsqueeze(-1))[:, br]
            logits_list, targets_list, uh, vh, _ = model.tree(
                z=None, b=b, frozen_u=fu, frozen_v=fv, root_emb=root)

            ue = any(int(uh[p][0].item()) != uf[0, p-1] for p in Au if p in uh)
            ve = any(int(vh[p][0].item()) != vf[0, p-1] for p in Av if p in vh)
            if ue or ve: errs += 1
            total += 1
    model.train()
    return errs / total


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N_VAL, KU, KV)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    Au_list = list(Au)
    Av_list = list(Av)
    b = make_path(N_VAL, N_VAL // 2)

    model = SnapshotFinetuneModel(d=D, hidden=HIDDEN, n_layers=N_LAYERS, max_levels=8)

    # Load best checkpoint
    ckpt_path = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N256.pt')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        own = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        model.load_state_dict(own, strict=False)
        print(f'Loaded {loaded} params from {ckpt_path}', flush=True)

    print(f'Params: {model.count_parameters():,}', flush=True)

    # Initial eval
    init_bler = evaluate_sequential(model, channel, N_VAL, b, Au, Av, fu, fv, EVAL_CW)
    print(f'Initial BLER: {init_bler:.4f} (SC={SC_BLER})', flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(42)
    t0 = time.time()
    best_bler = init_bler

    print(f'Training with snapshot supervision, {TOTAL_ITERS} iters', flush=True)

    for it in range(1, TOTAL_ITERS + 1):
        # Collect fresh snapshots
        snapshots, Z = collect_snapshots(channel, N_VAL, b, fu, fv, BATCH_SNAP, rng)

        # Train on snapshots
        loss = model.snapshot_loss(snapshots)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            bler = evaluate_sequential(model, channel, N_VAL, b, Au, Av, fu, fv, EVAL_CW)
            improved = ''
            if bler < best_bler:
                best_bler = bler
                improved = ' *BEST*'
            print(f'[{it:>4}/{TOTAL_ITERS}] snap_loss={loss.item():.4f} '
                  f'BLER={bler:.4f} (best={best_bler:.4f}, SC={SC_BLER}) '
                  f'{elapsed/60:.0f}min{improved}', flush=True)

    print(f'\nDONE: best={best_bler:.4f} (SC={SC_BLER})', flush=True)


if __name__ == '__main__':
    main()
