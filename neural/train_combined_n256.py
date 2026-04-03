#!/usr/bin/env python3
"""
train_combined_n256.py — Combined end-to-end + snapshot auxiliary loss at N=256.

For each training codeword:
1. Run NN decoder sequentially (standard CE loss on leaf decisions)
2. In parallel, run analytical SC decoder to get TRUE tensors at every edge
3. At each CalcLeft/CalcRight, add MSE loss between NN embedding output and
   the embedding of the TRUE analytical tensor (via logits2emb)

Total loss = CE_loss + lambda * snapshot_MSE_loss

The CE keeps the model functional. The MSE pushes each operation closer to
the analytical one. Together they should improve per-node accuracy without
destroying overall decoding ability.
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
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder, _make_mlp

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

BATCH = 4        # Small batch due to running analytical decoder in parallel
LR = 5e-5
TOTAL_ITERS = 50000
EVAL_EVERY = 2000
EVAL_CW = 1000
LAMBDA_SNAP = 0.1  # Weight for snapshot auxiliary loss

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'train_combined_n256.log')

# Analytical SC operations
def _circ_conv_single(A, B):
    """2x2 circular convolution in log domain for single tensors."""
    out = np.empty((2, 2), dtype=np.float64)
    out[0, 0] = np.logaddexp(np.logaddexp(A[0,0]+B[0,0], A[0,1]+B[0,1]),
                              np.logaddexp(A[1,0]+B[1,0], A[1,1]+B[1,1]))
    out[0, 1] = np.logaddexp(np.logaddexp(A[0,1]+B[0,0], A[0,0]+B[0,1]),
                              np.logaddexp(A[1,1]+B[1,0], A[1,0]+B[1,1]))
    out[1, 0] = np.logaddexp(np.logaddexp(A[1,0]+B[0,0], A[1,1]+B[0,1]),
                              np.logaddexp(A[0,0]+B[1,0], A[0,1]+B[1,1]))
    out[1, 1] = np.logaddexp(np.logaddexp(A[1,1]+B[0,0], A[1,0]+B[0,1]),
                              np.logaddexp(A[0,1]+B[1,0], A[0,0]+B[1,1]))
    return out

def _norm_prod_single(A, B):
    C = A + B
    s = np.logaddexp(np.logaddexp(C[0,0], C[0,1]), np.logaddexp(C[1,0], C[1,1]))
    return C - s


class CombinedTrainer:
    """
    Runs NN and analytical decoders in parallel, computing combined loss.
    """
    def __init__(self, model, channel, N, b, frozen_u, frozen_v):
        self.model = model
        self.channel = channel
        self.N = N
        self.n = int(math.log2(N))
        self.b = b
        self.frozen_u = frozen_u
        self.frozen_v = frozen_v
        self.br = bit_reversal_perm(self.n)

    def train_step(self, z_np, u_true_np, v_true_np):
        """
        Single training step with combined loss.
        z_np: (batch, N) numpy float
        u_true_np, v_true_np: (batch, N) numpy int (message bits)
        Returns: total_loss (torch scalar)
        """
        model = self.model
        N = self.N; n = self.n; d = model.d
        B = z_np.shape[0]
        br = self.br
        b = self.b

        z = torch.from_numpy(z_np).float()
        u_true = torch.from_numpy(u_true_np).float()
        v_true = torch.from_numpy(v_true_np).float()

        # ── NN: init embeddings ──
        br_t = torch.from_numpy(br).long()
        root = model.z_encoder(z.unsqueeze(-1))[:, br_t]  # (B, N, d)

        nn_edge = [None] * (2 * N)
        nn_edge[1] = root
        no_info = model.tree.no_info_emb.unsqueeze(0).unsqueeze(0)
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            nn_edge[beta] = no_info.expand(B, size, d).clone()

        # ── Analytical: init tensors ──
        # Build per-sample log_W leaf tensors
        anal_edge = [None] * (2 * N)
        log_W_batch = np.stack([build_log_W_leaf(z_np[i], self.channel) for i in range(B)])
        # (B, N, 2, 2), bit-reverse
        anal_edge[1] = log_W_batch[:, br]  # (B, N, 2, 2)

        LOG_QUARTER = np.log(0.25)
        _NEG_INF = -1e30
        _LOG_HALF = np.log(0.5)

        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            anal_edge[beta] = np.full((B, size, 2, 2), LOG_QUARTER)

        # ── Parallel tree walk ──
        nn_head = 1
        anal_head = 1
        u_hat, v_hat = {}, {}
        all_logits, all_targets = [], []
        snap_losses = []
        i_u, i_v = 0, 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = self.frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = self.frozen_v

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1

            # Navigate both decoders
            nn_head = self._nn_step_to(nn_head, target_vtx, nn_edge)
            self._anal_step_to(anal_head, target_vtx, anal_edge, u_hat, v_hat)
            anal_head = target_vtx

            # Save temps
            nn_temp = nn_edge[leaf_edge][:, 0].clone()
            anal_temp = anal_edge[leaf_edge][:, 0].copy()

            # CalcLeft or CalcRight at leaf level + snapshot loss
            if leaf_edge & 1 == 0:
                # NN CalcLeft
                parent = nn_edge[target_vtx]
                right = nn_edge[2 * target_vtx + 1]
                l = right.shape[1]
                inp = torch.cat([parent[:, :l], parent[:, l:], right], dim=-1)
                nn_edge[2 * target_vtx] = model.tree.calc_left_nn(inp)

                # Analytical CalcLeft
                for bi in range(B):
                    p = anal_edge[target_vtx][bi]
                    r = anal_edge[2 * target_vtx + 1][bi]
                    for j in range(r.shape[0]):
                        anal_edge[2 * target_vtx][bi, j] = _circ_conv_single(p[j], r[j])

                # Snapshot loss: compare NN output to analytical
                anal_out = anal_edge[2 * target_vtx]  # (B, l, 2, 2)
                snap_loss = self._compute_snap_loss(nn_edge[2 * target_vtx], anal_out)
                if snap_loss is not None:
                    snap_losses.append(snap_loss)
            else:
                # NN CalcRight
                parent = nn_edge[target_vtx]
                left = nn_edge[2 * target_vtx]
                l = left.shape[1]
                inp = torch.cat([parent[:, :l], parent[:, l:], left], dim=-1)
                nn_edge[2 * target_vtx + 1] = model.tree.calc_right_nn(inp)

                # Analytical CalcRight
                for bi in range(B):
                    p = anal_edge[target_vtx][bi]
                    le = anal_edge[2 * target_vtx][bi]
                    for j in range(le.shape[0]):
                        anal_edge[2 * target_vtx + 1][bi, j] = _norm_prod_single(p[j], le[j])

                anal_out = anal_edge[2 * target_vtx + 1]
                snap_loss = self._compute_snap_loss(nn_edge[2 * target_vtx + 1], anal_out)
                if snap_loss is not None:
                    snap_losses.append(snap_loss)

            # NN: get logits
            top_down = nn_edge[leaf_edge][:, 0]
            combined = top_down + nn_temp
            logits = model.tree.emb2logits(combined)

            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.float32)
            else:
                all_logits.append(logits)
                target = (u_true[:, i_t - 1] * 2 + v_true[:, i_t - 1]).long()
                all_targets.append(target)
                bit = u_true[:, i_t - 1] if gamma == 0 else v_true[:, i_t - 1]

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            # Set leaf embeddings (NN)
            new_emb = model._make_leaf_emb(u_hat.get(i_t), v_hat.get(i_t), B)
            nn_edge[leaf_edge] = new_emb.unsqueeze(1)

            # Set leaf tensors (analytical)
            for bi in range(B):
                new_leaf = np.full((2, 2), _NEG_INF)
                u_val = int(u_true_np[bi, i_t - 1]) if gamma == 0 else (int(u_hat[i_t][bi].item()) if i_t in u_hat else 0)
                v_val = int(v_true_np[bi, i_t - 1]) if gamma == 1 else (int(v_hat[i_t][bi].item()) if i_t in v_hat else 0)
                if i_t in self.frozen_u:
                    u_val = 0
                if i_t in self.frozen_v:
                    v_val = 0
                # Use true bits (genie)
                u_val = int(u_true_np[bi, i_t - 1])
                v_val = int(v_true_np[bi, i_t - 1])
                new_leaf[u_val, v_val] = 0.0
                anal_edge[leaf_edge][bi, 0] = new_leaf

        # Compute combined loss
        ce_loss = torch.tensor(0.0)
        if all_logits:
            ce_loss = F.cross_entropy(torch.stack(all_logits).reshape(-1, 4),
                                       torch.stack(all_targets).reshape(-1))

        snap_loss = torch.tensor(0.0)
        if snap_losses:
            snap_loss = torch.stack(snap_losses).mean()

        total = ce_loss + LAMBDA_SNAP * snap_loss
        return total, ce_loss.item(), snap_loss.item()

    def _compute_snap_loss(self, nn_emb, anal_tensor):
        """
        Compare NN embedding to analytical tensor.
        nn_emb: (B, M, d) torch tensor
        anal_tensor: (B, M, 2, 2) numpy array

        Convert analytical tensor to embedding via model's logits2emb,
        then MSE between the two embeddings.
        """
        B, M = nn_emb.shape[0], nn_emb.shape[1]

        # Clamp analytical values
        anal_clamped = np.clip(anal_tensor, -30.0, 0.0)
        anal_flat = torch.from_numpy(anal_clamped.reshape(B * M, 4)).float()

        # Convert to embedding space
        with torch.no_grad():
            anal_emb = self.model.tree.logits2emb(anal_flat)  # (B*M, d)

        nn_flat = nn_emb.reshape(B * M, -1)
        return F.mse_loss(nn_flat, anal_emb)

    def _nn_step_to(self, current, target, edge_data):
        if current == target:
            return current
        path = self.model._get_path(current, target)
        for beta in path:
            current = self.model._step_one(current, beta, edge_data)
        return current

    def _anal_step_to(self, current, target, edge_data, u_hat, v_hat):
        """Navigate analytical decoder to target vertex."""
        if current == target:
            return
        path_up, path_down = [], []
        c, t = current, target
        while c != t:
            if c > t:
                c >>= 1
                path_up.append(('up', c))
            else:
                path_down.append(('down', t))
                t >>= 1
        path_down.reverse()
        full_path = path_up + path_down

        B = edge_data[1].shape[0]
        for direction, beta in full_path:
            if direction == 'down':
                parent = beta >> 1
                if beta & 1 == 0:
                    p = edge_data[parent]
                    r = edge_data[2 * parent + 1]
                    l = r.shape[1]
                    result = np.empty_like(r)
                    for bi in range(B):
                        for j in range(l):
                            result[bi, j] = _circ_conv_single(p[bi, j], r[bi, j])
                    edge_data[2 * parent] = result
                else:
                    p = edge_data[parent]
                    le = edge_data[2 * parent]
                    l = le.shape[1]
                    result = np.empty_like(le)
                    for bi in range(B):
                        for j in range(l):
                            result[bi, j] = _norm_prod_single(p[bi, j], le[bi, j])
                    edge_data[2 * parent + 1] = result
            else:  # up = CalcParent
                curr = current if direction == 'up' else beta
                # Simple CalcParent for navigation
                l = edge_data[2 * curr]
                r = edge_data[2 * curr + 1]
                result = np.empty_like(l)
                for bi in range(B):
                    for j in range(l.shape[1]):
                        result[bi, j] = _circ_conv_single(l[bi, j], r[bi, j])
                edge_data[curr] = np.concatenate([result, r], axis=1)


class SimpleMLP_Gmac(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2, z_hidden=32):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

    def _make_leaf_emb(self, u_val, v_val, batch):
        lp = torch.full((batch, 4), -30.0)
        LH = math.log(0.5)
        if u_val is not None and v_val is not None:
            idx = (u_val.long() * 2 + v_val.long()).unsqueeze(1)
            lp.scatter_(1, idx, 0.0)
        elif u_val is not None:
            lp.scatter_(1, (u_val.long() * 2).unsqueeze(1), LH)
            lp.scatter_(1, (u_val.long() * 2 + 1).unsqueeze(1), LH)
        elif v_val is not None:
            lp.scatter_(1, v_val.long().unsqueeze(1), LH)
            lp.scatter_(1, (v_val.long() + 2).unsqueeze(1), LH)
        else:
            lp.fill_(math.log(0.25))
        return self.tree.logits2emb(lp)

    @staticmethod
    def _get_path(current, target):
        if current == target:
            return []
        path_up, path_down = [], []
        c, t = current, target
        while c != t:
            if c > t:
                c >>= 1
                path_up.append(c)
            else:
                path_down.append(t)
                t >>= 1
        path_down.reverse()
        return path_up + path_down

    def _step_one(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0:
                self.tree._neural_calc_left(current, edge_data)
            else:
                self.tree._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self.tree._pure_neural_calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: {current} -> {beta}")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    return design_from_file(mc_path, n, ku, kv)


def evaluate(model, channel, N, b, Au, Av, fu, fv, n_cw):
    model.eval()
    n = int(math.log2(N))
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            uf = np.zeros((1, N), dtype=int); vf = np.zeros((1, N), dtype=int)
            for p in Au: uf[0, p-1] = rng.integers(0, 2)
            for p in Av: vf[0, p-1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            root = model.z_encoder(zf.unsqueeze(-1))[:, br]
            _, _, uh, vh, _ = model.tree(z=None, b=b, frozen_u=fu, frozen_v=fv, root_emb=root)
            ue = any(int(uh[p][0].item()) != uf[0, p-1] for p in Au if p in uh)
            ve = any(int(vh[p][0].item()) != vf[0, p-1] for p in Av if p in vh)
            if ue or ve: errs += 1
            total += 1
    model.train()
    return errs / total


def get_lr(it):
    if it < 1000:
        return LR * it / 1000
    progress = (it - 1000) / max(1, TOTAL_ITERS - 1000)
    return LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N_VAL, KU, KV)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    b = make_path(N_VAL, N_VAL // 2)

    model = SimpleMLP_Gmac(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    ckpt = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N256.pt')
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
        model.load_state_dict(fixed, strict=False)
        print(f'Loaded checkpoint', flush=True)

    print(f'Params: {model.count_parameters():,}', flush=True)

    init_bler = evaluate(model, channel, N_VAL, b, Au, Av, fu, fv, EVAL_CW)
    print(f'Initial BLER: {init_bler:.4f} (SC={SC_BLER})', flush=True)
    print(f'Lambda_snap: {LAMBDA_SNAP}', flush=True)
    print(f'Batch: {BATCH}, LR: {LR}, Iters: {TOTAL_ITERS}', flush=True)

    trainer = CombinedTrainer(model, channel, N_VAL, b, fu, fv)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    t0 = time.time()
    best_bler = init_bler

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        lr_now = get_lr(it)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        uf = np.zeros((BATCH, N_VAL), dtype=int); vf = np.zeros((BATCH, N_VAL), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)

        total_loss, ce_val, snap_val = trainer.train_step(zf, uf, vf)

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            bler = evaluate(model, channel, N_VAL, b, Au, Av, fu, fv, EVAL_CW)
            improved = ''
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'combined_N256_best.pt'))
                improved = ' *BEST*'
            msg = (f'[{it:>5}/{TOTAL_ITERS}] CE={ce_val:.4f} snap={snap_val:.4f} '
                   f'BLER={bler:.4f} (best={best_bler:.4f}, SC={SC_BLER}) '
                   f'{elapsed/60:.0f}min lr={lr_now:.1e}{improved}')
            print(msg, flush=True)
            with open(LOG_FILE, 'a') as f:
                f.write(msg + '\n')

    print(f'\nDONE: best={best_bler:.4f}', flush=True)


if __name__ == '__main__':
    main()
