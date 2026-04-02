#!/usr/bin/env python3
"""
poc_snapshot_prob_n256.py — Snapshot training in probability domain at N=256.

Phase 1: Run analytical SC on 200 codewords, store all CalcLeft/CalcRight
          input/output tensors in PROBABILITY domain (no -inf, no NaN)
Phase 2: Train CalcLeft/CalcRight NNs against analytical targets using CE loss
          Use model's own logits2emb/emb2logits as converters
Phase 3: Fine-tune with standard end-to-end sequential CE
Phase 4: Evaluate BLER
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
from polar.decoder import build_log_W_leaf, decode_single
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

N_VAL = 256
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
SC_BLER = 0.005
KU = 123; KV = 123

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')

# ─── Analytical operations in prob domain ──────────────────────────────────

def circ_conv_prob(A, B):
    """Circular convolution in probability domain. A, B: (2,2) arrays."""
    out = np.zeros((2, 2))
    for u in range(2):
        for v in range(2):
            for u2 in range(2):
                for v2 in range(2):
                    out[u, v] += A[u ^ u2, v ^ v2] * B[u2, v2]
    return out

def norm_prod_prob(A, B):
    """Normalized element-wise product in probability domain."""
    C = A * B
    s = C.sum()
    if s > 0:
        return C / s
    return np.full((2, 2), 0.25)


# ─── Phase 1: Collect snapshot DB ──────────────────────────────────────────

def collect_snapshot_db(channel, N, b, fu, fv, Au, Av, n_codewords):
    """
    Run analytical SC decoder, record all CalcLeft/CalcRight operations
    in PROBABILITY domain.

    Returns: list of (op_type, in1_prob, in2_prob, out_prob) tuples
             each prob is a (2,2) numpy array
    """
    n = int(np.log2(N))
    br = bit_reversal_perm(n)
    rng = np.random.default_rng(42)

    db_left = []   # (in1, in2, out) for CalcLeft
    db_right = []  # (in1, in2, out) for CalcRight

    LOG_QUARTER = np.log(0.25)
    NEG_INF = -1e30
    LOG_HALF = np.log(0.5)

    t0 = time.time()
    for cw in range(n_codewords):
        # Generate codeword
        uf = np.zeros(N, dtype=int); vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        xf = polar_encode_batch(uf.reshape(1,-1))[0]
        yf = polar_encode_batch(vf.reshape(1,-1))[0]
        zf = channel.sample_batch(xf.reshape(1,-1).astype(float),
                                   yf.reshape(1,-1).astype(float))[0]

        # Build leaf tensors in log domain
        log_W = build_log_W_leaf(zf, channel)  # (N, 2, 2)
        log_W_br = log_W[br]

        # Init edges in log domain
        edge = [None] * (2 * N)
        edge[1] = log_W_br.copy()  # (N, 2, 2)
        for beta in range(2, 2*N):
            lev = beta.bit_length() - 1
            size = N >> lev
            edge[beta] = np.full((size, 2, 2), LOG_QUARTER)

        # Walk the tree, record operations
        dec_head = 1
        u_hat, v_hat = {}, {}
        i_u, i_v = 0, 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = fu
            else:
                i_v += 1; i_t = i_v; fdict = fv

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1

            # Navigate to target
            path_up, path_down = [], []
            c, t = dec_head, target_vtx
            while c != t:
                if c > t: c >>= 1; path_up.append(c)
                else: path_down.append(t); t >>= 1
            path_down.reverse()

            for beta in path_up:
                # CalcParent (go up)
                curr = dec_head
                l = edge[2*curr]; r = edge[2*curr+1]
                result = np.empty_like(l)
                for j in range(l.shape[0]):
                    log_res = np.empty((2,2))
                    for u in range(2):
                        for v in range(2):
                            log_res[u,v] = np.logaddexp(
                                np.logaddexp(l[j,u^0,v^0]+r[j,0,0], l[j,u^0,v^1]+r[j,0,1]),
                                np.logaddexp(l[j,u^1,v^0]+r[j,1,0], l[j,u^1,v^1]+r[j,1,1]))
                    result[j] = log_res
                edge[curr] = np.concatenate([result, r], axis=0)
                dec_head = beta

            for beta in path_down:
                parent = beta >> 1
                if beta & 1 == 0:  # CalcLeft
                    p = edge[parent]; r = edge[2*parent+1]
                    l = r.shape[0]
                    result = np.empty((l, 2, 2))
                    for j in range(l):
                        log_res = np.empty((2,2))
                        for u in range(2):
                            for v in range(2):
                                log_res[u,v] = np.logaddexp(
                                    np.logaddexp(p[j,u^0,v^0]+r[j,0,0], p[j,u^0,v^1]+r[j,0,1]),
                                    np.logaddexp(p[j,u^1,v^0]+r[j,1,0], p[j,u^1,v^1]+r[j,1,1]))
                        result[j] = log_res
                        # Record in prob domain
                        in1_prob = np.exp(np.clip(p[j], -30, 0))
                        in2_prob = np.exp(np.clip(r[j], -30, 0))
                        out_prob = np.exp(np.clip(log_res, -30, 0))
                        # Normalize
                        in1_prob /= max(in1_prob.sum(), 1e-10)
                        in2_prob /= max(in2_prob.sum(), 1e-10)
                        out_prob /= max(out_prob.sum(), 1e-10)
                        db_left.append((in1_prob.copy(), in2_prob.copy(), out_prob.copy()))
                    edge[2*parent] = result
                else:  # CalcRight
                    p = edge[parent]; le = edge[2*parent]
                    l = le.shape[0]
                    result = np.empty((l, 2, 2))
                    for j in range(l):
                        C = p[j] + le[j]
                        log_s = np.logaddexp(np.logaddexp(C[0,0],C[0,1]), np.logaddexp(C[1,0],C[1,1]))
                        log_res = C - log_s
                        result[j] = log_res
                        in1_prob = np.exp(np.clip(p[j], -30, 0))
                        in2_prob = np.exp(np.clip(le[j], -30, 0))
                        out_prob = np.exp(np.clip(log_res, -30, 0))
                        in1_prob /= max(in1_prob.sum(), 1e-10)
                        in2_prob /= max(in2_prob.sum(), 1e-10)
                        out_prob /= max(out_prob.sum(), 1e-10)
                        db_right.append((in1_prob.copy(), in2_prob.copy(), out_prob.copy()))
                    edge[2*parent+1] = result
                dec_head = beta

            # Leaf operation
            temp = edge[leaf_edge][0].copy()
            parent = target_vtx
            if leaf_edge & 1 == 0:
                p = edge[parent]; r = edge[2*parent+1]
                log_res = np.empty((2,2))
                for u in range(2):
                    for v in range(2):
                        log_res[u,v] = np.logaddexp(
                            np.logaddexp(p[0,u^0,v^0]+r[0,0,0], p[0,u^0,v^1]+r[0,0,1]),
                            np.logaddexp(p[0,u^1,v^0]+r[0,1,0], p[0,u^1,v^1]+r[0,1,1]))
                edge[2*parent] = log_res.reshape(1,2,2)
                in1_p = np.exp(np.clip(p[0],-30,0)); in1_p/=max(in1_p.sum(),1e-10)
                in2_p = np.exp(np.clip(r[0],-30,0)); in2_p/=max(in2_p.sum(),1e-10)
                out_p = np.exp(np.clip(log_res,-30,0)); out_p/=max(out_p.sum(),1e-10)
                db_left.append((in1_p.copy(), in2_p.copy(), out_p.copy()))
            else:
                p = edge[parent]; le = edge[2*parent]
                C = p[0] + le[0]
                log_s = np.logaddexp(np.logaddexp(C[0,0],C[0,1]),np.logaddexp(C[1,0],C[1,1]))
                log_res = C - log_s
                edge[2*parent+1] = log_res.reshape(1,2,2)
                in1_p = np.exp(np.clip(p[0],-30,0)); in1_p/=max(in1_p.sum(),1e-10)
                in2_p = np.exp(np.clip(le[0],-30,0)); in2_p/=max(in2_p.sum(),1e-10)
                out_p = np.exp(np.clip(log_res,-30,0)); out_p/=max(out_p.sum(),1e-10)
                db_right.append((in1_p.copy(), in2_p.copy(), out_p.copy()))

            # Set leaf (genie: use true bits)
            u_bit = uf[i_t-1]; v_bit = vf[i_t-1]
            new_leaf = np.full((2,2), NEG_INF)
            new_leaf[u_bit, v_bit] = 0.0
            edge[leaf_edge] = new_leaf.reshape(1,2,2)
            if gamma == 0: u_hat[i_t] = u_bit
            else: v_hat[i_t] = v_bit

        if (cw+1) % 50 == 0:
            print(f'  {cw+1}/{n_codewords} ({time.time()-t0:.0f}s)', flush=True)

    print(f'Snapshot DB: {len(db_left)} CalcLeft, {len(db_right)} CalcRight samples', flush=True)
    return db_left, db_right


# ─── Model ─────────────────────────────────────────────────────────────────

class SimpleMLP_Gmac(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2, z_hidden=32):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d))
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Phase 2: Snapshot training ────────────────────────────────────────────

def train_snapshot_phase(model, db_left, db_right, n_iters=5000, lr=1e-4, batch_size=256):
    """Train CalcLeft/CalcRight against analytical prob-domain targets."""
    # Convert DB to tensors
    left_in1 = torch.from_numpy(np.array([x[0].flatten() for x in db_left])).float()
    left_in2 = torch.from_numpy(np.array([x[1].flatten() for x in db_left])).float()
    left_out = torch.from_numpy(np.array([x[2].flatten() for x in db_left])).float()

    right_in1 = torch.from_numpy(np.array([x[0].flatten() for x in db_right])).float()
    right_in2 = torch.from_numpy(np.array([x[1].flatten() for x in db_right])).float()
    right_out = torch.from_numpy(np.array([x[2].flatten() for x in db_right])).float()

    print(f'  CalcLeft samples: {len(left_in1)}, CalcRight samples: {len(right_in1)}', flush=True)

    # Only train CalcLeft/CalcRight + logits2emb/emb2logits (converters)
    trainable = list(model.tree.calc_left_nn.parameters()) + \
                list(model.tree.calc_right_nn.parameters()) + \
                list(model.tree.emb2logits.parameters()) + \
                list(model.tree.logits2emb.parameters())
    opt = torch.optim.Adam(trainable, lr=lr)

    rng = np.random.default_rng(123)
    t0 = time.time()

    for it in range(1, n_iters + 1):
        # Sample batch for CalcLeft
        idx_l = rng.integers(0, len(left_in1), batch_size)
        in1_l = left_in1[idx_l]   # (batch, 4) prob domain
        in2_l = left_in2[idx_l]
        target_l = left_out[idx_l]

        # Convert prob inputs to log domain, then to embeddings
        log_in1 = torch.log(in1_l.clamp(min=1e-10))
        log_in2 = torch.log(in2_l.clamp(min=1e-10))
        emb1 = model.tree.logits2emb(log_in1)  # (batch, d)
        emb2 = model.tree.logits2emb(log_in2)

        # CalcLeft: inp = cat(parent_first, parent_second, right)
        # parent_first ≈ emb1, parent_second ≈ emb1, right = emb2
        inp_l = torch.cat([emb1, emb1, emb2], dim=-1)
        out_emb_l = model.tree.calc_left_nn(inp_l)
        pred_logits_l = model.tree.emb2logits(out_emb_l)  # (batch, 4)
        pred_prob_l = F.softmax(pred_logits_l, dim=-1)

        # CE loss against prob-domain target
        loss_l = -(target_l * torch.log(pred_prob_l.clamp(min=1e-10))).sum(dim=-1).mean()

        # Same for CalcRight
        idx_r = rng.integers(0, len(right_in1), batch_size)
        in1_r = right_in1[idx_r]
        in2_r = right_in2[idx_r]
        target_r = right_out[idx_r]

        log_in1_r = torch.log(in1_r.clamp(min=1e-10))
        log_in2_r = torch.log(in2_r.clamp(min=1e-10))
        emb1_r = model.tree.logits2emb(log_in1_r)
        emb2_r = model.tree.logits2emb(log_in2_r)
        inp_r = torch.cat([emb1_r, emb1_r, emb2_r], dim=-1)
        out_emb_r = model.tree.calc_right_nn(inp_r)
        pred_logits_r = model.tree.emb2logits(out_emb_r)
        pred_prob_r = F.softmax(pred_logits_r, dim=-1)
        loss_r = -(target_r * torch.log(pred_prob_r.clamp(min=1e-10))).sum(dim=-1).mean()

        loss = loss_l + loss_r
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if it % 1000 == 0:
            print(f'  [{it}/{n_iters}] loss_L={loss_l.item():.4f} loss_R={loss_r.item():.4f} '
                  f'({time.time()-t0:.0f}s)', flush=True)


# ─── Phase 3: End-to-end fine-tune ─────────────────────────────────────────

def train_e2e_phase(model, channel, N, b, Au, Av, fu, fv, n_iters=5000, lr=3e-5, batch=4):
    """Standard sequential CE training to restore end-to-end performance."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(456)
    br = torch.from_numpy(bit_reversal_perm(int(np.log2(N)))).long()
    t0 = time.time()

    for it in range(1, n_iters + 1):
        uf = np.zeros((batch, N), dtype=int); vf = np.zeros((batch, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, batch)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, batch)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        root = model.z_encoder(zf.unsqueeze(-1))[:, br]
        logits, targets, _, _, _ = model.tree(z=None, b=b, frozen_u=fu, frozen_v=fv,
                                               u_true=torch.from_numpy(uf).float(),
                                               v_true=torch.from_numpy(vf).float(),
                                               root_emb=root)
        if logits:
            loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4),
                                   torch.stack(targets).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if it % 1000 == 0:
            print(f'  [{it}/{n_iters}] CE={loss.item():.4f} ({time.time()-t0:.0f}s)', flush=True)


# ─── Evaluate ──────────────────────────────────────────────────────────────

def evaluate(model, channel, N, b, Au, Av, fu, fv, n_cw):
    model.eval()
    br = torch.from_numpy(bit_reversal_perm(int(np.log2(N)))).long()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(4, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            root = model.z_encoder(zf.unsqueeze(-1))[:, br]
            _, _, uh, vh, _ = model.tree(z=None, b=b, frozen_u=fu, frozen_v=fv, root_emb=root)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    n = int(np.log2(N_VAL))
    result = design_from_file(os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr6dB.npz'), n, KU, KV)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    b = make_path(N_VAL, N_VAL // 2)

    # Load model
    model = SimpleMLP_Gmac()
    ckpt = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N256.pt')
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    print(f'Loaded checkpoint, params={model.count_parameters():,}', flush=True)

    # Phase 0: Baseline eval
    print(f'\n=== Phase 0: Baseline ===', flush=True)
    bler0 = evaluate(model, channel, N_VAL, b, Au, Av, fu, fv, 1000)
    print(f'Baseline BLER: {bler0:.4f} (SC={SC_BLER})', flush=True)

    # Phase 1: Collect snapshots
    print(f'\n=== Phase 1: Collecting snapshot DB ===', flush=True)
    db_left, db_right = collect_snapshot_db(channel, N_VAL, b, fu, fv, Au, Av, 200)

    # Phase 2: Snapshot training
    print(f'\n=== Phase 2: Snapshot training (prob domain CE) ===', flush=True)
    train_snapshot_phase(model, db_left, db_right, n_iters=5000, lr=1e-4)
    bler2 = evaluate(model, channel, N_VAL, b, Au, Av, fu, fv, 1000)
    print(f'After snapshot: BLER={bler2:.4f}', flush=True)

    # Phase 3: End-to-end fine-tune
    print(f'\n=== Phase 3: End-to-end fine-tune ===', flush=True)
    train_e2e_phase(model, channel, N_VAL, b, Au, Av, fu, fv, n_iters=5000, lr=3e-5)
    bler3 = evaluate(model, channel, N_VAL, b, Au, Av, fu, fv, 1000)
    print(f'After E2E fine-tune: BLER={bler3:.4f}', flush=True)

    print(f'\n=== SUMMARY ===', flush=True)
    print(f'Baseline:       {bler0:.4f}', flush=True)
    print(f'After snapshot: {bler2:.4f}', flush=True)
    print(f'After E2E:      {bler3:.4f}', flush=True)
    print(f'SC reference:   {SC_BLER}', flush=True)


if __name__ == '__main__':
    main()
