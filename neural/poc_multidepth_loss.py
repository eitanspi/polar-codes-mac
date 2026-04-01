#!/usr/bin/env python3
"""
poc_multidepth_loss.py — POC: multi-depth auxiliary loss during sequential training.

CONCEPT: Add auxiliary CE loss at INTERMEDIATE edges during the tree walk.
At each CalcLeft/CalcRight output, apply emb2logits and compare against
the true intermediate joint (u,v) bits at that depth.

This gives gradients directly to early tree operations instead of flowing
through 500+ sequential steps.

Total loss = main_leaf_CE + lambda * aux_loss

Trains at N=256, Class B, SNR=6dB. Compares against baseline BLER ~0.019
and SC BLER ~0.005.
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# ─── Config ────────────────────────────────────────────────────────────────────
N = 256; n = 8
KU = 123; KV = 123
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
SC_BLER = 0.005
BASELINE_BLER = 0.019

BATCH = 8
LR = 5e-5
TOTAL_ITERS = 20000
EVAL_EVERY = 5000
EVAL_CW = 1000
AUX_LAMBDA = 0.1

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'poc_multidepth_loss.log')
CKPT_BEST = os.path.join(SAVE_DIR, 'multidepth_best.pt')


# ─── Model (same as baseline) ─────────────────────────────────────────────────
class SimpleMLP_Gmac(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = 16
        self.z_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ELU(), nn.Linear(32, 16))
        self.tree = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Precompute intermediate codeword bits at each butterfly stage ─────────────
def precompute_intermediate_bits(u_msg, v_msg, N, n):
    """
    Compute the codeword at each butterfly stage for both users.

    The encoder does:
        stage_0 = u[bit_reversal]
        stage_s: for each block of size 2*step (step=2^s):
            left ^= right

    We store stage_0 .. stage_n (stage_n = final codeword).

    Returns:
        u_stages: list of n+1 arrays, each (batch, N)
        v_stages: list of n+1 arrays, each (batch, N)

    The tree edge at level L (root=level 0, leaves=level n) corresponds
    to the encoding at stage (n - L).

    Edge beta at level L covers positions determined by beta's position
    in the tree. Specifically, edge beta covers a contiguous block of
    size N/2^L in the bit-reversed/butterfly layout.
    """
    br = bit_reversal_perm(n)
    batch = u_msg.shape[0]

    u_stages = []
    v_stages = []

    # Stage 0: after bit reversal
    xu = u_msg[:, br].copy()
    xv = v_msg[:, br].copy()
    u_stages.append(xu.copy())
    v_stages.append(xv.copy())

    # Stages 1..n: butterfly XOR
    step = 1
    for s in range(n):
        xur = xu.reshape(batch, N // (2 * step), 2, step)
        xvr = xv.reshape(batch, N // (2 * step), 2, step)
        xur[:, :, 0, :] ^= xur[:, :, 1, :]
        xvr[:, :, 0, :] ^= xvr[:, :, 1, :]
        u_stages.append(xu.copy())
        v_stages.append(xv.copy())
        step *= 2

    return u_stages, v_stages


def get_edge_positions(beta, N, n):
    """
    Get the contiguous block of positions that edge beta covers
    in the butterfly layout.

    Edge beta is at level L = floor(log2(beta)).
    The block size is N / 2^L.
    The block index within the level is beta - 2^L.

    Returns: (start, end) — slice [start:end] into the stage array.
    """
    L = beta.bit_length() - 1  # level: root=0, leaves=n
    block_size = N >> L
    block_idx = beta - (1 << L)
    start = block_idx * block_size
    end = start + block_size
    return start, end


def get_edge_stage(beta, n):
    """
    Which butterfly stage does edge beta correspond to?

    Edge at level L corresponds to stage (n - L).
    After stage s, step = 2^s sized blocks have been processed.

    Level 0 (root) -> stage n (fully encoded codeword)
    Level n (leaves) -> stage 0 (bit-reversed message)
    """
    L = beta.bit_length() - 1
    return n - L


# ─── Modified forward pass with auxiliary loss ────────────────────────────────
def forward_with_aux_loss(model, z, b, frozen_u, frozen_v,
                          u_true, v_true, root_emb,
                          u_stages, v_stages, aux_lambda=0.1):
    """
    Modified forward pass that adds auxiliary CE loss at intermediate edges.

    After each CalcLeft/CalcRight produces a new edge embedding, we:
    1. Apply emb2logits to get (B, M, 4) logits
    2. Look up the true intermediate (u,v) bits for that edge at that depth
    3. Add CE loss

    Returns: main_loss, aux_loss, total_loss, u_hat, v_hat
    """
    tree = model.tree
    B, NN, d = root_emb.shape
    device = root_emb.device
    nn_val = n  # number of levels

    edge_data = [None] * (2 * N)
    edge_data[1] = root_emb

    no_info = tree.no_info_emb.unsqueeze(0).unsqueeze(0)
    for beta in range(2, 2 * N):
        level = beta.bit_length() - 1
        size = N >> level
        edge_data[beta] = no_info.expand(B, size, d).clone()

    # Track auxiliary losses
    aux_losses = []

    def collect_aux_loss(beta_edge):
        """Collect aux loss for edge beta_edge after CalcLeft/CalcRight."""
        emb = edge_data[beta_edge]  # (B, M, d)
        logits = tree.emb2logits(emb)  # (B, M, 4)
        stage = get_edge_stage(beta_edge, nn_val)
        start, end = get_edge_positions(beta_edge, N, nn_val)

        # Get true bits at this stage
        u_bits = torch.from_numpy(u_stages[stage][:, start:end]).long().to(device)
        v_bits = torch.from_numpy(v_stages[stage][:, start:end]).long().to(device)
        targets = u_bits * 2 + v_bits  # (B, M) in {0,1,2,3}

        loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
        aux_losses.append(loss)

    # Modified _step_one that collects aux loss
    def step_one(current, beta, edge_data):
        if current == beta >> 1:
            # Going DOWN
            if beta & 1 == 0:
                tree._neural_calc_left(current, edge_data)
            else:
                tree._neural_calc_right(current, edge_data)
            collect_aux_loss(beta)
            return beta
        elif beta == current >> 1:
            # Going UP
            tree._pure_neural_calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: current={current}, target={beta}")

    def step_to(current, target, edge_data):
        if current == target:
            return current
        for beta in tree._get_path(current, target):
            current = step_one(current, beta, edge_data)
        return current

    dec_head = 1
    u_hat, v_hat = {}, {}
    all_logits, all_targets = [], []
    i_u, i_v = 0, 0

    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u; fdict = frozen_u
        else:
            i_v += 1; i_t = i_v; fdict = frozen_v

        leaf_edge = i_t + N - 1
        target_vtx = leaf_edge >> 1

        dec_head = step_to(dec_head, target_vtx, edge_data)

        temp = edge_data[leaf_edge][:, 0].clone()

        if leaf_edge & 1 == 0:
            tree._neural_calc_left(target_vtx, edge_data)
        else:
            tree._neural_calc_right(target_vtx, edge_data)
        # Aux loss on leaf edge too (will match the main loss target)
        # Skip this — leaves are already covered by main CE loss.

        top_down = edge_data[leaf_edge][:, 0]
        combined = top_down + temp
        logits = tree.emb2logits(combined)

        if i_t in fdict:
            bit = torch.full((B,), fdict[i_t], dtype=torch.float32, device=device)
        else:
            all_logits.append(logits)
            target = (u_true[:, i_t - 1] * 2 + v_true[:, i_t - 1]).long()
            all_targets.append(target)
            bit = u_true[:, i_t - 1] if gamma == 0 else v_true[:, i_t - 1]

        if gamma == 0:
            u_hat[i_t] = bit
        else:
            v_hat[i_t] = bit

        new_emb = tree._make_leaf_emb(
            u_hat.get(i_t), v_hat.get(i_t), B, device)
        edge_data[leaf_edge] = new_emb.unsqueeze(1)

    # Main leaf CE loss
    if all_logits:
        main_loss = F.cross_entropy(torch.stack(all_logits).reshape(-1, 4),
                                     torch.stack(all_targets).reshape(-1))
    else:
        main_loss = torch.tensor(0.0, device=device)

    # Aux loss (average over all intermediate edges)
    if aux_losses:
        aux_loss = torch.stack(aux_losses).mean()
    else:
        aux_loss = torch.tensor(0.0, device=device)

    total_loss = main_loss + aux_lambda * aux_loss
    return main_loss, aux_loss, total_loss, u_hat, v_hat


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model, channel, b, Au, Av, fu, fv, n_cw):
    model.eval()
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(8, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            root = model.z_encoder(zf.unsqueeze(-1))[:, br]
            _, _, uh, vh, _ = model.tree(
                z=None, b=b, frozen_u=fu, frozen_v=fv, root_emb=root)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1]
                        for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1]
                        for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


# ─── LR Schedule ──────────────────────────────────────────────────────────────
def get_lr(it):
    warmup = 1000
    if it < warmup:
        return LR * it / warmup
    progress = (it - warmup) / max(1, TOTAL_ITERS - warmup)
    return LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("POC: Multi-depth Auxiliary Loss")
    print(f"N={N}, Class B, SNR={SNR_DB}dB, lambda={AUX_LAMBDA}")
    print(f"Baseline BLER={BASELINE_BLER}, SC BLER={SC_BLER}")
    print("=" * 70, flush=True)

    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv, _, _, _ = design_from_file(
        os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{int(SNR_DB)}dB.npz'), n, KU, KV)
    b = make_path(N, N // 2)
    br_np = bit_reversal_perm(n)
    br = torch.from_numpy(br_np).long()

    model = SimpleMLP_Gmac()

    # Load checkpoint
    ckpt = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N256.pt')
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
        model.load_state_dict(fixed, strict=False)
        print(f'Loaded checkpoint: {ckpt}', flush=True)
    else:
        print(f'WARNING: no checkpoint found at {ckpt}', flush=True)

    print(f'Params: {model.count_parameters():,}', flush=True)

    # Initial evaluation
    init_bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW)
    print(f'Initial BLER: {init_bler:.4f} (baseline={BASELINE_BLER}, SC={SC_BLER})',
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(123)
    t0 = time.time()
    main_losses = []
    aux_losses_log = []
    best_bler = init_bler

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        lr_now = get_lr(it)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        # Generate data
        uf = np.zeros((BATCH, N), dtype=int)
        vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        root = model.z_encoder(zf.unsqueeze(-1))[:, br]

        # Precompute intermediate codeword bits
        u_stages, v_stages = precompute_intermediate_bits(uf, vf, N, n)

        # Forward with aux loss
        u_true_t = torch.from_numpy(uf).float()
        v_true_t = torch.from_numpy(vf).float()

        main_loss, aux_loss, total_loss, _, _ = forward_with_aux_loss(
            model, zf, b, fu, fv, u_true_t, v_true_t, root,
            u_stages, v_stages, aux_lambda=AUX_LAMBDA)

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        main_losses.append(main_loss.item())
        aux_losses_log.append(aux_loss.item())

        if it % 500 == 0:
            elapsed = time.time() - t0
            avg_main = np.mean(main_losses[-500:])
            avg_aux = np.mean(aux_losses_log[-500:])
            msg = (f'[{it:>6}/{TOTAL_ITERS}] main_CE={avg_main:.4f} '
                   f'aux_CE={avg_aux:.4f} lr={lr_now:.1e} '
                   f'{elapsed/60:.1f}min')
            print(msg, flush=True)

        # Evaluate
        if it % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            avg_main = np.mean(main_losses[-500:])
            avg_aux = np.mean(aux_losses_log[-500:])
            bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), CKPT_BEST)
                improved = ' *BEST*'

            msg = (f'[EVAL {it:>6}/{TOTAL_ITERS}] main_CE={avg_main:.4f} '
                   f'aux_CE={avg_aux:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}, baseline={BASELINE_BLER}, SC={SC_BLER}) '
                   f'{elapsed/60:.0f}min{improved}')
            print(msg, flush=True)
            with open(LOG_FILE, 'a') as f:
                f.write(msg + '\n')

    elapsed = time.time() - t0
    final_msg = (f'\nDONE: best_bler={best_bler:.4f} '
                 f'(baseline={BASELINE_BLER}, SC={SC_BLER}), '
                 f'{elapsed/3600:.1f}hr')
    print(final_msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(final_msg + '\n')


if __name__ == '__main__':
    main()
