#!/usr/bin/env python3
"""
poc_multidepth_scratch.py — Multi-depth auxiliary loss trained FROM SCRATCH.

Hypothesis: multi-depth aux loss failed when added to pre-trained model because
embeddings had already learned to be information carriers, not decodable at
intermediate levels. NPD's multi-depth loss works because it trains from scratch.

This POC trains TWO models from scratch with identical curriculum:
  1. BASELINE: leaf-only CE loss (standard sequential training)
  2. MULTI-DEPTH: leaf CE + lambda * mean(intermediate CE at every CalcLeft/CalcRight)

Curriculum: N=16 (10K) -> N=32 (15K) -> N=64 (30K) -> N=128 (30K)
All Class B (path_i = N//2), SNR=6dB, GaussianMAC.

At end of each stage, evaluates BLER on 1000 codewords.
"""
import sys, os, math, time, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path, design_gmac
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# ─── Config ──────────────────────────────────────────────────────────────────

D = 16
HIDDEN = 64
N_LAYERS = 2
Z_HIDDEN = 32

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
AUX_LAMBDA = 0.01

LR = 3e-4
WARMUP_FRAC = 0.05

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'poc_multidepth_scratch.log')

# Curriculum stages (reduced iters for POC due to CPU contention)
STAGES = [
    {'N': 16,  'ku': 8,  'kv': 8,  'iters': 8000,  'batch': 16, 'design': 'ga'},
    {'N': 32,  'ku': 15, 'kv': 15, 'iters': 12000, 'batch': 16, 'design': 'file'},
    {'N': 64,  'ku': 31, 'kv': 31, 'iters': 15000, 'batch': 8,  'design': 'file'},
    {'N': 128, 'ku': 62, 'kv': 62, 'iters': 10000, 'batch': 8,  'design': 'file'},
]

SC_REF = {32: 0.046, 64: 0.025, 128: 0.016}
EVAL_CW = 1000


# ─── Model ───────────────────────────────────────────────────────────────────

class SimpleMLP_Gmac(nn.Module):
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_hidden=Z_HIDDEN):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d))
        self.tree = PureNeuralCompGraphDecoder(
            d=d, hidden=hidden, n_layers=n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Precompute intermediate codeword bits at each butterfly stage ───────────

def precompute_intermediate_bits(u_msg, v_msg, N, n):
    """
    Compute the codeword at each butterfly stage for both users.

    stage_0 = u[bit_reversal]
    stage_s: for each block of size 2*step (step=2^s): left ^= right

    Returns u_stages, v_stages: each list of n+1 arrays (batch, N).

    Edge at tree level L corresponds to stage (n - L).
    Level 0 = root -> stage n = final codeword.
    Level n = leaves -> stage 0 = bit-reversed message.
    """
    br = bit_reversal_perm(n)
    batch = u_msg.shape[0]

    xu = u_msg[:, br].copy()
    xv = v_msg[:, br].copy()

    u_stages = [xu.copy()]
    v_stages = [xv.copy()]

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
    """Contiguous block [start:end] that edge beta covers in butterfly layout."""
    L = beta.bit_length() - 1
    block_size = N >> L
    block_idx = beta - (1 << L)
    start = block_idx * block_size
    end = start + block_size
    return start, end


def get_edge_stage(beta, n):
    """Butterfly stage for edge beta. Level L -> stage (n - L)."""
    L = beta.bit_length() - 1
    return n - L


# ─── Forward pass with optional multi-depth aux loss ─────────────────────────

def forward_with_optional_aux(model, z_tensor, b, frozen_u, frozen_v,
                               u_true, v_true, N, n,
                               u_stages=None, v_stages=None,
                               aux_lambda=0.0):
    """
    Forward pass through sequential SC tree decoder.

    If aux_lambda > 0 and u_stages/v_stages provided, adds intermediate CE loss
    at every CalcLeft/CalcRight output.

    Returns: main_loss, aux_loss, total_loss, u_hat, v_hat
    """
    tree = model.tree
    d = model.d
    device = z_tensor.device
    B = z_tensor.shape[0]

    br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
    root = model.z_encoder(z_tensor.unsqueeze(-1))[:, br]

    edge_data = [None] * (2 * N)
    edge_data[1] = root

    no_info = tree.no_info_emb.unsqueeze(0).unsqueeze(0)
    for beta in range(2, 2 * N):
        level = beta.bit_length() - 1
        size = N >> level
        edge_data[beta] = no_info.expand(B, size, d).clone()

    use_aux = aux_lambda > 0 and u_stages is not None
    aux_losses = []

    def collect_aux_loss(beta_edge):
        """Compute aux CE on intermediate edge embedding."""
        emb = edge_data[beta_edge]  # (B, M, d)
        logits = tree.emb2logits(emb)  # (B, M, 4)
        stage = get_edge_stage(beta_edge, n)
        start, end = get_edge_positions(beta_edge, N, n)

        u_bits = torch.from_numpy(u_stages[stage][:, start:end]).long().to(device)
        v_bits = torch.from_numpy(v_stages[stage][:, start:end]).long().to(device)
        targets = u_bits * 2 + v_bits

        loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
        aux_losses.append(loss)

    def step_one(current, beta):
        if current == beta >> 1:
            if beta & 1 == 0:
                tree._neural_calc_left(current, edge_data)
            else:
                tree._neural_calc_right(current, edge_data)
            if use_aux:
                collect_aux_loss(beta)
            return beta
        elif beta == current >> 1:
            tree._pure_neural_calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: current={current}, target={beta}")

    def step_to(current, target):
        if current == target:
            return current
        for beta in tree._get_path(current, target):
            current = step_one(current, beta)
        return current

    dec_head = 1
    u_hat, v_hat = {}, {}
    all_logits, all_targets = [], []
    i_u, i_v = 0, 0

    for step_idx in range(2 * N):
        gamma = b[step_idx]
        if gamma == 0:
            i_u += 1; i_t = i_u; fdict = frozen_u
        else:
            i_v += 1; i_t = i_v; fdict = frozen_v

        leaf_edge = i_t + N - 1
        target_vtx = leaf_edge >> 1

        dec_head = step_to(dec_head, target_vtx)

        temp = edge_data[leaf_edge][:, 0].clone()

        if leaf_edge & 1 == 0:
            tree._neural_calc_left(target_vtx, edge_data)
        else:
            tree._neural_calc_right(target_vtx, edge_data)
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
        main_loss = F.cross_entropy(
            torch.stack(all_logits).reshape(-1, 4),
            torch.stack(all_targets).reshape(-1))
    else:
        main_loss = torch.tensor(0.0, device=device)

    # Aux loss
    if aux_losses:
        aux_loss = torch.stack(aux_losses).mean()
    else:
        aux_loss = torch.tensor(0.0, device=device)

    total_loss = main_loss + aux_lambda * aux_loss
    return main_loss, aux_loss, total_loss, u_hat, v_hat


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, channel, N, n, b, Au, Av, fu, fv, n_cw):
    model.eval()
    tree = model.tree
    br_np = bit_reversal_perm(n)
    br = torch.from_numpy(br_np).long()
    errs = 0
    total = 0
    rng = np.random.default_rng(999)
    bs = 8

    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au:
                uf[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                vf[:, p - 1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            root = model.z_encoder(zf.unsqueeze(-1))[:, br]
            _, _, uh, vh, _ = tree(
                z=None, b=b, frozen_u=fu, frozen_v=fv, root_emb=root)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p - 1]
                        for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p - 1]
                        for p in Av if p in vh)
                if e:
                    errs += 1
            total += actual
    model.train()
    return errs / total


# ─── LR Schedule ─────────────────────────────────────────────────────────────

def get_lr(it, total_iters, lr, warmup_frac=WARMUP_FRAC):
    warmup = int(total_iters * warmup_frac)
    if it < warmup:
        return lr * it / max(1, warmup)
    progress = (it - warmup) / max(1, total_iters - warmup)
    return lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


# ─── Load design for a stage ─────────────────────────────────────────────────

def load_stage_design(stage):
    N = stage['N']
    n = int(math.log2(N))
    ku, kv = stage['ku'], stage['kv']

    if stage['design'] == 'ga':
        Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, sigma2=SIGMA2)
        return Au, Av, fu, fv
    else:
        mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{int(SNR_DB)}dB.npz')
        Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
        return Au, Av, fu, fv


# ─── Train one model through full curriculum ─────────────────────────────────

def train_model(model_name, model, use_aux, channel):
    """
    Train model through curriculum stages.

    Returns dict of {N: bler} for each evaluated stage.
    """
    results = {}
    rng = np.random.default_rng(42)
    t0_global = time.time()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    for stage_idx, stage in enumerate(STAGES):
        N = stage['N']
        n = int(math.log2(N))
        ku, kv = stage['ku'], stage['kv']
        total_iters = stage['iters']
        batch = stage['batch']

        Au, Av, fu, fv = load_stage_design(stage)
        b = make_path(N, N // 2)  # Class B

        print(f'\n{"="*60}')
        print(f'[{model_name}] Stage {stage_idx+1}: N={N}, ku={ku}, kv={kv}, '
              f'iters={total_iters}, batch={batch}, aux={use_aux}')
        print(f'{"="*60}', flush=True)

        # Reset optimizer state for new stage (fresh momentum)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

        losses = []
        t0 = time.time()

        model.train()
        for it in range(1, total_iters + 1):
            lr_now = get_lr(it, total_iters, LR)
            for pg in opt.param_groups:
                pg['lr'] = lr_now

            # Generate data
            uf = np.zeros((batch, N), dtype=int)
            vf = np.zeros((batch, N), dtype=int)
            for p in Au:
                uf[:, p - 1] = rng.integers(0, 2, batch)
            for p in Av:
                vf[:, p - 1] = rng.integers(0, 2, batch)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            u_true_t = torch.from_numpy(uf).float()
            v_true_t = torch.from_numpy(vf).float()

            # Precompute stages for aux loss
            if use_aux:
                u_stages, v_stages = precompute_intermediate_bits(uf, vf, N, n)
            else:
                u_stages, v_stages = None, None

            main_loss, aux_loss, total_loss, _, _ = forward_with_optional_aux(
                model, zf, b, fu, fv, u_true_t, v_true_t, N, n,
                u_stages, v_stages,
                aux_lambda=AUX_LAMBDA if use_aux else 0.0)

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            losses.append(total_loss.item())

            if it % 1000 == 0:
                elapsed = time.time() - t0
                avg_loss = np.mean(losses[-500:])
                speed = it / elapsed
                eta = (total_iters - it) / speed
                extra = ''
                if use_aux:
                    extra = f' aux={aux_loss.item():.4f}'
                print(f'  [{model_name}] N={N} [{it:>6}/{total_iters}] '
                      f'loss={avg_loss:.4f}{extra} lr={lr_now:.1e} '
                      f'{speed:.1f}it/s ETA={eta/60:.0f}min', flush=True)

        # Evaluate at end of stage
        elapsed_stage = time.time() - t0
        bler = evaluate(model, channel, N, n, b, Au, Av, fu, fv, EVAL_CW)
        sc_ref = SC_REF.get(N, None)
        ratio_str = f' ratio={bler/sc_ref:.2f}x' if sc_ref else ''
        sc_str = f' SC={sc_ref}' if sc_ref else ''

        results[N] = bler
        msg = (f'  [{model_name}] N={N} BLER={bler:.4f}{sc_str}{ratio_str} '
               f'({elapsed_stage/60:.1f}min)')
        print(msg, flush=True)
        with open(LOG_FILE, 'a') as f:
            f.write(msg + '\n')

    elapsed_total = time.time() - t0_global
    print(f'\n[{model_name}] Total training time: {elapsed_total/60:.1f}min', flush=True)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('POC: Multi-depth Auxiliary Loss TRAINED FROM SCRATCH')
    print(f'SNR={SNR_DB}dB, sigma2={SIGMA2:.4f}, aux_lambda={AUX_LAMBDA}')
    print(f'Curriculum: ' + ' -> '.join(
        f'N={s["N"]}({s["iters"]//1000}K)' for s in STAGES))
    print(f'SC references: {SC_REF}')
    print('=' * 70, flush=True)

    # Clear log
    with open(LOG_FILE, 'w') as f:
        f.write(f'Multi-depth from scratch POC\n')
        f.write(f'SNR={SNR_DB}dB, lambda={AUX_LAMBDA}\n')
        f.write(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')

    channel = GaussianMAC(sigma2=SIGMA2)

    # Create two models with same initialization
    torch.manual_seed(42)
    model_baseline = SimpleMLP_Gmac()
    print(f'Model params: {model_baseline.count_parameters():,}', flush=True)

    # Clone weights for multi-depth model (same init)
    torch.manual_seed(42)
    model_multidepth = SimpleMLP_Gmac()

    # Verify identical initialization
    for (n1, p1), (n2, p2) in zip(model_baseline.named_parameters(),
                                    model_multidepth.named_parameters()):
        assert torch.equal(p1.data, p2.data), f'Init mismatch: {n1}'
    print('Verified: both models have identical initialization.', flush=True)

    # Train baseline (leaf-only loss)
    print('\n' + '~' * 70)
    print('TRAINING BASELINE (leaf-only CE)')
    print('~' * 70, flush=True)
    t0 = time.time()
    results_baseline = train_model('BASELINE', model_baseline, use_aux=False, channel=channel)
    time_baseline = time.time() - t0

    # Train multi-depth (aux loss from scratch)
    print('\n' + '~' * 70)
    print('TRAINING MULTI-DEPTH (leaf CE + aux intermediate CE)')
    print('~' * 70, flush=True)
    t0 = time.time()
    results_multidepth = train_model('MULTIDEPTH', model_multidepth, use_aux=True, channel=channel)
    time_multidepth = time.time() - t0

    # Print comparison
    print('\n' + '=' * 70)
    print('RESULTS COMPARISON')
    print('=' * 70)
    print(f'{"N":>6} {"Baseline":>10} {"MultiDepth":>10} {"SC Ref":>10} {"BL/SC":>8} {"MD/SC":>8} {"Winner":>10}')
    print('-' * 68)

    for stage in STAGES:
        N = stage['N']
        bl = results_baseline.get(N, float('nan'))
        md = results_multidepth.get(N, float('nan'))
        sc = SC_REF.get(N, None)

        bl_ratio = f'{bl/sc:.2f}x' if sc else 'N/A'
        md_ratio = f'{md/sc:.2f}x' if sc else 'N/A'

        if bl < md:
            winner = 'BASELINE'
        elif md < bl:
            winner = 'MULTIDEPTH'
        else:
            winner = 'TIE'

        sc_str = f'{sc:.4f}' if sc else 'N/A'
        print(f'{N:>6} {bl:>10.4f} {md:>10.4f} {sc_str:>10} {bl_ratio:>8} {md_ratio:>8} {winner:>10}')

    print('-' * 68)
    print(f'Baseline training time:   {time_baseline/60:.1f} min')
    print(f'MultiDepth training time: {time_multidepth/60:.1f} min')
    print('=' * 70, flush=True)

    # Write final summary to log
    with open(LOG_FILE, 'a') as f:
        f.write(f'\n{"="*60}\nFINAL RESULTS\n{"="*60}\n')
        for stage in STAGES:
            N = stage['N']
            bl = results_baseline.get(N, float('nan'))
            md = results_multidepth.get(N, float('nan'))
            sc = SC_REF.get(N, None)
            sc_str = f'{sc}' if sc else 'N/A'
            f.write(f'N={N}: baseline={bl:.4f}, multidepth={md:.4f}, SC={sc_str}\n')
        f.write(f'\nBaseline time: {time_baseline/60:.1f}min\n')
        f.write(f'MultiDepth time: {time_multidepth/60:.1f}min\n')
        f.write(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()
