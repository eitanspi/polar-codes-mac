#!/usr/bin/env python3
"""
train_n512_long.py — Long training at N=512 with C++ accelerated forward pass.
Saves checkpoint every 5K iters. Stable cosine LR (no warm restarts).
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# Compile C++ extension
print('Compiling C++ extension...', flush=True)
fast_walk = load(name='fast_tree_walk',
                 sources=[os.path.join(os.path.dirname(__file__), 'csrc', 'fast_tree_walk.cpp')],
                 extra_cflags=['-O3', '-std=c++17'], verbose=False)
print('C++ extension ready.', flush=True)

N = 512; n = 9
KU = 246; KV = 246
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
SC_BLER = 0.001

BATCH = 4
LR = 5e-5
TOTAL_ITERS = 100000
EVAL_EVERY = 5000
SAVE_EVERY = 5000
EVAL_CW = 500

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'train_n512_long.log')
CKPT_LATEST = os.path.join(SAVE_DIR, 'n512_long_latest.pt')
CKPT_BEST = os.path.join(SAVE_DIR, 'n512_long_best.pt')


class SimpleMLP_Gmac(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = 16
        self.z_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 32), torch.nn.ELU(), torch.nn.Linear(32, 16))
        self.tree = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_schedule_flat(N, b, fu, fv):
    schedule = []; dec_head = 1; i_u = 0; i_v = 0
    for step in range(2*N):
        gamma = b[step]
        if gamma == 0: i_u += 1; i_t = i_u; is_frozen = int(i_t in fu)
        else: i_v += 1; i_t = i_v; is_frozen = int(i_t in fv)
        leaf_edge = i_t + N - 1; target_vtx = leaf_edge >> 1
        path_up, path_down = [], []
        c, t = dec_head, target_vtx
        while c != t:
            if c > t: c >>= 1; path_up.append(c)
            else: path_down.append(t); t >>= 1
        path_down.reverse()
        for beta in path_up: schedule.extend([2, dec_head]); dec_head = beta
        for beta in path_down:
            parent = beta >> 1
            if beta & 1 == 0: schedule.extend([0, parent])
            else: schedule.extend([1, parent])
            dec_head = beta
        op_type = 3 if (leaf_edge & 1 == 0) else 4
        schedule.extend([op_type, target_vtx, leaf_edge, i_t, gamma, is_frozen])
    return schedule


def get_mlp_weights(mlp):
    return [mlp[0].weight, mlp[0].bias, mlp[2].weight, mlp[2].bias, mlp[4].weight, mlp[4].bias]


def get_all_weights(model):
    t = model.tree
    return (
        get_mlp_weights(t.calc_left_nn),
        get_mlp_weights(t.calc_right_nn),
        [t.calc_parent_nn.gate_net[0].weight, t.calc_parent_nn.gate_net[0].bias,
         t.calc_parent_nn.gate_net[2].weight, t.calc_parent_nn.gate_net[2].bias],
        get_mlp_weights(t.calc_parent_nn.candidate_net),
        [t.parent_second_nn[0].weight, t.parent_second_nn[0].bias],
        get_mlp_weights(t.emb2logits),
        get_mlp_weights(t.logits2emb),
    )


def get_lr(it):
    warmup = 1000
    if it < warmup: return LR * it / warmup
    progress = (it - warmup) / max(1, TOTAL_ITERS - warmup)
    return LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate(model, channel, b, Au, Av, fu, fv, n_cw):
    model.eval()
    br = torch.from_numpy(bit_reversal_perm(n)).long()
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
    Au, Av, fu, fv, _, _, _ = design_from_file(
        os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz'), n, KU, KV)
    b = make_path(N, N // 2)
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    schedule_flat = build_schedule_flat(N, b, fu, fv)
    frozen_u_mask = torch.tensor([int(i+1 in fu) for i in range(N)], dtype=torch.bool)
    frozen_v_mask = torch.tensor([int(i+1 in fv) for i in range(N)], dtype=torch.bool)

    model = SimpleMLP_Gmac()

    # Load from N=256 best (curriculum) or N=512 checkpoint if exists
    ckpt = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N512.pt')
    if os.path.exists(CKPT_BEST):
        ckpt = CKPT_BEST
    elif os.path.exists(CKPT_LATEST):
        ckpt = CKPT_LATEST
    elif os.path.exists(os.path.join(SAVE_DIR, 'n256_long_best.pt')):
        ckpt = os.path.join(SAVE_DIR, 'n256_long_best.pt')

    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    print(f'Loaded: {ckpt}', flush=True)
    print(f'Params: {model.count_parameters():,}', flush=True)
    print(f'N={N}, ku={KU}, kv={KV}, SC~{SC_BLER}', flush=True)
    print(f'Using C++ tree walk forward', flush=True)

    init_bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW)
    print(f'Initial BLER: {init_bler:.4f} (SC={SC_BLER})', flush=True)
    print(f'Batch={BATCH}, LR={LR}, Iters={TOTAL_ITERS}', flush=True)
    print(f'Checkpoints every {SAVE_EVERY} iters', flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    best_bler = init_bler

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        lr_now = get_lr(it)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
        ut = torch.from_numpy(uf).float()
        vt = torch.from_numpy(vf).float()

        root = model.z_encoder(zf.unsqueeze(-1))[:, br]
        cl_w, cr_w, pg_w, pc_w, ps_w, el_w, le_w = get_all_weights(model)
        result = fast_walk.tree_walk_forward(root, schedule_flat, model.tree.no_info_emb,
            cl_w, cr_w, pg_w, pc_w, ps_w, el_w, le_w, ut, vt, frozen_u_mask, frozen_v_mask)

        logits_all, targets_all = result[0], result[1]
        if logits_all.numel() > 0:
            loss = F.cross_entropy(logits_all.reshape(-1, 4), targets_all.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        if it % SAVE_EVERY == 0:
            torch.save(model.state_dict(), CKPT_LATEST)

        if it % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-500:])
            bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), CKPT_BEST)
                improved = ' *BEST*'

            ratio = bler / max(SC_BLER, 1e-8)
            msg = (f'[{it:>6}/{TOTAL_ITERS}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}, SC={SC_BLER}, ratio={ratio:.1f}x) '
                   f'{elapsed/60:.0f}min lr={lr_now:.1e}{improved}')
            print(msg, flush=True)
            with open(LOG_FILE, 'a') as f:
                f.write(msg + '\n')

    elapsed = time.time() - t0
    print(f'\nDONE: best={best_bler:.4f} (SC={SC_BLER}), {elapsed/3600:.1f}hr', flush=True)


if __name__ == '__main__':
    main()
