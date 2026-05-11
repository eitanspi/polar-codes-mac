#!/usr/bin/env python3
"""
train_gmac_continue_allN.py — Continue training d=16 GMAC model at N=256, 512, 1024.

Same approach that worked at N=128 (0.027 → 0.019, 1.69x → 1.17x SC):
stable cosine decay lr, no warm restarts, long training budget.

The 48hr run used cosine warm restarts which disrupted learning.
This script loads best checkpoints and trains with stable lr.
"""
import sys, os, math, time, json, gc
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

# ─── Config ──────────────────────────────────────────────────────────────────

D = 16
HIDDEN = 64
N_LAYERS = 2

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'train_gmac_continue_allN_results.json')

# Each N gets its own budget — larger N = fewer iters (slower per iter)
CONFIGS = [
    {'N': 256,  'ku': 123, 'kv': 123, 'sc_bler': 0.005,
     'batch': 8,  'lr': 5e-5, 'iters': 100000, 'eval_every': 5000, 'eval_cw': 2000},
    {'N': 512,  'ku': 246, 'kv': 246, 'sc_bler': 0.001,
     'batch': 4,  'lr': 3e-5, 'iters': 50000,  'eval_every': 5000, 'eval_cw': 1000},
    {'N': 1024, 'ku': 492, 'kv': 492, 'sc_bler': 0.001,
     'batch': 2,  'lr': 2e-5, 'iters': 20000,  'eval_every': 5000, 'eval_cw': 500},
]


class SimpleMLP_Gmac(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2, z_hidden=32):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

    def forward(self, z, b, fu, fv, u_true=None, v_true=None):
        n = z.shape[1].bit_length() - 1
        br = torch.from_numpy(bit_reversal_perm(n)).long()
        root = self.z_encoder(z.unsqueeze(-1))[:, br]
        return self.tree(z=None, b=b, frozen_u=fu, frozen_v=fv,
                        u_true=u_true, v_true=v_true, root_emb=root)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
    return Au, Av, fu, fv


def evaluate(model, channel, N, b, Au, Av, fu, fv, n_cw):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    bs = max(1, min(8, 64 // max(1, N // 32)))
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


def get_lr(it, total, base_lr, warmup=2000):
    if it < warmup:
        return base_lr * it / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def train_one_N(cfg, results):
    N = cfg['N']
    ku, kv = cfg['ku'], cfg['kv']
    sc_bler = cfg['sc_bler']
    batch = cfg['batch']
    lr = cfg['lr']
    n_iters = cfg['iters']
    eval_every = cfg['eval_every']
    eval_cw = cfg['eval_cw']

    channel = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)  # Class B
    Au, Av, fu, fv = load_design(N, ku, kv)

    model = SimpleMLP_Gmac(d=D, hidden=HIDDEN, n_layers=N_LAYERS)

    # Load checkpoint
    ckpt_path = os.path.join(SAVE_DIR, f'ncg_gmac_mlp_N{N}.pt')
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, weights_only=True)
        fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
        model.load_state_dict(fixed, strict=False)
        print(f'  Loaded: {ckpt_path}', flush=True)
    else:
        print(f'  WARNING: no checkpoint for N={N}', flush=True)

    print(f'\n{"="*60}', flush=True)
    print(f'  N={N}, ku={ku}, kv={kv}, SC={sc_bler}', flush=True)
    print(f'  params={model.count_parameters():,}', flush=True)
    print(f'  batch={batch}, lr={lr}, iters={n_iters}', flush=True)
    print(f'  {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'{"="*60}', flush=True)

    init_bler = evaluate(model, channel, N, b, Au, Av, fu, fv, eval_cw)
    print(f'  Initial BLER: {init_bler:.4f} (SC={sc_bler}, ratio={init_bler/max(sc_bler,1e-8):.1f}x)', flush=True)

    best_bler = init_bler
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng()
    t0 = time.time()
    losses = []

    model.train()
    for it in range(1, n_iters + 1):
        lr_now = get_lr(it, n_iters, lr)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        u = np.zeros((batch, N), dtype=int); v = np.zeros((batch, N), dtype=int)
        for p in Au: u[:, p-1] = rng.integers(0, 2, batch)
        for p in Av: v[:, p-1] = rng.integers(0, 2, batch)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(channel.sample_batch(x, y)).float()

        logits, targets, _, _, _ = model(z, b, fu, fv,
            u_true=torch.from_numpy(u).float(),
            v_true=torch.from_numpy(v).float())

        if logits:
            loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4),
                                   torch.stack(targets).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        if it % eval_every == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-min(len(losses), 500):])
            bler = evaluate(model, channel, N, b, Au, Av, fu, fv, eval_cw)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(SAVE_DIR, f'ncg_gmac_mlp_N{N}.pt'))
                improved = ' *BEST*'

            ratio = bler / max(sc_bler, 1e-8)
            print(f'  [{it:>6}/{n_iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}, SC={sc_bler}, ratio={ratio:.1f}x) '
                  f'{elapsed/60:.0f}min lr={lr_now:.1e}{improved}', flush=True)

    model.load_state_dict(best_state)
    final_bler = evaluate(model, channel, N, b, Au, Av, fu, fv, eval_cw * 2)
    total_time = time.time() - t0

    print(f'\n  N={N} DONE: best={best_bler:.4f}, final={final_bler:.4f}, '
          f'{total_time/3600:.1f}hr', flush=True)

    results[str(N)] = {
        'N': N, 'ku': ku, 'kv': kv,
        'best_bler': best_bler, 'final_bler': final_bler,
        'init_bler': init_bler,
        'sc_bler': sc_bler,
        'ratio': best_bler / max(sc_bler, 1e-8),
        'iters': n_iters, 'time_hr': total_time / 3600,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    del model; gc.collect()


def main():
    print('=' * 60, flush=True)
    print('Continue d=16 GMAC training — N=256, 512, 1024', flush=True)
    print(f'Stable cosine lr (no warm restarts)', flush=True)
    print(f'SNR={SNR_DB}dB, Class B, MC design', flush=True)
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print('=' * 60, flush=True)

    results = {}
    for cfg in CONFIGS:
        train_one_N(cfg, results)

    print(f'\n{"="*60}', flush=True)
    print('ALL DONE', flush=True)
    for k, r in sorted(results.items(), key=lambda x: int(x[0])):
        print(f'  N={r["N"]}: {r["init_bler"]:.4f} -> {r["best_bler"]:.4f} '
              f'(SC={r["sc_bler"]}, ratio={r["ratio"]:.1f}x, {r["time_hr"]:.1f}hr)',
              flush=True)


if __name__ == '__main__':
    main()
