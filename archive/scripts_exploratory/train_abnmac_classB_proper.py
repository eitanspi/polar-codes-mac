#!/usr/bin/env python3
"""
train_abnmac_classB_proper.py — Full curriculum training for ABNMAC Class B NCG.

Prior attempts failed because:
  1. The v1 script used only 5K/15K/30K iters (needs 10x more)
  2. The v2 script trained Class C with a bad analytical design

This script uses:
  - Class B path (interleaved, path_i = N//2)
  - MC-based designs from designs/abnmac_B_n{n}.npz
  - Long curriculum: N=8 (15K) → N=16 (30K) → N=32 (80K) → N=64 (150K)
  - Cosine LR schedule per stage
  - Periodic eval with SC reference
  - Early checkpoint saving on best BLER

Target: SC BLER at these rates is ~0.012 at N=32, ~0.038 at N=64.
If NCG matches SC, this would be the first working ABNMAC NCG.

Output: saved_models/ncg_abnmac_classB_N{N}_best.pt
Log:    /tmp/abnmac_classB_proper.log
"""

import os, sys, math, time, json
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

D = 16; HIDDEN = 64; N_LAYERS = 2
BATCH = 32
LR = 3e-4
EVAL_EVERY = 2000
EVAL_CW = 300

CURRICULUM = [
    {'N': 8,   'iters': 15000,  'ku': 3,  'kv': 3},
    {'N': 16,  'iters': 30000,  'ku': 5,  'kv': 5},
    {'N': 32,  'iters': 80000,  'ku': 10, 'kv': 10},
    {'N': 64,  'iters': 150000, 'ku': 22, 'kv': 22},
]


def encode_z(zf):
    out = np.empty(zf.shape, dtype=np.int64)
    for idx in np.ndindex(zf.shape):
        zx, zy = zf[idx]; out[idx] = 2*int(zx) + int(zy)
    return out


def evaluate_nn(model, channel, b, Au, Av, fu, fv, N, n_cw, batch_size=25):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            zt = torch.from_numpy(encode_z(zf)).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / max(total, 1)


def train_stage(model, channel, stage, prev_ckpt=None):
    N = stage['N']; n = int(math.log2(N))
    iters = stage['iters']; ku = stage['ku']; kv = stage['kv']
    path_i = N // 2
    b = make_path(N, path_i)

    design_file = os.path.join(BASE, 'designs', f'abnmac_B_n{n}.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, ku, kv)

    print(f"\n{'='*70}", flush=True)
    print(f"  Stage N={N}, Class B, ku={ku}, kv={kv}, iters={iters}", flush=True)
    print(f"{'='*70}", flush=True)

    if prev_ckpt and os.path.exists(prev_ckpt):
        sd = torch.load(prev_ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded: {os.path.basename(prev_ckpt)}", flush=True)

    init_bler = evaluate_nn(model, channel, b, Au, Av, fu, fv, N, EVAL_CW)
    print(f"  Initial NN BLER: {init_bler:.4f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=LR * 0.01)

    rng = np.random.default_rng()
    t0 = time.time()
    best_bler = init_bler
    best_path = os.path.join(BASE, 'saved_models', f'ncg_abnmac_classB_N{N}_best.pt')
    losses = []

    model.train()
    for it in range(1, iters + 1):
        uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        zt = torch.from_numpy(encode_z(zf)).long()
        ut = torch.from_numpy(uf).float(); vt = torch.from_numpy(vf).float()

        logits, targets, _, _, _ = model(zt, b, fu, fv, u_true=ut, v_true=vt)
        if not logits:
            continue
        loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4),
                               torch.stack(targets).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if it % EVAL_EVERY == 0 or it == iters:
            bler = evaluate_nn(model, channel, b, Au, Av, fu, fv, N, EVAL_CW)
            recent = float(np.mean(losses[-1000:])) if losses else 0
            elapsed = (time.time() - t0) / 60
            improved = ""
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), best_path)
                improved = " *BEST*"
            print(f"  [{it:7d}/{iters}] loss={loss.item():.4f} (avg={recent:.4f}) "
                  f"BLER={bler:.4f} best={best_bler:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {elapsed:.1f}min{improved}",
                  flush=True)

    final_bler = evaluate_nn(model, channel, b, Au, Av, fu, fv, N, max(EVAL_CW, 500))
    print(f"  Stage N={N} done. Final BLER={final_bler:.4f} Best={best_bler:.4f} "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    if best_bler < 1.0:
        torch.save(model.state_dict(), best_path)
    return best_path, best_bler


def main():
    channel = ABNMAC()
    model = PureNeuralCompGraphDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS, vocab_size=4)
    print(f"\n  ABNMAC Class B proper curriculum training")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Curriculum: {[(s['N'], s['iters']) for s in CURRICULUM]}")
    print(f"  Total iters: {sum(s['iters'] for s in CURRICULUM):,}", flush=True)

    results = {'curriculum': CURRICULUM, 'stages': []}
    prev_ckpt = None
    for stage in CURRICULUM:
        ckpt, best_bler = train_stage(model, channel, stage, prev_ckpt)
        results['stages'].append({
            'N': stage['N'], 'ku': stage['ku'], 'kv': stage['kv'],
            'iters': stage['iters'], 'best_bler': best_bler, 'ckpt': ckpt
        })
        prev_ckpt = ckpt

        out = os.path.join(BASE, 'results', 'crc_scl_expansion',
                            'abnmac_classB_training.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)

        if best_bler >= 0.99:
            print(f"\n  WARNING: N={stage['N']} did not converge (BLER={best_bler:.3f}). "
                  f"Continuing curriculum anyway — transfer might help.", flush=True)

    print(f"\n  Training complete. Saved: {out}", flush=True)


if __name__ == '__main__':
    main()
