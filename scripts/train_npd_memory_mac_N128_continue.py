#!/usr/bin/env python3
"""
Continue training the N=128 BiGRU Stage 1 from a checkpoint with lower lr.
Uses flat lr (no cosine decay) for fine-tuning.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from neural.npd_memory_mac import ChainedNPD_MAC

from scripts.train_npd_memory_mac import (
    make_channel, make_batch, eval_stage1, eval_chained,
    eval_stage2_with_true_x, SNR_DB, RESULTS_DIR,
)


def load_design_n128(ku=30, kv=58):
    n = 7; N = 128
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list); Av = sorted(Av_list)
    fu_1 = {p: 0 for p in range(1, N+1) if p not in Au}
    fv_1 = {p: 0 for p in range(1, N+1) if p not in Av}
    fu_set = {p-1 for p in fu_1}; fv_set = {p-1 for p in fv_1}
    return Au, Av, fu_1, fv_1, fu_set, fv_set


def train_stage1_flat_lr(model, channel, N, Au, Av, frozen_u_set,
                         iters, batch, lr, ckpt_base, tag, log_file,
                         eval_every=2000, eval_cw=200, seed=142):
    """Stage 1 training with flat lr (no cosine decay)."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()

    opt = torch.optim.AdamW(model.stage1.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    best_bler = 1.0
    losses = []
    t0 = time.time()

    model.stage1.train()
    for it in range(1, iters + 1):
        u_msg, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, batch, rng)
        z_t = torch.from_numpy(z)
        emb = model.stage1.encode_channel(z_t)
        emb_npd = emb[:, br_t, :]
        x_cw_npd = torch.from_numpy(x_phys[:, br]).long()

        loss = model.stage1.tree.fast_ce(emb_npd, x_cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % eval_every == 0 or it == iters:
            bler = eval_stage1(model, channel, N, Au, Av, frozen_u_set,
                              n_cw=eval_cw, seed=999)
            avg_loss = float(np.mean(losses[-min(200, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                ckpt_path = os.path.join(ckpt_base, f'{tag}_best.pt')
                torch.save({
                    'state_dict': model.stage1.state_dict(),
                    'N': N, 'Au': Au, 'Av': Av,
                }, ckpt_path)
                marker = ' *BEST*'
            if it % 5000 == 0:
                torch.save({
                    'state_dict': model.stage1.state_dict(),
                    'N': N, 'Au': Au, 'Av': Av,
                }, os.path.join(ckpt_base, f'{tag}_iter{it}.pt'))
            msg = (f'  [S1c {it:>6}/{iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}')
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
    return best_bler


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=100000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--gru_layers', type=int, default=1)
    p.add_argument('--eval_cw', type=int, default=300)
    p.add_argument('--final_cw', type=int, default=2000)
    p.add_argument('--ckpt', type=str, default=None,
                   help='Stage 1 checkpoint to load (default: d16 best)')
    p.add_argument('--s2_ckpt', type=str, default=None,
                   help='Stage 2 checkpoint to load (default: d16 best)')
    args = p.parse_args()

    N = 128
    channel, ch_meta = make_channel('isi_mac')
    Au, Av, fu_1, fv_1, fu_set, fv_set = load_design_n128()

    suffix = f'd{args.d}'
    tag = f'isi_mac_bigru_L1_cont_{suffix}'
    s1_tag = f'{tag}_s1_N{N}'
    log_file = os.path.join(RESULTS_DIR, f'{tag}_N{N}.log')

    with open(log_file, 'a') as lf:
        lf.write(f'\n=== CONTINUATION {tag} N={N} {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        lf.write(f'lr={args.lr} iters={args.iters} batch={args.batch}\n')

    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=args.d, hidden=args.hidden, n_layers=args.n_layers,
                           encoder_type='bigru', gru_layers=args.gru_layers)

    # Load S1 checkpoint
    s1_ckpt = args.ckpt or os.path.join(RESULTS_DIR,
        f'isi_mac_bigru_L1_s1_N{N}_best_d{args.d}.pt')
    if os.path.exists(s1_ckpt):
        sd = torch.load(s1_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])
        print(f'  Loaded S1 from {os.path.basename(s1_ckpt)}', flush=True)
    else:
        print(f'  WARNING: S1 ckpt not found at {s1_ckpt}', flush=True)

    # Load S2 checkpoint
    s2_ckpt = args.s2_ckpt or os.path.join(RESULTS_DIR,
        f'isi_mac_bigru_L1_s2_N{N}_best_d{args.d}.pt')
    if os.path.exists(s2_ckpt):
        sd = torch.load(s2_ckpt, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd['state_dict'])
        print(f'  Loaded S2 from {os.path.basename(s2_ckpt)}', flush=True)

    # Pre-eval
    pre_bler = eval_stage1(model, channel, N, Au, Av, fu_set, n_cw=300, seed=999)
    print(f'  Pre-continuation S1 BLER: {pre_bler:.4f}', flush=True)

    # Train Stage 1 only
    t0 = time.time()
    s1_best = train_stage1_flat_lr(
        model, channel, N, Au, Av, fu_set,
        iters=args.iters, batch=args.batch, lr=args.lr,
        ckpt_base=RESULTS_DIR, tag=s1_tag, log_file=log_file,
        eval_every=2000, eval_cw=args.eval_cw,
    )
    s1_time = (time.time() - t0) / 60
    print(f'\n  Stage 1 continuation best BLER: {s1_best:.4f} ({s1_time:.1f} min)', flush=True)

    # Reload best
    s1_best_path = os.path.join(RESULTS_DIR, f'{s1_tag}_best.pt')
    if os.path.exists(s1_best_path):
        sd = torch.load(s1_best_path, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])

    # Chained eval
    print(f'\n  Chained inference ({args.final_cw} codewords)...', flush=True)
    chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set,
                           n_cw=args.final_cw, seed=777)
    print(f'  chained BLER={chained["bler_total"]:.4f} '
          f'(U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f})', flush=True)

    # Wilson CI
    n_cw = chained['n_cw']; errs = chained['errs_total']
    p_hat = errs / n_cw; z = 1.96
    denom = 1 + z*z / n_cw
    center = (p_hat + z*z / (2*n_cw)) / denom
    margin = z * math.sqrt(p_hat * (1-p_hat)/n_cw + z*z/(4*n_cw*n_cw)) / denom
    print(f'  95% Wilson CI: [{max(0, center-margin):.4f}, {min(1, center+margin):.4f}]', flush=True)

    result = {
        'channel': 'isi_mac', 'N': N, 'ku': 30, 'kv': 58,
        'd': args.d, 'hidden': args.hidden, 'gru_layers': args.gru_layers,
        's1_best_bler': float(s1_best), 'pre_s1_bler': float(pre_bler),
        'chained': {k: float(v) if isinstance(v, (float, int)) else v
                    for k, v in chained.items()},
        's1_time_min': s1_time, 'lr': args.lr, 'iters': args.iters,
    }
    out_json = os.path.join(RESULTS_DIR, f'{tag}_N{N}_results.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved {out_json}', flush=True)


if __name__ == '__main__':
    main()
