#!/usr/bin/env python3
"""
train_npd_memory_mac_N128.py
============================
Scale chained NPD for ISI-MAC Class C to N=128 using the BiGRU E^W encoder.

- Warm-start Stage 1 and Stage 2 from the N=64 BiGRU best checkpoints
  (shape-compatible — z_encoder.gru is independent of N, tree weights
  are input-dim-only). The warm-start copies weights directly.
- Stage 1: 60K iters, batch 8, lr 5e-4 cosine-decayed to 1e-4.
  Checkpoint every 5K iters; best kept separately.
- Stage 2: 40K iters (converges quickly with teacher forcing), batch 8.
- Rates: ku=30, kv=58 per GMAC_C n=7 snr=6dB design proxy.
- Final eval: 2000 codewords chained, Wilson CI.

Saves to class_c_npd/results/npd_memory_mac/isi_mac_bigru_L1_{s1,s2}_N128_*.pt
and result JSON to class_c_npd/results/npd_memory_mac/isi_mac_bigru_N128_results.json.

Usage:
    python scripts/train_npd_memory_mac_N128.py
    python scripts/train_npd_memory_mac_N128.py --stage1_iters 80000 --stage2_iters 40000
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

# Reuse helpers from the main training driver
from scripts.train_npd_memory_mac import (
    make_channel, train_stage1, train_stage2,
    eval_stage1, eval_stage2_with_true_x, eval_chained,
    SNR_DB, ISI_H, RESULTS_DIR,
)
from polar.design_mc import design_from_file
from neural.npd_memory_mac import ChainedNPD_MAC


def load_design_n128(ku: int = 30, kv: int = 58):
    """Load GMAC_C n=7 snr=6dB design as proxy for ISI-MAC N=128."""
    n = 7
    N = 2 ** n
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, _pu, _pv, _path_i = design_from_file(
        path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_1idx = {p: 0 for p in range(1, N + 1) if p not in Au}
    frozen_v_1idx = {p: 0 for p in range(1, N + 1) if p not in Av}
    fu_set = {p - 1 for p in frozen_u_1idx.keys()}
    fv_set = {p - 1 for p in frozen_v_1idx.keys()}
    return Au, Av, frozen_u_1idx, frozen_v_1idx, fu_set, fv_set


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_iters', type=int, default=60000)
    p.add_argument('--stage2_iters', type=int, default=40000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--gru_layers', type=int, default=1)
    p.add_argument('--ku', type=int, default=30)
    p.add_argument('--kv', type=int, default=58)
    p.add_argument('--eval_cw', type=int, default=200)
    p.add_argument('--final_cw', type=int, default=2000)
    p.add_argument('--no_warm', action='store_true',
                   help='Disable warm-start from N=64 BiGRU checkpoints')
    p.add_argument('--skip_stage1', action='store_true',
                   help='Skip Stage 1 training (e.g. already done, load best)')
    p.add_argument('--skip_stage2', action='store_true',
                   help='Skip Stage 2 training')
    p.add_argument('--save_json', type=str,
                   default=os.path.join(RESULTS_DIR, 'isi_mac_bigru_N128_results.json'))
    args = p.parse_args()

    N = 128
    n = 7
    ku, kv = args.ku, args.kv
    channel, ch_meta = make_channel('isi_mac')
    Au, Av, fu_1idx, fv_1idx, fu_set, fv_set = load_design_n128(ku, kv)

    tag_base = 'isi_mac_bigru_L1'
    s1_tag = f'{tag_base}_s1_N{N}'
    s2_tag = f'{tag_base}_s2_N{N}'
    log_file = os.path.join(RESULTS_DIR, f'{tag_base}_N{N}.log')

    with open(log_file, 'a') as lf:
        lf.write(f'\n=== {tag_base} N={N} started {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        lf.write(f'channel: isi_mac meta={ch_meta}\n')
        lf.write(f'ku={ku} kv={kv} Ru={ku/N:.3f} Rv={kv/N:.3f}\n')
        lf.write(f'args: stage1_iters={args.stage1_iters} stage2_iters={args.stage2_iters} '
                 f'batch={args.batch} lr={args.lr} d={args.d} hidden={args.hidden} '
                 f'n_layers={args.n_layers} gru_layers={args.gru_layers}\n')

    # Model
    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=args.d, hidden=args.hidden, n_layers=args.n_layers,
                           encoder_type='bigru', gru_layers=args.gru_layers)
    n_params = model.count_parameters()

    # Warm-start from N=64 BiGRU bests (same architecture — only differs in N)
    warm_s1 = os.path.join(RESULTS_DIR, f'{tag_base}_s1_N64_best.pt')
    warm_s2 = os.path.join(RESULTS_DIR, f'{tag_base}_s2_N64_best.pt')
    if not args.no_warm:
        if os.path.exists(warm_s1):
            try:
                sd = torch.load(warm_s1, weights_only=False, map_location='cpu')
                model.stage1.load_state_dict(sd['state_dict'])
                msg = f'  warm-start stage1 from {os.path.basename(warm_s1)}'
                print(msg, flush=True)
                with open(log_file, 'a') as lf:
                    lf.write(msg + '\n')
            except Exception as e:
                print(f'  warm-start stage1 FAILED: {e}', flush=True)
        else:
            print(f'  warm_s1 not found at {warm_s1}', flush=True)

        if os.path.exists(warm_s2):
            try:
                sd = torch.load(warm_s2, weights_only=False, map_location='cpu')
                model.stage2.load_state_dict(sd['state_dict'])
                msg = f'  warm-start stage2 from {os.path.basename(warm_s2)}'
                print(msg, flush=True)
                with open(log_file, 'a') as lf:
                    lf.write(msg + '\n')
            except Exception as e:
                print(f'  warm-start stage2 FAILED: {e}', flush=True)

    print(f'\n{"="*60}\nN={N} encoder=bigru L={args.gru_layers} '
          f'params={n_params:,}\n{"="*60}', flush=True)

    # Sanity: pre-training eval
    pre_bler_u = eval_stage1(model, channel, N, Au, Av, fu_set, n_cw=200, seed=999)
    msg = f'  pre-training Stage 1 BLER (after warm-start): {pre_bler_u:.4f}'
    print(msg, flush=True)
    with open(log_file, 'a') as lf:
        lf.write(msg + '\n')

    # ─── Stage 1 ─────────────────────────────────────────────
    s1_time = 0.0
    s1_best = 1.0
    if not args.skip_stage1:
        t0 = time.time()
        s1_best = train_stage1(
            model, channel, N, Au, Av, fu_set,
            iters=args.stage1_iters, batch=args.batch, lr=args.lr,
            ckpt_base=RESULTS_DIR, tag=s1_tag, log_file=log_file,
            eval_every=2000, eval_cw=args.eval_cw,
        )
        s1_time = (time.time() - t0) / 60
        print(f'\n  Stage 1 best BLER: {s1_best:.4f}  ({s1_time:.1f} min)', flush=True)
    else:
        print('  Skipping Stage 1 training (flag set).', flush=True)

    # Reload best Stage 1 before Stage 2 training
    s1_best_ckpt = os.path.join(RESULTS_DIR, f'{s1_tag}_best.pt')
    if os.path.exists(s1_best_ckpt):
        sd = torch.load(s1_best_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])
        if args.skip_stage1:
            s1_best = eval_stage1(model, channel, N, Au, Av, fu_set,
                                  n_cw=args.eval_cw, seed=999)
            print(f'  Reloaded Stage 1 best. Eval BLER={s1_best:.4f}', flush=True)

    # ─── Stage 2 ─────────────────────────────────────────────
    s2_time = 0.0
    s2_best = 1.0
    if not args.skip_stage2:
        t1 = time.time()
        s2_best = train_stage2(
            model, channel, N, Au, Av, fv_set,
            iters=args.stage2_iters, batch=args.batch, lr=args.lr,
            ckpt_base=RESULTS_DIR, tag=s2_tag, log_file=log_file,
            eval_every=2000, eval_cw=args.eval_cw,
        )
        s2_time = (time.time() - t1) / 60
        print(f'\n  Stage 2 best BLER(V|true X): {s2_best:.4f}  ({s2_time:.1f} min)', flush=True)
    else:
        print('  Skipping Stage 2 training.', flush=True)

    # Reload best Stage 2
    s2_best_ckpt = os.path.join(RESULTS_DIR, f'{s2_tag}_best.pt')
    if os.path.exists(s2_best_ckpt):
        sd = torch.load(s2_best_ckpt, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd['state_dict'])

    # ─── Chained inference ─────────────────────────────────────
    print(f'\n  Chained inference ({args.final_cw} codewords)...', flush=True)
    t2 = time.time()
    chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set,
                           n_cw=args.final_cw, seed=777)
    chained_time = (time.time() - t2) / 60
    print(f'  chained BLER={chained["bler_total"]:.4f} '
          f'(U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f}) '
          f'({chained_time:.1f} min)', flush=True)

    # Wilson CI for chained BLER
    n_cw = chained['n_cw']
    errs = chained['errs_total']
    p_hat = errs / n_cw
    z = 1.96
    denom = 1 + z*z / n_cw
    center = (p_hat + z*z / (2*n_cw)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n_cw + z*z / (4 * n_cw*n_cw)) / denom
    ci_lo = max(0.0, center - margin)
    ci_hi = min(1.0, center + margin)
    print(f'  chained BLER 95% Wilson CI: [{ci_lo:.4f}, {ci_hi:.4f}]', flush=True)

    result = {
        'channel': 'isi_mac', 'channel_meta': ch_meta,
        'N': N, 'ku': ku, 'kv': kv,
        'encoder': 'bigru', 'gru_layers': args.gru_layers,
        'd': args.d, 'hidden': args.hidden, 'n_layers': args.n_layers,
        'stage1_iters': args.stage1_iters, 'stage2_iters': args.stage2_iters,
        'batch': args.batch, 'lr': args.lr,
        'stage1_best_bler': float(s1_best),
        'stage2_best_bler_true_x': float(s2_best),
        'chained': {k: (float(v) if isinstance(v, (float, int)) else v)
                    for k, v in chained.items()},
        'chained_wilson_95ci': [float(ci_lo), float(ci_hi)],
        'stage1_time_min': s1_time,
        'stage2_time_min': s2_time,
        's1_ckpt': s1_best_ckpt, 's2_ckpt': s2_best_ckpt,
        'n_params': n_params,
        'warm_start': not args.no_warm,
        'pre_training_s1_bler': float(pre_bler_u),
    }

    with open(args.save_json, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\n  Results saved to {args.save_json}', flush=True)


if __name__ == '__main__':
    main()
