"""
Compare two NPD checkpoints by running both on the same evaluation set.

Useful for:
  - Checking if curriculum-trained model beats fresh-trained model
  - Comparing models trained at different iterations of the same run
  - Comparing different architectures or training setups

Usage:
  python -u class_c_npd/eval/compare_checkpoints.py \
    --ckpts ckpt1.pt ckpt2.pt --n 5 --snr 6.0 --n_cw 5000
"""
from __future__ import annotations
import os
import sys
import math
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design
from class_c_npd.training.train_stage import generate_stage1_batch, generate_stage2_batch
from class_c_npd.eval.chain_eval import wilson_ci


def load_npd(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model = NPDSingleUser(d=ckpt['d'], hidden=ckpt['hidden'],
                          n_layers=ckpt['n_layers'], z_dim=ckpt['z_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt


def evaluate_npd_stage(model, ckpt_meta, channel_name, n, snr_db, ku, kv, n_cw, seed=999):
    N = 1 << n
    channel = build_channel(channel_name,
                             sigma2=10 ** (-snr_db / 10) if channel_name == 'gmac' else None)
    Au, Av, fu, fv, _, _ = load_class_c_design(channel_name, n, snr_db=snr_db, ku=ku, kv=kv)

    stage = ckpt_meta.get('stage', 1)
    if stage == 1:
        gen_fn = generate_stage1_batch
        info = Au; frozen = fu; other = Av
    else:
        gen_fn = generate_stage2_batch
        info = Av; frozen = fv; other = Au

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    bs = 32
    total = 0

    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            u_true, features_npd, _ = gen_fn(channel, N, info, actual, rng, other)
            ft = torch.from_numpy(features_npd).float()
            if ft.dim() == 2:
                ft = ft.unsqueeze(-1)
            emb = model.encode_channel(ft)
            u_dec = model.decode(emb, frozen)
            for i in range(actual):
                if any(u_dec[i, p - 1].item() != u_true[i, p - 1] for p in info):
                    errs += 1
            total += actual

    bler = errs / n_cw
    ci_low, ci_high = wilson_ci(errs, n_cw)
    return {
        'bler': bler, 'errs': errs, 'n_cw': n_cw,
        'ci_low': ci_low, 'ci_high': ci_high,
        'stage': stage,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', type=str, nargs='+', required=True)
    parser.add_argument('--channel', type=str, default='gmac')
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--snr', type=float, default=6.0)
    parser.add_argument('--ku', type=int, required=True)
    parser.add_argument('--kv', type=int, required=True)
    parser.add_argument('--n_cw', type=int, default=5000)
    args = parser.parse_args()

    print(f'Comparing {len(args.ckpts)} checkpoints')
    print(f'  channel={args.channel}, N={1<<args.n}, SNR={args.snr}, ku={args.ku}, kv={args.kv}')
    print(f'  n_cw={args.n_cw}')
    print()

    results = []
    for ckpt_path in args.ckpts:
        if not os.path.exists(ckpt_path):
            print(f'  SKIP {ckpt_path} (not found)')
            continue
        print(f'  Evaluating {os.path.basename(ckpt_path)}...')
        model, meta = load_npd(ckpt_path)
        r = evaluate_npd_stage(model, meta, args.channel, args.n, args.snr,
                                args.ku, args.kv, args.n_cw)
        r['name'] = os.path.basename(ckpt_path)
        r['stage'] = meta.get('stage', '?')
        results.append(r)
        print(f'    BLER={r["bler"]:.4f}  CI=[{r["ci_low"]:.4f},{r["ci_high"]:.4f}]  '
              f'errs={r["errs"]}/{r["n_cw"]}  stage={r["stage"]}')

    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'{"checkpoint":<40}{"stage":<8}{"BLER":<10}{"95% CI":<22}')
    print('-' * 70)
    for r in results:
        ci = f'[{r["ci_low"]:.4f},{r["ci_high"]:.4f}]'
        print(f'{r["name"][:38]:<40}{r["stage"]:<8}{r["bler"]:<10.4f}{ci:<22}')


if __name__ == '__main__':
    main()
