"""
Multi-channel curriculum sweep for Class C NPD.

Runs the same N-curriculum across all supported channels:
  - GMAC at SNR=6dB (Gaussian noise)
  - BEMAC (binary erasure-style; Stage 2 trivially zero)
  - ABNMAC (asymmetric binary noise)
  - ISI-MAC (when designs available)

For each channel, trains Stage 1 + Stage 2 NPDs at each N in the
curriculum, with warm-starting between Ns. Reports a unified table
of (channel, N, SC BLER, NPD BLER, ratio).

This is the "headline" experiment — show that the chained NPD approach
inherits the proven scalability of single-user NPDs across multiple
MAC channels.
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
import torch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.training.train_stage import (
    generate_stage1_batch, generate_stage2_batch, evaluate_stage,
)
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design, design_summary
from class_c_npd.eval.chain_eval import chain_evaluate


# ─── Channel-specific configs ────────────────────────────────────────────────

CHANNELS = {
    'gmac': {
        'kwargs': {'sigma2': 10 ** (-6.0 / 10)},
        'snr_db': 6.0,
        'design_supports_n': [4, 5, 6, 7, 8, 9, 10],
        'description': 'Gaussian MAC at SNR=6dB',
        'capacity_per_user': (0.4645, 0.9119),  # I(X;Z), I(Y;Z|X)
    },
    'bemac': {
        'kwargs': {},
        'snr_db': None,
        'design_supports_n': [3, 4, 5, 6, 7, 8, 9, 10],
        'description': 'Binary erasure MAC (Stage 2 trivially zero)',
        'capacity_per_user': (0.5, 1.0),
    },
    'abnmac': {
        'kwargs': {'p_x': 0.1, 'p_y': 0.1},
        'snr_db': None,
        'design_supports_n': [3, 4, 5, 6, 7, 8, 9, 10],
        'description': 'Asymmetric binary noise MAC, p_x=p_y=0.1',
        'capacity_per_user': None,  # would need calc
    },
}


# Standard curriculum schedule
CURRICULUM = {
    16:  {'s1_iters': 30000, 's1_batch': 64, 's1_lr': 3e-4,
          's2_iters': 8000,  's2_batch': 64, 's2_lr': 3e-4},
    32:  {'s1_iters': 50000, 's1_batch': 64, 's1_lr': 2e-4,
          's2_iters': 12000, 's2_batch': 64, 's2_lr': 3e-4},
    64:  {'s1_iters': 80000, 's1_batch': 32, 's1_lr': 1e-4,
          's2_iters': 20000, 's2_batch': 32, 's2_lr': 2e-4},
    128: {'s1_iters': 120000, 's1_batch': 16, 's1_lr': 1e-4,
          's2_iters': 30000,  's2_batch': 16, 's2_lr': 1e-4},
}


def warm_start(model, ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    try:
        model.load_state_dict(ckpt['state_dict'])
        return True
    except Exception:
        return False


def train_one_stage(stage, channel_name, n, ku, kv, cfg, warm_from, tag):
    N = 1 << n
    chan_cfg = CHANNELS[channel_name]
    channel = build_channel(channel_name, **chan_cfg['kwargs'])

    Au, Av, fu, fv, pe_u, pe_v = load_class_c_design(
        channel_name, n, snr_db=chan_cfg['snr_db'] or 6.0,
        ku=ku, kv=kv,
    )
    print(f'  ' + design_summary(Au, Av, pe_u, pe_v).replace('\n', '\n  '))

    if stage == 1:
        gen_fn = generate_stage1_batch
        info = Au; frozen = fu; other = Av
        z_dim = channel.stage1_feature_dim
    else:
        gen_fn = generate_stage2_batch
        info = Av; frozen = fv; other = Au
        z_dim = channel.stage2_feature_dim

    torch.manual_seed(42 if stage == 1 else 43)
    model = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=z_dim)
    if warm_start(model, warm_from):
        print(f'  warm-started')

    iters = cfg[f's{stage}_iters']
    batch = cfg[f's{stage}_batch']
    lr = cfg[f's{stage}_lr']
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rng = np.random.default_rng(42 if stage == 1 else 43)
    t0 = time.time()
    losses = []
    best_bler = 1.0
    best_iter = 0
    save_dir = os.path.join(_ROOT, 'class_c_npd', 'results')
    ckpt_path = os.path.join(save_dir, f'{tag}_best.pt')

    eval_every = max(2000, iters // 12)

    model.train()
    for it in range(1, iters + 1):
        _, features_npd, cw_npd = gen_fn(channel, N, info, batch, rng, other)
        ft = torch.from_numpy(features_npd).float()
        if ft.dim() == 2:
            ft = ft.unsqueeze(-1)
        cw = torch.from_numpy(cw_npd).long()

        emb = model.encode_channel(ft)
        loss = model.fast_ce(emb, cw)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % eval_every == 0:
            bler = evaluate_stage(model, channel, N, info, frozen, gen_fn, 500, seed=999, other_info=other)
            avg_loss = float(np.mean(losses[-500:]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                best_iter = it
                torch.save({
                    'state_dict': model.state_dict(),
                    'd': 16, 'hidden': 64, 'n_layers': 2, 'z_dim': z_dim,
                    'channel': channel_name, 'stage': stage, 'N': N,
                    'Au': Au, 'Av': Av,
                }, ckpt_path)
                marker = ' *'
            print(f'  [{it:>6}/{iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}', flush=True)

    elapsed = (time.time() - t0) / 60
    return ckpt_path, best_bler, elapsed


def run_channel_sweep(channel_name, N_list):
    print('\n' + '#' * 70)
    print(f'# CHANNEL: {channel_name}')
    print(f'# {CHANNELS[channel_name]["description"]}')
    print('#' * 70)

    chan_cfg = CHANNELS[channel_name]
    snr_db = chan_cfg['snr_db'] or 6.0

    # Determine ku, kv per N from capacity (50% of capacity per user)
    if chan_cfg['capacity_per_user'] is not None:
        cap_u, cap_v = chan_cfg['capacity_per_user']
        get_ku_kv = lambda N: (max(1, round(0.5 * cap_u * N)),
                                max(1, round(0.5 * cap_v * N)))
    else:
        # Fallback: use pe_threshold
        get_ku_kv = lambda N: (None, None)

    results = {}
    prev_s1, prev_s2 = None, None
    t_total = time.time()

    for N in N_list:
        n = int(np.log2(N))
        if n not in chan_cfg['design_supports_n']:
            print(f'\n  Skipping N={N}: no design file')
            continue

        ku, kv = get_ku_kv(N)
        cfg = CURRICULUM.get(N)
        if cfg is None:
            cfg = CURRICULUM[max(CURRICULUM.keys())]  # use largest schedule
        print(f'\n=== N={N} | ku={ku} kv={kv} ===')

        # Stage 1
        print(f'-- Stage 1 --')
        s1_ckpt, s1_bler, s1_time = train_one_stage(
            stage=1, channel_name=channel_name, n=n, ku=ku, kv=kv, cfg=cfg,
            warm_from=prev_s1, tag=f'mc_{channel_name}_s1_N{N}',
        )
        # Stage 2
        print(f'-- Stage 2 --')
        s2_ckpt, s2_bler, s2_time = train_one_stage(
            stage=2, channel_name=channel_name, n=n, ku=ku, kv=kv, cfg=cfg,
            warm_from=prev_s2, tag=f'mc_{channel_name}_s2_N{N}',
        )

        # Chained eval
        print(f'-- Chained eval --')
        chain_results = chain_evaluate(
            stage1_ckpt=s1_ckpt, stage2_ckpt=s2_ckpt,
            channel_name=channel_name, n=n, snr_db=snr_db,
            ku=ku, kv=kv,
            n_cw=2000, batch=16, seed=999, verbose=False,
        )
        bler = chain_results['bler_total']
        print(f'  Chained BLER: {bler:.4f}')

        results[N] = {
            'channel': channel_name, 'N': N, 'ku': ku, 'kv': kv,
            'stage1_bler': float(s1_bler), 'stage1_time_min': float(s1_time),
            'stage2_bler': float(s2_bler), 'stage2_time_min': float(s2_time),
            'chained_bler': float(bler),
            'chained_ci_low': float(chain_results['ci_low']),
            'chained_ci_high': float(chain_results['ci_high']),
            'chained_errs_u': int(chain_results['errs_u']),
            'chained_errs_v': int(chain_results['errs_v']),
        }

        # Save incrementally
        out_path = os.path.join(_ROOT, 'class_c_npd', 'results',
                                f'multi_channel_{channel_name}.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        prev_s1, prev_s2 = s1_ckpt, s2_ckpt

    total_min = (time.time() - t_total) / 60
    print(f'\n{channel_name}: total wall {total_min:.1f} min')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', type=str, default='all',
                        choices=['all', 'gmac', 'bemac', 'abnmac'])
    parser.add_argument('--N_list', type=str, default='16,32,64')
    args = parser.parse_args()

    N_list = [int(x) for x in args.N_list.split(',')]

    channels = ['gmac', 'bemac', 'abnmac'] if args.channel == 'all' else [args.channel]

    print(f'Multi-channel Class C NPD sweep')
    print(f'Channels: {channels}')
    print(f'N values: {N_list}')
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    all_results = {}
    for ch in channels:
        if ch not in CHANNELS:
            print(f'Unknown channel: {ch}, skipping')
            continue
        all_results[ch] = run_channel_sweep(ch, N_list)

    # Final summary
    print('\n' + '=' * 70)
    print('MULTI-CHANNEL SUMMARY')
    print('=' * 70)
    for ch, ch_res in all_results.items():
        print(f'\n{ch}:')
        for N in sorted(ch_res.keys()):
            r = ch_res[N]
            print(f'  N={N}: chained BLER = {r["chained_bler"]:.4f} '
                  f'(CI [{r["chained_ci_low"]:.4f}, {r["chained_ci_high"]:.4f}])')

    print(f'\nFinished: {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
