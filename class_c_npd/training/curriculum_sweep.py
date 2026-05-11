"""
Curriculum sweep across N for GMAC Class C.

Trains Stage 1 and Stage 2 NPDs at N=16, 32, 64, 128, 256 (and beyond),
warm-starting each N from the previous N's checkpoint. The architecture
is N-independent (weight-shared tree ops), so warm-starting is sound.

Uses the FIXED training distribution (V matches inference for Stage 1,
X matches inference for Stage 2).

Each N gets a generous training budget. The full sweep runs unattended
and writes results to a single JSON.

Usage:
  python -u class_c_npd/training/curriculum_sweep.py
"""
from __future__ import annotations
import os
import sys
import time
import json
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
from class_c_npd.eval.chain_eval import chain_evaluate, wilson_ci


SNR_DB = 6.0

# SC reference at 50% capacity (from gmac_sc_reference_50pct.json)
SC_REF = {
    16:   {'ku': 4,   'kv': 7,   'sc_bler': 0.1626},
    32:   {'ku': 7,   'kv': 15,  'sc_bler': 0.0684},
    64:   {'ku': 15,  'kv': 29,  'sc_bler': 0.0266},
    128:  {'ku': 30,  'kv': 58,  'sc_bler': 0.0054},
    256:  {'ku': 59,  'kv': 117, 'sc_bler': 0.0020},
}

# Curriculum schedule — generous training at each N
CURRICULUM = {
    16:  {'s1_iters': 30000, 's1_batch': 64, 's1_lr': 3e-4,
          's2_iters': 10000, 's2_batch': 64, 's2_lr': 3e-4},
    32:  {'s1_iters': 50000, 's1_batch': 64, 's1_lr': 2e-4,
          's2_iters': 15000, 's2_batch': 64, 's2_lr': 3e-4},
    64:  {'s1_iters': 80000, 's1_batch': 32, 's1_lr': 1e-4,
          's2_iters': 25000, 's2_batch': 32, 's2_lr': 2e-4},
    128: {'s1_iters': 120000, 's1_batch': 16, 's1_lr': 1e-4,
          's2_iters': 40000,  's2_batch': 16, 's2_lr': 1e-4},
    256: {'s1_iters': 200000, 's1_batch': 8, 's1_lr': 5e-5,
          's2_iters': 60000,  's2_batch': 8, 's2_lr': 1e-4},
}

EVAL_CW = 5000


# ─── Save/load checkpoints ──────────────────────────────────────────────────

def save_ckpt(path, model, meta):
    torch.save({'state_dict': model.state_dict(), **meta}, path)

def load_ckpt(path):
    return torch.load(path, weights_only=False, map_location='cpu')


def warm_start(target_model: NPDSingleUser, source_path: str) -> bool:
    if not source_path or not os.path.exists(source_path):
        return False
    ckpt = load_ckpt(source_path)
    try:
        target_model.load_state_dict(ckpt['state_dict'])
        return True
    except Exception as e:
        print(f'  WARN: warm start failed: {e}')
        return False


# ─── Train one stage with curriculum ─────────────────────────────────────────

def train_one_stage(stage, N, ku, kv, cfg, warm_from, tag):
    n = int(np.log2(N))
    channel = build_channel('gmac', sigma2=10 ** (-SNR_DB / 10))
    Au, Av, fu, fv, pe_u, pe_v = load_class_c_design('gmac', n, snr_db=SNR_DB, ku=ku, kv=kv)

    print(f'\n--- Stage {stage} | N={N} | ku={ku} kv={kv} ---')
    print(f'  iters={cfg[f"s{stage}_iters"]} batch={cfg[f"s{stage}_batch"]} lr={cfg[f"s{stage}_lr"]}')
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
        print(f'  warm-started from {os.path.basename(warm_from)}')
    else:
        print(f'  fresh start')
    print(f'  params: {model.count_parameters():,}')

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
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f'{tag}_best.pt')

    eval_every = max(2000, iters // 15)

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
                save_ckpt(ckpt_path, model, {
                    'd': 16, 'hidden': 64, 'n_layers': 2, 'z_dim': z_dim,
                    'channel': 'gmac', 'stage': stage, 'N': N,
                    'Au': Au, 'Av': Av,
                })
                marker = ' *'
            print(f'  [{it:>6}/{iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f} @{best_iter}) {elapsed:.1f}min{marker}', flush=True)

    elapsed = (time.time() - t0) / 60
    print(f'  Stage {stage} done: best={best_bler:.4f} @ iter {best_iter}, {elapsed:.1f} min')
    return ckpt_path, best_bler, best_iter, elapsed


# ─── Main sweep ─────────────────────────────────────────────────────────────

def run_sweep(N_list):
    results = {}
    prev_s1_ckpt = None
    prev_s2_ckpt = None
    t_total = time.time()

    for N in N_list:
        n = int(np.log2(N))
        ref = SC_REF[N]
        cfg = CURRICULUM[N]
        sc_bler = ref['sc_bler']
        target = sc_bler * 1.5

        print('\n' + '=' * 70)
        print(f'CURRICULUM STAGE: N={N}')
        print(f'  SC BLER={sc_bler:.4f}, target (1.5x)={target:.4f}')
        print('=' * 70)

        # Stage 1
        s1_ckpt, s1_bler, s1_best_iter, s1_time = train_one_stage(
            stage=1, N=N, ku=ref['ku'], kv=ref['kv'], cfg=cfg,
            warm_from=prev_s1_ckpt, tag=f'curriculum_gmac_c_s1_N{N}',
        )

        # Stage 2
        s2_ckpt, s2_bler, s2_best_iter, s2_time = train_one_stage(
            stage=2, N=N, ku=ref['ku'], kv=ref['kv'], cfg=cfg,
            warm_from=prev_s2_ckpt, tag=f'curriculum_gmac_c_s2_N{N}',
        )

        # Chained eval
        print(f'\n--- Chained eval at N={N} ({EVAL_CW} CW) ---')
        t0 = time.time()
        chain_results = chain_evaluate(
            stage1_ckpt=s1_ckpt, stage2_ckpt=s2_ckpt,
            channel_name='gmac', n=n, snr_db=SNR_DB,
            ku=ref['ku'], kv=ref['kv'],
            n_cw=EVAL_CW, batch=16, seed=999, verbose=False,
        )
        eval_time = (time.time() - t0) / 60
        bler = chain_results['bler_total']
        ratio = bler / sc_bler if sc_bler > 0 else float('inf')
        passed = bler <= target

        print(f'  Chained: BLER={bler:.4f} CI=[{chain_results["ci_low"]:.4f},'
              f' {chain_results["ci_high"]:.4f}]  ratio={ratio:.2f}x  pass={passed}')

        results[N] = {
            'N': N,
            'ku': ref['ku'], 'kv': ref['kv'],
            'sc_bler': sc_bler, 'target_1_5x': target,
            'stage1': {'best_bler': float(s1_bler), 'best_iter': int(s1_best_iter),
                       'time_min': float(s1_time), 'ckpt': s1_ckpt},
            'stage2': {'best_bler': float(s2_bler), 'best_iter': int(s2_best_iter),
                       'time_min': float(s2_time), 'ckpt': s2_ckpt},
            'chained': {
                'bler': float(bler),
                'ci_low': float(chain_results['ci_low']),
                'ci_high': float(chain_results['ci_high']),
                'errs_u': int(chain_results['errs_u']),
                'errs_v': int(chain_results['errs_v']),
                'ratio_to_sc': float(ratio),
                'pass': bool(passed),
                'eval_time_min': float(eval_time),
                'eval_cw': EVAL_CW,
            },
        }

        # Save incrementally
        with open(os.path.join(_ROOT, 'class_c_npd', 'results', 'curriculum_sweep_results.json'), 'w') as f:
            json.dump({k: v for k, v in results.items()}, f, indent=2, default=str)

        prev_s1_ckpt = s1_ckpt
        prev_s2_ckpt = s2_ckpt

        elapsed_total = (time.time() - t_total) / 60
        print(f'\n  Cumulative time: {elapsed_total:.1f} min')

    # Final summary table
    print('\n' + '=' * 70)
    print('CURRICULUM SWEEP — FINAL SUMMARY')
    print('=' * 70)
    print(f'{"N":<6}{"SC":<10}{"NPD":<10}{"CI95":<24}{"ratio":<8}{"pass":<6}')
    print('-' * 70)
    for N in sorted(results.keys()):
        r = results[N]
        ci = f'[{r["chained"]["ci_low"]:.4f},{r["chained"]["ci_high"]:.4f}]'
        p = 'PASS' if r['chained']['pass'] else 'FAIL'
        print(f'{N:<6}{r["sc_bler"]:<10.4f}{r["chained"]["bler"]:<10.4f}'
              f'{ci:<24}{r["chained"]["ratio_to_sc"]:<8.2f}{p:<6}')
    total_min = (time.time() - t_total) / 60
    print(f'\nTotal sweep wall: {total_min:.1f} min')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_list', type=str, default='16,32,64',
                        help='Comma-separated list of N values')
    args = parser.parse_args()
    N_list = [int(x) for x in args.N_list.split(',')]
    print(f'GMAC Class C curriculum sweep: N values = {N_list}')
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    run_sweep(N_list)
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
