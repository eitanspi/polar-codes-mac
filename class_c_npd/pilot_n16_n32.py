"""
Pilot run: GMAC Class C NPD at N=16 and N=32.

Trains Stage 1 (U on mixture) and Stage 2 (V on clean) at each N,
then runs chained end-to-end evaluation. Compares to SC reference
at 50% of per-user capacity.

Rate: 50% of per-user capacity (from gmac_sc_reference_50pct.json)
  N=16: ku=4, kv=7, SC BLER=0.1626
  N=32: ku=7, kv=15, SC BLER=0.0684

Targets (1.5x SC):
  N=16: NPD BLER <= 0.244
  N=32: NPD BLER <= 0.103
"""
import os
import sys
import time
import json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from class_c_npd.training.train_stage import train_stage
from class_c_npd.eval.chain_eval import chain_evaluate


# ─── Config ──────────────────────────────────────────────────────────────────

SNR_DB = 6.0

# SC references from gmac_sc_reference_50pct.json
SC_REF = {
    16: {'ku': 4,  'kv': 7,  'sc_bler': 0.1626, 'target': 0.244},
    32: {'ku': 7,  'kv': 15, 'sc_bler': 0.0684, 'target': 0.103},
}

# Training schedule
TRAIN_CONFIG = {
    16: {
        's1_iters': 15000, 's1_batch': 64, 's1_lr': 3e-4,
        's2_iters': 5000,  's2_batch': 64, 's2_lr': 3e-4,
    },
    32: {
        's1_iters': 30000, 's1_batch': 64, 's1_lr': 3e-4,
        's2_iters': 10000, 's2_batch': 64, 's2_lr': 3e-4,
    },
}

EVAL_CW = 5000


def run_pilot_n(n, warm_start_s1=None, warm_start_s2=None):
    N = 1 << n
    ref = SC_REF[N]
    cfg = TRAIN_CONFIG[N]

    print('\n' + '=' * 70)
    print(f'PILOT — N={N} (n={n}), SNR={SNR_DB}dB, GMAC Class C')
    print(f'  ku={ref["ku"]}, kv={ref["kv"]}, SC BLER={ref["sc_bler"]:.4f}')
    print(f'  Target (1.5x SC): {ref["target"]:.4f}')
    print('=' * 70)

    # Stage 1 training
    print(f'\n[Stage 1] U on mixture channel ({cfg["s1_iters"]} iters)')
    t0 = time.time()
    s1_bler, s1_ckpt = train_stage(
        stage=1, channel_name='gmac', n=n, snr_db=SNR_DB,
        d=16, hidden=64, n_layers=2,
        batch=cfg['s1_batch'], lr=cfg['s1_lr'],
        total_iters=cfg['s1_iters'],
        eval_every=max(1000, cfg['s1_iters'] // 5),
        ku=ref['ku'], kv=ref['kv'],
        seed=42, tag=f'gmac_c_stage1_N{N}',
    )
    s1_time = (time.time() - t0) / 60
    print(f'\n  Stage 1 done: BLER={s1_bler:.4f} in {s1_time:.1f} min')

    # Stage 2 training
    print(f'\n[Stage 2] V on clean channel ({cfg["s2_iters"]} iters)')
    t0 = time.time()
    s2_bler, s2_ckpt = train_stage(
        stage=2, channel_name='gmac', n=n, snr_db=SNR_DB,
        d=16, hidden=64, n_layers=2,
        batch=cfg['s2_batch'], lr=cfg['s2_lr'],
        total_iters=cfg['s2_iters'],
        eval_every=max(1000, cfg['s2_iters'] // 5),
        ku=ref['ku'], kv=ref['kv'],
        seed=43, tag=f'gmac_c_stage2_N{N}',
    )
    s2_time = (time.time() - t0) / 60
    print(f'\n  Stage 2 done: BLER={s2_bler:.4f} in {s2_time:.1f} min')

    # Chained end-to-end evaluation
    print(f'\n[Chained eval] {EVAL_CW} codewords')
    t0 = time.time()
    results = chain_evaluate(
        stage1_ckpt=s1_ckpt, stage2_ckpt=s2_ckpt,
        channel_name='gmac', n=n, snr_db=SNR_DB,
        ku=ref['ku'], kv=ref['kv'],
        n_cw=EVAL_CW, batch=16, seed=999, verbose=False,
    )
    eval_time = (time.time() - t0) / 60
    print(f'  Chained BLER: {results["bler_total"]:.4f}  '
          f'(95% CI: [{results["ci_low"]:.4f}, {results["ci_high"]:.4f}])')
    print(f'  U errs: {results["errs_u"]}  V errs: {results["errs_v"]}  '
          f'(total={results["errs_total"]})')
    print(f'  Eval wall: {eval_time:.1f} min')

    summary = {
        'N': N, 'n': n,
        'ku': ref['ku'], 'kv': ref['kv'],
        'sc_bler': ref['sc_bler'],
        'target_1_5x': ref['target'],
        'stage1_bler': float(s1_bler),
        'stage1_time_min': float(s1_time),
        'stage2_bler': float(s2_bler),
        'stage2_time_min': float(s2_time),
        'chained_bler': float(results['bler_total']),
        'chained_ci_low': float(results['ci_low']),
        'chained_ci_high': float(results['ci_high']),
        'chained_errs_u': int(results['errs_u']),
        'chained_errs_v': int(results['errs_v']),
        'chained_errs_total': int(results['errs_total']),
        'eval_cw': EVAL_CW,
        'eval_time_min': float(eval_time),
        'ratio_to_sc': float(results['bler_total'] / ref['sc_bler']) if ref['sc_bler'] > 0 else float('inf'),
        'pass_1_5x': bool(results['bler_total'] <= ref['target']),
    }
    return summary


def main():
    print('GMAC Class C NPD pilot — N=16 and N=32')
    print(f'Target: NPD BLER within 1.5x of SC reference')
    print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    t_total = time.time()
    pilot_results = {}

    for n in [4, 5]:  # N=16, N=32
        summary = run_pilot_n(n)
        pilot_results[1 << n] = summary

    total_time = (time.time() - t_total) / 60

    # Final summary table
    print('\n' + '=' * 70)
    print('PILOT SUMMARY')
    print('=' * 70)
    print(f'{"N":<6}{"SC":<10}{"NPD":<10}{"CI95":<20}{"ratio":<10}{"pass":<8}')
    print('-' * 70)
    for N, s in sorted(pilot_results.items()):
        ci = f'[{s["chained_ci_low"]:.4f}, {s["chained_ci_high"]:.4f}]'
        pass_mark = 'PASS' if s['pass_1_5x'] else 'FAIL'
        print(f'{N:<6}{s["sc_bler"]:<10.4f}{s["chained_bler"]:<10.4f}{ci:<20}'
              f'{s["ratio_to_sc"]:<10.2f}{pass_mark:<8}')
    print()
    print(f'Total wall time: {total_time:.1f} min')
    print(f'Finish: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # Save
    with open(os.path.join(_HERE, 'results', 'pilot_n16_n32_results.json'), 'w') as f:
        json.dump(pilot_results, f, indent=2)
    print(f'\nResults saved to class_c_npd/results/pilot_n16_n32_results.json')


if __name__ == '__main__':
    main()
