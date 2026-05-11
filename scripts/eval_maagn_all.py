#!/usr/bin/env python3
"""
eval_maagn_all.py
=================
Re-evaluate all MA-AGN chained NPD checkpoints at N=16, 32, 64 and report
alongside memoryless GMAC SC baselines. Produces a consolidated JSON.

Use this when the per-N training script overwrote its JSON during sequential
runs.

Usage:
  python scripts/eval_maagn_all.py
"""
from __future__ import annotations
import os
import sys
import json
import math
import time

import numpy as np
import torch

torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.channels_memory_new import MAAGNMAC
from neural.npd_memory_mac import ChainedNPD_MAC

from scripts.train_npd_maagn_mac import (
    RATES, RESULTS_DIR, SNR_DB, ALPHA,
    load_design, eval_chained, eval_memoryless_sc,
)

# Configurations matching how checkpoints were trained.
#  - N=16: d=16, hidden=64
#  - N=32: d=16, hidden=64
#  - N=64: d=32, hidden=128
CONFIGS = {
    16: {'d': 16, 'hidden': 64, 'n_layers': 2, 'encoder_type': 'bigru',
         'window_size': 2, 'gru_layers': 1},
    32: {'d': 32, 'hidden': 128, 'n_layers': 2, 'encoder_type': 'bigru',
         'window_size': 2, 'gru_layers': 1},
    64: {'d': 32, 'hidden': 128, 'n_layers': 2, 'encoder_type': 'bigru',
         'window_size': 2, 'gru_layers': 1},
}


def main():
    all_results = {}
    t_total = time.time()

    for N in [16, 32, 64]:
        cfg = CONFIGS[N]
        ku, kv = RATES[N]
        channel = MAAGNMAC.from_snr_db(SNR_DB, alpha=ALPHA)
        Au, Av, fu_1idx, fv_1idx, fu_set, fv_set = load_design(N, ku, kv)

        s1_ckpt = os.path.join(RESULTS_DIR,
                               f'maagn_bigru_L1_s1_N{N}_best.pt')
        s2_ckpt = os.path.join(RESULTS_DIR,
                               f'maagn_bigru_L1_s2_N{N}_best.pt')
        if not os.path.exists(s1_ckpt) or not os.path.exists(s2_ckpt):
            print(f'N={N}: checkpoints missing, skipping')
            continue

        torch.manual_seed(42)
        model = ChainedNPD_MAC(d=cfg['d'], hidden=cfg['hidden'],
                               n_layers=cfg['n_layers'],
                               encoder_type=cfg['encoder_type'],
                               window_size=cfg['window_size'],
                               gru_layers=cfg['gru_layers'])

        sd1 = torch.load(s1_ckpt, weights_only=False, map_location='cpu')
        sd2 = torch.load(s2_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd1['state_dict'])
        model.stage2.load_state_dict(sd2['state_dict'])
        model.stage1.eval()
        model.stage2.eval()

        print(f'\nN={N} ku={ku} kv={kv}  params={model.count_parameters():,}')
        print('-' * 60)

        t0 = time.time()
        print('  Chained NPD inference (2000 codewords)...')
        chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set,
                               n_cw=2000, seed=777)
        print(f'    chained BLER={chained["bler_total"]:.4f} '
              f'(U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f}) '
              f'({(time.time()-t0)/60:.2f} min)')

        t0 = time.time()
        print('  Memoryless GMAC SC baseline (2000 codewords)...')
        baseline = eval_memoryless_sc(channel, N, Au, Av, fu_1idx, fv_1idx,
                                      n_cw=2000, seed=555)
        print(f'    memoryless SC BLER={baseline["bler_total"]:.4f} '
              f'(U={baseline["bler_u"]:.4f}, V={baseline["bler_v"]:.4f}) '
              f'({(time.time()-t0)/60:.2f} min)')

        improvement = (baseline['bler_total'] - chained['bler_total']) / max(
            baseline['bler_total'], 1e-6)
        ratio = chained['bler_total'] / max(baseline['bler_total'], 1e-6)
        print(f'  Improvement: {improvement*100:.1f}%  ratio={ratio:.3f}')

        all_results[str(N)] = {
            'channel': 'maagn',
            'channel_meta': {'alpha': ALPHA, 'snr_db': SNR_DB,
                             'sigma2': channel.sigma2},
            'N': N, 'ku': ku, 'kv': kv,
            'config': cfg,
            'chained': {k: (float(v) if isinstance(v, (float, int)) else v)
                        for k, v in chained.items()},
            'memoryless_sc': {k: (float(v) if isinstance(v, (float, int)) else v)
                              for k, v in baseline.items()},
            'improvement_ratio': ratio,
            's1_ckpt': s1_ckpt, 's2_ckpt': s2_ckpt,
        }

    out_json = os.path.join(RESULTS_DIR, 'maagn_consolidated_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nSaved: {out_json}')
    print(f'Total time: {(time.time()-t_total)/60:.1f} min')

    # Summary table
    print(f'\n{"N":<6}{"Chained BLER":<16}{"Memoryless SC":<16}{"Ratio":<10}'
          f'{"Improvement":<14}')
    for Ns, r in all_results.items():
        ch = r['chained']['bler_total']
        ms = r['memoryless_sc']['bler_total']
        ratio = ch / max(ms, 1e-6)
        impr = (ms - ch) / max(ms, 1e-6) * 100
        print(f'{Ns:<6}{ch:<16.4f}{ms:<16.4f}{ratio:<10.3f}{impr:<14.1f}%')


if __name__ == '__main__':
    main()
