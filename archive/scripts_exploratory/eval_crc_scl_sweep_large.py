#!/usr/bin/env python3
"""
eval_crc_scl_sweep_large.py — CRC-SCL at larger N values.
"""
import os, sys, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(4)

from scripts.eval_crc_scl_sweep import eval_config

EXTRA = [
    # GMAC Class B N=1024
    {'channel': 'gmac', 'cls': 'B', 'N': 1024, 'ku': 492, 'kv': 492, 'n_cw': 500},
    # BEMAC Class B N=512
    {'channel': 'bemac', 'cls': 'B', 'N': 512, 'ku': 256, 'kv': 358, 'n_cw': 500},
    # BEMAC Class C N=512
    {'channel': 'bemac', 'cls': 'C', 'N': 512, 'ku': 154, 'kv': 307, 'n_cw': 500},
    # GMAC Class C N=512
    {'channel': 'gmac', 'cls': 'C', 'N': 512, 'ku': 119, 'kv': 233, 'n_cw': 500},
]

out_dir = os.path.join(BASE, 'results', 'crc_scl_sweep')

for cfg in EXTRA:
    channel = cfg['channel']
    cls = cfg['cls']
    N = cfg['N']
    ku = cfg['ku']
    kv = cfg['kv']
    n_cw = cfg['n_cw']

    out_path = os.path.join(out_dir, f'{channel}_{cls}_crc_scl.json')
    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)

    key = str(N)
    if key in all_results and all_results[key].get('n_cw', 0) >= n_cw:
        print(f"[skip] {channel} {cls} N={N} already done")
        continue

    print(f"\n{channel} Class {cls} N={N}, ku={ku}, kv={kv}, n_cw={n_cw}")
    try:
        result = eval_config(channel, cls, N, ku, kv, n_cw, L=4)
        all_results[key] = result
        print(f"  DONE SC={result['sc_bler']:.4f} SCL={result['scl_bler']:.4f} CRC-SCL={result['crc_scl_bler']:.4f} [{result['time_s']:.0f}s]")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAILED: {e}")
        all_results[key] = {'error': str(e), 'N': N, 'ku': ku, 'kv': kv}

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

print("\nDone with large N configs.")
