#!/usr/bin/env python3
"""
eval_crc_scl_sweep_extra.py — Extra CRC-SCL configs at larger N.
"""
import os, sys, json
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(4)

from scripts.eval_crc_scl_sweep import eval_config

EXTRA_CONFIGS = [
    # GMAC Class B N=512
    {'channel': 'gmac', 'cls': 'B', 'N': 512,  'ku': 246, 'kv': 246, 'n_cw': 1000},
    # GMAC Class C N=256
    {'channel': 'gmac', 'cls': 'C', 'N': 256,  'ku': 59, 'kv': 117, 'n_cw': 2000},
    # BEMAC Class C N=256
    {'channel': 'bemac', 'cls': 'C', 'N': 256,  'ku': 77, 'kv': 154, 'n_cw': 2000},
    # BEMAC Class B N=256
    {'channel': 'bemac', 'cls': 'B', 'N': 256,  'ku': 128, 'kv': 178, 'n_cw': 2000},
    # ABNMAC Class B N=128
    {'channel': 'abnmac', 'cls': 'B', 'N': 128,  'ku': 45, 'kv': 45, 'n_cw': 2000},
    # ABNMAC Class C N=128
    {'channel': 'abnmac', 'cls': 'C', 'N': 128,  'ku': 26, 'kv': 51, 'n_cw': 2000},
]

out_dir = os.path.join(BASE, 'results', 'crc_scl_sweep')

for cfg in EXTRA_CONFIGS:
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

print("\nDone with extra configs.")
