#!/usr/bin/env python3
"""
eval_crc_scl_multimodel.py — Run CRC-SCL L=4 on multiple N=256 checkpoints.
"""
import os, sys, json, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import subprocess

MODELS = [
    'campaign_n256_sched_best.pt',
    'n256_long_best.pt',
]

N_CW = 1000
L = 4
T = 1.0
N = 256

out_dir = os.path.join(BASE, 'results', 'crc_scl_expansion', 'validation')

for model in MODELS:
    safe_name = model.replace('.pt', '')
    out_path = os.path.join(out_dir, f'N256_L4_{safe_name}_{N_CW}cw.json')
    if os.path.exists(out_path):
        print(f"  Skipping {model} — already done")
        continue
    cmd = [
        sys.executable, os.path.join(BASE, 'scripts', 'eval_crc_scl_validation.py'),
        '--model', model,
        '--N', str(N),
        '--L', str(L),
        '--T', str(T),
        '--n_cw', str(N_CW),
        '--out', out_path,
        '--progress_every', '100',
    ]
    print(f"\n  Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"  Done: {model}", flush=True)

print("\n  All done.")
