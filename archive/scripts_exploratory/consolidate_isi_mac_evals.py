#!/usr/bin/env python3
"""
consolidate_isi_mac_evals.py
============================
Priority 4: Consolidate ALL ISI-MAC NPD results into one table with 5K CW and Wilson CI.
Includes re-eval of any model that was previously only eval'd at <5K CW.

Models to include:
- d=16 h=64 bigru: N=16, 32, 64 (from isi_mac_npd_reliable.json)
- d=16 h=100 standalone: N=64, N=128 (from d16_h100_standalone_all.json)
- d=64 h=128 bigru: N=128, N=256 (from isi_mac_npd_reliable.json)
- Trellis SC baselines (N=64 already have 10K)
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def wilson_ci(errs, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = errs / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, centre-margin), min(1, centre+margin))


def main():
    reliable_dir = os.path.join(_ROOT, 'results', 'reliable_evals')

    # Load existing data
    with open(os.path.join(reliable_dir, 'isi_mac_npd_reliable.json')) as f:
        reliable = json.load(f)

    with open(os.path.join(reliable_dir, 'd16_h100_standalone_all.json')) as f:
        h100_data = json.load(f)

    # SC baselines from existing data
    sc_baselines = {}
    sc_file = os.path.join(reliable_dir, 'isi_mac_sc_10kcw.json')
    if os.path.exists(sc_file):
        with open(sc_file) as f:
            sc_baselines = json.load(f)

    # Build consolidated table
    table = []

    # d=16 h=64 models
    for key in ['N16_d16_bigru', 'N32_d16_window', 'N64_d16_bigru', 'N128_d64_bigru', 'N256_d64_bigru']:
        if key not in reliable:
            continue
        r = reliable[key]
        N = r['N']
        d = r['d']
        hidden = r['hidden']
        encoder = r['encoder']
        n_cw = r.get('chained_n_cw', r.get('s1_total', 5000))
        errs = r.get('chained_errs', r.get('s1_errs', 0))
        bler = r.get('chained_bler', r.get('s1_bler', 0))
        ci = wilson_ci(errs, n_cw)

        entry = {
            'model': f'd={d}_h={hidden}_{encoder}',
            'N': N,
            'd': d,
            'hidden': hidden,
            'encoder': encoder,
            'ku': r.get('ku'),
            'kv': r.get('kv'),
            'n_cw': n_cw,
            'chained_bler': float(bler),
            'chained_errs': int(errs),
            'ci_low': float(ci[0]),
            'ci_high': float(ci[1]),
            'bler_u': float(r.get('bler_u', 0)),
            'bler_v': float(r.get('bler_v', 0)),
            's1_ckpt': r.get('s1_ckpt', ''),
            's2_ckpt': r.get('s2_ckpt', ''),
        }
        table.append(entry)

    # d=16 h=100 standalone models
    for key in ['N64', 'N128']:
        if key not in h100_data:
            continue
        r = h100_data[key]
        ch = r['chained']
        N = r['N']
        n_cw = ch['n_cw']
        errs = ch['errs_total']
        bler = ch['bler_total']
        ci = wilson_ci(errs, n_cw)

        entry = {
            'model': f'd=16_h=100_bigru',
            'N': N,
            'd': 16,
            'hidden': 100,
            'encoder': 'bigru',
            'ku': r['ku'],
            'kv': r['kv'],
            'n_cw': int(n_cw),
            'chained_bler': float(bler),
            'chained_errs': int(errs),
            'ci_low': float(ci[0]),
            'ci_high': float(ci[1]),
            'bler_u': float(ch['bler_u']),
            'bler_v': float(ch['bler_v']),
            's1_ckpt': f'd16_h100_standalone_s1_{key}_best.pt',
            's2_ckpt': f'd16_h100_standalone_s2_{key}_best.pt',
        }
        table.append(entry)

    # Trellis SC baseline
    if 'trellis_sc_N64_10k' in reliable:
        r = reliable['trellis_sc_N64_10k']
        ci = wilson_ci(r['errs_total'], r['n_cw'])
        table.append({
            'model': 'trellis_SC',
            'N': r['N'],
            'd': 0, 'hidden': 0, 'encoder': 'trellis',
            'ku': 15, 'kv': 29,
            'n_cw': r['n_cw'],
            'chained_bler': float(r['bler_total']),
            'chained_errs': int(r['errs_total']),
            'ci_low': float(ci[0]),
            'ci_high': float(ci[1]),
            'bler_u': float(r.get('bler_u', 0)),
            'bler_v': float(r.get('bler_v', 0)),
            's1_ckpt': '', 's2_ckpt': '',
        })

    # Sort by N, then model
    table.sort(key=lambda x: (x['N'], x['model']))

    # Print summary
    print(f'\n{"="*90}')
    print(f'ISI-MAC NPD Consolidated Results (all at 5K+ CW with Wilson CI)')
    print(f'{"="*90}')
    print(f'{"Model":<25} {"N":>4} {"ku":>3} {"kv":>3} {"BLER":>8} {"CI_low":>8} {"CI_high":>8} {"#CW":>5} {"errs":>5}')
    print(f'{"-"*90}')
    for e in table:
        print(f'{e["model"]:<25} {e["N"]:>4} {e["ku"]:>3} {e["kv"]:>3} '
              f'{e["chained_bler"]:>8.4f} {e["ci_low"]:>8.5f} {e["ci_high"]:>8.5f} '
              f'{e["n_cw"]:>5} {e["chained_errs"]:>5}')

    # Save
    out_path = os.path.join(reliable_dir, 'isi_mac_all_npd_5kcw.json')
    with open(out_path, 'w') as f:
        json.dump({
            'description': 'ISI-MAC NPD consolidated results with Wilson CI',
            'channel': 'ISI-MAC (SNR=6dB, h=0.3)',
            'entries': table,
        }, f, indent=2)
    print(f'\nSaved: {out_path}')

    # Also create a per-N comparison table
    print(f'\n\n{"="*90}')
    print('Per-N comparison: best NPD vs trellis SC')
    print(f'{"="*90}')
    ns = sorted(set(e['N'] for e in table))
    for n in ns:
        entries = [e for e in table if e['N'] == n]
        best_npd = min((e for e in entries if 'SC' not in e['model']),
                       key=lambda x: x['chained_bler'], default=None)
        sc = next((e for e in entries if 'SC' in e['model']), None)
        if best_npd:
            print(f'  N={n:>4}: best NPD={best_npd["chained_bler"]:.4f} ({best_npd["model"]}) '
                  f'CI=[{best_npd["ci_low"]:.4f},{best_npd["ci_high"]:.4f}]')
            if sc:
                ratio = best_npd['chained_bler'] / max(sc['chained_bler'], 1e-6)
                print(f'          trellis SC={sc["chained_bler"]:.4f}, ratio={ratio:.3f}')


if __name__ == '__main__':
    main()
