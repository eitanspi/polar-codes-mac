"""
Print the current status of all running experiments and their results.

Usage: python -u class_c_npd/status.py
"""
import os
import sys
import json
import glob

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, 'results')


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def show_curriculum():
    print('=' * 70)
    print('CURRICULUM SWEEP — GMAC Class C')
    print('=' * 70)

    log = os.path.join(RESULTS_DIR, 'curriculum_sweep.log')
    if os.path.exists(log):
        with open(log) as f:
            lines = f.read().splitlines()
        # Print last 25 lines
        for line in lines[-25:]:
            print(' ', line)
        print()

    results = load_json(os.path.join(RESULTS_DIR, 'curriculum_sweep_results.json'))
    if results:
        print('Saved chained results so far:')
        print(f'{"N":<6}{"SC":<10}{"NPD":<10}{"95% CI":<22}{"ratio":<8}')
        print('-' * 60)
        for N in sorted(results.keys(), key=int):
            r = results[N]
            ch = r['chained']
            ci = f'[{ch["ci_low"]:.4f},{ch["ci_high"]:.4f}]'
            print(f'{N:<6}{r["sc_bler"]:<10.4f}{ch["bler"]:<10.4f}{ci:<22}'
                  f'{ch["ratio_to_sc"]:<8.2f}')
    else:
        print('  (no results saved yet)')


def show_snr_diagnostic():
    print()
    print('=' * 70)
    print('DIAGNOSTIC: NPD Stage 1 across SNR')
    print('=' * 70)

    log = os.path.join(RESULTS_DIR, 'diagnostic_snr_sweep.log')
    if os.path.exists(log):
        with open(log) as f:
            lines = f.read().splitlines()
        for line in lines[-15:]:
            print(' ', line)


def show_sc_reference():
    print()
    print('=' * 70)
    print('SC REFERENCE — GMAC Class C, 50% capacity')
    print('=' * 70)

    sc = load_json(os.path.join(RESULTS_DIR, 'gmac_sc_reference_50pct.json'))
    if sc:
        print(f'{"N":<6}{"ku":<6}{"kv":<6}{"R_u":<8}{"R_v":<8}{"SC BLER":<12}')
        print('-' * 60)
        for N in sorted(sc['results'].keys(), key=int):
            r = sc['results'][N]
            print(f'{N:<6}{r["ku"]:<6}{r["kv"]:<6}{r["R_u"]:<8.3f}{r["R_v"]:<8.3f}{r["sc_bler"]:<12.5f}')


def show_processes():
    print()
    print('=' * 70)
    print('RUNNING PROCESSES')
    print('=' * 70)
    import subprocess
    try:
        out = subprocess.check_output(['ps', 'aux'], text=True)
        for line in out.splitlines():
            if any(k in line for k in ['curriculum_sweep', 'diagnostic_snr', 'multi_channel', 'class_c_npd']):
                if 'grep' not in line and 'status.py' not in line:
                    parts = line.split()
                    pid, cpu = parts[1], parts[2]
                    cmd = ' '.join(parts[10:])
                    print(f'  PID={pid} CPU={cpu}%  {cmd[:90]}')
    except Exception:
        pass


def show_files():
    print()
    print('=' * 70)
    print('RESULT FILES')
    print('=' * 70)
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json')) +
                   glob.glob(os.path.join(RESULTS_DIR, '*.log')))
    for f in files:
        size_kb = os.path.getsize(f) / 1024
        print(f'  {os.path.basename(f):<45}  {size_kb:>6.1f} KB')

    ckpts = sorted(glob.glob(os.path.join(RESULTS_DIR, '*.pt')))
    if ckpts:
        print(f'  --- {len(ckpts)} checkpoints ---')
        for f in ckpts[:10]:
            print(f'    {os.path.basename(f)}')
        if len(ckpts) > 10:
            print(f'    ... and {len(ckpts) - 10} more')


def main():
    show_curriculum()
    show_snr_diagnostic()
    show_sc_reference()
    show_processes()
    show_files()


if __name__ == '__main__':
    main()
