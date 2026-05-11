#!/usr/bin/env python3
"""
check_progress.py — Quick status check of all running/completed training.
"""
import os, json, glob, subprocess, time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_processes():
    """Check if training processes are running."""
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    lines = [l for l in result.stdout.split('\n') if 'train_ising' in l or 'train_maagn' in l]
    lines = [l for l in lines if 'grep' not in l]
    if lines:
        print("RUNNING PROCESSES:")
        for l in lines:
            parts = l.split()
            pid, cpu, time_str = parts[1], parts[2], parts[9]
            cmd = ' '.join(parts[10:])
            print(f"  PID={pid} CPU={cpu}% TIME={time_str} CMD={cmd}")
    else:
        print("NO TRAINING PROCESSES RUNNING")
    print()

def check_logs():
    """Check latest log entries."""
    log_patterns = [
        ('Ising N=128', os.path.join(_ROOT, 'class_c_npd/results/npd_ising_mac/ising_d16_h100_N128.log')),
        ('Ising N=256', os.path.join(_ROOT, 'class_c_npd/results/npd_ising_mac/ising_d16_h100_N256.log')),
        ('MAAGN N=256', os.path.join(_ROOT, 'class_c_npd/results/npd_maagn_mac/maagn_d16_h100_N256.log')),
    ]
    print("TRAINING LOGS:")
    for name, path in log_patterns:
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
            if lines:
                last_lines = lines[-3:]
                print(f"  {name}: ({len(lines)} lines)")
                for l in last_lines:
                    print(f"    {l.rstrip()}")
            else:
                print(f"  {name}: (empty)")
        else:
            print(f"  {name}: NOT STARTED")
    print()

def check_checkpoints():
    """List new checkpoints."""
    dirs = [
        ('Ising', os.path.join(_ROOT, 'class_c_npd/results/npd_ising_mac')),
        ('MAAGN', os.path.join(_ROOT, 'class_c_npd/results/npd_maagn_mac')),
    ]
    print("LATEST CHECKPOINTS:")
    for name, d in dirs:
        pts = sorted(glob.glob(os.path.join(d, '*_N128*.pt')) + glob.glob(os.path.join(d, '*_N256*.pt')),
                     key=os.path.getmtime)
        if pts:
            for p in pts[-3:]:
                mtime = time.strftime('%H:%M', time.localtime(os.path.getmtime(p)))
                size_mb = os.path.getsize(p) / 1e6
                print(f"  {name}: {os.path.basename(p)} ({size_mb:.1f}MB, {mtime})")
        else:
            print(f"  {name}: no N=128/256 checkpoints yet")
    print()

def check_gpu_ckpts():
    """Check for GPU checkpoints."""
    gpu_dir = '/tmp/paper_style'
    if os.path.exists(gpu_dir):
        pts = glob.glob(os.path.join(gpu_dir, '*_final.pt'))
        if pts:
            print(f"GPU CHECKPOINTS: {len(pts)} found")
            for p in pts:
                print(f"  {os.path.basename(p)}")
        else:
            print("GPU CHECKPOINTS: none found")
    else:
        print("GPU CHECKPOINTS: /tmp/paper_style does not exist")
    print()

if __name__ == '__main__':
    print(f"=== Status Check {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    check_processes()
    check_logs()
    check_checkpoints()
    check_gpu_ckpts()
