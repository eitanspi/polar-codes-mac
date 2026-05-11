"""
Generate a markdown report summarizing all sweep results.

Usage: python -u class_c_npd/eval/generate_report.py [--out path/report.md]
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results')


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def make_report():
    sc = load_json(os.path.join(RESULTS_DIR, 'gmac_sc_reference_50pct.json'))
    sweep = load_json(os.path.join(RESULTS_DIR, 'curriculum_sweep_results.json'))

    lines = []
    lines.append('# Class C NPD: GMAC Class C Sweep Report')
    lines.append('')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')

    lines.append('## Setup')
    lines.append('')
    lines.append('- **Channel**: Gaussian MAC, SNR = 6 dB')
    lines.append('- **Path**: Class C (`path_i = N`), decode all U first then V')
    lines.append('- **Rate**: 50% of per-user capacity (R_u ≈ 0.232, R_v ≈ 0.456)')
    lines.append('- **Architecture**: single-user NPD per stage, d=16 hidden=64 (~21K params)')
    lines.append('- **Training**: fast_ce parallel, curriculum across N with warm-starting')
    lines.append('- **Evaluation**: 5000 codewords per N, Wilson 95% CIs')
    lines.append('')

    lines.append('## SC Reference')
    lines.append('')
    lines.append('| N | ku | kv | SC BLER | NPD target (1.5×) |')
    lines.append('|---|----|----|---------|---|')
    if sc:
        for N in sorted(sc['results'].keys(), key=int):
            r = sc['results'][N]
            target = r['sc_bler'] * 1.5
            lines.append(f'| {N} | {r["ku"]} | {r["kv"]} | {r["sc_bler"]:.4f} | {target:.4f} |')
    lines.append('')

    lines.append('## Curriculum Sweep Results')
    lines.append('')

    if sweep:
        lines.append('| N | Stage 1 BLER | Stage 2 BLER | Chained BLER | 95% CI | Ratio | Pass? |')
        lines.append('|---|---|---|---|---|---|---|')
        for N in sorted(sweep.keys(), key=int):
            r = sweep[N]
            chain = r['chained']
            ci = f'[{chain["ci_low"]:.4f}, {chain["ci_high"]:.4f}]'
            passed = '✓ PASS' if chain['pass'] else '✗ FAIL'
            lines.append(
                f'| {N} | {r["stage1"]["best_bler"]:.4f} | '
                f'{r["stage2"]["best_bler"]:.4f} | {chain["bler"]:.4f} | '
                f'{ci} | {chain["ratio_to_sc"]:.2f}x | {passed} |'
            )
        lines.append('')

        # Headline summary
        passed_Ns = [N for N in sweep.keys() if sweep[N]['chained']['pass']]
        failed_Ns = [N for N in sweep.keys() if not sweep[N]['chained']['pass']]
        lines.append('### Summary')
        lines.append('')
        lines.append(f'- **PASSED at**: {", ".join(sorted(passed_Ns, key=int)) if passed_Ns else "(none)"}')
        lines.append(f'- **FAILED at**: {", ".join(sorted(failed_Ns, key=int)) if failed_Ns else "(none)"}')
    else:
        lines.append('(curriculum sweep results not yet available)')

    lines.append('')

    lines.append('## Architecture diagram')
    lines.append('')
    lines.append('```')
    lines.append('       GMAC channel: z = (1-2X) + (1-2Y) + W')
    lines.append('              │')
    lines.append('              ▼')
    lines.append('  ┌──────────────────────┐')
    lines.append('  │  Stage 1 (NPD)       │     U on marginal channel')
    lines.append('  │  decode U from raw z │     mixture LLR via z_encoder')
    lines.append('  └──────────┬───────────┘')
    lines.append('             │ û')
    lines.append('             ▼')
    lines.append('   x̂ = polar_encode(û)')
    lines.append('             │')
    lines.append('             ▼')
    lines.append("    z' = z - (1 - 2·x̂)     subtract known U contribution")
    lines.append('             │')
    lines.append('             ▼')
    lines.append('  ┌──────────────────────┐')
    lines.append('  │  Stage 2 (NPD)       │     V on clean BPSK+AWGN')
    lines.append('  │  decode V from z\'    │     standard single-user NPD')
    lines.append('  └──────────┬───────────┘')
    lines.append('             │ v̂')
    lines.append('             ▼')
    lines.append('         output')
    lines.append('```')
    lines.append('')

    lines.append('## Reproduction')
    lines.append('')
    lines.append('```bash')
    lines.append('# Run smoke test (5 min)')
    lines.append('python -u class_c_npd/smoke_test.py')
    lines.append('')
    lines.append('# Run curriculum sweep')
    lines.append('python -u class_c_npd/training/curriculum_sweep.py --N_list 16,32,64,128')
    lines.append('')
    lines.append('# Generate this report')
    lines.append('python -u class_c_npd/eval/generate_report.py')
    lines.append('```')

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str,
                        default=os.path.join(RESULTS_DIR, 'sweep_report.md'))
    args = parser.parse_args()

    report = make_report()
    with open(args.out, 'w') as f:
        f.write(report)
    print(f'Wrote {args.out}')
    print()
    print(report[:2000])
    if len(report) > 2000:
        print(f'... ({len(report)} chars total)')


if __name__ == '__main__':
    main()
