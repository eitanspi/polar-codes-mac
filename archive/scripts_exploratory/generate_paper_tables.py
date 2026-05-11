#!/usr/bin/env python3
"""
generate_paper_tables.py
========================
Generate paper-style markdown tables from results.

Creates project_summary/PAPER_STYLE_TABLES.md with:
  Table A: ISI-MAC
  Table B: Ising MAC
  Table C: MA-AGN MAC
"""
import sys, os, json, math

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

RESULTS_DIR = os.path.join(_ROOT, 'results', 'paper_style')
OUTPUT_DIR = os.path.join(_ROOT, 'project_summary')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def wilson_ci(n_err, n_total, z=1.96):
    if n_total == 0:
        return (0.0, 1.0)
    p_hat = n_err / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_bler(r):
    """Format BLER with Wilson CI."""
    if r is None or 'bler_total' not in r:
        return '---'
    bler = r['bler_total']
    ci = r.get('wilson_95_ci', wilson_ci(r.get('errs_total', 0), r.get('n_cw', 1)))
    n_cw = r.get('n_cw', '?')
    if bler == 0:
        return f'0.0000 [0, {ci[1]:.4f}]'
    return f'{bler:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]'


def fmt_bler_short(r):
    """Short format."""
    if r is None or 'bler_total' not in r:
        return '---'
    bler = r['bler_total']
    return f'{bler:.4f}'


def get_best_npd(npd_data, channel_key, N):
    """Get best NPD result for a given N."""
    if npd_data is None or channel_key not in npd_data:
        return None
    best = None
    for key, r in npd_data[channel_key].items():
        if 'error' in r or 'bler_total' not in r:
            continue
        if r.get('N') != N:
            continue
        if best is None or r['bler_total'] < best['bler_total']:
            best = r
    return best


def get_npd_by_model(npd_data, channel_key, N, model_substr):
    """Get NPD result for specific model type at given N."""
    if npd_data is None or channel_key not in npd_data:
        return None
    for key, r in npd_data[channel_key].items():
        if 'error' in r or 'bler_total' not in r:
            continue
        if r.get('N') != N:
            continue
        if model_substr in key:
            return r
    return None


def ratio_str(npd_r, baseline_r):
    """Compute NPD/baseline ratio."""
    if npd_r is None or baseline_r is None:
        return '---'
    if 'bler_total' not in npd_r or 'bler_total' not in baseline_r:
        return '---'
    b_npd = npd_r['bler_total']
    b_base = baseline_r['bler_total']
    if b_base == 0:
        return 'inf'
    return f'{b_npd / b_base:.2f}'


def main():
    # Load all results
    trellis_isi = load_json('isi_mac_sc_baselines.json')
    memless_isi = load_json('isi_mac_memoryless_sc_baselines.json')
    ising = load_json('ising_mac_baselines.json')
    maagn = load_json('maagn_mac_baselines.json')
    npd_all = load_json('npd_all_channels_5kcw.json')

    lines = []
    lines.append('# Paper-Style BLER Tables')
    lines.append('')
    lines.append('Generated from `results/paper_style/` baseline and NPD evaluations.')
    lines.append('All entries include Wilson 95% confidence intervals.')
    lines.append('')

    # ====================================================================
    #  Table A: ISI-MAC
    # ====================================================================
    lines.append('## Table A: ISI-MAC (h=0.3, SNR=6 dB)')
    lines.append('')
    lines.append('| N | ku/kv | Chained Trellis SC | Memoryless SC | NPD (best) | NPD/Trellis |')
    lines.append('|---|-------|-------------------|---------------|------------|-------------|')

    for N in [16, 32, 64, 128, 256, 512, 1024]:
        key = f'N{N}'
        rates = {16:(4,7),32:(7,15),64:(15,29),128:(30,58),256:(59,117),512:(119,233),1024:(238,467)}
        ku, kv = rates[N]

        tr = trellis_isi.get(key) if trellis_isi else None
        ml = memless_isi.get(key) if memless_isi else None
        npd_best = get_best_npd(npd_all, 'isi_mac', N)

        lines.append(
            f'| {N} | {ku}/{kv} | {fmt_bler(tr)} | {fmt_bler(ml)} | '
            f'{fmt_bler(npd_best)} | {ratio_str(npd_best, tr)} |'
        )

    lines.append('')

    # ====================================================================
    #  Table B: Ising MAC
    # ====================================================================
    lines.append('## Table B: Ising MAC (p_flip=0.1, sigma2=0.251)')
    lines.append('')
    lines.append('| N | ku/kv | Chained Trellis SC | Memoryless SC | NPD d=16 h=100 | NPD/Memoryless |')
    lines.append('|---|-------|-------------------|---------------|----------------|----------------|')

    for N in [16, 32, 64]:
        key = f'N{N}'
        rates = {16:(4,7),32:(7,15),64:(15,29)}
        ku, kv = rates[N]

        tr = ising['trellis_sc'].get(key) if ising and 'trellis_sc' in ising else None
        ml = ising['memoryless_sc'].get(key) if ising and 'memoryless_sc' in ising else None
        npd_r = get_best_npd(npd_all, 'ising_mac', N)

        lines.append(
            f'| {N} | {ku}/{kv} | {fmt_bler(tr)} | {fmt_bler(ml)} | '
            f'{fmt_bler(npd_r)} | {ratio_str(npd_r, ml)} |'
        )

    lines.append('')

    # ====================================================================
    #  Table C: MA-AGN MAC
    # ====================================================================
    lines.append('## Table C: MA-AGN MAC (alpha=0.3, SNR=6 dB)')
    lines.append('')
    lines.append('| N | ku/kv | Memoryless SC | NPD (best) | NPD/Memoryless |')
    lines.append('|---|-------|---------------|------------|----------------|')

    for N in [16, 32, 64, 128]:
        key = f'N{N}'
        rates = {16:(4,7),32:(7,15),64:(15,29),128:(30,58)}
        ku, kv = rates[N]

        ml = maagn.get(key) if maagn else None
        npd_r = get_best_npd(npd_all, 'maagn_mac', N)

        lines.append(
            f'| {N} | {ku}/{kv} | {fmt_bler(ml)} | '
            f'{fmt_bler(npd_r)} | {ratio_str(npd_r, ml)} |'
        )

    lines.append('')
    lines.append('---')
    lines.append('Notes:')
    lines.append('- Wilson 95% CI format: BLER [CI_low, CI_high]')
    lines.append('- NPD/baseline < 1.0 means NPD is better')
    lines.append('- Trellis SC uses chained 2-stage decoder with forward-backward on 2-state trellis')
    lines.append('- Memoryless SC ignores channel memory, uses GMAC decoder')
    lines.append('- All baselines use 10K CW; NPD uses 5K CW')
    lines.append('')

    output = '\n'.join(lines)
    out_path = os.path.join(OUTPUT_DIR, 'PAPER_STYLE_TABLES.md')
    with open(out_path, 'w') as f:
        f.write(output)
    print(f'Saved: {out_path}')
    print()
    print(output)


if __name__ == '__main__':
    main()
