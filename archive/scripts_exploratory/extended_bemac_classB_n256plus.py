#!/usr/bin/env python3
"""
Re-evaluate BEMAC Class B (Ru=0.50, Rv=0.70) NN-SC and analytical SC at
N=256, 512, 1024 with enough codewords to get statistically stable BLER
estimates with Wilson 95% confidence intervals.

The original results were suspicious:
  N=256: 2 errors / 50K cw  -> BLER = 4e-5
  N=512: 0 errors / 10K cw  -> BLER = 0      (suspicious — should still see errors)
  N=1024: 1 error / 10K cw  -> BLER = 1e-4   (HIGHER than N=256, non-monotonic)

Output:
  to_git_v2/results/bemac/bemac_classB_Ru50_Rv70_nn_vs_sc/extended_eval.json
  to_git_v2/results/bemac/bemac_classB_Ru50_Rv70_nn_vs_sc/extended_eval_report.md
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'nn_mac'))

import math
import time
import json
import numpy as np
import torch

from polar.encoder import polar_encode_batch
from polar.channels import BEMAC
from polar.design import make_path
from polar.decoder import decode_single

# Reuse loaders from existing script
from eval_bemac_nn_scl import load_bemac_model, load_bemac_design


# ─── Wilson interval ──────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


# ─── Eval functions ───────────────────────────────────────────────────────────

def eval_nn_sc(model, N, b, Au, Av, fu, fv, n_cw, batch_size, seed=2025, log_every=2000):
    """NN-SC eval, returns (errors, total)."""
    model.eval()
    errs = 0
    total = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    last_log = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy((xf + yf).astype(np.int64)).long()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e:
                    errs += 1
            total += actual
            if total - last_log >= log_every:
                last_log = total
                p, lo, hi = wilson_ci(errs, total)
                print(f"    NN  [{total:>7d}/{n_cw}] errs={errs:>4d} BLER={p:.2e} CI[{lo:.2e},{hi:.2e}]  {time.time()-t0:.0f}s", flush=True)
    return errs, total


def eval_analytical_sc(N, b, Au, Av, fu, fv, channel, n_cw, seed=2025, log_every=2000):
    """Analytical SC eval (per-codeword, uses Numba parallel path for Class B)."""
    errs = 0
    total = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    last_log = 0
    while total < n_cw:
        # Generate one codeword at a time (analytical SC takes single z)
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        xf = polar_encode_batch(uf.reshape(1, N))[0]
        yf = polar_encode_batch(vf.reshape(1, N))[0]
        z = (xf + yf).astype(int).tolist()
        u_dec, v_dec = decode_single(N, z, b, fu, fv, channel)
        e = any(u_dec[p-1] != uf[p-1] for p in Au) or \
            any(v_dec[p-1] != vf[p-1] for p in Av)
        if e:
            errs += 1
        total += 1
        if total - last_log >= log_every:
            last_log = total
            p, lo, hi = wilson_ci(errs, total)
            print(f"    SC  [{total:>7d}/{n_cw}] errs={errs:>4d} BLER={p:.2e} CI[{lo:.2e},{hi:.2e}]  {time.time()-t0:.0f}s", flush=True)
    return errs, total


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    RU, RV = 0.50, 0.70
    PLAN = {
        256:  {'n_cw_nn': 100000, 'n_cw_sc': 100000, 'bs': 100},
        512:  {'n_cw_nn':  50000, 'n_cw_sc':  50000, 'bs':  50},
        1024: {'n_cw_nn':  20000, 'n_cw_sc':  20000, 'bs':  20},
    }

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'bemac',
                           'bemac_classB_Ru50_Rv70_nn_vs_sc')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'extended_eval.json')
    md_path = os.path.join(out_dir, 'extended_eval_report.md')

    results = {
        'experiment': 'BEMAC Class B, Ru=0.50, Rv=0.70',
        'purpose': 'Re-evaluate N=256/512/1024 with high CW counts to verify suspicious anomaly',
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'original': {
            '256':  {'nn_bler': 4e-05, 'sc_bler': 8e-05, 'cw': 50000},
            '512':  {'nn_bler': 0.0,   'sc_bler': 0.0,   'cw': 10000},
            '1024': {'nn_bler': 1e-04, 'sc_bler': 1e-04, 'cw': 10000},
        },
        'extended': {},
    }
    # Save scaffold so we have something even if it crashes mid-run
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    channel = BEMAC()

    for N in [256, 512, 1024]:
        cfg = PLAN[N]
        print(f"\n{'='*72}")
        print(f"  N={N}    NN cw={cfg['n_cw_nn']}    SC cw={cfg['n_cw_sc']}    bs={cfg['bs']}")
        print(f"{'='*72}", flush=True)

        Au, Av, fu, fv, ku, kv = load_bemac_design(N, RU, RV)
        b = make_path(N, N // 2)  # Class B
        model = load_bemac_model(N)

        print(f"  ku={ku}, kv={kv}, model params={model.count_parameters():,}", flush=True)

        # NN-SC
        print(f"\n  --- NN-SC ---", flush=True)
        nn_errs, nn_total = eval_nn_sc(model, N, b, Au, Av, fu, fv, cfg['n_cw_nn'], cfg['bs'])
        nn_bler, nn_lo, nn_hi = wilson_ci(nn_errs, nn_total)

        # analytical SC
        print(f"\n  --- Analytical SC ---", flush=True)
        sc_errs, sc_total = eval_analytical_sc(N, b, Au, Av, fu, fv, channel, cfg['n_cw_sc'])
        sc_bler, sc_lo, sc_hi = wilson_ci(sc_errs, sc_total)

        ratio = nn_bler / sc_bler if sc_bler > 0 else None

        results['extended'][str(N)] = {
            'N': N, 'ku': ku, 'kv': kv,
            'nn_errors': nn_errs, 'nn_cw': nn_total,
            'nn_bler': nn_bler, 'nn_ci_lo': nn_lo, 'nn_ci_hi': nn_hi,
            'sc_errors': sc_errs, 'sc_cw': sc_total,
            'sc_bler': sc_bler, 'sc_ci_lo': sc_lo, 'sc_ci_hi': sc_hi,
            'ratio_nn_over_sc': ratio,
        }

        # Save after each N
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n  N={N} SUMMARY: NN {nn_errs}/{nn_total}={nn_bler:.2e} | SC {sc_errs}/{sc_total}={sc_bler:.2e} | ratio={ratio}", flush=True)

    # Markdown report
    rows = ["| N | NN errors | NN cw | NN BLER (95% CI) | SC errors | SC cw | SC BLER (95% CI) | NN/SC |"]
    rows.append("|---|---|---|---|---|---|---|---|")
    for N in [256, 512, 1024]:
        r = results['extended'][str(N)]
        rows.append(f"| {N} | {r['nn_errors']} | {r['nn_cw']} | {r['nn_bler']:.2e} [{r['nn_ci_lo']:.2e}, {r['nn_ci_hi']:.2e}] | {r['sc_errors']} | {r['sc_cw']} | {r['sc_bler']:.2e} [{r['sc_ci_lo']:.2e}, {r['sc_ci_hi']:.2e}] | {r['ratio_nn_over_sc']} |")

    md = f"""# BEMAC Class B (Ru=0.50, Rv=0.70) — Extended Evaluation at N≥256

Re-ran NN-SC and analytical SC at N=256, 512, 1024 with high codeword counts
to verify the original suspicious values:

| N | Original NN BLER | Original SC BLER | Original CW |
|---|---|---|---|
| 256 | 4e-05 | 8e-05 | 50000 |
| 512 | 0.0 | 0.0 | 10000 |
| 1024 | 1e-04 | 1e-04 | 10000 |

## Extended results

{chr(10).join(rows)}

Wilson 95% CI used.
"""
    with open(md_path, 'w') as f:
        f.write(md)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == '__main__':
    main()
