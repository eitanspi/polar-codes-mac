#!/usr/bin/env python3
"""
Rigorous four-way comparison on BEMAC Class B (Ru=0.50, Rv=0.70):
  1. Analytical SC
  2. Neural SC (PureNeuralCompGraphDecoder)
  3. Analytical SCL (L=4)
  4. Neural SCL (L=4)  — BemacNeuralSCLDecoder

Goal: verify the unverified Session-5 claim that NN-SCL beats analytical SC
by ~8x at N=64, AND give the proper apples-to-apples comparison
NN-SCL vs analytical SCL.

For each N in {32, 64, 128} and each decoder:
  - run 10K codewords (or until ≥ 30 errors)
  - report BLER + Wilson 95% CI
  - report wall-clock decode time per codeword
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
from polar.decoder_scl import decode_single_list

from eval_bemac_nn_scl import (
    load_bemac_model, load_bemac_design,
    BemacNeuralSCLDecoder,
)

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results/bemac/bemac_classB_Ru50_Rv70_four_way"
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Wilson 95% CI ────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


# ─── Codeword generation ──────────────────────────────────────────────────────

def gen_batch(N, Au, Av, batch_size, rng):
    """Returns (uf, vf, z) where uf,vf are int arrays (B,N) and z is (B,N) int64."""
    uf = np.zeros((batch_size, N), dtype=int)
    vf = np.zeros((batch_size, N), dtype=int)
    for p in Au: uf[:, p-1] = rng.integers(0, 2, batch_size)
    for p in Av: vf[:, p-1] = rng.integers(0, 2, batch_size)
    xf = polar_encode_batch(uf)
    yf = polar_encode_batch(vf)
    z_batch = (xf + yf).astype(np.int64)
    return uf, vf, z_batch


# ─── Decoder evaluators ───────────────────────────────────────────────────────

def eval_nn_sc(model, N, b, Au, Av, fu, fv, n_cw, batch_size, seed=2025):
    """Returns dict with errors, total, time_total."""
    model.eval()
    errs = 0
    total = 0
    t_total = 0.0
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf, vf, z_batch = gen_batch(N, Au, Av, actual, rng)
            zf = torch.from_numpy(z_batch).long()
            t0 = time.perf_counter()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            t_total += time.perf_counter() - t0
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e:
                    errs += 1
            total += actual
    return {'errors': errs, 'total': total, 'time_total': t_total}


def eval_analytical_sc(N, b, Au, Av, fu, fv, channel, n_cw, seed=2025):
    errs = 0
    total = 0
    t_total = 0.0
    rng = np.random.default_rng(seed)
    while total < n_cw:
        uf, vf, z_batch = gen_batch(N, Au, Av, 1, rng)
        z = z_batch[0].astype(int).tolist()
        t0 = time.perf_counter()
        u_dec, v_dec = decode_single(N, z, b, fu, fv, channel)
        t_total += time.perf_counter() - t0
        e = any(u_dec[p-1] != uf[0, p-1] for p in Au) or \
            any(v_dec[p-1] != vf[0, p-1] for p in Av)
        if e:
            errs += 1
        total += 1
    return {'errors': errs, 'total': total, 'time_total': t_total}


def eval_analytical_scl(N, b, Au, Av, fu, fv, channel, n_cw, L=4, seed=2025):
    errs = 0
    total = 0
    t_total = 0.0
    rng = np.random.default_rng(seed)
    while total < n_cw:
        uf, vf, z_batch = gen_batch(N, Au, Av, 1, rng)
        z = z_batch[0].astype(int).tolist()
        t0 = time.perf_counter()
        u_dec, v_dec = decode_single_list(N, z, b, fu, fv, channel, L=L)
        t_total += time.perf_counter() - t0
        e = any(u_dec[p-1] != uf[0, p-1] for p in Au) or \
            any(v_dec[p-1] != vf[0, p-1] for p in Av)
        if e:
            errs += 1
        total += 1
    return {'errors': errs, 'total': total, 'time_total': t_total}


def eval_nn_scl(model, N, b, Au, Av, fu, fv, n_cw, L=4, seed=2025):
    decoder = BemacNeuralSCLDecoder(model, L=L)
    errs = 0
    total = 0
    t_total = 0.0
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        while total < n_cw:
            uf, vf, z_batch = gen_batch(N, Au, Av, 1, rng)
            zf = torch.from_numpy(z_batch).long()
            z_single = zf[0]
            t0 = time.perf_counter()
            uh, vh = decoder.decode(z_single, b, fu, fv)
            t_total += time.perf_counter() - t0
            e = any(uh.get(p, 0) != uf[0, p-1] for p in Au) or \
                any(vh.get(p, 0) != vf[0, p-1] for p in Av)
            if e:
                errs += 1
            total += 1
    return {'errors': errs, 'total': total, 'time_total': t_total}


# ─── Main loop ────────────────────────────────────────────────────────────────

def fmt_result(r):
    bler, lo, hi = wilson_ci(r['errors'], r['total'])
    ms_per_cw = r['time_total'] / r['total'] * 1000
    return {
        'errors': r['errors'],
        'total': r['total'],
        'bler': bler,
        'ci_lo': lo,
        'ci_hi': hi,
        'ms_per_cw': ms_per_cw,
    }


def main():
    RU, RV = 0.50, 0.70
    PLAN = {
        32:  {'nn_cw': 50000, 'sc_cw': 50000, 'scl_cw': 20000, 'nn_scl_cw': 20000, 'nn_bs': 100},
        64:  {'nn_cw': 50000, 'sc_cw': 50000, 'scl_cw': 20000, 'nn_scl_cw': 20000, 'nn_bs':  50},
        128: {'nn_cw': 30000, 'sc_cw': 30000, 'scl_cw': 10000, 'nn_scl_cw':  5000, 'nn_bs':  25},
    }

    channel = BEMAC()
    json_path = os.path.join(OUT_DIR, 'four_way_results.json')
    md_path = os.path.join(OUT_DIR, 'four_way_report.md')

    results = {
        'experiment': 'BEMAC Class B Ru=0.50 Rv=0.70 — four-way SC vs NN-SC vs SCL(L=4) vs NN-SCL(L=4)',
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data': {},
    }
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    for N in [32, 64, 128]:
        cfg = PLAN[N]
        print(f"\n{'='*72}")
        print(f"  N = {N}")
        print(f"{'='*72}", flush=True)

        Au, Av, fu, fv, ku, kv = load_bemac_design(N, RU, RV)
        b = make_path(N, N // 2)
        model = load_bemac_model(N)
        print(f"  ku={ku}, kv={kv}, model_params={model.count_parameters():,}", flush=True)

        per_N = {'N': N, 'ku': ku, 'kv': kv}

        # 1. Analytical SC
        print(f"\n  --- Analytical SC ({cfg['sc_cw']} cw) ---", flush=True)
        t0 = time.time()
        r = eval_analytical_sc(N, b, Au, Av, fu, fv, channel, cfg['sc_cw'])
        per_N['sc'] = fmt_result(r)
        print(f"    SC:      {r['errors']:>5d} errs / {r['total']:>5d} cw  BLER={per_N['sc']['bler']:.2e}  CI[{per_N['sc']['ci_lo']:.2e},{per_N['sc']['ci_hi']:.2e}]  {per_N['sc']['ms_per_cw']:.3f} ms/cw  [wall {time.time()-t0:.0f}s]", flush=True)

        # 2. Neural SC
        print(f"\n  --- Neural SC ({cfg['nn_cw']} cw, batch={cfg['nn_bs']}) ---", flush=True)
        t0 = time.time()
        r = eval_nn_sc(model, N, b, Au, Av, fu, fv, cfg['nn_cw'], cfg['nn_bs'])
        per_N['nn_sc'] = fmt_result(r)
        print(f"    NN-SC:   {r['errors']:>5d} errs / {r['total']:>5d} cw  BLER={per_N['nn_sc']['bler']:.2e}  CI[{per_N['nn_sc']['ci_lo']:.2e},{per_N['nn_sc']['ci_hi']:.2e}]  {per_N['nn_sc']['ms_per_cw']:.3f} ms/cw  [wall {time.time()-t0:.0f}s]", flush=True)

        # 3. Analytical SCL(L=4)
        print(f"\n  --- Analytical SCL(L=4) ({cfg['scl_cw']} cw) ---", flush=True)
        t0 = time.time()
        r = eval_analytical_scl(N, b, Au, Av, fu, fv, channel, cfg['scl_cw'], L=4)
        per_N['scl4'] = fmt_result(r)
        print(f"    SCL4:    {r['errors']:>5d} errs / {r['total']:>5d} cw  BLER={per_N['scl4']['bler']:.2e}  CI[{per_N['scl4']['ci_lo']:.2e},{per_N['scl4']['ci_hi']:.2e}]  {per_N['scl4']['ms_per_cw']:.3f} ms/cw  [wall {time.time()-t0:.0f}s]", flush=True)

        # 4. Neural SCL(L=4)
        print(f"\n  --- Neural SCL(L=4) ({cfg['nn_scl_cw']} cw) ---", flush=True)
        t0 = time.time()
        r = eval_nn_scl(model, N, b, Au, Av, fu, fv, cfg['nn_scl_cw'], L=4)
        per_N['nn_scl4'] = fmt_result(r)
        print(f"    NN-SCL4: {r['errors']:>5d} errs / {r['total']:>5d} cw  BLER={per_N['nn_scl4']['bler']:.2e}  CI[{per_N['nn_scl4']['ci_lo']:.2e},{per_N['nn_scl4']['ci_hi']:.2e}]  {per_N['nn_scl4']['ms_per_cw']:.3f} ms/cw  [wall {time.time()-t0:.0f}s]", flush=True)

        # Compute interesting ratios
        sc_b = per_N['sc']['bler']
        scl_b = per_N['scl4']['bler']
        nn_sc_b = per_N['nn_sc']['bler']
        nn_scl_b = per_N['nn_scl4']['bler']
        per_N['ratios'] = {
            'nn_sc_vs_sc':     (nn_sc_b / sc_b)  if sc_b > 0  else None,
            'nn_scl_vs_sc':    (nn_scl_b / sc_b) if sc_b > 0  else None,
            'nn_scl_vs_scl':   (nn_scl_b / scl_b) if scl_b > 0 else None,
            'nn_scl_vs_nn_sc': (nn_scl_b / nn_sc_b) if nn_sc_b > 0 else None,
        }

        print(f"\n  RATIOS at N={N}:")
        for k, v in per_N['ratios'].items():
            if v is not None:
                print(f"    {k:<22s}  {v:.3f}x")

        results['data'][str(N)] = per_N
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Markdown report
    lines = []
    lines.append("# BEMAC Class B (Ru=0.50, Rv=0.70) — Four-way decoder comparison\n")
    lines.append("Decoders: SC, Neural-SC, SCL(L=4), Neural-SCL(L=4). All four use the same frozen sets.\n")
    lines.append("\n## Summary table\n")
    lines.append("| N | SC BLER | NN-SC BLER | SCL(4) BLER | NN-SCL(4) BLER | NN-SC/SC | NN-SCL/SCL | NN-SCL/SC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for N in [32, 64, 128]:
        d = results['data'].get(str(N))
        if not d: continue
        r = d['ratios']
        lines.append(
            f"| {N} | {d['sc']['bler']:.2e} | {d['nn_sc']['bler']:.2e} | "
            f"{d['scl4']['bler']:.2e} | {d['nn_scl4']['bler']:.2e} | "
            f"{r['nn_sc_vs_sc']:.2f} | {r['nn_scl_vs_scl']:.2f} | {r['nn_scl_vs_sc']:.2f} |"
        )
    lines.append("\n## Timing (ms per codeword)\n")
    lines.append("| N | SC | NN-SC | SCL(4) | NN-SCL(4) |")
    lines.append("|---|---|---|---|---|")
    for N in [32, 64, 128]:
        d = results['data'].get(str(N))
        if not d: continue
        lines.append(
            f"| {N} | {d['sc']['ms_per_cw']:.3f} | {d['nn_sc']['ms_per_cw']:.3f} | "
            f"{d['scl4']['ms_per_cw']:.3f} | {d['nn_scl4']['ms_per_cw']:.3f} |"
        )

    with open(md_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == '__main__':
    main()
