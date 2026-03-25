"""
bler_vs_N_bemac.py — BLER vs block length for BE-MAC, all code classes.

Analytical (Bhattacharyya) design, SC (L=1), fixed rho=0.7.

Usage:
    python scripts/bler_vs_N_bemac.py --class A
    python scripts/bler_vs_N_bemac.py --class C
    python scripts/bler_vs_N_bemac.py --class A B C   # all three
"""

import os
import sys
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.encoder import polar_encode, polar_encode_batch, build_message, build_message_batch
from polar.channels import BEMAC
from polar.design import design_bemac, make_path
from polar.decoder import decode_batch

# ════════════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════════════

N_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024]
RHO = 0.693
TARGET_CW = 50000
MAX_TIME_PER_N = 600  # 10 min max per N
SEED = 42

CLASS_CONFIGS = {
    "A": {"Ru_dir": 0.75, "Rv_dir": 0.75, "path_i_frac": 0.375},
    "B": {"Ru_dir": 0.625, "Rv_dir": 0.875, "path_i_frac": 0.5},
    "C": {"Ru_dir": 0.5,  "Rv_dir": 1.0,   "path_i_frac": 1.0},
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg=""):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v, b, batch_size, rng, channel):
    info_u = rng.integers(0, 2, size=(batch_size, ku))
    info_v = rng.integers(0, 2, size=(batch_size, kv))
    U = build_message_batch(N, info_u, Au)
    V = build_message_batch(N, info_v, Av)
    X = polar_encode_batch(U)
    Y = polar_encode_batch(V)
    Z = channel.sample_batch(X, Y)

    results = decode_batch(N, Z.tolist(), b, frozen_u, frozen_v,
                           channel, vectorized=True)

    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])

    block_errors = 0
    u_bit_errors = 0
    v_bit_errors = 0
    for i, (u_dec, v_dec) in enumerate(results):
        ue = int(np.sum(np.array(u_dec)[u_info_idx] != info_u[i]))
        ve = int(np.sum(np.array(v_dec)[v_info_idx] != info_v[i]))
        u_bit_errors += ue
        v_bit_errors += ve
        if ue > 0 or ve > 0:
            block_errors += 1
    return block_errors, u_bit_errors, v_bit_errors


def run_class(code_class):
    cfg = CLASS_CONFIGS[code_class]
    channel = BEMAC()

    log(f"BLER vs N — BE-MAC Class {code_class}, SC (L=1), rho={RHO}")
    log(f"  Direction: ({cfg['Ru_dir']}, {cfg['Rv_dir']})")
    log(f"  N values: {N_VALUES}")
    log()

    all_results = []

    for N in N_VALUES:
        n = N.bit_length() - 1
        path_i = round(cfg["path_i_frac"] * N)
        b = make_path(N, path_i)

        ku = max(1, round(RHO * cfg["Ru_dir"] * N))
        kv = max(1, round(RHO * cfg["Rv_dir"] * N))
        ku = min(ku, N - 1)
        kv = min(kv, N - 1)
        Ru = ku / N
        Rv = kv / N

        log(f"N={N:>5d} (n={n:>2d}): ku={ku}, kv={kv}, Ru={Ru:.4f}, Rv={Rv:.4f}, "
            f"Ru+Rv={Ru+Rv:.4f}, path_i={path_i}")

        Au, Av, frozen_u, frozen_v, z_u, z_v = design_bemac(n, ku, kv)

        # Benchmark
        rng = np.random.default_rng(SEED)
        bs_test = min(20, TARGET_CW)
        t0 = time.perf_counter()
        sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v, b, bs_test, rng, channel)
        ms_per_cw = (time.perf_counter() - t0) / bs_test * 1000

        n_cw = min(TARGET_CW, max(5000, int(MAX_TIME_PER_N / (ms_per_cw / 1000))))
        batch_size = min(500, n_cw)

        log(f"  {ms_per_cw:.1f} ms/cw, running {n_cw} codewords...")

        block_errors = 0
        u_bit_errors = 0
        v_bit_errors = 0
        cw_done = 0
        t0 = time.time()

        while cw_done < n_cw:
            bs = min(batch_size, n_cw - cw_done)
            rng = np.random.default_rng(SEED + cw_done + n * 100000)
            be, ube, vbe = sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v,
                                     b, bs, rng, channel)
            block_errors += be
            u_bit_errors += ube
            v_bit_errors += vbe
            cw_done += bs

        elapsed = time.time() - t0
        bler = block_errors / n_cw
        ber_u = u_bit_errors / max(1, n_cw * ku)
        ber_v = v_bit_errors / max(1, n_cw * kv)

        log(f"  BLER={bler:.6f}  ({block_errors}/{n_cw})  {elapsed:.1f}s")

        all_results.append({
            "N": N, "n": n, "ku": ku, "kv": kv,
            "Ru": round(Ru, 6), "Rv": round(Rv, 6),
            "bler": bler, "ber_u": ber_u, "ber_v": ber_v,
            "block_errors": block_errors, "n_codewords": n_cw,
            "time_s": round(elapsed, 2),
            "path_i": path_i,
        })
        log()

    # Save JSON
    rho_tag = f"_rho{RHO:.1f}" if RHO != 0.5 else ""
    json_path = os.path.join(RESULTS_DIR, f"bler_vs_N_bemac_class{code_class}_L1{rho_tag}.json")
    output = {
        "description": f"BLER vs N — BE-MAC Class {code_class}, SC (L=1)",
        "channel": "BE-MAC",
        "class": code_class, "L": 1,
        "rho": RHO,
        "Ru_dir": cfg["Ru_dir"], "Rv_dir": cfg["Rv_dir"],
        "path_i_frac": cfg["path_i_frac"],
        "sum_rate_target": RHO * (cfg["Ru_dir"] + cfg["Rv_dir"]),
        "design": "analytical",
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Results saved to {json_path}")

    # Print summary
    log()
    log("=" * 70)
    log(f"  {'N':>5s}  {'n':>2s}  {'ku':>4s}  {'kv':>4s}  {'Ru+Rv':>7s}  {'BLER':>10s}  {'n_cw':>7s}")
    log("-" * 70)
    for r in all_results:
        log(f"  {r['N']:5d}  {r['n']:2d}  {r['ku']:4d}  {r['kv']:4d}  "
            f"{r['Ru']+r['Rv']:7.4f}  {r['bler']:10.6f}  {r['n_codewords']:7d}")
    log("=" * 70)

    return json_path, output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class", "-c", dest="classes", nargs="+",
                   choices=["A", "B", "C"], required=True)
    args = p.parse_args()

    for cls in args.classes:
        run_class(cls)
        log()


if __name__ == "__main__":
    main()
