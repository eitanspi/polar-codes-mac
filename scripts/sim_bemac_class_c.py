"""
sim_bemac_class_c.py
====================
Reproduce code class C = (0.5, 1) results from Önay ISIT 2013 for BE-MAC
with L=1 (SC decoder, no list).

Run:
    cd to_git
    python scripts/sim_bemac_class_c.py
"""

import os
import sys
import time
import json
import warnings
import numpy as np

# Suppress inf-arithmetic warnings from the decoder's f/g nodes (handled by fallback logic)
warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.encoder import polar_encode, build_message
from polar.channels import BEMAC
from polar.design import design_bemac, make_path
from polar.decoder import decode_single

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════

BLOCK_LENGTHS = [10, 12, 14]  # n values → N = 2^n
RHO_VALUES = np.linspace(0.5, 0.95, 8)
TOTAL_BUDGET_S = 2.0 * 3600  # 2 hours
SEED = 42
CHANNEL = BEMAC()
BENCHMARK_CODEWORDS = 30  # enough to measure multiprocessing overhead
BLER_SKIP_THRESHOLD = 0.5  # skip remaining rho values for this N when BLER exceeds this

# ════════════════════════════════════════════════════════════════════
#  OUTPUT
# ════════════════════════════════════════════════════════════════════

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "sim_bemac_class_c.json")


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════
#  WORKER FOR MULTIPROCESSING
# ════════════════════════════════════════════════════════════════════

def _sim_one_codeword(args):
    """
    Simulate one codeword: encode → channel → decode → check errors.
    Returns (block_error, u_bit_errors, v_bit_errors).
    """
    N, n, ku, kv, Au, Av, frozen_u, frozen_v, b, seed_i = args
    rng = np.random.default_rng(seed_i)
    channel = BEMAC()

    info_u = rng.integers(0, 2, size=ku).tolist()
    info_v = rng.integers(0, 2, size=kv).tolist()

    u = build_message(N, info_u, Au)
    v = build_message(N, info_v, Av)

    x = polar_encode(u.tolist())
    y = polar_encode(v.tolist())
    z = channel.sample_batch(np.array(x), np.array(y)).tolist()

    u_dec, v_dec = decode_single(N, z, b, frozen_u, frozen_v, channel)

    ue = sum(1 for p, bit in zip(Au, info_u) if u_dec[p - 1] != bit)
    ve = sum(1 for p, bit in zip(Av, info_v) if v_dec[p - 1] != bit)

    return (1 if (ue > 0 or ve > 0) else 0, ue, ve)


# ════════════════════════════════════════════════════════════════════
#  BENCHMARK
# ════════════════════════════════════════════════════════════════════

def benchmark(n_values, n_workers):
    """
    Benchmark wall-clock throughput at each N using the same multiprocessing
    setup as the real simulation, so time estimates are accurate.
    """
    timings = {}

    for n in n_values:
        N = 1 << n
        ku = round(0.7 * 0.5 * N)  # mid-range ρ
        kv = round(0.7 * 1.0 * N)
        Au, Av, frozen_u, frozen_v, _, _ = design_bemac(n, ku, kv)
        b = make_path(N, path_i=N)

        base_seed = SEED + n * 100000
        args_list = [
            (N, n, ku, kv, Au, Av, frozen_u, frozen_v, b, base_seed + i)
            for i in range(BENCHMARK_CODEWORDS)
        ]

        t0 = time.perf_counter()
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(_sim_one_codeword, args_list,
                                  chunksize=max(1, BENCHMARK_CODEWORDS // n_workers)))
        else:
            for a in args_list:
                _sim_one_codeword(a)
        elapsed = time.perf_counter() - t0

        wall_per_cw = elapsed / BENCHMARK_CODEWORDS
        timings[n] = wall_per_cw
        log(f"  N={N:>5d} (n={n:>2d}): {wall_per_cw*1000:.1f} ms/codeword wall-clock "
            f"({BENCHMARK_CODEWORDS} cw, {n_workers} workers, {elapsed:.2f}s)")

    return timings


# ════════════════════════════════════════════════════════════════════
#  MAIN SIMULATION
# ════════════════════════════════════════════════════════════════════

def main():
    t_global = time.time()

    # Determine number of workers for multiprocessing
    n_workers = min(os.cpu_count() or 1, 8)

    log("sim_bemac_class_c.py — BE-MAC Code Class C = (0.5, 1)")
    log(f"  Block lengths: N = {[1 << n for n in BLOCK_LENGTHS]}")
    log(f"  Rho values: {RHO_VALUES.tolist()}")
    log(f"  Total time budget: {TOTAL_BUDGET_S/3600:.1f} hours")
    log(f"  Workers: {n_workers}")
    log(f"  Seed: {SEED}")
    log()

    # ── Step 1: Benchmark (with multiprocessing, so wall-clock is accurate)
    log("=== Benchmarking decode speed (wall-clock with multiprocessing) ===")
    timings = benchmark(BLOCK_LENGTHS, n_workers)
    log()

    # Compute codewords per (N, rho) point
    n_block_lengths = len(BLOCK_LENGTHS)
    n_rho = len(RHO_VALUES)
    time_per_N = TOTAL_BUDGET_S / n_block_lengths
    time_per_point = time_per_N / n_rho

    codewords_per_point = {}
    for n in BLOCK_LENGTHS:
        cw = max(200, int(time_per_point / timings[n]))
        codewords_per_point[n] = cw
        N = 1 << n
        log(f"  N={N:>5d}: {cw} codewords/point "
            f"(~{cw * timings[n]:.0f}s per point, "
            f"~{cw * timings[n] * n_rho / 60:.0f} min total)")
    log()

    # ── Step 2: Main simulation loop
    all_results = []
    benchmark_meta = {n: {"N": 1 << n, "ms_per_codeword": timings[n] * 1000}
                      for n in BLOCK_LENGTHS}
    time_spent = 0.0

    for n_idx, n in enumerate(BLOCK_LENGTHS):
        N = 1 << n
        b = make_path(N, path_i=N)

        remaining_budget = TOTAL_BUDGET_S - time_spent
        remaining_Ns = len(BLOCK_LENGTHS) - n_idx
        time_for_this_N = remaining_budget / remaining_Ns
        n_cw = max(200, int((time_for_this_N / n_rho) / timings[n]))

        log("=" * 65)
        log(f"  N = {N}  (n = {n}),  {n_cw} codewords per rate point")
        log(f"  (budget for this N: ~{time_for_this_N/60:.0f} min)")
        log("=" * 65)

        skipped = False
        for rho_idx, rho in enumerate(RHO_VALUES):
            ku = round(rho * 0.5 * N)
            kv = round(rho * 1.0 * N)
            Ru = ku / N
            Rv = kv / N

            if skipped:
                result_row = {
                    "N": N, "n": n,
                    "Ru": round(Ru, 6), "Rv": round(Rv, 6),
                    "rho": round(float(rho), 6),
                    "ku": ku, "kv": kv, "L": 1,
                    "bler": None, "ber_u": None, "ber_v": None,
                    "block_errors": None, "n_codewords": 0,
                    "time_s": 0, "path_i": N, "skipped": True,
                }
                all_results.append(result_row)
                log(f"  rho={rho:.4f}  SKIPPED (previous BLER > {BLER_SKIP_THRESHOLD})")
                continue

            Au, Av, frozen_u, frozen_v, _, _ = design_bemac(n, ku, kv)

            base_seed = SEED + n * 100000 + rho_idx * 10000
            args_list = [
                (N, n, ku, kv, Au, Av, frozen_u, frozen_v, b, base_seed + i)
                for i in range(n_cw)
            ]

            t0 = time.time()

            if n_workers > 1 and n_cw > 10:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(
                        _sim_one_codeword, args_list,
                        chunksize=max(1, n_cw // (n_workers * 4))))
            else:
                results = [_sim_one_codeword(args) for args in args_list]

            elapsed = time.time() - t0

            block_errors = sum(r[0] for r in results)
            u_bit_errors = sum(r[1] for r in results)
            v_bit_errors = sum(r[2] for r in results)
            bler = block_errors / n_cw
            ber_u = u_bit_errors / max(1, n_cw * ku)
            ber_v = v_bit_errors / max(1, n_cw * kv)

            result_row = {
                "N": N, "n": n,
                "Ru": round(Ru, 6), "Rv": round(Rv, 6),
                "rho": round(float(rho), 6),
                "ku": ku, "kv": kv, "L": 1,
                "bler": bler,
                "ber_u": ber_u, "ber_v": ber_v,
                "block_errors": block_errors,
                "u_bit_errors": u_bit_errors, "v_bit_errors": v_bit_errors,
                "n_codewords": n_cw,
                "time_s": round(elapsed, 2),
                "path_i": N,
            }
            all_results.append(result_row)

            log(f"  rho={rho:.4f}  Ru={Ru:.4f}  Rv={Rv:.4f}  "
                f"ku={ku}  kv={kv}  "
                f"BLER={bler:.4f}  BER_u={ber_u:.4e}  BER_v={ber_v:.4e}  "
                f"({block_errors}/{n_cw})  {elapsed:.1f}s")

            if bler > BLER_SKIP_THRESHOLD:
                skipped = True
                log(f"  ** BLER {bler:.4f} > {BLER_SKIP_THRESHOLD} — "
                    f"skipping remaining rho for N={N} **")

            # Checkpoint after each rate point
            output = {
                "description": "BE-MAC Code Class C = (0.5, 1), Onay ISIT 2013",
                "channel": "BE-MAC",
                "decoder": "SC (L=1), efficient O(N log N)",
                "design": "analytical Bhattacharyya",
                "path": "0^N 1^N (path_i=N)",
                "seed": SEED,
                "bler_skip_threshold": BLER_SKIP_THRESHOLD,
                "timestamp": datetime.now().isoformat(),
                "benchmark": benchmark_meta,
                "results": all_results,
            }
            with open(JSON_PATH, "w") as f:
                json.dump(output, f, indent=2)

            time_spent += elapsed

        log()

    # ── Summary
    total_time = time.time() - t_global

    log("=" * 105)
    log("  SUMMARY")
    log("=" * 105)
    hdr = (f"{'N':>6} {'rho':>6} {'Ru':>7} {'Rv':>7} {'ku':>5} {'kv':>5} "
           f"{'BLER':>10} {'BER_u':>10} {'BER_v':>10} "
           f"{'blk_err':>7} {'n_cw':>6} {'time':>7}")
    log(hdr)
    log("-" * 105)
    for r in all_results:
        if r.get("skipped"):
            log(f"{r['N']:6d} {r['rho']:6.3f} {r['Ru']:7.4f} {r['Rv']:7.4f} "
                f"{r['ku']:5d} {r['kv']:5d}    SKIPPED")
        else:
            log(f"{r['N']:6d} {r['rho']:6.3f} {r['Ru']:7.4f} {r['Rv']:7.4f} "
                f"{r['ku']:5d} {r['kv']:5d} "
                f"{r['bler']:10.4f} {r['ber_u']:10.4e} {r['ber_v']:10.4e} "
                f"{r['block_errors']:7d} {r['n_codewords']:6d} {r['time_s']:6.1f}s")
    log("-" * 105)
    log(f"Total runtime: {total_time:.0f}s ({total_time/3600:.2f}h)")
    log(f"Results saved to {JSON_PATH}")
    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log()
        log("*** Interrupted by user ***")
        sys.exit(1)
