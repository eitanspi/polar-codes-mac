"""
run_design.py
=============
Unified MC genie design script for all three code classes (A, B, C).

Usage:
    python scripts/run_design.py --class C --N 1024 4096 16384 --hours 1
    python scripts/run_design.py --class B --N 1024 --hours 0.5
    python scripts/run_design.py --class A --N 1024 4096 --hours 2

Run from the project root (to_git/).
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="invalid value encountered",
                        category=RuntimeWarning)

from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Project root imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.encoder import polar_encode
from polar.channels import BEMAC
from polar.design import make_path
from polar.decoder import (
    build_log_W_leaf, _CompGraph, _norm_prod_single,
    _NEG_INF, _LOG_HALF, _LOG_QUARTER,
)

# ════════════════════════════════════════════════════════════════════
#  CODE CLASS CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════

CLASS_CONFIG = {
    "A": {"Ru": 0.75,  "Rv": 0.75,  "path_i": lambda N: round(0.375 * N)},
    "B": {"Ru": 0.625, "Rv": 0.875, "path_i": lambda N: round(0.5 * N)},
    "C": {"Ru": 0.5,   "Rv": 1.0,   "path_i": lambda N: N},
}

N_WORKERS = 8
BENCHMARK_CODEWORDS = 20


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════
#  O(N log N) GENIE-AIDED DECODER
# ════════════════════════════════════════════════════════════════════

def _genie_decode_interleaved(N, z, b, u_true, v_true, channel):
    """
    O(N log N) genie-aided SC decode for arbitrary monotone chain paths.

    Same algorithm as decoder.decode_single, but:
    - Always feeds the TRUE bit value (genie), not the decoded value
    - Records whether each bit's hard decision was correct
    """
    n = N.bit_length() - 1
    log_W = build_log_W_leaf(z, channel)
    graph = _CompGraph(n, log_W)

    u_hat = {}
    v_hat = {}
    i_u = 0
    i_v = 0

    u_errors = np.zeros(N, dtype=np.int32)
    v_errors = np.zeros(N, dtype=np.int32)

    for step in range(2 * N):
        gamma = b[step]

        if gamma == 0:
            i_u += 1
            i_t = i_u
        else:
            i_v += 1
            i_t = i_v

        leaf_edge = i_t + N - 1
        target_vertex = leaf_edge >> 1

        graph.step_to(target_vertex)

        temp = graph.edge_data[leaf_edge][0].copy()

        if leaf_edge & 1 == 0:
            graph.calc_left(target_vertex)
        else:
            graph.calc_right(target_vertex)

        top_down = graph.edge_data[leaf_edge][0]
        combined = _norm_prod_single(top_down, temp)

        # Hard decision
        if gamma == 0:
            p0 = np.logaddexp(combined[0, 0], combined[0, 1])
            p1 = np.logaddexp(combined[1, 0], combined[1, 1])
            decoded = 1 if p1 > p0 else 0
            true_val = u_true[i_t - 1]
            if decoded != true_val:
                u_errors[i_t - 1] = 1
            u_hat[i_t] = true_val
        else:
            p0 = np.logaddexp(combined[0, 0], combined[1, 0])
            p1 = np.logaddexp(combined[0, 1], combined[1, 1])
            decoded = 1 if p1 > p0 else 0
            true_val = v_true[i_t - 1]
            if decoded != true_val:
                v_errors[i_t - 1] = 1
            v_hat[i_t] = true_val

        # Set leaf to partially deterministic tensor (using TRUE values)
        new_leaf = np.full((2, 2), _NEG_INF, dtype=np.float64)
        u_val = u_hat.get(i_t)
        v_val = v_hat.get(i_t)

        if u_val is not None and v_val is not None:
            new_leaf[u_val, v_val] = 0.0
        elif u_val is not None:
            new_leaf[u_val, 0] = _LOG_HALF
            new_leaf[u_val, 1] = _LOG_HALF
        elif v_val is not None:
            new_leaf[0, v_val] = _LOG_HALF
            new_leaf[1, v_val] = _LOG_HALF
        else:
            new_leaf[:, :] = _LOG_QUARTER

        graph.edge_data[leaf_edge][0] = new_leaf

    return u_errors, v_errors


# ════════════════════════════════════════════════════════════════════
#  WORKER FOR MULTIPROCESSING
# ════════════════════════════════════════════════════════════════════

def _genie_one_codeword(args):
    """Process one genie-aided codeword. Returns (u_errors, v_errors)."""
    N, b, seed_i = args
    rng = np.random.default_rng(seed_i)
    channel = BEMAC()

    u = rng.integers(0, 2, size=N).tolist()
    v = rng.integers(0, 2, size=N).tolist()

    x = polar_encode(u)
    y = polar_encode(v)
    z = channel.sample_batch(np.array(x), np.array(y)).tolist()

    u_err, v_err = _genie_decode_interleaved(N, z, b, u, v, channel)
    return u_err, v_err


# ════════════════════════════════════════════════════════════════════
#  BENCHMARK
# ════════════════════════════════════════════════════════════════════

def benchmark_speed(N, b, n_workers, pool=None):
    """Benchmark genie decode speed, return seconds per codeword."""
    base_seed = 999_000
    args_list = [(N, b, base_seed + i) for i in range(BENCHMARK_CODEWORDS)]

    t0 = time.perf_counter()
    if pool is not None:
        list(pool.map(_genie_one_codeword, args_list,
                      chunksize=max(1, BENCHMARK_CODEWORDS // n_workers)))
    elif n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(_genie_one_codeword, args_list,
                              chunksize=max(1, BENCHMARK_CODEWORDS // n_workers)))
    else:
        for a in args_list:
            _genie_one_codeword(a)
    elapsed = time.perf_counter() - t0

    wall_per_cw = elapsed / BENCHMARK_CODEWORDS
    log(f"  Benchmark: {wall_per_cw*1000:.1f} ms/codeword "
        f"({BENCHMARK_CODEWORDS} cw, {n_workers} workers)")
    return wall_per_cw


# ════════════════════════════════════════════════════════════════════
#  MC GENIE DESIGN FOR ONE N
# ════════════════════════════════════════════════════════════════════

def run_design_one_N(N, b, time_budget_s, seed, n_trials_override, n_workers,
                     pool=None):
    """
    Run MC genie design for a single block length.

    Returns (u_error_rates, v_error_rates, n_trials_completed).
    """
    log(f"  Benchmarking N={N} ...")
    wall_per_cw = benchmark_speed(N, b, n_workers, pool=pool)

    if n_trials_override is not None:
        n_trials = n_trials_override
    else:
        n_trials = max(100, int(time_budget_s / wall_per_cw))

    log(f"  N={N}: {n_trials} trials, budget={time_budget_s:.0f}s, "
        f"est. {n_trials * wall_per_cw:.0f}s")

    u_err_counts = np.zeros(N, dtype=np.float64)
    v_err_counts = np.zeros(N, dtype=np.float64)
    completed = 0

    # Submit all trials at once to persistent pool for max throughput
    batch_size = max(1, n_workers * 4)
    t0 = time.time()
    report_interval = 10

    trial_idx = 0
    while trial_idx < n_trials:
        if time.time() - t0 > time_budget_s:
            log(f"  Time budget reached after {completed} trials")
            break

        # Build batch
        end = min(trial_idx + batch_size, n_trials)
        args_list = [(N, b, seed + i) for i in range(trial_idx, end)]

        if pool is not None and len(args_list) > 1:
            results = list(pool.map(_genie_one_codeword, args_list))
        elif n_workers > 1 and len(args_list) > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_genie_one_codeword, args_list))
        else:
            results = [_genie_one_codeword(a) for a in args_list]

        for u_err, v_err in results:
            u_err_counts += u_err
            v_err_counts += v_err
            completed += 1

        trial_idx = end

        # Progress reporting
        if completed >= report_interval:
            elapsed = time.time() - t0
            rate = completed / elapsed
            remaining = time_budget_s - elapsed
            eta_trials = min(n_trials - completed, int(rate * remaining))
            log(f"  trial {completed}/{n_trials}  "
                f"({elapsed:.0f}s, {rate:.1f} trials/s, "
                f"~{eta_trials} more, "
                f"~{max(0, remaining):.0f}s left)")
            if completed < 100:
                report_interval = completed + 10
            elif completed < 1000:
                report_interval = completed + 100
            else:
                report_interval = completed + 500

    total = time.time() - t0
    error_rates_u = u_err_counts / max(completed, 1)
    error_rates_v = v_err_counts / max(completed, 1)

    log(f"  Done: {completed} trials in {total:.1f}s ({completed/max(total,0.01):.1f} trials/s)")
    log(f"  U: {int(np.sum(error_rates_u < 0.001))} ch P_e<0.001, "
        f"{int(np.sum(error_rates_u < 0.01))} ch P_e<0.01, "
        f"{int(np.sum(error_rates_u < 0.1))} ch P_e<0.1")
    log(f"  V: {int(np.sum(error_rates_v < 0.001))} ch P_e<0.001, "
        f"{int(np.sum(error_rates_v < 0.01))} ch P_e<0.01, "
        f"{int(np.sum(error_rates_v < 0.1))} ch P_e<0.1")

    return error_rates_u, error_rates_v, completed


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MC genie design for polar codes on BE-MAC")
    parser.add_argument("--class", dest="code_class", required=True,
                        choices=["A", "B", "C"],
                        help="Code class: A=(0.75,0.75), B=(0.625,0.875), C=(0.5,1.0)")
    parser.add_argument("--N", type=int, nargs="+", required=True,
                        help="Block lengths (must be powers of 2)")
    parser.add_argument("--hours", type=float, default=1.0,
                        help="Total time budget in hours (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--trials", type=int, default=None,
                        help="Override auto-computed trial count per N")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing design files")

    args = parser.parse_args()

    # Validate N values
    for N in args.N:
        if N < 2 or (N & (N - 1)) != 0:
            parser.error(f"N={N} is not a power of 2")

    cfg = CLASS_CONFIG[args.code_class]
    n_workers = min(os.cpu_count() or 1, N_WORKERS)
    total_budget_s = args.hours * 3600
    budget_per_N = total_budget_s / len(args.N)

    designs_dir = os.path.join(os.path.dirname(__file__), "..", "designs")
    os.makedirs(designs_dir, exist_ok=True)

    n_bits = {N: N.bit_length() - 1 for N in args.N}

    log(f"run_design.py — MC genie design")
    log(f"  Class {args.code_class}: (Ru, Rv) = ({cfg['Ru']}, {cfg['Rv']})")
    log(f"  N values: {args.N}")
    log(f"  Total budget: {args.hours}h ({total_budget_s:.0f}s), "
        f"{budget_per_N:.0f}s per N")
    log(f"  Seed: {args.seed}, Workers: {n_workers}")
    log()

    # Use a single persistent pool for the entire run
    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None

    try:
        for N in args.N:
            n = n_bits[N]
            path_i = cfg["path_i"](N)
            out_path = os.path.join(designs_dir,
                                    f"bemac_{args.code_class}_n{n}.npz")

            log("=" * 65)
            log(f"  N={N} (n={n}), path_i={path_i}")
            log(f"  Output: {out_path}")
            log("=" * 65)

            # Skip if exists
            if os.path.exists(out_path) and not args.force:
                log(f"  Already exists, skipping. Use --force to overwrite.")
                log()
                continue

            b = make_path(N, path_i)

            error_rates_u, error_rates_v, n_completed = run_design_one_N(
                N, b, budget_per_N, args.seed, args.trials, n_workers,
                pool=pool)

            # Save
            np.savez(out_path,
                     u_error_rates=error_rates_u,
                     v_error_rates=error_rates_v,
                     path_i=np.array(path_i),
                     n_trials=np.array(n_completed),
                     seed=np.array(args.seed))
            log(f"  Saved to {out_path}")
            log()

    finally:
        if pool is not None:
            pool.shutdown(wait=False)

    log("All done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log()
        log("*** Interrupted by user ***")
        sys.exit(1)
