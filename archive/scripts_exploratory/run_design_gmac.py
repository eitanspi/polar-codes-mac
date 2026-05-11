"""
run_design_gmac.py
==================
MC genie design for Gaussian MAC polar codes.

Adapts run_design.py for GaussianMAC channel.
Uses the O(N log N) computational-graph genie decoder.

Usage:
    cd to_git
    python scripts/run_design_gmac.py --class C --N 8 16 32 64 128 256 512 1024 --hours 2 --snr-db 6
    python scripts/run_design_gmac.py --class B --N 1024 --hours 1 --snr-db 3
    python scripts/run_design_gmac.py --class A --N 1024 --hours 1 --snr-db 6
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.encoder import polar_encode
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.decoder import (
    build_log_W_leaf, _CompGraph, _norm_prod_single,
    _NEG_INF, _LOG_HALF, _LOG_QUARTER,
)

# ════════════════════════════════════════════════════════════════════
#  CODE CLASS CONFIGURATIONS
#  Same path_i fractions as BEMAC classes
# ════════════════════════════════════════════════════════════════════

CLASS_CONFIG = {
    "A": {"path_i": lambda N: round(0.375 * N)},
    "B": {"path_i": lambda N: round(0.5 * N)},
    "C": {"path_i": lambda N: N},
}

N_WORKERS = 8
BENCHMARK_CODEWORDS = 20

# Module-level channel (set by main, used by workers)
_CHANNEL = None


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════
#  O(N log N) GENIE-AIDED DECODER  (same as run_design.py)
# ════════════════════════════════════════════════════════════════════

def _genie_decode_interleaved(N, z, b, u_true, v_true, channel):
    """O(N log N) genie-aided SC decode for arbitrary monotone chain paths."""
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
    N, b, seed_i, sigma2 = args
    rng = np.random.default_rng(seed_i)
    channel = GaussianMAC(sigma2=sigma2)

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

def benchmark_speed(N, b, sigma2, n_workers, pool=None):
    base_seed = 999_000
    args_list = [(N, b, base_seed + i, sigma2) for i in range(BENCHMARK_CODEWORDS)]

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

def run_design_one_N(N, b, sigma2, time_budget_s, seed, n_trials_override,
                     n_workers, pool=None):
    log(f"  Benchmarking N={N} ...")
    wall_per_cw = benchmark_speed(N, b, sigma2, n_workers, pool=pool)

    if n_trials_override is not None:
        n_trials = n_trials_override
    else:
        n_trials = max(100, int(time_budget_s / wall_per_cw))

    log(f"  N={N}: {n_trials} trials, budget={time_budget_s:.0f}s, "
        f"est. {n_trials * wall_per_cw:.0f}s")

    u_err_counts = np.zeros(N, dtype=np.float64)
    v_err_counts = np.zeros(N, dtype=np.float64)
    completed = 0

    batch_size = max(1, n_workers * 4)
    t0 = time.time()
    report_interval = 10

    trial_idx = 0
    while trial_idx < n_trials:
        if time.time() - t0 > time_budget_s:
            log(f"  Time budget reached after {completed} trials")
            break

        end = min(trial_idx + batch_size, n_trials)
        args_list = [(N, b, seed + i, sigma2) for i in range(trial_idx, end)]

        if pool is not None and len(args_list) > 1:
            results = list(pool.map(_genie_one_codeword, args_list))
        else:
            results = [_genie_one_codeword(a) for a in args_list]

        for u_err, v_err in results:
            u_err_counts += u_err
            v_err_counts += v_err
            completed += 1

        trial_idx = end

        if completed >= report_interval:
            elapsed = time.time() - t0
            rate = completed / elapsed
            remaining = time_budget_s - elapsed
            log(f"  trial {completed}/{n_trials}  "
                f"({elapsed:.0f}s, {rate:.1f} trials/s, "
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

    log(f"  Done: {completed} trials in {total:.1f}s")
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
        description="MC genie design for polar codes on Gaussian MAC")
    parser.add_argument("--class", dest="code_class", required=True,
                        choices=["A", "B", "C"])
    parser.add_argument("--N", type=int, nargs="+", required=True)
    parser.add_argument("--snr-db", type=float, required=True,
                        help="Per-user SNR in dB")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    for N in args.N:
        if N < 2 or (N & (N - 1)) != 0:
            parser.error(f"N={N} is not a power of 2")

    sigma2 = 10.0 ** (-args.snr_db / 10.0)
    channel = GaussianMAC(sigma2=sigma2)
    I_ZX, I_ZY_X, I_ZXY = channel.capacity()

    cfg = CLASS_CONFIG[args.code_class]
    n_workers = min(os.cpu_count() or 1, N_WORKERS)
    total_budget_s = args.hours * 3600
    budget_per_N = total_budget_s / len(args.N)

    designs_dir = os.path.join(os.path.dirname(__file__), "..", "designs")
    os.makedirs(designs_dir, exist_ok=True)

    log(f"run_design_gmac.py — MC genie design for Gaussian MAC")
    log(f"  Class {args.code_class}")
    log(f"  SNR = {args.snr_db} dB, sigma2 = {sigma2:.4f}")
    log(f"  Capacity: I(Z;X)={I_ZX:.4f}, I(Z;Y|X)={I_ZY_X:.4f}, "
        f"I(Z;X,Y)={I_ZXY:.4f}")
    log(f"  N values: {args.N}")
    log(f"  Budget: {args.hours}h ({total_budget_s:.0f}s), {budget_per_N:.0f}s per N")
    log(f"  Workers: {n_workers}")
    log()

    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None

    try:
        for N in args.N:
            n = N.bit_length() - 1
            path_i = cfg["path_i"](N)
            snr_tag = f"snr{args.snr_db:.0f}dB"
            out_path = os.path.join(designs_dir,
                                    f"gmac_{args.code_class}_n{n}_{snr_tag}.npz")

            log("=" * 65)
            log(f"  N={N} (n={n}), path_i={path_i}")
            log(f"  Output: {out_path}")
            log("=" * 65)

            if os.path.exists(out_path) and not args.force:
                log(f"  Already exists, skipping. Use --force to overwrite.")
                log()
                continue

            b = make_path(N, path_i)

            error_rates_u, error_rates_v, n_completed = run_design_one_N(
                N, b, sigma2, budget_per_N, args.seed, args.trials, n_workers,
                pool=pool)

            np.savez(out_path,
                     u_error_rates=error_rates_u,
                     v_error_rates=error_rates_v,
                     path_i=np.array(path_i),
                     n_trials=np.array(n_completed),
                     seed=np.array(args.seed),
                     sigma2=np.array(sigma2),
                     snr_db=np.array(args.snr_db))
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
