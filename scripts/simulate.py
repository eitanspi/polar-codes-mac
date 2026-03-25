"""
simulate.py — Unified BE-MAC polar code simulation for all code classes.

Usage:
    cd to_git
    python scripts/simulate.py --class C --L 1
    python scripts/simulate.py --class C --L 32
    python scripts/simulate.py --class B --L 1 --N 1024 4096
    python scripts/simulate.py --class A --L 32 --hours 2
    python scripts/simulate.py --class C --L 1 --design mc
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.encoder import polar_encode, polar_encode_batch, build_message, build_message_batch
from polar.channels import BEMAC
from polar.design import design_bemac, make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_single, decode_batch
from polar.decoder_scl import decode_single_list, decode_batch_list

# ════════════════════════════════════════════════════════════════════
#  CODE CLASS CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════

CLASS_CONFIGS = {
    "A": {"Ru_dir": 0.75, "Rv_dir": 0.75, "path_i_frac": 0.375},   # 384/1024, from sweep
    "B": {"Ru_dir": 0.625, "Rv_dir": 0.875, "path_i_frac": 0.5},   # 512/1024, from sweep
    "C": {"Ru_dir": 0.5,  "Rv_dir": 1.0,   "path_i_frac": 1.0},    # path_i = N (extreme)
}

BENCHMARK_CODEWORDS = 30
BLER_SKIP_THRESHOLD = 0.5


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_path_i(cfg, N):
    """Resolve path_i from config fraction: frac * N, rounded to int."""
    frac = cfg.get("path_i_frac", 1.0)
    return round(frac * N)


# ════════════════════════════════════════════════════════════════════
#  WORKER
# ════════════════════════════════════════════════════════════════════

def _sim_one_codeword(args):
    """Encode → channel → decode → count errors."""
    N, n, ku, kv, Au, Av, frozen_u, frozen_v, b, L, seed_i = args
    rng = np.random.default_rng(seed_i)
    channel = BEMAC()

    info_u = rng.integers(0, 2, size=ku).tolist()
    info_v = rng.integers(0, 2, size=kv).tolist()

    u = build_message(N, info_u, Au)
    v = build_message(N, info_v, Av)

    x = polar_encode(u.tolist())
    y = polar_encode(v.tolist())
    z = channel.sample_batch(np.array(x), np.array(y)).tolist()

    if L == 1:
        u_dec, v_dec = decode_single(N, z, b, frozen_u, frozen_v, channel, log_domain=True)
    else:
        u_dec, v_dec = decode_single_list(N, z, b, frozen_u, frozen_v, channel, log_domain=True, L=L)

    ue = sum(1 for p, bit in zip(Au, info_u) if u_dec[p - 1] != bit)
    ve = sum(1 for p, bit in zip(Av, info_v) if v_dec[p - 1] != bit)
    return (1 if (ue > 0 or ve > 0) else 0, ue, ve)


# ════════════════════════════════════════════════════════════════════
#  BATCH WORKER (vectorised)
# ════════════════════════════════════════════════════════════════════

def _sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v, b, L,
               batch_size, rng, channel):
    """Encode → channel → batch-decode → count errors.  Single process."""
    info_u = rng.integers(0, 2, size=(batch_size, ku))
    info_v = rng.integers(0, 2, size=(batch_size, kv))

    U = build_message_batch(N, info_u, Au)
    V = build_message_batch(N, info_v, Av)
    X = polar_encode_batch(U)
    Y = polar_encode_batch(V)
    Z = channel.sample_batch(X, Y)

    if L == 1:
        results = decode_batch(N, Z.tolist(), b, frozen_u, frozen_v,
                               channel, vectorized=True)
    else:
        results = decode_batch_list(N, Z.tolist(), b, frozen_u, frozen_v,
                                    channel, L=L, vectorized=True)

    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])

    block_errors = 0
    u_bit_errors = 0
    v_bit_errors = 0
    for i, (u_dec, v_dec) in enumerate(results):
        u_dec_arr = np.array(u_dec)
        v_dec_arr = np.array(v_dec)
        ue = int(np.sum(u_dec_arr[u_info_idx] != info_u[i]))
        ve = int(np.sum(v_dec_arr[v_info_idx] != info_v[i]))
        u_bit_errors += ue
        v_bit_errors += ve
        if ue > 0 or ve > 0:
            block_errors += 1

    return block_errors, u_bit_errors, v_bit_errors


# ════════════════════════════════════════════════════════════════════
#  BENCHMARK
# ════════════════════════════════════════════════════════════════════

def benchmark(n_values, n_workers, cfg, L, seed, pool=None):
    """Measure wall-clock ms/codeword using batch-vectorised decoder."""
    timings = {}
    channel = BEMAC()
    for n in n_values:
        N = 1 << n
        pi = resolve_path_i(cfg, N)
        ku = round(0.7 * cfg["Ru_dir"] * N)
        kv = round(0.7 * cfg["Rv_dir"] * N)
        ku = max(1, min(ku, N - 1))
        kv = max(1, min(kv, N - 1))
        Au, Av, frozen_u, frozen_v, _, _ = design_bemac(n, ku, kv)
        b = make_path(N, path_i=pi)

        rng = np.random.default_rng(seed + n * 100000)

        t0 = time.perf_counter()
        _sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v, b, L,
                   BENCHMARK_CODEWORDS, rng, channel)
        elapsed = time.perf_counter() - t0

        wall_per_cw = elapsed / BENCHMARK_CODEWORDS
        timings[n] = wall_per_cw
        log(f"  N={N:>5d} (n={n:>2d}): {wall_per_cw*1000:.1f} ms/codeword "
            f"({BENCHMARK_CODEWORDS} cw, batch-vectorised, {elapsed:.2f}s)")

    return timings


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Unified BE-MAC polar code simulation")
    p.add_argument("--class", "-c", dest="code_class", required=True, choices=["A", "B", "C"],
                   help="Code class")
    p.add_argument("--L", type=int, default=1, help="List size (1=SC, >1=SCL)")
    p.add_argument("--N", type=int, nargs="+", default=[1024],
                   help="Block lengths (must be powers of 2)")
    p.add_argument("--rho", type=float, nargs="+", default=None,
                   help="Rho values (default: linspace(0.5, 0.95, 8))")
    p.add_argument("--hours", type=float, default=2.0, help="Time budget in hours")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--design", choices=["analytical", "mc"], default="analytical",
                   help="Design method")
    p.add_argument("--output", "-o", type=str, default=None, help="Custom output JSON path")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CLASS_CONFIGS[args.code_class]
    rho_values = np.array(args.rho) if args.rho else np.linspace(0.5, 0.95, 8)
    L = args.L
    seed = args.seed
    design_mode = args.design
    total_budget_s = args.hours * 3600

    # Convert N values to n (log2)
    n_values = []
    for N in args.N:
        n = int(np.log2(N))
        assert 1 << n == N, f"N={N} is not a power of 2"
        n_values.append(n)

    # Output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.output:
        json_path = args.output
    else:
        design_tag = f"_{design_mode}" if design_mode == "mc" else ""
        json_path = os.path.join(results_dir,
                                 f"sim_bemac_{args.code_class}{design_tag}_L{L}.json")

    # MC design file template
    mc_design_tmpl = os.path.join(script_dir, "..", "designs",
                                  f"bemac_{args.code_class}_n{{n}}.npz")

    n_workers = min(os.cpu_count() or 1, 8)
    decoder_name = f"SC (L=1)" if L == 1 else f"SCL (L={L})"

    log(f"simulate.py — BE-MAC Code Class {args.code_class} = "
        f"({cfg['Ru_dir']}, {cfg['Rv_dir']}), {decoder_name}")
    log(f"  Block lengths: N = {args.N}")
    log(f"  Rho values: {rho_values.tolist()}")
    log(f"  Design: {design_mode}")
    log(f"  Time budget: {args.hours:.1f} hours")
    log(f"  Mode: batch-vectorised (single process)")
    log(f"  Seed: {seed}")
    log(f"  Output: {json_path}")
    log()

    pool = None  # batch mode doesn't need multiprocessing

    try:
        # ── Benchmark ──
        log("=== Benchmarking decode speed ===")
        timings = benchmark(n_values, n_workers, cfg, L, seed, pool=pool)
        log()

        # Compute codewords per point
        n_rho = len(rho_values)
        for n in n_values:
            N = 1 << n
            time_per_point = (total_budget_s / len(n_values)) / n_rho
            cw = max(200, int(time_per_point / timings[n]))
            log(f"  N={N:>5d}: ~{cw} codewords/point "
                f"(~{cw * timings[n]:.0f}s/point, ~{cw * timings[n] * n_rho / 60:.0f} min total)")
        log()

        # ── Simulation loop ──
        all_results = []
        benchmark_meta = {str(n): {"N": 1 << n, "ms_per_codeword": round(timings[n] * 1000, 2)}
                          for n in n_values}
        time_spent = 0.0

        for n_idx, n in enumerate(n_values):
            N = 1 << n
            pi = resolve_path_i(cfg, N)
            b = make_path(N, path_i=pi)

            remaining_budget = total_budget_s - time_spent
            remaining_Ns = len(n_values) - n_idx
            time_for_this_N = remaining_budget / remaining_Ns
            n_cw = max(200, int((time_for_this_N / n_rho) / timings[n]))

            log("=" * 65)
            log(f"  N = {N}  (n = {n}),  {n_cw} codewords per rate point")
            log(f"  path_i = {pi}, budget ~{time_for_this_N/60:.0f} min")
            log("=" * 65)

            skipped = False
            for rho_idx, rho in enumerate(rho_values):
                ku = round(rho * cfg["Ru_dir"] * N)
                kv = round(rho * cfg["Rv_dir"] * N)
                ku = max(1, min(ku, N - 1))
                kv = max(1, min(kv, N - 1))
                Ru = ku / N
                Rv = kv / N

                if skipped:
                    result_row = {
                        "N": N, "n": n,
                        "Ru": round(Ru, 6), "Rv": round(Rv, 6),
                        "rho": round(float(rho), 6),
                        "ku": ku, "kv": kv, "L": L,
                        "bler": None, "ber_u": None, "ber_v": None,
                        "block_errors": None, "n_codewords": 0,
                        "time_s": 0, "path_i": pi, "skipped": True,
                    }
                    all_results.append(result_row)
                    log(f"  rho={rho:.4f}  SKIPPED (previous BLER > {BLER_SKIP_THRESHOLD})")
                    continue

                # Design
                if design_mode == "mc":
                    mc_path = mc_design_tmpl.format(n=n)
                    Au, Av, frozen_u, frozen_v, _, _, _ = design_from_file(mc_path, n, ku, kv)
                else:
                    Au, Av, frozen_u, frozen_v, _, _ = design_bemac(n, ku, kv)

                base_seed = seed + n * 100000 + rho_idx * 10000
                channel = BEMAC()
                batch_size = min(500, n_cw)

                block_errors = 0
                u_bit_errors = 0
                v_bit_errors = 0
                cw_done = 0

                t0 = time.time()
                while cw_done < n_cw:
                    bs = min(batch_size, n_cw - cw_done)
                    rng = np.random.default_rng(base_seed + cw_done)
                    be, ube, vbe = _sim_batch(
                        N, ku, kv, Au, Av, frozen_u, frozen_v, b, L,
                        bs, rng, channel)
                    block_errors += be
                    u_bit_errors += ube
                    v_bit_errors += vbe
                    cw_done += bs
                elapsed = time.time() - t0
                bler = block_errors / n_cw
                ber_u = u_bit_errors / max(1, n_cw * ku)
                ber_v = v_bit_errors / max(1, n_cw * kv)

                result_row = {
                    "N": N, "n": n,
                    "Ru": round(Ru, 6), "Rv": round(Rv, 6),
                    "rho": round(float(rho), 6),
                    "ku": ku, "kv": kv, "L": L,
                    "bler": bler,
                    "ber_u": ber_u, "ber_v": ber_v,
                    "block_errors": block_errors,
                    "n_codewords": n_cw,
                    "time_s": round(elapsed, 2),
                    "path_i": pi,
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

                # Checkpoint
                path_desc = f"path_i={pi}" if pi != N else "0^N 1^N (path_i=N)"
                output = {
                    "description": f"BE-MAC Code Class {args.code_class} = "
                                   f"({cfg['Ru_dir']}, {cfg['Rv_dir']}), {decoder_name}",
                    "channel": "BE-MAC",
                    "class": args.code_class,
                    "decoder": f"{decoder_name}, efficient O(N log N)",
                    "design": design_mode,
                    "path_i_frac": cfg["path_i_frac"],
                    "seed": seed,
                    "L": L,
                    "bler_skip_threshold": BLER_SKIP_THRESHOLD,
                    "timestamp": datetime.now().isoformat(),
                    "benchmark": benchmark_meta,
                    "results": all_results,
                }
                with open(json_path, "w") as f:
                    json.dump(output, f, indent=2)

                time_spent += elapsed

            log()

    finally:
        if pool is not None:
            pool.shutdown(wait=False)

    # ── Summary ──
    total_time = time_spent

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
    log(f"Total simulation time: {time_spent:.0f}s ({time_spent/3600:.2f}h)")
    log(f"Results saved to {json_path}")
    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log()
        log("*** Interrupted by user ***")
        sys.exit(1)
