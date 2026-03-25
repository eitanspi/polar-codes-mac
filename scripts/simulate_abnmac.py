"""
simulate_abnmac.py — Unified ABN-MAC polar code simulation for all code classes.

Reproduces the ABN-MAC experiments from Önay ISIT 2013 (Fig. 3 & 4).

ABN-MAC channel: Z = (X⊕Ex, Y⊕Ey),  (Ex,Ey) ~ p_noise
  Default noise: [[0.1286, 0.0175], [0.0175, 0.8364]]
  Capacity region (uniform inputs):
    I(Z;X)   ≈ 0.400,  I(Z;Y|X) ≈ 0.800,  I(Z;X,Y) ≈ 1.200
    Dominant face from (0.4, 0.8) to (0.8, 0.4)

Usage:
    cd to_git
    python scripts/simulate_abnmac.py --class C --L 1
    python scripts/simulate_abnmac.py --class C --L 32
    python scripts/simulate_abnmac.py --class B --L 1 --N 1024
    python scripts/simulate_abnmac.py --class A --L 32 --hours 2

    # Resume mode: accumulate more codewords on top of existing results
    python scripts/simulate_abnmac.py --class C --L 1 --hours 1 --resume --seed 100
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

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.encoder import polar_encode, polar_encode_batch, build_message, build_message_batch
from polar.channels import ABNMAC
from polar.design import design_abnmac, make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_single, decode_batch
from polar.decoder_scl import decode_single_list, decode_batch_list

# ════════════════════════════════════════════════════════════════════
#  CODE CLASS CONFIGURATIONS  (ABN-MAC dominant face: (0.4,0.8)↔(0.8,0.4))
#
#  Analogous to BEMAC classes from Önay 2013:
#    A → midpoint of dominant face
#    B → intermediate point
#    C → corner (decode all U first, then V)
# ════════════════════════════════════════════════════════════════════

CLASS_CONFIGS = {
    "A": {"Ru_dir": 0.6,  "Rv_dir": 0.6,  "path_i_frac": 0.375},
    "B": {"Ru_dir": 0.5,  "Rv_dir": 0.7,  "path_i_frac": 0.5},
    "C": {"Ru_dir": 0.4,  "Rv_dir": 0.8,  "path_i_frac": 1.0},
}

BENCHMARK_CODEWORDS = 30
BLER_SKIP_THRESHOLD = 0.1


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_path_i(cfg, N):
    """Resolve path_i from config fraction: frac * N, rounded to int."""
    frac = cfg.get("path_i_frac", 1.0)
    return round(frac * N)


# ════════════════════════════════════════════════════════════════════
#  RESUME: load previous accumulated results
# ════════════════════════════════════════════════════════════════════

def load_previous_results(json_path):
    """
    Load previous results from JSON, return dict keyed by (N, rho) ->
    {block_errors, u_bit_errors, v_bit_errors, n_codewords, time_s}.
    """
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path) as f:
            data = json.load(f)
        prev = {}
        for r in data.get("results", []):
            if r.get("skipped"):
                continue
            key = (r["N"], round(r["rho"], 6))
            prev[key] = {
                "block_errors": r.get("block_errors", 0) or 0,
                "u_bit_errors": r.get("u_bit_errors", 0) or 0,
                "v_bit_errors": r.get("v_bit_errors", 0) or 0,
                "n_codewords": r.get("n_codewords", 0) or 0,
                "time_s": r.get("time_s", 0) or 0,
            }
        return prev
    except (json.JSONDecodeError, KeyError):
        return {}


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

def benchmark(n_values, cfg, L, seed, channel):
    """Measure wall-clock ms/codeword using batch-vectorised decoder."""
    timings = {}
    for n in n_values:
        N = 1 << n
        pi = resolve_path_i(cfg, N)
        ku = round(0.7 * cfg["Ru_dir"] * N)
        kv = round(0.7 * cfg["Rv_dir"] * N)
        ku = max(1, min(ku, N - 1))
        kv = max(1, min(kv, N - 1))
        Au, Av, frozen_u, frozen_v, _, _ = design_abnmac(n, ku, kv)
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
    p = argparse.ArgumentParser(description="Unified ABN-MAC polar code simulation")
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
                   help="Design method (mc requires pre-computed .npz files)")
    p.add_argument("--resume", action="store_true",
                   help="Resume: load existing results and accumulate more codewords")
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

    channel = ABNMAC()
    I_ZX, I_ZY_X, I_ZXY = channel.capacity()

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

    # MC design file template
    mc_design_tmpl = os.path.join(script_dir, "..", "designs",
                                  f"abnmac_{args.code_class}_n{{n}}.npz")

    if args.output:
        json_path = args.output
    else:
        design_tag = f"_{design_mode}" if design_mode == "mc" else ""
        json_path = os.path.join(results_dir,
                                 f"sim_abnmac_{args.code_class}{design_tag}_L{L}.json")

    # Load previous results if resuming
    prev_results = {}
    if args.resume:
        prev_results = load_previous_results(json_path)
        if prev_results:
            total_prev_cw = sum(v["n_codewords"] for v in prev_results.values())
            log(f"  RESUME: loaded {len(prev_results)} points, "
                f"{total_prev_cw} total previous codewords from {json_path}")
        else:
            log(f"  RESUME: no previous results found, starting fresh")

    decoder_name = f"SC (L=1)" if L == 1 else f"SCL (L={L})"

    log(f"simulate_abnmac.py — ABN-MAC Code Class {args.code_class} = "
        f"({cfg['Ru_dir']}, {cfg['Rv_dir']}), {decoder_name}")
    log(f"  Channel: ABN-MAC, I(Z;X)={I_ZX:.4f}, I(Z;Y|X)={I_ZY_X:.4f}, "
        f"I(Z;X,Y)={I_ZXY:.4f}")
    log(f"  Block lengths: N = {args.N}")
    log(f"  Rho values: {rho_values.tolist()}")
    log(f"  Design: {design_mode}")
    log(f"  Time budget: {args.hours:.1f} hours")
    log(f"  Mode: batch-vectorised (single process)")
    log(f"  Seed: {seed}")
    log(f"  Output: {json_path}")
    log()

    # ── Benchmark ──
    log("=== Benchmarking decode speed ===")
    timings = benchmark(n_values, cfg, L, seed, channel)
    log()

    # Compute codewords per point
    n_rho = len(rho_values)
    for n in n_values:
        N = 1 << n
        time_per_point = (total_budget_s / len(n_values)) / n_rho
        cw = max(200, int(time_per_point / timings[n]))
        log(f"  N={N:>5d}: ~{cw} NEW codewords/point this round "
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
        n_cw_new = max(200, int((time_for_this_N / n_rho) / timings[n]))

        log("=" * 65)
        log(f"  N = {N}  (n = {n}),  {n_cw_new} NEW codewords per rate point")
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
                    "block_errors": None, "u_bit_errors": None, "v_bit_errors": None,
                    "n_codewords": 0,
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
                Au, Av, frozen_u, frozen_v, _, _ = design_abnmac(n, ku, kv)

            # Load accumulated counts from previous rounds
            key = (N, round(float(rho), 6))
            prev = prev_results.get(key, {})
            acc_block_errors = prev.get("block_errors", 0)
            acc_u_bit_errors = prev.get("u_bit_errors", 0)
            acc_v_bit_errors = prev.get("v_bit_errors", 0)
            acc_n_cw = prev.get("n_codewords", 0)
            acc_time = prev.get("time_s", 0)

            # Use seed offset based on how many codewords already done
            base_seed = seed + n * 100000 + rho_idx * 10000
            batch_size = min(500, n_cw_new)

            block_errors = 0
            u_bit_errors = 0
            v_bit_errors = 0
            cw_done = 0

            t0 = time.time()
            while cw_done < n_cw_new:
                bs = min(batch_size, n_cw_new - cw_done)
                # Offset RNG by accumulated codewords so we never repeat
                rng = np.random.default_rng(base_seed + acc_n_cw + cw_done)
                be, ube, vbe = _sim_batch(
                    N, ku, kv, Au, Av, frozen_u, frozen_v, b, L,
                    bs, rng, channel)
                block_errors += be
                u_bit_errors += ube
                v_bit_errors += vbe
                cw_done += bs
            elapsed = time.time() - t0

            # Accumulate with previous
            total_block_errors = acc_block_errors + block_errors
            total_u_bit_errors = acc_u_bit_errors + u_bit_errors
            total_v_bit_errors = acc_v_bit_errors + v_bit_errors
            total_n_cw = acc_n_cw + n_cw_new
            total_time = acc_time + elapsed

            bler = total_block_errors / total_n_cw
            ber_u = total_u_bit_errors / max(1, total_n_cw * ku)
            ber_v = total_v_bit_errors / max(1, total_n_cw * kv)

            result_row = {
                "N": N, "n": n,
                "Ru": round(Ru, 6), "Rv": round(Rv, 6),
                "rho": round(float(rho), 6),
                "ku": ku, "kv": kv, "L": L,
                "bler": bler,
                "ber_u": ber_u, "ber_v": ber_v,
                "block_errors": total_block_errors,
                "u_bit_errors": total_u_bit_errors,
                "v_bit_errors": total_v_bit_errors,
                "n_codewords": total_n_cw,
                "time_s": round(total_time, 2),
                "path_i": pi,
            }
            all_results.append(result_row)

            prev_tag = f" (+{acc_n_cw} prev)" if acc_n_cw > 0 else ""
            log(f"  rho={rho:.4f}  Ru={Ru:.4f}  Rv={Rv:.4f}  "
                f"ku={ku}  kv={kv}  "
                f"BLER={bler:.4f}  BER_u={ber_u:.4e}  BER_v={ber_v:.4e}  "
                f"({total_block_errors}/{total_n_cw}{prev_tag})  {elapsed:.1f}s")

            if bler > BLER_SKIP_THRESHOLD:
                skipped = True
                log(f"  ** BLER {bler:.4f} > {BLER_SKIP_THRESHOLD} — "
                    f"skipping remaining rho for N={N} **")

            # Checkpoint
            output = {
                "description": f"ABN-MAC Code Class {args.code_class} = "
                               f"({cfg['Ru_dir']}, {cfg['Rv_dir']}), {decoder_name}",
                "channel": "ABN-MAC",
                "class": args.code_class,
                "decoder": f"{decoder_name}, efficient O(N log N)",
                "design": design_mode,
                "path_i_frac": cfg["path_i_frac"],
                "seed": seed,
                "L": L,
                "capacity": {"I_ZX": round(I_ZX, 4),
                             "I_ZY_X": round(I_ZY_X, 4),
                             "I_ZXY": round(I_ZXY, 4)},
                "bler_skip_threshold": BLER_SKIP_THRESHOLD,
                "timestamp": datetime.now().isoformat(),
                "benchmark": benchmark_meta,
                "results": all_results,
            }
            with open(json_path, "w") as f:
                json.dump(output, f, indent=2)

            time_spent += elapsed

        log()

    # ── Summary ──
    log("=" * 115)
    log("  SUMMARY")
    log("=" * 115)
    hdr = (f"{'N':>6} {'rho':>6} {'Ru':>7} {'Rv':>7} {'ku':>5} {'kv':>5} "
           f"{'BLER':>10} {'BER_u':>10} {'BER_v':>10} "
           f"{'blk_err':>7} {'n_cw':>8} {'time':>7}")
    log(hdr)
    log("-" * 115)
    for r in all_results:
        if r.get("skipped"):
            log(f"{r['N']:6d} {r['rho']:6.3f} {r['Ru']:7.4f} {r['Rv']:7.4f} "
                f"{r['ku']:5d} {r['kv']:5d}    SKIPPED")
        else:
            log(f"{r['N']:6d} {r['rho']:6.3f} {r['Ru']:7.4f} {r['Rv']:7.4f} "
                f"{r['ku']:5d} {r['kv']:5d} "
                f"{r['bler']:10.4f} {r['ber_u']:10.4e} {r['ber_v']:10.4e} "
                f"{r['block_errors']:7d} {r['n_codewords']:8d} {r['time_s']:6.1f}s")
    log("-" * 115)
    log(f"Total simulation time this round: {time_spent:.0f}s ({time_spent/3600:.2f}h)")
    log(f"Results saved to {json_path}")
    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log()
        log("*** Interrupted by user ***")
        sys.exit(1)
