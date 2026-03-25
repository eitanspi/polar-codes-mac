"""
simulate_gmac.py — Gaussian MAC polar code simulation for all code classes.

Adapts simulate_abnmac.py for the Gaussian MAC channel.

Usage:
    cd to_git
    python scripts/simulate_gmac.py --class C --L 1 --snr-db 6 --N 8 16 32 64 128 256 512 1024
    python scripts/simulate_gmac.py --class C --L 32 --snr-db 6
    python scripts/simulate_gmac.py --class B --L 1 --snr-db 3 --design mc
    python scripts/simulate_gmac.py --class A --L 32 --snr-db 6 --hours 2
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.encoder import polar_encode, polar_encode_batch, build_message, build_message_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_single, decode_batch
from polar.decoder_scl import decode_single_list, decode_batch_list

# Import GA design from root-level design.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from design import design_gmac as design_gmac_ga


# ════════════════════════════════════════════════════════════════════
#  CODE CLASS CONFIGURATIONS
#  Same path_i fractions as BEMAC/ABNMAC
# ════════════════════════════════════════════════════════════════════

CLASS_CONFIGS = {
    "A": {"path_i_frac": 0.375},
    "B": {"path_i_frac": 0.5},
    "C": {"path_i_frac": 1.0},
}

BENCHMARK_CODEWORDS = 30
BLER_SKIP_THRESHOLD = 0.5


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_path_i(cfg, N):
    return round(cfg.get("path_i_frac", 1.0) * N)


# ════════════════════════════════════════════════════════════════════
#  BATCH WORKER (vectorised)
# ════════════════════════════════════════════════════════════════════

def _sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v, b, L,
               batch_size, rng, channel):
    info_u = rng.integers(0, 2, size=(batch_size, ku))
    info_v = rng.integers(0, 2, size=(batch_size, kv))

    U = build_message_batch(N, info_u, Au)
    V = build_message_batch(N, info_v, Av)
    X = polar_encode_batch(U)
    Y = polar_encode_batch(V)
    Z = channel.sample_batch(X, Y)

    # Gaussian MAC outputs are floats — convert to list of lists
    if Z.ndim == 2:
        Z_list = [Z[i].tolist() for i in range(Z.shape[0])]
    else:
        Z_list = Z.tolist()

    if L == 1:
        results = decode_batch(N, Z_list, b, frozen_u, frozen_v,
                               channel, vectorized=True)
    else:
        results = decode_batch_list(N, Z_list, b, frozen_u, frozen_v,
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

def benchmark(n_values, cfg, L, seed, channel, sigma2):
    timings = {}
    for n in n_values:
        N = 1 << n
        pi = resolve_path_i(cfg, N)
        # Use moderate rate for benchmarking
        ku = max(1, round(0.3 * N))
        kv = max(1, round(0.5 * N))
        Au, Av, frozen_u, frozen_v, _, _ = design_gmac_ga(n, ku, kv, sigma2)
        b = make_path(N, path_i=pi)

        rng = np.random.default_rng(seed + n * 100000)

        t0 = time.perf_counter()
        _sim_batch(N, ku, kv, Au, Av, frozen_u, frozen_v, b, L,
                   BENCHMARK_CODEWORDS, rng, channel)
        elapsed = time.perf_counter() - t0

        wall_per_cw = elapsed / BENCHMARK_CODEWORDS
        timings[n] = wall_per_cw
        log(f"  N={N:>5d} (n={n:>2d}): {wall_per_cw*1000:.1f} ms/codeword")

    return timings


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Gaussian MAC polar code simulation")
    p.add_argument("--class", "-c", dest="code_class", required=True,
                   choices=["A", "B", "C"])
    p.add_argument("--L", type=int, default=1, help="List size (1=SC, >1=SCL)")
    p.add_argument("--N", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--snr-db", type=float, required=True, help="Per-user SNR in dB")
    p.add_argument("--rho", type=float, nargs="+", default=None,
                   help="Rate scaling factors (default: linspace(0.5, 0.95, 8))")
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--design", choices=["ga", "mc"], default="ga",
                   help="Design method: ga (Gaussian Approximation) or mc")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", "-o", type=str, default=None)
    return p.parse_args()


def load_previous_results(json_path):
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


def main():
    args = parse_args()
    cfg = CLASS_CONFIGS[args.code_class]
    rho_values = np.array(args.rho) if args.rho else np.linspace(0.5, 0.95, 8)
    L = args.L
    seed = args.seed
    design_mode = args.design
    total_budget_s = args.hours * 3600

    sigma2 = 10.0 ** (-args.snr_db / 10.0)
    channel = GaussianMAC(sigma2=sigma2)
    I_ZX, I_ZY_X, I_ZXY = channel.capacity()

    # Compute max rates from capacity
    # For class C: U marginal cap = I_ZX, V conditional cap = I_ZY_X
    # For class A: swapped by symmetry
    # For class B: both get partial side info, use I_ZXY/2 as guideline
    if args.code_class == "C":
        Ru_max, Rv_max = I_ZX, I_ZY_X
    elif args.code_class == "A":
        Ru_max, Rv_max = I_ZY_X, I_ZX  # swapped
    else:
        Ru_max = I_ZXY / 2.0
        Rv_max = I_ZXY / 2.0

    n_values = []
    for N in args.N:
        n = int(np.log2(N))
        assert 1 << n == N, f"N={N} is not a power of 2"
        n_values.append(n)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    mc_design_tmpl = os.path.join(script_dir, "..", "designs",
                                  f"gmac_{args.code_class}_n{{n}}_snr{args.snr_db:.0f}dB.npz")

    if args.output:
        json_path = args.output
    else:
        design_tag = f"_{design_mode}" if design_mode == "mc" else ""
        json_path = os.path.join(results_dir,
                                 f"sim_gmac_{args.code_class}{design_tag}_L{L}"
                                 f"_snr{args.snr_db:.0f}dB.json")

    prev_results = {}
    if args.resume:
        prev_results = load_previous_results(json_path)
        if prev_results:
            log(f"  RESUME: loaded {len(prev_results)} points")

    decoder_name = f"SC (L=1)" if L == 1 else f"SCL (L={L})"

    log(f"simulate_gmac.py — Gaussian MAC Class {args.code_class}, {decoder_name}")
    log(f"  SNR = {args.snr_db} dB, sigma2 = {sigma2:.4f}")
    log(f"  Capacity: I(Z;X)={I_ZX:.4f}, I(Z;Y|X)={I_ZY_X:.4f}, "
        f"I(Z;X,Y)={I_ZXY:.4f}")
    log(f"  Max rates: Ru_max={Ru_max:.4f}, Rv_max={Rv_max:.4f}")
    log(f"  N = {args.N}")
    log(f"  Rho values: {rho_values.tolist()}")
    log(f"  Design: {design_mode}")
    log(f"  Budget: {args.hours:.1f} hours")
    log(f"  Output: {json_path}")
    log()

    # Benchmark
    log("=== Benchmarking ===")
    timings = benchmark(n_values, cfg, L, seed, channel, sigma2)
    log()

    n_rho = len(rho_values)
    for n in n_values:
        N = 1 << n
        time_per_point = (total_budget_s / len(n_values)) / n_rho
        cw = max(200, int(time_per_point / timings[n]))
        log(f"  N={N:>5d}: ~{cw} codewords/point")
    log()

    # Simulation
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
        log(f"  N = {N}  (n = {n}),  {n_cw_new} codewords/point")
        log(f"  path_i = {pi}, budget ~{time_for_this_N/60:.0f} min")
        log("=" * 65)

        skipped = False
        for rho_idx, rho in enumerate(rho_values):
            ku = round(rho * Ru_max * N)
            kv = round(rho * Rv_max * N)
            ku = max(1, min(ku, N - 1))
            kv = max(1, min(kv, N - 1))
            Ru = ku / N
            Rv = kv / N

            if skipped:
                all_results.append({
                    "N": N, "n": n, "Ru": round(Ru, 6), "Rv": round(Rv, 6),
                    "rho": round(float(rho), 6), "ku": ku, "kv": kv, "L": L,
                    "bler": None, "ber_u": None, "ber_v": None,
                    "block_errors": None, "n_codewords": 0,
                    "time_s": 0, "path_i": pi, "skipped": True,
                })
                log(f"  rho={rho:.4f}  SKIPPED")
                continue

            # Design
            if design_mode == "mc":
                mc_path = mc_design_tmpl.format(n=n)
                Au, Av, frozen_u, frozen_v, _, _, _ = design_from_file(mc_path, n, ku, kv)
            else:
                Au, Av, frozen_u, frozen_v, _, _ = design_gmac_ga(n, ku, kv, sigma2)

            key = (N, round(float(rho), 6))
            prev = prev_results.get(key, {})
            acc_block = prev.get("block_errors", 0)
            acc_ube = prev.get("u_bit_errors", 0)
            acc_vbe = prev.get("v_bit_errors", 0)
            acc_cw = prev.get("n_codewords", 0)
            acc_time = prev.get("time_s", 0)

            base_seed = seed + n * 100000 + rho_idx * 10000
            batch_size = min(500, n_cw_new)

            block_errors = 0
            u_bit_errors = 0
            v_bit_errors = 0
            cw_done = 0

            t0 = time.time()
            while cw_done < n_cw_new:
                bs = min(batch_size, n_cw_new - cw_done)
                rng = np.random.default_rng(base_seed + acc_cw + cw_done)
                be, ube, vbe = _sim_batch(
                    N, ku, kv, Au, Av, frozen_u, frozen_v, b, L,
                    bs, rng, channel)
                block_errors += be
                u_bit_errors += ube
                v_bit_errors += vbe
                cw_done += bs
            elapsed = time.time() - t0

            total_block = acc_block + block_errors
            total_ube = acc_ube + u_bit_errors
            total_vbe = acc_vbe + v_bit_errors
            total_cw = acc_cw + n_cw_new
            total_time = acc_time + elapsed

            bler = total_block / total_cw
            ber_u = total_ube / max(1, total_cw * ku)
            ber_v = total_vbe / max(1, total_cw * kv)

            result_row = {
                "N": N, "n": n, "Ru": round(Ru, 6), "Rv": round(Rv, 6),
                "rho": round(float(rho), 6), "ku": ku, "kv": kv, "L": L,
                "bler": bler, "ber_u": ber_u, "ber_v": ber_v,
                "block_errors": total_block, "u_bit_errors": total_ube,
                "v_bit_errors": total_vbe, "n_codewords": total_cw,
                "time_s": round(total_time, 2), "path_i": pi,
            }
            all_results.append(result_row)

            prev_tag = f" (+{acc_cw} prev)" if acc_cw > 0 else ""
            log(f"  rho={rho:.4f}  Ru={Ru:.4f}  Rv={Rv:.4f}  "
                f"ku={ku}  kv={kv}  "
                f"BLER={bler:.4f}  BER_u={ber_u:.4e}  BER_v={ber_v:.4e}  "
                f"({total_block}/{total_cw}{prev_tag})  {elapsed:.1f}s")

            if bler > BLER_SKIP_THRESHOLD:
                skipped = True

            # Checkpoint
            output = {
                "description": f"Gaussian MAC Class {args.code_class}, {decoder_name}",
                "channel": "Gaussian MAC",
                "snr_db": args.snr_db,
                "sigma2": sigma2,
                "class": args.code_class,
                "decoder": f"{decoder_name}, efficient O(N log N)",
                "design": design_mode,
                "path_i_frac": cfg["path_i_frac"],
                "seed": seed, "L": L,
                "capacity": {"I_ZX": round(I_ZX, 4), "I_ZY_X": round(I_ZY_X, 4),
                             "I_ZXY": round(I_ZXY, 4)},
                "Ru_max": round(Ru_max, 4), "Rv_max": round(Rv_max, 4),
                "timestamp": datetime.now().isoformat(),
                "benchmark": benchmark_meta,
                "results": all_results,
            }
            with open(json_path, "w") as f:
                json.dump(output, f, indent=2)

            time_spent += elapsed

        log()

    # Summary
    log("=" * 115)
    log("  SUMMARY")
    log("=" * 115)
    log(f"{'N':>6} {'rho':>6} {'Ru':>7} {'Rv':>7} {'ku':>5} {'kv':>5} "
        f"{'BLER':>10} {'BER_u':>10} {'BER_v':>10} "
        f"{'blk_err':>7} {'n_cw':>8} {'time':>7}")
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
    log(f"Total: {time_spent:.0f}s ({time_spent/3600:.2f}h)")
    log(f"Results saved to {json_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log()
        log("*** Interrupted ***")
        sys.exit(1)
