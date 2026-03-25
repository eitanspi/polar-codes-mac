#!/usr/bin/env python
"""
campaign_10h.py — 10-hour campaign focused on reliable N=1024 results.

1. Heavy MC designs for N=512,1024 (more trials = fewer pe=0 ties)
2. BLER sims with 5000 codewords at N=512,1024 for reliable stats
3. Uses GA tiebreaker for Class C only, pure MC for A/B

Runs one job at a time, ~1 core for sims, 4 cores for MC design.
"""

import os, sys, time, json, numpy as np, subprocess
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from polar.channels import GaussianMAC
from polar.encoder import polar_encode, build_message
from polar.design import make_path
from polar.design_mc import load_design, _select_info_frozen
from polar.decoder import decode_single
from polar.decoder_scl import decode_single_list
from design import ga_gmac

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), "..", "designs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "campaign_10h.json")
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")

os.makedirs(DESIGNS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_results():
    if os.path.exists(RESULTS_FILE):
        return json.load(open(RESULTS_FILE))
    return []

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def has_result(results, channel, cls, rho, N, L):
    for r in results:
        if (r["channel"] == channel and r["class"] == cls
                and abs(r["rho"] - rho) < 0.01 and r["N"] == N and r["L"] == L):
            return True
    return False


def get_design(cls, n, ku, kv, sigma2, snr_db):
    """Get design: GA tiebreaker for C, pure MC for A/B."""
    N = 1 << n
    design_file = os.path.join(DESIGNS_DIR, f"gmac_{cls}_n{n}_snr{snr_db:.0f}dB.npz")
    su, sv, pe_u_mc, pe_v_mc, _ = load_design(design_file)

    if cls == 'C':
        # Hybrid: GA tiebreaker for pe=0 channels
        pe_u_ga, pe_v_ga = ga_gmac(n, sigma2)
        pe_u_h = pe_u_mc.copy()
        pe_v_h = pe_v_mc.copy()
        for pe_mc, pe_ga, pe_h in [(pe_u_mc, pe_u_ga, pe_u_h),
                                    (pe_v_mc, pe_v_ga, pe_v_h)]:
            zero = pe_mc == 0
            if np.any(zero) and np.any(pe_mc > 0):
                mn = pe_mc[pe_mc > 0].min()
                ga_z = pe_ga[zero]
                if ga_z.max() > 0:
                    pe_h[zero] = ga_z / ga_z.max() * mn * 0.99
        su = np.argsort(pe_u_h)
        sv = np.argsort(pe_v_h)
    # else: su/sv already have polar tiebreaker from load_design

    Au, fu = _select_info_frozen(N, su, ku)
    Av, fv = _select_info_frozen(N, sv, kv)
    return Au, Av, fu, fv


def run_sim(channel, cls, rho, N, L, Au, Av, fu, fv, ncw, seed=42):
    pfrac = {'C': 1.0, 'A': 0.375, 'B': 0.5}
    b = make_path(N, path_i=round(pfrac[cls] * N))
    rng = np.random.default_rng(seed)
    errors = 0
    for _ in range(ncw):
        ku = len(Au); kv = len(Av)
        iu = rng.integers(0, 2, ku).tolist()
        iv = rng.integers(0, 2, kv).tolist()
        u = build_message(N, iu, Au)
        v = build_message(N, iv, Av)
        x = polar_encode(u.tolist())
        y = polar_encode(v.tolist())
        z = channel.sample_batch(np.array(x), np.array(y)).tolist()
        if L == 1:
            ud, vd = decode_single(N, z, b, fu, fv, channel, log_domain=True)
        else:
            ud, vd = decode_single_list(N, z, b, fu, fv, channel,
                                        log_domain=True, L=L)
        if not (all(ud[p - 1] == bit for p, bit in zip(Au, iu)) and
                all(vd[p - 1] == bit for p, bit in zip(Av, iv))):
            errors += 1
    return errors / ncw


def run_mc_design(cls, N, snr_db, hours):
    """Run MC design via subprocess."""
    log(f"  MC design: {cls} N={N} SNR={snr_db}dB ({hours}h)...")
    cmd = (f"python scripts/run_design_gmac.py --class {cls} --N {N} "
           f"--snr-db {snr_db} --hours {hours} --force")
    subprocess.run(cmd, shell=True, cwd=PROJECT_DIR,
                   capture_output=True, timeout=int(hours * 3600 + 300))
    # Check result
    n = N.bit_length() - 1
    f = os.path.join(DESIGNS_DIR, f"gmac_{cls}_n{n}_snr{snr_db:.0f}dB.npz")
    if os.path.exists(f):
        d = np.load(f)
        nt = int(d['n_trials']) if 'n_trials' in d else '?'
        log(f"  Done: {nt} trials")
    else:
        log(f"  FAILED")


def main():
    t_start = time.time()
    results = load_results()
    log(f"Campaign 10h starting. {len(results)} existing results.")

    CLASS_PFRAC = {'C': 1.0, 'A': 0.375, 'B': 0.5}
    RHOS = [0.3, 0.5, 0.7]

    # ═══ Phase 1: Heavy MC designs for N=512,1024 ═══
    log("\n" + "=" * 60)
    log("  PHASE 1: Heavy MC designs (N=512, 1024)")
    log("=" * 60)

    for snr_db in [6, 3]:
        for cls in ['C', 'A', 'B']:
            for N in [512, 1024]:
                # 1 hour per design = ~36K-70K trials
                run_mc_design(cls, N, snr_db, hours=1.0)

        elapsed_h = (time.time() - t_start) / 3600
        log(f"  Elapsed: {elapsed_h:.1f}h")

    # ═══ Phase 2: BLER simulations with 5000 codewords ═══
    log("\n" + "=" * 60)
    log("  PHASE 2: BLER simulations (5000 cw at N≤512, 3000 at N=1024)")
    log("=" * 60)

    for snr_db in [6, 3]:
        sigma2 = 10.0 ** (-snr_db / 10.0)
        ch = GaussianMAC(sigma2=sigma2)
        I_ZX, I_ZY_X, I_ZXY = ch.capacity()
        Ru_max = {'C': I_ZX, 'A': I_ZY_X, 'B': I_ZXY / 2}
        Rv_max = {'C': I_ZY_X, 'A': I_ZX, 'B': I_ZXY / 2}
        channel_name = f"gmac_snr{snr_db}dB"

        log(f"\n  --- SNR={snr_db}dB ---")

        for cls in ['C', 'A', 'B']:
            for rho in RHOS:
                for N in [8, 16, 32, 64, 128, 256, 512, 1024]:
                    for L in [1, 32]:
                        if has_result(results, channel_name, cls, rho, N, L):
                            continue

                        n = N.bit_length() - 1
                        ku = max(1, min(round(rho * Ru_max[cls] * N), N - 1))
                        kv = max(1, min(round(rho * Rv_max[cls] * N), N - 1))

                        try:
                            Au, Av, fu, fv = get_design(cls, n, ku, kv,
                                                        sigma2, snr_db)
                        except Exception as e:
                            log(f"  {cls} {rho} N={N} L={L}: design error {e}")
                            continue

                        # More codewords for larger N
                        if L == 1:
                            ncw = 5000 if N <= 512 else 3000
                        else:
                            ncw = 2000 if N <= 128 else (1000 if N <= 512 else 500)

                        seed = 42 + n * 1000 + int(rho * 100) + L * 10000 + snr_db * 100000

                        t0 = time.time()
                        bler = run_sim(ch, cls, rho, N, L, Au, Av, fu, fv,
                                       ncw, seed)
                        elapsed = time.time() - t0

                        row = {
                            "channel": channel_name, "class": cls, "rho": rho,
                            "N": N, "L": L, "ku": ku, "kv": kv,
                            "Ru": round(ku / N, 4), "Rv": round(kv / N, 4),
                            "sum_rate": round((ku + kv) / N, 4),
                            "bler": bler, "n_codewords": ncw,
                            "snr_db": snr_db, "time_s": round(elapsed, 1),
                            "design": "hybrid_GA" if cls == 'C' else "pure_MC",
                        }
                        results.append(row)
                        save_results(results)

                        log(f"  {cls} rho={rho} N={N:5d} L={L:2d} snr={snr_db}  "
                            f"ku={ku:4d} kv={kv:4d}  BLER={bler:.4f}  "
                            f"({ncw}cw, {elapsed:.0f}s)")

                        # Time check
                        elapsed_h = (time.time() - t_start) / 3600
                        if elapsed_h > 9.5:
                            log("  Time limit approaching!")
                            save_results(results)
                            return

    total_h = (time.time() - t_start) / 3600
    log(f"\n{'=' * 60}")
    log(f"  CAMPAIGN COMPLETE: {total_h:.1f}h, {len(results)} results")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
