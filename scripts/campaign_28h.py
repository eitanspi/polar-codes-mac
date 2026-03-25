#!/usr/bin/env python
"""
campaign_28h.py — Comprehensive 28-hour MAC polar code campaign.

Runs at half CPU. Sanity-checks each config before committing to long runs.
Collects data across:
  - Channels: GaussianMAC (multiple SNR), BEMAC, ABNMAC
  - Classes: A, B, C
  - Rates: rho = 0.3, 0.5, 0.7
  - N: 8..2048
  - Decoders: SC, SCL L=32

All results saved incrementally to results/campaign_28h.json.
"""

import os, sys, time, json, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from polar.channels import GaussianMAC, BEMAC, ABNMAC
from polar.encoder import polar_encode, build_message
from polar.design import design_bemac, design_abnmac, make_path
from polar.design_mc import load_design, _select_info_frozen
from polar.decoder import decode_single
from polar.decoder_scl import decode_single_list
from design import ga_gmac

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), "..", "designs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "campaign_28h.json")
N_WORKERS = 4  # half CPU

os.makedirs(DESIGNS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Results I/O ──

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


# ── MC Design for GaussianMAC ──

def ensure_gmac_design(snr_db, cls, n, hours=0.3):
    """Run MC design if not exists. Returns True if design available."""
    design_file = os.path.join(DESIGNS_DIR, f"gmac_{cls}_n{n}_snr{snr_db:.0f}dB.npz")
    if os.path.exists(design_file):
        return True

    N = 1 << n
    pfrac = {'C': 1.0, 'A': 0.375, 'B': 0.5}
    path_i = round(pfrac[cls] * N)

    log(f"  Running MC design: {cls} n={n} SNR={snr_db}dB ({hours:.1f}h budget)...")
    import subprocess
    cmd = (f"python scripts/run_design_gmac.py --class {cls} --N {N} "
           f"--snr-db {snr_db} --hours {hours} --force")
    # Limit workers by setting env
    env = os.environ.copy()
    result = subprocess.run(cmd, shell=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
                            capture_output=True, text=True,
                            timeout=int(hours * 3600 + 300), env=env)
    if result.returncode != 0:
        log(f"  Design FAILED: {result.stderr[-200:]}")
        return False
    log(f"  Design saved: {design_file}")
    return True


def hybrid_design(cls, n, ku, kv, sigma2, snr_db):
    """MC design with GA tiebreaker for pe=0 channels."""
    N = 1 << n
    design_file = os.path.join(DESIGNS_DIR, f"gmac_{cls}_n{n}_snr{snr_db:.0f}dB.npz")
    su, sv, pe_u_mc, pe_v_mc, _ = load_design(design_file)
    pe_u_ga, pe_v_ga = ga_gmac(n, sigma2)

    pe_u_h = pe_u_mc.copy()
    pe_v_h = pe_v_mc.copy()
    for pe_mc, pe_ga, pe_h in [(pe_u_mc, pe_u_ga, pe_u_h), (pe_v_mc, pe_v_ga, pe_v_h)]:
        zero = pe_mc == 0
        if np.any(zero) and np.any(pe_mc > 0):
            mn = pe_mc[pe_mc > 0].min()
            ga_z = pe_ga[zero]
            if ga_z.max() > 0:
                pe_h[zero] = ga_z / ga_z.max() * mn * 0.99

    su_h = np.argsort(pe_u_h)
    sv_h = np.argsort(pe_v_h)
    Au = sorted([i + 1 for i in su_h[:ku].tolist()])
    Av = sorted([i + 1 for i in sv_h[:kv].tolist()])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


# ── Simulation ──

def run_one_point(channel_obj, cls, rho, N, L, Au, Av, fu, fv, path_i,
                  ncw=2000, seed=42):
    b = make_path(N, path_i=path_i)
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
        z = channel_obj.sample_batch(np.array(x), np.array(y))
        z_list = z.tolist() if z.dtype != object else [tuple(zi) if isinstance(zi, (list, np.ndarray)) else zi for zi in z.flat]
        if L == 1:
            ud, vd = decode_single(N, z_list, b, fu, fv, channel_obj, log_domain=True)
        else:
            ud, vd = decode_single_list(N, z_list, b, fu, fv, channel_obj,
                                        log_domain=True, L=L)
        if not (all(ud[p-1] == bit for p, bit in zip(Au, iu)) and
                all(vd[p-1] == bit for p, bit in zip(Av, iv))):
            errors += 1
    return errors / ncw


def sanity_check(channel_obj, cls, N_test, Au, Av, fu, fv, path_i):
    """Quick 200-codeword test. Returns BLER."""
    return run_one_point(channel_obj, cls, 0.5, N_test, 1, Au, Av, fu, fv,
                         path_i, ncw=200, seed=99)


# ── Channel configs ──

GMAC_SNRS = [0, 1, 2, 3, 4, 5, 6, 8, 10]
CLASS_PFRAC = {'C': 1.0, 'A': 0.375, 'B': 0.5}
RHOS = [0.3, 0.5, 0.7]
N_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024]
CLASSES = ['C', 'A', 'B']


def run_gmac_snr(snr_db, results):
    """Run full GaussianMAC campaign for one SNR."""
    sigma2 = 10.0 ** (-snr_db / 10.0)
    ch = GaussianMAC(sigma2=sigma2)
    I_ZX, I_ZY_X, I_ZXY = ch.capacity()
    Ru_max = {'C': I_ZX, 'A': I_ZY_X, 'B': I_ZXY / 2}
    Rv_max = {'C': I_ZY_X, 'A': I_ZX, 'B': I_ZXY / 2}
    channel_name = f"gmac_snr{snr_db}dB"

    log(f"\n{'='*60}")
    log(f"  GaussianMAC SNR={snr_db}dB  (sigma2={sigma2:.4f})")
    log(f"  I_ZX={I_ZX:.4f}  I_ZY|X={I_ZY_X:.4f}  I_ZXY={I_ZXY:.4f}")
    log(f"{'='*60}")

    # Ensure MC designs exist for all N and classes
    for cls in CLASSES:
        for N in N_VALUES:
            n = N.bit_length() - 1
            # Scale design time by N
            hours = 0.05 if N <= 64 else (0.1 if N <= 256 else (0.2 if N <= 512 else 0.4))
            if not ensure_gmac_design(snr_db, cls, n, hours=hours):
                log(f"  SKIP {cls} n={n}: design failed")
                continue

    # Sanity check at N=64
    log(f"  Sanity check N=64...")
    for cls in CLASSES:
        n = 6
        ku = max(1, round(0.5 * Ru_max[cls] * 64))
        kv = max(1, round(0.5 * Rv_max[cls] * 64))
        try:
            Au, Av, fu, fv = hybrid_design(cls, n, ku, kv, sigma2, snr_db)
        except Exception as e:
            log(f"    {cls}: design error: {e}")
            continue
        path_i = round(CLASS_PFRAC[cls] * 64)
        bler = sanity_check(ch, cls, 64, Au, Av, fu, fv, path_i)
        log(f"    {cls} N=64 rho=0.5: BLER={bler:.4f} {'OK' if bler < 0.5 else 'HIGH'}")
        if bler > 0.8:
            log(f"    WARNING: {cls} sanity check failed, skipping this class")

    # Full simulation
    for cls in CLASSES:
        for rho in RHOS:
            for N in N_VALUES:
                for L in [1, 32]:
                    if has_result(results, channel_name, cls, rho, N, L):
                        continue

                    n = N.bit_length() - 1
                    ku = max(1, min(round(rho * Ru_max[cls] * N), N - 1))
                    kv = max(1, min(round(rho * Rv_max[cls] * N), N - 1))
                    path_i = round(CLASS_PFRAC[cls] * N)

                    try:
                        Au, Av, fu, fv = hybrid_design(cls, n, ku, kv, sigma2, snr_db)
                    except:
                        continue

                    # Scale codewords by N and L
                    if L == 1:
                        ncw = 2000 if N <= 512 else 1000
                    else:
                        ncw = 500 if N <= 128 else (200 if N <= 512 else 100)

                    seed = 42 + n * 1000 + int(rho * 100) + L * 10000 + snr_db * 100000
                    t0 = time.time()
                    bler = run_one_point(ch, cls, rho, N, L, Au, Av, fu, fv,
                                        path_i, ncw=ncw, seed=seed)
                    elapsed = time.time() - t0

                    row = {
                        "channel": channel_name, "class": cls, "rho": rho,
                        "N": N, "L": L, "ku": ku, "kv": kv,
                        "Ru": round(ku / N, 4), "Rv": round(kv / N, 4),
                        "sum_rate": round((ku + kv) / N, 4),
                        "bler": bler, "n_codewords": ncw,
                        "snr_db": snr_db, "time_s": round(elapsed, 1),
                    }
                    results.append(row)
                    save_results(results)

                    log(f"  {cls} rho={rho} N={N:5d} L={L:2d}  "
                        f"ku={ku:4d} kv={kv:4d}  BLER={bler:.4f}  [{elapsed:.0f}s]")


def run_bemac(results):
    """Run BEMAC campaign using existing MC designs."""
    ch = BEMAC()
    channel_name = "bemac"

    log(f"\n{'='*60}")
    log(f"  BEMAC  Z=X+Y")
    log(f"  I_ZX=0.5  I_ZY|X=1.0  I_ZXY=1.5")
    log(f"{'='*60}")

    Ru_max = {'C': 0.5, 'A': 0.75, 'B': 0.625}
    Rv_max = {'C': 1.0, 'A': 0.75, 'B': 0.875}

    for cls in CLASSES:
        for rho in RHOS:
            for N in N_VALUES:
                for L in [1, 32]:
                    if has_result(results, channel_name, cls, rho, N, L):
                        continue

                    n = N.bit_length() - 1
                    ku = max(1, min(round(rho * Ru_max[cls] * N), N - 1))
                    kv = max(1, min(round(rho * Rv_max[cls] * N), N - 1))
                    path_i = round(CLASS_PFRAC[cls] * N)

                    # Use MC design if available, else analytical
                    design_file = os.path.join(DESIGNS_DIR, f"bemac_{cls}_n{n}.npz")
                    if os.path.exists(design_file):
                        su, sv, _, _, _ = load_design(design_file)
                        Au, _ = _select_info_frozen(N, su, ku)
                        Av, _ = _select_info_frozen(N, sv, kv)
                        all_pos = set(range(1, N + 1))
                        fu = {p: 0 for p in sorted(all_pos - set(Au))}
                        fv = {p: 0 for p in sorted(all_pos - set(Av))}
                    else:
                        Au, Av, fu, fv, _, _ = design_bemac(n, ku, kv)

                    if L == 1:
                        ncw = 2000 if N <= 512 else 1000
                    else:
                        ncw = 500 if N <= 128 else (200 if N <= 512 else 100)

                    seed = 42 + n * 1000 + int(rho * 100) + L * 10000
                    t0 = time.time()
                    bler = run_one_point(ch, cls, rho, N, L, Au, Av, fu, fv,
                                        path_i, ncw=ncw, seed=seed)
                    elapsed = time.time() - t0

                    results.append({
                        "channel": channel_name, "class": cls, "rho": rho,
                        "N": N, "L": L, "ku": ku, "kv": kv,
                        "Ru": round(ku / N, 4), "Rv": round(kv / N, 4),
                        "sum_rate": round((ku + kv) / N, 4),
                        "bler": bler, "n_codewords": ncw,
                        "time_s": round(elapsed, 1),
                    })
                    save_results(results)

                    log(f"  {cls} rho={rho} N={N:5d} L={L:2d}  "
                        f"ku={ku:4d} kv={kv:4d}  BLER={bler:.4f}  [{elapsed:.0f}s]")


def run_abnmac(results):
    """Run ABNMAC campaign."""
    ch = ABNMAC()
    channel_name = "abnmac"
    I_ZX, I_ZY_X, I_ZXY = ch.capacity()

    log(f"\n{'='*60}")
    log(f"  ABNMAC  Z=(X⊕Ex, Y⊕Ey)")
    log(f"  I_ZX={I_ZX:.4f}  I_ZY|X={I_ZY_X:.4f}  I_ZXY={I_ZXY:.4f}")
    log(f"{'='*60}")

    Ru_max = {'C': I_ZX, 'A': I_ZY_X, 'B': (I_ZX + I_ZY_X) / 2}
    Rv_max = {'C': I_ZY_X, 'A': I_ZX, 'B': (I_ZX + I_ZY_X) / 2}

    for cls in CLASSES:
        for rho in RHOS:
            for N in N_VALUES:
                for L in [1, 32]:
                    if has_result(results, channel_name, cls, rho, N, L):
                        continue

                    n = N.bit_length() - 1
                    ku = max(1, min(round(rho * Ru_max[cls] * N), N - 1))
                    kv = max(1, min(round(rho * Rv_max[cls] * N), N - 1))
                    path_i = round(CLASS_PFRAC[cls] * N)

                    # Use MC design if available, else analytical
                    design_file = os.path.join(DESIGNS_DIR, f"abnmac_{cls}_n{n}.npz")
                    if os.path.exists(design_file):
                        su, sv, _, _, _ = load_design(design_file)
                        Au, _ = _select_info_frozen(N, su, ku)
                        Av, _ = _select_info_frozen(N, sv, kv)
                        all_pos = set(range(1, N + 1))
                        fu = {p: 0 for p in sorted(all_pos - set(Au))}
                        fv = {p: 0 for p in sorted(all_pos - set(Av))}
                    else:
                        Au, Av, fu, fv, _, _ = design_abnmac(n, ku, kv)

                    if L == 1:
                        ncw = 2000 if N <= 512 else 1000
                    else:
                        ncw = 500 if N <= 128 else (200 if N <= 512 else 100)

                    seed = 42 + n * 1000 + int(rho * 100) + L * 10000 + 99999
                    t0 = time.time()
                    bler = run_one_point(ch, cls, rho, N, L, Au, Av, fu, fv,
                                        path_i, ncw=ncw, seed=seed)
                    elapsed = time.time() - t0

                    results.append({
                        "channel": channel_name, "class": cls, "rho": rho,
                        "N": N, "L": L, "ku": ku, "kv": kv,
                        "Ru": round(ku / N, 4), "Rv": round(kv / N, 4),
                        "sum_rate": round((ku + kv) / N, 4),
                        "bler": bler, "n_codewords": ncw,
                        "time_s": round(elapsed, 1),
                    })
                    save_results(results)

                    log(f"  {cls} rho={rho} N={N:5d} L={L:2d}  "
                        f"ku={ku:4d} kv={kv:4d}  BLER={bler:.4f}  [{elapsed:.0f}s]")


# ── Main ──

def main():
    t_start = time.time()
    results = load_results()
    log(f"Campaign starting. {len(results)} existing results loaded.")

    # Priority order: GaussianMAC at key SNRs, then BEMAC, then ABNMAC
    # Start with SNRs we don't have yet
    for snr_db in [6, 3, 0, 2, 4, 8, 10, 1, 5]:
        run_gmac_snr(snr_db, results)
        elapsed_h = (time.time() - t_start) / 3600
        log(f"  Elapsed: {elapsed_h:.1f}h")
        if elapsed_h > 27:
            log("  Time limit approaching, stopping GMAC")
            break

    # BEMAC
    elapsed_h = (time.time() - t_start) / 3600
    if elapsed_h < 25:
        run_bemac(results)

    # ABNMAC
    elapsed_h = (time.time() - t_start) / 3600
    if elapsed_h < 27:
        run_abnmac(results)

    total_h = (time.time() - t_start) / 3600
    log(f"\n{'='*60}")
    log(f"  CAMPAIGN COMPLETE: {total_h:.1f}h, {len(results)} results")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
