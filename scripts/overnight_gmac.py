#!/usr/bin/env python
"""
overnight_gmac.py — Comprehensive overnight Gaussian MAC campaign.

Runs sequentially (1 CPU-heavy job at a time) over ~12 hours:
  Phase 1: MC designs for N=2048 at SNR=6dB (3 classes × 1h = 3h)
  Phase 2: MC designs for SNR=3dB, N=128..2048 (3 classes × 1h = 3h)
  Phase 3: BLER simulations at SNR=6dB and 3dB (all classes, SC + SCL L=32, ~4h)
  Phase 4: Generate plots (~30min)

Usage:
    cd to_git
    nohup python scripts/overnight_gmac.py > results/overnight_gmac.log 2>&1 &
"""

import os
import sys
import time
import json
import subprocess
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd, timeout_s=None):
    """Run a command and stream output."""
    log(f"  CMD: {cmd}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR,
                            capture_output=False, timeout=timeout_s)
    elapsed = time.time() - t0
    log(f"  Done in {elapsed/60:.1f} min (exit={result.returncode})")
    return result.returncode


def phase_design(snr_db, N_values, hours_per_class=1.0):
    """Run MC designs for given SNR and N values."""
    log(f"\n{'='*60}")
    log(f"  MC DESIGN: SNR={snr_db}dB, N={N_values}")
    log(f"{'='*60}")
    N_str = " ".join(str(N) for N in N_values)
    for cls in ["C", "A", "B"]:
        log(f"\n  --- Class {cls} ---")
        run_cmd(
            f"python scripts/run_design_gmac.py --class {cls} "
            f"--N {N_str} --snr-db {snr_db} --hours {hours_per_class} --force",
            timeout_s=int(hours_per_class * 3600 + 300)
        )


def phase_simulate(snr_db, L, hours_per_class=0.5, rho_str="0.3 0.5 0.7 0.9"):
    """Run BLER simulations."""
    log(f"\n{'='*60}")
    log(f"  SIMULATE: SNR={snr_db}dB, L={L}")
    log(f"{'='*60}")
    for cls in ["C", "A", "B"]:
        out = os.path.join(RESULTS_DIR,
                           f"sim_gmac_{cls}_mc_L{L}_snr{snr_db:.0f}dB_final.json")
        log(f"\n  --- Class {cls} L={L} ---")
        run_cmd(
            f"python scripts/simulate_gmac.py --class {cls} --L {L} "
            f"--snr-db {snr_db} --design mc --hours {hours_per_class} "
            f"--rho {rho_str} --output {out}",
            timeout_s=int(hours_per_class * 3600 + 300)
        )


def phase_plots():
    """Generate comprehensive plots from all results."""
    log(f"\n{'='*60}")
    log(f"  GENERATING PLOTS")
    log(f"{'='*60}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from polar.channels import GaussianMAC
    from polar.encoder import polar_encode, build_message
    from polar.design import make_path
    from polar.design_mc import load_design, _select_info_frozen
    from polar.decoder import decode_single
    from polar.decoder_scl import decode_single_list

    designs_dir = os.path.join(PROJECT_DIR, "designs")

    # Load all simulation results
    all_data = {}
    for f in sorted(os.listdir(RESULTS_DIR)):
        if not f.startswith("sim_gmac_") or not f.endswith("_final.json"):
            continue
        try:
            d = json.load(open(os.path.join(RESULTS_DIR, f)))
            snr = d.get("snr_db", 6)
            cls = d.get("class", "?")
            L = d.get("L", 1)
            for r in d.get("results", []):
                if r.get("skipped"):
                    continue
                key = (cls, r["rho"], r["N"], L, snr)
                all_data[key] = r["bler"]
        except:
            continue

    log(f"  Loaded {len(all_data)} data points from result files")

    # If not enough data from files, run quick simulations
    N_values = [8, 16, 32, 64, 128, 256, 512, 1024]
    # Check if N=2048 designs exist
    if os.path.exists(os.path.join(designs_dir, "gmac_C_n11_snr6dB.npz")):
        N_values.append(2048)
        log("  Including N=2048")

    snr_values = [6]
    if os.path.exists(os.path.join(designs_dir, "gmac_C_n10_snr3dB.npz")):
        snr_values.append(3)
        log("  Including SNR=3dB")

    rho_values = [0.3, 0.5, 0.7, 0.9]
    classes_cfg = {
        'C': {'pfrac': 1.0},
        'A': {'pfrac': 0.375},
        'B': {'pfrac': 0.5},
    }

    # Fill gaps with quick simulations
    for snr_db in snr_values:
        sigma2 = 10 ** (-snr_db / 10.0)
        ch = GaussianMAC(sigma2=sigma2)
        I_ZX, I_ZY_X, I_ZXY = ch.capacity()

        Ru_max = {'C': I_ZX, 'A': I_ZY_X, 'B': I_ZXY / 2}
        Rv_max = {'C': I_ZY_X, 'A': I_ZX, 'B': I_ZXY / 2}

        for cls in ['C', 'A', 'B']:
            for rho in rho_values:
                for N in N_values:
                    for L in [1, 32]:
                        key = (cls, rho, N, L, snr_db)
                        if key in all_data:
                            continue

                        n = N.bit_length() - 1
                        design_file = os.path.join(designs_dir,
                                                   f"gmac_{cls}_n{n}_snr{snr_db:.0f}dB.npz")
                        if not os.path.exists(design_file):
                            continue

                        su, sv, _, _, _ = load_design(design_file)
                        path_i = round(classes_cfg[cls]['pfrac'] * N)
                        ku = max(1, min(round(rho * Ru_max[cls] * N), N - 1))
                        kv = max(1, min(round(rho * Rv_max[cls] * N), N - 1))
                        Au, fu = _select_info_frozen(N, su, ku)
                        Av, fv = _select_info_frozen(N, sv, kv)
                        b = make_path(N, path_i=path_i)

                        ncw = 2000 if L == 1 else (500 if N <= 256 else (200 if N <= 512 else 100))
                        rng = np.random.default_rng(42 + n * 1000 + int(rho * 100) + L * 10000 + snr_db * 100000)

                        errors = 0
                        t0 = time.time()
                        for trial in range(ncw):
                            info_u = rng.integers(0, 2, ku).tolist()
                            info_v = rng.integers(0, 2, kv).tolist()
                            u = build_message(N, info_u, Au)
                            v = build_message(N, info_v, Av)
                            x = polar_encode(u.tolist())
                            y = polar_encode(v.tolist())
                            z = ch.sample_batch(np.array(x), np.array(y)).tolist()
                            if L == 1:
                                u_dec, v_dec = decode_single(N, z, b, fu, fv, ch, log_domain=True)
                            else:
                                u_dec, v_dec = decode_single_list(N, z, b, fu, fv, ch, log_domain=True, L=L)
                            if not (all(u_dec[p - 1] == bit for p, bit in zip(Au, info_u)) and
                                    all(v_dec[p - 1] == bit for p, bit in zip(Av, info_v))):
                                errors += 1

                        bler = errors / ncw
                        all_data[key] = bler
                        elapsed = time.time() - t0
                        log(f"  {cls} rho={rho} N={N:5d} L={L:2d} snr={snr_db} "
                            f"ku={ku:3d} kv={kv:3d} BLER={bler:.4f} [{elapsed:.1f}s]")

    # Save all data
    save_data = {str(k): v for k, v in all_data.items()}
    with open(os.path.join(RESULTS_DIR, "gmac_overnight_data.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    log(f"  Saved {len(all_data)} total data points")

    # ── PLOT GENERATION ──
    colors_rho = {0.3: 'tab:green', 0.5: 'tab:blue', 0.7: 'tab:orange', 0.9: 'tab:red'}
    styles_cls = {'C': ('tab:blue', '-', 'o'), 'A': ('tab:orange', '--', 's'), 'B': ('tab:green', '-.', 'D')}
    palette_N = {8: 'gray', 16: 'royalblue', 32: 'darkorange', 64: 'forestgreen',
                 128: 'crimson', 256: 'purple', 512: 'brown', 1024: 'teal', 2048: 'navy'}

    def get(cls, rho, N, L=1, snr=6):
        return all_data.get((cls, rho, N, L, snr), None)

    for snr_db in snr_values:
        sigma2 = 10 ** (-snr_db / 10.0)
        ch = GaussianMAC(sigma2=sigma2)
        I_ZX, I_ZY_X, I_ZXY = ch.capacity()
        Ru_max_d = {'C': I_ZX, 'A': I_ZY_X, 'B': I_ZXY / 2}
        Rv_max_d = {'C': I_ZY_X, 'A': I_ZX, 'B': I_ZXY / 2}
        snr_tag = f"snr{snr_db:.0f}dB"

        # Plot 1: BLER vs N, all classes, SC, rho=0.5
        fig, ax = plt.subplots(figsize=(10, 7))
        for cls in ['C', 'A', 'B']:
            xs, ys = [], []
            for N in N_values:
                v = get(cls, 0.5, N, 1, snr_db)
                if v is not None:
                    xs.append(N)
                    ys.append(max(v, 5e-4))
            if xs:
                c, ls, m = styles_cls[cls]
                ax.semilogy(xs, ys, color=c, linestyle=ls, marker=m,
                            linewidth=2.5, markersize=9, label=f'Class {cls}')
        ax.axhline(0.05, color='gray', linestyle=':', alpha=0.5, label='BLER=5%')
        ax.set_xlabel('Block length N', fontsize=14)
        ax.set_ylabel('BLER', fontsize=14)
        ax.set_title(f'Gaussian MAC: BLER vs N — SC, MC Design, SNR={snr_db}dB, ρ=0.5',
                     fontsize=13)
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim([3e-4, 1.5])
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'gmac_overnight_classes_{snr_tag}.png'), dpi=150)
        plt.close()

        # Plot 2: BLER vs N, multiple rho, per class (SC)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for ax, cls in zip(axes, ['C', 'A', 'B']):
            for rho in rho_values:
                xs, ys = [], []
                for N in N_values:
                    v = get(cls, rho, N, 1, snr_db)
                    if v is not None:
                        xs.append(N)
                        ys.append(max(v, 5e-4))
                if xs:
                    ax.semilogy(xs, ys, color=colors_rho[rho], marker='o',
                                linewidth=2, markersize=7, label=f'ρ={rho}')
            ax.set_xlabel('Block length N', fontsize=12)
            ax.set_title(f'Class {cls}', fontsize=14)
            ax.set_xscale('log', base=2)
            ax.legend(fontsize=9)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_ylim([3e-4, 1.5])
        axes[0].set_ylabel('BLER', fontsize=13)
        fig.suptitle(f'Gaussian MAC: BLER vs N — SC, MC Design, SNR={snr_db}dB',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'gmac_overnight_rho_{snr_tag}.png'), dpi=150)
        plt.close()

        # Plot 3: SC vs SCL L=32
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for ax, cls in zip(axes, ['C', 'A', 'B']):
            for rho in [0.5, 0.7]:
                for L, ls_style in [(1, '-'), (32, '--')]:
                    xs, ys = [], []
                    for N in N_values:
                        v = get(cls, rho, N, L, snr_db)
                        if v is not None:
                            xs.append(N)
                            ys.append(max(v, 5e-4))
                    if xs:
                        lbl = f'{"SC" if L==1 else "SCL L=32"} ρ={rho}'
                        ax.semilogy(xs, ys, color=colors_rho[rho], linestyle=ls_style,
                                    marker='o' if L == 1 else 's', linewidth=2,
                                    markersize=7, label=lbl)
            ax.set_xlabel('Block length N', fontsize=12)
            ax.set_title(f'Class {cls}', fontsize=14)
            ax.set_xscale('log', base=2)
            ax.legend(fontsize=8)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_ylim([3e-4, 1.5])
        axes[0].set_ylabel('BLER', fontsize=13)
        fig.suptitle(f'Gaussian MAC: SC vs SCL L=32 — MC Design, SNR={snr_db}dB',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'gmac_overnight_scl_{snr_tag}.png'), dpi=150)
        plt.close()

        # Plot 4: BLER vs sum rate
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for ax, cls in zip(axes, ['C', 'A', 'B']):
            for N in N_values:
                n = N.bit_length() - 1
                xs, ys = [], []
                for rho in rho_values:
                    ku = max(1, min(round(rho * Ru_max_d[cls] * N), N - 1))
                    kv = max(1, min(round(rho * Rv_max_d[cls] * N), N - 1))
                    sr = ku / N + kv / N
                    v = get(cls, rho, N, 1, snr_db)
                    if v is not None:
                        xs.append(sr)
                        ys.append(max(v, 5e-4))
                if xs:
                    ax.semilogy(xs, ys, color=palette_N[N], marker='o',
                                linewidth=1.5, markersize=6, label=f'$N=2^{{{n}}}$')
            ax.axvline(I_ZXY, color='red', linewidth=2, alpha=0.7,
                       label=f'$C_{{sum}}$={I_ZXY:.2f}')
            ax.set_xlabel('Sum rate (bits)', fontsize=12)
            ax.set_title(f'Class {cls}', fontsize=14)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_ylim([3e-4, 1.5])
        axes[0].set_ylabel('BLER', fontsize=13)
        fig.suptitle(f'Gaussian MAC: BLER vs Sum Rate — SC, MC Design, SNR={snr_db}dB',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'gmac_overnight_rate_{snr_tag}.png'), dpi=150)
        plt.close()

        log(f"  Saved 4 plots for SNR={snr_db}dB")

    # Plot 5: SNR comparison (if multiple SNRs)
    if len(snr_values) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        snr_colors = {3: 'tab:red', 6: 'tab:blue'}
        for ax, cls in zip(axes, ['C', 'A', 'B']):
            for snr_db in snr_values:
                xs, ys = [], []
                for N in N_values:
                    v = get(cls, 0.5, N, 1, snr_db)
                    if v is not None:
                        xs.append(N)
                        ys.append(max(v, 5e-4))
                if xs:
                    ax.semilogy(xs, ys, color=snr_colors.get(snr_db, 'black'),
                                marker='o', linewidth=2, markersize=7,
                                label=f'SNR={snr_db}dB')
            ax.set_xlabel('Block length N', fontsize=12)
            ax.set_title(f'Class {cls}', fontsize=14)
            ax.set_xscale('log', base=2)
            ax.legend(fontsize=10)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_ylim([3e-4, 1.5])
        axes[0].set_ylabel('BLER', fontsize=13)
        fig.suptitle('Gaussian MAC: BLER vs N — SC, MC Design, ρ=0.5', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'gmac_overnight_snr_compare.png'), dpi=150)
        plt.close()
        log("  Saved SNR comparison plot")

    log("  All plots complete!")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log("=" * 60)
    log("  OVERNIGHT GAUSSIAN MAC CAMPAIGN")
    log("=" * 60)

    # Phase 1: MC designs for N=2048 at SNR=6dB
    log("\n" + "=" * 60)
    log("  PHASE 1: MC Design N=2048, SNR=6dB")
    log("=" * 60)
    phase_design(snr_db=6, N_values=[2048], hours_per_class=1.5)

    # Phase 2: MC designs for SNR=3dB
    log("\n" + "=" * 60)
    log("  PHASE 2: MC Design SNR=3dB, N=128..2048")
    log("=" * 60)
    phase_design(snr_db=3, N_values=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                 hours_per_class=1.5)

    # Phase 3: BLER simulations
    log("\n" + "=" * 60)
    log("  PHASE 3: BLER Simulations")
    log("=" * 60)

    for snr_db in [6, 3]:
        phase_simulate(snr_db=snr_db, L=1, hours_per_class=0.5)
        phase_simulate(snr_db=snr_db, L=32, hours_per_class=0.5)

    # Phase 4: Plots
    phase_plots()

    total = time.time() - t_start
    log(f"\n{'='*60}")
    log(f"  CAMPAIGN COMPLETE: {total/3600:.1f} hours")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
