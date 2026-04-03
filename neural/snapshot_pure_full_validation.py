"""
snapshot_pure_full_validation.py
================================
Full validation of snapshot-pure decoder:
1. Train at N=512 (curriculum from N=256 checkpoint)
2. Train at N=1024 (curriculum from N=512)
3. Evaluate ALL N values (32..1024) with 5000 codewords
4. Save results to snapshot_pure_final_results.json
"""

import sys, os, time, datetime, math, json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural.snapshot_pure_scaling_v2 import (
    SnapshotModel, D_EMB, D_HIDDEN, EPS,
    generate_snapshots, train_snapshot,
    neural_sc_decode, eval_bler_neural, eval_bler_sc,
    prob_from_log, log as slog,
)
from polar.channels import GaussianMAC
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.decoder import build_log_W_leaf, decode_single
from polar.design import make_path
from polar.design_mc import design_from_file

DESIGN_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
SAVED_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'snapshot_pure_full_validation.log')

SNR_dB = 6.0
SIGMA2 = 10 ** (-SNR_dB / 10)

# All evaluation configs
CONFIGS = {
    32:   {'ku': 15,  'kv': 15,  'design': 'gmac_B_n5_snr6dB.npz'},
    64:   {'ku': 31,  'kv': 31,  'design': 'gmac_B_n6_snr6dB.npz'},
    128:  {'ku': 62,  'kv': 62,  'design': 'gmac_B_n7_snr6dB.npz'},
    256:  {'ku': 123, 'kv': 123, 'design': 'gmac_B_n8_snr6dB.npz'},
    512:  {'ku': 246, 'kv': 246, 'design': 'gmac_B_n9_snr6dB.npz'},
    1024: {'ku': 492, 'kv': 492, 'design': 'gmac_B_n10_snr6dB.npz'},
}

SC_REF_BLER = {32: 0.046, 64: 0.025, 128: 0.016, 256: 0.005, 512: 0.001, 1024: 0.001}

log_fh = None

def log(msg=""):
    print(msg, flush=True)
    if log_fh:
        log_fh.write(msg + "\n")
        log_fh.flush()


def get_design(N):
    """Get frozen sets for given N using MC design."""
    n = int(math.log2(N))
    cfg = CONFIGS[N]
    ku, kv = cfg['ku'], cfg['kv']
    path_i = N // 2  # Class B
    mc_path = os.path.join(DESIGN_DIR, cfg['design'])
    Au, Av, frozen_u, frozen_v, _, _, _ = design_from_file(mc_path, n, ku, kv)
    actual_ku = N - len(frozen_u)
    actual_kv = N - len(frozen_v)
    assert actual_ku == ku, f"ku mismatch: expected {ku}, got {actual_ku}"
    assert actual_kv == kv, f"kv mismatch: expected {kv}, got {actual_kv}"
    return frozen_u, frozen_v, ku, kv, path_i


def main():
    global log_fh
    log_fh = open(LOG_FILE, 'w')
    overall_t0 = time.time()
    TIME_BUDGET = 3600  # 1 hour

    log("=" * 70)
    log("Snapshot-Pure Full Validation")
    log(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"SNR={SNR_dB}dB, sigma2={SIGMA2:.6f}, Class B")
    log("=" * 70)

    # =========================================================================
    # Load N=256 checkpoint
    # =========================================================================
    model = SnapshotModel(d=D_EMB, h=D_HIDDEN)
    ckpt_256 = os.path.join(SAVED_DIR, 'snapshot_pure_v2_N256.pt')
    log(f"\nLoading N=256 checkpoint: {ckpt_256}")
    state = torch.load(ckpt_256, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {total_params} params loaded")

    # =========================================================================
    # PHASE 1: Train at N=512
    # =========================================================================
    log(f"\n{'='*70}")
    log("PHASE 1: Train at N=512")
    log(f"{'='*70}")

    N512 = 512
    n512 = 9
    frozen_u_512, frozen_v_512, ku_512, kv_512, path_i_512 = get_design(N512)
    log(f"  Design: N={N512}, ku={ku_512}, kv={kv_512}, path_i={path_i_512}")

    # Generate snapshots
    np.random.seed(42 + 512)
    t0 = time.time()
    left_s, right_s, parent_s = generate_snapshots(
        n512, N512, 100, SIGMA2, frozen_u_512, frozen_v_512, path_i_512)
    log(f"  Snapshots: {len(left_s)} L, {len(right_s)} R, {len(parent_s)} P "
        f"in {time.time()-t0:.1f}s")

    # Train
    import neural.snapshot_pure_scaling_v2 as spmod
    spmod.log_fh = log_fh  # share log file
    train_snapshot(model, left_s, right_s, parent_s,
                   n_iters=15000, batch_size=384, lr=3e-4, label="[N=512] ")

    # Save checkpoint
    ckpt_512 = os.path.join(SAVED_DIR, 'snapshot_pure_v2_N512.pt')
    torch.save(model.state_dict(), ckpt_512)
    log(f"  Saved: {ckpt_512}")

    # =========================================================================
    # PHASE 2: Train at N=1024
    # =========================================================================
    elapsed = time.time() - overall_t0
    log(f"\n{'='*70}")
    log(f"PHASE 2: Train at N=1024 (elapsed: {elapsed/60:.1f}min)")
    log(f"{'='*70}")

    N1024 = 1024
    n1024 = 10
    frozen_u_1024, frozen_v_1024, ku_1024, kv_1024, path_i_1024 = get_design(N1024)
    log(f"  Design: N={N1024}, ku={ku_1024}, kv={kv_1024}, path_i={path_i_1024}")

    # Generate snapshots
    np.random.seed(42 + 1024)
    t0 = time.time()
    left_s, right_s, parent_s = generate_snapshots(
        n1024, N1024, 50, SIGMA2, frozen_u_1024, frozen_v_1024, path_i_1024)
    log(f"  Snapshots: {len(left_s)} L, {len(right_s)} R, {len(parent_s)} P "
        f"in {time.time()-t0:.1f}s")

    # Train
    train_snapshot(model, left_s, right_s, parent_s,
                   n_iters=15000, batch_size=384, lr=3e-4, label="[N=1024] ")

    # Save checkpoint
    ckpt_1024 = os.path.join(SAVED_DIR, 'snapshot_pure_v2_N1024.pt')
    torch.save(model.state_dict(), ckpt_1024)
    log(f"  Saved: {ckpt_1024}")

    # =========================================================================
    # PHASE 3: Evaluate ALL N values
    # =========================================================================
    elapsed = time.time() - overall_t0
    log(f"\n{'='*70}")
    log(f"PHASE 3: Full Evaluation (elapsed: {elapsed/60:.1f}min)")
    log(f"{'='*70}")

    results = {}

    for N_val in [32, 64, 128, 256, 512, 1024]:
        elapsed = time.time() - overall_t0
        remaining = TIME_BUDGET - elapsed
        log(f"\n--- Evaluating N={N_val} (elapsed: {elapsed/60:.1f}min, "
            f"remaining: {remaining/60:.1f}min) ---")

        n_val = int(math.log2(N_val))
        frozen_u, frozen_v, ku, kv, path_i = get_design(N_val)

        # Decide number of codewords based on time
        if N_val >= 512 and remaining < 1800:
            num_cw = 2000
            log(f"  Using {num_cw} codewords (time-constrained)")
        else:
            num_cw = 5000
            log(f"  Using {num_cw} codewords")

        # Time a single SC decode to estimate
        np.random.seed(999 + N_val)
        t0 = time.time()
        test_bler = eval_bler_sc(n_val, N_val, 10, SIGMA2, frozen_u, frozen_v, path_i)
        sc_per_cw = (time.time() - t0) / 10
        est_sc_time = sc_per_cw * num_cw
        log(f"  SC estimate: {sc_per_cw:.3f}s/cw, total ~{est_sc_time/60:.1f}min")

        # If estimated time > 30 min for SC+Neural combined, reduce
        if est_sc_time > 900:  # 15 min for SC alone
            num_cw = min(num_cw, max(500, int(900 / sc_per_cw)))
            log(f"  Reduced to {num_cw} codewords")

        # Time single neural decode
        t0 = time.time()
        test_bler_n = eval_bler_neural(model, n_val, N_val, 5, SIGMA2,
                                        frozen_u, frozen_v, path_i)
        nn_per_cw = (time.time() - t0) / 5
        est_nn_time = nn_per_cw * num_cw
        log(f"  Neural estimate: {nn_per_cw:.3f}s/cw, total ~{est_nn_time/60:.1f}min")

        # If combined time too long, reduce further
        total_est = (sc_per_cw + nn_per_cw) * num_cw
        if total_est > 1800:  # 30 min combined
            num_cw = min(num_cw, max(500, int(1800 / (sc_per_cw + nn_per_cw))))
            log(f"  Further reduced to {num_cw} codewords (combined time constraint)")

        # Evaluate SC
        np.random.seed(12345 + N_val)
        t0 = time.time()
        sc_bler = eval_bler_sc(n_val, N_val, num_cw, SIGMA2, frozen_u, frozen_v, path_i)
        sc_time = time.time() - t0
        sc_errors = int(round(sc_bler * num_cw))
        log(f"  SC:     BLER={sc_bler:.6f} ({sc_errors}/{num_cw}) in {sc_time:.1f}s "
            f"(ref~{SC_REF_BLER.get(N_val, '?')})")

        # Evaluate Neural (same seed for same codewords)
        np.random.seed(12345 + N_val)
        t0 = time.time()
        nn_bler = eval_bler_neural(model, n_val, N_val, num_cw, SIGMA2,
                                    frozen_u, frozen_v, path_i)
        nn_time = time.time() - t0
        nn_errors = int(round(nn_bler * num_cw))
        log(f"  Neural: BLER={nn_bler:.6f} ({nn_errors}/{num_cw}) in {nn_time:.1f}s")

        ratio = nn_bler / max(sc_bler, 1e-10)
        log(f"  Ratio: {ratio:.3f}")

        results[N_val] = {
            'N': N_val, 'ku': ku, 'kv': kv,
            'nn_bler': nn_bler, 'sc_bler': sc_bler, 'ratio': ratio,
            'nn_errors': nn_errors, 'sc_errors': sc_errors,
            'n_codewords': num_cw,
            'sc_ref': SC_REF_BLER.get(N_val),
            'nn_time_s': round(nn_time, 1),
            'sc_time_s': round(sc_time, 1),
        }

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - overall_t0
    log(f"\n{'='*70}")
    log("FINAL SUMMARY: Snapshot-Pure Neural MAC Decoder")
    log(f"{'='*70}")
    log(f"Total time: {total_time/60:.1f} min")
    log(f"Model: {total_params} params, d={D_EMB}, hidden={D_HIDDEN}")
    log(f"SNR={SNR_dB}dB, Class B, sigma2={SIGMA2:.6f}")
    log("")
    log(f"{'N':>5s}  {'Neural BLER':>12s}  {'SC BLER':>12s}  {'Ratio':>7s}  "
        f"{'NN Err':>7s}  {'SC Err':>7s}  {'n_cw':>6s}")
    log(f"{'-'*5}  {'-'*12}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
    for N_val in sorted(results.keys()):
        r = results[N_val]
        log(f"{r['N']:5d}  {r['nn_bler']:12.6f}  {r['sc_bler']:12.6f}  "
            f"{r['ratio']:7.3f}  {r['nn_errors']:7d}  {r['sc_errors']:7d}  "
            f"{r['n_codewords']:6d}")
    log(f"{'='*70}")
    log(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__),
                                 'snapshot_pure_final_results.json')
    save_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': {
            'snr_dB': SNR_dB, 'sigma2': SIGMA2,
            'class': 'B', 'd_emb': D_EMB, 'd_hidden': D_HIDDEN,
            'total_params': total_params,
        },
        'results': {str(k): v for k, v in results.items()},
        'total_time_min': round(total_time / 60, 1),
    }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    log(f"\nResults saved to: {results_path}")

    log_fh.close()


if __name__ == "__main__":
    main()
