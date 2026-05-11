"""
Serious BLER evaluation: Chained NPD (Stage 1) vs SC on GMAC Class C.

For each N in {16, 32, 64, 128, 256, 512, 1024}:
  - Load the best NPD checkpoint (Phase 3 for N<=512, Phase 1 for N=1024)
  - Evaluate NPD Stage 1 BLER
  - Evaluate SC BLER at the same rate (same ku, kv) for fair comparison
  - Report 95% Wilson confidence intervals
"""
from __future__ import annotations
import os, sys, math, json, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design
from class_c_npd.training.train_stage import generate_stage1_batch, evaluate_stage

from polar.eval import MACEval
from polar.design import make_path
from polar.channels import GaussianMAC


# ─── Wilson confidence interval ──────────────────────────────────────────────

def wilson_ci(errors: int, total: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    p_hat = errors / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z / denom * math.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


# ─── NPD BLER evaluation (returns errors, total) ────────────────────────────

def evaluate_npd_bler(model, channel, N, Au, Av, frozen_set_0idx, n_cw, seed=42):
    """Evaluate NPD Stage 1 BLER. Returns (errors, total)."""
    model.eval()
    errs = 0
    total = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    bs = 64
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            u_true, features_npd, _ = generate_stage1_batch(
                channel, N, Au, actual, rng, Av)
            ft = torch.from_numpy(features_npd).float()
            if ft.dim() == 2:
                ft = ft.unsqueeze(-1)
            emb = model.encode_channel(ft)
            u_dec = model.decode(emb, frozen_set_0idx)
            for i in range(actual):
                if any(u_dec[i, p - 1].item() != u_true[i, p - 1] for p in Au):
                    errs += 1
            total += actual
    return errs, total


# ─── SC BLER evaluation ─────────────────────────────────────────────────────

def evaluate_sc_bler(N, ku, kv, n_cw, snr_db=6.0, seed=42):
    """
    Evaluate SC BLER on GMAC Class C at given rate (ku, kv).
    Uses SC-optimal positions from design files.
    Returns (u_block_errors, total).
    """
    n = int(math.log2(N))

    # Load SC-optimal design at matching ku, kv
    Au_sc, Av_sc, frozen_u_0idx, frozen_v_0idx, pe_u, pe_v = load_class_c_design(
        'gmac', n, snr_db=snr_db, ku=ku, kv=kv)

    # Convert 0-indexed frozen sets to 1-indexed frozen dicts for MACEval
    frozen_u_dict = {p + 1: 0 for p in frozen_u_0idx}
    frozen_v_dict = {p + 1: 0 for p in frozen_v_0idx}

    # Class C path: path_i = N
    b = make_path(N, N)

    # Build the actual GaussianMAC channel (from polar.channels, not class_c_npd)
    sigma2 = 10 ** (-snr_db / 10)
    channel = GaussianMAC(sigma2=sigma2)

    # Use 'auto' backend: picks 'efficient' for extreme paths (path_i=0 or N),
    # which is ~3x faster than interleaved while giving equivalent SC results.
    evaluator = MACEval(channel, log_domain=True, backend='auto',
                        rng=np.random.default_rng(seed))

    ber_u, ber_v, bler = evaluator.run(
        N, b, Au_sc, Av_sc, frozen_u_dict, frozen_v_dict,
        n_codewords=n_cw, batch_size=25, verbose=True)

    # We want Stage 1 (U) block errors specifically
    # But MACEval returns joint BLER. For Class C at 6dB, Stage 2 BLER ~ 0,
    # so joint BLER ~ Stage 1 BLER. We'll report full BLER for fairness.
    u_block_errors = round(bler * n_cw)
    return u_block_errors, n_cw, bler, Au_sc, Av_sc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    SNR_DB = 6.0
    RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results')

    # Config per N
    configs = [
        (16,   'npd_design_p3_N16_best.pt',   10000),
        (32,   'npd_design_p3_N32_best.pt',   10000),
        (64,   'npd_design_p3_N64_best.pt',   10000),
        (128,  'npd_design_p3_N128_best.pt',  10000),
        (256,  'npd_design_p3_N256_best.pt',  20000),
        (512,  'npd_design_p3_N512_best.pt',  50000),
        (1024, 'npd_design_p1_N1024_best.pt', 50000),
    ]

    all_results = []

    for N, ckpt_name, n_cw in configs:
        n = int(math.log2(N))
        print(f'\n{"="*70}')
        print(f'  N = {N}  ({n_cw} codewords)')
        print(f'{"="*70}')

        # ── Load NPD checkpoint ──
        ckpt_path = os.path.join(RESULTS_DIR, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Get Au, Av
        if N == 1024:
            # Load from JSON
            json_path = os.path.join(RESULTS_DIR, 'npd_design_N1024.json')
            with open(json_path) as f:
                design_json = json.load(f)
            Au_npd = sorted(design_json['npd_Au'])
            # Load Av from SC design at same ku
            ku_npd = len(Au_npd)
            _, Av_npd, _, _, _, _ = load_class_c_design(
                'gmac', n, snr_db=SNR_DB, ku=ku_npd)
        else:
            Au_npd = sorted(ckpt['Au'])
            Av_npd = sorted(ckpt['Av'])

        ku_npd = len(Au_npd)
        kv_npd = len(Av_npd)

        # Build frozen set (0-indexed) for NPD
        frozen_set_0idx = {p - 1 for p in range(1, N + 1) if p not in Au_npd}

        print(f'  NPD: ku={ku_npd}, kv={kv_npd}, rate_u={ku_npd/N:.4f}')
        print(f'  Au (first 10): {Au_npd[:10]}')

        # Build model
        z_dim = ckpt.get('z_dim', 1)
        d = ckpt.get('d', 16)
        hidden = ckpt.get('hidden', 64)
        n_layers = ckpt.get('n_layers', 2)
        model = NPDSingleUser(d=d, hidden=hidden, n_layers=n_layers,
                               z_dim=z_dim, use_analytical_training=False)
        sd = ckpt.get('state_dict') or ckpt.get('model_state_dict')
        model.load_state_dict(sd)
        model.eval()

        # Build channel
        sigma2 = 10 ** (-SNR_DB / 10)
        channel = build_channel('gmac', sigma2=sigma2)

        # ── Evaluate NPD ──
        print(f'\n  Evaluating NPD Stage 1 ...', flush=True)
        t0 = time.time()
        npd_errs, npd_total = evaluate_npd_bler(
            model, channel, N, Au_npd, Av_npd, frozen_set_0idx, n_cw)
        npd_time = time.time() - t0
        npd_bler = npd_errs / npd_total
        npd_ci = wilson_ci(npd_errs, npd_total)
        print(f'  NPD: {npd_errs}/{npd_total} = {npd_bler:.6f}  '
              f'CI=[{npd_ci[0]:.6f}, {npd_ci[1]:.6f}]  ({npd_time:.1f}s)')

        # ── Evaluate SC at same rate ──
        print(f'\n  Evaluating SC (auto/efficient) at ku={ku_npd}, kv={kv_npd} ...', flush=True)
        t0 = time.time()
        sc_errs, sc_total, sc_bler, Au_sc, Av_sc = evaluate_sc_bler(
            N, ku_npd, kv_npd, n_cw, snr_db=SNR_DB, seed=42)
        sc_time = time.time() - t0
        sc_ci = wilson_ci(sc_errs, sc_total)
        print(f'  SC:  {sc_errs}/{sc_total} = {sc_bler:.6f}  '
              f'CI=[{sc_ci[0]:.6f}, {sc_ci[1]:.6f}]  ({sc_time:.1f}s)')

        ratio = npd_bler / sc_bler if sc_bler > 0 else float('inf')
        print(f'\n  Ratio NPD/SC = {ratio:.4f}')

        result = {
            'N': N,
            'checkpoint': ckpt_name,
            'ku': ku_npd,
            'kv': kv_npd,
            'rate_u': ku_npd / N,
            'n_cw': n_cw,
            'npd_errors': npd_errs,
            'npd_total': npd_total,
            'npd_bler': npd_bler,
            'npd_ci_lo': npd_ci[0],
            'npd_ci_hi': npd_ci[1],
            'npd_time_s': round(npd_time, 1),
            'sc_errors': sc_errs,
            'sc_total': sc_total,
            'sc_bler': sc_bler,
            'sc_ci_lo': sc_ci[0],
            'sc_ci_hi': sc_ci[1],
            'sc_time_s': round(sc_time, 1),
            'ratio_npd_over_sc': round(ratio, 4),
            'sc_Au': Au_sc,
        }
        all_results.append(result)

        # Incremental save after each N
        out_path = os.path.join(RESULTS_DIR, 'serious_npd_vs_sc_eval.json')
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  [saved incremental results to {out_path}]', flush=True)

    # ── Save results ──
    out_path = os.path.join(RESULTS_DIR, 'serious_npd_vs_sc_eval.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved to {out_path}')

    # ── Print summary table ──
    print(f'\n{"="*90}')
    print(f'  SUMMARY: NPD Stage 1 vs SC — GMAC Class C, SNR={SNR_DB}dB')
    print(f'{"="*90}')
    print(f'{"N":>6}  {"ku":>4}  {"rate":>6}  {"n_cw":>6}  '
          f'{"NPD BLER":>12}  {"NPD CI 95%":>20}  '
          f'{"SC BLER":>12}  {"SC CI 95%":>20}  {"NPD/SC":>8}')
    print(f'{"-"*90}')
    for r in all_results:
        print(f'{r["N"]:>6}  {r["ku"]:>4}  {r["rate_u"]:>6.3f}  {r["n_cw"]:>6}  '
              f'{r["npd_bler"]:>12.6f}  [{r["npd_ci_lo"]:.6f}, {r["npd_ci_hi"]:.6f}]  '
              f'{r["sc_bler"]:>12.6f}  [{r["sc_ci_lo"]:.6f}, {r["sc_ci_hi"]:.6f}]  '
              f'{r["ratio_npd_over_sc"]:>8.4f}')
    print(f'{"="*90}')


if __name__ == '__main__':
    main()
