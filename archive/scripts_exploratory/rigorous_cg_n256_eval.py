#!/usr/bin/env python3
"""
Rigorous BLER evaluation of the CG (Computational Graph) neural decoder
at N=256 for the 2-user Gaussian MAC.

- Loads the CG model via the `tree.*` prefix wrapper format used by the
  training scripts (train_n256_long, train_30hr_campaign, etc.).
- Evaluates 5000 codewords on SNR=6dB GMAC using the Class B design file.
- Computes the 95% Wilson confidence interval.
- Supports batched evaluation for throughput.
- Saves partial progress every 500 codewords to a JSON file.

Output: results/cg_n256_rigorous_eval.json
"""
import sys, os, math, time, json, argparse
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

import numpy as np
import torch

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder


RESULTS_PATH = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results/cg_n256_rigorous_eval.json'


def wilson_ci(errs, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = errs / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (centre - margin, centre + margin)


def load_cg_model(ckpt_path, d=16, hidden=64, n_layers=2):
    """
    Load a CG decoder checkpoint.

    Checkpoints produced by the n256_long / campaign training scripts use a
    wrapper with a `tree.*` prefix and also include a discrete `embedding_z`
    we skip.  They also contain the `z_encoder.*` keys (no prefix) which we
    load directly into the GmacNeuralCompGraphDecoder's z_encoder.
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']

    model = GmacNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)
    model_sd = model.state_dict()

    has_tree_prefix = any(k.startswith('tree.') for k in sd.keys())

    loaded_keys = 0
    if has_tree_prefix:
        for k, v in sd.items():
            if k.startswith('tree.'):
                new_k = k[len('tree.'):]
                if 'embedding_z' in new_k:
                    continue  # skip discrete embedding (replaced by z_encoder)
                if new_k in model_sd and model_sd[new_k].shape == v.shape:
                    model_sd[new_k] = v
                    loaded_keys += 1
            elif k.startswith('z_encoder.'):
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    loaded_keys += 1
        model.load_state_dict(model_sd)
    else:
        model.load_state_dict(sd)
        loaded_keys = len(sd)

    model.eval()
    return model, loaded_keys


def make_batch(rng, Au, Av, N):
    """Sample a batch of info-bit patterns and encode into X, Y arrays."""
    B = 1  # single-sample path uses rng
    # kept for api compatibility but not used
    raise NotImplementedError


def run_eval(model, channel, b, Au, Av, frozen_u, frozen_v, N,
             n_total, seed=999, batch_size=8, tag='eval',
             save_every=500):
    """
    Run the BLER evaluation.

    Uses independent numpy RNGs for reproducibility:
      - rng_bits: to draw the info bits per codeword (seeded by seed)
      - global np.random: seeded once so channel sampling is reproducible
    """
    rng_bits = np.random.default_rng(seed)
    np.random.seed(seed + 1)  # for GaussianMAC.sample_batch

    errs = 0
    done = 0
    t0 = time.time()
    per_step_errs_log = []  # (count, n)

    au_idx = np.array(sorted(Au), dtype=np.int64) - 1  # 0-indexed
    av_idx = np.array(sorted(Av), dtype=np.int64) - 1

    while done < n_total:
        this_B = min(batch_size, n_total - done)
        U = np.zeros((this_B, N), dtype=np.int32)
        V = np.zeros((this_B, N), dtype=np.int32)
        U[:, au_idx] = rng_bits.integers(0, 2, size=(this_B, au_idx.size), dtype=np.int32)
        V[:, av_idx] = rng_bits.integers(0, 2, size=(this_B, av_idx.size), dtype=np.int32)

        X = polar_encode_batch(U)
        Y = polar_encode_batch(V)
        Z = channel.sample_batch(X, Y)
        z_t = torch.from_numpy(Z).float()

        with torch.no_grad():
            _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u, frozen_v)

        # u_hat, v_hat are dicts {1-indexed position: tensor of shape (B,)}
        # Gather info bits only and compare per-batch element.
        u_pred = np.zeros((this_B, au_idx.size), dtype=np.int32)
        for j, pos in enumerate(sorted(Au)):
            u_pred[:, j] = u_hat[pos].cpu().numpy().round().astype(np.int32)
        v_pred = np.zeros((this_B, av_idx.size), dtype=np.int32)
        for j, pos in enumerate(sorted(Av)):
            v_pred[:, j] = v_hat[pos].cpu().numpy().round().astype(np.int32)

        u_true_bits = U[:, au_idx]
        v_true_bits = V[:, av_idx]
        u_err_row = (u_pred != u_true_bits).any(axis=1)
        v_err_row = (v_pred != v_true_bits).any(axis=1)
        row_err = u_err_row | v_err_row
        errs += int(row_err.sum())
        done += this_B

        if done % save_every == 0 or done >= n_total:
            elapsed = time.time() - t0
            rate = done / elapsed
            bler = errs / done
            lo, hi = wilson_ci(errs, done)
            print(f'  [{tag}] [{done}/{n_total}] errs={errs} '
                  f'BLER={bler:.5f} CI=[{lo:.5f},{hi:.5f}] '
                  f'{rate:.2f} cw/s  eta={((n_total-done)/max(rate,1e-9)):.0f}s',
                  flush=True)
            per_step_errs_log.append({'done': done, 'errs': errs,
                                      'bler': bler, 'ci_low': lo, 'ci_high': hi})

    elapsed = time.time() - t0
    bler = errs / n_total
    lo, hi = wilson_ci(errs, n_total)
    return {
        'n_cw': n_total,
        'errs': errs,
        'bler': bler,
        'ci_low': lo,
        'ci_high': hi,
        'time_s': elapsed,
        'rate_cw_per_s': n_total / elapsed,
        'progress': per_step_errs_log,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-cw', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--seed', type=int, default=999)
    args = parser.parse_args()

    # ---- Setup -----------------------------------------------------------
    N = 256; n = 8; ku = 123; kv = 123
    SNR_DB = 6.0; SIGMA2 = 10 ** (-SNR_DB / 10)
    DESIGN_FILE = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs/gmac_B_n8_snr6dB.npz'

    print(f'Setup: N={N}, SNR={SNR_DB}dB, sigma2={SIGMA2:.5f}', flush=True)
    Au, Av, frozen_u, frozen_v, pe_u, pe_v, path_i = design_from_file(
        DESIGN_FILE, n, ku=ku, kv=kv)
    print(f'Design: ku={len(Au)}, kv={len(Av)}, path_i={path_i}', flush=True)
    b = make_path(N, path_i)
    channel = GaussianMAC(sigma2=SIGMA2)

    # ---- Candidate checkpoints ------------------------------------------
    checkpoints = [
        ('n256_long_best',
         '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/saved_models/n256_long_best.pt'),
        ('campaign_n256_sched_best',
         '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/saved_models/campaign_n256_sched_best.pt'),
    ]

    results = {
        'checkpoints': {},
        'sc_ref': 0.005,
        'channel': f'GMAC SNR={SNR_DB}dB',
        'frozen_design': 'gmac_B_n8_snr6dB.npz',
        'ku_kv': [ku, kv],
        'evaluation_seed': args.seed,
        'batch_size': args.batch_size,
        'path_i': int(path_i),
    }

    # ---- Benchmark (100 cw) each checkpoint first -----------------------
    print('\n=== Benchmark (100 cw) ===', flush=True)
    bench = {}
    for name, path in checkpoints:
        print(f'\n--- {name} ---', flush=True)
        model, nk = load_cg_model(path, d=16, hidden=64)
        print(f'  Loaded {nk} tensors, params={model.count_parameters()}', flush=True)
        r = run_eval(model, channel, b, Au, Av, frozen_u, frozen_v, N,
                     n_total=100, seed=args.seed + 100000,  # separate seed for bench
                     batch_size=args.batch_size, tag=name + '-bench',
                     save_every=100)
        bench[name] = r
        print(f'  Benchmark: {r["rate_cw_per_s"]:.2f} cw/s '
              f'→ projected {args.n_cw} cw: {args.n_cw/r["rate_cw_per_s"]:.0f}s '
              f'= {args.n_cw/r["rate_cw_per_s"]/60:.1f} min', flush=True)

    results['benchmark'] = bench
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'Benchmark saved to {RESULTS_PATH}', flush=True)

    if args.benchmark_only:
        return

    # ---- Full eval ------------------------------------------------------
    wall_start = time.time()
    for name, path in checkpoints:
        print(f'\n=== FULL EVAL: {name} ({args.n_cw} cw) ===', flush=True)
        model, _ = load_cg_model(path, d=16, hidden=64)
        r = run_eval(model, channel, b, Au, Av, frozen_u, frozen_v, N,
                     n_total=args.n_cw, seed=args.seed,
                     batch_size=args.batch_size, tag=name,
                     save_every=500)
        r['ratio_to_sc'] = r['bler'] / results['sc_ref']
        r['params'] = model.count_parameters()
        results['checkpoints'][name] = r
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f'  >> {name}: BLER={r["bler"]:.5f} '
              f'CI=[{r["ci_low"]:.5f}, {r["ci_high"]:.5f}] '
              f'ratio/SC={r["ratio_to_sc"]:.2f}x', flush=True)

    wall = time.time() - wall_start

    print('\n=== CG DECODER N=256 RIGOROUS EVALUATION ===', flush=True)
    for name, r in results['checkpoints'].items():
        print(f'\nCheckpoint: {name}.pt', flush=True)
        print(f'  Codewords: {r["n_cw"]}', flush=True)
        print(f'  Errors: {r["errs"]}', flush=True)
        print(f'  BLER: {r["bler"]:.5f}', flush=True)
        print(f'  95% CI: [{r["ci_low"]:.5f}, {r["ci_high"]:.5f}]', flush=True)
        print(f'  vs SC (0.005): {r["ratio_to_sc"]:.2f}x', flush=True)
    print(f'\nWall time: {wall/60:.1f} minutes', flush=True)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved to {RESULTS_PATH}', flush=True)


if __name__ == '__main__':
    main()
