#!/usr/bin/env python3
"""
n256_inference_errors.py

Run the CG decoder in FREE-RUNNING mode (no teacher forcing) at N=256.
Track per-STEP error rates: for each of the 246 info steps, count how
often the decoder's decision is wrong.

Then compare to SC's per-step error rates.

If errors cascade from early steps to late steps, we should see:
  - Early steps have error rates close to SC
  - Late steps have much higher error rates (from cascade)

Output: results/n256_inference_errors.json
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder_interleaved import decode_single
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 256; n_log = 8; ku = kv = 123
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
SEED = 42; N_CW = 5000; BATCH = 64


def load_nn(ckpt_path, d=16, hidden=64):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model = GmacNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=2)
    model_sd = model.state_dict()
    for k, v in sd.items():
        nk = k
        if nk.startswith('tree.'): nk = nk[5:]
        elif nk.startswith('z_enc.'): nk = 'z_encoder.' + nk[6:]
        if 'embedding_z' in nk: continue
        if nk in model_sd and model_sd[nk].shape == v.shape:
            model_sd[nk] = v
    model.load_state_dict(model_sd)
    model.eval()
    return model


def main():
    os.chdir(ROOT)
    Au, Av, frozen_u, frozen_v, pe_u, pe_v, path_i = design_from_file(
        f'designs/gmac_B_n{n_log}_snr6dB.npz', n_log, ku=ku, kv=kv)
    b = make_path(N, path_i)
    channel = GaussianMAC(sigma2=SIGMA2)

    # Build step metadata
    step_meta = []
    i_u, i_v = 0, 0
    for step in range(2*N):
        gamma = b[step]
        if gamma == 0: i_u += 1; i_t = i_u; fdict = frozen_u
        else: i_v += 1; i_t = i_v; fdict = frozen_v
        if i_t in fdict: continue
        step_meta.append({'step': step, 'pos': i_t, 'user': gamma,
                          'pe': float(pe_u[i_t-1]) if gamma==0 else float(pe_v[i_t-1])})
    n_info = len(step_meta)
    print(f"Info steps: {n_info}")

    # Generate data
    rng = np.random.default_rng(SEED)
    U_info = rng.integers(0, 2, size=(N_CW, ku), dtype=np.int32)
    V_info = rng.integers(0, 2, size=(N_CW, kv), dtype=np.int32)
    U_msg = build_message_batch(N, U_info, Au)
    V_msg = build_message_batch(N, V_info, Av)
    X = polar_encode_batch(U_msg)
    Y = polar_encode_batch(V_msg)
    np.random.seed(SEED + 7919)
    Z = channel.sample_batch(X, Y).astype(np.float32)

    # ── NN inference (free-running) ───────────────────────────────────────
    model = load_nn('saved_models/ncg_gmac_mlp_N256.pt')
    nn_errors = np.zeros(n_info, dtype=np.int64)
    nn_count = np.zeros(n_info, dtype=np.int64)

    print("Running NN (free-running)...", flush=True)
    t0 = time.time()
    for start in range(0, N_CW, BATCH):
        end = min(start + BATCH, N_CW)
        bs = end - start
        z_t = torch.from_numpy(Z[start:end]).float()
        with torch.no_grad():
            # No u_true/v_true → free-running mode
            _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u, frozen_v)
        # Count errors at each info position
        idx = 0
        i_u, i_v = 0, 0
        for step in range(2*N):
            gamma = b[step]
            if gamma == 0: i_u += 1; i_t = i_u; fdict = frozen_u
            else: i_v += 1; i_t = i_v; fdict = frozen_v
            if i_t in fdict: continue
            if gamma == 0:
                pred = u_hat[i_t].round().int().cpu().numpy()
                true_val = U_msg[start:end, i_t-1]
            else:
                pred = v_hat[i_t].round().int().cpu().numpy()
                true_val = V_msg[start:end, i_t-1]
            nn_errors[idx] += (pred != true_val).sum()
            nn_count[idx] += bs
            idx += 1
        if (start // BATCH) % 10 == 0:
            print(f"  NN {end}/{N_CW}", flush=True)
    nn_time = time.time() - t0
    nn_per_step = nn_errors / nn_count
    print(f"  NN done in {nn_time:.1f}s")

    # ── SC baseline per-step error rates ──────────────────────────────────
    # SC is slow, only use 1000 codewords
    N_SC = 1000
    sc_errors = np.zeros(n_info, dtype=np.int64)
    sc_count = np.zeros(n_info, dtype=np.int64)

    print(f"Running SC (free-running, {N_SC} cw)...", flush=True)
    t0 = time.time()
    for i in range(N_SC):
        z_list = Z[i].tolist()
        u_dec, v_dec = decode_single(N, z_list, b, frozen_u, frozen_v,
                                     channel, log_domain=False)
        u_dec = np.asarray(u_dec, dtype=np.int32)
        v_dec = np.asarray(v_dec, dtype=np.int32)
        idx = 0
        i_u, i_v = 0, 0
        for step in range(2*N):
            gamma = b[step]
            if gamma == 0: i_u += 1; i_t = i_u; fdict = frozen_u
            else: i_v += 1; i_t = i_v; fdict = frozen_v
            if i_t in fdict: continue
            if gamma == 0:
                if u_dec[i_t-1] != U_msg[i, i_t-1]:
                    sc_errors[idx] += 1
            else:
                if v_dec[i_t-1] != V_msg[i, i_t-1]:
                    sc_errors[idx] += 1
            sc_count[idx] += 1
            idx += 1
        if (i+1) % 200 == 0:
            print(f"  SC {i+1}/{N_SC}", flush=True)
    sc_time = time.time() - t0
    sc_per_step = sc_errors / sc_count
    print(f"  SC done in {sc_time:.1f}s")

    # ── Analysis ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("N=256 Inference-time per-step error rates")
    print("=" * 70)

    # Group by quintiles of step index
    n_groups = 5
    group_size = n_info // n_groups
    print(f"\nError rates by step quintile (early → late in tree walk):")
    print(f"{'quintile':<10}{'steps':<12}{'NN err rate':<14}{'SC err rate':<14}{'ratio':<8}")
    for g in range(n_groups):
        s = g * group_size
        e = (g+1) * group_size if g < n_groups-1 else n_info
        nn_rate = nn_per_step[s:e].mean()
        sc_rate = sc_per_step[s:e].mean()
        ratio = nn_rate / max(sc_rate, 1e-10)
        print(f"  Q{g+1:<7}{s}-{e:<10}{nn_rate:<14.6f}{sc_rate:<14.6f}{ratio:<8.2f}")

    # Top 20 positions with highest NN error rate
    sorted_idx = np.argsort(nn_per_step)[::-1]
    print(f"\nTop 20 NN error-rate positions:")
    print(f"{'step_idx':<10}{'step':<6}{'pos':<6}{'user':<6}"
          f"{'NN_err':>10}{'SC_err':>10}{'ratio':>8}{'pe':>8}")
    for i in sorted_idx[:20]:
        s = step_meta[i]
        user = 'U' if s['user']==0 else 'V'
        ratio = nn_per_step[i] / max(sc_per_step[i], 1e-10)
        print(f"{i:<10}{s['step']:<6}{s['pos']:<6}{user:<6}"
              f"{nn_per_step[i]:>10.4f}{sc_per_step[i]:>10.4f}"
              f"{ratio:>8.1f}{s['pe']:>8.4f}")

    # Overall
    print(f"\nOverall: NN mean err rate={nn_per_step.mean():.6f}, "
          f"SC mean err rate={sc_per_step.mean():.6f}, "
          f"ratio={nn_per_step.mean()/max(sc_per_step.mean(),1e-10):.2f}")

    # Correlation: does NN error rate grow with step index?
    step_indices = np.arange(n_info)
    from numpy import corrcoef
    corr_nn = corrcoef(step_indices, nn_per_step)[0,1]
    corr_sc = corrcoef(step_indices, sc_per_step)[0,1]
    print(f"Correlation(step_idx, NN_err): {corr_nn:.4f}")
    print(f"Correlation(step_idx, SC_err): {corr_sc:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    results = {
        'config': {'N': N, 'ku': ku, 'kv': kv, 'n_cw_nn': N_CW,
                   'n_cw_sc': N_SC, 'seed': SEED},
        'overall': {
            'nn_mean_err': float(nn_per_step.mean()),
            'sc_mean_err': float(sc_per_step.mean()),
            'ratio': float(nn_per_step.mean()/max(sc_per_step.mean(), 1e-10)),
            'corr_step_nn': float(corr_nn),
            'corr_step_sc': float(corr_sc),
        },
        'per_step': [
            {**step_meta[i],
             'nn_err_rate': float(nn_per_step[i]),
             'sc_err_rate': float(sc_per_step[i])}
            for i in range(n_info)
        ],
    }
    with open('results/n256_inference_errors.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/n256_inference_errors.json")


if __name__ == '__main__':
    main()
