#!/usr/bin/env python3
"""
GMAC Class B rate-BLER sweep at 6 dB.

Tests whether the GMAC neural decoder beats analytical SC at ANY rate point
across multiple block lengths. Memory says NN matches SC at the published
(0.48, 0.48) operating point — but maybe a different rate has a clear win.

For each N in {64, 128, 256}, sweep ku=kv ∈ several values and run both
NN-SC and analytical SC. Save BLER + Wilson CIs.

Each point uses 5000 cw — enough to detect 2-3x differences with confidence.
"""

import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.decoder import decode_single
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac


SNR_DB = 6
SIGMA2 = 10 ** (-SNR_DB / 10)


def load_design(N, ku, kv, snr_db=SNR_DB):
    n = int(math.log2(N))
    dp = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr{snr_db}dB.npz')
    if not os.path.exists(dp):
        raise FileNotFoundError(f"No design file: {dp}")
    d = np.load(dp)
    if 'u_error_rates' in d:
        su = np.argsort(d['u_error_rates'])
        sv = np.argsort(d['v_error_rates'])
    else:
        su = d['sorted_u']
        sv = d['sorted_v']
    Au = sorted([int(i+1) for i in su[:ku]])
    Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_gmac_model(N):
    ckpt = os.path.join(BASE, 'saved_models', f'ncg_gmac_mlp_N{N}.pt')
    if not os.path.exists(ckpt):
        return None
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    fixed = {}
    for k, v in sd.items():
        nk = k.replace('z_enc.', 'z_encoder.') if k.startswith('z_enc.') else k
        fixed[nk] = v
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def eval_sc(N, channel, b, Au, Av, fu, fv, n_cw, rng):
    errs = 0
    for _ in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int)).tolist()
        u_dec, v_dec = decode_single(N, z, b, fu, fv, channel, log_domain=True)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or \
           any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
    return errs


def eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, n_cw, rng, batch_size=50):
    model.eval()
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf).astype(np.float32)
            zt = torch.from_numpy(zf).float()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    return errs


def main():
    channel = GaussianMAC(sigma2=SIGMA2)

    # Sweep design — symmetric ku=kv only (the trained regime)
    # Each point is (N, ku=kv list, n_cw)
    plan = {
        64:  ([16, 20, 24, 28, 32, 36, 40], 5000),
        128: ([32, 40, 50, 56, 62, 70, 80], 5000),
        256: ([64, 80, 100, 115, 128, 140], 3000),
    }

    results = {}
    out_path = os.path.join(BASE, 'results', 'bemac', '..', 'gmac_classB_rate_sweep.json')
    out_path = os.path.normpath(out_path)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    for N in [64, 128, 256]:
        ks, n_cw = plan[N]
        print(f"\n{'='*72}\n  N = {N}, GMAC SNR=6dB Class B, {n_cw} cw\n{'='*72}", flush=True)
        model = load_gmac_model(N)
        if model is None:
            print(f"  No model for N={N}, skipping")
            continue
        b = make_path(N, N // 2)
        results[str(N)] = []

        for k in ks:
            try:
                Au, Av, fu, fv = load_design(N, k, k)
            except Exception as e:
                print(f"  k={k}: design error: {e}", flush=True)
                continue

            sum_rate = 2 * k / N
            print(f"\n  ku=kv={k} (R_sum={sum_rate:.3f})", flush=True)

            # NN-SC
            t0 = time.perf_counter()
            rng = np.random.default_rng(42 + N + k)
            nn_errs = eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, n_cw, rng)
            nn_time = time.perf_counter() - t0
            nn_bler, nn_lo, nn_hi = wilson_ci(nn_errs, n_cw)
            print(f"    NN-SC:  {nn_errs}/{n_cw} = {nn_bler:.3e}  CI[{nn_lo:.2e},{nn_hi:.2e}]  {nn_time:.0f}s ({nn_time/n_cw*1000:.2f} ms/cw)", flush=True)

            # SC
            t0 = time.perf_counter()
            rng2 = np.random.default_rng(42 + N + k)
            sc_errs = eval_sc(N, channel, b, Au, Av, fu, fv, n_cw, rng2)
            sc_time = time.perf_counter() - t0
            sc_bler, sc_lo, sc_hi = wilson_ci(sc_errs, n_cw)
            print(f"    SC:     {sc_errs}/{n_cw} = {sc_bler:.3e}  CI[{sc_lo:.2e},{sc_hi:.2e}]  {sc_time:.0f}s ({sc_time/n_cw*1000:.2f} ms/cw)", flush=True)

            ratio = nn_bler / sc_bler if sc_bler > 0 else None
            print(f"    RATIO:  {ratio:.3f}x" if ratio is not None else f"    RATIO:  —", flush=True)

            results[str(N)].append({
                'N': N, 'ku': k, 'kv': k, 'sum_rate': sum_rate,
                'nn_errors': nn_errs, 'nn_cw': n_cw,
                'nn_bler': nn_bler, 'nn_ci_lo': nn_lo, 'nn_ci_hi': nn_hi,
                'nn_ms_per_cw': nn_time / n_cw * 1000,
                'sc_errors': sc_errs, 'sc_cw': n_cw,
                'sc_bler': sc_bler, 'sc_ci_lo': sc_lo, 'sc_ci_hi': sc_hi,
                'sc_ms_per_cw': sc_time / n_cw * 1000,
                'ratio_nn_over_sc': ratio,
            })

            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
