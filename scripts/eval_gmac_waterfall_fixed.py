#!/usr/bin/env python3
"""
eval_gmac_waterfall_fixed.py — GMAC waterfall with FIXED frozen set.

The frozen set is designed at 6 dB and held constant across all SNR values.
Only the channel sigma2 varies, giving a proper waterfall curve.
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

# ── Configuration ──────────────────────────────────────────────────────────────

DESIGN_SNR = 6  # frozen set designed at this SNR
SNR_VALUES = [3, 4, 5, 6, 7, 8]
N_CW = 3000

CONFIGS = {
    64:  {'ku': 31, 'kv': 31, 'n': 6},
    128: {'ku': 62, 'kv': 62, 'n': 7},
}


def load_design_fixed(N, ku, kv):
    """Load frozen set from the 6 dB design file (used for ALL SNR points)."""
    n = int(math.log2(N))
    dp = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr{DESIGN_SNR}dB.npz')
    if not os.path.exists(dp):
        raise FileNotFoundError(f"Design file not found: {dp}")
    d = np.load(dp)
    if 'u_error_rates' in d:
        su = np.argsort(d['u_error_rates'])
        sv = np.argsort(d['v_error_rates'])
    else:
        su = d['sorted_u']
        sv = d['sorted_v']
    Au = sorted([int(i + 1) for i in su[:ku]])
    Av = sorted([int(i + 1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_gmac_model(N, device='cpu'):
    """Load GMAC neural decoder (SimpleMLP_Gmac)."""
    ckpt = os.path.join(BASE, 'saved_models', f'ncg_gmac_mlp_N{N}.pt')
    if not os.path.exists(ckpt):
        print(f"  WARNING: Checkpoint {ckpt} not found")
        return None
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    fixed = {}
    for k, v in sd.items():
        nk = k.replace('z_enc.', 'z_encoder.') if k.startswith('z_enc.') else k
        fixed[nk] = v
    model.load_state_dict(fixed, strict=False)
    model.to(device)
    model.eval()
    return model


def eval_sc(N, channel, b, Au, Av, fu, fv, n_cw, rng):
    errs = 0
    for i in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p - 1] = rng.integers(0, 2)
        for p in Av: vf[p - 1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int)).tolist()
        u_dec, v_dec = decode_single(N, z, b, fu, fv, channel, log_domain=True)
        if any(u_dec[p - 1] != uf[p - 1] for p in Au) or \
           any(v_dec[p - 1] != vf[p - 1] for p in Av):
            errs += 1
    return errs / n_cw


def eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, n_cw, rng, batch_size=25):
    model.eval()
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p - 1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf).astype(np.float32)
            zt = torch.from_numpy(zf).float()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p - 1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p - 1] for p in Av if p in vh)
                if e:
                    errs += 1
            total += actual
    return errs / n_cw


def main():
    results = {}

    print(f"\n{'=' * 80}")
    print(f"  GMAC Waterfall — Fixed Code (designed at {DESIGN_SNR} dB)")
    print(f"  {N_CW} codewords per (N, SNR) point")
    print(f"{'=' * 80}")

    for N in [64, 128]:
        cfg = CONFIGS[N]
        ku, kv = cfg['ku'], cfg['kv']
        b = make_path(N, N // 2)

        # Load frozen set ONCE — designed at 6 dB
        Au, Av, fu, fv = load_design_fixed(N, ku, kv)
        print(f"\n  N={N}: frozen set from {DESIGN_SNR} dB design "
              f"(|Au|={len(Au)}, |Av|={len(Av)})")

        model = load_gmac_model(N)
        if model is None:
            print(f"  Skipping N={N} — no model found")
            continue

        N_results = {}

        print(f"\n{'─' * 80}")
        print(f"  N={N}, ku={ku}, kv={kv}, design SNR={DESIGN_SNR} dB")
        print(f"{'─' * 80}")
        print(f"  {'SNR':>5s}  {'SC BLER':>10s}  {'NN-SC BLER':>12s}  {'Ratio':>8s}")
        print(f"  {'-' * 50}")

        for snr_db in SNR_VALUES:
            # Only sigma2 changes — frozen set stays fixed
            sigma2 = 10 ** (-snr_db / 10)
            channel = GaussianMAC(sigma2=sigma2)

            # SC
            rng = np.random.default_rng(42)
            t0 = time.time()
            bler_sc = eval_sc(N, channel, b, Au, Av, fu, fv, N_CW, rng)
            t_sc = time.time() - t0

            # NN-SC
            rng = np.random.default_rng(42)
            t0 = time.time()
            bs = max(2, min(50, 200 // max(1, N // 32)))
            bler_nn = eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, N_CW, rng, bs)
            t_nn = time.time() - t0

            ratio = bler_nn / max(bler_sc, 1e-9)
            print(f"  {snr_db:>4d}dB  {bler_sc:>10.5f}  {bler_nn:>12.5f}  {ratio:>7.2f}x"
                  f"  [SC:{t_sc:.0f}s, NN:{t_nn:.0f}s]")

            N_results[str(snr_db)] = {
                'snr_dB': snr_db,
                'sigma2': sigma2,
                'design_snr_dB': DESIGN_SNR,
                'sc_bler': bler_sc,
                'nn_sc_bler': bler_nn,
                'ratio': round(ratio, 3),
                'n_cw': N_CW,
            }

        results[str(N)] = {
            'N': N, 'ku': ku, 'kv': kv,
            'design_snr_dB': DESIGN_SNR,
            'snr_results': N_results,
        }

    # Save
    out_dir = os.path.join(BASE, 'results', 'gmac_snr6dB')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'gmac_waterfall_fixed_code.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY — Fixed code (designed at {DESIGN_SNR} dB)")
    print(f"{'=' * 80}")
    print(f"  {'SNR':>5s}", end='')
    for N in [64, 128]:
        print(f"  {'N=' + str(N) + ' SC':>10s}  {'N=' + str(N) + ' NN':>10s}  {'ratio':>6s}",
              end='')
    print()
    for snr_db in SNR_VALUES:
        print(f"  {snr_db:>4d}dB", end='')
        for N in [64, 128]:
            if str(N) in results:
                snr_r = results[str(N)]['snr_results'].get(str(snr_db), {})
                sc = snr_r.get('sc_bler', '-')
                nn = snr_r.get('nn_sc_bler', '-')
                r = snr_r.get('ratio', '-')
                print(f"  {sc:>10}  {nn:>10}  {r:>6}", end='')
        print()


if __name__ == '__main__':
    main()
