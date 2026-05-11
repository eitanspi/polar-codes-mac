#!/usr/bin/env python3
"""
eval_multi_snr_n128.py — Clean multi-SNR waterfall for N=128 NCG.

Fixed code design: use the existing N=128 GMAC Class B design (trained at 6 dB).
Only the channel sigma2 varies across SNR = {4, 5, 6, 7, 8} dB. This tests how
well the NCG trained at 6 dB generalizes to neighboring SNRs without retraining.

Also optionally runs BEMAC (which is channel-SNR-independent but has a control
degraded/perfect erasure variant — we skip BEMAC SNR sweep since BEMAC is
deterministic).

Outputs:
  results/multi_snr/gmac_N128_waterfall.json
"""

import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
torch.set_num_threads(2)

from polar.encoder import polar_encode, polar_encode_batch
from polar.decoder import decode_single
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac

SNR_VALUES = [4, 5, 6, 7, 8]
N_CW_SC = 2000
N_CW_NN = 1000


def load_gmac_design_fixed_6db(N, ku, kv):
    """Always load the 6 dB design — the code is fixed."""
    n = int(math.log2(N))
    dp = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz')
    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:ku]])
    Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_gmac_model(N):
    ckpt = os.path.join(BASE, 'saved_models', f'ncg_gmac_mlp_N{N}.pt')
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def eval_sc(N, channel, b, Au, Av, fu, fv, n_cw, seed=42):
    rng = np.random.default_rng(seed)
    errs = 0
    for i in range(n_cw):
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
    return errs / n_cw


def eval_ncg(model, N, channel, b, Au, Av, fu, fv, n_cw, seed=42, batch_size=20):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(seed)
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
    return errs / total


def main():
    N = 128
    ku, kv = 62, 62
    Au, Av, fu, fv = load_gmac_design_fixed_6db(N, ku, kv)
    b = make_path(N, N // 2)
    model = load_gmac_model(N)

    print(f"\n{'='*78}")
    print(f"  Multi-SNR waterfall — GMAC Class B, N={N}, ku={ku}, kv={kv}")
    print(f"  Fixed 6-dB design; channel sigma2 varies with SNR")
    print(f"{'='*78}")

    results = {'N': N, 'ku': ku, 'kv': kv,
               'design': f'gmac_B_n{int(math.log2(N))}_snr6dB.npz',
               'snr_results': {}}

    for snr_db in SNR_VALUES:
        sigma2 = 10 ** (-snr_db / 10)
        channel = GaussianMAC(sigma2=sigma2)
        print(f"\n  SNR = {snr_db} dB  (sigma2 = {sigma2:.4f})")

        t0 = time.time()
        bler_sc = eval_sc(N, channel, b, Au, Av, fu, fv, N_CW_SC)
        t_sc = time.time() - t0
        t0 = time.time()
        bler_nn = eval_ncg(model, N, channel, b, Au, Av, fu, fv, N_CW_NN)
        t_nn = time.time() - t0

        ratio = bler_nn / max(bler_sc, 1e-9)
        print(f"    SC   : {bler_sc:.5f}   [{t_sc:.0f}s, {N_CW_SC} cw]")
        print(f"    NCG  : {bler_nn:.5f}   [{t_nn:.0f}s, {N_CW_NN} cw]")
        print(f"    ratio: {ratio:.2f}x")
        results['snr_results'][str(snr_db)] = {
            'snr_db': snr_db, 'sigma2': sigma2,
            'sc_bler': bler_sc, 'ncg_bler': bler_nn,
            'ratio': round(ratio, 3),
            'n_cw_sc': N_CW_SC, 'n_cw_nn': N_CW_NN,
        }

        # Save after each SNR
        out_path = os.path.join(BASE, 'results', 'multi_snr',
                                 f'gmac_N{N}_waterfall.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
