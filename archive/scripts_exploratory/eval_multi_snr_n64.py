#!/usr/bin/env python3
"""
eval_multi_snr_n64.py — Multi-SNR waterfall for N=64 GMAC B.
Complements eval_multi_snr_n128.py. Same methodology: fixed 6-dB design,
vary sigma2 for SNR ∈ {4, 5, 6, 7, 8} dB.
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
N = 64
KU = KV = 31
N_CW_SC = 3000
N_CW_NN = 2000


def load_design():
    n = int(math.log2(N))
    d = np.load(os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz'))
    su = np.argsort(d['u_error_rates']); sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:KU]]); Av = sorted([int(i+1) for i in sv[:KV]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_model():
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(os.path.join(BASE, 'saved_models', f'ncg_gmac_mlp_N{N}.pt'),
                    map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def eval_sc(channel, Au, Av, fu, fv, b, n_cw, seed=42):
    rng = np.random.default_rng(seed)
    errs = 0
    for i in range(n_cw):
        uf = np.zeros(N, dtype=int); vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist()); y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int)).tolist()
        u_dec, v_dec = decode_single(N, z, b, fu, fv, channel, log_domain=True)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or \
           any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
    return errs / n_cw


def eval_ncg(model, channel, Au, Av, fu, fv, b, n_cw, seed=42, batch_size=40):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf).astype(np.float32)).float()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    return errs / total


def main():
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)
    model = load_model()

    print(f"\n{'='*78}")
    print(f"  Multi-SNR waterfall — GMAC B, N={N}, ku={KU}, kv={KV}")
    print(f"{'='*78}")

    results = {'N': N, 'ku': KU, 'kv': KV,
               'design': f'gmac_B_n{int(math.log2(N))}_snr6dB.npz',
               'snr_results': {}}

    for snr_db in SNR_VALUES:
        sigma2 = 10 ** (-snr_db / 10)
        channel = GaussianMAC(sigma2=sigma2)
        t0 = time.time()
        bler_sc = eval_sc(channel, Au, Av, fu, fv, b, N_CW_SC)
        t_sc = time.time() - t0
        t0 = time.time()
        bler_nn = eval_ncg(model, channel, Au, Av, fu, fv, b, N_CW_NN)
        t_nn = time.time() - t0
        ratio = bler_nn / max(bler_sc, 1e-9)
        print(f"  SNR={snr_db}dB  SC={bler_sc:.4f}[{t_sc:.0f}s]  NCG={bler_nn:.4f}[{t_nn:.0f}s]  ratio={ratio:.2f}x")
        results['snr_results'][str(snr_db)] = {
            'snr_db': snr_db, 'sigma2': sigma2,
            'sc_bler': bler_sc, 'ncg_bler': bler_nn,
            'ratio': round(ratio, 3),
            'n_cw_sc': N_CW_SC, 'n_cw_nn': N_CW_NN,
        }
        out = os.path.join(BASE, 'results', 'multi_snr', f'gmac_N{N}_waterfall.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == '__main__':
    main()
