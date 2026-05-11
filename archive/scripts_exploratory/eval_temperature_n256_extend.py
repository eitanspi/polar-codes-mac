#!/usr/bin/env python3
"""
eval_temperature_n256_extend.py — Extended scan of larger T values at N=256.

The first scan found T=3.0 as the best (BLER=0.013, 32% below T=1.0 baseline).
This script extends to T ∈ {2.5, 3.0, 4.0, 5.0, 7.0, 10.0}, with a larger cw
budget for the best Ts.
"""

import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
N = 256
KU = KV = 123


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


class TWrap(nn.Module):
    def __init__(self, inner, T):
        super().__init__(); self.inner = inner; self.T = T
    def forward(self, x): return self.inner(x) / self.T


def run_eval(model, channel, Au, Av, fu, fv, b, T, n_cw, batch_size=25, seed=42):
    orig = model.tree.emb2logits
    model.tree.emb2logits = TWrap(orig, T)
    errs = 0; total = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    try:
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
    finally:
        model.tree.emb2logits = orig
    return errs / total, time.time() - t0


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)
    model = load_model()

    # Load prior results to merge
    out_path = os.path.join(BASE, 'results', 'crc_scl_expansion',
                             'gmac_N256_temperature.json')
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    results['snr_db'] = SNR_DB; results['N'] = N
    results['ku'] = KU; results['kv'] = KV

    print(f"\n{'='*78}")
    print(f"  Extended temperature scan at N=256")
    print(f"{'='*78}")

    # Scan broader range with 5000 cw each for the ones of interest
    T_scan = [2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    for T in T_scan:
        print(f"\n  T = {T}  (5000 cw)")
        bler, tt = run_eval(model, channel, Au, Av, fu, fv, b, T, 5000,
                             batch_size=25)
        print(f"    BLER = {bler:.4f}  [{tt:.0f}s]")
        results[f'ncg_sc_T{T}_v2'] = {
            'bler': bler, 'n_cw': 5000, 'time_s': round(tt, 1),
            'batch_size': 25
        }
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Also a high-confidence baseline at T=1.0
    print(f"\n  T = 1.0 (high-confidence baseline, 5000 cw)")
    bler, tt = run_eval(model, channel, Au, Av, fu, fv, b, 1.0, 5000, batch_size=25)
    print(f"    T=1.0 (5000 cw): BLER = {bler:.4f}  [{tt:.0f}s]")
    results['ncg_sc_T1.0_v2'] = {
        'bler': bler, 'n_cw': 5000, 'time_s': round(tt, 1), 'batch_size': 25
    }
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
