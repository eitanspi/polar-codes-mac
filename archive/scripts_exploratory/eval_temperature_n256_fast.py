#!/usr/bin/env python3
"""
eval_temperature_n256_fast.py — Temperature scaling diagnostic at N=256 with
BATCHED inference so we finish within the ~30 min budget.

We monkey-patch the tree's emb2logits with a scaled version (logits /= T) and
call the model's normal forward (which does greedy NCG-SC with batching).
This gives per-codeword timings around 0.03-0.05s with batch 20-50.

Outputs: results/crc_scl_expansion/gmac_N256_temperature.json
"""

import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
BATCH = 25
N_CW_PER_T = 1000


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


class TemperatureEmb2Logits(nn.Module):
    """Wrapper that divides logits by T before returning."""
    def __init__(self, inner, T):
        super().__init__()
        self.inner = inner
        self.T = T

    def forward(self, x):
        return self.inner(x) / self.T


def evaluate_at_T(model, channel, Au, Av, fu, fv, b, T, n_cw, batch_size=25):
    # Install temperature-aware emb2logits
    original = model.tree.emb2logits
    model.tree.emb2logits = TemperatureEmb2Logits(original, T)
    errs = 0
    total = 0
    rng = np.random.default_rng(42)
    t0 = time.time()
    try:
        with torch.no_grad():
            while total < n_cw:
                actual = min(batch_size, n_cw - total)
                uf = np.zeros((actual, N), dtype=int)
                vf = np.zeros((actual, N), dtype=int)
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
                if total % 100 == 0:
                    print(f"    T={T}: {total}/{n_cw}  BLER={errs/total:.4f}  "
                          f"[{time.time()-t0:.0f}s, {(time.time()-t0)/total:.2f}s/cw]",
                          flush=True)
    finally:
        model.tree.emb2logits = original
    t_total = time.time() - t0
    return errs / total, t_total


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)
    model = load_model()

    print(f"\n{'='*78}")
    print(f"  Temperature Scaling — GMAC B N={N}, {N_CW_PER_T} cw per T")
    print(f"  SC baseline ≈ 0.006, NCG baseline (T=1) ≈ 0.023")
    print(f"{'='*78}")

    results = {'N': N, 'ku': KU, 'kv': KV, 'snr_db': SNR_DB,
               'n_cw_per_T': N_CW_PER_T, 'batch_size': BATCH}

    # Scan a range of Ts, including fractional ones for sharper posteriors
    T_values = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]
    for T in T_values:
        print(f"\n  T = {T}")
        bler, t = evaluate_at_T(model, channel, Au, Av, fu, fv, b, T,
                                 N_CW_PER_T, batch_size=BATCH)
        results[f'ncg_sc_T{T}'] = {'bler': bler, 'n_cw': N_CW_PER_T,
                                    'time_s': round(t, 1)}
        print(f"    T={T}: BLER={bler:.4f}  [{t:.0f}s]")

    out_path = os.path.join(BASE, 'results', 'crc_scl_expansion',
                             'gmac_N256_temperature.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
