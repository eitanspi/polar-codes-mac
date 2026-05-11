#!/usr/bin/env python3
"""Evaluate GMAC N=1024 main checkpoint + launch BEMAC N=1024 CRC-SCL."""

import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac

N = 1024; SIGMA2 = 10 ** (-6.0 / 10)
KU = KV = int(0.48 * N)  # 491


def load_design():
    n = int(math.log2(N))
    d = np.load(os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz'))
    su = np.argsort(d['u_error_rates']); sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:KU]]); Av = sorted([int(i+1) for i in sv[:KV]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def eval_bler(model, channel, Au, Av, fu, fv, b, n_cw=200, batch=5):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(42)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
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
            if total % 50 == 0:
                print(f"  {total}/{n_cw}  BLER={errs/total:.4f}", flush=True)
    return errs / total


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)
    print(f"\n  GMAC N={N} main checkpoint eval (200 cw, ku={KU}, kv={KV})", flush=True)

    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(os.path.join(BASE, 'saved_models', 'ncg_gmac_mlp_N1024.pt'),
                    map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()

    t0 = time.time()
    bler = eval_bler(model, channel, Au, Av, fu, fv, b, 200, batch=5)
    dt = time.time() - t0
    print(f"\n  ncg_gmac_mlp_N1024.pt  BLER = {bler:.4f}  [{dt:.0f}s]")

    out = os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N1024_main.json')
    with open(out, 'w') as f:
        json.dump({'N': N, 'ku': KU, 'kv': KV,
                    'ncg_gmac_mlp_N1024': {'bler': bler, 'n_cw': 200, 'time_s': round(dt, 1)}}, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
