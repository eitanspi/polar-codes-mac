#!/usr/bin/env python3
"""eval_all_n512_checkpoints.py — single-model BLER for all N=512 GMAC B checkpoints.
Analogous to eval_all_n256_checkpoints.py."""

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

N = 512; SIGMA2 = 10 ** (-6.0 / 10)
KU = int(0.48 * N); KV = int(0.48 * N)   # Class B symmetric
# Actually NCG_CHAPTER uses ku=kv=123 for N=256 ≈ 0.48. For N=512: 246/246.
KU, KV = 246, 246


def load_design():
    n = int(math.log2(N))
    d = np.load(os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz'))
    su = np.argsort(d['u_error_rates']); sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:KU]]); Av = sorted([int(i+1) for i in sv[:KV]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def try_load(name):
    p = os.path.join(BASE, 'saved_models', name)
    if not os.path.exists(p): return None
    try:
        sd = torch.load(p, map_location='cpu', weights_only=True)
    except Exception as e:
        return f'load_err: {e}'
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    try:
        missing, unexpected = model.load_state_dict(fixed, strict=False)
        return model
    except Exception as e:
        return f'state_err: {e}'


def eval_bler(model, channel, Au, Av, fu, fv, b, n_cw=500, batch=10):
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
    return errs / total


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)
    candidates = [
        'ncg_gmac_mlp_N512.pt',
        'campaign_n512_best.pt',
        'campaign_n512_latest.pt',
        'n512_long_best.pt',
        'n512_long_latest.pt',
    ]
    print(f"\n  All N={N} GMAC Class B checkpoints (500 cw each, ku={KU}, kv={KV})", flush=True)
    results = {}
    for name in candidates:
        m = try_load(name)
        if m is None:
            print(f"  [miss] {name}"); results[name] = 'not_found'; continue
        if isinstance(m, str):
            print(f"  [fail] {name}: {m[:60]}"); results[name] = m; continue
        t0 = time.time()
        try:
            bler = eval_bler(m, channel, Au, Av, fu, fv, b, 500, batch=10)
            dt = time.time() - t0
            print(f"  {name:35s}  BLER={bler:.4f}  [{dt:.0f}s]", flush=True)
            results[name] = {'bler': bler, 'n_cw': 500, 'time_s': round(dt, 1)}
        except Exception as e:
            print(f"  [run-fail] {name}: {e}"); results[name] = f'run_err: {e}'

    out = os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N512_all_checkpoints.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == '__main__':
    main()
