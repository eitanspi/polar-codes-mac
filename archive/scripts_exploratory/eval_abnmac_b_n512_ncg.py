#!/usr/bin/env python3
"""Eval ABNMAC B N=512 NCG best checkpoint with 3K CW for reliability."""
import os, sys, math, time, json
import numpy as np
import torch
torch.set_num_threads(4)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder


def encode_z(zf):
    out = np.empty(zf.shape, dtype=np.int64)
    for idx in np.ndindex(zf.shape):
        zx, zy = zf[idx]; out[idx] = 2*int(zx) + int(zy)
    return out


def main():
    N = 512; ku = 205; kv = 205
    n = int(math.log2(N))
    channel = ABNMAC()
    design_file = os.path.join(BASE, 'designs', f'abnmac_B_n{n}.npz')
    Au, Av, fu, fv, _, _, path_i = design_from_file(design_file, n, ku, kv)
    Au = sorted(Au); Av = sorted(Av)
    b = make_path(N, path_i)

    ckpt = os.path.join(BASE, 'saved_models', 'ncg_abnmac_classB_N512_best.pt')
    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=4)
    sd = torch.load(ckpt, weights_only=True, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.eval()

    n_cw = 3000
    rng = np.random.default_rng(42)
    errs = 0; total = 0; t0 = time.time()
    batch_sz = 5
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_sz, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            zt = torch.from_numpy(encode_z(zf)).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
            if total % 500 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  [{total}/{n_cw}] errs={errs} BLER={errs/total:.4f} ({elapsed:.1f}min)", flush=True)

    elapsed = (time.time() - t0) / 60
    bler = errs / total
    result = {'N': N, 'ku': ku, 'kv': kv, 'bler': bler, 'errs': errs, 'n_cw': total, 'time_min': round(elapsed, 1)}
    print(f"  BLER={bler:.4f} ({errs}/{total}) [{elapsed:.1f}min]", flush=True)

    out = os.path.join(BASE, 'results', 'abnmac_b_n512_ncg_eval.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out}", flush=True)


if __name__ == '__main__':
    main()
