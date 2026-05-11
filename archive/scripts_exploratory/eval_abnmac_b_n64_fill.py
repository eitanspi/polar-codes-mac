#!/usr/bin/env python3
"""ABNMAC B N=64 SC+NCG fill: both around 200/5K. Run 5K more each."""
import os, sys, math, time, json
import numpy as np
import torch
torch.set_num_threads(4)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_batch
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder


def encode_z(zf):
    out = np.empty(zf.shape, dtype=np.int64)
    for idx in np.ndindex(zf.shape):
        zx, zy = zf[idx]; out[idx] = 2*int(zx) + int(zy)
    return out


def main():
    N = 64; ku = 22; kv = 22
    n = int(math.log2(N))
    channel = ABNMAC()
    design_file = os.path.join(BASE, 'designs', f'abnmac_B_n{n}.npz')
    Au, Av, fu, fv, _, _, path_i = design_from_file(design_file, n, ku, kv)
    Au = sorted(Au); Av = sorted(Av)
    b = make_path(N, path_i)
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])

    results = {}
    n_cw = 5000

    # SC eval
    print("=== ABNMAC B N=64 SC ===", flush=True)
    rng = np.random.default_rng(600)
    errs = 0; total = 0; t0 = time.time()
    batch_sz = 10
    while total < n_cw:
        actual = min(batch_sz, n_cw - total)
        info_u = rng.integers(0, 2, size=(actual, len(Au)))
        info_v = rng.integers(0, 2, size=(actual, len(Av)))
        U = build_message_batch(N, info_u, Au)
        V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U)
        Y = polar_encode_batch(V)
        Z = channel.sample_batch(X, Y)
        res = decode_batch(N, Z.tolist(), b, fu, fv, channel, vectorized=True)
        for i, (u_dec, v_dec) in enumerate(res):
            u_dec_arr = np.array(u_dec); v_dec_arr = np.array(v_dec)
            if np.sum(u_dec_arr[u_info_idx] != info_u[i]) > 0 or \
               np.sum(v_dec_arr[v_info_idx] != info_v[i]) > 0:
                errs += 1
        total += actual
    bler_sc = errs / total
    results['SC'] = {'bler': bler_sc, 'errs': errs, 'n_cw': total}
    print(f"  SC BLER={bler_sc:.4f} ({errs}/{total})", flush=True)

    # NCG eval
    print("=== ABNMAC B N=64 NCG ===", flush=True)
    ckpt = os.path.join(BASE, 'saved_models', 'ncg_abnmac_classB_N64_best.pt')
    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=4)
    sd = torch.load(ckpt, weights_only=True, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.eval()

    rng = np.random.default_rng(601)
    errs = 0; total = 0; batch_sz = 25
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
    bler_ncg = errs / total
    results['NCG'] = {'bler': bler_ncg, 'errs': errs, 'n_cw': total}
    print(f"  NCG BLER={bler_ncg:.4f} ({errs}/{total})", flush=True)

    out = os.path.join(BASE, 'results', 'abnmac_b_n64_fill.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out}", flush=True)


if __name__ == '__main__':
    main()
