#!/usr/bin/env python3
"""Reliable eval of ABNMAC Class B NCG at N=256 with 2000 CW."""
import os, sys, math, json, time
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
torch.set_num_threads(4)

from polar.encoder import polar_encode_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

D = 16; HIDDEN = 64; N_LAYERS = 2; N = 256; n = 8
ku = 102; kv = 102

def encode_z(zf):
    out = np.empty(zf.shape, dtype=np.int64)
    for idx in np.ndindex(zf.shape):
        zx, zy = zf[idx]; out[idx] = 2*int(zx) + int(zy)
    return out

def main():
    channel = ABNMAC()
    design_file = os.path.join(BASE, 'designs', f'abnmac_B_n{n}.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, ku, kv)
    b = make_path(N, N // 2)

    model = PureNeuralCompGraphDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS, vocab_size=4)
    ckpt = os.path.join(BASE, 'saved_models', 'ncg_abnmac_classB_N256_best.pt')
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # NCG eval
    n_cw = 2000
    rng = np.random.default_rng(777)
    errs = 0; total = 0
    t0 = time.time()
    with torch.no_grad():
        while total < n_cw:
            actual = min(20, n_cw - total)
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
            if total % 200 == 0:
                print(f'  [{total}/{n_cw}] errs={errs}', flush=True)
    elapsed = time.time() - t0
    bler = errs / n_cw
    print(f'\nNCG N=256: BLER={bler:.4f} ({errs}/{n_cw}) ({elapsed:.1f}s)')

    # SC eval
    from polar.encoder import build_message_batch
    from polar.decoder import decode_batch
    rng2 = np.random.default_rng(777)
    sc_errs = 0; total2 = 0
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    t1 = time.time()
    while total2 < n_cw:
        actual = min(20, n_cw - total2)
        info_u = rng2.integers(0, 2, size=(actual, ku))
        info_v = rng2.integers(0, 2, size=(actual, kv))
        U = build_message_batch(N, info_u, Au); V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U); Y = polar_encode_batch(V)
        Z = channel.sample_batch(X, Y)
        results = decode_batch(N, Z.tolist(), b, fu, fv, channel, vectorized=True)
        for i, (u_dec, v_dec) in enumerate(results):
            ue = int(np.sum(np.array(u_dec)[u_info_idx] != info_u[i]))
            ve = int(np.sum(np.array(v_dec)[v_info_idx] != info_v[i]))
            if ue > 0 or ve > 0: sc_errs += 1
        total2 += actual
    sc_elapsed = time.time() - t1
    sc_bler = sc_errs / n_cw
    print(f'SC  N=256: BLER={sc_bler:.4f} ({sc_errs}/{n_cw}) ({sc_elapsed:.1f}s)')
    print(f'Ratio: {bler/max(sc_bler,1e-6):.2f}x')

    result = {'N': N, 'ku': ku, 'kv': kv, 'ncg_bler': bler, 'ncg_errs': errs, 'ncg_cw': n_cw,
              'sc_bler': sc_bler, 'sc_errs': sc_errs, 'sc_cw': n_cw}
    out = os.path.join(BASE, 'results', 'abnmac_classB_ncg_N256_eval.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out}')

if __name__ == '__main__':
    main()
