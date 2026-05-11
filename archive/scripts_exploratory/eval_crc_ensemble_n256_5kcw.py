#!/usr/bin/env python3
"""High-cw (5000) confirmation of the 3-model CRC ensemble at N=256.
Verifies the 3000-cw 0.0117 result with tighter confidence."""

import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'scripts'))
torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac
from eval_crc_scl_unified import compute_crc8, CRC_BITS

N = 256; KU = KV = 123
SIGMA2 = 10 ** (-6.0 / 10)
MODELS = [
    'campaign_n256_sched_best.pt',
    'n256_long_best.pt',
    'ncg_gmac_mlp_N256.pt',
]
FALLBACK = 'campaign_n256_sched_best.pt'


def load_design():
    n = int(math.log2(N))
    d = np.load(os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz'))
    su = np.argsort(d['u_error_rates']); sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:KU]]); Av = sorted([int(i+1) for i in sv[:KV]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_model(name):
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(os.path.join(BASE, 'saved_models', name), map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    try:
        model.load_state_dict(fixed, strict=True)
    except Exception:
        model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)
    crc_positions = Au[-CRC_BITS:]
    msg_positions = [p for p in Au if p not in crc_positions]

    print(f"\n  5000-cw confirmation of 3-model CRC ensemble at N={N}", flush=True)
    models = {n: load_model(n) for n in MODELS}
    model_names = list(models.keys())

    n_cw = 5000; batch = 25
    rng = np.random.default_rng(42)
    errs_per_model = {n: 0 for n in model_names}
    errs_ensemble = 0; errs_oracle_all_wrong = 0; n_no_crc_pass = 0; total = 0
    t0 = time.time()

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            for i in range(actual):
                msg_bits = [uf[i, p-1] for p in msg_positions]
                crc_vals = compute_crc8(msg_bits)
                for cp, cv in zip(crc_positions, crc_vals):
                    uf[i, cp-1] = cv
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf).astype(np.float32)).float()
            per_model = {}
            for name in model_names:
                _, _, uh, vh, _ = models[name](zf, b, fu, fv)
                per_model[name] = (uh, vh)

            for i in range(actual):
                errs = {}; crc_ok = {}
                for name in model_names:
                    uh, vh = per_model[name]
                    e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                        any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                    errs[name] = e; errs_per_model[name] += int(e)
                    uh_vec = {p: int(uh[p][i].item()) for p in Au if p in uh}
                    msg_bits = [uh_vec.get(p, 0) for p in msg_positions]
                    crc_dec = [uh_vec.get(p, 0) for p in crc_positions]
                    crc_ok[name] = (compute_crc8(msg_bits) == crc_dec)
                picked_err = None
                for name in model_names:
                    if crc_ok[name]:
                        picked_err = errs[name]; break
                if picked_err is None:
                    picked_err = errs[FALLBACK]; n_no_crc_pass += 1
                if picked_err: errs_ensemble += 1
                if all(errs.values()): errs_oracle_all_wrong += 1

            total += actual
            if total % 500 == 0:
                msg = f"  {total}/{n_cw}  "
                for name in model_names:
                    short = name.replace('.pt','').replace('_best','').replace('ncg_gmac_mlp_','main').replace('campaign_n256_sched','camp').replace('n256_long','long')[:10]
                    msg += f"{short}={errs_per_model[name]/total:.4f} "
                msg += f"ens={errs_ensemble/total:.4f}  oracle={errs_oracle_all_wrong/total:.4f}"
                print(msg, flush=True)

    print(f"\n  Final ({n_cw} cw, {time.time()-t0:.0f}s):")
    for name in model_names:
        print(f"    {name:40s}  BLER={errs_per_model[name]/n_cw:.4f}")
    print(f"    {'CRC-aided 3-ensemble':40s}  BLER={errs_ensemble/n_cw:.4f}")
    print(f"    {'Oracle all-wrong':40s}  BLER={errs_oracle_all_wrong/n_cw:.4f}")

    out = os.path.join(BASE, 'results', 'crc_scl_expansion', 'gmac_N256_crc_ensemble_5kcw.json')
    with open(out, 'w') as f:
        json.dump({
            'N': N, 'ku': KU, 'kv': KV, 'n_cw': n_cw,
            'models': model_names,
            'per_model_bler': {n: errs_per_model[n]/n_cw for n in model_names},
            'ensemble_bler': errs_ensemble / n_cw,
            'oracle_bler': errs_oracle_all_wrong / n_cw,
            'no_crc_pass_fraction': n_no_crc_pass / n_cw,
            'time_s': round(time.time()-t0, 1),
        }, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
