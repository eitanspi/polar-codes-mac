#!/usr/bin/env python3
"""
eval_crc_scl_campaign_hicw.py — High-cw confirmation of campaign_sched SCL.

First pass (300 cw) found BLER=0.0033 for SCL L=4 using campaign_n256_sched_best.pt,
beating SC (0.006). Confidence interval at 300 cw is wide; this script runs
3000 cw for L=4 and 2000 cw for L=8 to tighten the estimates.
"""

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
from eval_crc_scl_unified import UnifiedSCL, compute_crc8, CRC_BITS

N = 256; KU = KV = 123; SIGMA2 = 10 ** (-6.0 / 10)
CKPT = 'campaign_n256_sched_best.pt'


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
    sd = torch.load(os.path.join(BASE, 'saved_models', CKPT), map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def run_config(model, Au, Av, fu, fv, b, L, n_cw, seed=42):
    channel = GaussianMAC(sigma2=SIGMA2)
    decoder = UnifiedSCL(model, 'gmac', L=L)
    crc_positions = Au[-CRC_BITS:] if len(Au) > CRC_BITS else []
    msg_positions = [p for p in Au if p not in crc_positions]

    errs_scl = 0; errs_crc = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_cw):
            uf = np.zeros(N, dtype=int); vf = np.zeros(N, dtype=int)
            for p in Au: uf[p-1] = rng.integers(0, 2)
            for p in Av: vf[p-1] = rng.integers(0, 2)
            if crc_positions:
                msg_bits = [uf[p-1] for p in msg_positions]
                crc_vals = compute_crc8(msg_bits)
                for cp, cv in zip(crc_positions, crc_vals):
                    uf[cp-1] = cv
            xf = polar_encode_batch(uf.reshape(1, N))
            yf = polar_encode_batch(vf.reshape(1, N))
            z_np = channel.sample_batch(xf, yf)
            z_t = torch.from_numpy(z_np.astype(np.float32))[0].float()
            paths = decoder.decode_list(z_t, b, fu, fv)

            best = paths[0]
            if any(best['uh'].get(p, 0) != uf[p-1] for p in Au) or \
               any(best['vh'].get(p, 0) != vf[p-1] for p in Av):
                errs_scl += 1

            picked = None
            for cand in paths:
                uh = cand['uh']
                msg_bits = [uh.get(p, 0) for p in msg_positions]
                crc_dec = [uh.get(p, 0) for p in crc_positions]
                if compute_crc8(msg_bits) == crc_dec:
                    picked = cand; break
            if picked is None:
                picked = paths[0]
            if any(picked['uh'].get(p, 0) != uf[p-1] for p in Au) or \
               any(picked['vh'].get(p, 0) != vf[p-1] for p in Av):
                errs_crc += 1

            if (i+1) % 200 == 0 or i+1 == n_cw:
                print(f"    {i+1}/{n_cw}  SCL={errs_scl/(i+1):.4f}  CRC-SCL={errs_crc/(i+1):.4f}  ({(time.time()-t0)/(i+1):.2f}s/cw)",
                      flush=True)
    return {
        'bler_scl': errs_scl / n_cw,
        'bler_crc_scl': errs_crc / n_cw,
        'n_cw': n_cw, 'time_s': round(time.time()-t0, 1),
    }


def main():
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)
    model = load_model()

    print(f"\n{'='*78}")
    print(f"  High-cw CRC-SCL confirmation at N=256 using {CKPT}")
    print(f"{'='*78}")

    results = {'N': N, 'ku': KU, 'kv': KV, 'ckpt': CKPT}

    # L=4 with 3000 cw for tight CI
    print(f"\n  L = 4  (3000 cw, ~90 min at 1.7s/cw)")
    res = run_config(model, Au, Av, fu, fv, b, 4, 3000)
    results['L4_hicw'] = res
    print(f"    L=4 final: SCL={res['bler_scl']:.4f}  CRC-SCL={res['bler_crc_scl']:.4f}  [{res['time_s']:.0f}s]")

    out = os.path.join(BASE, 'results', 'crc_scl_expansion',
                        'gmac_N256_campaign_crc_scl_hicw.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
