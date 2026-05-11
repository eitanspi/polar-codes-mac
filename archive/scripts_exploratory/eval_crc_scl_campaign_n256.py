#!/usr/bin/env python3
"""
eval_crc_scl_campaign_n256.py — CRC-SCL on the BETTER N=256 model
`campaign_n256_sched_best.pt` (BLER 0.014 vs main 0.025).

We already know that plain SCL at N=256 using the main ncg_gmac_mlp_N256.pt
gets worse with L (0.020 → 0.040 → 0.060). Test whether the better
campaign_n256_sched_best.pt shares the same miscalibration — or whether
SCL actually helps with it.

Outputs: results/crc_scl_expansion/gmac_N256_campaign_crc_scl.json
"""

import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'scripts'))
torch.set_num_threads(2)

from polar.channels import GaussianMAC
from polar.design import make_path
from polar.encoder import polar_encode_batch
from neural.neural_scl import SimpleMLP_Gmac

# Re-use the unified SCL decoder
from eval_crc_scl_unified import UnifiedSCL, compute_crc8, CRC_BITS

N = 256; KU = KV = 123
SNR_DB = 6.0; SIGMA2 = 10 ** (-SNR_DB / 10)


def load_design():
    n = int(math.log2(N))
    d = np.load(os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz'))
    su = np.argsort(d['u_error_rates']); sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:KU]]); Av = sorted([int(i+1) for i in sv[:KV]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_model(ckpt_name):
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(os.path.join(BASE, 'saved_models', ckpt_name),
                    map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    try:
        model.load_state_dict(fixed, strict=True)
    except Exception:
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
            uh_scl = best['uh']; vh_scl = best['vh']
            if any(uh_scl.get(p, 0) != uf[p-1] for p in Au) or \
               any(vh_scl.get(p, 0) != vf[p-1] for p in Av):
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
            uh_crc = picked['uh']; vh_crc = picked['vh']
            if any(uh_crc.get(p, 0) != uf[p-1] for p in Au) or \
               any(vh_crc.get(p, 0) != vf[p-1] for p in Av):
                errs_crc += 1

            if (i+1) % 50 == 0 or i+1 == n_cw:
                print(f"    {i+1}/{n_cw}  SCL={errs_scl/(i+1):.4f}  "
                      f"CRC-SCL={errs_crc/(i+1):.4f}  ({(time.time()-t0)/(i+1):.2f}s/cw)",
                      flush=True)
    return {
        'bler_scl': errs_scl / n_cw,
        'bler_crc_scl': errs_crc / n_cw,
        'n_cw': n_cw, 'time_s': round(time.time()-t0, 1),
    }


def main():
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)
    CKPT = 'campaign_n256_sched_best.pt'
    model = load_model(CKPT)

    print(f"\n{'='*78}")
    print(f"  CRC-SCL at N=256 using {CKPT}")
    print(f"{'='*78}")

    # First confirm baseline
    # (Already done by ensemble: BLER=0.014)

    out_path = os.path.join(BASE, 'results', 'crc_scl_expansion',
                             'gmac_N256_campaign_crc_scl.json')
    results = {'N': N, 'ku': KU, 'kv': KV, 'ckpt': CKPT,
               'snr_db': SNR_DB, 'single_baseline_bler': 0.014}

    # Budgets: smaller because each cw at N=256 L=16 is expensive
    configs = [(4, 300), (8, 200), (16, 100)]
    for L, n_cw in configs:
        print(f"\n  L = {L}  ({n_cw} cw)", flush=True)
        res = run_config(model, Au, Av, fu, fv, b, L, n_cw)
        results[f'L{L}'] = res
        print(f"    L={L}  SCL={res['bler_scl']:.4f}  CRC-SCL={res['bler_crc_scl']:.4f}")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
