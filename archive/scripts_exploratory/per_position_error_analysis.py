#!/usr/bin/env python3
"""
per_position_error_analysis.py — Analyze per-position error rates for NN vs SC.

Runs NN-SC and SC at N=128 and N=256, collecting per-position error statistics.
Shows which positions the NN struggles with most.
"""

import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.decoder import decode_single
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
N_CW = 2000


def load_design(N, ku, kv):
    n = int(math.log2(N))
    dp = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz')
    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:ku]])
    Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_model(N):
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    ckpt = os.path.join(BASE, 'saved_models', f'ncg_gmac_mlp_N{N}.pt')
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    RATES = {128: (62, 62), 256: (123, 123)}

    results = {}

    for N in [128, 256]:
        ku, kv = RATES[N]
        Au, Av, fu, fv = load_design(N, ku, kv)
        b = make_path(N, N // 2)
        model = load_model(N)

        print(f"\n{'='*60}")
        print(f"  Per-Position Error Analysis — N={N}")
        print(f"  ku={ku}, kv={kv}, {N_CW} codewords")
        print(f"{'='*60}")

        # Track per-position errors
        nn_u_errors = np.zeros(N)
        nn_v_errors = np.zeros(N)
        sc_u_errors = np.zeros(N)
        sc_v_errors = np.zeros(N)
        nn_block_errors = 0
        sc_block_errors = 0

        rng = np.random.default_rng(42)

        for cw in range(N_CW):
            uf = np.zeros(N, dtype=int)
            vf = np.zeros(N, dtype=int)
            for p in Au: uf[p-1] = rng.integers(0, 2)
            for p in Av: vf[p-1] = rng.integers(0, 2)
            x = polar_encode(uf.tolist())
            y = polar_encode(vf.tolist())
            z = channel.sample_batch(
                np.array(x, dtype=int).reshape(1, N),
                np.array(y, dtype=int).reshape(1, N)
            )[0]

            # SC decode
            z_list = z.tolist()
            u_sc, v_sc = decode_single(N, z_list, b, fu, fv, channel, log_domain=True)
            sc_block_err = False
            for p in Au:
                if u_sc[p-1] != uf[p-1]:
                    sc_u_errors[p-1] += 1
                    sc_block_err = True
            for p in Av:
                if v_sc[p-1] != vf[p-1]:
                    sc_v_errors[p-1] += 1
                    sc_block_err = True
            if sc_block_err:
                sc_block_errors += 1

            # NN decode
            with torch.no_grad():
                zt = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
                _, _, uh, vh, _ = model(zt, b, fu, fv)
                nn_block_err = False
                for p in Au:
                    if p in uh and int(uh[p][0].item()) != uf[p-1]:
                        nn_u_errors[p-1] += 1
                        nn_block_err = True
                for p in Av:
                    if p in vh and int(vh[p][0].item()) != vf[p-1]:
                        nn_v_errors[p-1] += 1
                        nn_block_err = True
                if nn_block_err:
                    nn_block_errors += 1

            if (cw+1) % 500 == 0:
                print(f"  {cw+1}/{N_CW}  SC BLER={sc_block_errors/(cw+1):.4f}  "
                      f"NN BLER={nn_block_errors/(cw+1):.4f}", flush=True)

        # Analysis
        nn_u_ber = nn_u_errors / N_CW
        nn_v_ber = nn_v_errors / N_CW
        sc_u_ber = sc_u_errors / N_CW
        sc_v_ber = sc_v_errors / N_CW

        print(f"\n  Results N={N}:")
        print(f"    SC BLER: {sc_block_errors/N_CW:.4f}")
        print(f"    NN BLER: {nn_block_errors/N_CW:.4f}")
        print(f"    SC avg errors/cw: {(sc_u_errors.sum() + sc_v_errors.sum())/N_CW:.2f}")
        print(f"    NN avg errors/cw: {(nn_u_errors.sum() + nn_v_errors.sum())/N_CW:.2f}")

        # Top-10 worst positions for NN
        nn_ber_all = np.zeros(N)
        for p in Au: nn_ber_all[p-1] = nn_u_ber[p-1]
        for p in Av: nn_ber_all[p-1] = max(nn_ber_all[p-1], nn_v_ber[p-1])

        worst = np.argsort(nn_ber_all)[::-1][:10]
        print(f"\n    Top-10 worst NN positions:")
        for pos in worst:
            user = 'U' if (pos+1) in Au else 'V'
            sc_er = sc_u_ber[pos] if (pos+1) in Au else sc_v_ber[pos]
            nn_er = nn_u_ber[pos] if (pos+1) in Au else nn_v_ber[pos]
            print(f"      pos={pos+1:>4d} ({user})  NN BER={nn_er:.4f}  SC BER={sc_er:.4f}")

        results[str(N)] = {
            'sc_bler': sc_block_errors/N_CW,
            'nn_bler': nn_block_errors/N_CW,
            'sc_avg_errors': float((sc_u_errors.sum() + sc_v_errors.sum())/N_CW),
            'nn_avg_errors': float((nn_u_errors.sum() + nn_v_errors.sum())/N_CW),
            'nn_u_ber': nn_u_ber.tolist(),
            'nn_v_ber': nn_v_ber.tolist(),
            'sc_u_ber': sc_u_ber.tolist(),
            'sc_v_ber': sc_v_ber.tolist(),
        }

    out_path = os.path.join(BASE, 'results', 'gmac_snr6dB', 'per_position_error_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out_path}")


if __name__ == '__main__':
    main()
