#!/usr/bin/env python3
"""
eval_nn_scl_full.py — Full comparison: NN-SC vs NN-SCL(L=4) vs SC vs SCL(L=4)
at N=32, 64, 128, 256 using best available checkpoints.

Class B, SNR=6dB, GaussianMAC.
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neural'))
# neural_scl.py imports bare 'encoder', 'channels', 'design' from the legacy layout
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polar'))

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.decoder import decode_single
from polar.decoder_scl import decode_single_list
from polar.channels import GaussianMAC
from polar.design import make_path

from neural.ncg_pure_neural import PureNeuralCompGraphDecoder
from neural.neural_scl import SimpleMLP_Gmac, NeuralSCLDecoder


# ─── Config ────────────────────────────────────────────────────────────────────

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

CONFIGS = {
    32:  {'ku': 15, 'kv': 15, 'design': 'gmac_B_n5_snr6dB.npz',
           'ckpt': 'ncg_gmac_mlp_N32.pt'},
    64:  {'ku': 31, 'kv': 31, 'design': 'gmac_B_n6_snr6dB.npz',
           'ckpt': 'ncg_gmac_mlp_N64.pt'},
    128: {'ku': 62, 'kv': 62, 'design': 'gmac_B_n7_snr6dB.npz',
           'ckpt': 'ncg_gmac_mlp_N128.pt'},
    256: {'ku': 123, 'kv': 123, 'design': 'gmac_B_n8_snr6dB.npz',
           'ckpt': 'campaign_n256_sched_best.pt'},
}

N_CW_SC = 2000       # codewords for SC / NN-SC
N_CW_SCL = 1000      # codewords for SCL / NN-SCL (slower)

BASE = os.path.dirname(os.path.abspath(__file__))


# ─── Helpers ────────────────────────────────────────────────────────────────────

def load_design(N, ku, kv, design_file):
    """Load GMAC design from npz."""
    dp = os.path.join(BASE, 'designs', design_file)
    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:ku]])
    Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_nn_model(N, ckpt_name, device='cpu'):
    """Load SimpleMLP_Gmac from checkpoint."""
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    ckpt_path = os.path.join(BASE, 'saved_models', ckpt_name)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Handle key mismatch
    fixed = {}
    for k, v in sd.items():
        nk = k.replace('z_enc.', 'z_encoder.') if k.startswith('z_enc.') else k
        fixed[nk] = v
    model.load_state_dict(fixed, strict=False)
    model.to(device)
    model.eval()
    return model




# ─── Evaluation functions ───────────────────────────────────────────────────────

def eval_analytical_sc(N, channel, b, Au, Av, fu, fv, n_cw, rng):
    """Evaluate analytical SC decoder."""
    errs = 0
    for i in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(
            np.array(x, dtype=int).reshape(1, N),
            np.array(y, dtype=int).reshape(1, N)
        )[0].tolist()
        u_dec, v_dec = decode_single(N, z, b, fu, fv, channel, log_domain=True)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or \
           any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
    return errs / n_cw


def eval_analytical_scl(N, channel, b, Au, Av, fu, fv, n_cw, L, rng):
    """Evaluate analytical SCL decoder."""
    errs = 0
    for i in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(
            np.array(x, dtype=int).reshape(1, N),
            np.array(y, dtype=int).reshape(1, N)
        )[0].tolist()
        u_dec, v_dec = decode_single_list(N, z, b, fu, fv, channel,
                                           log_domain=True, L=L)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or \
           any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
    return errs / n_cw


def eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, n_cw, rng, batch_size=25):
    """Evaluate NN-SC (greedy, L=1)."""
    model.eval()
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            zt = torch.from_numpy(zf).float()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e:
                    errs += 1
            total += actual
    return errs / n_cw


def eval_nn_scl(model, N, channel, b, Au, Av, fu, fv, n_cw, L, rng):
    """Evaluate NN-SCL decoder."""
    decoder = NeuralSCLDecoder(model, L=L)
    errs = 0
    with torch.no_grad():
        for i in range(n_cw):
            uf = np.zeros(N, dtype=int)
            vf = np.zeros(N, dtype=int)
            for p in Au: uf[p-1] = rng.integers(0, 2)
            for p in Av: vf[p-1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf.reshape(1, N))
            yf = polar_encode_batch(vf.reshape(1, N))
            zf = channel.sample_batch(xf, yf)
            z_t = torch.from_numpy(zf[0]).float()
            uh, vh = decoder.decode(z_t, b, fu, fv)
            if any(uh.get(p, 0) != uf[p-1] for p in Au) or \
               any(vh.get(p, 0) != vf[p-1] for p in Av):
                errs += 1
    return errs / n_cw


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    results = {}

    print(f"\n{'='*78}")
    print(f"  Neural SCL Full Comparison — GMAC Class B, SNR={SNR_DB}dB")
    print(f"  SC/NN-SC: {N_CW_SC} codewords, SCL/NN-SCL: {N_CW_SCL} codewords")
    print(f"{'='*78}")

    for N in [32, 64, 128, 256]:
        cfg = CONFIGS[N]
        ku, kv = cfg['ku'], cfg['kv']
        n = int(math.log2(N))
        path_i = N // 2
        b = make_path(N, path_i)

        Au, Av, fu, fv = load_design(N, ku, kv, cfg['design'])

        print(f"\n{'─'*78}")
        print(f"  N={N}, n={n}, ku={ku}, kv={kv}, R_u={ku/N:.3f}, R_v={kv/N:.3f}")
        print(f"  Design: {cfg['design']}, Checkpoint: {cfg['ckpt']}")
        print(f"{'─'*78}")

        res = {'N': N, 'ku': ku, 'kv': kv, 'snr_dB': SNR_DB}

        # --- 1. Analytical SC ---
        print(f"  [1/4] Analytical SC ({N_CW_SC} cw) ...", end='', flush=True)
        rng_sc = np.random.default_rng(42)
        t0 = time.time()
        bler_sc = eval_analytical_sc(N, channel, b, Au, Av, fu, fv, N_CW_SC, rng_sc)
        t_sc = time.time() - t0
        print(f" BLER={bler_sc:.4f}  [{t_sc:.1f}s]")
        res['SC'] = {'bler': bler_sc, 'n_cw': N_CW_SC, 'time_s': round(t_sc, 1)}

        # --- 2. Analytical SCL L=4 ---
        print(f"  [2/4] Analytical SCL L=4 ({N_CW_SCL} cw) ...", end='', flush=True)
        rng_scl = np.random.default_rng(42)
        t0 = time.time()
        bler_scl = eval_analytical_scl(N, channel, b, Au, Av, fu, fv, N_CW_SCL, 4, rng_scl)
        t_scl = time.time() - t0
        print(f" BLER={bler_scl:.4f}  [{t_scl:.1f}s]")
        res['SCL_L4'] = {'bler': bler_scl, 'n_cw': N_CW_SCL, 'time_s': round(t_scl, 1)}

        # --- 3. NN-SC ---
        print(f"  [3/4] NN-SC ({N_CW_SC} cw) ...", end='', flush=True)
        model = load_nn_model(N, cfg['ckpt'])
        rng_nn = np.random.default_rng(42)
        t0 = time.time()
        bs = max(2, min(50, 200 // (N // 16)))
        bler_nnsc = eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, N_CW_SC, rng_nn, bs)
        t_nnsc = time.time() - t0
        print(f" BLER={bler_nnsc:.4f}  [{t_nnsc:.1f}s]")
        res['NN_SC'] = {'bler': bler_nnsc, 'n_cw': N_CW_SC, 'time_s': round(t_nnsc, 1)}

        # --- 4. NN-SCL L=4 ---
        print(f"  [4/4] NN-SCL L=4 ({N_CW_SCL} cw) ...", end='', flush=True)
        rng_nnscl = np.random.default_rng(42)
        t0 = time.time()
        bler_nnscl = eval_nn_scl(model, N, channel, b, Au, Av, fu, fv, N_CW_SCL, 4, rng_nnscl)
        t_nnscl = time.time() - t0
        print(f" BLER={bler_nnscl:.4f}  [{t_nnscl:.1f}s]")
        res['NN_SCL_L4'] = {'bler': bler_nnscl, 'n_cw': N_CW_SCL, 'time_s': round(t_nnscl, 1)}

        results[str(N)] = res

        # Summary for this N
        print(f"\n  {'Decoder':<18s}  {'BLER':>8s}  {'vs SC':>8s}  {'vs SCL':>8s}")
        print(f"  {'-'*50}")
        print(f"  {'SC':<18s}  {bler_sc:>8.4f}  {'1.00x':>8s}  {bler_sc/max(bler_scl,1e-9):>7.2f}x")
        print(f"  {'SCL (L=4)':<18s}  {bler_scl:>8.4f}  {bler_scl/max(bler_sc,1e-9):>7.2f}x  {'1.00x':>8s}")
        print(f"  {'NN-SC':<18s}  {bler_nnsc:>8.4f}  {bler_nnsc/max(bler_sc,1e-9):>7.2f}x  {bler_nnsc/max(bler_scl,1e-9):>7.2f}x")
        print(f"  {'NN-SCL (L=4)':<18s}  {bler_nnscl:>8.4f}  {bler_nnscl/max(bler_sc,1e-9):>7.2f}x  {bler_nnscl/max(bler_scl,1e-9):>7.2f}x")

    # ─── Final comparison table ─────────────────────────────────────────────
    print(f"\n\n{'='*78}")
    print(f"  FINAL COMPARISON TABLE — GMAC Class B, SNR=6dB")
    print(f"{'='*78}")
    print(f"  {'N':>5s}  {'ku':>4s}  {'SC':>8s}  {'SCL(4)':>8s}  {'NN-SC':>8s}  {'NN-SCL(4)':>10s}  {'NN-SCL/SCL':>11s}")
    print(f"  {'-'*62}")
    for N in [32, 64, 128, 256]:
        r = results[str(N)]
        sc = r['SC']['bler']
        scl = r['SCL_L4']['bler']
        nnsc = r['NN_SC']['bler']
        nnscl = r['NN_SCL_L4']['bler']
        ratio = f"{nnscl/scl:.2f}x" if scl > 0 else "N/A"
        print(f"  {N:>5d}  {r['ku']:>4d}  {sc:>8.4f}  {scl:>8.4f}  {nnsc:>8.4f}  {nnscl:>10.4f}  {ratio:>11s}")

    # ─── Save results ───────────────────────────────────────────────────────
    out_dir = os.path.join(BASE, 'results', 'gmac_snr6dB')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'nn_scl_full_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*78}\n")


if __name__ == '__main__':
    main()
