#!/usr/bin/env python3
"""
eval_bemac_comprehensive.py — Comprehensive BEMAC evaluation for paper.

Evaluates SC, SCL(L=4), NN-SC, NN-SCL(L=4) at N=32,64,128,256,512,1024
for BOTH Class B and Class C paths.

BEMAC: Z = X + Y in {0,1,2}. Neural decoder uses nn.Embedding(3, d=16).
"""

import os
import sys
import json
import time
import math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'polar'))

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.decoder import decode_single
from polar.decoder_scl import decode_single_list
from polar.channels import BEMAC
from polar.design import design_bemac, make_path
from polar.design_mc import design_from_file

from neural.ncg_pure_neural import PureNeuralCompGraphDecoder
from neural.neural_scl import NeuralSCLDecoder


# ─── Rate points ────────────────────────────────────────────────────────────

# Class C: path 0^N 1^N (all U first, then all V)
# Uses analytical Bhattacharyya design. Ru~0.50, Rv=1.0 (all V bits info)
BEMAC_CLASS_C = {
    32:  {'ku': 16, 'kv': 32},
    64:  {'ku': 32, 'kv': 64},
    128: {'ku': 64, 'kv': 128},
    256: {'ku': 128, 'kv': 256},
    512: {'ku': 256, 'kv': 512},
    1024: {'ku': 512, 'kv': 1024},
}

# Class B: interleaved path 0^{N/2} 1^N 0^{N/2}
# Uses MC design. Ru~0.50, Rv~0.70
BEMAC_CLASS_B = {
    32:  {'ku': 16, 'kv': 22},
    64:  {'ku': 32, 'kv': 45},
    128: {'ku': 64, 'kv': 90},
    256: {'ku': 128, 'kv': 179},
    512: {'ku': 256, 'kv': 358},
    1024: {'ku': 512, 'kv': 716},
}

N_CW_SC = 5000
N_CW_SCL = 2000
N_CW_NNSCL = 1000


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_design_class_c(N, ku, kv):
    """Analytical Bhattacharyya design for Class C."""
    n = int(math.log2(N))
    Au, Av, fu, fv, _, _ = design_bemac(n, ku, kv)
    b = make_path(N, path_i=N)  # 0^N 1^N
    return Au, Av, fu, fv, b


def get_design_class_b(N, ku, kv):
    """MC design for Class B (interleaved path)."""
    n = int(math.log2(N))
    design_file = os.path.join(BASE, 'designs', f'bemac_B_n{n}.npz')
    if not os.path.exists(design_file):
        print(f"  WARNING: MC design {design_file} not found, skipping")
        return None, None, None, None, None
    Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, ku, kv)
    b = make_path(N, path_i=N//2)  # 0^{N/2} 1^N 0^{N/2}
    return Au, Av, fu, fv, b


def load_bemac_nn_model(ckpt_name, device='cpu'):
    """Load PureNeuralCompGraphDecoder (BEMAC, discrete embedding)."""
    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=3)
    ckpt_path = os.path.join(BASE, 'saved_models', ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: Checkpoint {ckpt_path} not found")
        return None
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


# ─── Eval functions ─────────────────────────────────────────────────────────

def eval_sc(N, channel, b, Au, Av, fu, fv, n_cw, rng):
    errs = 0
    for i in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int)).tolist()
        u_dec, v_dec = decode_single(N, z, b, fu, fv, channel, log_domain=False)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or \
           any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
        if (i+1) % 1000 == 0:
            print(f"    SC: {i+1}/{n_cw}  BLER={errs/(i+1):.4f}", flush=True)
    return errs / n_cw


def eval_scl(N, channel, b, Au, Av, fu, fv, n_cw, L, rng):
    errs = 0
    for i in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int)).tolist()
        u_dec, v_dec = decode_single_list(N, z, b, fu, fv, channel,
                                           log_domain=False, L=L)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or \
           any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
        if (i+1) % 500 == 0:
            print(f"    SCL(L={L}): {i+1}/{n_cw}  BLER={errs/(i+1):.4f}", flush=True)
    return errs / n_cw


def eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, n_cw, rng, batch_size=25):
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
            # BEMAC: Z = X + Y, discrete output
            zf = channel.sample_batch(xf, yf)
            zt = torch.from_numpy(zf).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e:
                    errs += 1
            total += actual
            if total % 1000 == 0:
                print(f"    NN-SC: {total}/{n_cw}  BLER={errs/total:.4f}", flush=True)
    return errs / n_cw


def eval_nn_scl_bemac(model, N, channel, b, Au, Av, fu, fv, n_cw, L, rng):
    """NN-SCL for BEMAC — manual list decoding with discrete embedding."""
    # For BEMAC, we can't use the GMAC NeuralSCLDecoder directly since
    # it assumes a z_encoder. Instead, implement a simple list decoder
    # that forks at each non-frozen leaf.
    # For now, this is a simplified version that tests a few candidate
    # paths. Full SCL for BEMAC will be implemented separately if needed.
    print(f"    NN-SCL(L={L}) for BEMAC: not yet implemented, skipping")
    return None


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    channel = BEMAC()
    all_results = {}

    print(f"\n{'='*80}")
    print(f"  BEMAC Comprehensive Evaluation for Paper")
    print(f"  SC: {N_CW_SC} cw, SCL: {N_CW_SCL} cw, NN-SCL: {N_CW_NNSCL} cw")
    print(f"{'='*80}")

    for code_class, class_name, rates in [
        ('C', 'Class C', BEMAC_CLASS_C),
        ('B', 'Class B', BEMAC_CLASS_B),
    ]:
        class_results = {}
        print(f"\n\n{'#'*80}")
        print(f"  {class_name}: {'0^N 1^N' if code_class == 'C' else '0^{N/2} 1^N 0^{N/2}'}")
        print(f"{'#'*80}")

        for N in [32, 64, 128, 256, 512, 1024]:
            n = int(math.log2(N))
            cfg = rates[N]
            ku, kv = cfg['ku'], cfg['kv']

            print(f"\n{'─'*80}")
            print(f"  N={N}, ku={ku}, kv={kv}, Ru={ku/N:.3f}, Rv={kv/N:.3f}")
            print(f"{'─'*80}")

            # Get design
            if code_class == 'C':
                Au, Av, fu, fv, b = get_design_class_c(N, ku, kv)
            else:
                Au, Av, fu, fv, b = get_design_class_b(N, ku, kv)
                if b is None:
                    print(f"  Skipping N={N} (no MC design)")
                    continue

            res = {'N': N, 'ku': ku, 'kv': kv, 'class': class_name,
                   'Ru': ku/N, 'Rv': kv/N}

            # 1. Analytical SC
            print(f"  [1/4] SC ({N_CW_SC} cw)...", end='', flush=True)
            rng = np.random.default_rng(42)
            t0 = time.time()
            bler_sc = eval_sc(N, channel, b, Au, Av, fu, fv, N_CW_SC, rng)
            t_sc = time.time() - t0
            print(f" BLER={bler_sc:.4f}  [{t_sc:.1f}s]")
            res['SC'] = {'bler': bler_sc, 'time_s': round(t_sc, 1)}

            # 2. SCL(L=4) — skip for large N to save time
            if N <= 512:
                n_cw_scl = min(N_CW_SCL, 500 if N >= 512 else N_CW_SCL)
                print(f"  [2/4] SCL(L=4) ({n_cw_scl} cw)...", end='', flush=True)
                rng = np.random.default_rng(42)
                t0 = time.time()
                bler_scl = eval_scl(N, channel, b, Au, Av, fu, fv, n_cw_scl, 4, rng)
                t_scl = time.time() - t0
                print(f" BLER={bler_scl:.4f}  [{t_scl:.1f}s]")
                res['SCL_L4'] = {'bler': bler_scl, 'time_s': round(t_scl, 1)}
            else:
                print(f"  [2/4] SCL(L=4) — skipped for N={N} (too slow)")

            # 3. NN-SC
            ckpt = f'ncg_pure_neural_N{N}.pt'
            model = load_bemac_nn_model(ckpt)
            if model is not None:
                print(f"  [3/4] NN-SC ({N_CW_SC} cw)...", end='', flush=True)
                rng = np.random.default_rng(42)
                t0 = time.time()
                bs = max(2, min(50, 400 // (N // 16)))
                bler_nnsc = eval_nn_sc(model, N, channel, b, Au, Av, fu, fv,
                                        N_CW_SC, rng, bs)
                t_nnsc = time.time() - t0
                print(f" BLER={bler_nnsc:.4f}  [{t_nnsc:.1f}s]")
                res['NN_SC'] = {'bler': bler_nnsc, 'time_s': round(t_nnsc, 1)}
            else:
                print(f"  [3/4] NN-SC — no checkpoint for N={N}")

            # 4. NN-SCL(L=4) — skip for BEMAC (not primary contribution)
            print(f"  [4/4] NN-SCL(L=4) — skipped for BEMAC")

            class_results[str(N)] = res

            # Print summary for this N
            print(f"\n  Summary for N={N}:")
            sc_bler = res.get('SC', {}).get('bler', None)
            for dec_name in ['SC', 'SCL_L4', 'NN_SC', 'NN_SCL_L4']:
                if dec_name in res:
                    b_val = res[dec_name]['bler']
                    ratio = f"{b_val/sc_bler:.2f}x" if sc_bler and sc_bler > 0 else "N/A"
                    print(f"    {dec_name:<15s}  BLER={b_val:.5f}  (vs SC: {ratio})")

        all_results[class_name] = class_results

    # ─── Save results ───────────────────────────────────────────────────────
    out_path = os.path.join(BASE, 'results', 'bemac', 'bemac_comprehensive_paper.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to: {out_path}")

    # ─── Print final comparison tables ──────────────────────────────────────
    for class_name, class_results in all_results.items():
        print(f"\n{'='*80}")
        print(f"  FINAL TABLE — BEMAC {class_name}")
        print(f"{'='*80}")
        print(f"  {'N':>5s}  {'Ru':>5s}  {'Rv':>5s}  {'SC':>8s}  {'SCL(4)':>8s}  {'NN-SC':>8s}  {'NN-SCL(4)':>10s}")
        print(f"  {'-'*60}")
        for N in [32, 64, 128, 256, 512, 1024]:
            if str(N) not in class_results:
                continue
            r = class_results[str(N)]
            sc = r.get('SC', {}).get('bler', '-')
            scl = r.get('SCL_L4', {}).get('bler', '-')
            nnsc = r.get('NN_SC', {}).get('bler', '-')
            nnscl = r.get('NN_SCL_L4', {}).get('bler', '-')
            sc_s = f"{sc:.5f}" if isinstance(sc, float) else sc
            scl_s = f"{scl:.5f}" if isinstance(scl, float) else scl
            nnsc_s = f"{nnsc:.5f}" if isinstance(nnsc, float) else nnsc
            nnscl_s = f"{nnscl:.5f}" if isinstance(nnscl, float) else nnscl
            print(f"  {N:>5d}  {r['Ru']:>5.3f}  {r['Rv']:>5.3f}  {sc_s:>8s}  {scl_s:>8s}  {nnsc_s:>8s}  {nnscl_s:>10s}")


if __name__ == '__main__':
    main()
