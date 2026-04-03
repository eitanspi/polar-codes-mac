#!/usr/bin/env python3
"""
consolidate_bemac_results.py — Merge existing BEMAC results into one comprehensive file,
then run ONLY the missing evaluations (Class C, SCL(L=4) for larger N).
"""
import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.decoder import decode_single
from polar.decoder_scl import decode_single_list
from polar.channels import BEMAC
from polar.design import design_bemac, make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# ─── Existing results ──────────────────────────────────────────────────────

CLASS_B_EXISTING = {
    # From bemac_classB_Ru50_Rv70_nn_vs_sc
    'SC': {16: 0.0106, 32: 0.008, 64: 0.0056, 128: 0.002, 256: 8e-05, 512: 0.0, 1024: 0.0001},
    'NN_SC': {16: 0.0114, 32: 0.0088, 64: 0.003, 128: 0.0012, 256: 4e-05, 512: 0.0, 1024: 0.0001},
    'SCL_L32': {16: 0.008, 32: 0.0082, 64: 0.001, 128: 0.0006},
    'NN_SCL_L4': {32: 0.0073, 64: 0.0007, 128: 0.0007},
}

BEMAC_CLASS_B_RATES = {
    16:  {'ku': 8,   'kv': 11},
    32:  {'ku': 16,  'kv': 22},
    64:  {'ku': 32,  'kv': 45},
    128: {'ku': 64,  'kv': 90},
    256: {'ku': 128, 'kv': 179},
    512: {'ku': 256, 'kv': 358},
    1024:{'ku': 512, 'kv': 716},
}

BEMAC_CLASS_C_RATES = {
    32:  {'ku': 16,  'kv': 32},
    64:  {'ku': 32,  'kv': 64},
    128: {'ku': 64,  'kv': 128},
    256: {'ku': 128, 'kv': 256},
    512: {'ku': 256, 'kv': 512},
    1024:{'ku': 512, 'kv': 1024},
}

# ─── Eval helpers ──────────────────────────────────────────────────────────

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
                                           log_domain=True, L=L)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or \
           any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
        if (i+1) % 500 == 0:
            print(f"    SCL(L={L}): {i+1}/{n_cw}  BLER={errs/(i+1):.4f}", flush=True)
    return errs / n_cw

def eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, n_cw, rng, batch_size=25):
    model.eval()
    errs = 0; total = 0
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
            zt = torch.from_numpy(zf).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
            if total % 1000 == 0:
                print(f"    NN-SC: {total}/{n_cw}  BLER={errs/total:.4f}", flush=True)
    return errs / n_cw

def load_model(N):
    ckpt = os.path.join(BASE, 'saved_models', f'ncg_pure_neural_N{N}.pt')
    if not os.path.exists(ckpt):
        return None
    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=3)
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    channel = BEMAC()
    all_results = {'Class_B': {}, 'Class_C': {}}

    # ── Class B: consolidate existing + fill missing ──
    print("\n" + "="*80)
    print("  PHASE 1: Class B — consolidating existing results")
    print("="*80)

    for N in [16, 32, 64, 128, 256, 512, 1024]:
        rates = BEMAC_CLASS_B_RATES[N]
        ku, kv = rates['ku'], rates['kv']
        entry = {
            'N': N, 'ku': ku, 'kv': kv,
            'Ru': round(ku/N, 4), 'Rv': round(kv/N, 4),
            'SC': CLASS_B_EXISTING['SC'].get(N),
            'NN_SC': CLASS_B_EXISTING['NN_SC'].get(N),
            'SCL_L32': CLASS_B_EXISTING['SCL_L32'].get(N),
            'NN_SCL_L4': CLASS_B_EXISTING['NN_SCL_L4'].get(N),
        }

        # Run SCL(L=4) if missing and N <= 256
        if entry.get('SCL_L4') is None and N <= 256:
            n = int(math.log2(N))
            design_file = os.path.join(BASE, 'designs', f'bemac_B_n{n}.npz')
            if os.path.exists(design_file):
                Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, ku, kv)
                b = make_path(N, N//2)
                n_cw = 3000 if N <= 128 else 2000
                print(f"\n  Running SCL(L=4) for Class B N={N} ({n_cw} cw)...", flush=True)
                rng = np.random.default_rng(42)
                t0 = time.time()
                bler = eval_scl(N, channel, b, Au, Av, fu, fv, n_cw, 4, rng)
                print(f"  SCL(L=4) N={N}: BLER={bler:.5f}  [{time.time()-t0:.1f}s]")
                entry['SCL_L4'] = bler

        all_results['Class_B'][str(N)] = entry

    # ── Class C: run evaluations ──
    print("\n" + "="*80)
    print("  PHASE 2: Class C — running evaluations")
    print("="*80)

    for N in [32, 64, 128, 256, 512, 1024]:
        n = int(math.log2(N))
        rates = BEMAC_CLASS_C_RATES[N]
        ku, kv = rates['ku'], rates['kv']
        Au, Av, fu, fv, _, _ = design_bemac(n, ku, kv)
        b = make_path(N, N)  # 0^N 1^N

        entry = {'N': N, 'ku': ku, 'kv': kv,
                 'Ru': round(ku/N, 4), 'Rv': round(kv/N, 4)}

        # SC
        n_cw = 5000 if N <= 256 else 2000
        print(f"\n  SC Class C N={N} ({n_cw} cw)...", flush=True)
        rng = np.random.default_rng(42)
        t0 = time.time()
        entry['SC'] = eval_sc(N, channel, b, Au, Av, fu, fv, n_cw, rng)
        print(f"  SC N={N}: BLER={entry['SC']:.5f}  [{time.time()-t0:.1f}s]")

        # NN-SC
        model = load_model(N)
        if model is not None:
            print(f"  NN-SC Class C N={N} ({n_cw} cw)...", flush=True)
            rng = np.random.default_rng(42)
            t0 = time.time()
            bs = max(2, min(50, 400 // max(1, N // 16)))
            entry['NN_SC'] = eval_nn_sc(model, N, channel, b, Au, Av, fu, fv, n_cw, rng, bs)
            print(f"  NN-SC N={N}: BLER={entry['NN_SC']:.5f}  [{time.time()-t0:.1f}s]")

        # SCL(L=4) for N <= 128
        if N <= 128:
            n_cw_scl = 2000
            print(f"  SCL(L=4) Class C N={N} ({n_cw_scl} cw)...", flush=True)
            rng = np.random.default_rng(42)
            t0 = time.time()
            entry['SCL_L4'] = eval_scl(N, channel, b, Au, Av, fu, fv, n_cw_scl, 4, rng)
            print(f"  SCL(L=4) N={N}: BLER={entry['SCL_L4']:.5f}  [{time.time()-t0:.1f}s]")

        all_results['Class_C'][str(N)] = entry

        # Save intermediate results
        out_path = os.path.join(BASE, 'results', 'bemac', 'bemac_comprehensive_paper.json')
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ─── Print final tables ────────────────────────────────────────────────
    for cls_name, cls_data in all_results.items():
        print(f"\n{'='*80}")
        print(f"  BEMAC {cls_name}")
        print(f"{'='*80}")
        print(f"  {'N':>5s}  {'Ru':>5s}  {'Rv':>5s}  {'SC':>10s}  {'SCL(4)':>10s}  {'NN-SC':>10s}  {'NN/SC':>6s}")
        print(f"  {'-'*60}")
        for N_key in sorted(cls_data.keys(), key=int):
            r = cls_data[N_key]
            sc = r.get('SC')
            scl = r.get('SCL_L4')
            nnsc = r.get('NN_SC')
            ratio = f"{nnsc/sc:.2f}" if sc and nnsc and sc > 0 else "-"
            print(f"  {r['N']:>5d}  {r['Ru']:>5.3f}  {r['Rv']:>5.3f}  "
                  f"{sc if sc is not None else '-':>10}  "
                  f"{scl if scl is not None else '-':>10}  "
                  f"{nnsc if nnsc is not None else '-':>10}  "
                  f"{ratio:>6s}")

    print(f"\n  Results saved to: results/bemac/bemac_comprehensive_paper.json")

if __name__ == '__main__':
    main()
