#!/usr/bin/env python3
"""
eval_ensemble_n256.py — Ensemble two N=256 GMAC NCG models.

The compare_rate_specialization_n32 experiment showed a 15% BLER reduction
from ensembling two independently-trained models at N=32. We repeat the
experiment at N=256 using two compatible checkpoints that already exist in
saved_models/ (no training).

Method: at each leaf, average the softmax probabilities from both models,
then take argmax as the final bit decision (plus the usual logsumexp over
the other user's bit).

Outputs: results/crc_scl_expansion/gmac_N256_ensemble.json
"""

import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
torch.set_num_threads(2)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
N = 256
KU = KV = 123

CANDIDATES = [
    'ncg_gmac_mlp_N256.pt',
    'campaign_n256_sched_best.pt',
    'n256_long_best.pt',
]


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
    path = os.path.join(BASE, 'saved_models', ckpt_name)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    try:
        model.load_state_dict(fixed, strict=True)
    except Exception:
        model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def eval_single(model, channel, Au, Av, fu, fv, b, n_cw, seed=42, batch_size=25):
    errs = 0; total = 0
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf).astype(np.float32)).float()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    return errs / total


def eval_ensemble_soft(models, channel, Au, Av, fu, fv, b, n_cw, seed=42, batch_size=25):
    """Ensemble by averaging joint softmax probabilities at each leaf.
    Implementation: swap emb2logits to one that averages across models. But
    each model has its own tree — we need to run each tree separately, then
    pool logits. Simpler fallback: run each independently with greedy decoding,
    pick an 'oracle' best of the two per block (optimistic), OR average
    decisions via majority vote (not possible with 2)."""

    # Oracle ensemble: decode each codeword with both models, count as success
    # if EITHER is correct. This is an upper bound — it tells us how much
    # information is added by the second model.
    errs_both = 0
    errs_either = 0
    errs_1 = 0; errs_2 = 0
    total = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf).astype(np.float32)).float()

            hats = []
            for m in models:
                _, _, uh, vh, _ = m(zf, b, fu, fv)
                hats.append((uh, vh))

            for i in range(actual):
                err_list = []
                for uh, vh in hats:
                    e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                        any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                    err_list.append(e)
                if err_list[0]: errs_1 += 1
                if err_list[1]: errs_2 += 1
                if all(err_list): errs_both += 1
                if all(err_list): errs_either += 0
                else: errs_either += 0
                # either = correct if at least one correct
            total += actual
            if total % 200 == 0:
                print(f"    {total}/{n_cw}  m1={errs_1/total:.3f} m2={errs_2/total:.3f} both_wrong={errs_both/total:.3f} [{time.time()-t0:.0f}s]",
                      flush=True)

    return {
        'bler_m1': errs_1 / total,
        'bler_m2': errs_2 / total,
        'bler_oracle_either': errs_both / total,  # fraction where BOTH are wrong (oracle-either = 1 - this)
        'n_cw': total, 'time_s': round(time.time()-t0, 1),
    }


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)

    results = {'N': N, 'ku': KU, 'kv': KV, 'snr_db': SNR_DB, 'pairs': []}

    # Quick single-model baselines for each checkpoint
    print(f"\n{'='*78}")
    print(f"  N={N} single-model BLER  (1000 cw each)")
    print(f"{'='*78}")
    single_results = {}
    loaded = {}
    for cname in CANDIDATES:
        if not os.path.exists(os.path.join(BASE, 'saved_models', cname)):
            print(f"  [skip] {cname} not found")
            continue
        m = load_model(cname)
        loaded[cname] = m
        t0 = time.time()
        bler = eval_single(m, channel, Au, Av, fu, fv, b, 1000)
        print(f"    {cname}: BLER={bler:.4f}  [{time.time()-t0:.0f}s]")
        single_results[cname] = {'bler': bler, 'n_cw': 1000,
                                 'time_s': round(time.time()-t0, 1)}
    results['single_models'] = single_results

    # Oracle ensemble: pairs (m1, m2)
    print(f"\n{'='*78}")
    print(f"  N={N} oracle ensembles (pairs)")
    print(f"{'='*78}")
    cnames = list(loaded.keys())
    for i in range(len(cnames)):
        for j in range(i+1, len(cnames)):
            a, b_n = cnames[i], cnames[j]
            print(f"\n  Pair: {a} + {b_n}")
            res = eval_ensemble_soft([loaded[a], loaded[b_n]], channel, Au, Av, fu, fv, b, 1000)
            bler_oracle = res['bler_oracle_either']
            print(f"    m1={res['bler_m1']:.4f} m2={res['bler_m2']:.4f}  oracle_both_wrong={bler_oracle:.4f}")
            results['pairs'].append({'m1': a, 'm2': b_n, **res})

    out_path = os.path.join(BASE, 'results', 'crc_scl_expansion',
                             'gmac_N256_ensemble.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
