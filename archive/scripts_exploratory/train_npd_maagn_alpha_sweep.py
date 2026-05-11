#!/usr/bin/env python3
"""
train_npd_maagn_alpha_sweep.py
==============================
Quick experiment: train chained NPD at N=16 with alpha=0.5 to demonstrate
the regime where neural memory-aware decoding has a larger advantage over
memoryless GMAC SC.

Usage:
    python scripts/train_npd_maagn_alpha_sweep.py
"""
from __future__ import annotations
import os
import sys
import json
import math
import time

import numpy as np
import torch

torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels_memory_new import MAAGNMAC
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_single as gmac_decode_single
from neural.npd_memory_mac import ChainedNPD_MAC

# Config
N = 16
n = int(math.log2(N))
ku, kv = 4, 7
SNR_DB = 6.0
RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_maagn_mac')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_design_():
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au); Av = sorted(Av)
    fu_1idx = {p: 0 for p in range(1, N + 1) if p not in Au}
    fv_1idx = {p: 0 for p in range(1, N + 1) if p not in Av}
    fu_set = {p - 1 for p in fu_1idx.keys()}
    fv_set = {p - 1 for p in fv_1idx.keys()}
    return Au, Av, fu_1idx, fv_1idx, fu_set, fv_set


def make_batch(channel, Au, Av, batch, rng):
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au: u_msg[:, p - 1] = rng.integers(0, 2, batch)
    for p in Av: v_msg[:, p - 1] = rng.integers(0, 2, batch)
    x = polar_encode_batch(u_msg.astype(int))
    y = polar_encode_batch(v_msg.astype(int))
    z = channel.sample_batch(x.astype(int), y.astype(int))
    return u_msg, v_msg, z.astype(np.float32), x, y


def train_and_eval_alpha(alpha=0.5, iters_s1=20000, iters_s2=20000,
                         batch=32, lr=1e-3, d=16, hidden=64, n_cw_eval=2000):
    sigma2 = 10.0 ** (-SNR_DB / 10.0)
    channel = MAAGNMAC(sigma2=sigma2, alpha=alpha)
    gmac = GaussianMAC(sigma2=sigma2)
    Au, Av, fu_1idx, fv_1idx, fu_set, fv_set = load_design_()
    b = make_path(N, N)
    br = bit_reversal_perm(n)
    br_t = torch.tensor(br.copy(), dtype=torch.long)

    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=d, hidden=hidden, n_layers=2,
                           encoder_type='bigru', gru_layers=1)
    print(f'\nalpha={alpha} sigma2={sigma2:.4f} params={model.count_parameters():,}')

    # Train Stage 1
    opt = torch.optim.AdamW(model.stage1.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters_s1, eta_min=lr * 0.1)
    rng = np.random.default_rng(42)
    t0 = time.time()
    model.stage1.train()
    best_bler_s1 = 1.0
    for it in range(1, iters_s1 + 1):
        u_msg, v_msg, z, x, y = make_batch(channel, Au, Av, batch, rng)
        z_t = torch.from_numpy(z)
        emb = model.stage1.encode_channel(z_t)
        emb_npd = emb[:, br_t, :]
        x_cw_npd = torch.from_numpy(x[:, br]).long()
        loss = model.stage1.tree.fast_ce(emb_npd, x_cw_npd)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 5000 == 0:
            # Quick eval
            model.stage1.eval()
            rng_e = np.random.default_rng(999); np.random.seed(999)
            errs = 0
            with torch.no_grad():
                for _ in range(300 // batch + 1):
                    actual = min(batch, 300)
                    um, _, zz, _, _ = make_batch(channel, Au, Av, actual, rng_e)
                    zt = torch.from_numpy(zz)
                    em = model.stage1.encode_channel(zt)
                    emn = em[:, br_t, :]
                    uh = model.stage1.tree.decode(emn, fu_set)
                    for i in range(actual):
                        if any(int(uh[i, p-1]) != int(um[i, p-1]) for p in Au):
                            errs += 1
            bler = errs / 300
            if bler < best_bler_s1:
                best_bler_s1 = bler
            print(f'  [S1 {it}/{iters_s1}] loss={loss.item():.4f} BLER={bler:.4f} best={best_bler_s1:.4f} '
                  f'{(time.time()-t0)/60:.1f}min')
            model.stage1.train()
    s1_time = (time.time() - t0) / 60

    # Train Stage 2
    opt2 = torch.optim.AdamW(model.stage2.parameters(), lr=lr, weight_decay=1e-5)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=iters_s2, eta_min=lr * 0.1)
    rng2 = np.random.default_rng(43)
    model.stage1.eval(); model.stage2.train()
    t1 = time.time()
    for it in range(1, iters_s2 + 1):
        um, vm, z, x, y = make_batch(channel, Au, Av, batch, rng2)
        z_t = torch.from_numpy(z)
        side = torch.from_numpy((1.0 - 2.0 * x.astype(np.float32))).unsqueeze(-1)
        emb = model.stage2.encode_channel(z_t, side=side)
        emb_npd = emb[:, br_t, :]
        y_cw_npd = torch.from_numpy(y[:, br]).long()
        loss = model.stage2.tree.fast_ce(emb_npd, y_cw_npd)
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage2.parameters(), 1.0)
        opt2.step(); sched2.step()
    s2_time = (time.time() - t1) / 60

    # Chained eval
    model.stage1.eval(); model.stage2.eval()
    rng_c = np.random.default_rng(777); np.random.seed(777)
    errs_u = errs_v = errs_total = 0
    with torch.no_grad():
        for _ in range(n_cw_eval // batch):
            um, vm, z, xp, yp = make_batch(channel, Au, Av, batch, rng_c)
            z_t = torch.from_numpy(z)
            emb1 = model.stage1.encode_channel(z_t)[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb1, fu_set)
            x_hat = polar_encode_batch(u_hat.numpy().astype(int))
            side = torch.from_numpy((1.0 - 2.0 * x_hat.astype(np.float32))).unsqueeze(-1)
            emb2 = model.stage2.encode_channel(z_t, side=side)[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb2, fv_set)
            for i in range(batch):
                uw = any(int(u_hat[i, p-1]) != int(um[i, p-1]) for p in Au)
                vw = any(int(v_hat[i, p-1]) != int(vm[i, p-1]) for p in Av)
                if uw: errs_u += 1
                if vw: errs_v += 1
                if uw or vw: errs_total += 1
    chained_bler = errs_total / n_cw_eval

    # Memoryless SC baseline
    rng_b = np.random.default_rng(555); np.random.seed(555)
    base_total = 0
    for _ in range(n_cw_eval):
        um = np.zeros(N, int); vm = np.zeros(N, int)
        for p in Au: um[p-1] = rng_b.integers(0, 2)
        for p in Av: vm[p-1] = rng_b.integers(0, 2)
        x = polar_encode_batch(um[None, :])[0]
        y = polar_encode_batch(vm[None, :])[0]
        z = channel.sample_batch(x[None, :], y[None, :])[0]
        ud, vd = gmac_decode_single(N, z.tolist(), b, fu_1idx, fv_1idx, gmac)
        if (any(ud[p-1] != um[p-1] for p in Au) or
                any(vd[p-1] != vm[p-1] for p in Av)):
            base_total += 1
    baseline_bler = base_total / n_cw_eval
    improvement = (baseline_bler - chained_bler) / max(baseline_bler, 1e-6) * 100

    print(f'  alpha={alpha}: chained={chained_bler:.4f} baseline={baseline_bler:.4f} '
          f'improvement={improvement:.1f}%  S1 time={s1_time:.1f}min')
    return {
        'alpha': alpha, 'sigma2': sigma2, 'N': N, 'ku': ku, 'kv': kv,
        'chained_bler': chained_bler, 'baseline_bler': baseline_bler,
        'improvement_pct': improvement,
        'stage1_best_bler': best_bler_s1,
        's1_time_min': s1_time, 's2_time_min': s2_time,
    }


if __name__ == '__main__':
    results = {}
    for alpha in [0.3, 0.5, 0.7]:
        res = train_and_eval_alpha(alpha=alpha, iters_s1=20000, iters_s2=20000)
        results[f'alpha_{alpha}'] = res
    out = os.path.join(RESULTS_DIR, 'maagn_alpha_sweep_N16.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: {out}')
    print(f'\n{"alpha":<8}{"Chained":<12}{"Baseline":<12}{"Improvement":<14}')
    for k, r in results.items():
        print(f'{r["alpha"]:<8}{r["chained_bler"]:<12.4f}{r["baseline_bler"]:<12.4f}'
              f'{r["improvement_pct"]:<14.1f}%')
