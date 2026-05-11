#!/usr/bin/env python3
"""
train_d16_h100_standalone.py
============================
Priority 1: Train d=16, hidden=100 BiGRU models at N=64 and N=128
STANDALONE (from scratch, no curriculum warm-start).

Hypothesis: h=100 matters more than d. This tests whether d=16 h=100
can match or beat d=64 h=128 at N=128 (current best 0.030 chained BLER).

N=64:  ku=15, kv=29, 500K iters, ~2h
N=128: ku=30, kv=58, 500K iters, ~4h
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from neural.npd_memory_mac import ChainedNPD_MAC

# ---- Config ----
SNR_DB = 6.0
ISI_H = 0.3
D = 16
HIDDEN = 100
N_LAYERS = 2
GRU_LAYERS = 1
ENCODER_TYPE = 'bigru'
LR = 1e-3
BATCH = 32
ITERS = 200_000
CKPT_EVERY = 50_000
EVAL_EVERY = 5_000
EVAL_CW = 300
SEED = 42

CONFIGS = [
    # (N, ku, kv, n_log2)
    (64,  15, 29, 6),
    (128, 30, 58, 7),
]

RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(_ROOT, 'results', 'reliable_evals')
os.makedirs(OUT_DIR, exist_ok=True)


def make_channel():
    return ISIMAC.from_snr_db(SNR_DB, h=ISI_H)


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, _path_i = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_dict = {p: 0 for p in range(1, N+1) if p not in Au}
    frozen_v_dict = {p: 0 for p in range(1, N+1) if p not in Av}
    frozen_u_set = {p-1 for p in frozen_u_dict.keys()}
    frozen_v_set = {p-1 for p in frozen_v_dict.keys()}
    return Au, Av, frozen_u_dict, frozen_v_dict, frozen_u_set, frozen_v_set


def make_batch(channel, N, Au, Av, batch, rng):
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au:
        u_msg[:, p-1] = rng.integers(0, 2, batch)
    for p in Av:
        v_msg[:, p-1] = rng.integers(0, 2, batch)
    x_phys = polar_encode_batch(u_msg.astype(int))
    y_phys = polar_encode_batch(v_msg.astype(int))
    z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
    return u_msg, v_msg, z.astype(np.float32), x_phys, y_phys


def eval_stage1(model_stage1, channel, N, Au, Av, frozen_u_set, n_cw=500, batch=32, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model_stage1.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, _, z, _, _ = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            emb = model_stage1.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            u_hat = model_stage1.tree.decode(emb_npd, frozen_u_set)
            for i in range(actual):
                if any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au):
                    errs += 1
            total += actual
    model_stage1.train()
    return errs, total


def train_stage1(model, channel, N, Au, Av, frozen_u_set, tag):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()

    # Constant lr=1e-3 (paper recipe)
    opt = torch.optim.Adam(model.stage1.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)

    best_bler = 1.0
    best_errs = None
    losses = []
    t0 = time.time()
    log_path = os.path.join(RESULTS_DIR, f'{tag}.log')

    model.stage1.train()
    for it in range(1, ITERS + 1):
        u_msg, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, BATCH, rng)
        z_t = torch.from_numpy(z)

        emb = model.stage1.encode_channel(z_t)
        emb_npd = emb[:, br_t, :]
        x_cw_npd = torch.from_numpy(x_phys[:, br]).long()

        loss = model.stage1.tree.fast_ce(emb_npd, x_cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % CKPT_EVERY == 0:
            ckpt_path = os.path.join(RESULTS_DIR, f'{tag}_iter{it}.pt')
            torch.save({
                'state_dict': model.stage1.state_dict(),
                'N': N, 'Au': Au, 'Av': Av, 'iter': it,
                'd': D, 'hidden': HIDDEN,
            }, ckpt_path)
            print(f'  Saved checkpoint: {ckpt_path}', flush=True)

        if it % EVAL_EVERY == 0 or it == ITERS:
            errs, total = eval_stage1(model.stage1, channel, N, Au, Av, frozen_u_set,
                                       n_cw=EVAL_CW, seed=999)
            bler = errs / total
            avg_loss = float(np.mean(losses[-min(500, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                best_errs = errs
                ckpt_path = os.path.join(RESULTS_DIR, f'{tag}_best.pt')
                torch.save({
                    'state_dict': model.stage1.state_dict(),
                    'N': N, 'Au': Au, 'Av': Av, 'iter': it,
                    'd': D, 'hidden': HIDDEN, 'best_bler': best_bler,
                }, ckpt_path)
                marker = ' *BEST*'

            msg = (f'  [S1 {it:>7}/{ITERS}] loss={avg_loss:.4f} '
                   f'BLER={bler:.4f} ({errs}/{total}) '
                   f'best={best_bler:.4f} {elapsed:.1f}min{marker}')
            print(msg, flush=True)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

    return best_bler, best_errs


def train_stage2(model, channel, N, Au, Av, frozen_v_set, tag, iters=50_000):
    """Train Stage 2 (V given true U) - converges fast."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()

    opt = torch.optim.Adam(model.stage2.parameters(), lr=LR)
    rng = np.random.default_rng(SEED + 1)

    best_bler = 1.0
    losses = []
    t0 = time.time()
    log_path = os.path.join(RESULTS_DIR, f'{tag}.log')

    model.stage1.eval()
    model.stage2.train()

    for it in range(1, iters + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, BATCH, rng)
        z_t = torch.from_numpy(z)
        # Teacher forcing with TRUE x_phys
        side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)

        emb = model.stage2.encode_channel(z_t, side=side)
        emb_npd = emb[:, br_t, :]
        y_cw_npd = torch.from_numpy(y_phys[:, br]).long()

        loss = model.stage2.tree.fast_ce(emb_npd, y_cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage2.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % 5000 == 0:
            torch.save({
                'state_dict': model.stage2.state_dict(),
                'N': N, 'Au': Au, 'Av': Av, 'iter': it,
            }, os.path.join(RESULTS_DIR, f'{tag}_iter{it}.pt'))

        if it % 2000 == 0 or it == iters:
            # Eval V with true X
            errs_v = eval_stage2_true_x(model, channel, N, Au, Av, frozen_v_set,
                                         n_cw=300, seed=999)
            bler = errs_v / 300
            avg_loss = float(np.mean(losses[-min(200, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                torch.save({
                    'state_dict': model.stage2.state_dict(),
                    'N': N, 'Au': Au, 'Av': Av, 'iter': it,
                }, os.path.join(RESULTS_DIR, f'{tag}_best.pt'))
                marker = ' *BEST*'

            msg = (f'  [S2 {it:>6}/{iters}] loss={avg_loss:.4f} '
                   f'BLER(V|trueU)={bler:.4f} best={best_bler:.4f} '
                   f'{elapsed:.1f}min{marker}')
            print(msg, flush=True)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

    return best_bler


def eval_stage2_true_x(model, channel, N, Au, Av, frozen_v_set, n_cw=300, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage2.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(32, n_cw - total)
            _, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)
            emb = model.stage2.encode_channel(z_t, side=side)
            emb_npd = emb[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb_npd, frozen_v_set)
            for i in range(actual):
                if any(int(v_hat[i, p-1].item()) != int(v_msg[i, p-1]) for p in Av):
                    errs += 1
            total += actual
    model.stage2.train()
    return errs


def eval_chained(model, channel, N, Au, Av, frozen_u_set, frozen_v_set,
                 n_cw=5000, batch=32, seed=777):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage1.eval()
    model.stage2.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            emb1 = model.stage1.encode_channel(z_t)
            emb1_npd = emb1[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb1_npd, frozen_u_set)
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)
            side = torch.from_numpy((1.0 - 2.0 * x_hat.astype(np.float32))).unsqueeze(-1)
            emb2 = model.stage2.encode_channel(z_t, side=side)
            emb2_npd = emb2[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb2_npd, frozen_v_set)
            for i in range(actual):
                u_wrong = any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au)
                v_wrong = any(int(v_hat[i, p-1].item()) != int(v_msg[i, p-1]) for p in Av)
                if u_wrong: errs_u += 1
                if v_wrong: errs_v += 1
                if u_wrong or v_wrong: errs_total += 1
            total += actual
            if total % 1000 == 0:
                print(f'    Chained: {total}/{n_cw}, errs={errs_total}', flush=True)
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v,
        'errs_total': errs_total,
        'bler_u': errs_u/n_cw, 'bler_v': errs_v/n_cw,
        'bler_total': errs_total/n_cw,
    }


def run_one_N(N, ku, kv):
    print(f'\n{"="*70}')
    print(f'TRAINING N={N}, d={D}, hidden={HIDDEN}, BiGRU, {ITERS} iters')
    print(f'ku={ku}, kv={kv}, Ru={ku/N:.3f}, Rv={kv/N:.3f}')
    print(f'{"="*70}')

    channel = make_channel()
    Au, Av, fu_dict, fv_dict, fu_set, fv_set = load_design(N, ku, kv)
    print(f'|Au|={len(Au)}, |Av|={len(Av)}')

    tag_s1 = f'd16_h100_standalone_s1_N{N}'
    tag_s2 = f'd16_h100_standalone_s2_N{N}'

    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS,
                           encoder_type=ENCODER_TYPE, gru_layers=GRU_LAYERS)
    n_params = model.count_parameters()
    print(f'Total params: {n_params:,}')

    # Stage 1
    t0 = time.time()
    s1_best_bler, s1_best_errs = train_stage1(
        model, channel, N, Au, Av, fu_set, tag_s1)
    s1_time = (time.time() - t0) / 60
    print(f'\nStage 1 DONE: best BLER={s1_best_bler:.4f} ({s1_time:.1f} min)')

    # Reload best S1
    s1_ckpt = os.path.join(RESULTS_DIR, f'{tag_s1}_best.pt')
    if os.path.exists(s1_ckpt):
        sd = torch.load(s1_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])

    # Stage 2 (50K iters - converges fast)
    t1 = time.time()
    s2_best_bler = train_stage2(model, channel, N, Au, Av, fv_set, tag_s2, iters=50_000)
    s2_time = (time.time() - t1) / 60
    print(f'\nStage 2 DONE: best BLER(V|trueU)={s2_best_bler:.4f} ({s2_time:.1f} min)')

    # Reload best S2
    s2_ckpt = os.path.join(RESULTS_DIR, f'{tag_s2}_best.pt')
    if os.path.exists(s2_ckpt):
        sd = torch.load(s2_ckpt, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd['state_dict'])

    # Final chained eval (5K CW)
    print(f'\nChained eval (5K CW)...')
    chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set, n_cw=5000)
    print(f'Chained BLER={chained["bler_total"]:.4f} '
          f'(U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f})')

    result = {
        'N': N, 'd': D, 'hidden': HIDDEN, 'encoder': ENCODER_TYPE,
        'ku': ku, 'kv': kv, 'iters': ITERS, 'n_params': n_params,
        's1_best_bler': float(s1_best_bler),
        's2_best_bler_trueU': float(s2_best_bler),
        'chained': chained,
        's1_time_min': s1_time, 's2_time_min': s2_time,
    }

    # Save result
    out_path = os.path.join(OUT_DIR, f'd16_h100_standalone_N{N}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')

    return result


def main():
    all_results = {}
    for N, ku, kv, n_log2 in CONFIGS:
        result = run_one_N(N, ku, kv)
        all_results[f'N{N}'] = result

        # Save incrementally
        out_path = os.path.join(OUT_DIR, 'd16_h100_standalone_all.json')
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f'\n{"="*70}')
    print('ALL DONE')
    print(f'{"="*70}')
    for key, res in all_results.items():
        print(f'{key}: S1={res["s1_best_bler"]:.4f} '
              f'Chained={res["chained"]["bler_total"]:.4f}')


if __name__ == '__main__':
    main()
