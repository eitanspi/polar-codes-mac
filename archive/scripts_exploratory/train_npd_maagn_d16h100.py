#!/usr/bin/env python3
"""
train_npd_maagn_d16h100.py
==========================
Train chained NPD for MA-AGN MAC at N=64 and N=128 with d=16 h=100.

Previous results with d=32 h=128: N=32 chained=0.112, N=64 chained=0.066.
Both WORSE than memoryless SC (N=32: 0.077, N=64: 0.028). The d=32 h=128
model underfits at N=64.

Hypothesis: d=16 h=100 with longer training (100K iters) may do better
(it worked for ISI-MAC).

Channel: MAAGNMAC(sigma2=10^(-0.6), alpha=0.3)
Path: Class C
Design: GMAC_C proxy at SNR=6dB.
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
from polar.channels_memory_new import MAAGNMAC
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder import decode_single as gmac_decode_single
from neural.npd_memory_mac import ChainedNPD_MAC

# ---- Config ----
SIGMA2 = 10**(-0.6)   # = 0.251... , same as SNR=6dB
ALPHA = 0.3
SNR_DB = 6
D = 16
HIDDEN = 100
N_LAYERS = 2
GRU_LAYERS = 1
ENCODER_TYPE = 'bigru'
LR = 1e-3
SEED = 42

CONFIGS = {
    64:  {'ku': 15, 'kv': 29, 'batch': 16, 'iters_s1': 100_000, 'iters_s2': 50_000},
    128: {'ku': 30, 'kv': 58, 'batch': 8,  'iters_s1': 150_000, 'iters_s2': 50_000},
}

RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_maagn_mac')
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_channel():
    return MAAGNMAC(sigma2=SIGMA2, alpha=ALPHA)


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_dict = {p: 0 for p in range(1, N+1) if p not in Au}
    frozen_v_dict = {p: 0 for p in range(1, N+1) if p not in Av}
    frozen_u_set = {p-1 for p in frozen_u_dict}
    frozen_v_set = {p-1 for p in frozen_v_dict}
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


def train_stage(model, stage_num, channel, N, Au, Av, frozen_set,
                iters, batch, lr, tag, log_file):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()

    stage = model.stage1 if stage_num == 1 else model.stage2
    opt = torch.optim.AdamW(stage.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(SEED if stage_num == 1 else SEED + 1)

    best_bler = 1.0
    losses = []
    t0 = time.time()
    stage.train()
    if stage_num == 2:
        model.stage1.eval()

    eval_every = max(2000, iters // 20)

    for it in range(1, iters + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, batch, rng)
        z_t = torch.from_numpy(z)

        if stage_num == 1:
            emb = stage.encode_channel(z_t)
            target_phys = x_phys
        else:
            side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)
            emb = stage.encode_channel(z_t, side=side)
            target_phys = y_phys

        emb_npd = emb[:, br_t, :]
        cw_npd = torch.from_numpy(target_phys[:, br]).long()

        loss = stage.tree.fast_ce(emb_npd, cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % eval_every == 0 or it == iters:
            bler = eval_stage(model, stage_num, channel, N, Au, Av, frozen_set,
                              n_cw=300, seed=999)
            avg_loss = float(np.mean(losses[-min(200, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                torch.save({'state_dict': stage.state_dict(), 'N': N, 'Au': Au, 'Av': Av},
                           os.path.join(RESULTS_DIR, f'{tag}_best.pt'))
                marker = ' *BEST*'
            if it % 10000 == 0:
                torch.save({'state_dict': stage.state_dict(), 'N': N, 'Au': Au, 'Av': Av},
                           os.path.join(RESULTS_DIR, f'{tag}_iter{it}.pt'))
            msg = (f'  [S{stage_num} {it:>6}/{iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}')
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

    return best_bler


def eval_stage(model, stage_num, channel, N, Au, Av, frozen_set,
               n_cw=500, batch=32, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    stage = model.stage1 if stage_num == 1 else model.stage2
    stage.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    info_pos = Au if stage_num == 1 else Av
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            if stage_num == 1:
                emb = stage.encode_channel(z_t)
                target_msg = u_msg
            else:
                side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)
                emb = stage.encode_channel(z_t, side=side)
                target_msg = v_msg
            emb_npd = emb[:, br_t, :]
            hat = stage.tree.decode(emb_npd, frozen_set)
            for i in range(actual):
                if any(int(hat[i, p-1].item()) != int(target_msg[i, p-1]) for p in info_pos):
                    errs += 1
            total += actual
    stage.train()
    return errs / n_cw


def eval_chained(model, channel, N, Au, Av, fu_set, fv_set, n_cw=2000, batch=32, seed=777):
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
            u_hat = model.stage1.tree.decode(emb1_npd, fu_set)
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)
            side = torch.from_numpy((1.0 - 2.0 * x_hat.astype(np.float32))).unsqueeze(-1)
            emb2 = model.stage2.encode_channel(z_t, side=side)
            emb2_npd = emb2[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb2_npd, fv_set)
            for i in range(actual):
                u_wrong = any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au)
                v_wrong = any(int(v_hat[i, p-1].item()) != int(v_msg[i, p-1]) for p in Av)
                if u_wrong: errs_u += 1
                if v_wrong: errs_v += 1
                if u_wrong or v_wrong: errs_total += 1
            total += actual
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw, 'bler_total': errs_total / n_cw,
    }


def eval_memoryless_sc(channel, N, Au, Av, fu_dict, fv_dict, n_cw=2000, seed=555):
    gmac = GaussianMAC(sigma2=channel.sigma2)
    b = make_path(N, N)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    for _ in range(n_cw):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for p in Au: u_msg[p-1] = rng.integers(0, 2)
        for p in Av: v_msg[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u_msg[None, :])[0]
        y = polar_encode_batch(v_msg[None, :])[0]
        z = channel.sample_batch(x[None, :].astype(int), y[None, :].astype(int))
        u_dec, v_dec = gmac_decode_single(N, z[0].tolist(), b, fu_dict, fv_dict, gmac, log_domain=True)
        u_wrong = any(u_dec[p-1] != u_msg[p-1] for p in Au)
        v_wrong = any(v_dec[p-1] != v_msg[p-1] for p in Av)
        if u_wrong: errs_u += 1
        if v_wrong: errs_v += 1
        if u_wrong or v_wrong: errs_total += 1
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw, 'bler_total': errs_total / n_cw,
    }


def run_one_N(N, warm_s1=None, warm_s2=None):
    cfg = CONFIGS[N]
    ku, kv = cfg['ku'], cfg['kv']
    batch = cfg['batch']
    iters_s1 = cfg['iters_s1']
    iters_s2 = cfg['iters_s2']

    channel = make_channel()
    Au, Av, fu_dict, fv_dict, fu_set, fv_set = load_design(N, ku, kv)

    tag_base = f'maagn_d{D}_h{HIDDEN}'
    s1_tag = f'{tag_base}_s1_N{N}'
    s2_tag = f'{tag_base}_s2_N{N}'
    log_file = os.path.join(RESULTS_DIR, f'{tag_base}_N{N}.log')

    with open(log_file, 'a') as lf:
        lf.write(f'\n=== MA-AGN N={N} d={D} h={HIDDEN} started {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        lf.write(f'sigma2={SIGMA2:.6f} alpha={ALPHA} ku={ku} kv={kv}\n')

    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS,
                           encoder_type=ENCODER_TYPE, window_size=2,
                           gru_layers=GRU_LAYERS)

    if warm_s1 and os.path.exists(warm_s1):
        try:
            sd = torch.load(warm_s1, weights_only=False, map_location='cpu')
            model.stage1.load_state_dict(sd['state_dict'])
            print(f'  warm-start stage1 from {os.path.basename(warm_s1)}')
        except Exception as e:
            print(f'  warm-start stage1 failed: {e}')

    print(f'\n{"="*60}\nMA-AGN N={N} d={D} h={HIDDEN} params={model.count_parameters():,}\n{"="*60}')

    # Stage 1
    t0 = time.time()
    s1_best = train_stage(model, 1, channel, N, Au, Av, fu_set,
                          iters_s1, batch, LR, s1_tag, log_file)
    s1_time = (time.time() - t0) / 60
    print(f'\n  Stage 1 best BLER: {s1_best:.4f} ({s1_time:.1f} min)')

    s1_ckpt = os.path.join(RESULTS_DIR, f'{s1_tag}_best.pt')
    if os.path.exists(s1_ckpt):
        sd = torch.load(s1_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])

    # Stage 2
    t1 = time.time()
    s2_best = train_stage(model, 2, channel, N, Au, Av, fv_set,
                          iters_s2, batch, LR, s2_tag, log_file)
    s2_time = (time.time() - t1) / 60
    print(f'\n  Stage 2 best BLER(V|trueU): {s2_best:.4f} ({s2_time:.1f} min)')

    s2_ckpt = os.path.join(RESULTS_DIR, f'{s2_tag}_best.pt')
    if os.path.exists(s2_ckpt):
        sd = torch.load(s2_ckpt, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd['state_dict'])

    # Chained eval
    print(f'\n  Chained inference (5000 codewords)...')
    chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set, n_cw=5000, seed=777)
    print(f'  chained BLER={chained["bler_total"]:.4f} (U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f})')

    # Memoryless SC baseline
    print(f'\n  Memoryless GMAC SC baseline (3000 codewords)...')
    t_ref = time.time()
    baseline = eval_memoryless_sc(channel, N, Au, Av, fu_dict, fv_dict, n_cw=3000, seed=555)
    print(f'  memoryless SC BLER={baseline["bler_total"]:.4f} ({(time.time()-t_ref)/60:.1f} min)')

    result = {
        'channel': 'maagn', 'sigma2': float(SIGMA2), 'alpha': ALPHA,
        'N': N, 'ku': ku, 'kv': kv, 'd': D, 'hidden': HIDDEN,
        'stage1_best_bler': float(s1_best),
        'stage2_best_bler_true_x': float(s2_best),
        'chained': {k: float(v) if isinstance(v, (int, float)) else v for k, v in chained.items()},
        'memoryless_sc': {k: float(v) if isinstance(v, (int, float)) else v for k, v in baseline.items()},
        's1_time_min': s1_time, 's2_time_min': s2_time,
    }
    return result


def main():
    all_results = {}
    out_json = os.path.join(RESULTS_DIR, 'maagn_d16h100_results.json')
    t_total = time.time()

    prev_s1 = None
    for N in [64, 128]:
        res = run_one_N(N, warm_s1=prev_s1)
        all_results[str(N)] = res
        prev_s1 = os.path.join(RESULTS_DIR, f'maagn_d{D}_h{HIDDEN}_s1_N{N}_best.pt')
        with open(out_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Saved incremental: {out_json}')

    total_min = (time.time() - t_total) / 60
    print(f'\n{"="*60}\nAll done in {total_min:.1f} min')
    print(f'\n{"N":<6}{"S1 BLER":<12}{"Chained":<12}{"Memoryless SC":<16}{"Ratio":<10}')
    for Ns, r in all_results.items():
        ch = r['chained']['bler_total']
        ms = r['memoryless_sc']['bler_total']
        print(f'{Ns:<6}{r["stage1_best_bler"]:<12.4f}{ch:<12.4f}{ms:<16.4f}{ch/max(ms,1e-6):<10.3f}')


if __name__ == '__main__':
    main()
