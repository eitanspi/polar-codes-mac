#!/usr/bin/env python3
"""
train_npd_maagn_mac.py
======================
Training driver for the chained NPD for the Moving-Average Additive Gaussian
Noise MAC (MA-AGN MAC). See `polar.channels_memory_new.MAAGNMAC`.

The noise process is AR(1) with CONTINUOUS state. Unlike ISI-MAC or
Gilbert-Elliott MAC, there is NO finite-state trellis and hence NO
analytical SC decoder that exploits the memory. The practical baseline is
the memoryless GMAC SC — treating each output as independent Gaussian.
This is the "flagship" memory case from Aharoni et al. 2024 (NPD paper
Sec. VI-B) where neural chained NPD has unique value.

Pipeline at each N:
  1. Load GMAC_C MC-design frozen set (proxy — the memoryless GMAC with
     the same marginal variance is a close approximation for picking
     strong positions).
  2. Train Stage 1 (U on marginal) with fast_ce BCE across tree depths,
     using a BiGRU encoder to capture the MA-AGN memory.
  3. Freeze Stage 1 and train Stage 2 (V given true X via teacher forcing).
  4. Chained inference: Stage 1 -> Stage 2.
  5. Baseline: memoryless GMAC SC on the MA-AGN channel (no memory handled).

Saves checkpoints to class_c_npd/results/npd_maagn_mac/*.pt and a results
markdown to class_c_npd/results/npd_maagn_mac_results.md.

Usage:
  python scripts/train_npd_maagn_mac.py --N 16 --iters 20000
  python scripts/train_npd_maagn_mac.py --all  # all N
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)

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


# ─── Config ──────────────────────────────────────────────────────────────────

SNR_DB = 6.0      # matches GMAC_C n{n} snr6dB designs
ALPHA = 0.3       # AR(1) coefficient; matches ISI h = 0.3 config

# GMAC Class C rates at SNR=6dB (same as ISI-MAC training config)
RATES: Dict[int, Tuple[int, int]] = {
    16: (4, 7),
    32: (7, 15),
    64: (15, 29),
}

RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_maagn_mac')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Channel factory ─────────────────────────────────────────────────────────

def make_maagn(alpha=ALPHA, snr_db=SNR_DB):
    ch = MAAGNMAC.from_snr_db(snr_db, alpha=alpha)
    return ch, {'alpha': alpha, 'snr_db': snr_db, 'sigma2': ch.sigma2}


# ─── Design loader: GMAC_C proxy ─────────────────────────────────────────────

def load_design(N: int, ku: int, kv: int):
    """
    Load the GMAC_C MC-design (proxy for MA-AGN, since MC-designing MA-AGN
    is expensive and the memoryless GMAC is a close match in marginal
    variance). This approximation is documented in the results.
    """
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs',
                        f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Design file missing: {path}')
    Au_list, Av_list, fu_dict, fv_dict, _, _, _ = design_from_file(
        path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_dict_1idx = {p: 0 for p in range(1, N + 1) if p not in Au}
    frozen_v_dict_1idx = {p: 0 for p in range(1, N + 1) if p not in Av}
    frozen_u_set_0idx = {p - 1 for p in frozen_u_dict_1idx.keys()}
    frozen_v_set_0idx = {p - 1 for p in frozen_v_dict_1idx.keys()}
    return (Au, Av, frozen_u_dict_1idx, frozen_v_dict_1idx,
            frozen_u_set_0idx, frozen_v_set_0idx)


# ─── Training batch ─────────────────────────────────────────────────────────

def make_batch(channel, N: int, Au: list, Av: list, batch: int,
               rng: np.random.Generator):
    """Generate a fresh training batch (u, v, z, x, y)."""
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au:
        u_msg[:, p - 1] = rng.integers(0, 2, batch)
    for p in Av:
        v_msg[:, p - 1] = rng.integers(0, 2, batch)

    x_phys = polar_encode_batch(u_msg.astype(int))
    y_phys = polar_encode_batch(v_msg.astype(int))

    z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
    return u_msg, v_msg, z.astype(np.float32), x_phys, y_phys


# ─── Stage 1 training ───────────────────────────────────────────────────────

def train_stage1(model: ChainedNPD_MAC, channel, N: int, Au: list, Av: list,
                 frozen_u_set: set, iters: int, batch: int, lr: float,
                 ckpt_base: str, tag: str, log_file: str,
                 eval_every: int = 2000, eval_cw: int = 200, seed: int = 42):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long() if isinstance(br, np.ndarray) else torch.tensor(br, dtype=torch.long)

    opt = torch.optim.AdamW(model.stage1.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.1)
    rng = np.random.default_rng(seed)

    best_bler = 1.0
    losses = []
    t0 = time.time()

    model.stage1.train()
    for it in range(1, iters + 1):
        u_msg, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, batch, rng)
        z_t = torch.from_numpy(z)

        emb = model.stage1.encode_channel(z_t)
        emb_npd = emb[:, br_t, :]
        x_cw_npd = torch.from_numpy(x_phys[:, br]).long()

        loss = model.stage1.tree.fast_ce(emb_npd, x_cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if it % eval_every == 0 or it == iters:
            bler = eval_stage1(model, channel, N, Au, Av, frozen_u_set,
                               n_cw=eval_cw, seed=999)
            avg_loss = float(np.mean(losses[-min(200, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                ckpt_path = os.path.join(ckpt_base, f'{tag}_best.pt')
                torch.save({
                    'state_dict': model.stage1.state_dict(),
                    'N': N, 'Au': Au, 'Av': Av,
                }, ckpt_path)
                marker = ' *BEST*'
            if it % 5000 == 0:
                torch.save({
                    'state_dict': model.stage1.state_dict(),
                    'N': N, 'Au': Au, 'Av': Av,
                }, os.path.join(ckpt_base, f'{tag}_iter{it}.pt'))
            msg = (f'  [S1 {it:>6}/{iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}')
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

    return best_bler


def eval_stage1(model, channel, N, Au, Av, frozen_u_set,
                n_cw=500, batch=32, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long() if isinstance(br, np.ndarray) else torch.tensor(br, dtype=torch.long)
    model.stage1.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, _, z, _, _ = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            emb = model.stage1.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb_npd, frozen_u_set)
            for i in range(actual):
                if any(int(u_hat[i, p - 1].item()) != int(u_msg[i, p - 1]) for p in Au):
                    errs += 1
            total += actual
    model.stage1.train()
    return errs / n_cw


# ─── Stage 2 training ───────────────────────────────────────────────────────

def train_stage2(model, channel, N, Au, Av, frozen_v_set,
                 iters, batch, lr, ckpt_base, tag, log_file,
                 eval_every=2000, eval_cw=200, seed=42):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long() if isinstance(br, np.ndarray) else torch.tensor(br, dtype=torch.long)

    opt = torch.optim.AdamW(model.stage2.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.1)
    rng = np.random.default_rng(seed + 1)

    best_bler = 1.0
    losses = []
    t0 = time.time()

    model.stage1.eval()
    model.stage2.train()

    for it in range(1, iters + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, batch, rng)
        z_t = torch.from_numpy(z)
        side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)

        emb = model.stage2.encode_channel(z_t, side=side)
        emb_npd = emb[:, br_t, :]
        y_cw_npd = torch.from_numpy(y_phys[:, br]).long()

        loss = model.stage2.tree.fast_ce(emb_npd, y_cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage2.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if it % eval_every == 0 or it == iters:
            bler = eval_stage2_with_true_x(model, channel, N, Au, Av,
                                           frozen_v_set, n_cw=eval_cw, seed=999)
            avg_loss = float(np.mean(losses[-min(200, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                ckpt_path = os.path.join(ckpt_base, f'{tag}_best.pt')
                torch.save({
                    'state_dict': model.stage2.state_dict(),
                    'N': N, 'Au': Au, 'Av': Av,
                }, ckpt_path)
                marker = ' *BEST*'
            if it % 5000 == 0:
                torch.save({
                    'state_dict': model.stage2.state_dict(),
                    'N': N, 'Au': Au, 'Av': Av,
                }, os.path.join(ckpt_base, f'{tag}_iter{it}.pt'))
            msg = (f'  [S2 {it:>6}/{iters}] loss={avg_loss:.4f} BLER(V|trueU)={bler:.4f} '
                   f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}')
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

    return best_bler


def eval_stage2_with_true_x(model, channel, N, Au, Av, frozen_v_set,
                            n_cw=500, batch=32, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long() if isinstance(br, np.ndarray) else torch.tensor(br, dtype=torch.long)
    model.stage2.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            _, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)
            emb = model.stage2.encode_channel(z_t, side=side)
            emb_npd = emb[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb_npd, frozen_v_set)
            for i in range(actual):
                if any(int(v_hat[i, p - 1].item()) != int(v_msg[i, p - 1]) for p in Av):
                    errs += 1
            total += actual
    model.stage2.train()
    return errs / n_cw


# ─── Chained inference ──────────────────────────────────────────────────────

def eval_chained(model, channel, N, Au, Av, frozen_u_set, frozen_v_set,
                 n_cw=2000, batch=32, seed=777):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long() if isinstance(br, np.ndarray) else torch.tensor(br, dtype=torch.long)

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
                u_wrong = any(int(u_hat[i, p - 1].item()) != int(u_msg[i, p - 1]) for p in Au)
                v_wrong = any(int(v_hat[i, p - 1].item()) != int(v_msg[i, p - 1]) for p in Av)
                if u_wrong: errs_u += 1
                if v_wrong: errs_v += 1
                if u_wrong or v_wrong: errs_total += 1
            total += actual

    return {
        'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw,
        'bler_v': errs_v / n_cw,
        'bler_total': errs_total / n_cw,
    }


# ─── Memoryless GMAC SC baseline (on MA-AGN samples) ─────────────────────────

def eval_memoryless_sc(channel, N, Au, Av, frozen_u_dict, frozen_v_dict,
                       n_cw=2000, seed=555):
    """
    Baseline: sample from MA-AGN, decode using memoryless GMAC SC (ignoring
    memory). Uses the SAME sigma2 for the GMAC likelihood.
    """
    gmac = GaussianMAC(sigma2=channel.sigma2)
    b = make_path(N, N)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    errs_u = errs_v = errs_total = 0
    for i in range(n_cw):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for p in Au: u_msg[p - 1] = rng.integers(0, 2)
        for p in Av: v_msg[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u_msg[None, :])[0]
        y = polar_encode_batch(v_msg[None, :])[0]
        z = channel.sample_batch(x[None, :].astype(int), y[None, :].astype(int))[0]
        u_dec, v_dec = gmac_decode_single(
            N, z.tolist(), b, frozen_u_dict, frozen_v_dict, gmac,
            log_domain=True)
        u_wrong = any(u_dec[p - 1] != u_msg[p - 1] for p in Au)
        v_wrong = any(v_dec[p - 1] != v_msg[p - 1] for p in Av)
        if u_wrong: errs_u += 1
        if v_wrong: errs_v += 1
        if u_wrong or v_wrong: errs_total += 1
    return {
        'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw,
        'bler_v': errs_v / n_cw,
        'bler_total': errs_total / n_cw,
    }


# ─── Orchestration per N ────────────────────────────────────────────────────

def run_one_N(N, iters_stage1, iters_stage2, encoder_type, window_size,
              batch, lr, d, hidden, n_layers, gru_layers,
              warm_stage1_ckpt=None, warm_stage2_ckpt=None,
              eval_cw=300, final_cw=2000, baseline_cw=2000):
    ku, kv = RATES[N]
    channel, ch_meta = make_maagn()
    Au, Av, fu_1idx, fv_1idx, fu_set, fv_set = load_design(N, ku, kv)

    tag_base = f'maagn_{encoder_type}'
    if encoder_type == 'window':
        tag_base += f'_w{window_size}'
    if encoder_type == 'bigru':
        tag_base += f'_L{gru_layers}'

    s1_tag = f'{tag_base}_s1_N{N}'
    s2_tag = f'{tag_base}_s2_N{N}'
    log_file = os.path.join(RESULTS_DIR, f'{tag_base}_N{N}.log')

    with open(log_file, 'a') as lf:
        lf.write(f'\n=== {tag_base} N={N} started {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        lf.write(f'channel: maagn meta={ch_meta}\n')
        lf.write(f'ku={ku} kv={kv} Ru={ku/N:.3f} Rv={kv/N:.3f}\n')

    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=d, hidden=hidden, n_layers=n_layers,
                           encoder_type=encoder_type, window_size=window_size,
                           gru_layers=gru_layers)

    if warm_stage1_ckpt and os.path.exists(warm_stage1_ckpt):
        try:
            sd = torch.load(warm_stage1_ckpt, weights_only=False, map_location='cpu')
            model.stage1.load_state_dict(sd['state_dict'])
            print(f'  warm-start stage1 from {os.path.basename(warm_stage1_ckpt)}')
        except Exception as e:
            print(f'  warm-start stage1 failed: {e}')
    if warm_stage2_ckpt and os.path.exists(warm_stage2_ckpt):
        try:
            sd = torch.load(warm_stage2_ckpt, weights_only=False, map_location='cpu')
            model.stage2.load_state_dict(sd['state_dict'])
            print(f'  warm-start stage2 from {os.path.basename(warm_stage2_ckpt)}')
        except Exception as e:
            print(f'  warm-start stage2 failed: {e}')

    print(f'\n{"="*60}\nN={N} channel=maagn encoder={encoder_type} '
          f'iters(S1/S2)={iters_stage1}/{iters_stage2} '
          f'params={model.count_parameters():,}\n{"="*60}')

    # Stage 1
    t0 = time.time()
    s1_best = train_stage1(
        model, channel, N, Au, Av, fu_set,
        iters=iters_stage1, batch=batch, lr=lr,
        ckpt_base=RESULTS_DIR, tag=s1_tag, log_file=log_file,
        eval_every=max(2000, iters_stage1 // 10), eval_cw=eval_cw,
    )
    s1_time = (time.time() - t0) / 60
    print(f'\n  Stage 1 best BLER: {s1_best:.4f}  ({s1_time:.1f} min)')

    # Reload best Stage 1
    s1_best_ckpt = os.path.join(RESULTS_DIR, f'{s1_tag}_best.pt')
    if os.path.exists(s1_best_ckpt):
        sd = torch.load(s1_best_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])

    # Stage 2
    t1 = time.time()
    s2_best = train_stage2(
        model, channel, N, Au, Av, fv_set,
        iters=iters_stage2, batch=batch, lr=lr,
        ckpt_base=RESULTS_DIR, tag=s2_tag, log_file=log_file,
        eval_every=max(2000, iters_stage2 // 10), eval_cw=eval_cw,
    )
    s2_time = (time.time() - t1) / 60
    print(f'\n  Stage 2 best BLER(V|trueU): {s2_best:.4f}  ({s2_time:.1f} min)')

    # Reload best Stage 2
    s2_best_ckpt = os.path.join(RESULTS_DIR, f'{s2_tag}_best.pt')
    if os.path.exists(s2_best_ckpt):
        sd = torch.load(s2_best_ckpt, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd['state_dict'])

    # Chained NPD inference
    print(f'\n  Chained NPD inference ({final_cw} codewords)...')
    chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set,
                           n_cw=final_cw, seed=777)
    print(f'  chained BLER={chained["bler_total"]:.4f} '
          f'(U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f})')

    # Memoryless GMAC SC baseline
    print(f'\n  Memoryless GMAC SC baseline ({baseline_cw} codewords)...')
    t_ref = time.time()
    baseline = eval_memoryless_sc(channel, N, Au, Av, fu_1idx, fv_1idx,
                                  n_cw=baseline_cw, seed=555)
    print(f'  memoryless SC BLER={baseline["bler_total"]:.4f} '
          f'(U={baseline["bler_u"]:.4f}, V={baseline["bler_v"]:.4f}) '
          f'({(time.time()-t_ref)/60:.1f} min)')

    improvement = (baseline['bler_total'] - chained['bler_total']) / max(
        baseline['bler_total'], 1e-6)
    print(f'\n  Improvement: {improvement*100:.1f}% '
          f'(ratio chained/baseline = '
          f'{chained["bler_total"]/max(baseline["bler_total"],1e-6):.3f})')

    result = {
        'channel': 'maagn', 'channel_meta': ch_meta,
        'N': N, 'ku': ku, 'kv': kv,
        'encoder': encoder_type, 'window_size': window_size, 'd': d,
        'hidden': hidden, 'n_layers': n_layers, 'gru_layers': gru_layers,
        'stage1_best_bler': float(s1_best),
        'stage2_best_bler_true_x': float(s2_best),
        'chained': {k: (float(v) if isinstance(v, (float, int)) else v)
                    for k, v in chained.items()},
        'memoryless_sc': {k: (float(v) if isinstance(v, (float, int)) else v)
                          for k, v in baseline.items()},
        'improvement_ratio': float(chained['bler_total'] /
                                   max(baseline['bler_total'], 1e-6)),
        'stage1_iters': iters_stage1, 'stage2_iters': iters_stage2,
        'stage1_time_min': s1_time, 'stage2_time_min': s2_time,
        's1_ckpt': s1_best_ckpt, 's2_ckpt': s2_best_ckpt,
    }
    return result


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, nargs='+', default=None)
    p.add_argument('--encoder_type', type=str, default='bigru',
                   choices=['window', 'bigru'])
    p.add_argument('--window_size', type=int, default=2)
    p.add_argument('--gru_layers', type=int, default=1)
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--eval_cw', type=int, default=300)
    p.add_argument('--final_cw', type=int, default=2000)
    p.add_argument('--baseline_cw', type=int, default=2000)
    p.add_argument('--all', action='store_true')
    p.add_argument('--save_json', type=str, default=None)
    p.add_argument('--curriculum', action='store_true')
    args = p.parse_args()

    N_list = args.N if args.N is not None else ([16, 32, 64] if args.all else [16])

    # Per-N training schedules (from task brief)
    SCHED = {
        16: {'iters_s1': 20000, 'iters_s2': 20000, 'batch': 32, 'lr': 1e-3},
        32: {'iters_s1': 30000, 'iters_s2': 30000, 'batch': 16, 'lr': 5e-4},
        64: {'iters_s1': 40000, 'iters_s2': 40000, 'batch': 8,  'lr': 5e-4},
    }

    all_results = {}
    prev_s1 = None
    prev_s2 = None
    out_json = args.save_json or os.path.join(
        RESULTS_DIR, f'maagn_{args.encoder_type}_results.json')

    t_total = time.time()
    for N in N_list:
        sch = SCHED[N]
        warm_s1 = prev_s1 if args.curriculum else None
        warm_s2 = prev_s2 if args.curriculum else None

        res = run_one_N(
            N=N,
            iters_stage1=sch['iters_s1'], iters_stage2=sch['iters_s2'],
            encoder_type=args.encoder_type, window_size=args.window_size,
            batch=sch['batch'], lr=sch['lr'],
            d=args.d, hidden=args.hidden, n_layers=args.n_layers,
            gru_layers=args.gru_layers,
            warm_stage1_ckpt=warm_s1, warm_stage2_ckpt=warm_s2,
            eval_cw=args.eval_cw, final_cw=args.final_cw,
            baseline_cw=args.baseline_cw,
        )
        all_results[str(N)] = res
        prev_s1 = res['s1_ckpt']
        prev_s2 = res['s2_ckpt']

        with open(out_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Saved incremental results: {out_json}')

    total_min = (time.time() - t_total) / 60
    print(f'\n{"="*60}\nAll N complete in {total_min:.1f} min')
    print(f'Results saved to {out_json}')

    # Quick summary
    print(f'\n{"N":<6}{"S1 BLER":<12}{"Chained BLER":<16}{"Memoryless SC":<16}'
          f'{"Ratio":<10}')
    for Ns, r in all_results.items():
        s1 = r['stage1_best_bler']
        ch = r['chained']['bler_total']
        ms = r['memoryless_sc']['bler_total']
        ratio = ch / max(ms, 1e-6)
        print(f'{Ns:<6}{s1:<12.4f}{ch:<16.4f}{ms:<16.4f}{ratio:<10.3f}')


if __name__ == '__main__':
    main()
