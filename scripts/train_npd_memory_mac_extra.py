#!/usr/bin/env python3
"""
train_npd_memory_mac_extra.py
=============================
Extend the chained NPD for memory MAC channels to NEW memory channels beyond
ISI-MAC: Gilbert-Elliott MAC and Trapdoor MAC. Uses the existing
`neural.npd_memory_mac` module and mirrors scripts/train_npd_memory_mac.py
but allows specifying a custom channel factory + design loader per channel.

Budget: ~3 hours total. Focus on N=16, 32.

Saves checkpoints to class_c_npd/results/npd_memory_mac/*.pt and an overall
markdown summary.
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
from polar.channels_memory_new import GilbertElliottMAC, TrapdoorMAC
from polar.design_mc import design_from_file
from polar.decoder_trellis import decode_single as trellis_decode_single
from polar.decoder import decode_single as sc_decode_single

from neural.npd_memory_mac import ChainedNPD_MAC


# ─── Config ──────────────────────────────────────────────────────────────────

SNR_DB = 6.0
RATES: Dict[int, Tuple[int, int]] = {
    16: (4, 7),
    32: (7, 15),
    64: (15, 29),
}
RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Channel factories ──────────────────────────────────────────────────────

def make_channel(name: str):
    """
    Return (channel, meta_dict, design_path_template) for a given short name.

    design_path_template: used by load_design to pick the proxy Class C design.
      - Gilbert-Elliott (Gaussian emission) -> GMAC_C at matching SNR.
      - Trapdoor (binary) -> BEMAC_C (binary proxy).
    """
    name = name.lower()
    if name in ('ge_mac', 'gemac', 'gilbert_elliott', 'gilbert-elliott'):
        sigma2 = 10.0 ** (-SNR_DB / 10.0)
        ch = GilbertElliottMAC(
            p_gb=0.08, p_bg=0.4,
            sigma2_good=sigma2 * 0.8, sigma2_bad=sigma2 * 5.0,
        )
        meta = {
            'p_gb': ch.p_gb, 'p_bg': ch.p_bg,
            'sigma2_good': ch.sigma2_good, 'sigma2_bad': ch.sigma2_bad,
            'snr_db_ref': SNR_DB,
        }
        return ch, meta, 'gmac_C'
    if name in ('trapdoor', 'trapdoor_mac'):
        ch = TrapdoorMAC(p_noise=0.1)
        meta = {'p_noise': ch.p_noise}
        return ch, meta, 'bemac_C'
    raise ValueError(f'unknown channel {name!r}')


def load_design(channel_short: str, N: int, ku: int, kv: int):
    """Return Au, Av, frozen dicts (1-idx), frozen sets (0-idx)."""
    n = int(math.log2(N))
    _, _, design_family = make_channel(channel_short)
    if design_family == 'gmac_C':
        path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    elif design_family == 'bemac_C':
        path = os.path.join(_ROOT, 'designs', f'bemac_C_n{n}.npz')
    else:
        raise ValueError(f'no design for {channel_short!r}')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    fu_1idx = {p: 0 for p in range(1, N + 1) if p not in Au}
    fv_1idx = {p: 0 for p in range(1, N + 1) if p not in Av}
    fu_set = {p - 1 for p in fu_1idx.keys()}
    fv_set = {p - 1 for p in fv_1idx.keys()}
    return Au, Av, fu_1idx, fv_1idx, fu_set, fv_set, path


# ─── Training batch ─────────────────────────────────────────────────────────

def make_batch(channel, N: int, Au: list, Av: list, batch: int, rng: np.random.Generator):
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au:
        u_msg[:, p - 1] = rng.integers(0, 2, batch)
    for p in Av:
        v_msg[:, p - 1] = rng.integers(0, 2, batch)
    x_phys = polar_encode_batch(u_msg.astype(int))
    y_phys = polar_encode_batch(v_msg.astype(int))
    # Set np seed at the batch level so channel sampling is deterministic per rng step
    z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
    return u_msg, v_msg, z.astype(np.float32), x_phys, y_phys


# ─── Stage 1 / Stage 2 training (copy of train_npd_memory_mac.py routines)
# Kept here so we do NOT modify that file.

def train_stage1(model, channel, N, Au, Av, frozen_u_set, iters, batch, lr,
                 ckpt_base, tag, log_file, eval_every=2000, eval_cw=200, seed=42):
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
        if not torch.isfinite(loss):
            msg = f'  [S1 {it:>6}/{iters}] NaN loss, aborting'
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            return best_bler
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
                torch.save({'state_dict': model.stage1.state_dict(),
                            'N': N, 'Au': Au, 'Av': Av}, ckpt_path)
                marker = ' *BEST*'
            msg = (f'  [S1 {it:>6}/{iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}')
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
    return best_bler


def eval_stage1(model, channel, N, Au, Av, frozen_u_set, n_cw=500, batch=32, seed=999):
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


def train_stage2(model, channel, N, Au, Av, frozen_v_set, iters, batch, lr,
                 ckpt_base, tag, log_file, eval_every=2000, eval_cw=200, seed=42):
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
        if not torch.isfinite(loss):
            msg = f'  [S2 {it:>6}/{iters}] NaN loss, aborting'
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            return best_bler
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage2.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if it % eval_every == 0 or it == iters:
            bler = eval_stage2_with_true_x(model, channel, N, Au, Av, frozen_v_set,
                                            n_cw=eval_cw, seed=999)
            avg_loss = float(np.mean(losses[-min(200, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                ckpt_path = os.path.join(ckpt_base, f'{tag}_best.pt')
                torch.save({'state_dict': model.stage2.state_dict(),
                            'N': N, 'Au': Au, 'Av': Av}, ckpt_path)
                marker = ' *BEST*'
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
                if u_wrong:
                    errs_u += 1
                if v_wrong:
                    errs_v += 1
                if u_wrong or v_wrong:
                    errs_total += 1
            total += actual
    return {'n_cw': n_cw,
            'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
            'bler_u': errs_u / n_cw,
            'bler_v': errs_v / n_cw,
            'bler_total': errs_total / n_cw}


# ─── Trellis SC and memoryless SC baselines ─────────────────────────────────

def eval_trellis_sc(channel, N, Au, Av, frozen_u_dict, frozen_v_dict,
                    n_cw=300, seed=555):
    from polar.design import make_path
    b = make_path(N, N)  # Class C corner-rate
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    for i in range(n_cw):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for p in Au:
            u_msg[p - 1] = rng.integers(0, 2)
        for p in Av:
            v_msg[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u_msg[None, :])[0]
        y = polar_encode_batch(v_msg[None, :])[0]
        z = channel.sample_batch(x[None, :].astype(int), y[None, :].astype(int))[0]
        try:
            u_dec, v_dec = trellis_decode_single(
                N, z.tolist() if hasattr(z, 'tolist') else list(z),
                b, frozen_u_dict, frozen_v_dict, channel, log_domain=True)
        except Exception as e:
            print(f'  trellis eval iter {i} raised: {e}')
            continue
        u_wrong = any(u_dec[p - 1] != u_msg[p - 1] for p in Au)
        v_wrong = any(v_dec[p - 1] != v_msg[p - 1] for p in Av)
        if u_wrong: errs_u += 1
        if v_wrong: errs_v += 1
        if u_wrong or v_wrong: errs_total += 1
    return {'n_cw': n_cw,
            'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
            'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw,
            'bler_total': errs_total / n_cw}


def eval_memoryless_sc(channel, N, Au, Av, frozen_u_dict, frozen_v_dict,
                        n_cw=300, seed=444):
    """Memoryless SC baseline: builds (N,2,2) leaf using transition_prob(z,x,y,0)
    ignoring state memory. Shows how much memory matters."""
    from polar.design import make_path
    b = make_path(N, N)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    for i in range(n_cw):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for p in Au:
            u_msg[p - 1] = rng.integers(0, 2)
        for p in Av:
            v_msg[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u_msg[None, :])[0]
        y = polar_encode_batch(v_msg[None, :])[0]
        z = channel.sample_batch(x[None, :].astype(int), y[None, :].astype(int))[0]
        try:
            u_dec, v_dec = sc_decode_single(
                N, z.tolist() if hasattr(z, 'tolist') else list(z),
                b, frozen_u_dict, frozen_v_dict, channel, log_domain=True)
        except Exception as e:
            print(f'  memoryless SC iter {i} raised: {e}')
            continue
        u_wrong = any(u_dec[p - 1] != u_msg[p - 1] for p in Au)
        v_wrong = any(v_dec[p - 1] != v_msg[p - 1] for p in Av)
        if u_wrong: errs_u += 1
        if v_wrong: errs_v += 1
        if u_wrong or v_wrong: errs_total += 1
    return {'n_cw': n_cw,
            'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
            'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw,
            'bler_total': errs_total / n_cw}


# ─── Orchestration per N ────────────────────────────────────────────────────

def run_one_N(channel_name, N, iters, encoder_type, window_size, batch, lr,
              d, hidden, n_layers, gru_layers,
              eval_cw=200, final_cw=2000, trellis_cw=300, memoryless_cw=300,
              skip_trellis=False, tag_suffix=''):
    ku, kv = RATES[N]
    channel, ch_meta, _ = make_channel(channel_name)
    Au, Av, fu_1idx, fv_1idx, fu_set, fv_set, design_path = load_design(
        channel_name, N, ku, kv)

    tag_base = f'{channel_name}_{encoder_type}'
    if encoder_type == 'window':
        tag_base += f'_w{window_size}'
    if encoder_type == 'bigru':
        tag_base += f'_L{gru_layers}'
    if tag_suffix:
        tag_base += f'_{tag_suffix}'
    s1_tag = f'{tag_base}_s1_N{N}'
    s2_tag = f'{tag_base}_s2_N{N}'
    log_file = os.path.join(RESULTS_DIR, f'{tag_base}_N{N}.log')

    with open(log_file, 'a') as lf:
        lf.write(f'\n=== {tag_base} N={N} started {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        lf.write(f'channel: {channel_name} meta={ch_meta}\n')
        lf.write(f'design proxy: {os.path.basename(design_path)}\n')
        lf.write(f'ku={ku} kv={kv} Ru={ku/N:.3f} Rv={kv/N:.3f}\n')

    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=d, hidden=hidden, n_layers=n_layers,
                           encoder_type=encoder_type, window_size=window_size,
                           gru_layers=gru_layers)
    print(f'\n{"="*60}\nchannel={channel_name} N={N} encoder={encoder_type} '
          f'iters={iters} params={model.count_parameters():,}\n{"="*60}')

    t0 = time.time()
    s1_best = train_stage1(
        model, channel, N, Au, Av, fu_set, iters=iters, batch=batch, lr=lr,
        ckpt_base=RESULTS_DIR, tag=s1_tag, log_file=log_file,
        eval_every=max(2000, iters // 10), eval_cw=eval_cw)
    s1_time = (time.time() - t0) / 60

    s1_best_ckpt = os.path.join(RESULTS_DIR, f'{s1_tag}_best.pt')
    if os.path.exists(s1_best_ckpt):
        sd = torch.load(s1_best_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])

    t1 = time.time()
    s2_best = train_stage2(
        model, channel, N, Au, Av, fv_set, iters=iters, batch=batch, lr=lr,
        ckpt_base=RESULTS_DIR, tag=s2_tag, log_file=log_file,
        eval_every=max(2000, iters // 10), eval_cw=eval_cw)
    s2_time = (time.time() - t1) / 60

    s2_best_ckpt = os.path.join(RESULTS_DIR, f'{s2_tag}_best.pt')
    if os.path.exists(s2_best_ckpt):
        sd = torch.load(s2_best_ckpt, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd['state_dict'])

    print(f'\n  Chained inference ({final_cw} codewords)...')
    chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set,
                           n_cw=final_cw, seed=777)
    print(f'  chained BLER={chained["bler_total"]:.4f} '
          f'(U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f})')

    mem_sc = None
    if memoryless_cw and memoryless_cw > 0:
        print(f'\n  Memoryless SC baseline ({memoryless_cw} codewords)...')
        t_ref = time.time()
        try:
            mem_sc = eval_memoryless_sc(channel, N, Au, Av, fu_1idx, fv_1idx,
                                        n_cw=memoryless_cw, seed=444)
            print(f'  memoryless SC BLER={mem_sc["bler_total"]:.4f} '
                  f'({(time.time()-t_ref)/60:.1f} min)')
        except Exception as e:
            print(f'  memoryless SC failed: {e}')
            mem_sc = {'error': str(e)}

    trellis = None
    if (not skip_trellis) and trellis_cw and trellis_cw > 0:
        print(f'\n  Trellis SC baseline ({trellis_cw} codewords)...')
        t_ref = time.time()
        try:
            trellis = eval_trellis_sc(channel, N, Au, Av, fu_1idx, fv_1idx,
                                      n_cw=trellis_cw, seed=555)
            print(f'  trellis SC BLER={trellis["bler_total"]:.4f} '
                  f'({(time.time()-t_ref)/60:.1f} min)')
        except Exception as e:
            print(f'  trellis SC failed: {e}')
            trellis = {'error': str(e)}

    result = {
        'channel': channel_name, 'channel_meta': ch_meta,
        'design_proxy': os.path.basename(design_path),
        'N': N, 'ku': ku, 'kv': kv,
        'encoder': encoder_type, 'window_size': window_size, 'd': d,
        'hidden': hidden, 'n_layers': n_layers, 'gru_layers': gru_layers,
        'stage1_best_bler': float(s1_best),
        'stage2_best_bler_true_x': float(s2_best),
        'chained': {k: (float(v) if isinstance(v, (float, int)) else v)
                    for k, v in chained.items()},
        'memoryless_sc': mem_sc,
        'trellis_sc': trellis,
        'stage1_iters': iters, 'stage2_iters': iters,
        'stage1_time_min': s1_time, 'stage2_time_min': s2_time,
        's1_ckpt': s1_best_ckpt, 's2_ckpt': s2_best_ckpt,
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--channel', type=str, required=True,
                   choices=['ge_mac', 'trapdoor_mac'])
    p.add_argument('--N', type=int, nargs='+', default=[16])
    p.add_argument('--iters', type=int, default=20000)
    p.add_argument('--encoder_type', type=str, default='window',
                   choices=['window', 'bigru'])
    p.add_argument('--window_size', type=int, default=2)
    p.add_argument('--gru_layers', type=int, default=1)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--eval_cw', type=int, default=200)
    p.add_argument('--final_cw', type=int, default=2000)
    p.add_argument('--trellis_cw', type=int, default=300)
    p.add_argument('--memoryless_cw', type=int, default=300)
    p.add_argument('--skip_trellis', action='store_true')
    p.add_argument('--save_json', type=str, default=None)
    p.add_argument('--ku', type=int, default=None, help='override rate: info bits for U')
    p.add_argument('--kv', type=int, default=None, help='override rate: info bits for V')
    p.add_argument('--tag_suffix', type=str, default='', help='extra tag suffix for file names')
    args = p.parse_args()

    # Apply rate override
    if args.ku is not None and args.kv is not None:
        for N in args.N:
            RATES[N] = (args.ku, args.kv)
            print(f'  rate override: N={N} ku={args.ku} kv={args.kv}')

    all_results = {}
    out_json = args.save_json or os.path.join(
        RESULTS_DIR, f'{args.channel}_{args.encoder_type}_extra_results.json')

    t_total = time.time()
    for N in args.N:
        res = run_one_N(
            channel_name=args.channel, N=N, iters=args.iters,
            encoder_type=args.encoder_type, window_size=args.window_size,
            batch=args.batch, lr=args.lr,
            d=args.d, hidden=args.hidden, n_layers=args.n_layers,
            gru_layers=args.gru_layers,
            eval_cw=args.eval_cw, final_cw=args.final_cw,
            trellis_cw=args.trellis_cw, memoryless_cw=args.memoryless_cw,
            skip_trellis=args.skip_trellis, tag_suffix=args.tag_suffix)
        all_results[str(N)] = res
        with open(out_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Saved incremental results: {out_json}')

    total_min = (time.time() - t_total) / 60
    print(f'\n{"="*60}\nAll N complete in {total_min:.1f} min')
    print(f'Results saved to {out_json}')
    print(f'\n{"N":<6}{"S1 BLER":<12}{"Chained":<12}{"MemSC":<12}{"TrellisSC":<12}')
    for Ns, r in all_results.items():
        s1 = r['stage1_best_bler']
        ch = r['chained']['bler_total']
        ms = r.get('memoryless_sc') or {}
        ts = r.get('trellis_sc') or {}
        ms_str = f"{ms.get('bler_total', float('nan')):.4f}" if isinstance(ms, dict) and 'bler_total' in ms else '-'
        ts_str = f"{ts.get('bler_total', float('nan')):.4f}" if isinstance(ts, dict) and 'bler_total' in ts else '-'
        print(f'{Ns:<6}{s1:<12.4f}{ch:<12.4f}{ms_str:<12}{ts_str:<12}')


if __name__ == '__main__':
    main()
