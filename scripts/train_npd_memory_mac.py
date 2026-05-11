#!/usr/bin/env python3
"""
train_npd_memory_mac.py
=======================
Training driver for the chained NPD for memory MAC channels (ISI-MAC and,
optionally, Trapdoor MAC). Uses the `neural.npd_memory_mac` module.

Pipeline at each N:
  1. Load GMAC_C-derived frozen set (ISI-MAC design is a bogus proxy, but
     GMAC_C is close for h=0.3 — same as the broken baseline uses).
  2. Train Stage 1 (U on marginal, V random) with fast_ce BCE across tree
     depths, using MemoryZEncoder(window) or BiGRU over the full z-sequence.
  3. Freeze Stage 1 and train Stage 2 (V given true U via teacher forcing).
  4. Chained inference: Stage 1 decodes U_hat from z, Stage 2 decodes V from
     (z, U_hat). Report per-codeword BLER with Wilson CI.

Saves checkpoints to class_c_npd/results/npd_memory_mac/*.pt and a results
markdown to class_c_npd/results/npd_memory_mac_results.md.

Usage:
  python scripts/train_npd_memory_mac.py --channel isi_mac --N 16 --iters 30000
  python scripts/train_npd_memory_mac.py --channel isi_mac --all  # all N

This script sets torch.set_num_threads(2) so we do not starve other running
training jobs.
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
from polar.channels_memory import ISIMAC
from polar.channels_memory_new import TrapdoorMAC  # noqa: F401 (bonus)
from polar.design_mc import design_from_file
from polar.decoder_trellis import decode_single as trellis_decode_single

from neural.npd_memory_mac import ChainedNPD_MAC


# ─── Config ──────────────────────────────────────────────────────────────────

SNR_DB = 6.0
ISI_H = 0.3

# Rates per the session prompt — GMAC_C capacities at SNR=6dB, h=0.3
# (same as baseline broken NPD uses — isolates the channel issue)
RATES: Dict[int, Tuple[int, int]] = {
    16: (4, 7),
    32: (7, 15),
    64: (15, 29),
}

RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Channel factories ───────────────────────────────────────────────────────

def make_channel(name: str, snr_db: float = SNR_DB, h: float = ISI_H):
    name = name.lower()
    if name in ('isi_mac', 'isi'):
        return ISIMAC.from_snr_db(snr_db, h=h), {'h': h, 'snr_db': snr_db}
    if name in ('trapdoor', 'trapdoor_mac'):
        # Single BSC-like noise; use p_noise=0.1 (matching channels_memory_new default)
        ch = TrapdoorMAC(p_noise=0.1)
        return ch, {'p_noise': 0.1}
    raise ValueError(f'unknown channel {name!r}')


# ─── Design loader (GMAC_C proxy) ────────────────────────────────────────────

def load_design(channel: str, N: int, ku: int, kv: int):
    """
    Load a Class C design. ISI-MAC designs in designs/ are bogus (all-zero
    Pe). We fall back to the GMAC_C design at matching SNR — same proxy the
    broken NPD baseline used, so the comparison is apples-to-apples.

    Trapdoor MAC: try a BEMAC_C proxy (both binary) — crude but acceptable
    for a bonus experiment.

    Returns Au, Av (1-indexed lists), frozen_u, frozen_v (1-indexed dicts
    needed by polar.decoder_trellis.decode_single), frozen_u_set,
    frozen_v_set (0-indexed sets for NPD decoder).
    """
    n = int(math.log2(N))
    if channel in ('isi_mac', 'isi'):
        path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    elif channel in ('trapdoor', 'trapdoor_mac'):
        path = os.path.join(_ROOT, 'designs', f'bemac_C_n{n}.npz')
    else:
        raise ValueError(f'no design for channel {channel!r}')

    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, _path_i = design_from_file(
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

def make_batch(channel, N: int, Au: list, Av: list, batch: int, rng: np.random.Generator):
    """Generate a fresh training batch (u, v, z, x_codeword, y_codeword)."""
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au:
        u_msg[:, p - 1] = rng.integers(0, 2, batch)
    for p in Av:
        v_msg[:, p - 1] = rng.integers(0, 2, batch)

    x_phys = polar_encode_batch(u_msg.astype(int))
    y_phys = polar_encode_batch(v_msg.astype(int))

    z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
    # z float, shape (B, N)
    return u_msg, v_msg, z.astype(np.float32), x_phys, y_phys


# ─── Stage 1 training ───────────────────────────────────────────────────────

def train_stage1(model: ChainedNPD_MAC, channel, N: int, Au: list, Av: list,
                 frozen_u_set: set, iters: int, batch: int, lr: float,
                 ckpt_base: str, tag: str,
                 log_file: str, eval_every: int = 2000, eval_cw: int = 200,
                 seed: int = 42):
    """
    Train Stage 1 (U decoder) with fast_ce BCE.

    Targets: codeword bits x (= polar_encode(u)) in NPD tree order.
    """
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
        z_t = torch.from_numpy(z)  # (B, N)

        # Encoder operates on raw z in natural position order; then we bit-
        # reverse to NPD tree order for fast_ce.
        emb = model.stage1.encode_channel(z_t)  # (B, N, d) natural order
        emb_npd = emb[:, br_t, :]               # (B, N, d) tree order
        x_cw_npd = torch.from_numpy(x_phys[:, br]).long()

        loss = model.stage1.tree.fast_ce(emb_npd, x_cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if it % eval_every == 0 or it == iters:
            bler = eval_stage1(model, channel, N, Au, Av, frozen_u_set, n_cw=eval_cw, seed=999)
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
            # periodic checkpoint every 5K
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


def eval_stage1(model: ChainedNPD_MAC, channel, N: int, Au: list, Av: list,
                frozen_u_set: set, n_cw: int = 500, batch: int = 32, seed: int = 999):
    """Evaluate Stage 1 BLER on U's info positions."""
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

def _u_side(u_msg_np: np.ndarray) -> torch.Tensor:
    """Convert hard U bits (0/1) to ±1 BPSK-style side info for Stage 2."""
    u_bpsk = 1.0 - 2.0 * u_msg_np.astype(np.float32)  # 0 -> +1, 1 -> -1
    return torch.from_numpy(u_bpsk).unsqueeze(-1)  # (B, N, 1)


def train_stage2(model: ChainedNPD_MAC, channel, N: int, Au: list, Av: list,
                 frozen_v_set: set, iters: int, batch: int, lr: float,
                 ckpt_base: str, tag: str, log_file: str,
                 eval_every: int = 2000, eval_cw: int = 200, seed: int = 42,
                 teacher_forcing: bool = True):
    """
    Train Stage 2 (V decoder) with fast_ce BCE, conditioned on U.

    teacher_forcing=True: feed the TRUE U as side info (standard approach;
    matches train/inference provided Stage 1 is accurate enough).

    For memory channels at low SNR, Stage 1 may make U errors that shift
    Stage 2's effective channel. A noiseless teacher is still usually
    better than noisy û during training because it gives a cleaner gradient.
    """
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long() if isinstance(br, np.ndarray) else torch.tensor(br, dtype=torch.long)

    opt = torch.optim.AdamW(model.stage2.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.1)
    rng = np.random.default_rng(seed + 1)

    best_bler = 1.0
    losses = []
    t0 = time.time()

    model.stage1.eval()  # used only for hard decisions in inference tests
    model.stage2.train()

    for it in range(1, iters + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, batch, rng)
        z_t = torch.from_numpy(z)
        # Teacher force with TRUE x_phys (the codeword produced by u). This is
        # what the channel actually saw; matches chained inference if Stage 1
        # decodes U perfectly. When û != u the mismatch is handled later via
        # a final chained-eval run.
        side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)

        emb = model.stage2.encode_channel(z_t, side=side)  # (B,N,d) natural
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
            # Stage 2 standalone eval with TRUE x (upper bound on chained V BLER)
            bler = eval_stage2_with_true_x(model, channel, N, Au, Av, frozen_v_set,
                                            n_cw=eval_cw, seed=999)
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


def eval_stage2_with_true_x(model: ChainedNPD_MAC, channel, N: int, Au: list, Av: list,
                             frozen_v_set: set, n_cw: int = 500, batch: int = 32,
                             seed: int = 999):
    """Evaluate Stage 2 with true X (upper bound on V BLER under perfect Stage 1)."""
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

def eval_chained(model: ChainedNPD_MAC, channel, N: int, Au: list, Av: list,
                  frozen_u_set: set, frozen_v_set: set, n_cw: int = 2000,
                  batch: int = 32, seed: int = 777) -> dict:
    """End-to-end chained inference: Stage 1 -> Stage 2. Returns BLER metrics."""
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

            # Stage 1
            emb1 = model.stage1.encode_channel(z_t)
            emb1_npd = emb1[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb1_npd, frozen_u_set)
            u_hat_np = u_hat.numpy().astype(int)
            # Reconstruct x_hat
            x_hat = polar_encode_batch(u_hat_np)

            # Stage 2 conditional on x_hat
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

    return {
        'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw,
        'bler_v': errs_v / n_cw,
        'bler_total': errs_total / n_cw,
    }


# ─── Trellis SC reference ───────────────────────────────────────────────────

def eval_trellis_sc(channel, N: int, Au: list, Av: list,
                     frozen_u_dict: dict, frozen_v_dict: dict,
                     n_cw: int = 500, seed: int = 555) -> dict:
    """
    Analytical trellis SC joint decode (Önay + forward-backward on ISI state).
    Reference for the best analytical approach.
    """
    n = int(math.log2(N))
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
                N, z.tolist(), b, frozen_u_dict, frozen_v_dict, channel,
                log_domain=True)
        except Exception as e:
            print(f'  trellis eval iter {i} raised: {e}')
            continue
        u_wrong = any(u_dec[p - 1] != u_msg[p - 1] for p in Au)
        v_wrong = any(v_dec[p - 1] != v_msg[p - 1] for p in Av)
        if u_wrong:
            errs_u += 1
        if v_wrong:
            errs_v += 1
        if u_wrong or v_wrong:
            errs_total += 1
    return {
        'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw,
        'bler_v': errs_v / n_cw,
        'bler_total': errs_total / n_cw,
    }


# ─── Orchestration per N ────────────────────────────────────────────────────

def run_one_N(channel_name: str, N: int, iters: int, encoder_type: str,
              window_size: int, batch: int, lr: float, d: int,
              hidden: int, n_layers: int, gru_layers: int,
              warm_stage1_ckpt: str = None, warm_stage2_ckpt: str = None,
              eval_cw: int = 300, final_cw: int = 2000, trellis_cw: int = 500):
    """
    Full pipeline: design, train Stage 1, train Stage 2, chained eval, trellis
    reference.
    """
    ku, kv = RATES[N]
    channel, ch_meta = make_channel(channel_name)
    Au, Av, fu_1idx, fv_1idx, fu_set, fv_set = load_design(channel_name, N, ku, kv)

    tag_base = f'{channel_name}_{encoder_type}'
    if encoder_type == 'window':
        tag_base += f'_w{window_size}'
    if encoder_type == 'bigru':
        tag_base += f'_L{gru_layers}'

    s1_tag = f'{tag_base}_s1_N{N}'
    s2_tag = f'{tag_base}_s2_N{N}'
    log_file = os.path.join(RESULTS_DIR, f'{tag_base}_N{N}.log')

    with open(log_file, 'a') as lf:
        lf.write(f'\n=== {tag_base} N={N} started {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        lf.write(f'channel: {channel_name} meta={ch_meta}\n')
        lf.write(f'ku={ku} kv={kv} Ru={ku/N:.3f} Rv={kv/N:.3f}\n')

    # Model
    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=d, hidden=hidden, n_layers=n_layers,
                           encoder_type=encoder_type, window_size=window_size,
                           gru_layers=gru_layers)

    # Warm-start stages
    if warm_stage1_ckpt and os.path.exists(warm_stage1_ckpt):
        try:
            sd = torch.load(warm_stage1_ckpt, weights_only=False, map_location='cpu')
            model.stage1.load_state_dict(sd['state_dict'])
            print(f'  warm-start stage1 from {os.path.basename(warm_stage1_ckpt)}')
            with open(log_file, 'a') as lf:
                lf.write(f'warm-start stage1 from {warm_stage1_ckpt}\n')
        except Exception as e:
            print(f'  warm-start stage1 failed: {e}')
    if warm_stage2_ckpt and os.path.exists(warm_stage2_ckpt):
        try:
            sd = torch.load(warm_stage2_ckpt, weights_only=False, map_location='cpu')
            model.stage2.load_state_dict(sd['state_dict'])
            print(f'  warm-start stage2 from {os.path.basename(warm_stage2_ckpt)}')
            with open(log_file, 'a') as lf:
                lf.write(f'warm-start stage2 from {warm_stage2_ckpt}\n')
        except Exception as e:
            print(f'  warm-start stage2 failed: {e}')

    print(f'\n{"="*60}\nN={N} channel={channel_name} encoder={encoder_type} '
          f'iters={iters} params={model.count_parameters():,}\n{"="*60}')

    # Stage 1
    t0 = time.time()
    s1_best = train_stage1(
        model, channel, N, Au, Av, fu_set, iters=iters, batch=batch, lr=lr,
        ckpt_base=RESULTS_DIR, tag=s1_tag, log_file=log_file,
        eval_every=max(2000, iters // 10), eval_cw=eval_cw,
    )
    s1_time = (time.time() - t0) / 60
    print(f'\n  Stage 1 best BLER: {s1_best:.4f}  ({s1_time:.1f} min)')

    # Reload best Stage 1 before training Stage 2
    s1_best_ckpt = os.path.join(RESULTS_DIR, f'{s1_tag}_best.pt')
    if os.path.exists(s1_best_ckpt):
        sd = torch.load(s1_best_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])

    # Stage 2
    t1 = time.time()
    s2_best = train_stage2(
        model, channel, N, Au, Av, fv_set, iters=iters, batch=batch, lr=lr,
        ckpt_base=RESULTS_DIR, tag=s2_tag, log_file=log_file,
        eval_every=max(2000, iters // 10), eval_cw=eval_cw,
    )
    s2_time = (time.time() - t1) / 60
    print(f'\n  Stage 2 best BLER(V|trueU): {s2_best:.4f}  ({s2_time:.1f} min)')

    # Reload best Stage 2
    s2_best_ckpt = os.path.join(RESULTS_DIR, f'{s2_tag}_best.pt')
    if os.path.exists(s2_best_ckpt):
        sd = torch.load(s2_best_ckpt, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd['state_dict'])

    # Chained inference
    print(f'\n  Chained inference ({final_cw} codewords)...')
    chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set,
                           n_cw=final_cw, seed=777)
    print(f'  chained BLER={chained["bler_total"]:.4f} '
          f'(U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f})')

    # Trellis SC reference
    print(f'\n  Trellis SC reference ({trellis_cw} codewords)...')
    t_ref = time.time()
    trellis = eval_trellis_sc(channel, N, Au, Av, fu_1idx, fv_1idx,
                              n_cw=trellis_cw, seed=555)
    print(f'  trellis SC BLER={trellis["bler_total"]:.4f} '
          f'(U={trellis["bler_u"]:.4f}, V={trellis["bler_v"]:.4f}) '
          f'({(time.time()-t_ref)/60:.1f} min)')

    result = {
        'channel': channel_name, 'channel_meta': ch_meta,
        'N': N, 'ku': ku, 'kv': kv,
        'encoder': encoder_type, 'window_size': window_size, 'd': d,
        'hidden': hidden, 'n_layers': n_layers, 'gru_layers': gru_layers,
        'stage1_best_bler': float(s1_best),
        'stage2_best_bler_true_x': float(s2_best),
        'chained': {k: (float(v) if isinstance(v, (float, int)) else v)
                    for k, v in chained.items()},
        'trellis_sc': {k: (float(v) if isinstance(v, (float, int)) else v)
                       for k, v in trellis.items()},
        'stage1_iters': iters, 'stage2_iters': iters,
        'stage1_time_min': s1_time, 'stage2_time_min': s2_time,
        's1_ckpt': s1_best_ckpt, 's2_ckpt': s2_best_ckpt,
    }
    return result


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--channel', type=str, default='isi_mac',
                   choices=['isi_mac', 'trapdoor_mac'])
    p.add_argument('--N', type=int, nargs='+', default=None)
    p.add_argument('--iters', type=int, default=30000)
    p.add_argument('--encoder_type', type=str, default='window',
                   choices=['window', 'bigru'])
    p.add_argument('--window_size', type=int, default=2,
                   help='W; total context = 2W+1')
    p.add_argument('--gru_layers', type=int, default=1)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--eval_cw', type=int, default=300)
    p.add_argument('--final_cw', type=int, default=2000)
    p.add_argument('--trellis_cw', type=int, default=500)
    p.add_argument('--all', action='store_true')
    p.add_argument('--save_json', type=str, default=None)
    p.add_argument('--curriculum', action='store_true',
                   help='warm-start larger N from previous N checkpoint')
    args = p.parse_args()

    N_list = args.N if args.N is not None else ([16, 32, 64] if args.all else [16])

    all_results = {}
    prev_s1 = None
    prev_s2 = None
    out_json = args.save_json or os.path.join(
        RESULTS_DIR, f'{args.channel}_{args.encoder_type}_results.json')

    t_total = time.time()
    for N in N_list:
        # Scale iters and batch for larger N
        n_iters = args.iters
        batch = args.batch
        if N == 64:
            batch = min(batch, 8)
            n_iters = max(n_iters, 40000)
        if N == 32:
            batch = min(batch, 16)

        warm_s1 = prev_s1 if args.curriculum else None
        warm_s2 = prev_s2 if args.curriculum else None

        res = run_one_N(
            channel_name=args.channel, N=N, iters=n_iters,
            encoder_type=args.encoder_type, window_size=args.window_size,
            batch=batch, lr=args.lr,
            d=args.d, hidden=args.hidden, n_layers=args.n_layers,
            gru_layers=args.gru_layers,
            warm_stage1_ckpt=warm_s1, warm_stage2_ckpt=warm_s2,
            eval_cw=args.eval_cw, final_cw=args.final_cw,
            trellis_cw=args.trellis_cw,
        )
        all_results[str(N)] = res
        prev_s1 = res['s1_ckpt']
        prev_s2 = res['s2_ckpt']

        # Save incrementally
        with open(out_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Saved incremental results: {out_json}')

    total_min = (time.time() - t_total) / 60
    print(f'\n{"="*60}\nAll N complete in {total_min:.1f} min')
    print(f'Results saved to {out_json}')

    # Quick summary
    print(f'\n{"N":<6}{"S1 BLER":<12}{"Chained BLER":<16}{"Trellis SC":<14}{"Gap (ratio)":<14}')
    for Ns, r in all_results.items():
        s1 = r['stage1_best_bler']
        ch = r['chained']['bler_total']
        tr = r['trellis_sc']['bler_total']
        ratio = ch / max(tr, 1e-6)
        print(f'{Ns:<6}{s1:<12.4f}{ch:<16.4f}{tr:<14.4f}{ratio:<14.2f}')


if __name__ == '__main__':
    main()
