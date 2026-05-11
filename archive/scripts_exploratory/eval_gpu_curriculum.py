#!/usr/bin/env python3
"""
eval_gpu_curriculum.py
======================
Evaluate GPU curriculum checkpoints with CORRECT info-only BLER.
Also trains Stage 2 and runs chained eval.
"""
from __future__ import annotations
import math
import os
import sys
import time
import numpy as np
import torch

torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from neural.npd_memory_mac import ChainedNPD_MAC

SNR_DB = 6.0
ISI_H = 0.3

RATES = {
    16:  (4, 7),
    32:  (7, 15),
    64:  (15, 29),
    128: (30, 58),
    256: (59, 117),
}

RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, _path_i = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_set = {p - 1 for p in range(1, N + 1) if p not in Au}
    frozen_v_set = {p - 1 for p in range(1, N + 1) if p not in Av}
    return Au, Av, frozen_u_set, frozen_v_set


def make_channel():
    return ISIMAC.from_snr_db(SNR_DB, h=ISI_H)


def make_batch(channel, N, Au, Av, batch, rng):
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


def eval_stage1_info_only(model, channel, N, Au, frozen_u_set, n_cw=2000, batch=16, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    Av = sorted(set(range(1, N+1)) - {p for p in range(1, N+1) if p-1 in frozen_u_set})  # dummy
    # Use full Av for batch generation (doesn't matter for U eval)
    ku, kv = RATES[N]
    _, Av_real, _, _ = load_design(N, ku, kv)

    model.stage1.eval()
    rng = np.random.default_rng(seed)
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, _, z, _, _ = make_batch(channel, N, Au, Av_real, actual, rng)
            z_t = torch.from_numpy(z)
            emb = model.stage1.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb_npd, frozen_u_set)
            for i in range(actual):
                if any(int(u_hat[i, p - 1].item()) != int(u_msg[i, p - 1]) for p in Au):
                    errs += 1
            total += actual
    return errs / total


def load_gpu_checkpoint_stage1(ckpt_path, d=16, hidden=64, n_layers=2, gru_layers=1):
    model = ChainedNPD_MAC(d=d, hidden=hidden, n_layers=n_layers,
                           encoder_type="bigru", gru_layers=gru_layers)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.stage1.load_state_dict(sd)
    model.eval()
    return model


def train_stage2(model, channel, N, Au, Av, frozen_v_set, iters=30000, batch=32,
                 lr=3e-4, tag="gpu_curriculum", seed=42):
    """Train Stage 2 with frozen Stage 1, teacher forcing with true U."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()

    # Freeze stage 1
    for p in model.stage1.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(model.stage2.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.1)
    rng = np.random.default_rng(seed + 1)

    best_bler = 1.0
    losses = []
    t0 = time.time()
    model.stage2.train()

    for it in range(1, iters + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, batch, rng)
        z_t = torch.from_numpy(z)
        # Teacher forcing: true x as side info
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

        if it % 2000 == 0 or it == iters:
            # Eval with true x
            bler = eval_stage2_true_x(model, channel, N, Au, Av, frozen_v_set, n_cw=200)
            avg_loss = float(np.mean(losses[-min(200, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                ckpt_path = os.path.join(RESULTS_DIR, f'{tag}_s2_N{N}_best.pt')
                torch.save({'state_dict': model.stage2.state_dict(), 'N': N}, ckpt_path)
                marker = ' *BEST*'
            if it % 5000 == 0 or it == iters:
                torch.save({'state_dict': model.stage2.state_dict(), 'N': N},
                           os.path.join(RESULTS_DIR, f'{tag}_s2_N{N}_final.pt'))
            print(f'  [S2 {it:>6}/{iters}] loss={avg_loss:.4f} BLER(V|trueU)={bler:.4f} '
                  f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}', flush=True)

    return best_bler


def eval_stage2_true_x(model, channel, N, Au, Av, frozen_v_set, n_cw=500, batch=16, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage2.eval()
    rng = np.random.default_rng(seed)
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
    return errs / total


def eval_chained(model, channel, N, Au, Av, frozen_u_set, frozen_v_set,
                 n_cw=2000, batch=16, seed=777):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage1.eval()
    model.stage2.eval()
    rng = np.random.default_rng(seed)

    errs_u = errs_v = errs_total = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)

            # Stage 1: decode U
            emb1 = model.stage1.encode_channel(z_t)
            emb1_npd = emb1[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb1_npd, frozen_u_set)
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)

            # Stage 2: decode V given u_hat
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
        'bler_u': errs_u / n_cw,
        'bler_v': errs_v / n_cw,
        'bler_total': errs_total / n_cw,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['eval_s1', 'train_s2', 'eval_chained', 'full_pipeline'],
                        default='eval_s1')
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--ckpt', type=str, default=None, help='Stage 1 checkpoint path')
    parser.add_argument('--s2_ckpt', type=str, default=None, help='Stage 2 checkpoint path')
    parser.add_argument('--n_cw', type=int, default=2000)
    parser.add_argument('--s2_iters', type=int, default=30000)
    parser.add_argument('--s2_batch', type=int, default=32)
    parser.add_argument('--s2_lr', type=float, default=3e-4)
    args = parser.parse_args()

    N = args.N
    ku, kv = RATES[N]
    Au, Av, frozen_u_set, frozen_v_set = load_design(N, ku, kv)
    channel = make_channel()

    ckpt = args.ckpt or os.path.join(RESULTS_DIR, f'gpu_curriculum_s1_N{N}_final.pt')

    if args.mode == 'eval_s1':
        print(f"Evaluating Stage 1 for N={N} (ku={ku}, kv={kv})")
        print(f"Checkpoint: {ckpt}")
        model = load_gpu_checkpoint_stage1(ckpt)
        bler = eval_stage1_info_only(model, channel, N, Au, frozen_u_set, n_cw=args.n_cw)
        print(f"N={N} Stage 1 BLER (info-only): {bler:.4f} ({int(bler*args.n_cw)}/{args.n_cw})")

    elif args.mode == 'train_s2':
        print(f"Training Stage 2 for N={N}")
        model = load_gpu_checkpoint_stage1(ckpt)
        train_stage2(model, channel, N, Au, Av, frozen_v_set,
                     iters=args.s2_iters, batch=args.s2_batch, lr=args.s2_lr)

    elif args.mode == 'eval_chained':
        print(f"Chained eval for N={N}")
        model = load_gpu_checkpoint_stage1(ckpt)
        s2_ckpt = args.s2_ckpt or os.path.join(RESULTS_DIR, f'gpu_curriculum_s2_N{N}_best.pt')
        sd = torch.load(s2_ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.stage2.load_state_dict(sd)
        res = eval_chained(model, channel, N, Au, Av, frozen_u_set, frozen_v_set, n_cw=args.n_cw)
        print(f"N={N} Chained: BLER_U={res['bler_u']:.4f} BLER_V={res['bler_v']:.4f} "
              f"BLER_total={res['bler_total']:.4f}")

    elif args.mode == 'full_pipeline':
        print(f"=== Full pipeline for N={N} ===")
        print(f"Stage 1 checkpoint: {ckpt}")
        model = load_gpu_checkpoint_stage1(ckpt)

        # Eval Stage 1
        bler_s1 = eval_stage1_info_only(model, channel, N, Au, frozen_u_set, n_cw=args.n_cw)
        print(f"\nN={N} Stage 1 BLER (info-only): {bler_s1:.4f}")

        # Train Stage 2
        print(f"\nTraining Stage 2...")
        best_s2 = train_stage2(model, channel, N, Au, Av, frozen_v_set,
                                iters=args.s2_iters, batch=args.s2_batch, lr=args.s2_lr)
        print(f"\nN={N} Stage 2 best BLER (V|trueU): {best_s2:.4f}")

        # Load best S2 and run chained eval
        s2_ckpt = os.path.join(RESULTS_DIR, f'gpu_curriculum_s2_N{N}_best.pt')
        sd = torch.load(s2_ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.stage2.load_state_dict(sd)
        res = eval_chained(model, channel, N, Au, Av, frozen_u_set, frozen_v_set, n_cw=args.n_cw)
        print(f"\nN={N} Chained: BLER_U={res['bler_u']:.4f} BLER_V={res['bler_v']:.4f} "
              f"BLER_total={res['bler_total']:.4f}")
