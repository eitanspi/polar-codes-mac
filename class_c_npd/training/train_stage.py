"""
Train a single stage (Stage 1 or Stage 2) of the Class C chained NPD.

Usage:
  python -m class_c_npd.training.train_stage --stage 1 --channel gmac --n 5
  python -m class_c_npd.training.train_stage --stage 2 --channel gmac --n 5

Stage 1 trains U on the MARGINAL channel (V is random interference).
Stage 2 trains V on the CLEAN conditional channel (X̂ is subtracted).

Both stages use the same NPDSingleUser model class, just with different
training data generators.
"""
from __future__ import annotations
import os
import sys
import math
import time
import json
import argparse
from typing import Callable

import numpy as np
import torch

# Add to_git_v2 root for 'polar' imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch

from class_c_npd.models.npd_single_user import NPDSingleUser, npd_encode
from class_c_npd.channels.mac_channel import build_channel, MACChannelC
from class_c_npd.channels.frozen_sets import load_class_c_design, design_summary


# ─── Training batch generators ───────────────────────────────────────────────

def generate_stage1_batch(channel: MACChannelC, N: int, Au: list, batch: int, rng: np.random.Generator, Av: list = None):
    """
    Stage 1 training batch (decode U on marginal channel).

    U message bits are random at info positions, 0 at frozen.
    V message bits are random at info positions (Av), 0 at frozen —
    matching the inference-time distribution of V.
    (The previous version used uniform V over {0,1}^N which gave a
    different joint distribution and created a train-inference gap.)

    If Av is None, falls back to uniform V (for backward compat).
    """
    u_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au:
        u_msg[:, p - 1] = rng.integers(0, 2, batch)
    x_phys = polar_encode_batch(u_msg.astype(int))

    # V matches inference: random bits at info positions, 0 at frozen.
    if Av is not None:
        v_msg = np.zeros((batch, N), dtype=np.int8)
        for p in Av:
            v_msg[:, p - 1] = rng.integers(0, 2, batch)
        y_phys = polar_encode_batch(v_msg.astype(int))
    else:
        v_random = rng.integers(0, 2, (batch, N)).astype(np.int8)
        y_phys = polar_encode_batch(v_random.astype(int))

    z = channel.sample_z(x_phys.astype(int), y_phys.astype(int))
    features = channel.stage1_features(z)

    # NPD tree order: bit-reverse the features and the codeword targets
    from polar.encoder import bit_reversal_perm
    n = int(math.log2(N))
    br = bit_reversal_perm(n)

    features_npd = features[..., br] if features.ndim == 2 else features[:, br, :]
    x_npd = x_phys[:, br]  # codeword in NPD tree order == npd_encode(u_msg)

    return u_msg, features_npd, x_npd


def generate_stage2_batch(channel: MACChannelC, N: int, Av: list, batch: int, rng: np.random.Generator, Au: list = None):
    """
    Stage 2 training batch (decode V on clean channel given X).

    V message bits random at info, 0 at frozen.
    X is drawn from the same distribution as inference: random at Au
    info positions, 0 at frozen. If Au is None, falls back to uniform X.
    (The previous version used fully uniform X which mismatches inference.)
    """
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Av:
        v_msg[:, p - 1] = rng.integers(0, 2, batch)

    y_phys = polar_encode_batch(v_msg.astype(int))

    # X matches inference distribution
    if Au is not None:
        u_msg = np.zeros((batch, N), dtype=np.int8)
        for p in Au:
            u_msg[:, p - 1] = rng.integers(0, 2, batch)
        x_phys = polar_encode_batch(u_msg.astype(int))
    else:
        u_random = rng.integers(0, 2, (batch, N)).astype(np.int8)
        x_phys = polar_encode_batch(u_random.astype(int))

    z = channel.sample_z(x_phys.astype(int), y_phys.astype(int))
    features = channel.stage2_features(z, x_phys.astype(int))

    from polar.encoder import bit_reversal_perm
    n = int(math.log2(N))
    br = bit_reversal_perm(n)

    features_npd = features[..., br] if features.ndim == 2 else features[:, br, :]
    y_npd = y_phys[:, br]

    return v_msg, features_npd, y_npd


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_stage(model: NPDSingleUser, channel: MACChannelC, N: int,
                   info_positions: list, frozen_set: set,
                   gen_fn: Callable, n_cw: int = 500, seed: int = 999,
                   other_info: list = None) -> float:
    """Evaluate BLER on the stage's own channel.

    other_info: for stage 1, pass Av so the V interference matches inference.
                for stage 2, pass Au so the X side info matches inference.
    """
    model.eval()
    errs = 0
    total = 0
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    bs = 32
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            if other_info is not None:
                u_true, features_npd, _ = gen_fn(channel, N, info_positions, actual, rng, other_info)
            else:
                u_true, features_npd, _ = gen_fn(channel, N, info_positions, actual, rng)
            ft = torch.from_numpy(features_npd).float()
            if ft.dim() == 2:
                ft = ft.unsqueeze(-1)
            emb = model.encode_channel(ft)
            u_dec = model.decode(emb, frozen_set)
            for i in range(actual):
                if any(u_dec[i, p - 1].item() != u_true[i, p - 1] for p in info_positions):
                    errs += 1
            total += actual
    model.train()
    return errs / n_cw


# ─── Training loop ───────────────────────────────────────────────────────────

def train_stage(stage: int, channel_name: str, n: int,
                snr_db: float = 6.0, d: int = 16, hidden: int = 64,
                n_layers: int = 2, batch: int = 32, lr: float = 3e-4,
                total_iters: int = 30000, eval_every: int = 2000,
                ku: int = None, kv: int = None, pe_threshold: float = 0.01,
                save_dir: str = None, tag: str = None, seed: int = None):
    """Train a single stage and save the best checkpoint."""
    N = 1 << n

    # Set up channel
    channel_kwargs = {}
    if channel_name in ('gmac', 'gaussian'):
        channel_kwargs['sigma2'] = 10 ** (-snr_db / 10)
    channel = build_channel(channel_name, **channel_kwargs)

    # Load design
    Au, Av, frozen_u, frozen_v, pe_u, pe_v = load_class_c_design(
        channel_name, n, snr_db=snr_db, ku=ku, kv=kv, pe_threshold=pe_threshold)

    print('=' * 60, flush=True)
    print(f'Stage {stage} training — channel={channel_name}, N={N}', flush=True)
    if channel_name in ('gmac', 'gaussian'):
        print(f'  SNR={snr_db} dB, sigma2={channel_kwargs["sigma2"]:.4f}', flush=True)
    print(design_summary(Au, Av, pe_u, pe_v), flush=True)
    print('=' * 60, flush=True)

    # Pick the right generator and info set for this stage
    if stage == 1:
        gen_fn = generate_stage1_batch
        info_positions = Au
        frozen_set = frozen_u
        z_dim = channel.stage1_feature_dim
        other_info = Av   # V's info positions — used so V interference at
                          # training matches inference-time distribution
    elif stage == 2:
        gen_fn = generate_stage2_batch
        info_positions = Av
        frozen_set = frozen_v
        z_dim = channel.stage2_feature_dim
        other_info = Au   # U's info positions — similarly
    else:
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    # Model
    if seed is not None:
        torch.manual_seed(seed)
    model = NPDSingleUser(d=d, hidden=hidden, n_layers=n_layers, z_dim=z_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f'Model: d={d} hidden={hidden} n_layers={n_layers} z_dim={z_dim} '
          f'params={model.count_parameters():,}', flush=True)

    rng = np.random.default_rng(seed)
    t0 = time.time()
    losses = []
    best_bler = 1.0

    # Save path
    if save_dir is None:
        save_dir = os.path.join(_ROOT, 'class_c_npd', 'results')
    os.makedirs(save_dir, exist_ok=True)
    if tag is None:
        tag = f'{channel_name}_stage{stage}_N{N}'
    ckpt_path = os.path.join(save_dir, f'{tag}_best.pt')

    model.train()
    for it in range(1, total_iters + 1):
        _, features_npd, cw_npd = gen_fn(channel, N, info_positions, batch, rng, other_info)
        ft = torch.from_numpy(features_npd).float()
        if ft.dim() == 2:
            ft = ft.unsqueeze(-1)
        cw = torch.from_numpy(cw_npd).long()

        emb = model.encode_channel(ft)
        loss = model.fast_ce(emb, cw)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % eval_every == 0:
            bler = evaluate_stage(model, channel, N, info_positions, frozen_set, gen_fn, 500, seed=999, other_info=other_info)
            avg_loss = float(np.mean(losses[-500:]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                torch.save({'state_dict': model.state_dict(),
                            'd': d, 'hidden': hidden, 'n_layers': n_layers, 'z_dim': z_dim,
                            'channel': channel_name, 'stage': stage, 'N': N,
                            'Au': Au, 'Av': Av}, ckpt_path)
                marker = ' *BEST*'
            print(f'  [{it:>6}/{total_iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}', flush=True)

    elapsed = (time.time() - t0) / 60
    print(f'\nStage {stage} done in {elapsed:.1f} min. Best BLER: {best_bler:.4f}', flush=True)
    print(f'Checkpoint: {ckpt_path}', flush=True)

    return best_bler, ckpt_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2])
    parser.add_argument('--channel', type=str, default='gmac',
                        choices=['gmac', 'bemac', 'abnmac'])
    parser.add_argument('--n', type=int, default=5, help='log2(N), default 5 => N=32')
    parser.add_argument('--snr', type=float, default=6.0, help='SNR in dB for GMAC')
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--iters', type=int, default=30000)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--ku', type=int, default=None)
    parser.add_argument('--kv', type=int, default=None)
    parser.add_argument('--pe_threshold', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_stage(
        stage=args.stage, channel_name=args.channel, n=args.n, snr_db=args.snr,
        d=args.d, hidden=args.hidden, n_layers=args.n_layers,
        batch=args.batch, lr=args.lr, total_iters=args.iters,
        eval_every=args.eval_every, ku=args.ku, kv=args.kv,
        pe_threshold=args.pe_threshold, seed=args.seed,
    )


if __name__ == '__main__':
    main()
