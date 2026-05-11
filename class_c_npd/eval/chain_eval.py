"""
End-to-end chained evaluation of the two-stage Class C NPD decoder.

Pipeline:
  1. Generate random (u_msg, v_msg) info bits
  2. Encode: x = polar_encode(u), y = polar_encode(v)
  3. Channel: z = MAC(x, y)
  4. Stage 1: decode u_hat from z using Stage 1 NPD on marginal features
  5. Reconstruct: x_hat = polar_encode(u_hat)
  6. Stage 2: decode v_hat from z using Stage 2 NPD on clean features z' = z - (1-2*x_hat)
  7. Block error = (u_hat != u) OR (v_hat != v) at info positions

Reports BLER with 95% Wilson confidence intervals.
"""
from __future__ import annotations
import os
import sys
import math
import json
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design


def wilson_ci(errs: int, n: int, z: float = 1.96):
    """95% Wilson confidence interval for a Bernoulli proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = errs / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_model_from_checkpoint(ckpt_path: str) -> NPDSingleUser:
    """Rebuild NPDSingleUser from a saved checkpoint."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model = NPDSingleUser(
        d=ckpt['d'], hidden=ckpt['hidden'],
        n_layers=ckpt['n_layers'], z_dim=ckpt['z_dim'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def chain_evaluate(
    stage1_ckpt: str,
    stage2_ckpt: str,
    channel_name: str,
    n: int,
    snr_db: float = 6.0,
    ku: int = None,
    kv: int = None,
    pe_threshold: float = 0.01,
    n_cw: int = 5000,
    batch: int = 8,
    seed: int = 999,
    verbose: bool = True,
) -> dict:
    """End-to-end chained evaluation with rigorous BLER."""
    N = 1 << n

    # Set up channel
    channel_kwargs = {}
    if channel_name in ('gmac', 'gaussian'):
        channel_kwargs['sigma2'] = 10 ** (-snr_db / 10)
    channel = build_channel(channel_name, **channel_kwargs)

    # Load design
    Au, Av, frozen_u, frozen_v, pe_u, pe_v = load_class_c_design(
        channel_name, n, snr_db=snr_db, ku=ku, kv=kv, pe_threshold=pe_threshold,
    )

    # Load models
    model_s1 = load_model_from_checkpoint(stage1_ckpt)
    model_s2 = load_model_from_checkpoint(stage2_ckpt)

    br = bit_reversal_perm(n)

    if verbose:
        print(f'Chained eval — {channel_name}, N={N}, ku={len(Au)}, kv={len(Av)}')
        print(f'  Stage 1 ckpt: {stage1_ckpt}')
        print(f'  Stage 2 ckpt: {stage2_ckpt}')
        print(f'  Eval codewords: {n_cw}')

    # Evaluation loop
    rng_bits = np.random.default_rng(seed)
    np.random.seed(seed)

    errs_total = 0
    errs_u = 0
    errs_v = 0
    errs_u_or_v_conditional = 0  # V errs given U was correct
    total = 0

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)

            # Generate random messages
            u_msg = np.zeros((actual, N), dtype=np.int8)
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Au:
                u_msg[:, p - 1] = rng_bits.integers(0, 2, actual)
            for p in Av:
                v_msg[:, p - 1] = rng_bits.integers(0, 2, actual)

            # Encode
            x_phys = polar_encode_batch(u_msg.astype(int))
            y_phys = polar_encode_batch(v_msg.astype(int))

            # Transmit through MAC
            z = channel.sample_z(x_phys.astype(int), y_phys.astype(int))

            # Stage 1: decode U on marginal channel
            features1 = channel.stage1_features(z)
            features1_npd = features1[..., br] if features1.ndim == 2 else features1[:, br, :]
            ft1 = torch.from_numpy(features1_npd).float()
            if ft1.dim() == 2:
                ft1 = ft1.unsqueeze(-1)
            emb1 = model_s1.encode_channel(ft1)
            u_hat = model_s1.decode(emb1, frozen_u)  # (B, N) in natural order

            # Reconstruct X_hat and compute residual channel
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)

            features2 = channel.stage2_features(z, x_hat.astype(int))
            features2_npd = features2[..., br] if features2.ndim == 2 else features2[:, br, :]
            ft2 = torch.from_numpy(features2_npd).float()
            if ft2.dim() == 2:
                ft2 = ft2.unsqueeze(-1)
            emb2 = model_s2.encode_channel(ft2)
            v_hat = model_s2.decode(emb2, frozen_v)

            # Count errors
            for i in range(actual):
                u_wrong = any(u_hat[i, p - 1].item() != u_msg[i, p - 1] for p in Au)
                v_wrong = any(v_hat[i, p - 1].item() != v_msg[i, p - 1] for p in Av)
                if u_wrong:
                    errs_u += 1
                if v_wrong:
                    errs_v += 1
                if u_wrong or v_wrong:
                    errs_total += 1
                if (not u_wrong) and v_wrong:
                    errs_u_or_v_conditional += 1

            total += actual
            if verbose and total % 500 == 0:
                bler = errs_total / total
                print(f'  [{total}/{n_cw}]  BLER={bler:.5f}  (U={errs_u}  V={errs_v}  both={errs_total})')

    bler_total = errs_total / n_cw
    bler_u = errs_u / n_cw
    bler_v = errs_v / n_cw
    ci_low, ci_high = wilson_ci(errs_total, n_cw)

    results = {
        'channel': channel_name,
        'N': N,
        'ku': len(Au),
        'kv': len(Av),
        'snr_db': snr_db,
        'n_cw': n_cw,
        'errs_total': errs_total,
        'errs_u': errs_u,
        'errs_v': errs_v,
        'bler_total': float(bler_total),
        'bler_u': float(bler_u),
        'bler_v': float(bler_v),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'stage1_ckpt': os.path.basename(stage1_ckpt),
        'stage2_ckpt': os.path.basename(stage2_ckpt),
    }

    if verbose:
        print()
        print('=== FINAL ===')
        print(f'  BLER total: {bler_total:.5f}  (95% CI: [{ci_low:.5f}, {ci_high:.5f}])')
        print(f'  BLER U only: {bler_u:.5f}  ({errs_u}/{n_cw})')
        print(f'  BLER V only: {bler_v:.5f}  ({errs_v}/{n_cw})')
        print(f'  V-errs given U correct: {errs_u_or_v_conditional}/{n_cw}')

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1_ckpt', type=str, required=True)
    parser.add_argument('--stage2_ckpt', type=str, required=True)
    parser.add_argument('--channel', type=str, default='gmac')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--snr', type=float, default=6.0)
    parser.add_argument('--ku', type=int, default=None)
    parser.add_argument('--kv', type=int, default=None)
    parser.add_argument('--pe_threshold', type=float, default=0.01)
    parser.add_argument('--n_cw', type=int, default=5000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--save_json', type=str, default=None)
    args = parser.parse_args()

    results = chain_evaluate(
        stage1_ckpt=args.stage1_ckpt, stage2_ckpt=args.stage2_ckpt,
        channel_name=args.channel, n=args.n, snr_db=args.snr,
        ku=args.ku, kv=args.kv, pe_threshold=args.pe_threshold,
        n_cw=args.n_cw, batch=args.batch, seed=args.seed,
    )

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or '.', exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to {args.save_json}')


if __name__ == '__main__':
    main()
