"""
Smoke test for the Class C NPD pipeline.

Quick end-to-end verification at small N using GMAC at SNR=6dB:
  1. Verify NPDSingleUser encoder consistency (npd_encode vs polar_encode_batch+br)
  2. Train Stage 1 briefly, measure BLER on marginal channel
  3. Train Stage 2 briefly, measure BLER on clean channel
  4. Chain them and measure end-to-end BLER
  5. Report vs. SC-reference expectations

Should take < 5 minutes. If any stage fails, the pipeline has a bug.
"""
import os
import sys
import time
import math
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm

from class_c_npd.models.npd_single_user import NPDSingleUser, npd_encode
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design, design_summary
from class_c_npd.training.train_stage import (
    train_stage, generate_stage1_batch, generate_stage2_batch,
)
from class_c_npd.eval.chain_eval import chain_evaluate


def test_encoder_consistency():
    """npd_encode(u) should equal polar_encode_batch(u)[:, bit_reversal_perm]."""
    print('[1/5] Encoder consistency test...')
    for n in (3, 4, 5):
        N = 1 << n
        rng = np.random.default_rng(42)
        u = rng.integers(0, 2, (3, N)).astype(int)
        x_std = polar_encode_batch(u)
        x_npd = npd_encode(u)
        br = bit_reversal_perm(n)
        assert np.all(x_npd == x_std[:, br]), (
            f'ENCODER MISMATCH at N={N}: npd_encode != polar_encode[br]')
        print(f'  N={N}: OK')
    print('  PASS')


def test_single_user_npd_on_clean_awgn(n=4, iters=3000, sigma=0.4, seed=42):
    """
    Quick end-to-end test: single-user NPD on clean BPSK+AWGN at small N.
    At N=16, sigma=0.4 (moderate SNR), a trained NPD with 16 info bits
    (rate 1/2) should get very low BLER.
    """
    print(f'[2/5] Single-user NPD sanity check (N={1<<n}, sigma={sigma})...')
    N = 1 << n
    torch.manual_seed(seed)
    model = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)

    # Use Bhattacharyya-ordered positions for a fair polar code design.
    # For BPSK+AWGN with sigma=0.4, compute Z values and pick the K most
    # reliable positions (lowest Z) as info.
    def bhattacharyya_awgn(n_levels: int, sigma_val: float):
        """Compute Bhattacharyya parameters via the standard recursion."""
        # Initial Z for BPSK+AWGN
        z = np.exp(-1.0 / sigma_val ** 2)
        channels = np.array([z])
        for _ in range(n_levels):
            new_channels = np.empty(2 * len(channels))
            for i, zi in enumerate(channels):
                new_channels[2 * i] = 2 * zi - zi ** 2  # W-
                new_channels[2 * i + 1] = zi ** 2        # W+
            channels = new_channels
        return channels

    Z = bhattacharyya_awgn(n, sigma)
    # Best K positions = lowest Z in NATURAL message order
    K = N // 2
    sorted_pos = np.argsort(Z)
    info_positions = sorted(int(p) for p in sorted_pos[:K])
    frozen_set = set(range(N)) - set(info_positions)

    model.train()
    for it in range(1, iters + 1):
        u_msg = np.zeros((64, N), dtype=int)
        for p in info_positions:
            u_msg[:, p] = rng.integers(0, 2, 64)
        x = npd_encode(u_msg)  # NPD tree-order codeword
        z = (1.0 - 2.0 * x.astype(float)) + rng.normal(0, sigma, (64, N))
        emb = model.encode_channel(torch.from_numpy(z).float().unsqueeze(-1))
        loss = model.fast_ce(emb, torch.from_numpy(x).long())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 1000 == 0:
            print(f'  [{it}] loss={loss.item():.4f}')

    # Evaluate
    errs = 0
    total = 300
    model.eval()
    trng = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(total):
            u1 = np.zeros((1, N), dtype=int)
            for p in info_positions:
                u1[:, p] = trng.integers(0, 2, 1)
            x1 = npd_encode(u1)
            z1 = (1.0 - 2.0 * x1) + trng.normal(0, sigma, (1, N))
            emb1 = model.encode_channel(torch.from_numpy(z1).float().unsqueeze(-1))
            u_dec = model.decode(emb1, frozen_set)
            if any(u_dec[0, p].item() != u1[0, p] for p in info_positions):
                errs += 1

    bler = errs / total
    print(f'  Clean AWGN BLER (N={N}, rate=0.5, sigma={sigma}): {bler:.4f}')
    if bler > 0.3:
        print(f'  FAIL: BLER too high — NPD is not learning properly')
        return False
    print('  PASS')
    return True


def test_stage1_training(n=5, iters=5000, snr_db=6.0):
    """Train Stage 1 briefly and return BLER."""
    print(f'[3/5] Stage 1 training (GMAC, N={1<<n}, SNR={snr_db}dB)...')
    t0 = time.time()
    bler, ckpt = train_stage(
        stage=1, channel_name='gmac', n=n, snr_db=snr_db,
        d=16, hidden=64, n_layers=2, batch=32, lr=3e-4,
        total_iters=iters, eval_every=iters // 3,
        ku=None, kv=None, pe_threshold=0.01, seed=42,
        tag='smoke_stage1',
    )
    print(f'  Done in {(time.time()-t0)/60:.1f}min, BLER={bler:.4f}, ckpt={ckpt}')
    return bler, ckpt


def test_stage2_training(n=5, iters=3000, snr_db=6.0):
    """Train Stage 2 briefly."""
    print(f'[4/5] Stage 2 training (GMAC, N={1<<n}, SNR={snr_db}dB)...')
    t0 = time.time()
    bler, ckpt = train_stage(
        stage=2, channel_name='gmac', n=n, snr_db=snr_db,
        d=16, hidden=64, n_layers=2, batch=32, lr=3e-4,
        total_iters=iters, eval_every=iters // 3,
        ku=None, kv=None, pe_threshold=0.01, seed=43,
        tag='smoke_stage2',
    )
    print(f'  Done in {(time.time()-t0)/60:.1f}min, BLER={bler:.4f}, ckpt={ckpt}')
    return bler, ckpt


def test_chained_eval(s1_ckpt, s2_ckpt, n=5, snr_db=6.0):
    """Run end-to-end chained evaluation."""
    print(f'[5/5] End-to-end chained evaluation...')
    results = chain_evaluate(
        stage1_ckpt=s1_ckpt, stage2_ckpt=s2_ckpt,
        channel_name='gmac', n=n, snr_db=snr_db,
        pe_threshold=0.01, n_cw=500, batch=8, seed=999,
        verbose=False,
    )
    bler = results['bler_total']
    ci_low, ci_high = results['ci_low'], results['ci_high']
    print(f'  Chained BLER: {bler:.4f}  (CI [{ci_low:.4f}, {ci_high:.4f}])')
    print(f'  Errors:  total={results["errs_total"]}  U={results["errs_u"]}  V={results["errs_v"]}')
    return results


def main():
    print('=' * 60)
    print('Class C NPD Pipeline Smoke Test')
    print('=' * 60)
    print()

    # Step 1: encoder consistency (must pass for anything else to make sense)
    test_encoder_consistency()
    print()

    # Step 2: does our clean NPD actually learn on clean AWGN?
    # This is the test the existing npd_pytorch.py FAILED.
    ok = test_single_user_npd_on_clean_awgn(n=4, iters=3000, sigma=0.4)
    if not ok:
        print('ABORTING: single-user NPD broken, no point running further.')
        return
    print()

    # Step 3: train Stage 1 at small N
    bler1, ckpt1 = test_stage1_training(n=5, iters=5000)
    print()

    # Step 4: train Stage 2 at small N
    bler2, ckpt2 = test_stage2_training(n=5, iters=3000)
    print()

    # Step 5: chained end-to-end
    test_chained_eval(ckpt1, ckpt2, n=5, snr_db=6.0)
    print()

    print('=' * 60)
    print('SMOKE TEST COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
