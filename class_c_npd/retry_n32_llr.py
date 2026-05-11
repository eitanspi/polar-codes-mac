"""
Retry N=32 Stage 1 with ANALYTICAL LLR as input instead of raw z.

Hypothesis:
  The d=16 model plateaus at BLER=0.108 on Stage 1 not because the tree
  operations are weak, but because the z_encoder MLP struggles to learn
  the Gaussian-mixture LLR from raw z.

Test:
  Replace the z_encoder input with the closed-form analytical LLR:
    LLR(x|z) = log[(N(z;+2,s^2) + N(z;0,s^2)) / (N(z;0,s^2) + N(z;-2,s^2))]

  This gives the network the "correct" per-position feature directly.
  If Stage 1 now reaches SC-level BLER, the bottleneck was the z_encoder,
  not the tree operations. If it still plateaus, the tree operations are
  the bottleneck.

Note: this is not strictly "pure NPD" — it uses channel-specific knowledge
in the input. But it's a clean diagnostic, and for the eventual "paper
quality" result, you can either leave the LLR input in (it's still a
standard approach in the NPD literature to use LLR inputs) or go back to
raw z with a bigger z_encoder.
"""
import os
import sys
import time
import math
import json
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.channels.frozen_sets import load_class_c_design, design_summary
from class_c_npd.eval.chain_eval import wilson_ci


# ─── Config ──────────────────────────────────────────────────────────────────

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
SIGMA = math.sqrt(SIGMA2)

N = 32
n = 5
KU, KV = 7, 15
SC_BLER = 0.0684
TARGET = 0.103

S1_ITERS = 30000
BATCH = 64
LR = 3e-4

EVAL_CW = 5000


# ─── Analytical mixture LLR for Stage 1 ──────────────────────────────────────

def mixture_llr_stage1(z: np.ndarray, sigma2: float) -> np.ndarray:
    """
    LLR(x | z) for the Gaussian mixture channel:
      p(z | x=0) = (1/2)[N(z; +2, s^2) + N(z; 0, s^2)]
      p(z | x=1) = (1/2)[N(z; 0, s^2) + N(z; -2, s^2)]
    Returns log[p(z|x=0) / p(z|x=1)].
    """
    z = z.astype(np.float64)
    def log_N(m):
        return -0.5 * (z - m) ** 2 / sigma2
    log_p0 = np.logaddexp(log_N(+2.0), log_N(0.0))
    log_p1 = np.logaddexp(log_N(0.0), log_N(-2.0))
    return (log_p0 - log_p1).astype(np.float32)


# ─── Batch generator that uses LLR input ────────────────────────────────────

def generate_stage1_batch_llr(N, Au, batch, rng, br):
    """Like generate_stage1_batch but feeds analytical LLR instead of raw z."""
    u_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au:
        u_msg[:, p - 1] = rng.integers(0, 2, batch)

    x_phys = polar_encode_batch(u_msg.astype(int))
    # V is truly uniform random (worst-case interference for U)
    v_random = rng.integers(0, 2, (batch, N))
    y_phys = polar_encode_batch(v_random)

    # Channel
    bx = 1.0 - 2.0 * x_phys.astype(np.float64)
    by = 1.0 - 2.0 * y_phys.astype(np.float64)
    w = rng.normal(0.0, SIGMA, (batch, N))
    z = (bx + by + w).astype(np.float32)

    # Feature = analytical LLR
    llr = mixture_llr_stage1(z, SIGMA2)

    # Bit-reverse to NPD tree order
    features_npd = llr[:, br]
    x_npd = x_phys[:, br]

    return u_msg, features_npd, x_npd


# ─── Evaluation with LLR input ──────────────────────────────────────────────

def evaluate_stage1_llr(model, N, Au, frozen, n_cw, br):
    model.eval()
    errs = 0
    rng = np.random.default_rng(999)
    bs = 32
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            u_true, features_npd, _ = generate_stage1_batch_llr(N, Au, actual, rng, br)
            ft = torch.from_numpy(features_npd).float().unsqueeze(-1)
            emb = model.encode_channel(ft)
            u_dec = model.decode(emb, frozen)
            for i in range(actual):
                if any(u_dec[i, p - 1].item() != u_true[i, p - 1] for p in Au):
                    errs += 1
            total += actual
    model.train()
    return errs / n_cw


def train_stage1_llr():
    """Train Stage 1 with analytical LLR input."""
    Au, Av, fu, fv, pe_u, pe_v = load_class_c_design('gmac', n, snr_db=SNR_DB, ku=KU, kv=KV)
    br = bit_reversal_perm(n)

    print('=' * 60)
    print(f'Stage 1 training — GMAC, N={N}, LLR INPUT')
    print(design_summary(Au, Av, pe_u, pe_v))
    print('=' * 60)

    torch.manual_seed(42)
    model = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f'Model: params={model.count_parameters():,}  batch={BATCH}  lr={LR}  iters={S1_ITERS}')

    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    best_bler = 1.0
    best_iter = 0
    ckpt_path = os.path.join(_HERE, 'results', 'gmac_c_stage1_N32_llr_best.pt')

    model.train()
    for it in range(1, S1_ITERS + 1):
        _, features_npd, cw_npd = generate_stage1_batch_llr(N, Au, BATCH, rng, br)
        ft = torch.from_numpy(features_npd).float().unsqueeze(-1)
        cw = torch.from_numpy(cw_npd).long()
        emb = model.encode_channel(ft)
        loss = model.fast_ce(emb, cw)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % 2000 == 0:
            bler = evaluate_stage1_llr(model, N, Au, fu, 500, br)
            avg_loss = float(np.mean(losses[-500:]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                best_iter = it
                torch.save({
                    'state_dict': model.state_dict(),
                    'd': 16, 'hidden': 64, 'n_layers': 2, 'z_dim': 1,
                    'Au': Au, 'N': N, 'stage': 1, 'uses_llr_input': True,
                }, ckpt_path)
                marker = ' *BEST*'
            print(f'  [{it:>6}/{S1_ITERS}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f} @{best_iter}) {elapsed:.1f}min{marker}',
                  flush=True)

    elapsed = (time.time() - t0) / 60
    print(f'\nStage 1 (LLR input) done in {elapsed:.1f} min. Best BLER: {best_bler:.4f} at iter {best_iter}')

    # Full eval on best checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    final_bler = evaluate_stage1_llr(model, N, Au, fu, EVAL_CW, br)
    errs = int(round(final_bler * EVAL_CW))
    ci_low, ci_high = wilson_ci(errs, EVAL_CW)

    print(f'\nRigorous eval ({EVAL_CW} CW): BLER={final_bler:.4f}  CI [{ci_low:.4f}, {ci_high:.4f}]')

    return {
        'stage1_best_bler_500cw': float(best_bler),
        'stage1_best_iter': int(best_iter),
        'stage1_final_bler_5000cw': float(final_bler),
        'stage1_ci_low': float(ci_low),
        'stage1_ci_high': float(ci_high),
        'elapsed_min': float(elapsed),
    }


def main():
    print(f'N={N} Stage 1 — ANALYTICAL LLR INPUT experiment')
    print(f'SC reference: {SC_BLER:.4f}  Target (1.5x): {TARGET:.4f}')
    print(f'Previous best (raw z, 80K iters): Stage 1 BLER=0.108, Chained=0.125')
    print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    t_total = time.time()

    stage1_results = train_stage1_llr()

    # Note: we don't need to retrain Stage 2 — it uses z'=z-(1-2x_hat)
    # which is already clean BPSK+AWGN, and Stage 2 at N=32 already works.
    # We reuse the existing stage2 checkpoint from the retry run.

    print('\n' + '=' * 60)
    print('SUMMARY — N=32 Stage 1 with analytical LLR')
    print('=' * 60)
    print(f'SC reference:  {SC_BLER:.4f}')
    print(f'Target (1.5x): {TARGET:.4f}')
    print(f'Stage 1 BLER (5000 CW): {stage1_results["stage1_final_bler_5000cw"]:.4f}')
    print(f'  95% CI: [{stage1_results["stage1_ci_low"]:.4f}, {stage1_results["stage1_ci_high"]:.4f}]')
    print(f'  vs raw z (0.108): ', end='')
    delta = stage1_results['stage1_final_bler_5000cw'] - 0.108
    if delta < -0.02:
        print(f'MUCH BETTER (by {-delta:.3f})')
    elif delta < 0:
        print(f'slightly better (by {-delta:.3f})')
    elif delta < 0.02:
        print(f'roughly same')
    else:
        print(f'worse (by {delta:.3f})')

    total_time = (time.time() - t_total) / 60
    print(f'\nTotal wall: {total_time:.1f} min')
    print(f'Finish: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # Save
    out = {
        'N': N, 'sc_bler': SC_BLER, 'target': TARGET,
        'experiment': 'stage1_analytical_llr_input',
        **stage1_results,
    }
    with open(os.path.join(_HERE, 'results', 'retry_n32_llr_results.json'), 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
