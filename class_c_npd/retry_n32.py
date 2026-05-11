"""
Retry N=32 with 80K Stage 1 iterations.

The original pilot showed Stage 1 was still converging at 30K iters
(trajectory: 0.284 → 0.228 → 0.142 → 0.134). This run gives it enough
iterations to finish, and warm-starts from the N=16 checkpoint to speed
up early convergence.
"""
import os
import sys
import time
import json
import torch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.training.train_stage import train_stage
from class_c_npd.eval.chain_eval import chain_evaluate


SNR_DB = 6.0
N = 32
n = 5
KU, KV = 7, 15
SC_BLER = 0.0684
TARGET = 0.103  # 1.5x SC

S1_ITERS = 80000
S2_ITERS = 15000
S1_BATCH = 64
S1_LR = 3e-4

EVAL_CW = 5000


def load_warm_start_weights(target_model: NPDSingleUser, source_ckpt_path: str):
    """Load compatible weights from a smaller-N checkpoint. All weights are
    N-independent in this NPD architecture (shared z_encoder, checknode,
    bitnode, emb2llr), so any smaller N can warm-start any larger N."""
    if not os.path.exists(source_ckpt_path):
        print(f'  (no warm-start checkpoint at {source_ckpt_path}, starting fresh)')
        return False
    ckpt = torch.load(source_ckpt_path, weights_only=False, map_location='cpu')
    try:
        target_model.load_state_dict(ckpt['state_dict'])
        print(f'  Warm-started from: {source_ckpt_path}')
        return True
    except Exception as e:
        print(f'  WARN: could not load warm-start weights: {e}')
        return False


def train_stage_with_warm_start(
    stage, n, snr_db, ku, kv, total_iters, batch, lr,
    warm_start_from=None, tag=None,
):
    """Like train_stage but optionally warm-starts from a checkpoint."""
    from class_c_npd.training.train_stage import (
        generate_stage1_batch, generate_stage2_batch, evaluate_stage
    )
    from class_c_npd.channels.mac_channel import build_channel
    from class_c_npd.channels.frozen_sets import load_class_c_design, design_summary

    N = 1 << n
    channel_kwargs = {'sigma2': 10 ** (-snr_db / 10)}
    channel = build_channel('gmac', **channel_kwargs)
    Au, Av, fu, fv, pe_u, pe_v = load_class_c_design(
        'gmac', n, snr_db=snr_db, ku=ku, kv=kv)

    print('=' * 60)
    print(f'Stage {stage} training — GMAC, N={N}, SNR={snr_db}dB')
    print(design_summary(Au, Av, pe_u, pe_v))
    print('=' * 60)

    if stage == 1:
        gen_fn = generate_stage1_batch
        info = Au
        frozen = fu
        z_dim = channel.stage1_feature_dim
    else:
        gen_fn = generate_stage2_batch
        info = Av
        frozen = fv
        z_dim = channel.stage2_feature_dim

    torch.manual_seed(42 if stage == 1 else 43)
    model = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=z_dim)

    if warm_start_from:
        load_warm_start_weights(model, warm_start_from)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f'Model: params={model.count_parameters():,}  batch={batch}  lr={lr}  iters={total_iters}')

    rng = np.random.default_rng(42 if stage == 1 else 43)
    t0 = time.time()
    losses = []
    best_bler = 1.0
    best_iter = 0
    save_dir = os.path.join(_HERE, 'results')
    ckpt_path = os.path.join(save_dir, f'{tag}_best.pt')

    eval_every = max(2000, total_iters // 10)
    model.train()
    for it in range(1, total_iters + 1):
        _, features_npd, cw_npd = gen_fn(channel, N, info, batch, rng)
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
            bler = evaluate_stage(model, channel, N, info, frozen, gen_fn, 500, seed=999)
            avg_loss = float(np.mean(losses[-500:]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                best_iter = it
                torch.save({
                    'state_dict': model.state_dict(),
                    'd': 16, 'hidden': 64, 'n_layers': 2, 'z_dim': z_dim,
                    'channel': 'gmac', 'stage': stage, 'N': N,
                    'Au': Au, 'Av': Av,
                }, ckpt_path)
                marker = ' *BEST*'
            print(f'  [{it:>6}/{total_iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f} @{best_iter}) {elapsed:.1f}min{marker}',
                  flush=True)

    elapsed = (time.time() - t0) / 60
    print(f'\nStage {stage} done in {elapsed:.1f} min. Best BLER: {best_bler:.4f} at iter {best_iter}')
    return best_bler, ckpt_path, elapsed


def main():
    print(f'Retry N={N} — 80K Stage 1 iters + warm start from N=16')
    print(f'SC reference: {SC_BLER:.4f}, target (1.5x): {TARGET:.4f}')
    print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    t_total = time.time()

    n16_s1_ckpt = os.path.join(_HERE, 'results', 'gmac_c_stage1_N16_best.pt')
    n16_s2_ckpt = os.path.join(_HERE, 'results', 'gmac_c_stage2_N16_best.pt')

    # Stage 1
    print('\n[Stage 1]')
    s1_bler, s1_ckpt, s1_time = train_stage_with_warm_start(
        stage=1, n=n, snr_db=SNR_DB, ku=KU, kv=KV,
        total_iters=S1_ITERS, batch=S1_BATCH, lr=S1_LR,
        warm_start_from=n16_s1_ckpt,
        tag=f'gmac_c_stage1_N{N}_retry',
    )

    # Stage 2
    print('\n[Stage 2]')
    s2_bler, s2_ckpt, s2_time = train_stage_with_warm_start(
        stage=2, n=n, snr_db=SNR_DB, ku=KU, kv=KV,
        total_iters=S2_ITERS, batch=S1_BATCH, lr=S1_LR,
        warm_start_from=n16_s2_ckpt,
        tag=f'gmac_c_stage2_N{N}_retry',
    )

    # Chained eval
    print('\n[Chained eval]')
    t0 = time.time()
    results = chain_evaluate(
        stage1_ckpt=s1_ckpt, stage2_ckpt=s2_ckpt,
        channel_name='gmac', n=n, snr_db=SNR_DB,
        ku=KU, kv=KV,
        n_cw=EVAL_CW, batch=16, seed=999, verbose=False,
    )
    eval_time = (time.time() - t0) / 60
    print(f'  Chained BLER: {results["bler_total"]:.4f}  '
          f'(95% CI: [{results["ci_low"]:.4f}, {results["ci_high"]:.4f}])')
    print(f'  Errs:  U={results["errs_u"]}  V={results["errs_v"]}  total={results["errs_total"]}')

    total_time = (time.time() - t_total) / 60
    bler = results['bler_total']
    ratio = bler / SC_BLER
    passed = bler <= TARGET

    print('\n' + '=' * 60)
    print('N=32 RETRY RESULTS')
    print('=' * 60)
    print(f'SC reference:   {SC_BLER:.4f}')
    print(f'Target (1.5x):  {TARGET:.4f}')
    print(f'Stage 1 BLER:   {s1_bler:.4f} (in {s1_time:.1f} min)')
    print(f'Stage 2 BLER:   {s2_bler:.4f} (in {s2_time:.1f} min)')
    print(f'Chained BLER:   {bler:.4f}  (CI [{results["ci_low"]:.4f}, {results["ci_high"]:.4f}])')
    print(f'Ratio to SC:    {ratio:.2f}x')
    print(f'Pass?           {"YES" if passed else "NO"}')
    print(f'Total wall:     {total_time:.1f} min')

    out = {
        'N': N, 'sc_bler': SC_BLER, 'target_1_5x': TARGET,
        'stage1_bler': float(s1_bler), 'stage1_time_min': float(s1_time),
        'stage2_bler': float(s2_bler), 'stage2_time_min': float(s2_time),
        'chained_bler': float(bler),
        'chained_ci_low': float(results['ci_low']),
        'chained_ci_high': float(results['ci_high']),
        'chained_errs_u': int(results['errs_u']),
        'chained_errs_v': int(results['errs_v']),
        'ratio_to_sc': float(ratio),
        'pass_1_5x': bool(passed),
        'total_time_min': float(total_time),
        's1_iters': S1_ITERS,
    }
    with open(os.path.join(_HERE, 'results', 'retry_n32_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved to class_c_npd/results/retry_n32_results.json')


if __name__ == '__main__':
    main()
