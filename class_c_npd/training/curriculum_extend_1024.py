#!/usr/bin/env python3
"""
Extend GMAC Class C chained NPD curriculum to N=256, 512, 1024.
Warm-starts from existing N=128 curriculum checkpoint.
Saves checkpoints at every eval + final chained eval.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import torch

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.training.train_stage import (
    generate_stage1_batch, generate_stage2_batch, evaluate_stage,
)
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design, design_summary
from class_c_npd.eval.chain_eval import chain_evaluate, wilson_ci

SNR_DB = 6.0

SC_REF = {
    16:   {'ku': 4,   'kv': 7,   'sc_bler': 0.1626},
    32:   {'ku': 7,   'kv': 15,  'sc_bler': 0.0684},
    64:   {'ku': 15,  'kv': 29,  'sc_bler': 0.0266},
    128:  {'ku': 30,  'kv': 58,  'sc_bler': 0.0054},
    256:  {'ku': 59,  'kv': 117, 'sc_bler': 0.0020},
    512:  {'ku': 119, 'kv': 233, 'sc_bler': 0.0008},
    1024: {'ku': 238, 'kv': 467, 'sc_bler': 0.0002},
}

CURRICULUM = {
    256:  {'s1_iters': 200000, 's1_batch': 8,  's1_lr': 5e-5,
           's2_iters': 60000,  's2_batch': 8,  's2_lr': 1e-4},
    512:  {'s1_iters': 200000, 's1_batch': 4,  's1_lr': 3e-5,
           's2_iters': 60000,  's2_batch': 4,  's2_lr': 5e-5},
    1024: {'s1_iters': 150000, 's1_batch': 2,  's1_lr': 2e-5,
           's2_iters': 40000,  's2_batch': 2,  's2_lr': 3e-5},
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_one_stage(stage, N, ku, kv, cfg, warm_from):
    n = int(np.log2(N))
    channel = build_channel('gmac', sigma2=10 ** (-SNR_DB / 10))
    Au, Av, fu, fv, pe_u, pe_v = load_class_c_design('gmac', n, snr_db=SNR_DB, ku=ku, kv=kv)

    gen_fn = generate_stage1_batch if stage == 1 else generate_stage2_batch
    info = Au if stage == 1 else Av
    frozen = fu if stage == 1 else fv
    other = Av if stage == 1 else Au
    z_dim = channel.stage1_feature_dim if stage == 1 else channel.stage2_feature_dim

    iters = cfg[f's{stage}_iters']
    batch = cfg[f's{stage}_batch']
    lr = cfg[f's{stage}_lr']
    eval_every = max(5000, iters // 15)

    print(f'\n--- Stage {stage} | N={N} | ku={ku} kv={kv} ---')
    print(f'  iters={iters} batch={batch} lr={lr} eval_every={eval_every}')

    model = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=z_dim,
                           use_analytical_training=False)
    if warm_from and os.path.exists(warm_from):
        try:
            ckpt = torch.load(warm_from, weights_only=False, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
            print(f'  warm-started from {warm_from}')
        except Exception as e:
            print(f'  warm-start failed: {e}')
    else:
        print(f'  training from scratch')

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(42 if stage == 1 else 43)
    t0 = time.time()
    best_bler = 1.0
    losses = []

    tag = f'curriculum_gmac_c_s{stage}_N{N}'
    best_path = os.path.join(RESULTS_DIR, f'{tag}_best.pt')

    model.train()
    for it in range(1, iters + 1):
        _, features, cw = gen_fn(channel, N, info, batch, rng, other)
        ft = torch.from_numpy(features).float()
        if ft.dim() == 2:
            ft = ft.unsqueeze(-1)
        emb = model.encode_channel(ft)
        loss = model.fast_ce(emb, torch.from_numpy(cw).long())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % eval_every == 0:
            model.eval()
            n_eval = min(500, max(100, 2000 // max(1, N // 64)))
            bler = evaluate_stage(model, channel, N, info, frozen, gen_fn,
                                  n_eval, seed=999, other_info=other)
            model.train()
            elapsed = (time.time() - t0) / 60
            avg_loss = np.mean(losses[-500:])
            marker = ''
            if bler < best_bler:
                best_bler = bler
                marker = ' *'
                torch.save({'state_dict': model.state_dict(),
                            'z_dim': z_dim, 'N': N, 'Au': Au, 'Av': Av,
                            'stage': stage, 'iter': it}, best_path)
            # Save periodic checkpoint
            iter_path = os.path.join(RESULTS_DIR, f'{tag}_iter{it}.pt')
            torch.save({'state_dict': model.state_dict(),
                        'z_dim': z_dim, 'N': N, 'Au': Au, 'Av': Av,
                        'stage': stage, 'iter': it}, iter_path)
            print(f'  [{it:>6}/{iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}', flush=True)

    elapsed = (time.time() - t0) / 60
    print(f'  Done: {elapsed:.1f}min, best BLER={best_bler:.4f}')
    return best_path, best_bler, elapsed


def main():
    print('=' * 60)
    print('GMAC Class C NPD Curriculum — Extend to N=1024')
    print(f'SNR={SNR_DB}dB')
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)

    results = {}

    # Warm-start from npd_design checkpoints (neural mode, not analytical)
    prev_s1 = os.path.join(RESULTS_DIR, 'npd_design_p3_N128_best.pt')
    prev_s2 = os.path.join(RESULTS_DIR, 'npd_design_p3_N64_best.pt')
    # Fallback
    if not os.path.exists(prev_s1):
        prev_s1 = os.path.join(RESULTS_DIR, 'npd_design_p3_N256_best.pt')
    if not os.path.exists(prev_s2):
        prev_s2 = None

    for N in [256, 512, 1024]:
        ref = SC_REF[N]
        ku, kv = ref['ku'], ref['kv']
        cfg = CURRICULUM[N]

        print(f'\n{"="*60}')
        print(f'N={N} | ku={ku} kv={kv} | SC BLER={ref["sc_bler"]}')
        print(f'{"="*60}')

        s1_path, s1_bler, s1_time = train_one_stage(
            1, N, ku, kv, cfg, prev_s1)

        s2_path, s2_bler, s2_time = train_one_stage(
            2, N, ku, kv, cfg, prev_s2)

        # Chained evaluation
        print(f'\n  Chained evaluation...')
        n_cw = min(5000, max(500, 10000 // max(1, N // 32)))
        try:
            cr = chain_evaluate(s1_path, s2_path, 'gmac', int(np.log2(N)),
                                SNR_DB, ku, kv, n_cw=n_cw, batch=max(1, 16 // max(1, N // 64)),
                                seed=999, verbose=False)
            bler = cr['bler_total']
            ci_lo, ci_hi = cr['ci_low'], cr['ci_high']
            ratio = bler / max(ref['sc_bler'], 1e-8)
            print(f'  Chained BLER={bler:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}] '
                  f'(SC={ref["sc_bler"]}, ratio={ratio:.2f}x)')
        except Exception as e:
            print(f'  Chained eval failed: {e}')
            bler = -1; ci_lo = ci_hi = -1; ratio = -1

        results[N] = {
            'N': N, 'ku': ku, 'kv': kv,
            'sc_bler': ref['sc_bler'],
            's1_bler': s1_bler, 's1_time_min': s1_time,
            's2_bler': s2_bler, 's2_time_min': s2_time,
            'chained_bler': bler, 'ci_low': ci_lo, 'ci_high': ci_hi,
            'ratio': ratio,
        }

        with open(os.path.join(RESULTS_DIR, 'curriculum_extend_1024.json'), 'w') as f:
            json.dump(results, f, indent=2)

        prev_s1 = s1_path
        prev_s2 = s2_path

    print(f'\n{"="*60}')
    print('ALL DONE')
    for N, r in sorted(results.items()):
        print(f'  N={N}: NPD BLER={r["chained_bler"]:.4f} '
              f'(SC={r["sc_bler"]}, ratio={r["ratio"]:.2f}x)')
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
