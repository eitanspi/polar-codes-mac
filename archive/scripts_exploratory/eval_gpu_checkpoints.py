#!/usr/bin/env python3
"""
eval_gpu_checkpoints.py
=======================
Poll for GPU paper-style checkpoints, eval Stage 1, train Stage 2, save results.

Expected checkpoints: /tmp/paper_style/isi_N{X}_final.pt
Model: ChainedNPD_MAC(d=16, hidden=100, n_layers=2, encoder_type="bigru", gru_layers=1)

For each checkpoint found:
  1. Load into model.stage1
  2. Eval Stage 1 BLER at 5K CW
  3. Train Stage 2 for 30K iters
  4. Eval chained BLER at 5K CW
  5. Save to results/paper_style/npd_paper_style_evals.json
"""
from __future__ import annotations
import json, math, os, sys, time, glob
import numpy as np
import torch

torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.design_mc import design_from_file
from neural.npd_memory_mac import ChainedNPD_MAC

# ISI-MAC channel (same as the GPU training script)
from polar.channels_memory import ISIMAC

SIGMA2 = 10**(-0.6)  # = 0.2512, ~6 dB
H = 0.3  # ISI tap h=0.3 (matching GPU training script)
SNR_DB = 6
D = 16
HIDDEN = 100
N_LAYERS = 2
GRU_LAYERS = 1
ENCODER_TYPE = 'bigru'
LR = 1e-3
SEED = 42

SRC_DIR = '/tmp/paper_style'
RESULTS_DIR = os.path.join(_ROOT, 'results', 'paper_style')
OUT_JSON = os.path.join(RESULTS_DIR, 'npd_paper_style_evals.json')

# N -> (ku, kv) mapping (from design files, Class C @ SNR=6dB)
N_CONFIGS = {
    16:   (4, 7),
    32:   (7, 15),
    64:   (15, 29),
    128:  (30, 58),
    256:  (59, 117),
    512:  (119, 233),
    1024: (238, 467),
}


def load_design(N):
    ku, kv = N_CONFIGS[N]
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    fu_set = {p-1 for p in range(1, N+1) if p not in Au}
    fv_set = {p-1 for p in range(1, N+1) if p not in Av}
    return Au, Av, fu_set, fv_set


def make_batch_isi(N, Au, Av, batch, rng, channel):
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au:
        u_msg[:, p-1] = rng.integers(0, 2, batch)
    for p in Av:
        v_msg[:, p-1] = rng.integers(0, 2, batch)
    x_phys = polar_encode_batch(u_msg.astype(int))
    y_phys = polar_encode_batch(v_msg.astype(int))
    z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
    return u_msg, v_msg, np.asarray(z, dtype=np.float32), x_phys, y_phys


def eval_stage1(model, channel, N, Au, Av, fu_set, n_cw=5000, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage1.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    bs = min(32, max(4, 512 // N))
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            u_msg, v_msg, z, x_phys, y_phys = make_batch_isi(N, Au, Av, actual, rng, channel)
            z_t = torch.from_numpy(z)
            emb = model.stage1.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb_npd, fu_set)
            for i in range(actual):
                if any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au):
                    errs += 1
            total += actual
    return errs / n_cw


def train_stage2(model, channel, N, Au, Av, fv_set, iters=30000, batch=16, seed=43):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage1.eval()
    model.stage2.train()
    opt = torch.optim.AdamW(model.stage2.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    best_bler = 1.0
    batch_size = min(batch, max(2, 256 // N))

    for it in range(1, iters + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch_isi(N, Au, Av, batch_size, rng, channel)
        z_t = torch.from_numpy(z)
        side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)
        emb = model.stage2.encode_channel(z_t, side=side)
        emb_npd = emb[:, br_t, :]
        cw_npd = torch.from_numpy(y_phys[:, br]).long()
        loss = model.stage2.tree.fast_ce(emb_npd, cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage2.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            bler = eval_stage2_quick(model, channel, N, Au, Av, fv_set, n_cw=300, seed=998)
            if bler < best_bler:
                best_bler = bler
            print(f'    S2 [{it}/{iters}] BLER={bler:.4f} (best={best_bler:.4f})', flush=True)

    return best_bler


def eval_stage2_quick(model, channel, N, Au, Av, fv_set, n_cw=300, seed=998):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage2.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    bs = min(16, max(2, 256 // N))
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            u_msg, v_msg, z, x_phys, y_phys = make_batch_isi(N, Au, Av, actual, rng, channel)
            z_t = torch.from_numpy(z)
            side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)
            emb = model.stage2.encode_channel(z_t, side=side)
            emb_npd = emb[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb_npd, fv_set)
            for i in range(actual):
                if any(int(v_hat[i, p-1].item()) != int(v_msg[i, p-1]) for p in Av):
                    errs += 1
            total += actual
    model.stage2.train()
    return errs / n_cw


def eval_chained(model, channel, N, Au, Av, fu_set, fv_set, n_cw=5000, seed=777):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage1.eval()
    model.stage2.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    total = 0
    bs = min(32, max(2, 512 // N))
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            u_msg, v_msg, z, x_phys, _ = make_batch_isi(N, Au, Av, actual, rng, channel)
            z_t = torch.from_numpy(z)
            emb1 = model.stage1.encode_channel(z_t)
            emb1_npd = emb1[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb1_npd, fu_set)
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)
            side = torch.from_numpy((1.0 - 2.0 * x_hat.astype(np.float32))).unsqueeze(-1)
            emb2 = model.stage2.encode_channel(z_t, side=side)
            emb2_npd = emb2[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb2_npd, fv_set)
            for i in range(actual):
                u_wrong = any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au)
                v_wrong = any(int(v_hat[i, p-1].item()) != int(v_msg[i, p-1]) for p in Av)
                if u_wrong: errs_u += 1
                if v_wrong: errs_v += 1
                if u_wrong or v_wrong: errs_total += 1
            total += actual
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u/n_cw, 'bler_v': errs_v/n_cw, 'bler_total': errs_total/n_cw,
    }


def wilson_ci(errs, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = errs / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, centre-margin), min(1, centre+margin))


def process_checkpoint(ckpt_path):
    """Process a single GPU checkpoint."""
    basename = os.path.basename(ckpt_path)
    # Parse N from filename like isi_N64_final.pt
    parts = basename.replace('.pt', '').split('_')
    N = None
    for p in parts:
        if p.startswith('N'):
            try:
                N = int(p[1:])
                break
            except ValueError:
                pass
    if N is None:
        print(f'  Could not parse N from {basename}, skipping')
        return None

    if N not in N_CONFIGS:
        print(f'  N={N} not in config, skipping')
        return None

    print(f'\n  Processing {basename} (N={N})...')
    Au, Av, fu_set, fv_set = load_design(N)
    channel = ISIMAC(sigma2=SIGMA2, h=H)

    torch.manual_seed(SEED)
    model = ChainedNPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS,
                           encoder_type=ENCODER_TYPE, gru_layers=GRU_LAYERS)

    # Load Stage 1
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    try:
        # GPU checkpoints may be raw state_dicts (not wrapped in {'state_dict': ...})
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.stage1.load_state_dict(ckpt['state_dict'])
        else:
            # The checkpoint IS the state_dict
            model.stage1.load_state_dict(ckpt)
    except Exception as e:
        print(f'  Failed to load: {e}')
        return None

    # Eval Stage 1
    print(f'  Eval Stage 1 (5K CW)...')
    t0 = time.time()
    s1_bler = eval_stage1(model, channel, N, Au, Av, fu_set, n_cw=5000, seed=999)
    s1_time = (time.time() - t0) / 60
    print(f'  S1 BLER={s1_bler:.4f} ({s1_time:.1f} min)')

    # Train Stage 2
    print(f'  Training Stage 2 (30K iters)...')
    t1 = time.time()
    s2_bler = train_stage2(model, channel, N, Au, Av, fv_set, iters=30000, batch=16, seed=43)
    s2_time = (time.time() - t1) / 60
    print(f'  S2 best BLER={s2_bler:.4f} ({s2_time:.1f} min)')

    # Eval chained
    print(f'  Eval chained (5K CW)...')
    t2 = time.time()
    chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set, n_cw=5000, seed=777)
    ch_ci = wilson_ci(chained['errs_total'], 5000)
    ch_time = (time.time() - t2) / 60
    print(f'  Chained BLER={chained["bler_total"]:.4f} CI=[{ch_ci[0]:.4f},{ch_ci[1]:.4f}] ({ch_time:.1f} min)')

    return {
        'checkpoint': basename,
        'N': N,
        'ku': N_CONFIGS[N][0],
        'kv': N_CONFIGS[N][1],
        's1_bler': float(s1_bler),
        's2_best_bler': float(s2_bler),
        'chained': {k: float(v) for k, v in chained.items()},
        'chained_ci': list(ch_ci),
        's1_eval_time_min': s1_time,
        's2_train_time_min': s2_time,
    }


def main():
    # Load existing results
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Find new checkpoints
    ckpts = sorted(glob.glob(os.path.join(SRC_DIR, '*_final.pt')))
    if not ckpts:
        print(f'No checkpoints found in {SRC_DIR}')
        print('To fetch: scp -P 30067 eitansp@132.72.137.66:~/polar_project/class_c_npd/results/npd_paper_style/*_final.pt /tmp/paper_style/')
        return

    processed = set(all_results.keys())
    new_ckpts = [c for c in ckpts if os.path.basename(c) not in processed]

    if not new_ckpts:
        print(f'All {len(ckpts)} checkpoints already processed.')
        return

    print(f'Found {len(new_ckpts)} new checkpoint(s) to process.')

    for ckpt_path in new_ckpts:
        result = process_checkpoint(ckpt_path)
        if result:
            all_results[os.path.basename(ckpt_path)] = result
            with open(OUT_JSON, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f'  Saved to {OUT_JSON}')

    print(f'\nDone. Total results: {len(all_results)}')


if __name__ == '__main__':
    main()
