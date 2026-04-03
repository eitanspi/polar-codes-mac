#!/usr/bin/env python3
"""
train_abnmac.py — Train PureNeuralCompGraphDecoder for ABNMAC channel (Class B).

ABNMAC: Z = (X XOR E_x, Y XOR E_y) with correlated noise (E_x, E_y).
Output alphabet: {(0,0),(0,1),(1,0),(1,1)} -> encoded as integers 0..3.
Uses vocab_size=4 for the embedding layer.

Curriculum: N=32 (5K iters) -> N=64 (15K iters) -> N=128 (30K iters)
Class B path: path_i = N//2
Design: MC-based from precomputed designs/abnmac_B_n{n}.npz files.
"""

import sys
import os
import math
import time
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.encoder import polar_encode_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# ─── Directories ────────────────────────────────────────────────────────────

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
DESIGNS_DIR = os.path.join(BASE_DIR, 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'train_abnmac.log')

os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Config ─────────────────────────────────────────────────────────────────

D = 16
HIDDEN = 64
N_LAYERS = 2
BATCH_SIZE = 32
LR = 3e-4
EVAL_EVERY = 1000
EVAL_CODEWORDS = 500

# ABNMAC capacities (default noise): I(Z;X) ~ 0.400, I(Z;Y|X) ~ 0.800
# For Class B (symmetric interleaved path), use conservative symmetric rates.
# MC design shows good channels (pe < 0.01):
#   N=32: ~11 U, ~10 V;  N=64: ~27 U, ~22 V;  N=128: ~57 U, ~50 V
CURRICULUM = [
    {'N': 32,  'iters': 5000,  'ku': 10, 'kv': 10},
    {'N': 64,  'iters': 15000, 'ku': 22, 'kv': 22},
    {'N': 128, 'iters': 30000, 'ku': 45, 'kv': 45},
]

# ─── Logging ────────────────────────────────────────────────────────────────

logger = logging.getLogger('train_abnmac')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_FILE, mode='w')
fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S'))
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(fh)
logger.addHandler(sh)


# ─── ABNMAC output encoding ────────────────────────────────────────────────

def encode_abnmac_output(z_tuples):
    """Convert ABNMAC output tuples (zx, zy) to integers: zx*2 + zy."""
    result = np.empty(z_tuples.shape, dtype=np.int64)
    for idx in np.ndindex(z_tuples.shape):
        zx, zy = z_tuples[idx]
        result[idx] = zx * 2 + zy
    return result


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_nn(model, channel, N, b, Au, Av, fu, fv, n_cw, batch_size=25):
    """Evaluate neural decoder BLER."""
    model.eval()
    errs = 0
    total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au:
                uf[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                vf[:, p - 1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            zf_int = encode_abnmac_output(zf)
            zt = torch.from_numpy(zf_int).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                err = False
                for p in Au:
                    if p in uh and int(uh[p][i].item()) != uf[i, p - 1]:
                        err = True
                        break
                if not err:
                    for p in Av:
                        if p in vh and int(vh[p][i].item()) != vf[i, p - 1]:
                            err = True
                            break
                if err:
                    errs += 1
            total += actual
    model.train()
    return errs / max(total, 1)


def evaluate_sc(channel, N, b, Au, Av, fu, fv, n_cw):
    """SC reference baseline BLER."""
    from polar.decoder import decode_single
    from polar.encoder import polar_encode
    rng = np.random.default_rng(42)
    errs = 0
    for _ in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au:
            uf[p - 1] = rng.integers(0, 2)
        for p in Av:
            vf[p - 1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int))
        z_list = z.tolist()
        z_tuples = [tuple(zz) if isinstance(zz, list) else zz for zz in z_list]
        u_dec, v_dec = decode_single(N, z_tuples, b, fu, fv, channel, log_domain=True)
        if any(u_dec[p - 1] != uf[p - 1] for p in Au) or \
           any(v_dec[p - 1] != vf[p - 1] for p in Av):
            errs += 1
    return errs / n_cw


# ─── Training one stage ────────────────────────────────────────────────────

def train_stage(model, channel, stage_cfg, prev_ckpt=None):
    """Train at one N value with cosine LR schedule."""
    N = stage_cfg['N']
    total_iters = stage_cfg['iters']
    ku = stage_cfg['ku']
    kv = stage_cfg['kv']
    n = int(math.log2(N))
    path_i = N // 2  # Class B

    # Load MC design
    design_file = os.path.join(DESIGNS_DIR, f'abnmac_B_n{n}.npz')
    if not os.path.exists(design_file):
        logger.error(f"Design file not found: {design_file}")
        raise FileNotFoundError(design_file)

    Au, Av, fu, fv, pe_u, pe_v, file_path_i = design_from_file(design_file, n, ku, kv)
    b = make_path(N, path_i)

    logger.info(f"{'=' * 60}")
    logger.info(f"  Stage N={N}, Class B (path_i={path_i})")
    logger.info(f"  ku={ku}, kv={kv}, Ru={ku / N:.3f}, Rv={kv / N:.3f}")
    logger.info(f"  iters={total_iters}, batch={BATCH_SIZE}, lr={LR}")
    logger.info(f"  |Au|={len(Au)}, |Av|={len(Av)}")
    logger.info(f"{'=' * 60}")

    # Load previous stage checkpoint for curriculum transfer
    if prev_ckpt and os.path.exists(prev_ckpt):
        sd = torch.load(prev_ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(sd, strict=False)
        logger.info(f"  Loaded checkpoint: {os.path.basename(prev_ckpt)}")

    # SC baseline
    n_sc = min(500, 200 if N >= 128 else 500)
    logger.info(f"  Computing SC baseline ({n_sc} cw)...")
    sc_bler = evaluate_sc(channel, N, b, Au, Av, fu, fv, n_sc)
    logger.info(f"  SC BLER: {sc_bler:.4f}")

    # Initial NN eval
    init_bler = evaluate_nn(model, channel, N, b, Au, Av, fu, fv, EVAL_CODEWORDS)
    logger.info(f"  Initial NN BLER: {init_bler:.4f}")

    # Optimizer with cosine schedule
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_iters, eta_min=LR * 0.01)

    rng = np.random.default_rng()
    t0 = time.time()
    losses = []
    best_bler = init_bler
    best_ckpt = os.path.join(SAVE_DIR, f'ncg_abnmac_N{N}_best.pt')

    model.train()
    for it in range(1, total_iters + 1):
        # Generate training batch
        uf = np.zeros((BATCH_SIZE, N), dtype=int)
        vf = np.zeros((BATCH_SIZE, N), dtype=int)
        for p in Au:
            uf[:, p - 1] = rng.integers(0, 2, BATCH_SIZE)
        for p in Av:
            vf[:, p - 1] = rng.integers(0, 2, BATCH_SIZE)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        zf_int = encode_abnmac_output(zf)

        zt = torch.from_numpy(zf_int).long()
        ut = torch.from_numpy(uf).float()
        vt = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _, _ = model(
            zt, b, fu, fv, u_true=ut, v_true=vt)

        if len(all_logits) > 0:
            logits = torch.stack(all_logits, dim=1)
            targets = torch.stack(all_targets, dim=1)
            loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            losses.append(loss.item())

        # Evaluate
        if it % EVAL_EVERY == 0 or it == 1:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-200:]) if losses else 0
            bler = evaluate_nn(model, channel, N, b, Au, Av, fu, fv, EVAL_CODEWORDS)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), best_ckpt)
                improved = ' *BEST*'

            ratio = bler / max(sc_bler, 1e-8)
            cur_lr = scheduler.get_last_lr()[0]
            msg = (f'  N={N} [{it:>6}/{total_iters}] loss={avg_loss:.4f} '
                   f'BLER={bler:.4f} (best={best_bler:.4f}, SC={sc_bler:.4f}, '
                   f'ratio={ratio:.2f}x) lr={cur_lr:.6f} '
                   f'{elapsed / 60:.1f}min{improved}')
            logger.info(msg)

    # Final eval and save
    final_bler = evaluate_nn(model, channel, N, b, Au, Av, fu, fv, EVAL_CODEWORDS)
    if final_bler < best_bler:
        best_bler = final_bler
        torch.save(model.state_dict(), best_ckpt)

    logger.info(f"  Stage N={N} done. Final BLER={final_bler:.4f} "
                f"Best={best_bler:.4f} SC={sc_bler:.4f}")

    return best_ckpt, best_bler, sc_bler


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    logger.info('#' * 60)
    logger.info('  ABNMAC Neural SC Decoder — Class B Curriculum Training')
    logger.info(f'  Channel: Z=(X XOR E_x, Y XOR E_y), vocab_size=4')
    logger.info(f'  d={D}, hidden={HIDDEN}, n_layers={N_LAYERS}')
    logger.info(f'  batch_size={BATCH_SIZE}, lr={LR}')
    logger.info(f'  Curriculum: {[(s["N"], s["iters"]) for s in CURRICULUM]}')
    logger.info('#' * 60)

    channel = ABNMAC()
    I_ZX, I_ZY_X, I_ZXY = channel.capacity()
    logger.info(f'  ABNMAC capacity: I(Z;X)={I_ZX:.4f}, '
                f'I(Z;Y|X)={I_ZY_X:.4f}, I(Z;X,Y)={I_ZXY:.4f}')

    # Create model with vocab_size=4 for ABNMAC's 4-symbol output
    model = PureNeuralCompGraphDecoder(
        d=D, hidden=HIDDEN, n_layers=N_LAYERS, vocab_size=4)
    logger.info(f'  Parameters: {model.count_parameters():,}')

    results = {}
    prev_ckpt = None

    for stage_cfg in CURRICULUM:
        N = stage_cfg['N']
        ckpt, best_bler, sc_bler = train_stage(
            model, channel, stage_cfg, prev_ckpt=prev_ckpt)
        prev_ckpt = ckpt

        results[str(N)] = {
            'N': N,
            'ku': stage_cfg['ku'],
            'kv': stage_cfg['kv'],
            'best_bler': best_bler,
            'sc_bler': sc_bler,
            'ratio': best_bler / max(sc_bler, 1e-8),
            'iters': stage_cfg['iters'],
        }

    # Save results
    results_dir = os.path.join(BASE_DIR, 'results', 'abnmac')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'abnmac_classB_nn_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'\nResults saved to: {results_path}')

    # Summary
    logger.info(f'\n{"=" * 60}')
    logger.info(f'  ABNMAC Class B Neural Decoder Results')
    logger.info(f'{"=" * 60}')
    logger.info(f'  {"N":>5s}  {"ku":>4s}  {"kv":>4s}  {"SC":>8s}  {"NN":>8s}  {"Ratio":>8s}')
    logger.info(f'  {"-" * 45}')
    for key, r in results.items():
        ratio_str = f"{r['ratio']:.2f}x" if r['sc_bler'] > 0 else "-"
        logger.info(f"  {r['N']:>5d}  {r['ku']:>4d}  {r['kv']:>4d}  "
                     f"{r['sc_bler']:.4f}  {r['best_bler']:.4f}  {ratio_str:>8s}")
    logger.info(f'{"=" * 60}')


if __name__ == '__main__':
    main()
