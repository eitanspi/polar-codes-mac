#!/usr/bin/env python3
"""
poc_no_frozen.py — POC: Train Without Frozen Set for GMAC Neural SC Decoder.

Concept (NPD / Aharoni et al.):
  During training, ALL N positions carry random info bits (no frozen set).
  The NNs learn general tree operations from fully-loaded codewords.
  Frozen set is only applied at inference time.

This loads the existing SimpleMLP_Gmac checkpoint and fine-tunes it
with fully-loaded training. Evaluation uses the real MC frozen sets.
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# ─── Config ──────────────────────────────────────────────────────────────────

N = 128
D = 16
HIDDEN = 64
N_LAYERS = 2

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

KU = 62
KV = 62
SC_BLER = 0.016
NN_BEST_BLER = 0.019

BATCH = 16
LR = 8e-5
TOTAL_ITERS = 200000
EVAL_EVERY = 5000
EVAL_CW = 3000
WARMUP_ITERS = 2000

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'poc_no_frozen_results.json')

TIME_LIMIT_HOURS = 1.5


# ─── Model (same as train_gmac_continue.py) ─────────────────────────────────

class SimpleMLP_Gmac(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2, z_hidden=32):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

    def forward(self, z, b, fu, fv, u_true=None, v_true=None):
        n = z.shape[1].bit_length() - 1
        br = torch.from_numpy(bit_reversal_perm(n)).long()
        root = self.z_encoder(z.unsqueeze(-1))[:, br]
        return self.tree(z=None, b=b, frozen_u=fu, frozen_v=fv,
                        u_true=u_true, v_true=v_true, root_emb=root)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Helpers ────────────────────────────────────────────────────────────────

def load_design():
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, KU, KV)
    return Au, Av, fu, fv


def evaluate(model, channel, b, Au, Av, fu, fv, n_cw):
    """Evaluate with REAL frozen sets (standard inference)."""
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    bs = 8
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


def get_lr(it):
    """Linear warmup then cosine decay to 0.01x."""
    if it < WARMUP_ITERS:
        return LR * it / WARMUP_ITERS
    progress = (it - WARMUP_ITERS) / max(1, TOTAL_ITERS - WARMUP_ITERS)
    return LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)  # Class B
    Au, Av, fu, fv = load_design()

    model = SimpleMLP_Gmac(d=D, hidden=HIDDEN, n_layers=N_LAYERS)

    # Load existing checkpoint
    ckpt_path = os.path.join(SAVE_DIR, f'ncg_gmac_mlp_N{N}.pt')
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, weights_only=True)
        fixed = {}
        for k, v in sd.items():
            nk = k.replace('z_enc.', 'z_encoder.') if k.startswith('z_enc.') else k
            fixed[nk] = v
        model.load_state_dict(fixed, strict=False)
        print(f'Loaded checkpoint: {ckpt_path}', flush=True)
    else:
        print(f'WARNING: No checkpoint at {ckpt_path}, training from scratch', flush=True)

    print(f'Model params: {model.count_parameters():,}', flush=True)
    print(f'N={N}, ku={KU}, kv={KV}, SC={SC_BLER}, NN_best={NN_BEST_BLER}', flush=True)
    print(f'batch={BATCH}, lr={LR}, iters={TOTAL_ITERS}', flush=True)
    print(f'Schedule: linear warmup {WARMUP_ITERS} iters + cosine decay', flush=True)
    print(f'TRAINING MODE: No frozen set (all {N} positions are info bits)', flush=True)
    print(f'EVAL MODE: Real frozen sets (ku={KU}, kv={KV})', flush=True)
    print(f'Time limit: {TIME_LIMIT_HOURS} hours', flush=True)

    # Initial eval with real frozen sets
    init_bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW)
    print(f'Initial BLER: {init_bler:.4f} (SC={SC_BLER}, ratio={init_bler/SC_BLER:.2f}x, '
          f'NN_best={NN_BEST_BLER})', flush=True)

    best_bler = init_bler
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng()
    t0 = time.time()
    losses = []
    history = []

    # Empty frozen dicts for training (no frozen positions)
    fu_train = {}
    fv_train = {}

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        # Check time limit
        elapsed_hrs = (time.time() - t0) / 3600
        if elapsed_hrs > TIME_LIMIT_HOURS:
            print(f'\nTime limit reached ({elapsed_hrs:.2f} hrs). Stopping.', flush=True)
            break

        # Set lr
        lr_now = get_lr(it)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        # Generate fully-loaded codewords: ALL positions are random info bits
        u = rng.integers(0, 2, (BATCH, N)).astype(int)
        v = rng.integers(0, 2, (BATCH, N)).astype(int)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(channel.sample_batch(x, y)).float()

        # Forward pass with EMPTY frozen dicts (no frozen positions)
        logits, targets, _, _, _ = model(z, b, fu_train, fv_train,
            u_true=torch.from_numpy(u).float(),
            v_true=torch.from_numpy(v).float())

        if logits:
            loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4),
                                   torch.stack(targets).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        if it % 500 == 0 and it < EVAL_EVERY:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-min(len(losses), 500):])
            print(f'[{it:>7}] loss={avg_loss:.4f} {elapsed/60:.0f}min lr={lr_now:.1e}',
                  flush=True)

        if it % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-min(len(losses), 500):])

            # Evaluate with REAL frozen sets
            bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(best_state,
                          os.path.join(SAVE_DIR, f'ncg_gmac_mlp_N{N}_nofrozen.pt'))
                improved = ' *BEST*'

            ratio_sc = bler / max(SC_BLER, 1e-8)
            ratio_nn = bler / max(NN_BEST_BLER, 1e-8)
            print(f'[{it:>7}/{TOTAL_ITERS}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}, SC={SC_BLER}={ratio_sc:.2f}x, '
                  f'NN={NN_BEST_BLER}={ratio_nn:.2f}x) '
                  f'{elapsed/60:.0f}min lr={lr_now:.1e}{improved}', flush=True)

            history.append({
                'iter': it, 'loss': avg_loss, 'bler': bler,
                'best_bler': best_bler, 'lr': lr_now,
                'time_min': elapsed / 60,
            })

            # Save results
            results = {
                'N': N, 'ku': KU, 'kv': KV, 'd': D, 'hidden': HIDDEN,
                'sc_bler': SC_BLER, 'nn_best_bler': NN_BEST_BLER,
                'init_bler': init_bler,
                'best_bler': best_bler, 'total_iters': it,
                'mode': 'no_frozen_training',
                'history': history,
            }
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # Final evaluation with more codewords
    model.load_state_dict(best_state)
    final_bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW * 2)
    elapsed = time.time() - t0

    print(f'\n{"="*70}', flush=True)
    print(f'POC NO-FROZEN TRAINING COMPLETE', flush=True)
    print(f'{"="*70}', flush=True)
    print(f'Total time: {elapsed/3600:.2f} hours, {it-1} iters', flush=True)
    print(f'Initial BLER:  {init_bler:.4f}', flush=True)
    print(f'Best BLER:     {best_bler:.4f}', flush=True)
    print(f'Final BLER:    {final_bler:.4f}', flush=True)
    print(f'SC baseline:   {SC_BLER:.4f} (ratio: {final_bler/SC_BLER:.2f}x)', flush=True)
    print(f'NN baseline:   {NN_BEST_BLER:.4f} (ratio: {final_bler/NN_BEST_BLER:.2f}x)',
          flush=True)
    if final_bler < SC_BLER:
        print(f'RESULT: BEATS SC!', flush=True)
    elif final_bler < NN_BEST_BLER:
        print(f'RESULT: Beats previous NN best, not yet SC', flush=True)
    else:
        print(f'RESULT: No improvement over NN best', flush=True)

    # Save final
    results = {
        'N': N, 'ku': KU, 'kv': KV, 'd': D, 'hidden': HIDDEN,
        'sc_bler': SC_BLER, 'nn_best_bler': NN_BEST_BLER,
        'init_bler': init_bler,
        'best_bler': best_bler, 'final_bler': final_bler,
        'total_iters': it - 1,
        'total_hours': elapsed / 3600,
        'mode': 'no_frozen_training',
        'history': history,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
