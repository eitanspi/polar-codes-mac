#!/usr/bin/env python3
"""
train_abnmac.py — Train neural SC decoder for ABNMAC channel.

ABNMAC: Z = (X⊕E_x, Y⊕E_y) with correlated noise.
Output alphabet: {(0,0),(0,1),(1,0),(1,1)} — 4 discrete symbols.

Uses PureNeuralCompGraphDecoder with vocab_size=4 (discrete embedding).
Curriculum: N=16 → 32 → 64 → 128.
"""

import sys, os, math, time, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'train_abnmac.log')

# ABNMAC capacities (default noise):
# I(Z;X) ≈ 0.400 bits, I(Z;Y|X) ≈ 0.800 bits
# Symmetric rate: ~0.42

# Rate points for Class B (interleaved path, symmetric rate ~0.42)
RATES = {
    16:  {'ku': 7,  'kv': 7},
    32:  {'ku': 13, 'kv': 13},
    64:  {'ku': 27, 'kv': 27},
    128: {'ku': 54, 'kv': 54},
}

# SC baselines (will be computed in-script)
SC_BASELINES = {}


def encode_abnmac_output(z_tuples):
    """Convert ABNMAC output tuples to integer labels for embedding.
    (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    """
    result = np.empty(z_tuples.shape, dtype=np.int64)
    for idx in np.ndindex(z_tuples.shape):
        zx, zy = z_tuples[idx]
        result[idx] = zx * 2 + zy
    return result


def evaluate(model, channel, b, Au, Av, fu, fv, N, n_cw, batch_size=25):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            zf_int = encode_abnmac_output(zf)
            zt = torch.from_numpy(zf_int).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


def eval_sc(channel, b, Au, Av, fu, fv, N, n_cw):
    """Analytical SC baseline."""
    from polar.decoder import decode_single
    from polar.encoder import polar_encode
    rng = np.random.default_rng(42)
    errs = 0
    for i in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int))
        z_list = z.tolist()
        # Convert to tuples
        z_tuples = [tuple(zz) if isinstance(zz, list) else zz for zz in z_list]
        u_dec, v_dec = decode_single(N, z_tuples, b, fu, fv, channel, log_domain=True)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or \
           any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
    return errs / n_cw


def train_at_N(N, model, channel, prev_ckpt=None, total_iters=30000, lr=1e-3,
               batch_size=16, eval_every=2000):
    n = int(math.log2(N))
    rates = RATES[N]
    ku, kv = rates['ku'], rates['kv']

    # Try MC design first, fall back to analytical
    design_file = os.path.join(DESIGNS_DIR, f'abnmac_B_n{n}.npz')
    if os.path.exists(design_file):
        Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, ku, kv)
        b = make_path(N, N//2)
        path_name = 'Class B (MC)'
    else:
        from polar.design import design_abnmac
        Au, Av, fu, fv, _, _ = design_abnmac(n, ku, kv)
        b = make_path(N, N)
        path_name = 'Class C (analytical)'

    print(f"\n{'='*60}")
    print(f"  Training ABNMAC at N={N} ({path_name})")
    print(f"  ku={ku}, kv={kv}, Ru={ku/N:.3f}, Rv={kv/N:.3f}")
    print(f"  iters={total_iters}, lr={lr}, batch={batch_size}")
    print(f"{'='*60}")

    # Load previous checkpoint if curriculum
    if prev_ckpt and os.path.exists(prev_ckpt):
        sd = torch.load(prev_ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded checkpoint: {prev_ckpt}")

    # SC baseline
    print(f"  Computing SC baseline ({min(2000, 500 if N >= 128 else 2000)} cw)...", flush=True)
    n_sc_cw = min(2000, 500 if N >= 128 else 2000)
    sc_bler = eval_sc(channel, b, Au, Av, fu, fv, N, n_sc_cw)
    SC_BASELINES[N] = sc_bler
    print(f"  SC BLER: {sc_bler:.4f}")

    # Evaluate initial
    init_bler = evaluate(model, channel, b, Au, Av, fu, fv, N, 500)
    print(f"  Initial NN BLER: {init_bler:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_iters, eta_min=lr*0.01)

    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    best_bler = init_bler

    model.train()
    for it in range(1, total_iters + 1):
        uf = np.zeros((batch_size, N), dtype=int)
        vf = np.zeros((batch_size, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, batch_size)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, batch_size)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
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

        if it % eval_every == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-200:]) if losses else 0
            bler = evaluate(model, channel, b, Au, Av, fu, fv, N, 500)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                ckpt_path = os.path.join(SAVE_DIR, f'ncg_abnmac_N{N}.pt')
                torch.save(model.state_dict(), ckpt_path)
                improved = ' *BEST*'

            ratio = bler / max(sc_bler, 1e-8)
            msg = (f'  [{it:>6}/{total_iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}, SC={sc_bler:.4f}, ratio={ratio:.2f}x) '
                   f'{elapsed/60:.0f}min{improved}')
            print(msg, flush=True)
            with open(LOG_FILE, 'a') as f:
                f.write(msg + '\n')

    # Final save
    ckpt_path = os.path.join(SAVE_DIR, f'ncg_abnmac_N{N}.pt')
    torch.save(model.state_dict(), ckpt_path)
    final_bler = evaluate(model, channel, b, Au, Av, fu, fv, N, 1000)
    print(f"  Final BLER: {final_bler:.4f} (SC={sc_bler:.4f}, ratio={final_bler/max(sc_bler,1e-8):.2f}x)")

    return ckpt_path, final_bler


def main():
    channel = ABNMAC()

    print(f"\n{'#'*60}")
    print(f"  ABNMAC Neural SC Decoder Training")
    print(f"  Channel: Z=(X⊕E_x, Y⊕E_y)")
    print(f"  Vocab size: 4 (discrete embedding)")
    print(f"{'#'*60}")

    # Create model with vocab_size=4 (ABNMAC has 4 output symbols)
    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=4)
    print(f"  Parameters: {model.count_parameters():,}")

    results = {}
    prev_ckpt = None

    for N in [16, 32, 64, 128]:
        iters = {16: 15000, 32: 30000, 64: 50000, 128: 80000}[N]
        lr = {16: 3e-3, 32: 1e-3, 64: 5e-4, 128: 3e-4}[N]
        bs = {16: 32, 32: 16, 64: 8, 128: 4}[N]

        ckpt, final_bler = train_at_N(
            N, model, channel, prev_ckpt=prev_ckpt,
            total_iters=iters, lr=lr, batch_size=bs, eval_every=max(1000, iters//10))
        prev_ckpt = ckpt

        results[str(N)] = {
            'N': N, 'ku': RATES[N]['ku'], 'kv': RATES[N]['kv'],
            'nn_bler': final_bler,
            'sc_bler': SC_BASELINES.get(N),
        }

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'abnmac',
                            'abnmac_nn_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  ABNMAC Neural Decoder Results")
    print(f"{'='*60}")
    print(f"  {'N':>5s}  {'SC':>8s}  {'NN-SC':>8s}  {'Ratio':>8s}")
    print(f"  {'-'*35}")
    for N_s, r in results.items():
        sc = r.get('sc_bler')
        nn = r.get('nn_bler')
        ratio = f"{nn/sc:.2f}x" if sc and nn and sc > 0 else "-"
        print(f"  {r['N']:>5d}  {sc:.4f}  {nn:.4f}  {ratio:>8s}")


if __name__ == '__main__':
    main()
