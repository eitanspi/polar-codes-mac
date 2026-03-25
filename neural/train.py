"""
train_ncg.py — Training script for Neural Computational Graph SC Decoder.

Workflow:
  1. Micro-batch overfit test (100 fixed samples) — validates architecture
  2. Full training with teacher forcing on Class B path
  3. Evaluation: NN BLER vs analytical SC BLER

Usage:
    python -m neural.train [--phase overfit|train|eval|all]
"""

import os
import argparse
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

from polar.encoder import polar_encode_batch
from polar.channels import BEMAC
from polar.design import make_path
from polar.design_mc import design_bemac_mc
from neural.neural_comp_graph import NeuralCompGraphDecoder


# ─── Data generation ─────────────────────────────────────────────────────────

def generate_batch(N, batch_size, rng=None):
    """Generate random (z, u, v) for BEMAC training."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.integers(0, 2, (batch_size, N))
    v = rng.integers(0, 2, (batch_size, N))
    x = polar_encode_batch(u)
    y = polar_encode_batch(v)
    z = x + y
    return (torch.from_numpy(z).long(),
            torch.from_numpy(u).float(),
            torch.from_numpy(v).float())


def compute_loss(logits_list, targets_list):
    """4-class CE loss over all info-bit positions."""
    if not logits_list:
        return torch.tensor(0.0)
    all_logits = torch.stack(logits_list)    # (S, B, 4)
    all_targets = torch.stack(targets_list)  # (S, B)
    return F.cross_entropy(all_logits.reshape(-1, 4), all_targets.reshape(-1))


# ─── Phase 1: Micro-batch overfit test ───────────────────────────────────────

def overfit_test(N=8, d=16, hidden=64, n_layers=2, n_iters=3000,
                 batch_size=16, n_samples=100, lr=1e-3):
    """
    Validate architecture by overfitting to a tiny fixed dataset.
    If loss doesn't approach 0, the architecture is broken.
    """
    print("=" * 60)
    print("PHASE 1: Micro-batch overfit test")
    print(f"  N={N}, d={d}, hidden={hidden}, samples={n_samples}")
    print("=" * 60)

    n = int(math.log2(N))
    path_i = N // 2
    b = make_path(N, path_i)

    # Create model
    model = NeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)
    print(f"  Parameters: {model.count_parameters()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr / 10)

    # Generate FIXED dataset (all positions as info)
    rng = np.random.default_rng(42)
    z_all, u_all, v_all = generate_batch(N, n_samples, rng)

    t0 = time.time()
    for it in range(1, n_iters + 1):
        # Sample mini-batch from fixed dataset
        idx = torch.randint(0, n_samples, (batch_size,))
        z = z_all[idx]
        u = u_all[idx]
        v = v_all[idx]

        logits_list, targets_list, _, _ = model(
            z, b, {}, {}, u_true=u, v_true=v)
        loss = compute_loss(logits_list, targets_list)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if it % 500 == 0 or it == 1:
            elapsed = time.time() - t0
            print(f"  iter {it:5d}: loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  "
                  f"time={elapsed:.1f}s")

    final_loss = loss.item()
    status = "PASS" if final_loss < 0.1 else "FAIL"
    print(f"\n  Overfit test: loss={final_loss:.4f} → {status}")

    if status == "FAIL":
        print("  ⚠ Architecture cannot overfit tiny dataset. CalcParent strategy may be broken.")
    return model, status


# ─── Phase 2: Full training ─────────────────────────────────────────────────

def train_full(N=8, d=16, hidden=64, n_layers=2, n_iters=20000,
               batch_size=64, lr=1e-3, model=None):
    """Train on random data with teacher forcing."""
    print("\n" + "=" * 60)
    print("PHASE 2: Full training")
    print(f"  N={N}, d={d}, hidden={hidden}, iters={n_iters}, batch={batch_size}")
    print("=" * 60)

    n = int(math.log2(N))
    path_i = N // 2
    b = make_path(N, path_i)

    if model is None:
        model = NeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)
    print(f"  Parameters: {model.count_parameters()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr / 20)

    rng = np.random.default_rng()
    t0 = time.time()

    for it in range(1, n_iters + 1):
        z, u, v = generate_batch(N, batch_size, rng)

        # Training with all-info (no frozen set) — same as NPD convention
        logits_list, targets_list, _, _ = model(
            z, b, {}, {}, u_true=u, v_true=v)
        loss = compute_loss(logits_list, targets_list)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if it % 2000 == 0 or it == 1:
            elapsed = time.time() - t0
            print(f"  iter {it:5d}: loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  "
                  f"time={elapsed:.1f}s")

    print(f"  Final loss: {loss.item():.4f}")
    return model


# ─── Phase 3: Evaluation ────────────────────────────────────────────────────

def evaluate(model, N=8, n_codewords=2000, batch_size=100):
    """Evaluate NN BLER vs SC BLER for Class B rate points."""
    print("\n" + "=" * 60)
    print("PHASE 3: Evaluation")
    print("=" * 60)

    n = int(math.log2(N))
    path_i = N // 2
    b = make_path(N, path_i)
    channel = BEMAC()

    # Rate points to test
    rate_points = [
        (int(N * 0.5),   int(N * 0.7)),    # Ru=0.5,  Rv≈0.7  (round)
        (int(N * 0.625), int(N * 0.875)),   # Ru=0.625, Rv=0.875
    ]
    # Fix rounding for N=8
    if N == 8:
        rate_points = [(4, 6), (5, 7)]

    for ku, kv in rate_points:
        print(f"\n  --- Ru={ku/N:.3f}, Rv={kv/N:.3f} (ku={ku}, kv={kv}) ---")

        # Design frozen set for Class B
        Au, Av, frozen_u, frozen_v, _, _ = design_bemac_mc(
            n, ku, kv, mc_trials=2000, seed=42, verbose=False,
            path_i=path_i)

        # NN evaluation
        nn_bler = evaluate_nn(model, N, b, Au, Av, frozen_u, frozen_v,
                              n_codewords, batch_size)

        # SC evaluation
        sc_bler = evaluate_sc(N, b, Au, Av, frozen_u, frozen_v,
                              channel, n_codewords, batch_size)

        ratio = nn_bler / max(sc_bler, 1e-6)
        status = "✓ MATCH" if ratio < 1.5 else ("★ BEATS" if ratio < 1.0 else "✗ FAIL")
        print(f"  NN BLER = {nn_bler:.4f}")
        print(f"  SC BLER = {sc_bler:.4f}")
        print(f"  Ratio   = {ratio:.2f}  {status}")


def evaluate_nn(model, N, b, Au, Av, frozen_u, frozen_v,
                n_codewords, batch_size):
    """Compute NN BLER."""
    model.eval()
    errors = 0
    total = 0
    rng = np.random.default_rng(123)

    with torch.no_grad():
        for start in range(0, n_codewords, batch_size):
            bs = min(batch_size, n_codewords - start)

            # Generate with frozen set
            u_full = np.zeros((bs, N), dtype=int)
            v_full = np.zeros((bs, N), dtype=int)
            for pos in Au:
                u_full[:, pos - 1] = rng.integers(0, 2, bs)
            for pos in Av:
                v_full[:, pos - 1] = rng.integers(0, 2, bs)

            x = polar_encode_batch(u_full)
            y = polar_encode_batch(v_full)
            z = x + y

            z_t = torch.from_numpy(z).long()
            _, _, u_hat, v_hat = model(z_t, b, frozen_u, frozen_v)

            # Check block errors
            for i in range(bs):
                err = False
                for pos in Au:
                    if pos in u_hat and int(u_hat[pos][i].item()) != u_full[i, pos - 1]:
                        err = True; break
                if not err:
                    for pos in Av:
                        if pos in v_hat and int(v_hat[pos][i].item()) != v_full[i, pos - 1]:
                            err = True; break
                if err:
                    errors += 1
            total += bs

    model.train()
    return errors / total


def evaluate_sc(N, b, Au, Av, frozen_u, frozen_v, channel,
                n_codewords, batch_size):
    """Compute SC BLER using analytical decoder."""
    from polar.eval import MACEval
    sc_eval = MACEval(channel, backend='interleaved')
    _, _, bler = sc_eval.run(N, b, Au, Av, frozen_u, frozen_v,
                             n_codewords=n_codewords,
                             batch_size=batch_size, verbose=False)
    return bler


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all',
                        choices=['overfit', 'train', 'eval', 'all'])
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--train_iters', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_codewords', type=int, default=2000)
    args = parser.parse_args()

    model = None

    if args.phase in ('overfit', 'all'):
        model, status = overfit_test(
            N=args.N, d=args.d, hidden=args.hidden, n_layers=args.n_layers,
            lr=args.lr)
        if status == 'FAIL' and args.phase == 'all':
            print("\nStopping: overfit test failed.")
            return

    if args.phase in ('train', 'all'):
        model = train_full(
            N=args.N, d=args.d, hidden=args.hidden, n_layers=args.n_layers,
            n_iters=args.train_iters, batch_size=args.batch_size,
            lr=args.lr, model=model)

        # Save model
        save_path = os.path.join('saved_models',
                                 f'ncg_N{args.N}_d{args.d}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"  Saved: {save_path}")

    if args.phase in ('eval', 'all'):
        if model is None:
            model = NeuralCompGraphDecoder(
                d=args.d, hidden=args.hidden, n_layers=args.n_layers)
            load_path = os.path.join('saved_models',
                                     f'ncg_N{args.N}_d{args.d}.pt')
            model.load_state_dict(torch.load(load_path, weights_only=True))
            print(f"  Loaded: {load_path}")

        evaluate(model, N=args.N, n_codewords=args.eval_codewords)


if __name__ == '__main__':
    main()
