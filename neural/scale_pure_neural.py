"""
scale_pure_neural.py — Curriculum scaling of Pure Neural CalcParent decoder.

Scales N=16 -> N=32 -> N=64 -> N=128 using curriculum learning.
The model weights are shared across N (weight-shared tree operations),
so we just load the previous checkpoint and fine-tune on larger N data.
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from polar.encoder import polar_encode_batch
from polar.channels import BEMAC
from polar.design import make_path
from polar.design_mc import design_bemac_mc
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

SAVED_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

# ─── Config per N ─────────────────────────────────────────────────────────────

STAGES = [
    {"N": 32,  "iters": 20000, "batch": 48, "lr": 3e-4,   "prev_N": 16},
    {"N": 64,  "iters": 15000, "batch": 32, "lr": 2e-4,   "prev_N": 32},
    {"N": 128, "iters": 12000, "batch": 24, "lr": 1.5e-4, "prev_N": 64},
]

D = 16
HIDDEN = 64
N_LAYERS = 2
EVAL_BATCH = 100
LOG_EVERY = 1000


# ─── Data generation ─────────────────────────────────────────────────────────

def generate_batch(N, batch_size, rng=None):
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
    if not logits_list:
        return torch.tensor(0.0)
    all_logits = torch.stack(logits_list)
    all_targets = torch.stack(targets_list)
    return F.cross_entropy(all_logits.reshape(-1, 4), all_targets.reshape(-1))


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_nn(model, N, b, Au, Av, frozen_u, frozen_v,
                n_codewords, batch_size):
    model.eval()
    errors = 0
    total = 0
    rng = np.random.default_rng(123)

    with torch.no_grad():
        for start in range(0, n_codewords, batch_size):
            bs = min(batch_size, n_codewords - start)

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
            _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u, frozen_v)

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


def evaluate_sc(N, b, Au, Av, frozen_u, frozen_v, n_codewords):
    from polar.eval import MACEval
    channel = BEMAC()
    sc_eval = MACEval(channel, backend='interleaved')
    _, _, bler = sc_eval.run(N, b, Au, Av, frozen_u, frozen_v,
                             n_codewords=n_codewords,
                             batch_size=100, verbose=False)
    return bler


# ─── Training one stage ──────────────────────────────────────────────────────

def train_stage(model, N, n_iters, batch_size, lr):
    n = int(math.log2(N))
    path_i = N // 2
    b = make_path(N, path_i)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iters, eta_min=lr / 10)

    rng = np.random.default_rng()
    model.train()

    t0 = time.time()
    losses = []

    for it in range(1, n_iters + 1):
        z, u, v = generate_batch(N, batch_size, rng)

        logits_list, targets_list, _, _, _ = model(
            z, b, {}, {}, u_true=u, v_true=v, distill_alpha=0.0)

        loss = compute_loss(logits_list, targets_list)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if it % LOG_EVERY == 0 or it == 1:
            avg = np.mean(losses[-min(100, len(losses)):])
            elapsed = time.time() - t0
            cur_lr = scheduler.get_last_lr()[0]
            print(f"    iter {it:6d}/{n_iters}: loss={loss.item():.4f}  "
                  f"avg100={avg:.4f}  lr={cur_lr:.6f}  "
                  f"time={elapsed:.1f}s", flush=True)

    avg_final = np.mean(losses[-100:])
    print(f"    Training done. Final avg loss (last 100): {avg_final:.4f}", flush=True)
    return avg_final


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70, flush=True)
    print("Pure Neural CalcParent — Curriculum Scaling N=32/64/128", flush=True)
    print("=" * 70, flush=True)

    results_all = {}

    for stage in STAGES:
        N = stage["N"]
        prev_N = stage["prev_N"]
        n_iters = stage["iters"]
        batch_size = stage["batch"]
        lr = stage["lr"]
        path_i = N // 2
        n = int(math.log2(N))
        ku = round(0.5 * N)
        kv = round(0.7 * N)

        print(f"\n{'='*70}", flush=True)
        print(f"  STAGE N={N}: fine-tune from N={prev_N}", flush=True)
        print(f"  iters={n_iters}, batch={batch_size}, lr={lr}", flush=True)
        print(f"  path_i={path_i}, ku={ku}, kv={kv}", flush=True)
        print(f"{'='*70}", flush=True)

        try:
            # Create model
            model = PureNeuralCompGraphDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
            print(f"  Parameters: {model.count_parameters()}", flush=True)

            # Load previous checkpoint
            prev_path = os.path.join(SAVED_DIR, f'ncg_pure_neural_N{prev_N}.pt')
            if os.path.exists(prev_path):
                state = torch.load(prev_path, weights_only=True, map_location='cpu')
                model.load_state_dict(state, strict=True)
                print(f"  Loaded checkpoint: {prev_path}", flush=True)
            else:
                print(f"  WARNING: No checkpoint at {prev_path}, training from scratch!", flush=True)

            # Train
            print(f"\n  Training N={N}...", flush=True)
            train_stage(model, N, n_iters, batch_size, lr)

            # Save checkpoint
            save_path = os.path.join(SAVED_DIR, f'ncg_pure_neural_N{N}.pt')
            torch.save(model.state_dict(), save_path)
            print(f"  Saved: {save_path}", flush=True)

            # Design code for evaluation
            print(f"\n  Designing code (n={n}, ku={ku}, kv={kv})...", flush=True)
            b = make_path(N, path_i)
            Au, Av, frozen_u, frozen_v, _, _ = design_bemac_mc(
                n, ku, kv, mc_trials=2000, seed=42, verbose=False, path_i=path_i)
            print(f"  |Au|={len(Au)}, |Av|={len(Av)}", flush=True)

            # Evaluate NN
            eval_cw = 2000
            print(f"  Evaluating NN BLER ({eval_cw} codewords)...", flush=True)
            nn_bler = evaluate_nn(model, N, b, Au, Av, frozen_u, frozen_v,
                                  eval_cw, EVAL_BATCH)
            print(f"  NN BLER = {nn_bler:.4f}", flush=True)

            # Evaluate SC
            print(f"  Evaluating SC BLER ({eval_cw} codewords)...", flush=True)
            sc_bler = evaluate_sc(N, b, Au, Av, frozen_u, frozen_v, eval_cw)
            print(f"  SC BLER = {sc_bler:.4f}", flush=True)

            ratio = nn_bler / max(sc_bler, 1e-6)
            print(f"  NN/SC ratio = {ratio:.2f}", flush=True)

            results_all[N] = {
                "nn_bler": nn_bler,
                "sc_bler": sc_bler,
                "ratio": ratio,
            }

        except Exception as e:
            print(f"  FAILED N={N}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results_all[N] = {"error": str(e)}

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY — Pure Neural CalcParent Curriculum Scaling", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'N':>5s}  {'NN BLER':>10s}  {'SC BLER':>10s}  {'Ratio':>8s}", flush=True)
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}", flush=True)
    for N_val in [32, 64, 128]:
        r = results_all.get(N_val, {})
        if "error" in r:
            print(f"  {N_val:>5d}  FAILED: {r['error']}", flush=True)
        elif r:
            print(f"  {N_val:>5d}  {r['nn_bler']:>10.4f}  {r['sc_bler']:>10.4f}  {r['ratio']:>8.2f}",
                  flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
