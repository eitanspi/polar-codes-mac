"""
scale_large.py — Memory-lean curriculum scaling N=256/512/1024.

Smaller batch sizes and explicit cleanup to avoid OOM on macOS.
"""

import os
import gc
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

STAGES = [
    {"N": 256,  "iters": 8000, "batch": 8, "lr": 1.5e-4, "prev_N": 128,
     "eval_cw": 500,  "mc_trials": 2000},
    {"N": 512,  "iters": 6000, "batch": 4, "lr": 1e-4,   "prev_N": 256,
     "eval_cw": 300,  "mc_trials": 300},
    {"N": 1024, "iters": 4000, "batch": 2, "lr": 7e-5,   "prev_N": 512,
     "eval_cw": 200,  "mc_trials": 80},
]

D = 16
HIDDEN = 64
N_LAYERS = 2
LOG_EVERY = 500


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


def evaluate_nn(model, N, b, Au, Av, frozen_u, frozen_v, n_codewords, batch_size):
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
                        err = True
                        break
                if not err:
                    for pos in Av:
                        if pos in v_hat and int(v_hat[pos][i].item()) != v_full[i, pos - 1]:
                            err = True
                            break
                if err:
                    errors += 1
            total += bs

            # Free intermediate tensors
            del z_t, u_hat, v_hat
            gc.collect()

    model.train()
    return errors / total


def evaluate_sc(N, b, Au, Av, frozen_u, frozen_v, n_codewords):
    from polar.eval import MACEval
    channel = BEMAC()
    sc_eval = MACEval(channel, backend='interleaved')
    _, _, bler = sc_eval.run(N, b, Au, Av, frozen_u, frozen_v,
                             n_codewords=n_codewords,
                             batch_size=50, verbose=False)
    del sc_eval, channel
    gc.collect()
    return bler


def train_stage(model, N, n_iters, batch_size, lr):
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

        # Free graph every iteration
        del z, u, v, logits_list, targets_list, loss

        if it % LOG_EVERY == 0 or it == 1:
            avg = np.mean(losses[-min(100, len(losses)):])
            elapsed = time.time() - t0
            cur_lr = scheduler.get_last_lr()[0]
            print(f"    iter {it:6d}/{n_iters}: loss={losses[-1]:.4f}  "
                  f"avg100={avg:.4f}  lr={cur_lr:.6f}  "
                  f"time={elapsed:.1f}s", flush=True)

    avg_final = np.mean(losses[-100:])
    print(f"    Training done. Final avg loss (last 100): {avg_final:.4f}", flush=True)
    return avg_final


def main():
    print("=" * 70, flush=True)
    print("Pure Neural CalcParent — Lean Curriculum Scaling N=256/512/1024", flush=True)
    print("=" * 70, flush=True)

    results_all = {}

    for stage in STAGES:
        N = stage["N"]
        prev_N = stage["prev_N"]
        n_iters = stage["iters"]
        batch_size = stage["batch"]
        lr = stage["lr"]
        eval_cw = stage["eval_cw"]
        mc_trials = stage["mc_trials"]
        path_i = N // 2
        n = int(math.log2(N))
        ku = round(0.5 * N)
        kv = round(0.7 * N)

        print(f"\n{'='*70}", flush=True)
        print(f"  STAGE N={N}: fine-tune from N={prev_N}", flush=True)
        print(f"  iters={n_iters}, batch={batch_size}, lr={lr}", flush=True)
        print(f"  path_i={path_i}, ku={ku}, kv={kv}, eval_cw={eval_cw}", flush=True)
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
                del state
                gc.collect()
                print(f"  Loaded checkpoint: {prev_path}", flush=True)
            else:
                print(f"  WARNING: No checkpoint at {prev_path}, training from scratch!", flush=True)

            # Skip training if checkpoint already exists
            save_path = os.path.join(SAVED_DIR, f'ncg_pure_neural_N{N}.pt')
            if os.path.exists(save_path):
                state = torch.load(save_path, weights_only=True, map_location='cpu')
                model.load_state_dict(state, strict=True)
                del state; gc.collect()
                print(f"  Checkpoint exists, skipping training: {save_path}", flush=True)
            else:
                # Train
                print(f"\n  Training N={N}...", flush=True)
                train_stage(model, N, n_iters, batch_size, lr)
                torch.save(model.state_dict(), save_path)
                print(f"  Saved: {save_path}", flush=True)

            # Design code for evaluation
            b = make_path(N, path_i)
            design_path = os.path.join(os.path.dirname(__file__), '..',
                                       'designs', f'bemac_B_n{n}.npz')
            if os.path.exists(design_path):
                d = np.load(design_path)
                eu = d['u_error_rates']; ev = d['v_error_rates']
                su = np.argsort(eu); sv = np.argsort(ev)
                Au = sorted([int(i+1) for i in su[:ku]])
                Av = sorted([int(i+1) for i in sv[:kv]])
                all_pos = set(range(1, N+1))
                frozen_u = {p: 0 for p in sorted(all_pos - set(Au))}
                frozen_v = {p: 0 for p in sorted(all_pos - set(Av))}
                del d, eu, ev, su, sv
                print(f"  Loaded pre-computed design: {design_path}", flush=True)
            else:
                print(f"  No pre-computed design, computing (mc_trials={mc_trials})...", flush=True)
                Au, Av, frozen_u, frozen_v, _, _ = design_bemac_mc(
                    n, ku, kv, mc_trials=mc_trials, seed=42, verbose=False, path_i=path_i)
            print(f"  |Au|={len(Au)}, |Av|={len(Av)}", flush=True)

            # Evaluate NN
            eval_batch = max(1, min(20, 200 // (N // 16)))
            print(f"  Evaluating NN BLER ({eval_cw} codewords, eval_batch={eval_batch})...", flush=True)
            nn_bler = evaluate_nn(model, N, b, Au, Av, frozen_u, frozen_v,
                                  eval_cw, eval_batch)
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

        # Explicit cleanup between stages
        try:
            del model
        except NameError:
            pass
        gc.collect()
        print(f"  Stage N={N} cleanup done.", flush=True)

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY — Pure Neural CalcParent Curriculum Scaling (Large N, v2)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'N':>5s}  {'NN BLER':>10s}  {'SC BLER':>10s}  {'Ratio':>8s}", flush=True)
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}", flush=True)
    for N_val in [256, 512, 1024]:
        r = results_all.get(N_val, {})
        if "error" in r:
            print(f"  {N_val:>5d}  FAILED: {r['error']}", flush=True)
        elif r:
            print(f"  {N_val:>5d}  {r['nn_bler']:>10.4f}  {r['sc_bler']:>10.4f}  {r['ratio']:>8.2f}",
                  flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
