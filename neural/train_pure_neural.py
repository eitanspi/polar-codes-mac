"""
train_pure_neural.py — Train Pure Neural CalcParent with Knowledge Distillation.

Loads pre-trained weights for CalcLeft, CalcRight, Emb2Logits, Logits2Emb,
EmbeddingZ from the existing ncg_N16_d16.pt checkpoint, then trains
NeuralCalcParent with a 3-phase curriculum:

  Phase A (10K iters): distill_alpha=1.0, only CalcParent params trained
  Phase B (10K iters): alpha decays 1.0->0.0, only CalcParent params trained
  Phase C (10K iters): alpha=0, finetune ALL params jointly at low LR

Evaluates BLER at each phase boundary.
"""

import os
import json
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

from polar.encoder import polar_encode_batch
from polar.channels import BEMAC
from polar.design import make_path
from polar.design_mc import design_bemac_mc
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# ─── Config ──────────────────────────────────────────────────────────────────

N = 16
D = 16
HIDDEN = 64
N_LAYERS = 2
BATCH_SIZE = 64
PATH_I = N // 2  # Class B
LR = 1e-3

PHASE_A_ITERS = 4000
PHASE_B_ITERS = 4000
PHASE_C_ITERS = 4000

EVAL_CODEWORDS = 3000
EVAL_BATCH = 200

# Eval rate point
KU = 8
KV = 11


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
                             batch_size=200, verbose=False)
    return bler


# ─── Load pre-trained weights ────────────────────────────────────────────────

def load_pretrained(model):
    """Load pre-trained weights from ncg_N16_d16.pt into matching layers."""
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'ncg_N16_d16.pt')
    if not os.path.exists(ckpt_path):
        print(f"WARNING: No checkpoint at {ckpt_path}, training from scratch.", flush=True)
        return

    state = torch.load(ckpt_path, weights_only=True, map_location='cpu')

    compatible_state = {}
    for key, val in state.items():
        if key in model.state_dict() and model.state_dict()[key].shape == val.shape:
            compatible_state[key] = val

    model.load_state_dict(compatible_state, strict=False)
    print(f"  Loaded {len(compatible_state)} params from checkpoint, "
          f"{len(model.state_dict()) - len(compatible_state)} new params.", flush=True)


# ─── Training ────────────────────────────────────────────────────────────────

def get_param_groups(model):
    """Split parameters into CalcParent-new and pre-trained-other."""
    calc_parent_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'calc_parent_nn' in name or 'parent_second_nn' in name:
            calc_parent_params.append(param)
        else:
            other_params.append(param)
    return calc_parent_params, other_params


def train_phase(model, b, phase_name, n_iters, alpha_start, alpha_end,
                optimizer, rng, scheduler=None, log_every=1000):
    """Train one phase with given distillation alpha schedule."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {phase_name}: {n_iters} iters, alpha {alpha_start:.2f} -> {alpha_end:.2f}",
          flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    losses = []

    for it in range(1, n_iters + 1):
        frac = (it - 1) / max(n_iters - 1, 1)
        alpha = alpha_start + (alpha_end - alpha_start) * frac

        z, u, v = generate_batch(N, BATCH_SIZE, rng)

        logits_list, targets_list, _, _, distill_loss = model(
            z, b, {}, {}, u_true=u, v_true=v, distill_alpha=alpha)

        ce_loss = compute_loss(logits_list, targets_list)
        total_loss = ce_loss + alpha * distill_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(total_loss.item())

        if it % log_every == 0 or it == 1:
            elapsed = time.time() - t0
            lr_str = ""
            if scheduler is not None:
                lr_str = f"  lr={scheduler.get_last_lr()[0]:.6f}"
            print(f"  iter {it:5d}: CE={ce_loss.item():.4f}  "
                  f"distill={distill_loss.item():.4f}  "
                  f"alpha={alpha:.3f}  "
                  f"total={total_loss.item():.4f}{lr_str}  "
                  f"time={elapsed:.1f}s", flush=True)

    avg_loss = np.mean(losses[-100:])
    print(f"  {phase_name} done. Avg loss (last 100): {avg_loss:.4f}", flush=True)
    return avg_loss


def main():
    print("=" * 60, flush=True)
    print("Pure Neural CalcParent — Knowledge Distillation Training", flush=True)
    print(f"  N={N}, d={D}, hidden={HIDDEN}, path_i={PATH_I}", flush=True)
    print(f"  Phases: A={PHASE_A_ITERS}, B={PHASE_B_ITERS}, C={PHASE_C_ITERS}", flush=True)
    print("=" * 60, flush=True)

    n = int(math.log2(N))
    b = make_path(N, PATH_I)

    # Design frozen set for evaluation
    Au, Av, frozen_u, frozen_v, _, _ = design_bemac_mc(
        n, KU, KV, mc_trials=2000, seed=42, verbose=False, path_i=PATH_I)
    print(f"  Eval: ku={KU}, kv={KV}, |Au|={len(Au)}, |Av|={len(Av)}", flush=True)

    # Create model
    model = PureNeuralCompGraphDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    print(f"  Total parameters: {model.count_parameters()}", flush=True)

    # Load pre-trained weights
    load_pretrained(model)

    calc_parent_params, other_params = get_param_groups(model)
    print(f"  CalcParent params: {sum(p.numel() for p in calc_parent_params)}", flush=True)
    print(f"  Other params: {sum(p.numel() for p in other_params)}", flush=True)

    rng = np.random.default_rng()

    # ─── Phase A: Train ONLY CalcParent with strong distillation ─────────
    for p in other_params:
        p.requires_grad = False

    optimizer_a = torch.optim.Adam(calc_parent_params, lr=LR)
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_a, T_max=PHASE_A_ITERS, eta_min=LR / 10)

    train_phase(model, b, "Phase A (distill CalcParent only)", PHASE_A_ITERS,
                alpha_start=1.0, alpha_end=1.0, optimizer=optimizer_a,
                rng=rng, scheduler=scheduler_a, log_every=500)

    print("\n  Evaluating after Phase A...", flush=True)
    bler_a = evaluate_nn(model, N, b, Au, Av, frozen_u, frozen_v,
                         EVAL_CODEWORDS, EVAL_BATCH)
    print(f"  Phase A BLER: {bler_a:.4f}", flush=True)

    # ─── Phase B: Still only CalcParent, decay distillation ──────────────
    optimizer_b = torch.optim.Adam(calc_parent_params, lr=LR * 0.5)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, T_max=PHASE_B_ITERS, eta_min=LR / 20)

    train_phase(model, b, "Phase B (decay distill, CalcParent only)", PHASE_B_ITERS,
                alpha_start=1.0, alpha_end=0.0, optimizer=optimizer_b,
                rng=rng, scheduler=scheduler_b, log_every=500)

    print("\n  Evaluating after Phase B...", flush=True)
    bler_b = evaluate_nn(model, N, b, Au, Av, frozen_u, frozen_v,
                         EVAL_CODEWORDS, EVAL_BATCH)
    print(f"  Phase B BLER: {bler_b:.4f}", flush=True)

    # ─── Phase C: Finetune ALL params at low LR, pure neural ─────────────
    for p in other_params:
        p.requires_grad = True

    optimizer_c = torch.optim.Adam([
        {'params': calc_parent_params, 'lr': LR * 0.1},
        {'params': other_params, 'lr': LR * 0.05},
    ])
    scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_c, T_max=PHASE_C_ITERS, eta_min=LR / 100)

    train_phase(model, b, "Phase C (finetune all, pure neural)", PHASE_C_ITERS,
                alpha_start=0.0, alpha_end=0.0, optimizer=optimizer_c,
                rng=rng, scheduler=scheduler_c, log_every=500)

    print("\n  Evaluating after Phase C...", flush=True)
    bler_c = evaluate_nn(model, N, b, Au, Av, frozen_u, frozen_v,
                         EVAL_CODEWORDS, EVAL_BATCH)
    print(f"  Phase C BLER: {bler_c:.4f}", flush=True)

    # SC baseline
    print("\n  Computing SC baseline...", flush=True)
    sc_bler = evaluate_sc(N, b, Au, Av, frozen_u, frozen_v, EVAL_CODEWORDS)
    print(f"  SC BLER: {sc_bler:.4f}", flush=True)

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models',
                             'ncg_pure_neural_N16.pt')
    torch.save(model.state_dict(), save_path)
    print(f"  Saved model: {save_path}", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  SC BLER:        {sc_bler:.4f}", flush=True)
    print(f"  NN Phase A:     {bler_a:.4f}  (distill alpha=1.0, CalcParent only)", flush=True)
    print(f"  NN Phase B:     {bler_b:.4f}  (alpha 1.0->0.0, CalcParent only)", flush=True)
    print(f"  NN Phase C:     {bler_c:.4f}  (pure neural, all params)", flush=True)
    best_bler = min(bler_a, bler_b, bler_c)
    ratio = best_bler / max(sc_bler, 1e-6)
    print(f"  Best NN BLER:   {best_bler:.4f}", flush=True)
    print(f"  Best / SC:      {ratio:.2f}", flush=True)
    if ratio <= 1.5:
        print("  Result: SUCCESS — Pure neural CalcParent approaches SC", flush=True)
    else:
        print("  Result: Needs more work", flush=True)
    print("=" * 60, flush=True)

    # Save results
    results = {
        "N": N,
        "d": D,
        "hidden": HIDDEN,
        "path_i": PATH_I,
        "ku": KU,
        "kv": KV,
        "sc_bler": sc_bler,
        "nn_bler_phase_a": bler_a,
        "nn_bler_phase_b": bler_b,
        "nn_bler_phase_c": bler_c,
        "best_nn_bler": best_bler,
        "ratio_best_vs_sc": ratio,
        "phase_a_iters": PHASE_A_ITERS,
        "phase_b_iters": PHASE_B_ITERS,
        "phase_c_iters": PHASE_C_ITERS,
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'pure_neural_calcparent.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}", flush=True)

    return results


if __name__ == '__main__':
    main()
