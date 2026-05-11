#!/usr/bin/env python3
"""
poc_gumbel_scl.py — POC: Differentiable list decoding via Gumbel-Softmax.

Hypothesis: teacher-forced training produces miscalibrated probabilities at
N=256 because the model never sees its own errors propagate. Gumbel-Softmax
replaces hard true-bit leaf embeddings with soft model predictions during
training, so the model learns that bad confidence -> bad downstream decisions.

Three experiments:
  1. Pure Gumbel-Softmax (soft, every leaf)
  2. Mixed scheduled sampling (ramp p_gumbel 0 -> 0.5)
  3. Gumbel hard=True (straight-through estimator)

Key implementation detail: to keep backward pass tractable at N=256 (246 info
positions), we detach the Gumbel embedding every `detach_every` info positions.
This limits gradient chain to ~20 steps while still teaching local calibration.

Temperature schedule: tau = max(0.1, 1.0 - iter * 0.9 / 1500)
"""
import sys, os, math, time
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

# ─── Config ─────────────────────────────────────────────────────────────────

N = 256; n = 8
KU = 123; KV = 123
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)

BATCH = 4
LR = 1e-5
TOTAL_ITERS = 2000
EVAL_EVERY = 500
EVAL_CW = 500
PRINT_EVERY = 100
DETACH_EVERY = 20  # detach gradient chain every K info positions

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')


# ─── Model (same architecture as campaign) ──────────────────────────────────

class SimpleMLP_Gmac(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = 16
        self.z_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ELU(), nn.Linear(32, 16))
        self.tree = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Gumbel-Softmax forward pass ────────────────────────────────────────────

def forward_gumbel(model, z, b, frozen_u, frozen_v, u_true, v_true,
                   tau=1.0, p_gumbel=1.0, hard=False, rng=None,
                   detach_every=DETACH_EVERY):
    """
    Modified forward pass using Gumbel-Softmax leaf embeddings.

    For non-frozen info positions:
      - With probability p_gumbel: leaf embedding = logits2emb(gumbel_softmax(logits))
        (differentiable soft decision based on model's own prediction)
      - With probability (1-p_gumbel): leaf embedding = _make_leaf_emb(true bit)
        (standard teacher forcing)

    Loss is always CE(logits, true_target) regardless of which embedding is used.

    To keep backward tractable, Gumbel embeddings are detached every
    `detach_every` info positions.

    Parameters
    ----------
    tau : float -- Gumbel-Softmax temperature (1.0=soft, 0.1=nearly hard)
    p_gumbel : float -- probability of using Gumbel embedding vs teacher forcing
    hard : bool -- if True, use straight-through estimator
    rng : np.random.Generator -- for mixed mode coin flips
    detach_every : int -- detach gradient chain every K info positions
    """
    tree = model.tree
    B, N_ = z.shape
    device = z.device
    d = tree.d

    br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
    root = model.z_encoder(z.unsqueeze(-1))[:, br]

    edge_data = [None] * (2 * N)
    edge_data[1] = root

    no_info = tree.no_info_emb.unsqueeze(0).unsqueeze(0)
    for beta in range(2, 2 * N):
        level = beta.bit_length() - 1
        size = N >> level
        edge_data[beta] = no_info.expand(B, size, d).clone()

    dec_head = 1
    u_hat, v_hat = {}, {}
    all_logits, all_targets = [], []
    i_u, i_v = 0, 0
    gumbel_count = 0  # count consecutive Gumbel positions for detach

    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u; fdict = frozen_u
        else:
            i_v += 1; i_t = i_v; fdict = frozen_v

        leaf_edge = i_t + N - 1
        target_vtx = leaf_edge >> 1

        dec_head = tree._step_to(dec_head, target_vtx, edge_data, None)

        temp = edge_data[leaf_edge][:, 0].clone()

        if leaf_edge & 1 == 0:
            tree._neural_calc_left(target_vtx, edge_data)
        else:
            tree._neural_calc_right(target_vtx, edge_data)
        top_down = edge_data[leaf_edge][:, 0]

        if tree.use_combine_nn:
            combined = tree.combine_nn(torch.cat([top_down, temp], dim=-1))
        else:
            combined = top_down + temp
        logits = tree.emb2logits(combined)

        if i_t in fdict:
            # Frozen: use known value
            bit = torch.full((B,), fdict[i_t], dtype=torch.float32, device=device)
            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit
            new_emb = tree._make_leaf_emb(
                u_hat.get(i_t), v_hat.get(i_t), B, device)
            edge_data[leaf_edge] = new_emb.unsqueeze(1)
            gumbel_count = 0  # reset on frozen
        else:
            # Non-frozen: collect loss, decide leaf embedding
            target = (u_true[:, i_t - 1] * 2 + v_true[:, i_t - 1]).long()
            all_logits.append(logits)
            all_targets.append(target)

            true_bit = u_true[:, i_t - 1] if gamma == 0 else v_true[:, i_t - 1]

            # Decide: Gumbel or teacher forcing?
            use_gumbel_here = (rng is None) or (rng.random() < p_gumbel)

            if use_gumbel_here and p_gumbel > 0:
                gumbel_count += 1
                # Gumbel-perturbed soft log-probs (avoid exp->log roundtrip)
                # Add Gumbel noise, divide by tau, then log_softmax
                gumbel_noise = -torch.log(-torch.log(
                    torch.rand_like(logits).clamp(min=1e-10) + 1e-10) + 1e-10)
                perturbed = (logits + gumbel_noise) / tau
                soft_lp = F.log_softmax(perturbed, dim=-1)
                if hard:
                    # Straight-through: hard forward, soft backward
                    idx = soft_lp.argmax(dim=-1, keepdim=True)
                    hard_lp = torch.full_like(soft_lp, -30.0)
                    hard_lp.scatter_(1, idx, 0.0)
                    soft_lp = hard_lp - soft_lp.detach() + soft_lp
                leaf_emb = tree.logits2emb(soft_lp)

                # Detach periodically to limit backward graph
                if detach_every > 0 and gumbel_count % detach_every == 0:
                    leaf_emb = leaf_emb.detach()

                if gamma == 0:
                    u_hat[i_t] = true_bit
                else:
                    v_hat[i_t] = true_bit

                edge_data[leaf_edge] = leaf_emb.unsqueeze(1)
            else:
                # Teacher forcing (standard)
                gumbel_count = 0
                bit = true_bit
                if gamma == 0:
                    u_hat[i_t] = bit
                else:
                    v_hat[i_t] = bit
                new_emb = tree._make_leaf_emb(
                    u_hat.get(i_t), v_hat.get(i_t), B, device)
                edge_data[leaf_edge] = new_emb.unsqueeze(1)

    return all_logits, all_targets


# ─── Evaluation (greedy decode, no teacher forcing) ─────────────────────────

def evaluate(model, channel, b, Au, Av, fu, fv, n_cw):
    model.eval()
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    errs = 0; total = 0
    eval_rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(8, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = eval_rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = eval_rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            root = model.z_encoder(zf.unsqueeze(-1))[:, br]
            _, _, uh, vh, _ = model.tree(
                z=None, b=b, frozen_u=fu, frozen_v=fv, root_emb=root)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


# ─── Calibration measurement ────────────────────────────────────────────────

def measure_calibration(model, channel, b, Au, Av, fu, fv, n_cw=300):
    """
    Measure expected calibration error (ECE).
    For each non-frozen position, compare predicted confidence to actual accuracy.
    """
    model.eval()
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    all_confs = []
    all_correct = []
    eval_rng = np.random.default_rng(1234)

    with torch.no_grad():
        total = 0
        while total < n_cw:
            actual = min(8, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = eval_rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = eval_rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            root = model.z_encoder(zf.unsqueeze(-1))[:, br]
            logits_list, targets_list, _, _, _ = model.tree(
                z=None, b=b, frozen_u=fu, frozen_v=fv,
                u_true=torch.from_numpy(uf).float(),
                v_true=torch.from_numpy(vf).float(),
                root_emb=root)

            if logits_list:
                logits = torch.stack(logits_list)  # (T, B, 4)
                targets = torch.stack(targets_list)  # (T, B)
                probs = F.softmax(logits, dim=-1)  # (T, B, 4)
                confs, preds = probs.max(dim=-1)  # (T, B)
                correct = (preds == targets).float()
                all_confs.append(confs.cpu().flatten())
                all_correct.append(correct.cpu().flatten())
            total += actual

    model.train()
    confs = torch.cat(all_confs)
    correct = torch.cat(all_correct)

    # ECE with 10 bins
    n_bins = 10
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() > 0:
            avg_conf = confs[mask].mean().item()
            avg_acc = correct[mask].mean().item()
            ece += mask.sum().item() / len(confs) * abs(avg_conf - avg_acc)

    avg_conf = confs.mean().item()
    avg_acc = correct.mean().item()
    return ece, avg_conf, avg_acc


# ─── Training loop for one experiment ───────────────────────────────────────

def run_experiment(name, model, channel, b, Au, Av, fu, fv, baseline_bler,
                   p_gumbel_fn, tau_fn, hard=False, rng=None):
    """
    Run one Gumbel training experiment.

    p_gumbel_fn: callable(it) -> float
    tau_fn: callable(it) -> float
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    train_rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    results_at_eval = []

    for it in range(1, TOTAL_ITERS + 1):
        tau = tau_fn(it)
        p_gumbel = p_gumbel_fn(it)

        # Generate data
        uf = np.zeros((BATCH, N), dtype=int)
        vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = train_rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = train_rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        logits_list, targets_list = forward_gumbel(
            model, zf, b, fu, fv,
            torch.from_numpy(uf).float(),
            torch.from_numpy(vf).float(),
            tau=tau, p_gumbel=p_gumbel, hard=hard, rng=rng)

        if logits_list:
            loss = F.cross_entropy(
                torch.stack(logits_list).reshape(-1, 4),
                torch.stack(targets_list).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        if it % PRINT_EVERY == 0:
            avg_loss = np.mean(losses[-100:]) if losses else 0
            elapsed = time.time() - t0
            print(f"  [{it:>5}/{TOTAL_ITERS}] loss={avg_loss:.4f} tau={tau:.3f} "
                  f"p={p_gumbel:.3f} ({elapsed:.0f}s)", flush=True)

        if it % EVAL_EVERY == 0:
            bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW)
            ece, conf, acc = measure_calibration(model, channel, b, Au, Av, fu, fv)
            results_at_eval.append((it, bler, ece, conf, acc))
            print(f"  >>> BLER={bler:.4f} (base={baseline_bler:.4f}) "
                  f"ECE={ece:.4f} conf={conf:.4f} acc={acc:.4f} "
                  f"gap={conf-acc:+.4f}", flush=True)

    # Final eval
    bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW)
    ece, conf, acc = measure_calibration(model, channel, b, Au, Av, fu, fv)
    elapsed = time.time() - t0
    print(f"  FINAL: BLER={bler:.4f} ECE={ece:.4f} conf={conf:.4f} "
          f"acc={acc:.4f} gap={conf-acc:+.4f} ({elapsed:.0f}s)", flush=True)
    return bler, ece, conf, acc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("POC: Gumbel-Softmax Training for Better Probability Calibration")
    print("=" * 70)
    print(f"N={N}, KU={KU}, KV={KV}, SNR={SNR_DB}dB")
    print(f"batch={BATCH}, lr={LR}, iters={TOTAL_ITERS}, detach_every={DETACH_EVERY}")

    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv, _, _, _ = design_from_file(
        os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz'), n, KU, KV)
    b = make_path(N, N // 2)

    # ── Load baseline model ─────────────────────────────────────────────
    ckpt_path = os.path.join(SAVE_DIR, 'campaign_n256_sched_best.pt')
    print(f"\nLoading checkpoint: {ckpt_path}")

    def make_model():
        m = SimpleMLP_Gmac()
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        m.load_state_dict(sd, strict=False)
        return m

    baseline = make_model()
    baseline.eval()
    n_info = 2*N - len(fu) - len(fv)
    print(f"Parameters: {baseline.count_parameters():,}, info positions: {n_info}")

    # ── Baseline evaluation ─────────────────────────────────────────────
    print("\n--- Baseline (pre-trained, no Gumbel) ---")
    baseline_bler = evaluate(baseline, channel, b, Au, Av, fu, fv, EVAL_CW)
    baseline_ece, baseline_conf, baseline_acc = measure_calibration(
        baseline, channel, b, Au, Av, fu, fv)
    print(f"  BLER:       {baseline_bler:.4f}")
    print(f"  ECE:        {baseline_ece:.4f}")
    print(f"  Avg conf:   {baseline_conf:.4f}")
    print(f"  Avg acc:    {baseline_acc:.4f}")
    print(f"  Conf - Acc: {baseline_conf - baseline_acc:+.4f} "
          f"({'overconfident' if baseline_conf > baseline_acc else 'underconfident'})")

    t_total = time.time()

    # Temperature schedule for all experiments
    def tau_fn(it):
        return max(0.1, 1.0 - it * 0.9 / 1500)

    # ══════════════════════════════════════════════════════════════════════
    #  EXPERIMENT 1: Pure Gumbel-Softmax training (soft)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Pure Gumbel-Softmax (p_gumbel=1.0, hard=False)")
    print("=" * 70)
    g1_bler, g1_ece, g1_conf, g1_acc = run_experiment(
        "gumbel_soft", make_model(), channel, b, Au, Av, fu, fv,
        baseline_bler, p_gumbel_fn=lambda it: 1.0, tau_fn=tau_fn, hard=False)

    # ══════════════════════════════════════════════════════════════════════
    #  EXPERIMENT 2: Mixed training (scheduled sampling + Gumbel)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Mixed Gumbel (p_gumbel: 0.0 -> 0.5 over 1K iters)")
    print("=" * 70)
    coin_rng = np.random.default_rng(123)
    g2_bler, g2_ece, g2_conf, g2_acc = run_experiment(
        "mixed", make_model(), channel, b, Au, Av, fu, fv,
        baseline_bler, p_gumbel_fn=lambda it: min(0.5, 0.5 * it / 1000),
        tau_fn=tau_fn, hard=False, rng=coin_rng)

    # ══════════════════════════════════════════════════════════════════════
    #  EXPERIMENT 3: Gumbel hard=True (straight-through estimator)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Gumbel hard=True (straight-through estimator)")
    print("=" * 70)
    g3_bler, g3_ece, g3_conf, g3_acc = run_experiment(
        "gumbel_hard", make_model(), channel, b, Au, Av, fu, fv,
        baseline_bler, p_gumbel_fn=lambda it: 1.0, tau_fn=tau_fn, hard=True)

    # ══════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    fmt = f"{'Method':<30} {'BLER':>8} {'ECE':>8} {'Conf':>8} {'Acc':>8} {'Gap':>8}"
    print(fmt)
    print("-" * 70)
    rows = [
        ("Baseline (teacher forced)", baseline_bler, baseline_ece, baseline_conf, baseline_acc),
        ("Gumbel soft (p=1.0)", g1_bler, g1_ece, g1_conf, g1_acc),
        ("Mixed (p: 0->0.5)", g2_bler, g2_ece, g2_conf, g2_acc),
        ("Gumbel hard (STE)", g3_bler, g3_ece, g3_conf, g3_acc),
    ]
    for name, bler, ece, conf, acc in rows:
        print(f"{name:<30} {bler:>8.4f} {ece:>8.4f} {conf:>8.4f} {acc:>8.4f} {conf-acc:>+8.4f}")

    print(f"\nTotal time: {elapsed/60:.1f} min")

    results = {
        'baseline': {'bler': baseline_bler, 'ece': baseline_ece},
        'gumbel_soft': {'bler': g1_bler, 'ece': g1_ece},
        'mixed': {'bler': g2_bler, 'ece': g2_ece},
        'gumbel_hard': {'bler': g3_bler, 'ece': g3_ece},
    }
    best_method = min(results, key=lambda k: results[k]['bler'])
    print(f"\nBest BLER method: {best_method} (BLER={results[best_method]['bler']:.4f})")

    if results[best_method]['bler'] < baseline_bler:
        print(">>> Gumbel-Softmax IMPROVED BLER over baseline!")
    else:
        print(">>> Gumbel-Softmax did NOT improve BLER over baseline.")

    best_ece_method = min(['gumbel_soft', 'mixed', 'gumbel_hard'],
                          key=lambda k: results[k]['ece'])
    if results[best_ece_method]['ece'] < baseline_ece:
        print(f">>> Best calibration: {best_ece_method} "
              f"(ECE={results[best_ece_method]['ece']:.4f} vs baseline={baseline_ece:.4f})")
    else:
        print(">>> No calibration improvement observed.")


if __name__ == '__main__':
    main()
