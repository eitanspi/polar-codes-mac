#!/usr/bin/env python3
"""
train_allinfo_n256.py

Train the CG decoder at N=256 with ALL positions as info (ku=kv=256).
This is above capacity (rate 1.0/user vs capacity ~0.48/user), so the
model will fail at many positions. The per-position MI after training
tells us which positions the CG decoder finds easy vs hard — the basis
for CG-optimal frozen set design.

After training, measures per-position MI and saves a ranking.
"""
import os, sys, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from polar.encoder import polar_encode_batch, build_message_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

# ─── Config ──────────────────────────────────────────────────────────────

N = 256
n_log = 8
SNR_DB = 12.0  # high SNR so rate 2.0 is below capacity
SIGMA2 = 10 ** (-SNR_DB / 10)

# ALL positions are info — no frozen bits
ku = N  # 256
kv = N  # 256
PATH_I = N // 2  # 128 for Class B

LR = 1e-3
BATCH = 16
TOTAL_ITERS = 10000
MI_EVAL_EVERY = 2000
MI_EVAL_CW = 500  # codewords for MI measurement
CKPT_OUT = 'saved_models/n256_allinfo_12dB.pt'


def main():
    os.chdir(ROOT)

    print("=" * 60)
    print(f"All-Info Training — N={N}, ku={ku}, kv={kv}")
    print(f"SNR={SNR_DB}dB, LR={LR}, batch={BATCH}")
    print("=" * 60)

    b = make_path(N, PATH_I)
    channel = GaussianMAC(sigma2=SIGMA2)

    # All positions are info → no frozen positions
    Au = list(range(1, N + 1))
    Av = list(range(1, N + 1))
    frozen_u = {}
    frozen_v = {}

    # Build model from scratch (random init)
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    print(f"Model: {model.count_parameters()} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_ITERS, eta_min=LR / 10)

    model.train()
    losses = []
    t0 = time.time()

    # ─── Build step metadata (which step → which position/user) ───────
    step_meta = []
    i_u, i_v = 0, 0
    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1
            i_t = i_u
        else:
            i_v += 1
            i_t = i_v
        # No frozen positions, so every step is info
        step_meta.append({'step': step, 'pos': i_t, 'user': gamma})
    n_info_steps = len(step_meta)
    print(f"Total info steps: {n_info_steps} (should be {2*N})")

    for it in range(1, TOTAL_ITERS + 1):
        # Generate batch — all positions are info
        rng = np.random.default_rng(it * 1000 + 42)
        U_msg = rng.integers(0, 2, size=(BATCH, N), dtype=np.int32)
        V_msg = rng.integers(0, 2, size=(BATCH, N), dtype=np.int32)
        X = polar_encode_batch(U_msg)
        Y = polar_encode_batch(V_msg)
        np.random.seed(it * 1000 + 43)
        Z = channel.sample_batch(X, Y).astype(np.float32)

        z_t = torch.from_numpy(Z).float()
        u_t = torch.from_numpy(U_msg.astype(np.float32))
        v_t = torch.from_numpy(V_msg.astype(np.float32))

        all_logits, all_targets, _, _, _ = model(
            z_t, b, frozen_u, frozen_v, u_true=u_t, v_true=v_t
        )

        logits_cat = torch.cat([l.unsqueeze(0) for l in all_logits], dim=0)
        targets_cat = torch.cat([t.unsqueeze(0) for t in all_targets], dim=0)
        loss = F.cross_entropy(logits_cat.view(-1, 4), targets_cat.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if it % 500 == 0:
            avg_loss = np.mean(losses[-500:])
            elapsed = time.time() - t0
            print(f"[{it:5d}/{TOTAL_ITERS}] loss={avg_loss:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} {elapsed:.0f}s",
                  flush=True)

        # Periodic MI evaluation
        if it % MI_EVAL_EVERY == 0 or it == TOTAL_ITERS:
            print(f"\n  === MI evaluation at iter {it} ===")
            _measure_mi(model, b, frozen_u, frozen_v, channel,
                        step_meta, n_info_steps, it)

    # Save final model
    torch.save(model.state_dict(), CKPT_OUT)
    print(f"\nSaved {CKPT_OUT}")

    # Final MI measurement and ranking
    print("\n" + "=" * 60)
    print("Final per-position MI ranking")
    print("=" * 60)
    mi_u, mi_v = _measure_mi(model, b, frozen_u, frozen_v, channel,
                              step_meta, n_info_steps, TOTAL_ITERS,
                              return_per_pos=True)

    # Save ranking
    ranking = {
        'config': {
            'N': N, 'ku_train': ku, 'kv_train': kv,
            'SNR_dB': SNR_DB, 'total_iters': TOTAL_ITERS,
        },
        'u_mi_per_position': {int(p): float(mi) for p, mi in mi_u.items()},
        'v_mi_per_position': {int(p): float(mi) for p, mi in mi_v.items()},
        # Sorted: best positions first
        'u_ranking': [int(p) for p, _ in sorted(mi_u.items(),
                                                  key=lambda x: -x[1])],
        'v_ranking': [int(p) for p, _ in sorted(mi_v.items(),
                                                  key=lambda x: -x[1])],
    }

    os.makedirs('results', exist_ok=True)
    with open('results/n256_allinfo_ranking.json', 'w') as f:
        json.dump(ranking, f, indent=2)
    print("Saved results/n256_allinfo_ranking.json")

    # Show top and bottom positions
    u_sorted = sorted(mi_u.items(), key=lambda x: -x[1])
    v_sorted = sorted(mi_v.items(), key=lambda x: -x[1])

    print(f"\nU positions — top 10 (best MI):")
    for p, mi in u_sorted[:10]:
        print(f"  pos {p:3d}: MI = {mi:.4f} bits")
    print(f"U positions — bottom 10 (worst MI):")
    for p, mi in u_sorted[-10:]:
        print(f"  pos {p:3d}: MI = {mi:.4f} bits")

    print(f"\nV positions — top 10 (best MI):")
    for p, mi in v_sorted[:10]:
        print(f"  pos {p:3d}: MI = {mi:.4f} bits")
    print(f"V positions — bottom 10 (worst MI):")
    for p, mi in v_sorted[-10:]:
        print(f"  pos {p:3d}: MI = {mi:.4f} bits")

    # Compare with genie design
    genie_data = np.load('designs/gmac_B_n8_snr6dB.npz')
    pe_u = genie_data['u_error_rates']
    pe_v = genie_data['v_error_rates']

    from polar.design_mc import design_from_file
    Au_genie, Av_genie, _, _, _, _, _ = design_from_file(
        'designs/gmac_B_n8_snr6dB.npz', n_log, ku=123, kv=123)

    # CG-optimal: top 123 by MI
    Au_cg = set(ranking['u_ranking'][:123])
    Av_cg = set(ranking['v_ranking'][:123])
    Au_genie_set = set(Au_genie)
    Av_genie_set = set(Av_genie)

    overlap_u = len(Au_cg & Au_genie_set)
    overlap_v = len(Av_cg & Av_genie_set)
    print(f"\nFrozen set comparison (ku=kv=123):")
    print(f"  U overlap: {overlap_u}/123 ({overlap_u/123*100:.1f}%)")
    print(f"  V overlap: {overlap_v}/123 ({overlap_v/123*100:.1f}%)")
    print(f"  U positions in CG but not genie: "
          f"{sorted(Au_cg - Au_genie_set)[:20]}...")
    print(f"  U positions in genie but not CG: "
          f"{sorted(Au_genie_set - Au_cg)[:20]}...")


def _measure_mi(model, b, frozen_u, frozen_v, channel,
                step_meta, n_info_steps, iteration,
                return_per_pos=False):
    """Measure per-step MI under teacher forcing on fresh codewords."""
    model.eval()
    N_size = N
    n_eval = MI_EVAL_CW

    rng = np.random.default_rng(77777 + iteration)
    U_msg = rng.integers(0, 2, size=(n_eval, N_size), dtype=np.int32)
    V_msg = rng.integers(0, 2, size=(n_eval, N_size), dtype=np.int32)
    X = polar_encode_batch(U_msg)
    Y = polar_encode_batch(V_msg)
    np.random.seed(77777 + iteration)
    Z = channel.sample_batch(X, Y).astype(np.float32)

    ce_sum = np.zeros(n_info_steps, dtype=np.float64)
    count = np.zeros(n_info_steps, dtype=np.int64)

    batch = 64
    for start in range(0, n_eval, batch):
        end = min(start + batch, n_eval)
        z_t = torch.from_numpy(Z[start:end]).float()
        u_t = torch.from_numpy(U_msg[start:end].astype(np.float32))
        v_t = torch.from_numpy(V_msg[start:end].astype(np.float32))
        with torch.no_grad():
            all_logits, all_targets, _, _, _ = model(
                z_t, b, frozen_u, frozen_v, u_true=u_t, v_true=v_t)
        for idx in range(len(all_logits)):
            log_probs = F.log_softmax(all_logits[idx], dim=-1)
            gamma = step_meta[idx]['user']
            pos = step_meta[idx]['pos']
            # Compute per-user binary CE
            if gamma == 0:
                lp0 = torch.logsumexp(log_probs[:, :2], dim=1)
                lp1 = torch.logsumexp(log_probs[:, 2:], dim=1)
                true_val = u_t[:, pos - 1].long()
            else:
                lp0 = torch.logsumexp(log_probs[:, [0, 2]], dim=1)
                lp1 = torch.logsumexp(log_probs[:, [1, 3]], dim=1)
                true_val = v_t[:, pos - 1].long()
            ce = -torch.where(true_val == 0, lp0, lp1)
            ce_sum[idx] += ce.sum().item()
            count[idx] += (end - start)

    mean_ce = ce_sum / count
    mi_bits = np.clip(1.0 - mean_ce / math.log(2), 0, 1)

    # Aggregate by user and position
    mi_u = {}  # pos → MI for U-steps
    mi_v = {}
    for idx, meta in enumerate(step_meta):
        pos = meta['pos']
        if meta['user'] == 0:
            mi_u[pos] = mi_bits[idx]
        else:
            mi_v[pos] = mi_bits[idx]

    avg_mi = mi_bits.mean()
    min_mi = mi_bits.min()
    n_low = (mi_bits < 0.5).sum()
    n_high = (mi_bits > 0.9).sum()
    print(f"  MI: avg={avg_mi:.4f}, min={min_mi:.4f}, "
          f"<0.5: {n_low}/{n_info_steps}, >0.9: {n_high}/{n_info_steps}")

    model.train()
    if return_per_pos:
        return mi_u, mi_v
    return None


if __name__ == '__main__':
    main()
