#!/usr/bin/env python3
"""
n256_mi_diagnostic.py

Per-position mutual-information diagnostic for the CG decoder at N=256.

For each non-frozen step in the tree walk, this script computes:
  - 4-class cross-entropy against the true (u,v) target
  - Marginalized binary cross-entropy for the active user at that step
  - Mutual information: MI = 1 - BCE_binary / log(2)

These are compared to the genie-aided per-position Pe from the design file.

Output:
  results/n256_mi_diagnostic.json
  results/n256_mi_diagnostic.md
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 256
n_log = 8
ku = kv = 123
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
SEED = 42
N_CW = 5000
BATCH = 64


def load_nn(ckpt_path, d=16, hidden=64):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model = GmacNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=2)
    model_sd = model.state_dict()
    for k, v in sd.items():
        nk = k
        if nk.startswith('tree.'):
            nk = nk[5:]
        elif nk.startswith('z_enc.'):
            nk = 'z_encoder.' + nk[6:]
        if 'embedding_z' in nk:
            continue
        if nk in model_sd and model_sd[nk].shape == v.shape:
            model_sd[nk] = v
    model.load_state_dict(model_sd)
    model.eval()
    return model


def H_binary(p):
    """Binary entropy in bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def main():
    os.chdir(ROOT)

    # ── Load design ───────────────────────────────────────────────────────
    Au, Av, frozen_u, frozen_v, pe_u_raw, pe_v_raw, path_i = design_from_file(
        f'designs/gmac_B_n{n_log}_snr6dB.npz', n_log, ku=ku, kv=kv
    )
    b = make_path(N, path_i)
    channel = GaussianMAC(sigma2=SIGMA2)

    Au_set = set(Au)
    Av_set = set(Av)

    # pe_u_raw, pe_v_raw are 0-indexed, shape (256,)
    # pe_u_raw[i] = genie Pe for user U at 0-indexed position i
    # Our positions are 1-indexed: pe for 1-indexed p → pe_u_raw[p-1]

    # ── Build step metadata ───────────────────────────────────────────────
    # Walk the path and record for each non-frozen step:
    #   - which 1-indexed position it corresponds to
    #   - which user (0=U, 1=V)
    #   - position type (both_info, u_only, v_only)
    step_meta = []
    i_u, i_v = 0, 0
    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1
            i_t = i_u
            fdict = frozen_u
        else:
            i_v += 1
            i_t = i_v
            fdict = frozen_v

        if i_t in fdict:
            continue  # frozen — no logits emitted

        # Position type
        if i_t in Au_set and i_t in Av_set:
            pos_type = 'both_info'
        elif i_t in Au_set:
            pos_type = 'u_only'
        elif i_t in Av_set:
            pos_type = 'v_only'
        else:
            pos_type = 'unknown'

        step_meta.append({
            'step': step,
            'pos': i_t,       # 1-indexed
            'user': gamma,    # 0=U, 1=V
            'pos_type': pos_type,
            'pe_u': float(pe_u_raw[i_t - 1]),
            'pe_v': float(pe_v_raw[i_t - 1]),
        })

    n_info_steps = len(step_meta)
    print(f"Total non-frozen steps: {n_info_steps}")
    print(f"  both_info: {sum(1 for s in step_meta if s['pos_type']=='both_info')}")
    print(f"  u_only: {sum(1 for s in step_meta if s['pos_type']=='u_only')}")
    print(f"  v_only: {sum(1 for s in step_meta if s['pos_type']=='v_only')}")

    # ── Generate test data ────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    U_info = rng.integers(0, 2, size=(N_CW, ku), dtype=np.int32)
    V_info = rng.integers(0, 2, size=(N_CW, kv), dtype=np.int32)
    U_msg = build_message_batch(N, U_info, Au)
    V_msg = build_message_batch(N, V_info, Av)
    X = polar_encode_batch(U_msg)
    Y = polar_encode_batch(V_msg)
    np.random.seed(SEED + 7919)
    Z = channel.sample_batch(X, Y).astype(np.float32)

    # ── Load model ────────────────────────────────────────────────────────
    ckpt = 'saved_models/ncg_gmac_mlp_N256.pt'
    model = load_nn(ckpt)
    print(f"Loaded {ckpt}")

    # ── Run teacher-forced forward and accumulate per-step CE ─────────────
    # Accumulators: sum of CE per step, and count
    ce_4class_sum = np.zeros(n_info_steps, dtype=np.float64)
    ce_user_sum = np.zeros(n_info_steps, dtype=np.float64)
    correct_user_count = np.zeros(n_info_steps, dtype=np.int64)
    count = np.zeros(n_info_steps, dtype=np.int64)

    t0 = time.time()
    for start in range(0, N_CW, BATCH):
        end = min(start + BATCH, N_CW)
        bs = end - start
        z_t = torch.from_numpy(Z[start:end]).float()
        u_t = torch.from_numpy(U_msg[start:end].astype(np.float32))
        v_t = torch.from_numpy(V_msg[start:end].astype(np.float32))

        with torch.no_grad():
            all_logits, all_targets, _, _, _ = model(
                z_t, b, frozen_u, frozen_v, u_true=u_t, v_true=v_t
            )

        # all_logits: list of (bs, 4) tensors, one per non-frozen step
        # all_targets: list of (bs,) long tensors
        assert len(all_logits) == n_info_steps, \
            f"Expected {n_info_steps} logit steps, got {len(all_logits)}"

        for idx in range(n_info_steps):
            logits = all_logits[idx]  # (bs, 4)
            targets = all_targets[idx]  # (bs,)

            # 4-class CE (per sample, then sum)
            ce4 = F.cross_entropy(logits, targets, reduction='none')  # (bs,)
            ce_4class_sum[idx] += ce4.sum().item()

            # Marginalize to get per-user binary CE
            log_probs = F.log_softmax(logits, dim=-1)  # (bs, 4)
            gamma = step_meta[idx]['user']
            if gamma == 0:  # U-step: marginalize over v
                # P(u=0) = P(class 0) + P(class 1)
                # P(u=1) = P(class 2) + P(class 3)
                lp_u0 = torch.logsumexp(log_probs[:, :2], dim=1)
                lp_u1 = torch.logsumexp(log_probs[:, 2:], dim=1)
                u_true_step = u_t[:, step_meta[idx]['pos'] - 1].long()
                ce_u = -torch.where(u_true_step == 0, lp_u0, lp_u1)
                ce_user_sum[idx] += ce_u.sum().item()
                pred = (lp_u1 > lp_u0).long()
                correct_user_count[idx] += (pred == u_true_step).sum().item()
            else:  # V-step: marginalize over u
                # P(v=0) = P(class 0) + P(class 2)
                # P(v=1) = P(class 1) + P(class 3)
                lp_v0 = torch.logsumexp(log_probs[:, [0, 2]], dim=1)
                lp_v1 = torch.logsumexp(log_probs[:, [1, 3]], dim=1)
                v_true_step = v_t[:, step_meta[idx]['pos'] - 1].long()
                ce_v = -torch.where(v_true_step == 0, lp_v0, lp_v1)
                ce_user_sum[idx] += ce_v.sum().item()
                pred = (lp_v1 > lp_v0).long()
                correct_user_count[idx] += (pred == v_true_step).sum().item()

            count[idx] += bs

        if (start // BATCH) % 10 == 0:
            print(f"  {end}/{N_CW}", flush=True)

    elapsed = time.time() - t0
    print(f"Forward pass done in {elapsed:.1f}s")

    # ── Compute MI per step ───────────────────────────────────────────────
    mean_ce4 = ce_4class_sum / count
    mean_ce_user = ce_user_sum / count
    accuracy = correct_user_count / count

    # MI in bits
    # For binary user prediction: MI = 1 - BCE / log(2)
    # (max MI = 1 bit when BCE = 0; MI = 0 when BCE = log(2))
    mi_user = 1.0 - mean_ce_user / math.log(2)
    mi_user = np.clip(mi_user, 0, 1)

    # 4-class MI: MI = log2(4) - CE/log(2) = 2 - CE/log(2)
    mi_4class = 2.0 - mean_ce4 / math.log(2)
    mi_4class = np.clip(mi_4class, 0, 2)

    # Genie MI per step
    genie_mi = np.zeros(n_info_steps)
    for idx, meta in enumerate(step_meta):
        pe = meta['pe_u'] if meta['user'] == 0 else meta['pe_v']
        genie_mi[idx] = 1.0 - H_binary(pe)

    mi_gap = genie_mi - mi_user  # positive = model is worse than genie

    # ── Enrich step_meta with computed values ─────────────────────────────
    for idx in range(n_info_steps):
        step_meta[idx].update({
            'mi_user': float(mi_user[idx]),
            'mi_4class': float(mi_4class[idx]),
            'genie_mi': float(genie_mi[idx]),
            'mi_gap': float(mi_gap[idx]),
            'accuracy': float(accuracy[idx]),
            'mean_ce4': float(mean_ce4[idx]),
            'mean_ce_user': float(mean_ce_user[idx]),
        })

    # ── Sort by MI gap (largest first) ────────────────────────────────────
    sorted_by_gap = sorted(step_meta, key=lambda x: -x['mi_gap'])

    # ── Tree depth for each position ──────────────────────────────────────
    # In the SC tree walk, the position i_t maps to leaf edge i_t + N - 1.
    # The "depth" of the leaf in the binary tree = log2(N) = 8 for all leaves.
    # But conceptually, the "effective depth" of a position in the polar
    # tree relates to how early it is decoded (step index).
    # Let's also compute the tree-walk step index for correlation.

    # ── Summary statistics ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("N=256 Per-Position MI Diagnostic")
    print("=" * 70)
    print(f"Model: {ckpt}, d=16, hidden=64")
    print(f"Test: {N_CW} codewords, seed={SEED}, SNR={SNR_DB}dB")
    print(f"Total info steps: {n_info_steps}")

    # Distribution of MI gaps
    bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    print("\nMI gap distribution (genie - model, in bits):")
    for i in range(len(bins) - 1):
        count_bin = sum(1 for g in mi_gap if bins[i] <= g < bins[i + 1])
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count_bin} steps")
    count_neg = sum(1 for g in mi_gap if g < 0)
    print(f"  < 0 (model > genie): {count_neg} steps")

    # Overall stats
    print(f"\nMI gap: mean={mi_gap.mean():.4f}, median={np.median(mi_gap):.4f}, "
          f"max={mi_gap.max():.4f}")
    print(f"User accuracy: mean={accuracy.mean():.4f}, min={accuracy.min():.4f}")
    print(f"User MI: mean={mi_user.mean():.4f}, min={mi_user.min():.4f}")

    # Top 30 worst gaps
    print(f"\nTop 30 worst MI gaps:")
    print(f"{'step':<6}{'pos':<6}{'user':<6}{'type':<12}"
          f"{'genie_MI':>10}{'model_MI':>10}{'gap':>8}{'acc':>8}{'pe':>8}")
    print("-" * 76)
    for s in sorted_by_gap[:30]:
        user = 'U' if s['user'] == 0 else 'V'
        pe = s['pe_u'] if s['user'] == 0 else s['pe_v']
        print(f"{s['step']:<6}{s['pos']:<6}{user:<6}{s['pos_type']:<12}"
              f"{s['genie_mi']:>10.4f}{s['mi_user']:>10.4f}"
              f"{s['mi_gap']:>8.4f}{s['accuracy']:>8.4f}{pe:>8.4f}")

    # Check correlation with step index (early vs late in the walk)
    steps_arr = np.array([s['step'] for s in step_meta])
    from numpy import corrcoef
    corr = corrcoef(steps_arr, mi_gap)[0, 1]
    print(f"\nCorrelation(step_index, MI_gap) = {corr:.4f}")

    # Check correlation with genie Pe
    pe_arr = np.array([s['pe_u'] if s['user'] == 0 else s['pe_v']
                       for s in step_meta])
    corr_pe = corrcoef(pe_arr, mi_gap)[0, 1]
    print(f"Correlation(genie_Pe, MI_gap) = {corr_pe:.4f}")

    # ── By position type ──────────────────────────────────────────────────
    for ptype in ['both_info', 'u_only', 'v_only']:
        mask = [s['pos_type'] == ptype for s in step_meta]
        g = mi_gap[mask]
        if len(g) > 0:
            print(f"\n{ptype}: n={len(g)}, mean_gap={g.mean():.4f}, "
                  f"max_gap={g.max():.4f}, mean_acc={accuracy[mask].mean():.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    report = {
        'config': {
            'N': N, 'ku': ku, 'kv': kv, 'n_cw': N_CW, 'seed': SEED,
            'SNR_dB': SNR_DB, 'checkpoint': ckpt,
        },
        'summary': {
            'n_info_steps': n_info_steps,
            'mi_gap_mean': float(mi_gap.mean()),
            'mi_gap_median': float(np.median(mi_gap)),
            'mi_gap_max': float(mi_gap.max()),
            'mi_gap_std': float(mi_gap.std()),
            'accuracy_mean': float(accuracy.mean()),
            'accuracy_min': float(accuracy.min()),
            'mi_user_mean': float(mi_user.mean()),
            'corr_step_gap': float(corr),
            'corr_pe_gap': float(corr_pe),
        },
        'worst_30': sorted_by_gap[:30],
        'all_steps': step_meta,
    }
    with open('results/n256_mi_diagnostic.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\nSaved results/n256_mi_diagnostic.json")

    # ── Write markdown report ─────────────────────────────────────────────
    with open('results/n256_mi_diagnostic.md', 'w') as f:
        f.write("# N=256 Per-Position MI Diagnostic\n\n")
        f.write(f"**Model:** `{ckpt}` (d=16, hidden=64, 39K params)\n")
        f.write(f"**Test:** {N_CW} codewords, seed={SEED}, SNR={SNR_DB} dB, "
                f"Class B symmetric\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total non-frozen steps:** {n_info_steps}\n")
        f.write(f"- **Mean MI gap (genie - model):** {mi_gap.mean():.4f} bits\n")
        f.write(f"- **Max MI gap:** {mi_gap.max():.4f} bits\n")
        f.write(f"- **Mean per-user accuracy:** {accuracy.mean():.4f}\n")
        f.write(f"- **Min per-user accuracy:** {accuracy.min():.4f}\n")
        f.write(f"- **Correlation(step_index, gap):** {corr:.4f}\n")
        f.write(f"- **Correlation(genie_Pe, gap):** {corr_pe:.4f}\n\n")

        f.write("## MI gap distribution\n\n")
        f.write("| gap range | count |\n")
        f.write("|-----------|-------|\n")
        for i in range(len(bins) - 1):
            c = sum(1 for g in mi_gap if bins[i] <= g < bins[i + 1])
            f.write(f"| [{bins[i]:.2f}, {bins[i+1]:.2f}) | {c} |\n")
        f.write(f"| < 0 (model > genie) | {count_neg} |\n\n")

        f.write("## By position type\n\n")
        f.write("| type | n | mean gap | max gap | mean acc |\n")
        f.write("|------|---|----------|---------|----------|\n")
        for ptype in ['both_info', 'u_only', 'v_only']:
            mask = np.array([s['pos_type'] == ptype for s in step_meta])
            g = mi_gap[mask]
            a = accuracy[mask]
            if len(g) > 0:
                f.write(f"| {ptype} | {len(g)} | {g.mean():.4f} | "
                        f"{g.max():.4f} | {a.mean():.4f} |\n")

        f.write("\n## Top 30 worst MI gaps\n\n")
        f.write("| step | pos | user | type | genie MI | model MI | gap | acc | pe |\n")
        f.write("|------|-----|------|------|----------|----------|-----|-----|----|\n")
        for s in sorted_by_gap[:30]:
            user = 'U' if s['user'] == 0 else 'V'
            pe = s['pe_u'] if s['user'] == 0 else s['pe_v']
            f.write(f"| {s['step']} | {s['pos']} | {user} | {s['pos_type']} | "
                    f"{s['genie_mi']:.4f} | {s['mi_user']:.4f} | {s['mi_gap']:.4f} | "
                    f"{s['accuracy']:.4f} | {pe:.4f} |\n")

        f.write("\n## Interpretation\n\n")
        # Will be filled after seeing results
        f.write("(See console output for full analysis)\n")

    print("Saved results/n256_mi_diagnostic.md")


if __name__ == '__main__':
    main()
