#!/usr/bin/env python3
"""
analyze_cg_bottleneck.py — Per-position MI analysis for the CG decoder at N=256 Class B.

Loads the trained n256_long_best.pt checkpoint, runs teacher-forced forward passes,
and measures per-position cross-entropy (CE) for each info position. Then compares
with the genie (MC) design error rates to identify positions where the CG decoder
struggles relative to the SC genie.

This is the key diagnostic for NPD-guided frozen set design: if positions that the
CG finds hard are different from what the genie says are hard, then CG-optimal design
could improve the CG decoder's BLER significantly (as demonstrated at N=32 Class C).
"""
import os, sys, math, time, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# ─── Config ──────────────────────────────────────────────────────────────
N = 256
n = 8
KU = 123
KV = 123
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
PATH_I = N // 2  # Class B

EVAL_CW = 2000  # codewords for MI measurement (batched)
EVAL_BATCH = 32

CKPT = 'saved_models/n256_long_best.pt'
DESIGN_FILE = f'designs/gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz'


class SimpleMLP_Gmac(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = 16
        self.z_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 32), torch.nn.ELU(), torch.nn.Linear(32, 16))
        self.tree = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    print("=" * 70)
    print(f"CG Decoder Per-Position MI Analysis — N={N} Class B")
    print(f"Checkpoint: {CKPT}")
    print(f"Design: {DESIGN_FILE}")
    print(f"SNR={SNR_DB}dB, ku={KU}, kv={KV}, eval codewords={EVAL_CW}")
    print("=" * 70)

    # Load design
    Au, Av, frozen_u, frozen_v, pe_u_genie, pe_v_genie, path_i_design = \
        design_from_file(DESIGN_FILE, n, KU, KV)
    b = make_path(N, PATH_I)
    channel = GaussianMAC(sigma2=SIGMA2)

    Au_set = set(Au)
    Av_set = set(Av)

    print(f"Genie design: path_i={path_i_design}, |Au|={len(Au)}, |Av|={len(Av)}")
    print(f"Path: {PATH_I} (Class B)")

    # Load model
    model = SimpleMLP_Gmac()
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    print(f"Model loaded: {model.count_parameters():,} params")

    br = torch.from_numpy(bit_reversal_perm(n)).long()

    # ─── Build step metadata ─────────────────────────────────────────────
    step_meta = []
    info_step_indices = []  # index into all_logits for info steps
    i_u, i_v = 0, 0
    info_idx = 0
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

        if i_t not in fdict:
            step_meta.append({
                'logit_idx': info_idx,
                'pos': i_t,
                'user': gamma,
                'step': step,
            })
            info_idx += 1

    n_info = len(step_meta)
    print(f"Info steps: {n_info}")

    # ─── Run teacher-forced forward passes ───────────────────────────────
    print(f"\nRunning teacher-forced evaluation ({EVAL_CW} codewords, batch {EVAL_BATCH})...")
    t0 = time.time()

    # For 4-class CE per info step
    ce4_sum = np.zeros(n_info, dtype=np.float64)
    # For per-user binary CE
    ce_binary_sum = np.zeros(n_info, dtype=np.float64)
    count = np.zeros(n_info, dtype=np.int64)

    rng = np.random.default_rng(42)

    for start in range(0, EVAL_CW, EVAL_BATCH):
        end = min(start + EVAL_BATCH, EVAL_CW)
        bs = end - start

        U_msg = np.zeros((bs, N), dtype=np.int32)
        V_msg = np.zeros((bs, N), dtype=np.int32)
        for p in Au:
            U_msg[:, p - 1] = rng.integers(0, 2, bs)
        for p in Av:
            V_msg[:, p - 1] = rng.integers(0, 2, bs)

        X = polar_encode_batch(U_msg)
        Y = polar_encode_batch(V_msg)
        Z = channel.sample_batch(X, Y).astype(np.float32)

        z_t = torch.from_numpy(Z).float()
        u_t = torch.from_numpy(U_msg.astype(np.float32))
        v_t = torch.from_numpy(V_msg.astype(np.float32))

        # Forward with teacher forcing
        root = model.z_encoder(z_t.unsqueeze(-1))[:, br]
        with torch.no_grad():
            all_logits, all_targets, _, _, _ = model.tree(
                z=None, b=b, frozen_u=frozen_u, frozen_v=frozen_v,
                u_true=u_t, v_true=v_t, root_emb=root)

        for meta in step_meta:
            idx = meta['logit_idx']
            logits = all_logits[idx]  # (bs, 4)
            target = all_targets[idx]  # (bs,)

            # 4-class CE
            ce4 = F.cross_entropy(logits, target, reduction='none')
            ce4_sum[idx] += ce4.sum().item()

            # Per-user binary CE
            log_probs = F.log_softmax(logits, dim=-1)
            gamma = meta['user']
            pos = meta['pos']
            if gamma == 0:
                lp0 = torch.logsumexp(log_probs[:, :2], dim=1)
                lp1 = torch.logsumexp(log_probs[:, 2:], dim=1)
                true_val = u_t[:, pos - 1].long()
            else:
                lp0 = torch.logsumexp(log_probs[:, [0, 2]], dim=1)
                lp1 = torch.logsumexp(log_probs[:, [1, 3]], dim=1)
                true_val = v_t[:, pos - 1].long()
            ce_bin = -torch.where(true_val == 0, lp0, lp1)
            ce_binary_sum[idx] += ce_bin.sum().item()
            count[idx] += bs

        if (start + EVAL_BATCH) % 256 == 0:
            print(f"  {start + EVAL_BATCH}/{EVAL_CW} done", flush=True)

    elapsed = time.time() - t0
    print(f"Evaluation done in {elapsed:.1f}s")

    # ─── Compute MI per position ─────────────────────────────────────────
    mean_ce_binary = ce_binary_sum / count
    mean_ce4 = ce4_sum / count
    mi_bits = np.clip(1.0 - mean_ce_binary / math.log(2), 0, 1)

    # ─── Aggregate by user and position ──────────────────────────────────
    u_positions = {}  # pos → {mi, ce4, ce_bin, step}
    v_positions = {}
    for meta in step_meta:
        idx = meta['logit_idx']
        pos = meta['pos']
        entry = {
            'pos': pos,
            'mi': float(mi_bits[idx]),
            'ce4': float(mean_ce4[idx]),
            'ce_bin': float(mean_ce_binary[idx]),
            'step': meta['step'],
        }
        if meta['user'] == 0:
            u_positions[pos] = entry
        else:
            v_positions[pos] = entry

    # ─── Compare with genie Pe ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Per-Position Analysis: CG MI vs Genie Pe")
    print("=" * 70)

    # Genie Pe for info positions (0-indexed in pe_u_genie)
    for user_label, positions, pe_genie, A_set in [
            ('U', u_positions, pe_u_genie, Au_set),
            ('V', v_positions, pe_v_genie, Av_set)]:
        print(f"\n--- User {user_label} ({len(positions)} info positions) ---")
        print(f"{'Pos':>4s} {'Step':>5s} {'CG_MI':>7s} {'CG_CE4':>7s} "
              f"{'Genie_Pe':>9s} {'Genie_MI':>9s} {'Gap':>7s}")

        entries = []
        for pos in sorted(positions.keys()):
            e = positions[pos]
            genie_pe = float(pe_genie[pos - 1]) if pe_genie is not None else 0.0
            genie_mi = max(0.0, 1.0 - genie_pe / math.log(2)) if genie_pe < 0.693 else 0.0
            # Simpler: genie_pe is error probability not CE. Approximate MI:
            # For BSC(p): MI = 1 - H(p) = 1 - (-p*log(p) - (1-p)*log(1-p))/log(2)
            if 0 < genie_pe < 1:
                H_p = -(genie_pe * math.log(genie_pe) + (1 - genie_pe) * math.log(1 - genie_pe))
                genie_mi = max(0.0, 1.0 - H_p / math.log(2))
            elif genie_pe == 0:
                genie_mi = 1.0
            else:
                genie_mi = 0.0

            gap = e['mi'] - genie_mi
            entries.append((pos, e, genie_pe, genie_mi, gap))

        # Sort by gap (ascending = worst first)
        entries.sort(key=lambda x: x[4])

        for pos, e, genie_pe, genie_mi, gap in entries[:15]:
            print(f"{pos:4d} {e['step']:5d} {e['mi']:7.4f} {e['ce4']:7.4f} "
                  f"{genie_pe:9.6f} {genie_mi:9.4f} {gap:+7.4f}")
        print("  ...")
        for pos, e, genie_pe, genie_mi, gap in entries[-5:]:
            print(f"{pos:4d} {e['step']:5d} {e['mi']:7.4f} {e['ce4']:7.4f} "
                  f"{genie_pe:9.6f} {genie_mi:9.4f} {gap:+7.4f}")

    # ─── Count bottleneck positions ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("Bottleneck Summary")
    print("=" * 70)

    all_mi = np.array([user_pos[p]['mi']
                       for user_pos in [u_positions, v_positions]
                       for p in sorted(user_pos.keys())])

    # Rebuild properly
    all_entries = []
    for user_label, positions, pe_genie in [
            ('U', u_positions, pe_u_genie),
            ('V', v_positions, pe_v_genie)]:
        for pos in sorted(positions.keys()):
            e = positions[pos]
            genie_pe = float(pe_genie[pos - 1]) if pe_genie is not None else 0.0
            if 0 < genie_pe < 1:
                H_p = -(genie_pe * math.log(genie_pe) + (1 - genie_pe) * math.log(1 - genie_pe))
                genie_mi = max(0.0, 1.0 - H_p / math.log(2))
            elif genie_pe == 0:
                genie_mi = 1.0
            else:
                genie_mi = 0.0
            all_entries.append({
                'user': user_label,
                'pos': pos,
                'cg_mi': e['mi'],
                'genie_mi': genie_mi,
                'genie_pe': genie_pe,
                'gap': e['mi'] - genie_mi,
                'ce4': e['ce4'],
                'step': e['step'],
            })

    all_cg_mi = np.array([e['cg_mi'] for e in all_entries])
    all_genie_mi = np.array([e['genie_mi'] for e in all_entries])
    all_gaps = np.array([e['gap'] for e in all_entries])

    print(f"Total info positions: {len(all_entries)}")
    print(f"CG MI:   mean={all_cg_mi.mean():.4f}, min={all_cg_mi.min():.4f}, "
          f"< 0.5: {(all_cg_mi < 0.5).sum()}, "
          f"< 0.8: {(all_cg_mi < 0.8).sum()}, "
          f"< 0.9: {(all_cg_mi < 0.9).sum()}")
    print(f"Genie MI: mean={all_genie_mi.mean():.4f}, min={all_genie_mi.min():.4f}, "
          f"< 0.5: {(all_genie_mi < 0.5).sum()}, "
          f"< 0.8: {(all_genie_mi < 0.8).sum()}, "
          f"< 0.9: {(all_genie_mi < 0.9).sum()}")
    print(f"Gap (CG - genie): mean={all_gaps.mean():.4f}, "
          f"min={all_gaps.min():.4f}, max={all_gaps.max():.4f}")

    n_bottleneck = (all_gaps < -0.1).sum()
    n_severe = (all_gaps < -0.2).sum()
    print(f"\nPositions where CG is much worse than genie:")
    print(f"  Gap < -0.1: {n_bottleneck} positions")
    print(f"  Gap < -0.2: {n_severe} positions")

    # ─── CG-optimal design: what if we swap? ─────────────────────────────
    print("\n" + "=" * 70)
    print("CG-Optimal Design Analysis")
    print("=" * 70)

    # For U: rank all 256 positions by CG_MI from the all-info experiment
    # But we only have MI for info positions here. Instead, estimate:
    # frozen positions have MI=1.0 (no error possible since they're known).
    # So the question is: among the genie info positions, which ones does
    # the CG find hardest? Could swapping them for currently-frozen positions
    # help?

    # We can only compare within the current info set.
    # Let's find the worst info positions and see if they're borderline in the genie design.
    worst_entries = sorted(all_entries, key=lambda e: e['cg_mi'])

    print("\nWorst CG-MI info positions (candidates for freezing):")
    print(f"{'User':>4s} {'Pos':>4s} {'CG_MI':>7s} {'Genie_Pe':>9s} {'Genie_MI':>9s}")
    for e in worst_entries[:20]:
        print(f"{e['user']:>4s} {e['pos']:4d} {e['cg_mi']:7.4f} "
              f"{e['genie_pe']:9.6f} {e['genie_mi']:9.4f}")

    print("\nBest CG-MI info positions (most reliable):")
    for e in worst_entries[-10:]:
        print(f"{e['user']:>4s} {e['pos']:4d} {e['cg_mi']:7.4f} "
              f"{e['genie_pe']:9.6f} {e['genie_mi']:9.4f}")

    # ─── Identify what the genie thinks are the borderline frozen positions ──
    # These are the ones just outside the info set
    print("\nBorderline frozen positions (candidates for unfreezing):")
    print(f"{'User':>4s} {'Pos':>4s} {'Genie_Pe':>9s} {'Genie_MI':>9s} {'Status':>8s}")

    # Sort ALL positions by genie pe
    for user_label, A_set, pe_genie in [('U', Au_set, pe_u_genie), ('V', Av_set, pe_v_genie)]:
        sorted_pos = sorted(range(1, N + 1), key=lambda p: float(pe_genie[p - 1]))
        # The positions just outside the info set
        borderline = []
        for p in sorted_pos:
            if p not in A_set:
                gpe = float(pe_genie[p - 1])
                if 0 < gpe < 1:
                    H_p = -(gpe * math.log(gpe) + (1 - gpe) * math.log(1 - gpe))
                    gmi = max(0.0, 1.0 - H_p / math.log(2))
                elif gpe == 0:
                    gmi = 1.0
                else:
                    gmi = 0.0
                borderline.append((p, gpe, gmi))
        for p, gpe, gmi in borderline[:10]:
            print(f"{user_label:>4s} {p:4d} {gpe:9.6f} {gmi:9.4f} frozen")

    # ─── Estimate potential improvement ──────────────────────────────────
    print("\n" + "=" * 70)
    print("Estimated Impact of CG-Guided Design")
    print("=" * 70)

    # The worst CG-MI positions are likely causing most block errors.
    # If the worst info position has CG_MI=0.7 and the best frozen position
    # has genie_MI=0.95, swapping could reduce that position's error by ~5x.

    # Count positions where CG MI < some threshold
    for thresh in [0.95, 0.9, 0.8, 0.7, 0.5]:
        n_below = sum(1 for e in all_entries if e['cg_mi'] < thresh)
        print(f"  CG MI < {thresh}: {n_below} positions")

    # BLER is approximately dominated by the worst position
    worst_mi = min(e['cg_mi'] for e in all_entries)
    approx_worst_pe = max(0.001, 1.0 - worst_mi) if worst_mi < 1 else 0.5
    print(f"\n  Worst CG MI: {worst_mi:.4f} (approx pe ~ {approx_worst_pe:.4f})")
    print(f"  If we could freeze the worst positions and pick better ones,")
    print(f"  BLER should improve proportionally to the reduction in worst-case pe.")

    # ─── Save results ────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    results = {
        'config': {
            'N': N, 'ku': KU, 'kv': KV, 'snr_dB': SNR_DB,
            'path_i': PATH_I, 'eval_cw': EVAL_CW,
            'checkpoint': CKPT,
        },
        'per_position': all_entries,
        'summary': {
            'cg_mi_mean': float(all_cg_mi.mean()),
            'cg_mi_min': float(all_cg_mi.min()),
            'genie_mi_mean': float(all_genie_mi.mean()),
            'genie_mi_min': float(all_genie_mi.min()),
            'gap_mean': float(all_gaps.mean()),
            'n_bottleneck_01': int(n_bottleneck),
            'n_severe_02': int(n_severe),
        }
    }
    with open('results/cg_n256_bottleneck.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/cg_n256_bottleneck.json")


if __name__ == '__main__':
    main()
