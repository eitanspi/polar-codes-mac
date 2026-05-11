#!/usr/bin/env python3
"""
Compare two N=32 NCG GMAC models trained differently:
  - Model A: rate-1 trained (class_c_npd/results/ncg_r1_32/iter300000.pt)
  - Model B: SC-design rate 0.594 trained (saved_models/ncg_gmac_mlp_N32.pt)

E1: per-info-position MI at SC design rate (teacher-forced)
E4: per-codeword error correlation at SC design rate (free-running)
"""

import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder
from neural.neural_scl import SimpleMLP_Gmac

N = 32
SIGMA2 = 10 ** (-0.6)  # 6 dB
DEVICE = torch.device('cpu')

MODEL_A_PATH = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd',
                            'results', 'ncg_r1_32', 'iter300000.pt')
MODEL_B_PATH = os.path.join(os.path.dirname(__file__), '..',
                            'saved_models', 'ncg_gmac_mlp_N32.pt')
DESIGN_PATH = os.path.join(os.path.dirname(__file__), '..', 'designs',
                           'gmac_B_n5_snr6dB.npz')


def load_model_A():
    m = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2).to(DEVICE)
    sd = torch.load(MODEL_A_PATH, map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


def load_model_B():
    m = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32).to(DEVICE)
    sd = torch.load(MODEL_B_PATH, map_location=DEVICE, weights_only=True)
    fixed = {}
    for k, v in sd.items():
        nk = k.replace('z_enc.', 'z_encoder.') if k.startswith('z_enc.') else k
        fixed[nk] = v
    missing, unexpected = m.load_state_dict(fixed, strict=False)
    if missing:
        print(f"[warn] Model B missing keys: {missing}")
    if unexpected:
        print(f"[warn] Model B unexpected keys: {unexpected}")
    m.eval()
    return m


def binary_entropy(p):
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def sample_sc_codewords(n_cw, Au, Av, rng):
    """Generate codewords with u=0 at frozen positions, random at info positions."""
    u = np.zeros((n_cw, N), dtype=int)
    v = np.zeros((n_cw, N), dtype=int)
    for p in Au:
        u[:, p - 1] = rng.integers(0, 2, n_cw)
    for p in Av:
        v[:, p - 1] = rng.integers(0, 2, n_cw)
    return u, v


def e1_measure_mi(model, ch, b, fu, fv, Au, Av, n_cw, batch, seed):
    """
    Teacher-forced per-position MI. Model is told about frozen positions
    (fu, fv); logits are collected at info positions. Returns per-info-position
    MI as dicts keyed by 1-indexed position.
    """
    rng = np.random.default_rng(seed)
    # Build per-info-position accumulators
    h_u = {p: 0.0 for p in Au}
    h_v = {p: 0.0 for p in Av}
    cnt = 0

    done = 0
    while done < n_cw:
        B = min(batch, n_cw - done)
        u, v = sample_sc_codewords(B, Au, Av, rng)
        x = polar_encode_batch(u)
        y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float().to(DEVICE)
        ut = torch.from_numpy(u).float().to(DEVICE)
        vt = torch.from_numpy(v).float().to(DEVICE)

        with torch.no_grad():
            L, _, _, _, _ = model(z, b, fu, fv, u_true=ut, v_true=vt)

        # L contains logits only for non-frozen positions, in the order they
        # are visited. We must track which position each logit corresponds to.
        idx_u = 0
        idx_v = 0
        logit_idx = 0
        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                idx_u += 1
                i_t = idx_u
                if i_t in fu:  # frozen -> no logit appended
                    continue
                lg = L[logit_idx]
                logit_idx += 1
                lp0 = torch.logsumexp(lg[:, :2], 1)
                lp1 = torch.logsumexp(lg[:, 2:], 1)
                p0 = torch.sigmoid(lp0 - lp1).cpu().numpy()
                h = binary_entropy(p0).sum()
                h_u[i_t] += h
            else:
                idx_v += 1
                i_t = idx_v
                if i_t in fv:
                    continue
                lg = L[logit_idx]
                logit_idx += 1
                lp0 = torch.logsumexp(lg[:, [0, 2]], 1)
                lp1 = torch.logsumexp(lg[:, [1, 3]], 1)
                p0 = torch.sigmoid(lp0 - lp1).cpu().numpy()
                h = binary_entropy(p0).sum()
                h_v[i_t] += h
        cnt += B
        done += B

    mi_u = {p: 1.0 - h_u[p] / cnt for p in Au}
    mi_v = {p: 1.0 - h_v[p] / cnt for p in Av}
    return mi_u, mi_v


def e4_decode(model, ch, b, fu, fv, Au, Av, n_cw, batch, seed):
    """
    Free-running (inference) decoding. Returns per-codeword fail array (bool).
    Uses fixed seed so the SAME codewords are generated.
    """
    rng = np.random.default_rng(seed)
    Au_sorted = sorted(Au)
    Av_sorted = sorted(Av)
    fail = np.zeros(n_cw, dtype=bool)
    done = 0
    while done < n_cw:
        B = min(batch, n_cw - done)
        u, v = sample_sc_codewords(B, Au, Av, rng)
        x = polar_encode_batch(u)
        y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float().to(DEVICE)

        with torch.no_grad():
            _, _, u_hat, v_hat, _ = model(z, b, fu, fv)

        u_pred = torch.stack([u_hat[p] for p in Au_sorted], dim=1).cpu().numpy().astype(int)
        v_pred = torch.stack([v_hat[p] for p in Av_sorted], dim=1).cpu().numpy().astype(int)
        u_true = u[:, [p - 1 for p in Au_sorted]]
        v_true = v[:, [p - 1 for p in Av_sorted]]
        ue = (u_pred != u_true).any(1)
        ve = (v_pred != v_true).any(1)
        fail[done:done + B] = ue | ve
        done += B
    return fail


def main():
    print("Loading models...")
    mA = load_model_A()
    mB = load_model_B()
    print(f"  Model A params: {sum(p.numel() for p in mA.parameters()):,}")
    print(f"  Model B params: {sum(p.numel() for p in mB.parameters()):,}")

    ch = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)

    # SC design (ku=13, kv=25)
    ku, kv = 13, 25
    Au, Av, fu, fv, _, _, _ = design_from_file(DESIGN_PATH, 5, ku, kv)
    Au = sorted(Au)
    Av = sorted(Av)
    print(f"  ku={ku}, kv={kv}  rate={(ku+kv)/(2*N):.3f}")
    print(f"  Au: {Au}")
    print(f"  Av: {Av}")

    # ─────────── E1: Per-info-position MI ───────────
    print("\n" + "=" * 72)
    print("E1: Per-info-position MI at SC design rate (5000 cw, teacher-forced)")
    print("=" * 72)
    N_CW_E1 = 5000
    BATCH = 256
    SEED_E1 = 42

    t0 = time.time()
    miA_u, miA_v = e1_measure_mi(mA, ch, b, fu, fv, Au, Av, N_CW_E1, BATCH, SEED_E1)
    tA = time.time() - t0
    print(f"  Model A done in {tA:.1f}s")

    t0 = time.time()
    miB_u, miB_v = e1_measure_mi(mB, ch, b, fu, fv, Au, Av, N_CW_E1, BATCH, SEED_E1)
    tB = time.time() - t0
    print(f"  Model B done in {tB:.1f}s")

    sumA_u = sum(miA_u.values())
    sumA_v = sum(miA_v.values())
    sumB_u = sum(miB_u.values())
    sumB_v = sum(miB_v.values())
    print(f"\n  sum MI_U: Model A = {sumA_u:.4f}, Model B = {sumB_u:.4f}")
    print(f"  sum MI_V: Model A = {sumA_v:.4f}, Model B = {sumB_v:.4f}")

    # Per-position table + differences
    print("\n  Per-position MI (info positions only):")
    print(f"  {'pos':>4s}  {'kind':>4s}  {'MI_A':>8s}  {'MI_B':>8s}  {'A-B':>9s}")
    diffs = []  # (abs_diff, pos, kind, miA, miB)
    for p in Au:
        a = miA_u[p]; b_ = miB_u[p]
        diffs.append((abs(a - b_), p, 'U', a, b_))
    for p in Av:
        a = miA_v[p]; b_ = miB_v[p]
        diffs.append((abs(a - b_), p, 'V', a, b_))

    # Print all (for reference)
    for _, p, k, a, b_ in sorted(diffs, key=lambda x: (x[2], x[1])):
        print(f"  {p:>4d}  {k:>4s}  {a:>8.4f}  {b_:>8.4f}  {a - b_:>+9.4f}")

    diffs.sort(reverse=True)
    print("\n  Top-5 biggest-differing info positions (by |A-B|):")
    print(f"  {'pos':>4s}  {'kind':>4s}  {'MI_A':>8s}  {'MI_B':>8s}  {'A-B':>9s}")
    for _, p, k, a, b_ in diffs[:5]:
        print(f"  {p:>4d}  {k:>4s}  {a:>8.4f}  {b_:>8.4f}  {a - b_:>+9.4f}")

    # MI < 0.9 counts
    weakA = [(p, 'U', miA_u[p]) for p in Au if miA_u[p] < 0.9] + \
            [(p, 'V', miA_v[p]) for p in Av if miA_v[p] < 0.9]
    weakB = [(p, 'U', miB_u[p]) for p in Au if miB_u[p] < 0.9] + \
            [(p, 'V', miB_v[p]) for p in Av if miB_v[p] < 0.9]
    print(f"\n  Model A: {len(weakA)} info positions with MI<0.9: {weakA}")
    print(f"  Model B: {len(weakB)} info positions with MI<0.9: {weakB}")

    # ─────────── E4: Per-codeword error correlation ───────────
    print("\n" + "=" * 72)
    print("E4: Per-codeword error correlation (2000 cw, free-running)")
    print("=" * 72)
    N_CW_E4 = 2000
    SEED_E4 = 42

    t0 = time.time()
    failA = e4_decode(mA, ch, b, fu, fv, Au, Av, N_CW_E4, BATCH, SEED_E4)
    tA = time.time() - t0
    print(f"  Model A done in {tA:.1f}s")

    t0 = time.time()
    failB = e4_decode(mB, ch, b, fu, fv, Au, Av, N_CW_E4, BATCH, SEED_E4)
    tB = time.time() - t0
    print(f"  Model B done in {tB:.1f}s")

    blerA = failA.mean()
    blerB = failB.mean()
    both_ok = int(((~failA) & (~failB)).sum())
    only_A = int((failB & ~failA).sum())  # A correct, B wrong
    only_B = int((failA & ~failB).sum())  # B correct, A wrong
    both_bad = int((failA & failB).sum())

    print(f"\n  BLER Model A (rate-1 trained):     {blerA:.4f}  ({int(failA.sum())}/{N_CW_E4})")
    print(f"  BLER Model B (SC-design trained):  {blerB:.4f}  ({int(failB.sum())}/{N_CW_E4})")
    print(f"\n  Contingency table:")
    print(f"    both correct:  {both_ok:>5d} / {N_CW_E4}")
    print(f"    only A correct:{only_A:>5d} / {N_CW_E4}")
    print(f"    only B correct:{only_B:>5d} / {N_CW_E4}")
    print(f"    both wrong:    {both_bad:>5d} / {N_CW_E4}")

    # Independence check: expected under independence
    exp_both_bad = blerA * blerB * N_CW_E4
    print(f"\n  Expected 'both wrong' under independence: {exp_both_bad:.1f}")
    print(f"  Observed 'both wrong': {both_bad}")
    oracle_fail = both_bad / N_CW_E4
    print(f"  Oracle BLER (min of either): {oracle_fail:.4f}")

    # Chi-squared / phi correlation (simple proxy)
    # phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
    a11 = both_bad; a10 = only_B  # A fails
    a01 = only_A; a00 = both_ok
    num = a11 * a00 - a10 * a01
    row1 = (a11 + a10); row0 = (a01 + a00)
    col1 = (a11 + a01); col0 = (a10 + a00)
    den = (row1 * row0 * col1 * col0) ** 0.5 if row1 * row0 * col1 * col0 > 0 else 1.0
    phi = num / den if den > 0 else 0.0
    print(f"  Phi correlation of failures: {phi:+.4f}  "
          f"(positive = errors happen on same codewords)")


if __name__ == '__main__':
    main()
