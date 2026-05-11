#!/usr/bin/env python3
"""
class_c_chained_npd.py — Class C MAC decoder via two chained single-user NPDs.

Approach:
  - Class C path (path_i=N in this codebase) decodes U first then V.
  - Stage 1: decode U on the mixture channel Z = (1-2X) + (1-2Y_random) + W
             where Y is treated as interference (single-user NPD over a
             non-Gaussian channel).
  - Stage 2: subtract (1-2Y_true) from Z to get Z' = (1-2X) + W, a clean
             BPSK+AWGN single-user channel, then decode V.
  Wait — convention check. This codebase uses path_i=N ("Class C") to mean
  decode U first then V. So Stage 1 = U on mixture (X→Z marginal), Stage 2 =
  V on clean AWGN conditional on known X. That's what we do.

Both stages are single-user polar decoding problems, so we can reuse the
proven NPDSingleUser module with its sign-flip bitnode + fast_ce training
(which works correctly for binary decisions — only the 4-class MAC version
had exposure bias problems).
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from neural.npd_pytorch import NPDSingleUser, npd_encode

# ─── Config ──────────────────────────────────────────────────────────────────
D = 16
HIDDEN = 64
N_LAYERS = 2
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
SIGMA = math.sqrt(SIGMA2)

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
os.makedirs(RESULTS_DIR, exist_ok=True)

# N=256 Class C design
N = 256
n = 8

# Rate choice: match Class B's total rate of 246 bits (ku=kv=123)
# for a fair comparison. Allocate asymmetrically favoring V (clean channel).
# Stage 1 U gets ku_C, Stage 2 V gets kv_C, sum = 246.
# Based on design file: U has 81 positions with Pe<0.01; V has 212 with Pe<0.01.
KU_C = 60   # stage 1: U on mixture, conservative choice (Pe < 0.005 for best 60)
KV_C = 186  # stage 2: V on clean AWGN, 60+186 = 246 = same as Class B

# Training config
BATCH_S1 = 32
BATCH_S2 = 32
LR = 3e-4
ITERS_S1 = 40000
ITERS_S2 = 30000
EVAL_EVERY = 2000
EVAL_CW = 500

FINAL_EVAL_CW = 5000

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_class_c_design():
    """Load Class C N=256 design."""
    path = os.path.join(DESIGNS_DIR, f'gmac_C_n{n}_snr{SNR_DB:.0f}dB.npz')
    Au, Av, fu, fv, pe_u, pe_v, path_i = design_from_file(path, n, KU_C, KV_C)
    print(f'Loaded Class C design: path_i={path_i}, N={N}')
    print(f'  |Au|={len(Au)}, |Av|={len(Av)}, ku={KU_C}, kv={KV_C}')
    print(f'  U Pe max (chosen): {max(pe_u[p-1] for p in Au):.4f}')
    print(f'  V Pe max (chosen): {max(pe_v[p-1] for p in Av):.4f}')
    print(f'  Expected union bound BLER: ~{sum(pe_u[p-1] for p in Au) + sum(pe_v[p-1] for p in Av):.4f}')
    return Au, Av, fu, fv, pe_u, pe_v


def generate_batch_stage1(Au, batch, rng):
    """
    Generate training data for Stage 1 (decode U on mixture channel).

    U info bits are random, V is completely random (independent uniform).
    Return:
      u_npd:   (B, N) NPD-order message bits (0 at frozen, random at info)
      x_cw:    (B, N) NPD-encoded codeword for U
      z:       (B, N) channel output Z = (1-2X) + (1-2Y) + W

    The single-user NPD uses its own tree order (no bit-reversal). Frozen
    positions are in the NPD tree order, which happens to be the same as
    the natural-order positions because the NPD encoder and the standard
    polar encoder differ only by a final bit-reversal.

    Actually: npd_encode(u)[br] == polar_encode(u). So if we feed u directly
    to npd_encode to get x_cw, then X in natural order is polar_encode(u) =
    npd_encode(u)[br], so we need to bit-reverse x_cw to get physical codeword.

    For the channel: we need the PHYSICAL codeword transmitted, which is in
    natural order. So:
      u_phys = u (natural-order message, with frozen=0)
      x_phys = polar_encode_batch(u) (natural-order codeword)
      x_npd = npd_encode(u) = x_phys[br] (NPD tree-order codeword)

    For training with fast_ce, the NPD sees channel samples z mapped via ey
    to embeddings. The fast_ce target is x_npd (NPD tree order).

    But z itself must match the tree order of the embeddings. The NPD's
    ey is applied position-wise, so if we want the tree leaf at position i
    to correspond to physical position i, we need z in NPD tree order too.
    In natural physical order, position i of z is z[i]. In NPD tree order,
    leaf i corresponds to physical position br[i] (because x_npd[i] = x_phys[br[i]]).

    So we should apply bit-reversal to z as well: z_npd = z[br].

    Wait — let me re-verify. From npd_pytorch.py self-test:
      x_std = polar_encode_batch(u)   # natural order
      x_npd = npd_encode(u)           # NPD tree order
      assert np.all(x_npd == x_std[:, br])  # i.e. x_npd[i] = x_std[br[i]]

    So x_npd[i] = x_std[br[i]]. Channel output z is generated from x_std (physical),
    so z[j] corresponds to x_std[j]. In NPD tree order, leaf i needs to see
    the channel output for x_std[br[i]] = x_npd[i]. So z_npd[i] = z[br[i]].
    """
    # Au is 1-indexed. Build u in natural order.
    u_nat = np.zeros((batch, N), dtype=int)
    for p in Au:
        u_nat[:, p - 1] = rng.integers(0, 2, batch)

    # Physical codeword for X
    x_phys = polar_encode_batch(u_nat)

    # Y is random uniform (not polar-encoded — just uniform bits to simulate interference)
    y_phys = rng.integers(0, 2, (batch, N))

    # Channel output
    noise = rng.normal(0, SIGMA, (batch, N))
    z_nat = (1.0 - 2.0 * x_phys.astype(float)) + (1.0 - 2.0 * y_phys.astype(float)) + noise

    # Convert to NPD tree order via bit-reversal
    br = bit_reversal_perm(n)
    z_npd = z_nat[:, br]
    x_npd = x_phys[:, br]  # = npd_encode(u_nat)

    return u_nat, x_phys, y_phys, z_nat, z_npd, x_npd


def generate_batch_stage2(Av, batch, rng):
    """
    Generate training data for Stage 2 (decode V on clean AWGN given U).

    Channel (given U is known): Z' = Z - (1-2X) = (1-2Y) + W
    """
    v_nat = np.zeros((batch, N), dtype=int)
    for p in Av:
        v_nat[:, p - 1] = rng.integers(0, 2, batch)

    y_phys = polar_encode_batch(v_nat)
    noise = rng.normal(0, SIGMA, (batch, N))
    zp_nat = (1.0 - 2.0 * y_phys.astype(float)) + noise  # clean BPSK+AWGN

    br = bit_reversal_perm(n)
    zp_npd = zp_nat[:, br]
    y_npd = y_phys[:, br]  # = npd_encode(v_nat)

    return v_nat, y_phys, zp_nat, zp_npd, y_npd


def frozen_set_npd(Au_1idx):
    """
    Build frozen set for NPD decoder in NPD tree order (0-indexed).

    Au is 1-indexed in natural message order. Frozen positions are
    natural positions not in Au, converted to 0-indexed.

    For NPD: leaf i in the tree corresponds to natural message position br[i]
    (because the NPD encoder + bit-reversal = standard encoder).

    Wait — message positions, not codeword positions. Let me re-check.

    Standard polar encoder: x = G_N u = B_N F^⊗n u
    So x[br[i]] = (F^⊗n u)[i]  — the "tree position" i in the F^⊗n
    decomposition corresponds to physical position br[i].

    For the MESSAGE side: u in natural order = u in tree order, because
    we apply B_N to the codeword, not the message. So u[i] is the message
    bit at position i regardless of whether we think in natural or tree order.

    But NPD's decode uses leaf_idx in its own traversal order. Looking at
    the self-test:
      frozen_set = set(range(k))  # positions 0..k-1 are frozen
      u_dec = model.decode(emb, frozen_set)
      if any(u_d[0, p].item() != u1[0, p] for p in range(k, N))  # check info

    So u_dec[0, p] corresponds to natural message position p. The NPD stores
    decoded bits at natural positions. So frozen set in NPD is in natural
    0-indexed message positions.

    That means: frozen_set = {p-1 for p in 1..N if p not in Au_1idx}
    """
    return {p - 1 for p in range(1, N + 1) if p not in Au_1idx}


# ─── Training ────────────────────────────────────────────────────────────────

def train_stage(stage_name, model, gen_fn, Au_or_Av, iters, batch_size, lr, ckpt_path, eval_every=EVAL_EVERY):
    """Train a single-user NPD stage with fast_ce."""
    br = bit_reversal_perm(n)
    frozen = frozen_set_npd(Au_or_Av)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng()
    t0 = time.time()
    losses = []
    best_bler = 1.0

    print(f'\n=== Stage {stage_name} training ===')
    print(f'  N={N}, params={model.count_parameters():,}, batch={batch_size}, lr={lr}, iters={iters}')

    model.train()
    for it in range(1, iters + 1):
        data = gen_fn(Au_or_Av, batch_size, rng)
        # Extract z_npd and x_npd (or y_npd) — the last two returns
        z_npd = data[-2]
        cw_npd = data[-1]

        z_t = torch.from_numpy(z_npd).float().unsqueeze(-1)
        cw_t = torch.from_numpy(cw_npd).long()

        emb = model.ey(z_t)
        loss = model.fast_ce(emb, cw_t)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % eval_every == 0:
            # Quick eval on the SAME channel used for training
            bler = evaluate_stage(model, gen_fn, Au_or_Av, frozen, n_cw=200, seed=999)
            avg_loss = np.mean(losses[-500:])
            elapsed = (time.time() - t0) / 60
            improved = ''
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), ckpt_path)
                improved = ' *BEST*'
            print(f'  [{stage_name}][{it:>6}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}) {elapsed:.1f}min{improved}', flush=True)

    # Final full eval
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    bler_final = evaluate_stage(model, gen_fn, Au_or_Av, frozen, n_cw=EVAL_CW, seed=999)
    print(f'  [{stage_name}] FINAL BLER={bler_final:.4f} (best during training={best_bler:.4f})')
    return best_bler, bler_final


def evaluate_stage(model, gen_fn, Au_or_Av, frozen, n_cw, seed):
    """Evaluate a single stage on its own channel (not chained)."""
    model.eval()
    errs = 0
    rng = np.random.default_rng(seed)
    bs = 32
    with torch.no_grad():
        total = 0
        while total < n_cw:
            actual = min(bs, n_cw - total)
            data = gen_fn(Au_or_Av, actual, rng)
            u_or_v_nat = data[0]
            z_npd = data[-2]
            z_t = torch.from_numpy(z_npd).float().unsqueeze(-1)
            emb = model.ey(z_t)
            u_dec = model.decode(emb, frozen)  # (B, N) in natural message order
            for i in range(actual):
                if any(u_dec[i, p - 1].item() != u_or_v_nat[i, p - 1] for p in Au_or_Av):
                    errs += 1
            total += actual
    model.train()
    return errs / n_cw


# ─── End-to-end chained evaluation ───────────────────────────────────────────

def evaluate_e2e(model_s1, model_s2, Au, Av, n_cw, seed):
    """
    Chained end-to-end decoding:
      1. Receive Z from GMAC
      2. Stage 1 decodes U from Z (on mixture channel)
      3. Reconstruct X̂ from Û
      4. Compute Z' = Z - (1-2X̂)
      5. Stage 2 decodes V from Z' (on clean AWGN)
      6. Block error = (Û != U) OR (V̂ != V) at info positions
    """
    model_s1.eval()
    model_s2.eval()
    br = bit_reversal_perm(n)
    fu = frozen_set_npd(Au)
    fv = frozen_set_npd(Av)

    channel = GaussianMAC(sigma2=SIGMA2)
    rng_bits = np.random.default_rng(seed)
    np.random.seed(seed)  # channel uses global rng

    errs = 0
    u_errs = 0
    v_errs = 0
    bs = 16
    total = 0

    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            # Generate true messages
            u_nat = np.zeros((actual, N), dtype=int)
            v_nat = np.zeros((actual, N), dtype=int)
            for p in Au: u_nat[:, p - 1] = rng_bits.integers(0, 2, actual)
            for p in Av: v_nat[:, p - 1] = rng_bits.integers(0, 2, actual)

            # Encode
            x_phys = polar_encode_batch(u_nat)
            y_phys = polar_encode_batch(v_nat)

            # Transmit through GMAC (single channel realization)
            z_nat = channel.sample_batch(x_phys, y_phys)

            # Stage 1: decode U from z_nat (on mixture channel)
            z_npd = z_nat[:, br]
            z_t = torch.from_numpy(z_npd).float().unsqueeze(-1)
            emb_s1 = model_s1.ey(z_t)
            u_dec = model_s1.decode(emb_s1, fu)  # (B, N) natural

            # Reconstruct X̂ from decoded U
            u_dec_np = u_dec.numpy().astype(int)
            x_hat = polar_encode_batch(u_dec_np)

            # Residual: Z' = Z - (1-2X̂)
            zp_nat = z_nat - (1.0 - 2.0 * x_hat.astype(float))

            # Stage 2: decode V from zp_nat (on clean AWGN given U)
            zp_npd = zp_nat[:, br]
            zp_t = torch.from_numpy(zp_npd).float().unsqueeze(-1)
            emb_s2 = model_s2.ey(zp_t)
            v_dec = model_s2.decode(emb_s2, fv)

            # Block error analysis
            for i in range(actual):
                ue = any(u_dec[i, p - 1].item() != u_nat[i, p - 1] for p in Au)
                ve = any(v_dec[i, p - 1].item() != v_nat[i, p - 1] for p in Av)
                if ue: u_errs += 1
                if ve: v_errs += 1
                if ue or ve: errs += 1
            total += actual

    return errs / n_cw, u_errs / n_cw, v_errs / n_cw


def wilson_ci(errs, n, z=1.96):
    if n == 0: return (0, 1)
    p = errs / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - margin, centre + margin)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f'Class C Chained NPD Decoder — N={N}, SNR={SNR_DB}dB, SIGMA2={SIGMA2:.4f}')
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # Load design
    Au, Av, fu, fv, pe_u, pe_v = load_class_c_design()

    # Stage 1: U decoder (mixture channel)
    print(f'\n{"="*60}')
    model_s1 = NPDSingleUser(d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_dim=1)
    ckpt_s1 = os.path.join(SAVE_DIR, 'class_c_stage1.pt')
    best_s1, final_s1 = train_stage(
        'S1-U-mixture', model_s1, generate_batch_stage1, Au,
        iters=ITERS_S1, batch_size=BATCH_S1, lr=LR, ckpt_path=ckpt_s1,
    )
    model_s1.load_state_dict(torch.load(ckpt_s1, weights_only=True))

    # Stage 2: V decoder (clean AWGN)
    print(f'\n{"="*60}')
    model_s2 = NPDSingleUser(d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_dim=1)
    ckpt_s2 = os.path.join(SAVE_DIR, 'class_c_stage2.pt')
    best_s2, final_s2 = train_stage(
        'S2-V-clean', model_s2, generate_batch_stage2, Av,
        iters=ITERS_S2, batch_size=BATCH_S2, lr=LR, ckpt_path=ckpt_s2,
    )
    model_s2.load_state_dict(torch.load(ckpt_s2, weights_only=True))

    # End-to-end evaluation
    print(f'\n{"="*60}')
    print(f'End-to-end chained evaluation ({FINAL_EVAL_CW} codewords)')
    t0 = time.time()
    bler_e2e, u_err_rate, v_err_rate = evaluate_e2e(
        model_s1, model_s2, Au, Av, n_cw=FINAL_EVAL_CW, seed=999,
    )
    elapsed = (time.time() - t0) / 60
    errs_total = int(round(bler_e2e * FINAL_EVAL_CW))
    ci_low, ci_high = wilson_ci(errs_total, FINAL_EVAL_CW)

    print(f'\n=== FINAL RESULTS (Class C, N={N}) ===')
    print(f'  Stage 1 (U on mixture) final BLER:  {final_s1:.4f}')
    print(f'  Stage 2 (V on clean AWGN) BLER:     {final_s2:.4f}')
    print(f'  Chained BLER (5000 cw):             {bler_e2e:.4f}')
    print(f'    95% Wilson CI:                    [{ci_low:.4f}, {ci_high:.4f}]')
    print(f'    U error rate:                      {u_err_rate:.4f}')
    print(f'    V error rate:                      {v_err_rate:.4f}')
    print(f'  SC reference (Class B 4-class MAC): 0.0050')
    print(f'  CG decoder (Class B 4-class MAC):   0.0170')
    print(f'  Class C NPD (this work):            {bler_e2e:.4f}')
    print(f'  Eval wall time:                     {elapsed:.1f} min')

    results = {
        'config': {
            'N': N, 'n': n, 'ku_C': KU_C, 'kv_C': KV_C,
            'snr_db': SNR_DB, 'sigma2': SIGMA2,
            'd': D, 'hidden': HIDDEN, 'n_layers': N_LAYERS,
            'batch_s1': BATCH_S1, 'batch_s2': BATCH_S2,
            'lr': LR, 'iters_s1': ITERS_S1, 'iters_s2': ITERS_S2,
        },
        'stage1': {
            'best_bler': float(best_s1),
            'final_bler': float(final_s1),
            'params': model_s1.count_parameters(),
        },
        'stage2': {
            'best_bler': float(best_s2),
            'final_bler': float(final_s2),
            'params': model_s2.count_parameters(),
        },
        'e2e': {
            'n_cw': FINAL_EVAL_CW,
            'errs': errs_total,
            'bler': float(bler_e2e),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'u_error_rate': float(u_err_rate),
            'v_error_rate': float(v_err_rate),
        },
        'reference': {
            'sc_class_b': 0.0050,
            'cg_class_b': 0.0170,
        },
    }
    with open(os.path.join(RESULTS_DIR, 'class_c_npd_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to results/class_c_npd_results.json')
    print(f'Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
