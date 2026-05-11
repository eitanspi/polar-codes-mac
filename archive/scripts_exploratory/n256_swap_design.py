#!/usr/bin/env python3
"""
n256_swap_design.py

Targeted code redesign for N=256 Class B:

1. From inference error analysis, identify the K worst U-info
   positions for the NN (highest per-step error rate).
2. From genie Pe, identify K currently-frozen U-positions with
   the lowest Pe (best candidates to become info).
3. Swap them: freeze the K worst info positions, unfreeze the
   K best frozen positions.
4. Do the same for V.
5. Retrain the CG decoder with the swapped frozen set.
6. Evaluate both NN and SC with the new frozen set.

This directly tests the code-design-mismatch hypothesis:
if the NN gap is due to trying to decode positions it can't
handle, swapping them for better ones should help.
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder_interleaved import decode_single
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 256; n_log = 8
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
K_SWAP = 5  # how many positions to swap per user (conservative)

# Training config
LR = 5e-4
BATCH = 16
TOTAL_ITERS = 15000
EVAL_EVERY = 1000
EVAL_CW = 1000
CKPT_OUT = 'saved_models/n256_swap5_best.pt'


def load_nn(ckpt_path, d=16, hidden=64):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model = GmacNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=2, z_hidden=32)
    model_sd = model.state_dict()
    for k, v in sd.items():
        nk = k
        if nk.startswith('tree.'): nk = nk[5:]
        elif nk.startswith('z_enc.'): nk = 'z_encoder.' + nk[6:]
        if 'embedding_z' in nk: continue
        if nk in model_sd and model_sd[nk].shape == v.shape:
            model_sd[nk] = v
    model.load_state_dict(model_sd)
    return model


def evaluate_nn(model, Z, b, frozen_u, frozen_v, Au, Av,
                U_msg, V_msg, n_cw, batch=64):
    model.eval()
    errors = 0
    for start in range(0, n_cw, batch):
        end = min(start + batch, n_cw)
        z_t = torch.from_numpy(Z[start:end]).float()
        with torch.no_grad():
            _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u, frozen_v)
        bs = end - start
        u_dec = np.zeros((bs, N), dtype=np.int32)
        v_dec = np.zeros((bs, N), dtype=np.int32)
        for pos, val in u_hat.items():
            u_dec[:, pos-1] = val.round().int().cpu().numpy()
        for pos, val in v_hat.items():
            v_dec[:, pos-1] = val.round().int().cpu().numpy()
        for i in range(bs):
            u_bad = any(u_dec[i, p-1] != U_msg[start+i, p-1] for p in Au)
            v_bad = any(v_dec[i, p-1] != V_msg[start+i, p-1] for p in Av)
            if u_bad or v_bad: errors += 1
    model.train()
    return errors / n_cw


def evaluate_sc(Z, b, frozen_u, frozen_v, channel, Au, Av,
                U_msg, V_msg, n_cw):
    errors = 0
    for i in range(n_cw):
        z_list = Z[i].tolist()
        u_dec, v_dec = decode_single(
            N, z_list, b, frozen_u, frozen_v, channel, log_domain=False)
        u_dec = np.asarray(u_dec, dtype=np.int32)
        v_dec = np.asarray(v_dec, dtype=np.int32)
        u_bad = any(u_dec[p-1] != U_msg[i, p-1] for p in Au)
        v_bad = any(v_dec[p-1] != V_msg[i, p-1] for p in Av)
        if u_bad or v_bad: errors += 1
    return errors / n_cw


def main():
    os.chdir(ROOT)

    # Load genie design
    Au_genie, Av_genie, frozen_u_genie, frozen_v_genie, pe_u, pe_v, path_i = \
        design_from_file('designs/gmac_B_n8_snr6dB.npz', n_log, ku=123, kv=123)
    b = make_path(N, path_i)
    channel = GaussianMAC(sigma2=SIGMA2)

    Au_genie_set = set(Au_genie)
    Av_genie_set = set(Av_genie)

    # Load inference error data
    with open('results/n256_inference_errors.json') as f:
        inf_data = json.load(f)

    # Per-step NN error rates, mapped to (pos, user)
    u_err_by_pos = {}
    v_err_by_pos = {}
    for s in inf_data['per_step']:
        pos = s['pos']
        if s['user'] == 0:
            u_err_by_pos[pos] = s['nn_err_rate']
        else:
            v_err_by_pos[pos] = s['nn_err_rate']

    # ─── Identify swap candidates ─────────────────────────────────────

    # Worst U-info positions by inference error rate
    u_info_by_err = sorted(
        [(pos, u_err_by_pos.get(pos, 0)) for pos in Au_genie],
        key=lambda x: -x[1]
    )
    u_to_freeze = [pos for pos, _ in u_info_by_err[:K_SWAP]]

    # Best currently-frozen U positions by genie Pe
    u_frozen_by_pe = sorted(
        [(pos, pe_u[pos-1]) for pos in range(1, N+1) if pos not in Au_genie_set],
        key=lambda x: x[1]
    )
    u_to_unfreeze = [pos for pos, _ in u_frozen_by_pe[:K_SWAP]]

    # Same for V
    v_info_by_err = sorted(
        [(pos, v_err_by_pos.get(pos, 0)) for pos in Av_genie],
        key=lambda x: -x[1]
    )
    v_to_freeze = [pos for pos, _ in v_info_by_err[:K_SWAP]]

    v_frozen_by_pe = sorted(
        [(pos, pe_v[pos-1]) for pos in range(1, N+1) if pos not in Av_genie_set],
        key=lambda x: x[1]
    )
    v_to_unfreeze = [pos for pos, _ in v_frozen_by_pe[:K_SWAP]]

    print("=" * 60)
    print(f"Code Redesign: swap {K_SWAP} positions per user")
    print("=" * 60)
    print(f"\nU positions to FREEZE (worst NN inference err):")
    for pos, err in u_info_by_err[:K_SWAP]:
        print(f"  pos {pos:3d}: NN err={err:.4f}, genie Pe={pe_u[pos-1]:.4f}")
    print(f"\nU positions to UNFREEZE (best genie Pe among frozen):")
    for pos, pe in u_frozen_by_pe[:K_SWAP]:
        print(f"  pos {pos:3d}: genie Pe={pe:.4f}")
    print(f"\nV positions to FREEZE (worst NN inference err):")
    for pos, err in v_info_by_err[:K_SWAP]:
        print(f"  pos {pos:3d}: NN err={err:.4f}, genie Pe={pe_v[pos-1]:.4f}")
    print(f"\nV positions to UNFREEZE (best genie Pe among frozen):")
    for pos, pe in v_frozen_by_pe[:K_SWAP]:
        print(f"  pos {pos:3d}: genie Pe={pe:.4f}")

    # ─── Build swapped frozen set ─────────────────────────────────────

    Au_new = set(Au_genie) - set(u_to_freeze) | set(u_to_unfreeze)
    Av_new = set(Av_genie) - set(v_to_freeze) | set(v_to_unfreeze)

    assert len(Au_new) == 123, f"Expected 123 U-info, got {len(Au_new)}"
    assert len(Av_new) == 123, f"Expected 123 V-info, got {len(Av_new)}"

    frozen_u_new = {p: 0 for p in range(1, N+1) if p not in Au_new}
    frozen_v_new = {p: 0 for p in range(1, N+1) if p not in Av_new}

    overlap_u = len(Au_new & Au_genie_set)
    overlap_v = len(Av_new & Av_genie_set)
    print(f"\nOverlap with genie: U={overlap_u}/123, V={overlap_v}/123")

    Au_new_list = sorted(Au_new)
    Av_new_list = sorted(Av_new)

    # ─── Evaluate SC with BOTH frozen sets ────────────────────────────

    rng_eval = np.random.default_rng(999)
    n_eval = 2000
    U_info_eval = rng_eval.integers(0, 2, size=(n_eval, 123), dtype=np.int32)
    V_info_eval = rng_eval.integers(0, 2, size=(n_eval, 123), dtype=np.int32)

    # Genie design
    U_msg_g = build_message_batch(N, U_info_eval, Au_genie)
    V_msg_g = build_message_batch(N, V_info_eval, Av_genie)
    X_g = polar_encode_batch(U_msg_g)
    Y_g = polar_encode_batch(V_msg_g)
    np.random.seed(999)
    Z_g = channel.sample_batch(X_g, Y_g).astype(np.float32)

    print(f"\nEvaluating SC with GENIE design ({n_eval} cw)...", flush=True)
    sc_bler_genie = evaluate_sc(Z_g, b, frozen_u_genie, frozen_v_genie,
                                channel, Au_genie, Av_genie,
                                U_msg_g, V_msg_g, n_eval)
    print(f"  SC genie BLER: {sc_bler_genie:.4f}")

    # Swapped design
    U_msg_s = build_message_batch(N, U_info_eval, Au_new_list)
    V_msg_s = build_message_batch(N, V_info_eval, Av_new_list)
    X_s = polar_encode_batch(U_msg_s)
    Y_s = polar_encode_batch(V_msg_s)
    np.random.seed(999)
    Z_s = channel.sample_batch(X_s, Y_s).astype(np.float32)

    print(f"Evaluating SC with SWAPPED design ({n_eval} cw)...", flush=True)
    sc_bler_swapped = evaluate_sc(Z_s, b, frozen_u_new, frozen_v_new,
                                  channel, Au_new_list, Av_new_list,
                                  U_msg_s, V_msg_s, n_eval)
    print(f"  SC swapped BLER: {sc_bler_swapped:.4f}")

    # ─── Train NN with swapped design ─────────────────────────────────

    print(f"\nTraining NN with swapped design ({TOTAL_ITERS} iters)...",
          flush=True)
    print(f"  Warm-starting from N=256 genie checkpoint", flush=True)

    model = load_nn('saved_models/ncg_gmac_mlp_N256.pt')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_ITERS, eta_min=LR/10)
    model.train()

    best_bler = 1.0
    t0 = time.time()
    losses = []

    for it in range(1, TOTAL_ITERS + 1):
        rng = np.random.default_rng(it * 1000 + 42)
        U_info = rng.integers(0, 2, size=(BATCH, 123), dtype=np.int32)
        V_info = rng.integers(0, 2, size=(BATCH, 123), dtype=np.int32)
        U_msg = build_message_batch(N, U_info, Au_new_list)
        V_msg = build_message_batch(N, V_info, Av_new_list)
        X = polar_encode_batch(U_msg)
        Y = polar_encode_batch(V_msg)
        np.random.seed(it * 1000 + 43)
        Z = channel.sample_batch(X, Y).astype(np.float32)

        z_t = torch.from_numpy(Z).float()
        u_t = torch.from_numpy(U_msg.astype(np.float32))
        v_t = torch.from_numpy(V_msg.astype(np.float32))

        all_logits, all_targets, _, _, _ = model(
            z_t, b, frozen_u_new, frozen_v_new, u_true=u_t, v_true=v_t)

        logits_cat = torch.cat([l.unsqueeze(0) for l in all_logits], dim=0)
        targets_cat = torch.cat([t.unsqueeze(0) for t in all_targets], dim=0)
        loss = F.cross_entropy(logits_cat.view(-1, 4), targets_cat.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if it % EVAL_EVERY == 0 or it == 1:
            # Eval NN with swapped design
            rng_e = np.random.default_rng(888)
            U_ie = rng_e.integers(0, 2, size=(EVAL_CW, 123), dtype=np.int32)
            V_ie = rng_e.integers(0, 2, size=(EVAL_CW, 123), dtype=np.int32)
            U_me = build_message_batch(N, U_ie, Au_new_list)
            V_me = build_message_batch(N, V_ie, Av_new_list)
            Xe = polar_encode_batch(U_me)
            Ye = polar_encode_batch(V_me)
            np.random.seed(888)
            Ze = channel.sample_batch(Xe, Ye).astype(np.float32)
            bler = evaluate_nn(model, Ze, b, frozen_u_new, frozen_v_new,
                               Au_new_list, Av_new_list, U_me, V_me, EVAL_CW)
            avg_loss = np.mean(losses[-EVAL_EVERY:]) if len(losses) >= EVAL_EVERY else np.mean(losses)
            improved = ""
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), CKPT_OUT)
                improved = " *BEST*"
            elapsed = time.time() - t0
            print(f"[{it:5d}/{TOTAL_ITERS}] loss={avg_loss:.4f} "
                  f"bler={bler:.4f} best={best_bler:.4f} "
                  f"{elapsed:.0f}s{improved}", flush=True)

    # ─── Final summary ────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"| Decoder | Frozen set  | BLER    |")
    print(f"|---------|-------------|---------|")
    print(f"| SC      | Genie       | {sc_bler_genie:.4f}  |")
    print(f"| SC      | Swapped     | {sc_bler_swapped:.4f}  |")
    print(f"| NN      | Genie       | 0.0208  |  (from prior validation)")
    print(f"| NN      | Swapped     | {best_bler:.4f}  |")
    print(f"\nSwapped {K_SWAP} positions per user. "
          f"Overlap: U={overlap_u}/123, V={overlap_v}/123")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/n256_swap_design.json', 'w') as f:
        json.dump({
            'K_SWAP': K_SWAP,
            'sc_bler_genie': sc_bler_genie,
            'sc_bler_swapped': sc_bler_swapped,
            'nn_bler_genie': 0.0208,
            'nn_bler_swapped': best_bler,
            'overlap_u': overlap_u,
            'overlap_v': overlap_v,
            'u_to_freeze': u_to_freeze,
            'u_to_unfreeze': u_to_unfreeze,
            'v_to_freeze': v_to_freeze,
            'v_to_unfreeze': v_to_unfreeze,
        }, f, indent=2)
    print("Saved results/n256_swap_design.json")


if __name__ == '__main__':
    main()
