#!/usr/bin/env python3
"""
train_error_inject.py

Fine-tune the CG decoder at N=256 with targeted error injection.

During teacher-forced training, at the ~20 highest-error positions
(identified by n256_inference_errors.py), the committed bit is randomly
flipped with probability p_flip before being embedded. The loss target
stays correct. This teaches the downstream tree ops to handle wrong
inputs.

Usage:
    python neural/train_error_inject.py
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
from polar.design_mc import design_from_file
from polar.design import make_path
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

# ─── Config ──────────────────────────────────────────────────────────────

N = 256; n_log = 8; ku = kv = 123
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)

# Top-20 highest NN inference error positions (from n256_inference_errors.py)
# These are 1-indexed code positions where the U-step has the most errors.
INJECT_POSITIONS = {
    177, 209, 225, 201, 179, 241, 185, 211, 227, 210,
    170, 231, 235, 229, 169, 195, 202, 203, 204, 178,
}

P_FLIP = 0.05        # probability of flipping the committed bit at inject positions
P_SELF = 0.15        # probability of using model's own prediction (scheduled sampling)
EMB_NOISE = 0.1      # std of Gaussian noise added to committed embedding
MODE = 'self_play'   # 'flip', 'self_play', or 'emb_noise'
LR = 1e-4
BATCH = 16
EVAL_EVERY = 500
EVAL_CW = 1000
TOTAL_ITERS = 15000
CKPT_IN = 'saved_models/ncg_gmac_mlp_N256.pt'
CKPT_OUT = 'saved_models/n256_error_inject_best.pt'


# ─── Model loading ───────────────────────────────────────────────────────

def load_nn(ckpt_path, d=16, hidden=64):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model = GmacNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=2)
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        nk = k
        if nk.startswith('tree.'): nk = nk[5:]
        elif nk.startswith('z_enc.'): nk = 'z_encoder.' + nk[6:]
        if 'embedding_z' in nk: continue
        if nk in model_sd and model_sd[nk].shape == v.shape:
            model_sd[nk] = v
            loaded += 1
    model.load_state_dict(model_sd)
    return model, loaded


# ─── Modified forward with error injection ────────────────────────────────

def forward_with_injection(model, z, b, frozen_u, frozen_v,
                           u_true, v_true,
                           inject_positions, mode='self_play',
                           p_flip=0.05, p_self=0.15, emb_noise_std=0.1,
                           training=True):
    """
    Same as model.forward() with teacher forcing, but with perturbation
    at inject_positions to build robustness to inference errors.

    Modes:
      'flip'      — flip the committed bit with probability p_flip
      'self_play' — use model's own argmax prediction with probability p_self
      'emb_noise' — add Gaussian noise to committed embedding
    """
    B_size, N_size = z.shape
    device = z.device
    d = model.d

    br = torch.from_numpy(bit_reversal_perm(
        int(math.log2(N_size)))).long().to(device)
    root = model.z_encoder(z.unsqueeze(-1))[:, br]

    n = int(math.log2(N_size))
    edge_data = [None] * (2 * N_size)
    edge_data[1] = root

    no_info = model.no_info_emb.unsqueeze(0).unsqueeze(0)
    for beta in range(2, 2 * N_size):
        level = beta.bit_length() - 1
        size = N_size >> level
        edge_data[beta] = no_info.expand(B_size, size, d).clone()

    dec_head = 1
    u_hat, v_hat = {}, {}
    all_logits, all_targets = [], []
    i_u, i_v = 0, 0

    for step in range(2 * N_size):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u; fdict = frozen_u
        else:
            i_v += 1; i_t = i_v; fdict = frozen_v

        leaf_edge = i_t + N_size - 1
        target_vtx = leaf_edge >> 1
        dec_head = model._step_to(dec_head, target_vtx, edge_data)

        temp = edge_data[leaf_edge][:, 0].clone()
        if leaf_edge & 1 == 0:
            model._neural_calc_left(target_vtx, edge_data)
        else:
            model._neural_calc_right(target_vtx, edge_data)
        top_down = edge_data[leaf_edge][:, 0]

        combined = top_down + temp
        logits = model.emb2logits(combined)

        inject_emb = False  # flag for embedding-level noise

        if i_t in fdict:
            bit = torch.full((B_size,), fdict[i_t],
                             dtype=torch.float32, device=device)
        else:
            # Loss target is always the true joint label
            target = (u_true[:, i_t - 1] * 2 + v_true[:, i_t - 1]).long()
            all_logits.append(logits)
            all_targets.append(target)

            # Committed bit: default = true value
            true_bit = u_true[:, i_t - 1] if gamma == 0 else v_true[:, i_t - 1]
            bit = true_bit

            # ─── PERTURBATION AT INJECT POSITIONS ────────────────
            if training and i_t in inject_positions:
                if mode == 'flip':
                    flip_mask = (torch.rand(B_size, device=device) < p_flip).float()
                    bit = bit * (1 - flip_mask) + (1 - bit) * flip_mask

                elif mode == 'self_play':
                    # Use model's own prediction with probability p_self
                    with torch.no_grad():
                        if gamma == 0:
                            p0 = torch.logsumexp(logits[:, :2], dim=1)
                            p1 = torch.logsumexp(logits[:, 2:], dim=1)
                        else:
                            p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                            p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                        model_bit = (p1 > p0).float()
                    use_model = (torch.rand(B_size, device=device) < p_self).float()
                    bit = true_bit * (1 - use_model) + model_bit * use_model

                elif mode == 'emb_noise':
                    inject_emb = True  # handle after embedding
            # ─────────────────────────────────────────────────────

        if gamma == 0:
            u_hat[i_t] = bit
        else:
            v_hat[i_t] = bit

        new_emb = model._make_leaf_emb(
            u_hat.get(i_t), v_hat.get(i_t), B_size, device)

        # Embedding-level noise injection
        if inject_emb:
            noise = torch.randn_like(new_emb) * emb_noise_std
            new_emb = new_emb + noise

        edge_data[leaf_edge] = new_emb.unsqueeze(1)

    return all_logits, all_targets, u_hat, v_hat


# ─── Evaluation ───────────────────────────────────────────────────────────

def evaluate(model, Z_eval, b, frozen_u, frozen_v, Au, Av,
             U_msg_eval, V_msg_eval, n_cw, batch=64):
    model.eval()
    errors = 0
    for start in range(0, n_cw, batch):
        end = min(start + batch, n_cw)
        z_t = torch.from_numpy(Z_eval[start:end]).float()
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
            u_bad = any(u_dec[i, p-1] != U_msg_eval[start+i, p-1] for p in Au)
            v_bad = any(v_dec[i, p-1] != V_msg_eval[start+i, p-1] for p in Av)
            if u_bad or v_bad:
                errors += 1
    model.train()
    return errors / n_cw


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    os.chdir(ROOT)
    print("=" * 60)
    print("Targeted Error Injection Training — N=256")
    print("=" * 60)
    print(f"Mode: {MODE}")
    print(f"Inject positions: {sorted(INJECT_POSITIONS)}")
    print(f"p_flip: {P_FLIP}, p_self: {P_SELF}, emb_noise: {EMB_NOISE}")
    print(f"LR: {LR}, batch: {BATCH}")
    print(f"Checkpoint: {CKPT_IN}")

    Au, Av, frozen_u, frozen_v, _, _, path_i = design_from_file(
        f'designs/gmac_B_n{n_log}_snr6dB.npz', n_log, ku=ku, kv=kv)
    b = make_path(N, path_i)
    channel = GaussianMAC(sigma2=SIGMA2)

    model, loaded = load_nn(CKPT_IN)
    print(f"Loaded {loaded} params from {CKPT_IN}")

    # Generate fixed eval set
    rng_eval = np.random.default_rng(999)
    U_info_eval = rng_eval.integers(0, 2, size=(EVAL_CW, ku), dtype=np.int32)
    V_info_eval = rng_eval.integers(0, 2, size=(EVAL_CW, kv), dtype=np.int32)
    U_msg_eval = build_message_batch(N, U_info_eval, Au)
    V_msg_eval = build_message_batch(N, V_info_eval, Av)
    X_eval = polar_encode_batch(U_msg_eval)
    Y_eval = polar_encode_batch(V_msg_eval)
    np.random.seed(999)
    Z_eval = channel.sample_batch(X_eval, Y_eval).astype(np.float32)

    # Initial eval
    init_bler = evaluate(model, Z_eval, b, frozen_u, frozen_v,
                         Au, Av, U_msg_eval, V_msg_eval, EVAL_CW)
    print(f"Initial BLER: {init_bler:.4f}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_ITERS, eta_min=LR/10)

    model.train()
    best_bler = init_bler
    losses = []
    t0 = time.time()

    for it in range(1, TOTAL_ITERS + 1):
        # Generate batch
        rng_train = np.random.default_rng(it * 1000 + 42)
        U_info = rng_train.integers(0, 2, size=(BATCH, ku), dtype=np.int32)
        V_info = rng_train.integers(0, 2, size=(BATCH, kv), dtype=np.int32)
        U_msg = build_message_batch(N, U_info, Au)
        V_msg = build_message_batch(N, V_info, Av)
        X = polar_encode_batch(U_msg)
        Y = polar_encode_batch(V_msg)
        np.random.seed(it * 1000 + 43)
        Z = channel.sample_batch(X, Y).astype(np.float32)

        z_t = torch.from_numpy(Z).float()
        u_t = torch.from_numpy(U_msg.astype(np.float32))
        v_t = torch.from_numpy(V_msg.astype(np.float32))

        all_logits, all_targets, _, _ = forward_with_injection(
            model, z_t, b, frozen_u, frozen_v, u_t, v_t,
            INJECT_POSITIONS, mode=MODE,
            p_flip=P_FLIP, p_self=P_SELF, emb_noise_std=EMB_NOISE,
            training=True
        )

        logits_cat = torch.cat([l.unsqueeze(0) for l in all_logits], dim=0)
        targets_cat = torch.cat([t.unsqueeze(0) for t in all_targets], dim=0)
        loss = F.cross_entropy(
            logits_cat.view(-1, 4), targets_cat.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if it % EVAL_EVERY == 0 or it == 1:
            bler = evaluate(model, Z_eval, b, frozen_u, frozen_v,
                            Au, Av, U_msg_eval, V_msg_eval, EVAL_CW)
            avg_loss = np.mean(losses[-EVAL_EVERY:])
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]

            improved = ""
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), CKPT_OUT)
                improved = " *BEST*"

            print(f"[{it:5d}/{TOTAL_ITERS}] loss={avg_loss:.4f} "
                  f"bler={bler:.4f} best={best_bler:.4f} "
                  f"lr={lr_now:.2e} {elapsed:.0f}s{improved}", flush=True)

    print(f"\nDone. Best BLER: {best_bler:.4f}")
    print(f"Saved to: {CKPT_OUT}")

    # Final eval with more codewords
    if best_bler < init_bler:
        model.load_state_dict(torch.load(CKPT_OUT, map_location='cpu',
                                         weights_only=True))
        final_bler = evaluate(model, Z_eval, b, frozen_u, frozen_v,
                              Au, Av, U_msg_eval, V_msg_eval, EVAL_CW)
        print(f"Final best checkpoint BLER: {final_bler:.4f} "
              f"(init was {init_bler:.4f})")


if __name__ == '__main__':
    main()
