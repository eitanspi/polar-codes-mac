#!/usr/bin/env python3
"""
train_freerun.py

Fine-tune the CG decoder at N=256 using FREE-RUNNING training.

Unlike teacher forcing (which feeds back the true bits), free-running
training feeds back the model's OWN predictions at every step. The loss
is still computed against the true (u,v) targets.

This directly exposes the model to its own error cascade during training.

Two modes:
  'full_freerun' — all steps use model predictions (aggressive)
  'mix'          — each step uses model prediction with prob p_free,
                   true bit with prob (1-p_free) (softer)
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from polar.encoder import polar_encode_batch, build_message_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 256; n_log = 8; ku = kv = 123
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)

MODE = 'mix'      # 'full_freerun' or 'mix'
P_FREE = 0.3      # for 'mix' mode: prob of using model's own prediction
LR = 5e-5
BATCH = 16
EVAL_EVERY = 500
EVAL_CW = 1000
TOTAL_ITERS = 15000
CKPT_IN = 'saved_models/ncg_gmac_mlp_N256.pt'
CKPT_OUT = 'saved_models/n256_freerun_best.pt'


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
            model_sd[nk] = v; loaded += 1
    model.load_state_dict(model_sd)
    return model, loaded


def forward_freerun(model, z, b, frozen_u, frozen_v, u_true, v_true,
                    mode='mix', p_free=0.3):
    """
    Forward pass with free-running committed bits.

    In 'full_freerun' mode: every non-frozen step uses the model's argmax.
    In 'mix' mode: each non-frozen step uses model prediction with prob p_free.

    Loss is always against true targets.
    """
    B_size, N_size = z.shape
    device = z.device
    d = model.d

    br = torch.from_numpy(bit_reversal_perm(
        int(math.log2(N_size)))).long().to(device)
    root = model.z_encoder(z.unsqueeze(-1))[:, br]

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

        if i_t in fdict:
            bit = torch.full((B_size,), fdict[i_t],
                             dtype=torch.float32, device=device)
        else:
            # Always compute loss against true target
            target = (u_true[:, i_t - 1] * 2 + v_true[:, i_t - 1]).long()
            all_logits.append(logits)
            all_targets.append(target)

            # Get model's own prediction (detached — no gradient through argmax)
            with torch.no_grad():
                if gamma == 0:
                    p0 = torch.logsumexp(logits[:, :2], dim=1)
                    p1 = torch.logsumexp(logits[:, 2:], dim=1)
                else:
                    p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                    p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                model_bit = (p1 > p0).float()

            true_bit = u_true[:, i_t - 1] if gamma == 0 else v_true[:, i_t - 1]

            if mode == 'full_freerun':
                bit = model_bit
            elif mode == 'mix':
                use_model = (torch.rand(B_size, device=device) < p_free).float()
                bit = true_bit * (1 - use_model) + model_bit * use_model
            else:
                bit = true_bit

        if gamma == 0:
            u_hat[i_t] = bit
        else:
            v_hat[i_t] = bit

        new_emb = model._make_leaf_emb(
            u_hat.get(i_t), v_hat.get(i_t), B_size, device)
        edge_data[leaf_edge] = new_emb.unsqueeze(1)

    return all_logits, all_targets


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


def main():
    os.chdir(ROOT)
    print("=" * 60)
    print(f"Free-Running Training — N=256 — mode={MODE}")
    print("=" * 60)
    print(f"p_free: {P_FREE}, LR: {LR}, batch: {BATCH}")

    Au, Av, frozen_u, frozen_v, _, _, path_i = design_from_file(
        f'designs/gmac_B_n{n_log}_snr6dB.npz', n_log, ku=ku, kv=kv)
    b = make_path(N, path_i)
    channel = GaussianMAC(sigma2=SIGMA2)

    model, loaded = load_nn(CKPT_IN)
    print(f"Loaded {loaded} params from {CKPT_IN}")

    # Fixed eval set
    rng_eval = np.random.default_rng(999)
    U_info_eval = rng_eval.integers(0, 2, size=(EVAL_CW, ku), dtype=np.int32)
    V_info_eval = rng_eval.integers(0, 2, size=(EVAL_CW, kv), dtype=np.int32)
    U_msg_eval = build_message_batch(N, U_info_eval, Au)
    V_msg_eval = build_message_batch(N, V_info_eval, Av)
    X_eval = polar_encode_batch(U_msg_eval)
    Y_eval = polar_encode_batch(V_msg_eval)
    np.random.seed(999)
    Z_eval = channel.sample_batch(X_eval, Y_eval).astype(np.float32)

    init_bler = evaluate(model, Z_eval, b, frozen_u, frozen_v,
                         Au, Av, U_msg_eval, V_msg_eval, EVAL_CW)
    print(f"Initial BLER: {init_bler:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_ITERS, eta_min=LR/10)

    model.train()
    best_bler = init_bler
    losses = []
    t0 = time.time()

    for it in range(1, TOTAL_ITERS + 1):
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

        all_logits, all_targets = forward_freerun(
            model, z_t, b, frozen_u, frozen_v, u_t, v_t,
            mode=MODE, p_free=P_FREE
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

        if it % EVAL_EVERY == 0 or it == 1:
            bler = evaluate(model, Z_eval, b, frozen_u, frozen_v,
                            Au, Av, U_msg_eval, V_msg_eval, EVAL_CW)
            avg_loss = np.mean(losses[-EVAL_EVERY:])
            elapsed = time.time() - t0
            improved = ""
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), CKPT_OUT)
                improved = " *BEST*"
            print(f"[{it:5d}/{TOTAL_ITERS}] loss={avg_loss:.4f} "
                  f"bler={bler:.4f} best={best_bler:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} "
                  f"{elapsed:.0f}s{improved}", flush=True)

    print(f"\nDone. Best BLER: {best_bler:.4f} (init: {init_bler:.4f})")
    print(f"Saved: {CKPT_OUT}")


if __name__ == '__main__':
    main()
