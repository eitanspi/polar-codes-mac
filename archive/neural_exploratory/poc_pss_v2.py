#!/usr/bin/env python3
"""
poc_pss_v2.py — Two-phase PSS: fast_ce pretraining then PSS fine-tuning.

Key insight from v1: fast_ce achieves 98.6% per-leaf accuracy under teacher
forcing, but sequential decode BLER=1.0 due to error cascades. The model has
the RIGHT capacity — the problem is purely exposure bias at inference time.

Approach:
  Phase 1: fast_ce pretraining (15K iters) — learn the tree operations
  Phase 2: PSS fine-tuning (25K iters) — close the inference gap

Also tests a "self-decode" evaluation: run the model's own parallel predictions
through the tree to compute parallel BLER (without sequential error cascade).
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file

D = 16; HIDDEN = 64; N_LAYERS = 2
SNR_DB = 6.0; SIGMA2 = 10 ** (-SNR_DB / 10)
TRAIN_N = 32; BATCH = 128; EVAL_CW = 2000

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
SC_BLER = 0.046

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class PSSDecoder(nn.Module):
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def bitnode(self, e_odd, e_even, uv_left):
        u_left = uv_left // 2; v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([e_odd[:, :, :h] * u_sign, e_odd[:, :, h:] * v_sign], dim=-1)
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def tree_pass(self, emb, joint_cw, pred_left=None, p_sample=0.0):
        B, N, d = emb.shape; n = int(math.log2(N))
        all_losses = []; predictions = []

        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 4), joint_cw.reshape(-1)))
        predictions.append(logits.detach().argmax(dim=-1))

        E_chunks = [emb]; J_chunks = [joint_cw]
        for depth in range(n):
            E_odds, E_evens, J_odds, J_evens = [], [], [], []
            for e, j in zip(E_chunks, J_chunks):
                M = e.shape[1]
                E_odds.append(e.reshape(B, M//2, 2, d)[:, :, 0, :])
                E_evens.append(e.reshape(B, M//2, 2, d)[:, :, 1, :])
                J_odds.append(j.reshape(B, M//2, 2)[:, :, 0])
                J_evens.append(j.reshape(B, M//2, 2)[:, :, 1])

            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)
            J_odd = torch.cat(J_odds, 1); J_even = torch.cat(J_evens, 1)

            u_o = J_odd // 2; v_o = J_odd % 2
            u_e = J_even // 2; v_e = J_even % 2
            J_left_true = (u_o ^ u_e) * 2 + (v_o ^ v_e)

            if pred_left is not None and p_sample > 0:
                mask = (torch.rand(B, J_left_true.shape[1]) < p_sample).long()
                J_left = J_left_true * (1 - mask) + pred_left[depth] * mask
            else:
                J_left = J_left_true

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)

            nc = 2 ** depth; cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1); er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left_true, cs, 1); jr = torch.split(J_even, cs, 1)
            E_chunks = []; J_chunks = []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]; J_chunks += [c, dd]

            e_all = torch.cat(E_chunks, 1); j_all = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_all)
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4), j_all.reshape(-1)))
            predictions.append(logits.detach().argmax(dim=-1))

        return torch.stack(all_losses).mean(), predictions

    def get_pred_left(self, preds, B, N):
        """Convert per-depth predictions to pred_left format for PSS."""
        n = int(math.log2(N))
        pred_left = []
        pred_chunks = [preds[0]]
        for depth in range(n):
            J_odds, J_evens = [], []
            for j in pred_chunks:
                M = j.shape[1]
                J_odds.append(j.reshape(B, M//2, 2)[:, :, 0])
                J_evens.append(j.reshape(B, M//2, 2)[:, :, 1])
            J_odd = torch.cat(J_odds, 1); J_even = torch.cat(J_evens, 1)
            u_o = J_odd // 2; v_o = J_odd % 2
            u_e = J_even // 2; v_e = J_even % 2
            J_left_pred = (u_o ^ u_e) * 2 + (v_o ^ v_e)
            pred_left.append(J_left_pred)
            nc = 2 ** depth; cs = (N // 2) // nc
            jl_s = torch.split(J_left_pred, cs, 1)
            jr_s = torch.split(J_even, cs, 1)
            pred_chunks = []
            for l, r in zip(jl_s, jr_s): pred_chunks += [l, r]
        return pred_left

    def sc_decode(self, emb, frozen_u, frozen_v):
        B = emb.shape[0]; N = emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]
        def _decode(eb):
            bs = eb.shape[1]
            if bs == 1:
                logits = self.emb2logits(eb[:, 0, :])
                idx = leaf_idx[0]; leaf_idx[0] += 1
                u_frz = idx in frozen_u; v_frz = idx in frozen_v
                if u_frz and v_frz: dec = torch.zeros(B, dtype=torch.long)
                elif u_frz: dec = (logits[:, 1] > logits[:, 0]).long()
                elif v_frz: dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else: dec = logits.argmax(dim=-1)
                u_hat[:, idx] = dec // 2; v_hat[:, idx] = dec % 2
                return dec.unsqueeze(1)
            half = bs // 2
            e_odd = eb[:, 0::2, :]; e_even = eb[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            uv_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, uv_left)
            uv_right = _decode(e_right)
            return torch.cat([uv_left, uv_right], 1)
        with torch.no_grad(): _decode(emb)
        return u_hat, v_hat


def load_design():
    n = int(math.log2(TRAIN_N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, 15, 15)
    return Au, Av


def evaluate(model, channel, Au, Av, br, n_cw=EVAL_CW):
    model.eval()
    N = TRAIN_N
    fu_set = {p - 1 for p in range(1, N + 1) if p not in Au}
    fv_set = {p - 1 for p in range(1, N + 1) if p not in Av}
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(32, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p - 1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode(emb, fu_set, fv_set)
            for i in range(actual):
                ue = any(u_dec[i, p-1].item() != uf[i, p-1] for p in Au)
                ve = any(v_dec[i, p-1].item() != vf[i, p-1] for p in Av)
                if ue or ve: errs += 1
            total += actual
    model.train()
    return errs / total


def main():
    N = TRAIN_N; n = int(math.log2(N))
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av = load_design()
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    model = PSSDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'PSS v2 — N={N}, d={D}, params={params:,}, SC BLER={SC_BLER}', flush=True)

    rng = np.random.default_rng()

    def gen_batch():
        uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        joint_cw = torch.from_numpy(xf * 2 + yf).long()[:, br]
        return emb, joint_cw

    best_bler = 1.0

    # ─── Phase 1: fast_ce pretraining ─────────────────────────────────────
    P1_ITERS = 15000; P1_LR = 3e-4
    opt = torch.optim.Adam(model.parameters(), lr=P1_LR)
    print(f'\n--- Phase 1: fast_ce pretraining ({P1_ITERS} iters, lr={P1_LR}) ---', flush=True)
    t0 = time.time()
    losses = []
    model.train()
    for it in range(1, P1_ITERS + 1):
        emb, joint_cw = gen_batch()
        loss, _ = model.tree_pass(emb, joint_cw)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if it % 5000 == 0:
            bler = evaluate(model, channel, Au, Av, br, 500)
            if bler < best_bler: best_bler = bler
            print(f'  P1[{it}] loss={np.mean(losses[-500:]):.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}) {(time.time()-t0)/60:.1f}min', flush=True)

    # Save Phase 1 checkpoint
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'pss_v2_phase1.pt'))
    print(f'  Phase 1 done. Best BLER: {best_bler:.4f}', flush=True)

    # ─── Phase 2: PSS fine-tuning with multiple p values ──────────────────
    for p_sample in [0.3, 0.5, 0.7, 1.0]:
        P2_ITERS = 25000; P2_LR = 1e-4
        # Reload Phase 1 checkpoint for fair comparison
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'pss_v2_phase1.pt'), weights_only=True))
        opt = torch.optim.Adam(model.parameters(), lr=P2_LR)
        print(f'\n--- Phase 2: PSS p={p_sample} ({P2_ITERS} iters, lr={P2_LR}) ---', flush=True)

        t0 = time.time()
        losses = []
        p2_best = best_bler

        model.train()
        for it in range(1, P2_ITERS + 1):
            emb, joint_cw = gen_batch()

            # Pass 1: collect predictions (no grad)
            with torch.no_grad():
                _, preds = model.tree_pass(emb, joint_cw)
            pred_left = model.get_pred_left(preds, BATCH, N)

            # Pass 2: train with mixed bits
            loss, _ = model.tree_pass(emb, joint_cw, pred_left=pred_left, p_sample=p_sample)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

            if it % 5000 == 0:
                bler = evaluate(model, channel, Au, Av, br, 1000)
                if bler < p2_best: p2_best = bler
                print(f'  PSS[{it}] p={p_sample} loss={np.mean(losses[-500:]):.4f} '
                      f'BLER={bler:.4f} (best={p2_best:.4f}) {(time.time()-t0)/60:.1f}min', flush=True)

        # Also continue fast_ce (no PSS) as control
        if p_sample == 0.3:  # only do this once
            model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'pss_v2_phase1.pt'), weights_only=True))
            opt = torch.optim.Adam(model.parameters(), lr=P2_LR)
            print(f'\n--- Phase 2 control: continued fast_ce ({P2_ITERS} iters) ---', flush=True)
            t0 = time.time(); ctrl_best = best_bler; losses = []
            model.train()
            for it in range(1, P2_ITERS + 1):
                emb, joint_cw = gen_batch()
                loss, _ = model.tree_pass(emb, joint_cw)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); losses.append(loss.item())
                if it % 5000 == 0:
                    bler = evaluate(model, channel, Au, Av, br, 1000)
                    if bler < ctrl_best: ctrl_best = bler
                    print(f'  CTRL[{it}] loss={np.mean(losses[-500:]):.4f} '
                          f'BLER={bler:.4f} (best={ctrl_best:.4f}) '
                          f'{(time.time()-t0)/60:.1f}min', flush=True)

    print(f'\nDone!', flush=True)


if __name__ == '__main__':
    main()
