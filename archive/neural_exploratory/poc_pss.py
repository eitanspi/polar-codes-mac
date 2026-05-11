#!/usr/bin/env python3
"""
poc_pss.py — Parallel Scheduled Sampling (PSS) for MAC neural polar decoder.

The core problem: fast_ce uses teacher forcing (true bits at BitNodes), giving
O(log N) gradient depth but 7x worse BLER than SC due to exposure bias.

PSS idea:
  Pass 1 (no grad): Run fast_ce with teacher forcing, collect model predictions.
  Pass 2 (with grad): Run fast_ce again, but at each BitNode, randomly substitute
    true bits with model predictions from Pass 1 (with probability p).

This exposes the model to its own error patterns while keeping O(log N) depth.

Architecture: Sign-based BitNode (proven in poc_joint_fastce.py to get BLER=0.34).
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

# ─── Config ──────────────────────────────────────────────────────────────────

D = 16
HIDDEN = 64
N_LAYERS = 2

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

TRAIN_N = 32
BATCH = 128
LR = 3e-4
TOTAL_ITERS = 40000
EVAL_EVERY = 2000
EVAL_CW = 2000

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

SC_REF = {
    32:  {'ku': 15,  'kv': 15,  'sc_bler': 0.046},
    64:  {'ku': 31,  'kv': 31,  'sc_bler': 0.025},
    128: {'ku': 62,  'kv': 62,  'sc_bler': 0.016},
}

# ─── Model ───────────────────────────────────────────────────────────────────

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class PSS_MAC_Decoder(nn.Module):
    """
    Sign-based BitNode architecture (matches poc_joint_fastce.py).

    BitNode: split embedding into u-half and v-half, apply sign flips
    based on decoded (u,v). Residual = signed_embedding + e_even.
    """
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d),
        )
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        # Sign-based bitnode: input is (e_signed, e_even) = 2d
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def bitnode(self, e_odd, e_even, uv_left):
        """
        Sign-based BitNode: splits e_odd into u-half and v-half,
        applies sign flips based on joint (u,v) decision.
        """
        u_left = uv_left // 2
        v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)  # 0->+1, 1->-1
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([e_odd[:, :, :h] * u_sign,
                              e_odd[:, :, h:] * v_sign], dim=-1)
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def _tree_pass(self, emb, joint_cw, pred_left_per_depth=None, p_sample=0.0):
        """
        Single pass through the tree.

        Args:
            emb: (B, N, d) bit-reversed channel embeddings
            joint_cw: (B, N) bit-reversed joint codeword = u_cw*2 + v_cw
            pred_left_per_depth: if given, predicted uv_left at each depth
            p_sample: probability of using predictions instead of true bits

        Returns:
            loss, predictions_per_depth (list of predicted joint at each depth+1 level)
        """
        B, N, d = emb.shape
        n = int(math.log2(N))

        all_losses = []
        predictions = []

        # Depth 0: predict from raw embeddings
        logits = self.emb2logits(emb)
        loss = F.cross_entropy(logits.reshape(-1, 4), joint_cw.reshape(-1))
        all_losses.append(loss)
        predictions.append(logits.detach().argmax(dim=-1))

        E_chunks = [emb]
        J_chunks = [joint_cw]

        for depth in range(n):
            E_odds, E_evens, J_odds, J_evens = [], [], [], []
            for e, j in zip(E_chunks, J_chunks):
                M = e.shape[1]
                E_odds.append(e.reshape(B, M // 2, 2, d)[:, :, 0, :])
                E_evens.append(e.reshape(B, M // 2, 2, d)[:, :, 1, :])
                J_odds.append(j.reshape(B, M // 2, 2)[:, :, 0])
                J_evens.append(j.reshape(B, M // 2, 2)[:, :, 1])

            E_odd = torch.cat(E_odds, 1)
            E_even = torch.cat(E_evens, 1)
            J_odd = torch.cat(J_odds, 1)
            J_even = torch.cat(J_evens, 1)

            # True left-child bits via per-user XOR
            u_o = J_odd // 2; v_o = J_odd % 2
            u_e = J_even // 2; v_e = J_even % 2
            J_left_true = (u_o ^ u_e) * 2 + (v_o ^ v_e)
            J_right = J_even

            # Mix with predictions if PSS
            if pred_left_per_depth is not None and p_sample > 0:
                pred = pred_left_per_depth[depth]
                mask = (torch.rand(B, J_left_true.shape[1]) < p_sample).long()
                J_left = J_left_true * (1 - mask) + pred * mask
            else:
                J_left = J_left_true

            # CheckNode and BitNode
            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)

            # Interleave left/right chunks
            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left_true, cs, 1)  # targets always true
            jr = torch.split(J_right, cs, 1)

            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

            # Loss at this depth
            e_all = torch.cat(E_chunks, 1)
            j_all = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_all)
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4), j_all.reshape(-1)))
            predictions.append(logits.detach().argmax(dim=-1))

        return torch.stack(all_losses).mean(), predictions

    def fast_ce(self, emb, joint_cw):
        loss, _ = self._tree_pass(emb, joint_cw)
        return loss

    def fast_ce_pss(self, emb, joint_cw, p_sample=0.5):
        """Two-pass PSS: collect predictions, then train with mixed bits."""
        B, N = joint_cw.shape
        n = int(math.log2(N))

        # Pass 1: teacher-forced, collect predictions (no grad)
        with torch.no_grad():
            _, preds = self._tree_pass(emb, joint_cw)

        # Derive predicted uv_left at each depth from Pass 1 predictions
        pred_left_per_depth = []
        pred_chunks = [preds[0]]  # depth-0 predictions

        for depth in range(n):
            J_odds, J_evens = [], []
            for j in pred_chunks:
                M = j.shape[1]
                J_odds.append(j.reshape(B, M // 2, 2)[:, :, 0])
                J_evens.append(j.reshape(B, M // 2, 2)[:, :, 1])
            J_odd = torch.cat(J_odds, 1)
            J_even = torch.cat(J_evens, 1)
            u_o = J_odd // 2; v_o = J_odd % 2
            u_e = J_even // 2; v_e = J_even % 2
            J_left_pred = (u_o ^ u_e) * 2 + (v_o ^ v_e)
            pred_left_per_depth.append(J_left_pred)

            nc = 2 ** depth
            cs = (N // 2) // nc
            jl_splits = torch.split(J_left_pred, cs, 1)
            jr_splits = torch.split(J_even, cs, 1)
            pred_chunks = []
            for l, r in zip(jl_splits, jr_splits):
                pred_chunks += [l, r]

        # Pass 2: train with mixed bits
        loss, _ = self._tree_pass(emb, joint_cw,
                                  pred_left_per_depth=pred_left_per_depth,
                                  p_sample=p_sample)
        return loss

    def sc_decode(self, emb, frozen_u, frozen_v):
        """Sequential SC decode for evaluation."""
        B = emb.shape[0]
        N = emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(emb_block):
            bs = emb_block.shape[1]
            if bs == 1:
                logits = self.emb2logits(emb_block[:, 0, :])
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                u_frz = idx in frozen_u
                v_frz = idx in frozen_v
                if u_frz and v_frz:
                    dec = torch.zeros(B, dtype=torch.long)
                elif u_frz:
                    dec = (logits[:, 1] > logits[:, 0]).long()
                elif v_frz:
                    dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else:
                    dec = logits.argmax(dim=-1)
                u_hat[:, idx] = dec // 2
                v_hat[:, idx] = dec % 2
                return dec.unsqueeze(1)

            half = bs // 2
            e_odd = emb_block[:, 0::2, :]
            e_even = emb_block[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            uv_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, uv_left)
            uv_right = _decode(e_right)
            return torch.cat([uv_left, uv_right], 1)

        with torch.no_grad():
            _decode(emb)
        return u_hat, v_hat


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_design(N):
    n = int(math.log2(N))
    ref = SC_REF[N]
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ref['ku'], ref['kv'])
    return Au, Av, fu, fv


def evaluate(model, channel, N, Au, Av):
    model.eval()
    n = int(math.log2(N))
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    fu_set = {p - 1 for p in range(1, N + 1) if p not in Au}
    fv_set = {p - 1 for p in range(1, N + 1) if p not in Av}
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < EVAL_CW:
            actual = min(32, EVAL_CW - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p - 1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode(emb, fu_set, fv_set)
            for i in range(actual):
                ue = any(u_dec[i, p - 1].item() != uf[i, p - 1] for p in Au)
                ve = any(v_dec[i, p - 1].item() != vf[i, p - 1] for p in Av)
                if ue or ve: errs += 1
            total += actual
    model.train()
    return errs / total


def train_one(mode, p_sample=0.5):
    N = TRAIN_N
    n = int(math.log2(N))
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design(N)
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    model = PSS_MAC_Decoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng()

    tag = f'{mode}' + (f'_p{p_sample}' if mode == 'pss' else '')
    print(f'\n{"="*60}', flush=True)
    print(f'Training: {tag} | N={N} d={D} params={model.count_params():,}', flush=True)
    print(f'batch={BATCH} lr={LR} iters={TOTAL_ITERS}', flush=True)
    print(f'{"="*60}', flush=True)

    t0 = time.time()
    losses = []
    best_bler = 1.0

    # PSS: ramp p from 0 to target over first 5K iters
    def get_p(it):
        if mode != 'pss': return 0.0
        return p_sample * min(1.0, it / 5000)

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        uf = np.zeros((BATCH, N), dtype=int)
        vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        joint_cw = torch.from_numpy(xf * 2 + yf).long()[:, br]

        if mode == 'fast_ce':
            loss = model.fast_ce(emb, joint_cw)
        else:
            loss = model.fast_ce_pss(emb, joint_cw, p_sample=get_p(it))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-500:])
            bler = evaluate(model, channel, N, Au, Av)
            sc = SC_REF[N]['sc_bler']
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(),
                           os.path.join(SAVE_DIR, f'pss_{tag}_N{N}_best.pt'))
            ratio = bler / max(sc, 1e-8)
            print(f'[{it:>6}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}, {ratio:.1f}x SC) '
                  f'p={get_p(it):.2f} {elapsed/60:.1f}min', flush=True)

    return tag, best_bler


def main():
    print('PSS Proof of Concept — MAC Neural Polar Decoder', flush=True)
    print(f'N={TRAIN_N}, SNR={SNR_DB}dB, Class B', flush=True)
    print(f'SC reference BLER: {SC_REF[TRAIN_N]["sc_bler"]}', flush=True)
    print(f'fast_ce expected BLER: ~0.34 (7.4x SC)', flush=True)

    results = {}

    # 1. Baseline: plain fast_ce
    tag, best = train_one('fast_ce')
    results[tag] = best

    # 2. PSS with different p values
    for p in [0.3, 0.5, 0.7]:
        tag, best = train_one('pss', p_sample=p)
        results[tag] = best

    print(f'\n{"="*60}', flush=True)
    print(f'SUMMARY — N={TRAIN_N}, SC BLER={SC_REF[TRAIN_N]["sc_bler"]}', flush=True)
    print(f'{"="*60}', flush=True)
    for tag, bler in results.items():
        ratio = bler / SC_REF[TRAIN_N]['sc_bler']
        print(f'  {tag:20s}  BLER={bler:.4f}  ({ratio:.1f}x SC)', flush=True)


if __name__ == '__main__':
    main()
