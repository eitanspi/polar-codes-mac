#!/usr/bin/env python3
"""
poc_two_phase_iterative.py — Two-Phase Iterative Refinement for MAC Polar Codes

Key idea: decompose the joint MAC decode into single-user phases, each using
fast_ce (O(log N) gradient depth). Iterate to converge.

Architecture:
  Phase 1: decoder_u_marginal(z) → u_hat      (U from marginal channel)
  Phase 2: decoder_v_cond(z, x_hat) → v_hat   (V given decoded U)
  Phase 3: decoder_u_refine(z, y_hat) → u_hat  (U given decoded V - refinement)
  Iterate Phases 2-3 for convergence.

Training: all phases use teacher forcing (true bits) → all fast_ce parallelizable.
Inference: iterate with estimated bits.

This is the FIRST attempt to get O(log N) training for MAC Class B paths.
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

# ─── Config ──────────────────────────────────────────────────────────────────

D = 16
HIDDEN = 64
N_LAYERS = 2
N_REFINE_ITERS = 2  # number of refinement iterations at inference

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')

SC_REF = {
    32:  {'ku': 15,  'kv': 15,  'sc_bler': 0.046},
    64:  {'ku': 31,  'kv': 31,  'sc_bler': 0.025},
    128: {'ku': 62,  'kv': 62,  'sc_bler': 0.016},
    256: {'ku': 123, 'kv': 123, 'sc_bler': 0.005},
}

# ─── Single-user NPD decoder ────────────────────────────────────────────────

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class NPDDecoder(nn.Module):
    """Single-user NPD decoder with fast_ce training (Aharoni et al.)."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_dim=1):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ELU(), nn.Linear(hidden, d),
        )
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.emb2llr = _make_mlp(d, hidden, 1, n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def bitnode(self, e_odd, e_even, u_left):
        u_sign = 2.0 * u_left.float().unsqueeze(-1) - 1.0
        u_sign = u_sign.expand_as(e_odd)
        e_signed = e_odd * u_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce(self, emb, x):
        """Parallel teacher-forced binary CE. O(log N) sequential depth."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        pred = self.emb2llr(emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred, x.float(), reduction='mean')
        all_losses.append(loss)

        E_chunks = [emb]
        X_chunks = [x]

        for depth in range(n):
            E_odds, E_evens = [], []
            X_odds, X_evens = [], []
            for e_chunk, x_chunk in zip(E_chunks, X_chunks):
                M = e_chunk.shape[1]
                e_r = e_chunk.reshape(B, M // 2, 2, d)
                E_odds.append(e_r[:, :, 0, :])
                E_evens.append(e_r[:, :, 1, :])
                x_r = x_chunk.reshape(B, M // 2, 2)
                X_odds.append(x_r[:, :, 0])
                X_evens.append(x_r[:, :, 1])

            E_odd = torch.cat(E_odds, dim=1)
            E_even = torch.cat(E_evens, dim=1)
            X_odd = torch.cat(X_odds, dim=1)
            X_even = torch.cat(X_evens, dim=1)

            X_left = X_odd ^ X_even
            X_right = X_even

            inp = torch.cat([E_odd, E_even], dim=-1)
            e_left = self.checknode(inp)
            e_right = self.bitnode(E_odd, E_even, X_left)

            n_chunks = 2 ** depth
            chunk_size = (N // 2) // n_chunks
            e_lefts = torch.split(e_left, chunk_size, dim=1)
            e_rights = torch.split(e_right, chunk_size, dim=1)
            x_lefts = torch.split(X_left, chunk_size, dim=1)
            x_rights = torch.split(X_right, chunk_size, dim=1)

            E_chunks = []
            X_chunks = []
            for el, er, xl, xr in zip(e_lefts, e_rights, x_lefts, x_rights):
                E_chunks.append(el)
                E_chunks.append(er)
                X_chunks.append(xl)
                X_chunks.append(xr)

            e_all = torch.cat(E_chunks, dim=1)
            x_all = torch.cat(X_chunks, dim=1)
            pred = self.emb2llr(e_all).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, x_all.float(), reduction='mean')
            all_losses.append(loss)

        return torch.stack(all_losses).mean()

    def decode_sequential(self, emb, frozen_set):
        """Sequential SC decode for inference. Returns message bits in u_hat."""
        B = emb.shape[0]
        N = emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(emb_block):
            """Returns CODEWORD of this subtree (needed by parent's bitnode)."""
            block_size = emb_block.shape[1]
            if block_size == 1:
                llr = self.emb2llr(emb_block[:, 0, :]).squeeze(-1)
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                if idx in frozen_set:
                    dec = torch.zeros(B, dtype=torch.long)
                else:
                    dec = (llr > 0).long()  # BCE convention: positive → bit 1
                u_hat[:, idx] = dec
                return dec.unsqueeze(1)  # at leaf, codeword = message bit

            e_odd = emb_block[:, 0::2, :]
            e_even = emb_block[:, 1::2, :]
            inp = torch.cat([e_odd, e_even], dim=-1)
            e_left = self.checknode(inp)
            x_left = _decode(e_left)  # left child codeword
            e_right = self.bitnode(e_odd, e_even, x_left)
            x_right = _decode(e_right)  # right child codeword
            # Reconstruct parent codeword via butterfly:
            # x[0::2] = x_left ^ x_right, x[1::2] = x_right
            x_parent = torch.zeros(B, block_size, dtype=torch.long)
            x_parent[:, 0::2] = x_left ^ x_right
            x_parent[:, 1::2] = x_right
            return x_parent

        with torch.no_grad():
            _decode(emb)
        return u_hat


# ─── Iterative Two-Phase MAC Decoder ─────────────────────────────────────────

class IterativeNPD_MAC(nn.Module):
    """
    Two-phase iterative refinement decoder for MAC polar codes.

    Phase 1: U marginal decoder (z only) — O(log N) via fast_ce
    Phase 2: V conditional decoder (z, x_bpsk) — O(log N) via fast_ce
    Phase 3: U refinement decoder (z, y_bpsk) — O(log N) via fast_ce

    Inference: iterate Phase 2 → Phase 3 for convergence.
    Total training depth: O(log N) per phase × 3 phases = O(log N).
    """

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.decoder_u_marginal = NPDDecoder(d=d, hidden=hidden, n_layers=n_layers, z_dim=1)
        self.decoder_v_cond = NPDDecoder(d=d, hidden=hidden, n_layers=n_layers, z_dim=2)
        self.decoder_u_refine = NPDDecoder(d=d, hidden=hidden, n_layers=n_layers, z_dim=2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, z, x_cw, y_cw, br):
        """
        Training: all phases teacher-forced, all parallelized via fast_ce.

        z: (B, N) channel output
        x_cw: (B, N) User U codeword bits
        y_cw: (B, N) User V codeword bits
        br: bit-reversal permutation
        """
        B, N = z.shape

        # Phase 1: U from marginal channel (z only)
        emb_u = self.decoder_u_marginal.z_encoder(z.unsqueeze(-1))[:, br]
        loss_u1 = self.decoder_u_marginal.fast_ce(emb_u, x_cw[:, br])

        # Phase 2: V from conditional channel (z, x_true)
        x_bpsk = (1.0 - 2.0 * x_cw.float())
        zv_input = torch.stack([z, x_bpsk], dim=-1)
        emb_v = self.decoder_v_cond.z_encoder(zv_input)[:, br]
        loss_v = self.decoder_v_cond.fast_ce(emb_v, y_cw[:, br])

        # Phase 3: U refinement from (z, y_true) — the key new component
        y_bpsk = (1.0 - 2.0 * y_cw.float())
        zu_input = torch.stack([z, y_bpsk], dim=-1)
        emb_u2 = self.decoder_u_refine.z_encoder(zu_input)[:, br]
        loss_u2 = self.decoder_u_refine.fast_ce(emb_u2, x_cw[:, br])

        return loss_u1 + loss_v + loss_u2

    @torch.no_grad()
    def decode(self, z, frozen_u_set, frozen_v_set, n_iters=N_REFINE_ITERS):
        """
        Iterative inference:
        1. U marginal → x_hat
        2. V conditional on x_hat → y_hat
        3. U refinement on y_hat → x_hat_new
        4. Repeat 2-3 for n_iters
        """
        B, N = z.shape
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long()

        self.eval()

        # Phase 1: U marginal decode
        emb_u = self.decoder_u_marginal.z_encoder(z.unsqueeze(-1))[:, br]
        u_msg = self.decoder_u_marginal.decode_sequential(emb_u, frozen_u_set)
        x_hat_cw = torch.from_numpy(polar_encode_batch(u_msg.numpy())).float()

        for it in range(n_iters):
            # Phase 2: V conditional on current x_hat
            x_bpsk = 1.0 - 2.0 * x_hat_cw
            zv_input = torch.stack([z, x_bpsk], dim=-1)
            emb_v = self.decoder_v_cond.z_encoder(zv_input)[:, br]
            v_msg = self.decoder_v_cond.decode_sequential(emb_v, frozen_v_set)
            y_hat_cw = torch.from_numpy(polar_encode_batch(v_msg.numpy())).float()

            # Phase 3: U refinement on current y_hat
            y_bpsk = 1.0 - 2.0 * y_hat_cw
            zu_input = torch.stack([z, y_bpsk], dim=-1)
            emb_u2 = self.decoder_u_refine.z_encoder(zu_input)[:, br]
            u_msg = self.decoder_u_refine.decode_sequential(emb_u2, frozen_u_set)
            x_hat_cw = torch.from_numpy(polar_encode_batch(u_msg.numpy())).float()

        # Final V decode with best x_hat
        x_bpsk = 1.0 - 2.0 * x_hat_cw
        zv_input = torch.stack([z, x_bpsk], dim=-1)
        emb_v = self.decoder_v_cond.z_encoder(zv_input)[:, br]
        v_msg = self.decoder_v_cond.decode_sequential(emb_v, frozen_v_set)

        return u_msg, v_msg


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, channel, N, Au, Av, fu, fv, n_cw, n_iters=N_REFINE_ITERS):
    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}
    errs_u = 0; errs_v = 0; errs_block = 0; total = 0
    rng = np.random.default_rng(999)
    bs = max(1, min(32, 256 // max(1, N // 16)))

    model.eval()
    while total < n_cw:
        actual = min(bs, n_cw - total)
        uf = np.zeros((actual, N), dtype=int)
        vf = np.zeros((actual, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        u_dec, v_dec = model.decode(zf, fu_set, fv_set, n_iters=n_iters)

        for i in range(actual):
            ue = any(u_dec[i, p-1].item() != uf[i, p-1] for p in Au)
            ve = any(v_dec[i, p-1].item() != vf[i, p-1] for p in Av)
            if ue: errs_u += 1
            if ve: errs_v += 1
            if ue or ve: errs_block += 1
        total += actual

    model.train()
    return errs_block / total, errs_u / total, errs_v / total


# ─── Main Training Loop ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=64)
    parser.add_argument('--iters', type=int, default=30000)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--eval_cw', type=int, default=1000)
    parser.add_argument('--refine_iters', type=int, default=2)
    args = parser.parse_args()

    N = args.N
    ref = SC_REF[N]
    ku, kv = ref['ku'], ref['kv']
    sc_bler = ref['sc_bler']
    n = int(math.log2(N))

    channel = GaussianMAC(sigma2=SIGMA2)

    # Load MC design
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{int(SNR_DB)}dB.npz')
    if os.path.exists(mc_path):
        Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
    else:
        from polar.design import design_gmac
        Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)

    br = torch.from_numpy(bit_reversal_perm(n)).long()

    model = IterativeNPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS)

    print(f'{"="*70}', flush=True)
    print(f'  Two-Phase Iterative Refinement — MAC Polar Decoder POC', flush=True)
    print(f'  N={N}, ku={ku}, kv={kv}, SNR={SNR_DB}dB', flush=True)
    print(f'  d={D}, hidden={HIDDEN}, refine_iters={args.refine_iters}', flush=True)
    print(f'  U_marginal params: {model.decoder_u_marginal.count_parameters():,}', flush=True)
    print(f'  V_conditional params: {model.decoder_v_cond.count_parameters():,}', flush=True)
    print(f'  U_refinement params: {model.decoder_u_refine.count_parameters():,}', flush=True)
    print(f'  Total: {model.count_parameters():,} params', flush=True)
    print(f'  Batch={args.batch}, LR={args.lr}, Iters={args.iters}', flush=True)
    print(f'  Training depth: O(log N) = {n} per phase (vs O(N log N) = {N*n} sequential)', flush=True)
    print(f'{"="*70}', flush=True)

    # Analytical SC baseline
    print(f'  SC BLER baseline: {sc_bler}', flush=True)

    # Marginal channel capacity check
    cap = channel.capacity()
    print(f'  I(Z;X) = {cap[0]:.4f}, I(Z;Y|X) = {cap[1]:.4f}, R_u = {ku/N:.4f}', flush=True)
    if ku/N > cap[0]:
        print(f'  WARNING: R_u={ku/N:.3f} > I(Z;X)={cap[0]:.3f} — Phase 1 alone cannot succeed!', flush=True)
        print(f'  This is expected for Class B. Iterative refinement should help.', flush=True)
    print(flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    t0 = time.time()
    best_bler = 1.0

    model.train()
    for it in range(1, args.iters + 1):
        # Cosine LR
        progress = it / args.iters
        lr_now = args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        # Generate batch
        uf = np.zeros((args.batch, N), dtype=int)
        vf = np.zeros((args.batch, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, args.batch)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, args.batch)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        loss = model(zf, torch.from_numpy(xf).long(), torch.from_numpy(yf).long(), br)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % args.eval_every == 0:
            elapsed = time.time() - t0

            # Evaluate with different numbers of refinement iterations
            bler_0, bu0, bv0 = evaluate(model, channel, N, Au, Av, fu, fv, args.eval_cw, n_iters=0)
            bler_1, bu1, bv1 = evaluate(model, channel, N, Au, Av, fu, fv, args.eval_cw, n_iters=1)
            bler_2, bu2, bv2 = evaluate(model, channel, N, Au, Av, fu, fv, args.eval_cw, n_iters=2)

            improved = ''
            if bler_2 < best_bler:
                best_bler = bler_2
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'iterative_npd_mac_N{N}.pt'))
                improved = ' *BEST*'

            ratio = bler_2 / max(sc_bler, 1e-8)
            print(f'[{it:>6}/{args.iters}] loss={loss.item():.4f} '
                  f'iter0={bler_0:.4f}(U:{bu0:.3f},V:{bv0:.3f}) '
                  f'iter1={bler_1:.4f}(U:{bu1:.3f},V:{bv1:.3f}) '
                  f'iter2={bler_2:.4f}(U:{bu2:.3f},V:{bv2:.3f}) '
                  f'SC={sc_bler} ratio={ratio:.2f}x '
                  f'{elapsed/60:.0f}min lr={lr_now:.1e}{improved}', flush=True)

    elapsed = time.time() - t0
    print(f'\n{"="*70}', flush=True)
    print(f'  DONE: best BLER (2 iters) = {best_bler:.4f}, SC = {sc_bler}', flush=True)
    print(f'  Ratio: {best_bler/max(sc_bler,1e-8):.2f}x', flush=True)
    print(f'  Time: {elapsed/60:.0f} min', flush=True)
    print(f'{"="*70}', flush=True)


if __name__ == '__main__':
    main()
