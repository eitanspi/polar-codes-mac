#!/usr/bin/env python3
"""
poc_two_phase_v2.py — Two-Phase Iterative Refinement for MAC Polar Codes (WORKING)

Uses the corrected NPD implementation with:
  - Bit-reversal mapped frozen sets
  - Bit-reversal permuted channel output
  - Proper codeword reconstruction in decode

Architecture:
  Phase 1: U_marginal(z) → u_hat          (marginal channel, fast_ce O(log N))
  Phase 2: V_cond(z, x_bpsk) → v_hat      (conditional on X, fast_ce O(log N))
  Phase 3: U_refine(z, y_bpsk) → u_hat     (conditional on Y, fast_ce O(log N))
  Iterate Phases 2-3 for convergence.
"""
import sys, os, math, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from neural.npd_pytorch import NPDSingleUser, npd_encode
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file

# ─── Config ──────────────────────────────────────────────────────────────────

D = 16
HIDDEN = 64
N_LAYERS = 2
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

SC_REF = {
    32:  {'ku': 15,  'kv': 15,  'sc_bler': 0.046},
    64:  {'ku': 31,  'kv': 31,  'sc_bler': 0.025},
    128: {'ku': 62,  'kv': 62,  'sc_bler': 0.016},
    256: {'ku': 123, 'kv': 123, 'sc_bler': 0.005},
}


# ─── Two-Phase Iterative MAC Decoder ─────────────────────────────────────────

class TwoPhaseIterativeMAC(nn.Module):
    """Three NPD decoders for iterative MAC decoding."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.decoder_u_marginal = NPDSingleUser(d=d, hidden=hidden, n_layers=n_layers, z_dim=1)
        self.decoder_v_cond = NPDSingleUser(d=d, hidden=hidden, n_layers=n_layers, z_dim=2)
        self.decoder_u_refine = NPDSingleUser(d=d, hidden=hidden, n_layers=n_layers, z_dim=2)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, z_npd, x_std, y_std, br):
        """
        Training forward. All phases teacher-forced → all O(log N).

        z_npd: (B, N) channel output in NPD order (z_standard[:, br])
        x_std: (B, N) User U standard codeword
        y_std: (B, N) User V standard codeword
        br: bit-reversal permutation
        """
        # NPD-order codewords and BPSK
        x_npd = x_std[:, br]
        y_npd = y_std[:, br]
        x_bpsk_npd = (1.0 - 2.0 * x_npd.float())
        y_bpsk_npd = (1.0 - 2.0 * y_npd.float())

        # Phase 1: U marginal (z only)
        emb_u = self.decoder_u_marginal.ey(z_npd.unsqueeze(-1))
        loss_u1 = self.decoder_u_marginal.fast_ce(emb_u, x_npd)

        # Phase 2: V conditional on TRUE X
        zv_input = torch.stack([z_npd, x_bpsk_npd], dim=-1)
        emb_v = self.decoder_v_cond.ey(zv_input)
        loss_v = self.decoder_v_cond.fast_ce(emb_v, y_npd)

        # Phase 3: U refinement conditional on TRUE Y
        zu_input = torch.stack([z_npd, y_bpsk_npd], dim=-1)
        emb_u2 = self.decoder_u_refine.ey(zu_input)
        loss_u2 = self.decoder_u_refine.fast_ce(emb_u2, x_npd)

        return loss_u1 + loss_v + loss_u2

    @torch.no_grad()
    def decode(self, z_npd, fu_npd, fv_npd, Au_std, Av_std, br, n_iters=2):
        """
        Iterative inference.

        Returns: u_msg_std, v_msg_std (standard-order message bits)
        """
        B, N = z_npd.shape
        self.eval()

        # Phase 1: U marginal
        emb_u = self.decoder_u_marginal.ey(z_npd.unsqueeze(-1))
        u_msg_npd = self.decoder_u_marginal.decode(emb_u, fu_npd)

        # Map NPD message to standard, encode to get codeword
        u_msg_std = self._npd_msg_to_std(u_msg_npd, br)
        x_hat_std = torch.from_numpy(polar_encode_batch(u_msg_std.numpy())).float()
        x_hat_npd_bpsk = 1.0 - 2.0 * x_hat_std[:, br].float()

        for it in range(n_iters):
            # Phase 2: V conditional on current X estimate
            zv_input = torch.stack([z_npd, x_hat_npd_bpsk], dim=-1)
            emb_v = self.decoder_v_cond.ey(zv_input)
            v_msg_npd = self.decoder_v_cond.decode(emb_v, fv_npd)

            v_msg_std = self._npd_msg_to_std(v_msg_npd, br)
            y_hat_std = torch.from_numpy(polar_encode_batch(v_msg_std.numpy())).float()
            y_hat_npd_bpsk = 1.0 - 2.0 * y_hat_std[:, br].float()

            # Phase 3: U refinement conditional on current Y estimate
            zu_input = torch.stack([z_npd, y_hat_npd_bpsk], dim=-1)
            emb_u2 = self.decoder_u_refine.ey(zu_input)
            u_msg_npd = self.decoder_u_refine.decode(emb_u2, fu_npd)

            u_msg_std = self._npd_msg_to_std(u_msg_npd, br)
            x_hat_std = torch.from_numpy(polar_encode_batch(u_msg_std.numpy())).float()
            x_hat_npd_bpsk = 1.0 - 2.0 * x_hat_std[:, br].float()

        # Final V decode
        zv_input = torch.stack([z_npd, x_hat_npd_bpsk], dim=-1)
        emb_v = self.decoder_v_cond.ey(zv_input)
        v_msg_npd = self.decoder_v_cond.decode(emb_v, fv_npd)
        v_msg_std = self._npd_msg_to_std(v_msg_npd, br)

        return u_msg_std, v_msg_std

    def _npd_msg_to_std(self, msg_npd, br):
        """Convert NPD-order message bits to standard order.
        NPD leaf index i corresponds to standard position br[i],
        so standard position j has NPD leaf index inv_br[j] = br[j] (self-inverse)."""
        B, N = msg_npd.shape
        msg_std = torch.zeros_like(msg_npd)
        for i in range(N):
            msg_std[:, br[i]] = msg_npd[:, i]
        return msg_std


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model, channel, N, Au, Av, fu_npd, fv_npd, Au_std, Av_std, br,
             n_cw, n_iters=2):
    errs_u = 0; errs_v = 0; errs_block = 0; total = 0
    rng = np.random.default_rng(999)
    bs = max(1, min(16, 128 // max(1, N // 16)))

    model.eval()
    while total < n_cw:
        actual = min(bs, n_cw - total)
        uf = np.zeros((actual, N), dtype=int)
        vf = np.zeros((actual, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        zf_npd = torch.from_numpy(zf[:, br]).float()

        u_dec, v_dec = model.decode(zf_npd, fu_npd, fv_npd, Au_std, Av_std, br, n_iters)

        for i in range(actual):
            ue = any(u_dec[i, p-1].item() != uf[i, p-1] for p in Au)
            ve = any(v_dec[i, p-1].item() != vf[i, p-1] for p in Av)
            if ue: errs_u += 1
            if ve: errs_v += 1
            if ue or ve: errs_block += 1
        total += actual

    return errs_block / total, errs_u / total, errs_v / total


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--eval_cw', type=int, default=500)
    args = parser.parse_args()

    N = args.N
    n = int(math.log2(N))
    ref = SC_REF[N]
    ku, kv = ref['ku'], ref['kv']
    sc_bler = ref['sc_bler']

    channel = GaussianMAC(sigma2=SIGMA2)
    br = bit_reversal_perm(n)
    br_torch = torch.from_numpy(br).long()

    # Load design
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{int(SNR_DB)}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
    Au_std = [p-1 for p in Au]
    Av_std = [p-1 for p in Av]

    # Map frozen sets to NPD indexing
    fu_npd = {int(br[p-1]) for p in fu}
    fv_npd = {int(br[p-1]) for p in fv}

    model = TwoPhaseIterativeMAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS)

    cap = channel.capacity()
    print(f'{"="*70}')
    print(f'  Two-Phase Iterative MAC Decoder v2 (FIXED)')
    print(f'  N={N}, ku={ku}, kv={kv}, SNR={SNR_DB}dB')
    print(f'  I(Z;X)={cap[0]:.4f}, I(Z;Y|X)={cap[1]:.4f}, R_u={ku/N:.4f}')
    print(f'  Total params: {model.count_parameters():,}')
    print(f'  Training depth: O(log N) = {n} per phase')
    if ku/N > cap[0]:
        print(f'  NOTE: R_u > I(Z;X) — Phase 1 alone cannot succeed, refinement needed')
    print(f'{"="*70}')
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
        zf = channel.sample_batch(xf, yf)

        # Convert to NPD order
        zf_npd = torch.from_numpy(zf[:, br]).float()
        xf_t = torch.from_numpy(xf).long()
        yf_t = torch.from_numpy(yf).long()

        loss = model(zf_npd, xf_t, yf_t, br_torch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % args.eval_every == 0:
            elapsed = time.time() - t0

            bler_0, bu0, bv0 = evaluate(model, channel, N, Au, Av, fu_npd, fv_npd,
                                         Au_std, Av_std, br, args.eval_cw, n_iters=0)
            bler_1, bu1, bv1 = evaluate(model, channel, N, Au, Av, fu_npd, fv_npd,
                                         Au_std, Av_std, br, args.eval_cw, n_iters=1)
            bler_2, bu2, bv2 = evaluate(model, channel, N, Au, Av, fu_npd, fv_npd,
                                         Au_std, Av_std, br, args.eval_cw, n_iters=2)

            improved = ''
            if bler_2 < best_bler:
                best_bler = bler_2
                torch.save(model.state_dict(),
                           os.path.join(SAVE_DIR, f'two_phase_v2_N{N}.pt'))
                improved = ' *BEST*'

            ratio = bler_2 / max(sc_bler, 1e-8)
            print(f'[{it:>6}/{args.iters}] loss={loss.item():.4f} '
                  f'iter0={bler_0:.4f}(U:{bu0:.3f},V:{bv0:.3f}) '
                  f'iter1={bler_1:.4f}(U:{bu1:.3f},V:{bv1:.3f}) '
                  f'iter2={bler_2:.4f}(U:{bu2:.3f},V:{bv2:.3f}) '
                  f'SC={sc_bler} ratio={ratio:.2f}x '
                  f'{elapsed/60:.0f}min lr={lr_now:.1e}{improved}', flush=True)

    elapsed = time.time() - t0
    print(f'\nDONE: best={best_bler:.4f} SC={sc_bler} ratio={best_bler/max(sc_bler,1e-8):.2f}x '
          f'{elapsed/60:.0f}min')


if __name__ == '__main__':
    main()
