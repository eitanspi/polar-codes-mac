#!/usr/bin/env python3
"""
POC: Joint 4-class fast_ce for MAC with SC decode evaluation.

Train with parallel fast_ce (O(log N) gradient depth), evaluate with
sequential SC decode (following the interleaved path b).
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, polar_encode, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file

D = 16
HIDDEN = 64
N_LAYERS = 2
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')


def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class JointMACDecoder(nn.Module):
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def bitnode(self, e_odd, e_even, uv_left):
        """uv_left: (B, M) integer 0-3."""
        u_left = uv_left // 2
        v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([e_odd[:, :, :h] * u_sign, e_odd[:, :, h:] * v_sign], dim=-1)
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce(self, emb, joint_cw):
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 4), joint_cw.reshape(-1), reduction='mean'))

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

            u_o = J_odd // 2; v_o = J_odd % 2
            u_e = J_even // 2; v_e = J_even % 2
            J_left = (u_o ^ u_e) * 2 + (v_o ^ v_e)
            J_right = J_even

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left, cs, 1)
            jr = torch.split(J_right, cs, 1)

            E_chunks = []
            J_chunks = []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

            e_all = torch.cat(E_chunks, 1)
            j_all = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_all)
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4), j_all.reshape(-1), reduction='mean'))

        return torch.stack(all_losses).mean()

    def sc_decode(self, emb, frozen_u, frozen_v):
        """
        Sequential SC decode matching the fast_ce tree structure.
        Returns message-domain u_hat, v_hat.
        frozen_u, frozen_v: sets of 0-indexed frozen positions.
        """
        B = emb.shape[0]
        N = emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(emb_block):
            block_size = emb_block.shape[1]
            if block_size == 1:
                logits = self.emb2logits(emb_block[:, 0, :])  # (B, 4)
                idx = leaf_idx[0]
                leaf_idx[0] += 1

                u_frozen = idx in frozen_u
                v_frozen = idx in frozen_v

                if u_frozen and v_frozen:
                    dec = torch.zeros(B, dtype=torch.long)
                elif u_frozen:
                    # u=0, pick v from classes {0,1}
                    dec = (logits[:, 1] > logits[:, 0]).long()
                elif v_frozen:
                    # v=0, pick u from classes {0,2}
                    dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else:
                    dec = logits.argmax(dim=-1)

                u_hat[:, idx] = dec // 2
                v_hat[:, idx] = dec % 2
                return dec.unsqueeze(1)  # (B, 1)

            half = block_size // 2
            e_odd = emb_block[:, 0::2, :]
            e_even = emb_block[:, 1::2, :]

            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            uv_left = _decode(e_left)  # (B, half)

            e_right = self.bitnode(e_odd, e_even, uv_left)
            _decode(e_right)

            return torch.cat([uv_left, (u_hat[:, leaf_idx[0] - half:leaf_idx[0]] * 2 +
                                         v_hat[:, leaf_idx[0] - half:leaf_idx[0]])], dim=1)

        with torch.no_grad():
            _decode(emb)
        return u_hat, v_hat


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        return design_from_file(mc_path, n, ku, kv)
    from polar.design import design_gmac
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv, None, None, None


def evaluate_bler(model, channel, N, Au, Av, fu, fv, n_cw):
    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}
    n = int(math.log2(N))
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    errs = 0
    rng = np.random.default_rng(999)

    model.eval()
    with torch.no_grad():
        for _ in range(n_cw):
            uf = np.zeros((1, N), dtype=int)
            vf = np.zeros((1, N), dtype=int)
            for p in Au: uf[0, p - 1] = rng.integers(0, 2)
            for p in Av: vf[0, p - 1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode(emb, fu_set, fv_set)
            ue = any(u_dec[0, p - 1].item() != uf[0, p - 1] for p in Au)
            ve = any(v_dec[0, p - 1].item() != vf[0, p - 1] for p in Av)
            if ue or ve:
                errs += 1
    model.train()
    return errs / n_cw


def main():
    N = 32
    ku, kv = 15, 15
    sc_bler = 0.046

    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]

    model = JointMACDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'N={N}, d={D}, hidden={HIDDEN}, params={params:,}', flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    br = torch.from_numpy(bit_reversal_perm(int(math.log2(N)))).long()
    t0 = time.time()

    for it in range(1, 50001):
        uf = np.zeros((128, N), dtype=int)
        vf = np.zeros((128, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, 128)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, 128)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        joint = torch.from_numpy(xf * 2 + yf).long()[:, br]

        loss = model.fast_ce(emb, joint)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            elapsed = time.time() - t0
            bler = evaluate_bler(model, channel, N, Au, Av, fu, fv, 500)
            print(f'[{it:>5}/50000] loss={loss.item():.4f} BLER={bler:.4f} '
                  f'(SC={sc_bler}) {elapsed/60:.1f}min', flush=True)

    # Final eval
    print(f'\nFinal eval (2000 cw):', flush=True)
    bler = evaluate_bler(model, channel, N, Au, Av, fu, fv, 2000)
    print(f'BLER={bler:.4f} (SC={sc_bler})', flush=True)

    # Test at N=64
    Au64, Av64, fu64, fv64 = load_design(64, 31, 31)[:4]
    bler64 = evaluate_bler(model, channel, 64, Au64, Av64, fu64, fv64, 500)
    print(f'N=64 BLER={bler64:.4f} (SC=0.025)', flush=True)


if __name__ == '__main__':
    main()
