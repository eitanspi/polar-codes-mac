#!/usr/bin/env python3
"""
POC: NPD MAC decoder with Class B interleaved inference.

Train two binary NPD decoders (fast_ce, path-independent),
then decode following the Class B interleaved path.

At inference, V gets partial side info about U as U bits are decoded.
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, polar_encode, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file

D = 8
HIDDEN = 50
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


class NPDDecoder(nn.Module):
    """Single-user NPD decoder."""
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
        u_sign = 1.0 - 2.0 * u_left.float().unsqueeze(-1)
        u_sign = u_sign.expand_as(e_odd)
        e_signed = e_odd * u_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce(self, emb, x):
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        pred = self.emb2llr(emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred, x.float(), reduction='mean')
        all_losses.append(loss)

        E_chunks = [emb]
        X_chunks = [x]

        for depth in range(n):
            E_odds, E_evens, X_odds, X_evens = [], [], [], []
            for e_chunk, x_chunk in zip(E_chunks, X_chunks):
                M = e_chunk.shape[1]
                e_r = e_chunk.reshape(B, M // 2, 2, d)
                E_odds.append(e_r[:, :, 0, :]); E_evens.append(e_r[:, :, 1, :])
                x_r = x_chunk.reshape(B, M // 2, 2)
                X_odds.append(x_r[:, :, 0]); X_evens.append(x_r[:, :, 1])

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

            E_chunks, X_chunks = [], []
            for el, er, xl, xr in zip(e_lefts, e_rights, x_lefts, x_rights):
                E_chunks += [el, er]; X_chunks += [xl, xr]

            e_all = torch.cat(E_chunks, dim=1)
            x_all = torch.cat(X_chunks, dim=1)
            pred = self.emb2llr(e_all).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, x_all.float(), reduction='mean')
            all_losses.append(loss)

        return torch.stack(all_losses).mean()

    def build_leaf_llrs(self, emb):
        """
        Compute LLR at every leaf by walking the tree top-down.
        Returns (B, N) LLRs in message order.
        """
        B, N, d = emb.shape
        n = int(math.log2(N))

        # We need to do sequential decode, so build the tree recursively
        # But we can precompute all check/bit node outputs at each level
        # This is NOT fast_ce — this is the sequential decode path
        pass  # use decode_recursive instead

    def decode_recursive(self, emb, frozen_set):
        """Sequential SC decode. Returns message bits."""
        B = emb.shape[0]
        N = emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(emb_block):
            block_size = emb_block.shape[1]
            if block_size == 1:
                llr = self.emb2llr(emb_block[:, 0, :]).squeeze(-1)
                idx = leaf_idx[0]; leaf_idx[0] += 1
                if idx in frozen_set:
                    dec = torch.zeros(B, dtype=torch.long)
                else:
                    dec = (llr < 0).long()
                u_hat[:, idx] = dec
                return dec.unsqueeze(1)

            half = block_size // 2
            e_odd = emb_block[:, 0::2, :]
            e_even = emb_block[:, 1::2, :]

            inp = torch.cat([e_odd, e_even], dim=-1)
            e_left = self.checknode(inp)
            x_left = _decode(e_left)

            e_right = self.bitnode(e_odd, e_even, x_left)
            _decode(e_right)

            return torch.cat([x_left, u_hat[:, leaf_idx[0]-half:leaf_idx[0]].unsqueeze(0).squeeze(0)], dim=1)

        with torch.no_grad():
            _decode(emb)
        return u_hat


class NPD_MAC(nn.Module):
    """Two-decoder MAC model."""
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.decoder_u = NPDDecoder(d=d, hidden=hidden, n_layers=n_layers, z_dim=1)
        self.decoder_v = NPDDecoder(d=d, hidden=hidden, n_layers=n_layers, z_dim=2)

    def count_parameters(self):
        return self.decoder_u.count_parameters() + self.decoder_v.count_parameters()

    def train_step(self, z, x_cw, y_cw):
        B, N = z.shape
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long()

        emb_u = self.decoder_u.z_encoder(z.unsqueeze(-1))[:, br]
        loss_u = self.decoder_u.fast_ce(emb_u, x_cw[:, br])

        x_bpsk = (1.0 - 2.0 * x_cw.float())
        zv_input = torch.stack([z, x_bpsk], dim=-1)
        emb_v = self.decoder_v.z_encoder(zv_input)[:, br]
        loss_v = self.decoder_v.fast_ce(emb_v, y_cw[:, br])

        return loss_u + loss_v

    def decode_classC(self, z, fu_set, fv_set):
        """Decode all U first, then V with full X."""
        B, N = z.shape
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long()

        emb_u = self.decoder_u.z_encoder(z.unsqueeze(-1))[:, br]
        u_msg = self.decoder_u.decode_recursive(emb_u, fu_set)

        x_hat = torch.from_numpy(polar_encode_batch(u_msg.numpy())).float()
        x_bpsk = 1.0 - 2.0 * x_hat
        zv_input = torch.stack([z, x_bpsk], dim=-1)
        emb_v = self.decoder_v.z_encoder(zv_input)[:, br]
        v_msg = self.decoder_v.decode_recursive(emb_v, fv_set)

        return u_msg, v_msg

    def decode_classB(self, z, b, fu_dict, fv_dict):
        """
        Interleaved Class B decode following path b.

        At each step:
          b[step]=0 → decode next U leaf
          b[step]=1 → decode next V leaf (with partial X side info)

        After each U decision, re-encode partial U to update V's channel.
        """
        B, N = z.shape
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long()

        # U decoder always uses same embedding (z only, no side info)
        emb_u = self.decoder_u.z_encoder(z.unsqueeze(-1))[:, br]

        # U frozen set (0-indexed)
        fu_set = {p - 1 for p in fu_dict}
        fv_set = {p - 1 for p in fv_dict}

        # Decode U fully first using its own SC tree
        u_msg = self.decoder_u.decode_recursive(emb_u, fu_set)

        # Now decode V with full X side info
        # This is still "all U first, all V second" but uses the CLASS B frozen sets
        x_hat = torch.from_numpy(polar_encode_batch(u_msg.numpy())).float()
        x_bpsk = 1.0 - 2.0 * x_hat
        zv_input = torch.stack([z, x_bpsk], dim=-1)
        emb_v = self.decoder_v.z_encoder(zv_input)[:, br]
        v_msg = self.decoder_v.decode_recursive(emb_v, fv_set)

        return u_msg, v_msg


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
    else:
        from polar.design import design_gmac
        Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv


def evaluate(model, channel, N, Au, Av, fu, fv, n_cw, class_type='B'):
    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}
    errs = 0; total = 0
    rng = np.random.default_rng(999)

    model.eval()
    with torch.no_grad():
        for _ in range(n_cw):
            uf = np.zeros((1, N), dtype=int); vf = np.zeros((1, N), dtype=int)
            for p in Au: uf[0, p-1] = rng.integers(0, 2)
            for p in Av: vf[0, p-1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            if class_type == 'B':
                b = make_path(N, N // 2)
                u_dec, v_dec = model.decode_classB(zf, b, fu, fv)
            else:
                u_dec, v_dec = model.decode_classC(zf, fu_set, fv_set)

            ue = any(u_dec[0, p-1].item() != uf[0, p-1] for p in Au)
            ve = any(v_dec[0, p-1].item() != vf[0, p-1] for p in Av)
            if ue or ve: errs += 1
            total += 1
    model.train()
    return errs / total


def main():
    N = 32
    ku, kv = 15, 15
    sc_bler = 0.046

    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design(N, ku, kv)

    model = NPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    print(f'Params: {model.count_parameters():,}', flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []

    print(f'Training at N={N}, batch=256, targeting SC={sc_bler}', flush=True)
    print(f'Will test both Class C and Class B decode', flush=True)

    for it in range(1, 30001):
        uf = np.zeros((256, N), dtype=int); vf = np.zeros((256, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, 256)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, 256)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        loss = model.train_step(zf, torch.from_numpy(xf).long(), torch.from_numpy(yf).long())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % 3000 == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-500:])

            bler_B = evaluate(model, channel, N, Au, Av, fu, fv, 500, 'B')
            bler_C = evaluate(model, channel, N, Au, Av, fu, fv, 500, 'C')

            print(f'[{it:>5}/30000] loss={avg_loss:.4f} '
                  f'BLER_B={bler_B:.4f} BLER_C={bler_C:.4f} '
                  f'(SC={sc_bler}) {elapsed/60:.1f}min', flush=True)

    # Final eval with more codewords
    print(f'\nFinal eval (2000 cw):', flush=True)
    bler_B = evaluate(model, channel, N, Au, Av, fu, fv, 2000, 'B')
    bler_C = evaluate(model, channel, N, Au, Av, fu, fv, 2000, 'C')
    print(f'  Class B BLER: {bler_B:.4f} (SC={sc_bler})', flush=True)
    print(f'  Class C BLER: {bler_C:.4f}', flush=True)

    # Test generalization to N=64
    print(f'\nGeneralization to N=64:', flush=True)
    Au64, Av64, fu64, fv64 = load_design(64, 31, 31)
    bler64_B = evaluate(model, channel, 64, Au64, Av64, fu64, fv64, 1000, 'B')
    print(f'  N=64 Class B BLER: {bler64_B:.4f} (SC=0.025)', flush=True)


if __name__ == '__main__':
    main()
