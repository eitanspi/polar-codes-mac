#!/usr/bin/env python3
"""
train_gmac_npd_mac.py — Two separate NPD decoders for MAC (one per user).

Instead of joint 4-class prediction, use two independent binary NPD decoders:
  Decoder U: z -> u_hat  (marginal channel, no side info)
  Decoder V: (z, x_hat) -> v_hat  (conditional channel, uses decoded U codeword)

Each decoder is a standard single-user NPD with fast_ce training (binary CE,
gradient depth O(log N), proven to scale to N=1024+).

For Class C (path 0^N 1^N): decode all U first, then all V with X known.
For Class B (interleaved): decode all U first (marginal), then all V (conditional).
This is suboptimal for Class B but gives a working baseline.
"""
import sys, os, math, time, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file

# ─── Config ──────────────────────────────────────────────────────────────────

D = 8
HIDDEN = 50
N_LAYERS = 2

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'train_gmac_npd_mac_results.json')

SC_REF = {
    32:  {'ku': 15,  'kv': 15,  'sc_bler': 0.046},
    64:  {'ku': 31,  'kv': 31,  'sc_bler': 0.025},
    128: {'ku': 62,  'kv': 62,  'sc_bler': 0.016},
    256: {'ku': 123, 'kv': 123, 'sc_bler': 0.005},
    512: {'ku': 246, 'kv': 246, 'sc_bler': 0.001},
}

TRAIN_N = 128
BATCH = 128
LR = 3e-4
TOTAL_ITERS = 50000
EVAL_EVERY = 2000
EVAL_CW = 2000


# ─── Single-user NPD decoder (following Aharoni et al.) ────────────────────

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class NPDDecoder(nn.Module):
    """Single-user NPD decoder with fast_ce training."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_dim=1):
        super().__init__()
        self.d = d

        # Channel embedding
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ELU(), nn.Linear(hidden, d),
        )

        # CheckNode (f-node): (e_odd, e_even) -> e_left
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)

        # BitNode (g-node): (e_odd * u_sign, e_even) -> e_right + residual
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)

        # Embedding to LLR (binary decision)
        self.emb2llr = _make_mlp(d, hidden, 1, n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def bitnode(self, e_odd, e_even, u_left):
        """BitNode with residual. u_left: (batch, M) binary bits."""
        u_sign = 1.0 - 2.0 * u_left.float().unsqueeze(-1)  # (batch, M, 1)
        u_sign = u_sign.expand_as(e_odd)  # (batch, M, d)
        e_signed = e_odd * u_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even  # residual!

    def fast_ce(self, emb, x):
        """
        Parallel teacher-forced binary CE across all tree depths.

        Args:
            emb: (batch, N, d) — channel embeddings
            x:   (batch, N) — true CODEWORD bits (0/1)

        Returns:
            total_loss: scalar
        """
        B, N, d = emb.shape
        n = int(math.log2(N))

        all_losses = []

        # Depth 0: predict from raw embeddings
        pred = self.emb2llr(emb).squeeze(-1)  # (batch, N)
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

            # Left child bits: XOR
            X_left = X_odd ^ X_even
            # Right child bits: identity
            X_right = X_even

            # CheckNode -> left embeddings
            inp = torch.cat([E_odd, E_even], dim=-1)
            e_left = self.checknode(inp)

            # BitNode -> right embeddings (teacher-forced with TRUE left bits)
            e_right = self.bitnode(E_odd, E_even, X_left)

            # Split and interleave
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

            # Loss at this depth
            e_all = torch.cat(E_chunks, dim=1)
            x_all = torch.cat(X_chunks, dim=1)
            pred = self.emb2llr(e_all).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, x_all.float(), reduction='mean')
            all_losses.append(loss)

        return torch.stack(all_losses).mean()

    def decode_recursive(self, emb, frozen_set):
        """Sequential SC decode. Returns message bits in order."""
        B = emb.shape[0]
        N = emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(emb_block):
            block_size = emb_block.shape[1]
            if block_size == 1:
                llr = self.emb2llr(emb_block[:, 0, :]).squeeze(-1)  # (B,)
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                if idx in frozen_set:
                    dec = torch.zeros(B, dtype=torch.long)
                else:
                    dec = (llr < 0).long()  # LLR < 0 means bit=1
                u_hat[:, idx] = dec
                return dec.unsqueeze(1)

            half = block_size // 2
            e_odd = emb_block[:, 0::2, :]
            e_even = emb_block[:, 1::2, :]

            # CheckNode -> left
            inp = torch.cat([e_odd, e_even], dim=-1)
            e_left = self.checknode(inp)
            x_left = _decode(e_left)  # (B, half)

            # BitNode -> right
            e_right = self.bitnode(e_odd, e_even, x_left)
            x_right = _decode(e_right)

            return torch.cat([x_left, x_right], dim=1)

        with torch.no_grad():
            _decode(emb)
        return u_hat


# ─── MAC Decoder: two NPDs ─────────────────────────────────────────────────

class NPD_MAC_Decoder(nn.Module):
    """
    Two-user MAC decoder using two independent NPD decoders.

    Decoder U: processes z alone (marginal channel)
    Decoder V: processes (z, x_hat) together (conditional channel)
    """

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.decoder_u = NPDDecoder(d=d, hidden=hidden, n_layers=n_layers, z_dim=1)
        self.decoder_v = NPDDecoder(d=d, hidden=hidden, n_layers=n_layers, z_dim=2)

    def count_parameters(self):
        return self.decoder_u.count_parameters() + self.decoder_v.count_parameters()

    def forward(self, z, x_cw, y_cw):
        """
        Training forward pass.

        z: (B, N) channel output
        x_cw: (B, N) User U codeword
        y_cw: (B, N) User V codeword
        """
        B, N = z.shape
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long()

        # Decoder U: z only
        emb_u = self.decoder_u.z_encoder(z.unsqueeze(-1))[:, br]
        loss_u = self.decoder_u.fast_ce(emb_u, x_cw[:, br])

        # Decoder V: (z, x_hat) — use TRUE x for teacher forcing during training
        x_bpsk = (1.0 - 2.0 * x_cw.float())  # BPSK: 0->+1, 1->-1
        zv_input = torch.stack([z, x_bpsk], dim=-1)  # (B, N, 2)
        emb_v = self.decoder_v.z_encoder(zv_input)[:, br]
        loss_v = self.decoder_v.fast_ce(emb_v, y_cw[:, br])

        return loss_u + loss_v

    def decode(self, z, frozen_u_set, frozen_v_set):
        """
        Sequential decode: first U, then V conditioned on decoded U.
        """
        B, N = z.shape
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long()

        self.eval()
        with torch.no_grad():
            # Decode U
            emb_u = self.decoder_u.z_encoder(z.unsqueeze(-1))[:, br]
            u_msg = self.decoder_u.decode_recursive(emb_u, frozen_u_set)

            # Re-encode U to get x_hat (codeword domain)
            x_hat = torch.from_numpy(
                polar_encode_batch(u_msg.numpy())).float()

            # Decode V conditioned on x_hat
            x_bpsk = 1.0 - 2.0 * x_hat
            zv_input = torch.stack([z, x_bpsk], dim=-1)
            emb_v = self.decoder_v.z_encoder(zv_input)[:, br]
            v_msg = self.decoder_v.decode_recursive(emb_v, frozen_v_set)

        return u_msg, v_msg


# ─── Training ──────────────────────────────────────────────────────────────

def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
    else:
        from polar.design import design_gmac
        Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv


def evaluate(model, channel, N, Au, Av, fu, fv, n_cw):
    fu_set = {p - 1 for p in fu}
    fv_set = {p - 1 for p in fv}
    errs = 0; total = 0
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

        u_dec, v_dec = model.decode(zf, fu_set, fv_set)

        for i in range(actual):
            ue = any(u_dec[i, p-1].item() != uf[i, p-1] for p in Au)
            ve = any(v_dec[i, p-1].item() != vf[i, p-1] for p in Av)
            if ue or ve:
                errs += 1
        total += actual
    model.train()
    return errs / total


def get_lr(it, total, base_lr, warmup=2000):
    if it < warmup:
        return base_lr * it / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    N = TRAIN_N
    ref = SC_REF[N]
    ku, kv = ref['ku'], ref['kv']
    sc_bler = ref['sc_bler']

    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design(N, ku, kv)

    model = NPD_MAC_Decoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)

    print(f'{"="*60}', flush=True)
    print(f'NPD MAC Decoder (two binary decoders)', flush=True)
    print(f'N={N}, d={D}, hidden={HIDDEN}', flush=True)
    print(f'Decoder U params: {model.decoder_u.count_parameters():,}', flush=True)
    print(f'Decoder V params: {model.decoder_v.count_parameters():,}', flush=True)
    print(f'Total params: {model.count_parameters():,}', flush=True)
    print(f'batch={BATCH}, lr={LR}, iters={TOTAL_ITERS}', flush=True)
    print(f'{"="*60}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    best_bler = 1.0

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        lr_now = get_lr(it, TOTAL_ITERS, LR)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        loss = model(zf, torch.from_numpy(xf).long(), torch.from_numpy(yf).long())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-min(len(losses), 500):])
            bler = evaluate(model, channel, N, Au, Av, fu, fv, EVAL_CW)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'npd_mac_N{N}.pt'))
                improved = ' *BEST*'

            ratio = bler / max(sc_bler, 1e-8)
            print(f'[{it:>6}/{TOTAL_ITERS}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}, SC={sc_bler}, ratio={ratio:.1f}x) '
                  f'{elapsed/60:.0f}min lr={lr_now:.1e}{improved}', flush=True)

            # Cross-N generalization test
            if it % (EVAL_EVERY * 5) == 0:
                for test_N in [32, 64, 256]:
                    if test_N == N:
                        continue
                    tr = SC_REF.get(test_N)
                    if tr is None:
                        continue
                    tAu, tAv, tfu, tfv = load_design(test_N, tr['ku'], tr['kv'])
                    tb = evaluate(model, channel, test_N, tAu, tAv, tfu, tfv, 1000)
                    print(f'  [N={test_N}] BLER={tb:.4f} (SC={tr["sc_bler"]})', flush=True)

    elapsed = time.time() - t0
    print(f'\nDONE: best={best_bler:.4f} (SC={sc_bler}), {elapsed/60:.0f}min', flush=True)


if __name__ == '__main__':
    main()
