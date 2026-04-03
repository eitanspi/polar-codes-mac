#!/usr/bin/env python3
"""
dine_mac.py — DINE (Differentiable Information Neural Estimator) for MAC channels.

Learns a z_encoder that maps channel outputs to embeddings without knowing
the channel transition probabilities. Instead of using analytical W(z|x,y),
trains the z_encoder using a MINE-like contrastive objective that maximizes
mutual information between the embedding and the true (x,y) pairs.

This enables the neural decoder to work on UNKNOWN channels.

Architecture:
    z_encoder: z → R^d  (same as in ncg_gmac.py)
    Trained via contrastive loss:
        L = E[T(z,x)] - log E[exp(T(z,x'))]
    where x' is a shuffled version of x, and T is a critic network.

After training the z_encoder, the full decoder (tree operations) is trained
end-to-end using the standard cross-entropy loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder, _make_mlp


class MINECritic(nn.Module):
    """Critic network for MINE-based mutual information estimation."""

    def __init__(self, z_dim, x_dim, hidden=128, n_layers=2):
        super().__init__()
        self.net = _make_mlp(z_dim + x_dim, hidden, 1, n_layers)

    def forward(self, z_emb, x_enc):
        """
        Parameters
        ----------
        z_emb : (B, d) — channel embedding
        x_enc : (B, x_dim) — encoded (x, y) input pair

        Returns
        -------
        score : (B, 1) — critic score T(z, x)
        """
        return self.net(torch.cat([z_emb, x_enc], dim=-1))


class DINEZEncoder(nn.Module):
    """
    DINE z_encoder: learns channel embeddings via contrastive training.

    Phase 1: Train z_encoder + critic using MINE objective
    Phase 2: Freeze z_encoder, train tree decoder with standard CE loss
    (or fine-tune end-to-end)
    """

    def __init__(self, d=16, z_hidden=32, n_contrastive=5):
        super().__init__()
        self.d = d
        self.n_contrastive = n_contrastive

        # z_encoder: continuous channel output → d-dim embedding
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden),
            nn.ELU(),
            nn.Linear(z_hidden, d),
        )

        # Critic for MINE: (d + 4) → 1
        # x_enc is 4-dim one-hot encoding of (x, y) ∈ {(0,0),(0,1),(1,0),(1,1)}
        self.critic = MINECritic(z_dim=d, x_dim=4, hidden=128, n_layers=2)

    def encode(self, z):
        """Encode channel outputs to embeddings.

        Parameters
        ----------
        z : (B, N) float — channel outputs

        Returns
        -------
        emb : (B, N, d) — embeddings
        """
        return self.z_encoder(z.unsqueeze(-1))

    def mine_loss(self, z, x, y):
        """
        Compute MINE loss for training z_encoder.

        Uses the Donsker-Varadhan representation:
            I(Z; X,Y) >= E[T(z, xy)] - log E[exp(T(z, x'y'))]

        Parameters
        ----------
        z : (B, N) float — channel outputs
        x : (B, N) int — user 1 codeword bits
        y : (B, N) int — user 2 codeword bits

        Returns
        -------
        loss : scalar — negative MINE bound (to minimize)
        """
        B, N = z.shape

        # Encode z
        z_emb = self.encode(z)  # (B, N, d)
        z_flat = z_emb.reshape(B * N, self.d)  # (B*N, d)

        # Encode (x, y) as one-hot
        xy_idx = (x * 2 + y).reshape(B * N)  # (B*N,)
        xy_onehot = F.one_hot(xy_idx.long(), 4).float()  # (B*N, 4)

        # Positive pairs: T(z_i, xy_i)
        t_pos = self.critic(z_flat, xy_onehot).squeeze(-1)  # (B*N,)

        # Negative pairs: T(z_i, xy_j) with shuffled xy
        t_neg_list = []
        for _ in range(self.n_contrastive):
            perm = torch.randperm(B * N)
            xy_shuffled = xy_onehot[perm]
            t_neg = self.critic(z_flat, xy_shuffled).squeeze(-1)
            t_neg_list.append(t_neg)

        t_neg = torch.cat(t_neg_list, dim=0)  # (n_contrastive * B*N,)

        # MINE loss: - (E[T_pos] - log E[exp(T_neg)])
        loss = -(t_pos.mean() - torch.logsumexp(t_neg, dim=0) + np.log(len(t_neg)))

        return loss


def train_dine_z_encoder(channel, N, n_iters=10000, batch_size=64,
                          lr=1e-3, d=16, z_hidden=32):
    """
    Phase 1: Train z_encoder using MINE on raw channel data.

    No polar coding involved — just (x, y) → z → embedding.
    """
    print(f"\n  DINE Phase 1: Training z_encoder (N={N}, {n_iters} iters)")
    dine = DINEZEncoder(d=d, z_hidden=z_hidden)
    opt = torch.optim.Adam(dine.parameters(), lr=lr)

    rng = np.random.default_rng(42)
    losses = []

    for it in range(1, n_iters + 1):
        # Generate random (x, y) pairs and channel outputs
        x = rng.integers(0, 2, size=(batch_size, N)).astype(np.int32)
        y = rng.integers(0, 2, size=(batch_size, N)).astype(np.int32)
        z = channel.sample_batch(x, y).astype(np.float32)

        xt = torch.from_numpy(x).float()
        yt = torch.from_numpy(y).float()
        zt = torch.from_numpy(z).float()

        loss = dine.mine_loss(zt, xt, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if it % (n_iters // 10) == 0:
            avg_loss = np.mean(losses[-200:])
            # MINE bound on MI: -loss ≈ I(Z; X,Y)
            mi_est = -avg_loss
            print(f"    [{it}/{n_iters}] MINE loss={avg_loss:.4f}, MI estimate={mi_est:.4f} bits")

    # Compute final MI estimate
    final_mi = -np.mean(losses[-500:])
    true_mi = channel.capacity()[2] if hasattr(channel, 'capacity') else None
    print(f"  DINE MI estimate: {final_mi:.4f}" +
          (f" (true: {true_mi:.4f})" if true_mi else ""))

    return dine


def train_decoder_with_dine(dine, channel, N, ku, kv, Au, Av, fu, fv, b,
                             n_iters=30000, batch_size=16, lr=5e-4):
    """
    Phase 2: Train full decoder using DINE z_encoder.

    The z_encoder is either frozen or fine-tuned.
    """
    print(f"\n  DINE Phase 2: Training decoder (N={N}, {n_iters} iters)")

    d = dine.d
    tree = PureNeuralCompGraphDecoder(d=d, hidden=64, n_layers=2)

    # Combine z_encoder and tree
    class DINEDecoder(nn.Module):
        def __init__(self, z_enc, tree):
            super().__init__()
            self.z_encoder = z_enc.z_encoder
            self.tree = tree

        def forward(self, z, b, fu, fv, u_true=None, v_true=None):
            n = int(math.log2(z.shape[1]))
            br = torch.from_numpy(bit_reversal_perm(n)).long()
            root = self.z_encoder(z.unsqueeze(-1))[:, br]
            return self.tree(z=None, b=b, frozen_u=fu, frozen_v=fv,
                             u_true=u_true, v_true=v_true, root_emb=root)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    model = DINEDecoder(dine, tree)
    print(f"  Parameters: {model.count_parameters():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iters, eta_min=lr * 0.01)

    rng = np.random.default_rng(42)
    losses = []
    best_bler = 1.0

    model.train()
    for it in range(1, n_iters + 1):
        uf = np.zeros((batch_size, N), dtype=int)
        vf = np.zeros((batch_size, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, batch_size)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, batch_size)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf).astype(np.float32)
        zt = torch.from_numpy(zf).float()
        ut = torch.from_numpy(uf).float()
        vt = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _, _ = model(
            zt, b, fu, fv, u_true=ut, v_true=vt)

        if len(all_logits) > 0:
            logits = torch.stack(all_logits, dim=1)
            targets = torch.stack(all_targets, dim=1)
            loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            losses.append(loss.item())

        if it % max(1000, n_iters // 10) == 0:
            # Evaluate
            model.eval()
            errs = 0; total = 0
            eval_rng = np.random.default_rng(999)
            with torch.no_grad():
                while total < 500:
                    actual = min(25, 500 - total)
                    uf2 = np.zeros((actual, N), dtype=int)
                    vf2 = np.zeros((actual, N), dtype=int)
                    for p in Au: uf2[:, p-1] = eval_rng.integers(0, 2, actual)
                    for p in Av: vf2[:, p-1] = eval_rng.integers(0, 2, actual)
                    xf2 = polar_encode_batch(uf2); yf2 = polar_encode_batch(vf2)
                    zf2 = channel.sample_batch(xf2, yf2).astype(np.float32)
                    zt2 = torch.from_numpy(zf2).float()
                    _, _, uh, vh, _ = model(zt2, b, fu, fv)
                    for i in range(actual):
                        e = any(int(uh[p][i].item()) != uf2[i, p-1] for p in Au if p in uh) or \
                            any(int(vh[p][i].item()) != vf2[i, p-1] for p in Av if p in vh)
                        if e: errs += 1
                    total += actual

            bler = errs / total
            improved = ''
            if bler < best_bler:
                best_bler = bler
                improved = ' *BEST*'

            avg_loss = np.mean(losses[-200:]) if losses else 0
            print(f"    [{it}/{n_iters}] loss={avg_loss:.4f} BLER={bler:.4f} "
                  f"(best={best_bler:.4f}){improved}")
            model.train()

    return model, best_bler


def main():
    import json, time

    SNR_DB = 6.0
    sigma2 = 10 ** (-SNR_DB / 10)
    channel = GaussianMAC(sigma2=sigma2)

    N = 32
    n = int(math.log2(N))
    ku, kv = 15, 15

    DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
    dp = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr6dB.npz')
    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:ku]])
    Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    b = make_path(N, N // 2)

    print(f"\n{'#'*60}")
    print(f"  DINE/MINE Unknown Channel Decoder — GMAC N={N}")
    print(f"  SNR={SNR_DB}dB, ku={ku}, kv={kv}")
    print(f"  True channel: Z = (1-2X) + (1-2Y) + W, σ²={sigma2:.4f}")
    print(f"  The z_encoder will learn this WITHOUT knowing the formula!")
    print(f"{'#'*60}")

    # Phase 1: Train z_encoder with MINE
    t0 = time.time()
    dine = train_dine_z_encoder(channel, N, n_iters=5000, batch_size=64,
                                 lr=1e-3, d=16, z_hidden=32)
    print(f"  Phase 1 time: {time.time()-t0:.0f}s")

    # Phase 2: Train decoder
    t0 = time.time()
    model, best_bler = train_decoder_with_dine(
        dine, channel, N, ku, kv, Au, Av, fu, fv, b,
        n_iters=20000, batch_size=16, lr=1e-3)
    print(f"  Phase 2 time: {time.time()-t0:.0f}s")

    # Compare with analytical z_encoder (known channel)
    print(f"\n  DINE decoder BLER: {best_bler:.4f}")

    # Save results
    results = {
        'N': N, 'snr_dB': SNR_DB,
        'dine_bler': best_bler,
        'note': 'z_encoder learned without knowing channel formula',
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'dine_mac_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
