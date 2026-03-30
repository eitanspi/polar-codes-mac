#!/usr/bin/env python3
"""
train_gmac_fastce.py — Parallel teacher-forced training for GMAC MAC neural decoder.

Adapts NPD's fast_ce approach (Aharoni et al.) to the 2-user MAC setting.

Key insight: instead of sequential SC tree walk (O(N log N) gradient depth),
process ALL positions at each tree depth in parallel using TRUE bits.
Gradient depth becomes O(log N) regardless of N.

Architecture (following NPD):
  - z_encoder: maps continuous channel output to d-dim embedding
  - CheckNode (f-node): MLP(concat(e_odd, e_even)) -> d-dim
  - BitNode (g-node):   MLP(concat(e_odd * u_sign, e_even)) + residual -> d-dim
  - emb2logits: maps d-dim embedding to 4-class logits (joint u,v decision)

Training:
  - At each depth, split into even/odd, apply CheckNode and BitNode in parallel
  - BitNode receives TRUE left-child bits (teacher forcing)
  - Compute 4-class CE loss at every depth for every position
  - Total loss = mean over all depths and positions

MAC adaptation:
  - 4-class output: (u=0,v=0), (u=0,v=1), (u=1,v=0), (u=1,v=1)
  - Bit transform: u_upper = u_odd XOR u_even (same for each user independently)
  - BitNode uses joint (u,v) sign: u_sign encodes the XOR'd joint state
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
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'train_gmac_fastce_results.json')

SC_REF = {
    32:  {'ku': 15,  'kv': 15,  'sc_bler': 0.046},
    64:  {'ku': 31,  'kv': 31,  'sc_bler': 0.025},
    128: {'ku': 62,  'kv': 62,  'sc_bler': 0.016},
    256: {'ku': 123, 'kv': 123, 'sc_bler': 0.005},
    512: {'ku': 246, 'kv': 246, 'sc_bler': 0.001},
    1024:{'ku': 492, 'kv': 492, 'sc_bler': 0.001},
}

TRAIN_N = 256  # Train at this N
BATCH = 64     # Can use large batch since fast_ce is memory-efficient
LR = 3e-4
TOTAL_ITERS = 100000
EVAL_EVERY = 2000
EVAL_CW = 2000
WARMUP = 2000


# ─── Model ──────────────────────────────────────────────────────────────────

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class FastCE_MAC_Decoder(nn.Module):
    """
    MAC neural decoder trained with parallel fast_ce.

    4-class output per position: joint (u,v) in {00, 01, 10, 11}.
    """

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d

        # Channel embedding: continuous z -> d-dim
        self.z_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d),
        )

        # CheckNode (f-node): (e_odd, e_even) -> e_left
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)

        # BitNode (g-node): (e_odd, e_even, uv_onehot) -> e_right + residual
        # Input: 2d + 4 (4-class one-hot for joint (u,v) of left child)
        self.bitnode_mlp = _make_mlp(2 * d + 4, hidden, d, n_layers)

        # Embedding to 4-class logits (joint u,v)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def bitnode(self, e_odd, e_even, uv_left):
        """
        BitNode (g-node) with residual connection.

        uv_left: (batch, M, 1) integer class index 0-3 for joint (u,v).
        Uses one-hot encoding concatenated with embeddings.
        """
        # One-hot encode the 4-class joint decision
        uv_int = uv_left.squeeze(-1).long()  # (batch, M)
        uv_onehot = F.one_hot(uv_int, 4).float()  # (batch, M, 4)

        inp = torch.cat([e_odd, e_even, uv_onehot], dim=-1)  # (batch, M, 2d+4)
        return self.bitnode_mlp(inp) + e_odd + e_even  # residual!

    def fast_ce(self, emb, u_codeword, v_codeword):
        """
        Parallel teacher-forced CE computation across all tree depths.

        Args:
            emb: (batch, N, d) — channel embeddings (bit-reversed)
            u_codeword: (batch, N) — true U codeword bits
            v_codeword: (batch, N) — true V codeword bits

        Returns:
            total_loss: scalar — mean CE across all depths and positions
        """
        B, N, d = emb.shape
        n = int(math.log2(N))

        # Joint target: 4-class index = u*2 + v
        joint = (u_codeword * 2 + v_codeword).long()  # (batch, N)

        all_losses = []

        # Depth 0: predict from raw embeddings
        logits = self.emb2logits(emb)  # (batch, N, 4)
        loss = F.cross_entropy(logits.reshape(-1, 4), joint.reshape(-1), reduction='mean')
        all_losses.append(loss)

        # Current embeddings and joint bits, stored as list of chunks
        E_chunks = [emb]        # list of tensors, each (batch, chunk_size, d)
        UV_chunks = [joint]     # list of tensors, each (batch, chunk_size)

        for depth in range(n):
            E_odds, E_evens = [], []
            UV_odds, UV_evens = [], []

            # Split each chunk into odd/even (interleaved)
            for e_chunk, uv_chunk in zip(E_chunks, UV_chunks):
                M = e_chunk.shape[1]
                e_r = e_chunk.reshape(B, M // 2, 2, d)
                e_odd = e_r[:, :, 0, :]   # positions 0,2,4,...
                e_even = e_r[:, :, 1, :]  # positions 1,3,5,...
                E_odds.append(e_odd)
                E_evens.append(e_even)

                uv_r = uv_chunk.reshape(B, M // 2, 2)
                UV_odds.append(uv_r[:, :, 0])
                UV_evens.append(uv_r[:, :, 1])

            # Concatenate all chunks for parallel processing
            E_odd = torch.cat(E_odds, dim=1)    # (batch, N/2, d)
            E_even = torch.cat(E_evens, dim=1)  # (batch, N/2, d)
            UV_odd = torch.cat(UV_odds, dim=1)  # (batch, N/2)
            UV_even = torch.cat(UV_evens, dim=1)

            # Compute left-child (f-node) joint bits: XOR per user
            # joint = u*2 + v. XOR each user independently:
            # u_left = u_odd XOR u_even, v_left = v_odd XOR v_even
            u_odd = UV_odd // 2;  v_odd = UV_odd % 2
            u_even = UV_even // 2; v_even = UV_even % 2
            u_left = u_odd ^ u_even  # XOR
            v_left = v_odd ^ v_even
            uv_left = u_left * 2 + v_left  # (batch, N/2)

            # Right-child bits are just the even bits
            uv_right = UV_even  # (batch, N/2)

            # CheckNode: compute left-child embeddings
            inp_check = torch.cat([E_odd, E_even], dim=-1)  # (batch, N/2, 2d)
            e_left = self.checknode(inp_check)  # (batch, N/2, d)

            # BitNode: compute right-child embeddings using TRUE left bits
            e_right = self.bitnode(E_odd, E_even, uv_left.unsqueeze(-1))  # (batch, N/2, d)

            # Split back into chunks and interleave [left_0, right_0, left_1, right_1, ...]
            n_chunks = 2 ** depth
            chunk_size = (N // 2) // n_chunks
            e_lefts = torch.split(e_left, chunk_size, dim=1)
            e_rights = torch.split(e_right, chunk_size, dim=1)
            uv_lefts = torch.split(uv_left, chunk_size, dim=1)
            uv_rights = torch.split(uv_right, chunk_size, dim=1)

            E_chunks = []
            UV_chunks = []
            for l, r, uvl, uvr in zip(e_lefts, e_rights, uv_lefts, uv_rights):
                E_chunks.append(l)
                E_chunks.append(r)
                UV_chunks.append(uvl)
                UV_chunks.append(uvr)

            # Compute loss at this depth
            e_all = torch.cat(E_chunks, dim=1)   # (batch, N, d)
            uv_all = torch.cat(UV_chunks, dim=1) # (batch, N)
            logits = self.emb2logits(e_all)
            loss = F.cross_entropy(logits.reshape(-1, 4), uv_all.reshape(-1), reduction='mean')
            all_losses.append(loss)

        return torch.stack(all_losses).mean()

    def decode_sequential(self, z, b, frozen_u, frozen_v):
        """
        Sequential SC decoding for evaluation (not for training).
        Recursive structure matching fast_ce: split even/odd, checknode left,
        bitnode right using decoded left bits.

        Returns message-domain bits u_hat, v_hat (batch, N).
        Frozen sets are 0-indexed.
        """
        B, N = z.shape
        n = int(math.log2(N))

        br = torch.from_numpy(bit_reversal_perm(n)).long()
        emb = self.z_encoder(z.unsqueeze(-1))[:, br]  # (B, N, d)

        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]  # mutable counter for leaf position

        def _decode(emb_block):
            """Recursively decode. Returns joint decisions (B, block_size)."""
            block_size = emb_block.shape[1]

            if block_size == 1:
                # Leaf: make decision
                logits = self.emb2logits(emb_block[:, 0, :])  # (B, 4)
                idx = leaf_idx[0]
                leaf_idx[0] += 1

                if idx in frozen_u and idx in frozen_v:
                    dec = torch.zeros(B, dtype=torch.long)
                elif idx in frozen_u:
                    # u frozen to 0, pick best v from classes 0,1
                    dec = (logits[:, 1] > logits[:, 0]).long()
                elif idx in frozen_v:
                    # v frozen to 0, pick best u from classes 0,2
                    dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else:
                    dec = logits.argmax(dim=-1)

                u_hat[:, idx] = dec // 2
                v_hat[:, idx] = dec % 2
                return dec.unsqueeze(1)  # (B, 1)

            half = block_size // 2
            e_odd = emb_block[:, 0::2, :]   # (B, half, d)
            e_even = emb_block[:, 1::2, :]  # (B, half, d)

            # CheckNode -> left child embeddings
            inp = torch.cat([e_odd, e_even], dim=-1)
            e_left = self.checknode(inp)  # (B, half, d)

            # Decode left subtree -> get left decisions
            uv_left = _decode(e_left)  # (B, half)

            # BitNode -> right child embeddings using left decisions
            self.bitnode(e_odd, e_even, uv_left.unsqueeze(-1).float())
            e_right = self.bitnode(e_odd, e_even, uv_left.unsqueeze(-1).float())

            # Decode right subtree
            uv_right = _decode(e_right)  # (B, half)

            # Reconstruct codeword bits for parent:
            # x_odd = u_left XOR u_right (per user), x_even = u_right
            # But we don't need to return x, just the message bits
            return torch.cat([uv_left, uv_right], dim=1)  # (B, block_size)

        _decode(emb)
        return u_hat, v_hat

    def forward(self, z, u_true, v_true):
        """Training forward pass using fast_ce."""
        B, N = z.shape
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long()
        emb = self.z_encoder(z.unsqueeze(-1))[:, br]

        # u_true, v_true are message-domain (pre-encode).
        # We need codeword-domain bits for the tree structure.
        # The polar transform F^⊗n maps message bits to codeword bits.
        # fast_ce operates on the codeword domain.
        return self.fast_ce(emb, u_true, v_true)


# ─── Training helpers ───────────────────────────────────────────────────────

def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
    else:
        from polar.design import design_gmac
        Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv


def evaluate(model, channel, N, Au, Av, fu_0idx, fv_0idx, n_cw):
    """Evaluate BLER using sequential SC decoding."""
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    bs = max(1, min(32, 256 // max(1, N // 16)))

    # Convert frozen dicts to 0-indexed sets for decode_sequential
    fu_set = {p - 1 for p in fu_0idx}
    fv_set = {p - 1 for p in fv_0idx}

    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            u_dec, v_dec = model.decode_sequential(zf, None, fu_set, fv_set)

            for i in range(actual):
                ue = any(u_dec[i, p-1].item() != uf[i, p-1] for p in Au)
                ve = any(v_dec[i, p-1].item() != vf[i, p-1] for p in Av)
                if ue or ve:
                    errs += 1
            total += actual
    model.train()
    return errs / total


def get_lr(it, total, base_lr, warmup=WARMUP):
    if it < warmup:
        return base_lr * it / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    N = TRAIN_N
    n = int(math.log2(N))
    ref = SC_REF[N]
    ku, kv = ref['ku'], ref['kv']
    sc_bler = ref['sc_bler']

    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design(N, ku, kv)

    model = FastCE_MAC_Decoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)

    print(f'{"="*60}', flush=True)
    print(f'fast_ce MAC Decoder Training', flush=True)
    print(f'N={N}, d={D}, hidden={HIDDEN}, params={model.count_parameters():,}', flush=True)
    print(f'batch={BATCH}, lr={LR}, iters={TOTAL_ITERS}', flush=True)
    print(f'SNR={SNR_DB}dB, Class B, MC design', flush=True)
    print(f'{"="*60}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng()
    t0 = time.time()
    losses = []
    best_bler = 1.0
    best_state = None

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        lr_now = get_lr(it, TOTAL_ITERS, LR)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        # Generate training batch: random messages -> encode -> channel
        uf = np.zeros((BATCH, N), dtype=int)
        vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        # fast_ce needs CODEWORD bits (post-encoding), not message bits
        u_cw = torch.from_numpy(xf).long()
        v_cw = torch.from_numpy(yf).long()

        loss = model(zf, u_cw, v_cw)

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
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(SAVE_DIR, f'ncg_gmac_fastce_N{N}.pt'))
                improved = ' *BEST*'

            ratio = bler / max(sc_bler, 1e-8)
            print(f'[{it:>6}/{TOTAL_ITERS}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}, SC={sc_bler}, ratio={ratio:.1f}x) '
                  f'{elapsed/60:.0f}min lr={lr_now:.1e}{improved}', flush=True)

            # Also test at different N to check generalization
            if it % (EVAL_EVERY * 5) == 0:
                for test_N in [64, 128, 512]:
                    if test_N == N:
                        continue
                    test_ref = SC_REF.get(test_N)
                    if test_ref is None:
                        continue
                    test_Au, test_Av, test_fu, test_fv = load_design(
                        test_N, test_ref['ku'], test_ref['kv'])
                    test_bler = evaluate(model, channel, test_N,
                                       test_Au, test_Av, test_fu, test_fv,
                                       min(EVAL_CW, 1000))
                    print(f'  [generalize] N={test_N}: BLER={test_bler:.4f} '
                          f'(SC={test_ref["sc_bler"]})', flush=True)

    # Final
    elapsed = time.time() - t0
    print(f'\nDONE: best_bler={best_bler:.4f} (SC={sc_bler}), '
          f'{elapsed/60:.0f}min', flush=True)

    results = {
        'N': N, 'd': D, 'hidden': HIDDEN,
        'best_bler': best_bler, 'sc_bler': sc_bler,
        'params': model.count_parameters(),
        'iters': TOTAL_ITERS, 'time_min': elapsed / 60,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
