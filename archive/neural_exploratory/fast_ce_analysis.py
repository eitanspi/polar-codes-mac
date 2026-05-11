#!/usr/bin/env python3
"""
Comprehensive analysis: Where does the fast_ce gap open for 4-class MAC?

Tests at N=4, 8, 16, 32 with:
1. 4-class MAC fast_ce (FIXED codeword-domain decode)
2. Analytical SC reference
3. Per-depth accuracy analysis
4. CheckNode vs BitNode ablation
5. Single-user NPD reference

All results logged to fast_ce_analysis.log
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, polar_encode, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path, design_gmac
from polar.design_mc import design_from_file

LOG_FILE = os.path.join(os.path.dirname(__file__), 'fast_ce_analysis.log')

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')

# Reference SC BLER values (from campaigns)
SC_REF = {
    4:  {'ku': 2,  'kv': 2,  'sc_bler': None},  # will compute
    8:  {'ku': 3,  'kv': 3,  'sc_bler': None},
    16: {'ku': 7,  'kv': 7,  'sc_bler': None},
    32: {'ku': 15, 'kv': 15, 'sc_bler': 0.046},
    64: {'ku': 31, 'kv': 31, 'sc_bler': 0.025},
}


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')


def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════════════
#  NPD-style encoder (no bit reversal, matches tree structure)
# ═══════════════════════════════════════════════════════════════════════

def npd_encode(u):
    """Recursive polar encode matching NPD tree structure. No bit-reversal."""
    N = u.shape[-1]
    if N == 1:
        return u.copy()
    u_odd = u[..., 0::2]
    u_even = u[..., 1::2]
    x_left = npd_encode(u_odd ^ u_even)
    x_right = npd_encode(u_even)
    x = np.empty_like(u)
    x[..., 0::2] = x_left
    x[..., 1::2] = x_right
    return x


def npd_encode_joint(u, v):
    """NPD-encode two users. Returns joint codeword u*2+v in NPD order."""
    xu = npd_encode(u)
    xv = npd_encode(v)
    return xu, xv


# ═══════════════════════════════════════════════════════════════════════
#  4-class MAC fast_ce decoder (FIXED: codeword-domain decode)
# ═══════════════════════════════════════════════════════════════════════

class MAC4ClassDecoder(nn.Module):
    """
    4-class MAC decoder with fast_ce training and FIXED codeword-domain decode.
    Uses one-hot encoding for 4-class bitnode conditioning.
    """
    def __init__(self, d=16, hidden=64, n_layers=2):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d)
        )
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        # BitNode: one-hot 4-class conditioning
        self.bitnode_mlp = _make_mlp(2 * d + 4, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def bitnode(self, e_odd, e_even, uv_left):
        """uv_left: (B, M) integer 0-3."""
        uv_oh = F.one_hot(uv_left.long(), 4).float()  # (B, M, 4)
        inp = torch.cat([e_odd, e_even, uv_oh], dim=-1)
        return self.bitnode_mlp(inp) + e_odd + e_even

    def fast_ce(self, emb, joint_cw):
        """
        Parallel teacher-forced training in codeword domain.
        joint_cw: (B, N) with values 0-3 = xu*2+xv in NPD order.
        """
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []
        depth_accs = []

        logits = self.emb2logits(emb)
        loss = F.cross_entropy(logits.reshape(-1, 4), joint_cw.reshape(-1))
        all_losses.append(loss)
        acc = (logits.argmax(-1) == joint_cw).float().mean().item()
        depth_accs.append(acc)

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

            # XOR per user
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

            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

            e_all = torch.cat(E_chunks, 1)
            j_all = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_all)
            loss = F.cross_entropy(logits.reshape(-1, 4), j_all.reshape(-1))
            all_losses.append(loss)
            acc = (logits.argmax(-1) == j_all).float().mean().item()
            depth_accs.append(acc)

        return torch.stack(all_losses).mean(), depth_accs

    @torch.no_grad()
    def decode_cw_domain(self, emb, frozen_u, frozen_v):
        """
        FIXED SC decode: returns CODEWORD domain at each level.
        frozen_u, frozen_v: sets of 0-indexed positions.
        """
        B, N, _ = emb.shape
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(e_block):
            block_size = e_block.shape[1]
            if block_size == 1:
                logits = self.emb2logits(e_block[:, 0, :])
                idx = leaf_idx[0]
                leaf_idx[0] += 1

                uf = idx in frozen_u
                vf = idx in frozen_v
                if uf and vf:
                    dec = torch.zeros(B, dtype=torch.long)
                elif uf:
                    dec = (logits[:, 1] > logits[:, 0]).long()
                elif vf:
                    dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else:
                    dec = logits.argmax(dim=-1)

                u_hat[:, idx] = dec // 2
                v_hat[:, idx] = dec % 2
                return dec.unsqueeze(1)  # codeword = message at leaf

            e_odd = e_block[:, 0::2, :]
            e_even = e_block[:, 1::2, :]

            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            cw_left = _decode(e_left)

            e_right = self.bitnode(e_odd, e_even, cw_left)
            cw_right = _decode(e_right)

            # Reconstruct parent codeword via butterfly
            u_l = cw_left // 2;  v_l = cw_left % 2
            u_r = cw_right // 2; v_r = cw_right % 2
            cw_odd = (u_l ^ u_r) * 2 + (v_l ^ v_r)
            cw_even = cw_right
            result = torch.zeros(B, block_size, dtype=torch.long)
            result[:, 0::2] = cw_odd
            result[:, 1::2] = cw_even
            return result

        _decode(emb)
        return u_hat, v_hat

    @torch.no_grad()
    def decode_msg_domain(self, emb, frozen_u, frozen_v):
        """
        BUGGY SC decode: returns MESSAGE domain (old broken version).
        This is what the original code did.
        """
        B, N, _ = emb.shape
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(e_block):
            block_size = e_block.shape[1]
            if block_size == 1:
                logits = self.emb2logits(e_block[:, 0, :])
                idx = leaf_idx[0]
                leaf_idx[0] += 1

                uf = idx in frozen_u
                vf = idx in frozen_v
                if uf and vf:
                    dec = torch.zeros(B, dtype=torch.long)
                elif uf:
                    dec = (logits[:, 1] > logits[:, 0]).long()
                elif vf:
                    dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else:
                    dec = logits.argmax(dim=-1)

                u_hat[:, idx] = dec // 2
                v_hat[:, idx] = dec % 2
                return dec.unsqueeze(1)

            e_odd = e_block[:, 0::2, :]
            e_even = e_block[:, 1::2, :]

            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            uv_left = _decode(e_left)

            e_right = self.bitnode(e_odd, e_even, uv_left)
            uv_right = _decode(e_right)

            # BUG: returns message domain, not codeword domain
            return torch.cat([uv_left, uv_right], dim=1)

        _decode(emb)
        return u_hat, v_hat


# ═══════════════════════════════════════════════════════════════════════
#  Single-user NPD decoder
# ═══════════════════════════════════════════════════════════════════════

class SingleUserNPD(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d)
        )
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.emb2llr = _make_mlp(d, hidden, 1, n_layers)

    def bitnode(self, e_odd, e_even, u_left):
        u_sign = (2.0 * u_left.float() - 1.0).unsqueeze(-1).expand_as(e_odd)
        e_signed = e_odd * u_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce(self, emb, x_cw):
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        pred = self.emb2llr(emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(pred, x_cw.float())
        all_losses.append(loss)

        V = [x_cw]
        E = [emb]

        for depth in range(n):
            V_odds, V_evens, E_odds, E_evens = [], [], [], []
            for v, e in zip(V, E):
                V_odds.append(v[:, 0::2]); V_evens.append(v[:, 1::2])
                E_odds.append(e[:, 0::2, :]); E_evens.append(e[:, 1::2, :])
            V_odd = torch.cat(V_odds, 1)
            V_even = torch.cat(V_evens, 1)
            E_odd = torch.cat(E_odds, 1)
            E_even = torch.cat(E_evens, 1)

            v_xor = V_odd ^ V_even
            v_id = V_even

            nc = 2 ** depth
            cs = (N // 2) // nc
            v_xor_c = torch.split(v_xor, cs, 1)
            v_id_c = torch.split(v_id, cs, 1)
            V_new = []
            for vx, vi in zip(v_xor_c, v_id_c):
                V_new.append(vx); V_new.append(vi)
            V_left = torch.cat(V_new[0::2], 1)

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, V_left)

            el_c = torch.split(e_left, cs, 1)
            er_c = torch.split(e_right, cs, 1)
            E_new = []
            for el, er in zip(el_c, er_c):
                E_new.append(el); E_new.append(er)

            e_all = torch.cat(E_new, 1)
            v_all = torch.cat(V_new, 1)
            pred = self.emb2llr(e_all).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(pred, v_all.float())
            all_losses.append(loss)

            V = V_new
            E = E_new

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def decode(self, emb, frozen_set):
        B, N, _ = emb.shape
        u_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(e_block):
            block_size = e_block.shape[1]
            if block_size == 1:
                llr = self.emb2llr(e_block[:, 0, :]).squeeze(-1)
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                if idx in frozen_set:
                    dec = torch.zeros(B, dtype=torch.long)
                else:
                    dec = (llr > 0).long()
                u_hat[:, idx] = dec
                return dec.unsqueeze(1)

            e_odd = e_block[:, 0::2, :]
            e_even = e_block[:, 1::2, :]
            e_top = self.checknode(torch.cat([e_odd, e_even], -1))
            u1_cw = _decode(e_top)
            e_bot = self.bitnode(e_odd, e_even, u1_cw)
            u2_cw = _decode(e_bot)

            x = torch.zeros(B, block_size, dtype=torch.long)
            x[:, 0::2] = u1_cw ^ u2_cw
            x[:, 1::2] = u2_cw
            return x

        _decode(emb)
        return u_hat


# ═══════════════════════════════════════════════════════════════════════
#  Design loading / analytical SC decoder
# ═══════════════════════════════════════════════════════════════════════

def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
        return Au, Av, fu, fv
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv


def compute_npd_frozen(Au, Av, fu, fv, N):
    """Convert 1-indexed frozen dicts to 0-indexed sets in NPD order."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)

    # Standard polar: info positions are 1-indexed
    # NPD order: position i in standard = position br[i] in NPD
    # We need frozen sets in NPD leaf order

    # Build 0-indexed info sets
    Au_0 = {p - 1 for p in Au}
    Av_0 = {p - 1 for p in Av}

    # All positions in NPD leaf order
    # NPD leaf j corresponds to standard position br[j]
    # So if standard position br[j] is frozen, NPD position j is frozen
    fu_npd = set()
    fv_npd = set()
    for j in range(N):
        std_pos = br[j]
        if std_pos not in Au_0:
            fu_npd.add(j)
        if std_pos not in Av_0:
            fv_npd.add(j)

    return fu_npd, fv_npd


def eval_analytical_sc(N, Au, Av, fu, fv, channel, n_cw=2000):
    """Evaluate analytical SC decoder BLER."""
    from polar.decoder import decode_batch
    b = make_path(N, N)  # Class C: all U first, then V
    rng = np.random.default_rng(999)
    errs = 0
    bs = min(64, n_cw)
    for start in range(0, n_cw, bs):
        actual = min(bs, n_cw - start)
        uf = np.zeros((actual, N), dtype=int)
        vf = np.zeros((actual, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, actual)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, actual)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        results = decode_batch(N, zf, b, fu, fv, channel)
        for i in range(actual):
            u_dec, v_dec = results[i]
            ue = any(u_dec[p - 1] != uf[i, p - 1] for p in Au)
            ve = any(v_dec[p - 1] != vf[i, p - 1] for p in Av)
            if ue or ve:
                errs += 1
    return errs / n_cw


# ═══════════════════════════════════════════════════════════════════════
#  Training and evaluation
# ═══════════════════════════════════════════════════════════════════════

def train_mac_fast_ce(N, Au, Av, fu_npd, fv_npd, channel, d=16, hidden=64,
                      n_layers=2, n_iters=20000, batch_size=128, lr=3e-4,
                      seed=42, eval_every=5000, eval_cw=1000):
    """Train 4-class MAC decoder with fast_ce, evaluate with FIXED decode."""
    torch.manual_seed(seed)
    model = MAC4ClassDecoder(d=d, hidden=hidden, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = int(math.log2(N))
    br = bit_reversal_perm(n)

    # Info positions in NPD order (0-indexed)
    Au_npd = set(range(N)) - fu_npd
    Av_npd = set(range(N)) - fv_npd

    best_bler_cw = 1.0
    best_bler_msg = 1.0
    depth_accs_history = []

    model.train()
    for it in range(1, n_iters + 1):
        # Generate data: random messages, encode, channel
        uf = np.zeros((batch_size, N), dtype=int)
        vf = np.zeros((batch_size, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, batch_size)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, batch_size)

        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)

        # NPD-encode for fast_ce targets
        xu_npd = npd_encode(uf)
        xv_npd = npd_encode(vf)
        joint_cw = torch.from_numpy(xu_npd * 2 + xv_npd).long()

        # Channel embeddings in NPD order
        zf_t = torch.from_numpy(zf).float()
        emb = model.z_encoder(zf_t.unsqueeze(-1))
        # NPD order: emb[:, br] maps standard order -> NPD order
        # Actually, npd_encode(u) = standard_encode(u)[br]
        # So z in standard order, emb = z_encoder(z), then emb[:, br] gives NPD order
        # Wait -- the channel output z is in STANDARD order (position i corresponds to
        # the i-th channel use). The NPD tree visits positions in bit-reversed order.
        # So we need emb[:, br] to match NPD tree leaf order.
        emb_npd = emb[:, br]

        loss, depth_accs = model.fast_ce(emb_npd, joint_cw)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % eval_every == 0 or it == n_iters:
            model.eval()
            # Evaluate with FIXED codeword-domain decode
            errs_cw = 0
            errs_msg = 0
            eval_rng = np.random.default_rng(999)
            with torch.no_grad():
                for _ in range(eval_cw):
                    uf1 = np.zeros((1, N), dtype=int)
                    vf1 = np.zeros((1, N), dtype=int)
                    for p in Au: uf1[0, p - 1] = eval_rng.integers(0, 2)
                    for p in Av: vf1[0, p - 1] = eval_rng.integers(0, 2)
                    xf1 = polar_encode_batch(uf1)
                    yf1 = polar_encode_batch(vf1)
                    zf1 = channel.sample_batch(xf1, yf1)
                    zf1_t = torch.from_numpy(zf1).float()
                    emb1 = model.z_encoder(zf1_t.unsqueeze(-1))[:, br]

                    # FIXED decode (codeword domain)
                    u_dec_cw, v_dec_cw = model.decode_cw_domain(emb1, fu_npd, fv_npd)
                    # Check: NPD leaf j -> standard position br[j]
                    ue_cw = any(u_dec_cw[0, j].item() != uf1[0, br[j]] for j in range(N) if j not in fu_npd)
                    ve_cw = any(v_dec_cw[0, j].item() != vf1[0, br[j]] for j in range(N) if j not in fv_npd)
                    if ue_cw or ve_cw:
                        errs_cw += 1

                    # BUGGY decode (message domain) for comparison
                    u_dec_msg, v_dec_msg = model.decode_msg_domain(emb1, fu_npd, fv_npd)
                    ue_msg = any(u_dec_msg[0, j].item() != uf1[0, br[j]] for j in range(N) if j not in fu_npd)
                    ve_msg = any(v_dec_msg[0, j].item() != vf1[0, br[j]] for j in range(N) if j not in fv_npd)
                    if ue_msg or ve_msg:
                        errs_msg += 1

            bler_cw = errs_cw / eval_cw
            bler_msg = errs_msg / eval_cw
            best_bler_cw = min(best_bler_cw, bler_cw)
            best_bler_msg = min(best_bler_msg, bler_msg)
            depth_accs_history.append(depth_accs)

            log(f'  [{it:>6}/{n_iters}] loss={loss.item():.4f} '
                f'BLER_cw={bler_cw:.4f} BLER_msg={bler_msg:.4f} '
                f'depth_acc={[f"{a:.3f}" for a in depth_accs]}')
            model.train()

    return model, best_bler_cw, best_bler_msg, depth_accs_history


def train_single_user_npd(N, frozen_set_std, channel, d=16, hidden=64,
                          n_layers=2, n_iters=10000, batch_size=128, lr=3e-4,
                          seed=42, eval_cw=1000):
    """Train single-user NPD with fast_ce."""
    torch.manual_seed(seed)
    model = SingleUserNPD(d=d, hidden=hidden, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = int(math.log2(N))
    br = bit_reversal_perm(n)

    # Convert frozen set to NPD order
    info_std_0idx = set(range(N)) - {p - 1 for p in frozen_set_std}
    frozen_npd = set()
    for j in range(N):
        if br[j] not in info_std_0idx:
            frozen_npd.add(j)

    # Use N/2 as rate (half the bits are info)
    ku = N - len(frozen_set_std)

    model.train()
    for it in range(1, n_iters + 1):
        uf = np.zeros((batch_size, N), dtype=int)
        for p_std_0 in info_std_0idx:
            uf[:, p_std_0] = rng.integers(0, 2, batch_size)

        xf = npd_encode(uf)
        # Channel: BPSK + AWGN (single user)
        sigma = math.sqrt(SIGMA2)
        zf = (1.0 - 2.0 * xf) + rng.normal(0, sigma, xf.shape)

        zf_t = torch.from_numpy(zf).float()
        emb = model.z_encoder(zf_t.unsqueeze(-1))
        x_cw = torch.from_numpy(xf).long()

        loss = model.fast_ce(emb, x_cw)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Evaluate
    model.eval()
    errs = 0
    eval_rng = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(eval_cw):
            uf1 = np.zeros((1, N), dtype=int)
            for p_std_0 in info_std_0idx:
                uf1[0, p_std_0] = eval_rng.integers(0, 2)
            xf1 = npd_encode(uf1)
            sigma = math.sqrt(SIGMA2)
            zf1 = (1.0 - 2.0 * xf1) + eval_rng.normal(0, sigma, xf1.shape)
            zf1_t = torch.from_numpy(zf1).float()
            emb1 = model.z_encoder(zf1_t.unsqueeze(-1))
            u_dec = model.decode(emb1, frozen_npd)
            if any(u_dec[0, j].item() != uf1[0, br[j]] for j in range(N) if j not in frozen_npd):
                errs += 1
    bler = errs / eval_cw
    return model, bler


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Clear log
    with open(LOG_FILE, 'w') as f:
        f.write(f"Fast CE Analysis - Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

    channel = GaussianMAC(sigma2=SIGMA2)

    # ─── PART 1: Compute SC reference BLER at all N ────────────────────
    log("=" * 70)
    log("PART 1: Analytical SC Reference BLER")
    log("=" * 70)

    sc_blers = {}
    for N in [4, 8, 16, 32]:
        ref = SC_REF[N]
        ku, kv = ref['ku'], ref['kv']
        Au, Av, fu, fv = load_design(N, ku, kv)
        n_cw = 5000 if N <= 16 else 2000
        bler = eval_analytical_sc(N, Au, Av, fu, fv, channel, n_cw)
        sc_blers[N] = bler
        log(f"  N={N:3d}, ku={ku}, kv={kv}: SC BLER = {bler:.4f}")

    # ─── PART 2: Train fast_ce at each N ───────────────────────────────
    log("\n" + "=" * 70)
    log("PART 2: fast_ce Training at Each N (4-class MAC)")
    log("=" * 70)

    D = 16
    HIDDEN = 64
    fastce_results = {}

    for N in [4, 8, 16, 32]:
        log(f"\n--- N={N} ---")
        ref = SC_REF[N]
        ku, kv = ref['ku'], ref['kv']
        Au, Av, fu, fv = load_design(N, ku, kv)
        fu_npd, fv_npd = compute_npd_frozen(Au, Av, fu, fv, N)

        n_iters = max(5000, min(30000, N * 1000))
        eval_every = n_iters // 5
        eval_cw = 2000 if N <= 16 else 1000

        log(f"  Training: d={D}, hidden={HIDDEN}, iters={n_iters}, "
            f"ku={ku}, kv={kv}, |fu_npd|={len(fu_npd)}, |fv_npd|={len(fv_npd)}")

        t0 = time.time()
        model, best_cw, best_msg, daccs = train_mac_fast_ce(
            N, Au, Av, fu_npd, fv_npd, channel,
            d=D, hidden=HIDDEN, n_layers=2,
            n_iters=n_iters, batch_size=128, lr=3e-4,
            eval_every=eval_every, eval_cw=eval_cw
        )
        elapsed = time.time() - t0

        sc_bler = sc_blers[N]
        fastce_results[N] = {
            'best_bler_cw': best_cw,
            'best_bler_msg': best_msg,
            'sc_bler': sc_bler,
            'ratio_cw': best_cw / max(sc_bler, 1e-6),
            'ratio_msg': best_msg / max(sc_bler, 1e-6),
            'time_s': elapsed,
        }

        log(f"  RESULT N={N}: BLER_cw={best_cw:.4f} BLER_msg={best_msg:.4f} "
            f"SC={sc_bler:.4f} ratio_cw={best_cw/max(sc_bler,1e-6):.2f}x "
            f"ratio_msg={best_msg/max(sc_bler,1e-6):.2f}x ({elapsed:.0f}s)")

    # ─── PART 3: Per-depth accuracy analysis ───────────────────────────
    log("\n" + "=" * 70)
    log("PART 3: Per-Depth Accuracy Analysis")
    log("=" * 70)

    # Train fresh models and track depth accuracy over training
    for N in [8, 16, 32]:
        log(f"\n--- N={N}: Depth accuracy during training ---")
        ref = SC_REF[N]
        ku, kv = ref['ku'], ref['kv']
        Au, Av, fu, fv = load_design(N, ku, kv)
        fu_npd, fv_npd = compute_npd_frozen(Au, Av, fu, fv, N)
        n = int(math.log2(N))

        torch.manual_seed(123)
        model = MAC4ClassDecoder(d=D, hidden=HIDDEN, n_layers=2)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        rng = np.random.default_rng(123)
        br = bit_reversal_perm(n)

        n_iters = 15000
        model.train()
        for it in range(1, n_iters + 1):
            uf = np.zeros((128, N), dtype=int)
            vf = np.zeros((128, N), dtype=int)
            for p in Au: uf[:, p - 1] = rng.integers(0, 2, 128)
            for p in Av: vf[:, p - 1] = rng.integers(0, 2, 128)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)

            xu_npd = npd_encode(uf)
            xv_npd = npd_encode(vf)
            joint_cw = torch.from_numpy(xu_npd * 2 + xv_npd).long()
            zf_t = torch.from_numpy(zf).float()
            emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br]

            loss, daccs = model.fast_ce(emb, joint_cw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if it % 5000 == 0 or it == n_iters:
                log(f"  it={it}: depth_acc={[f'{a:.3f}' for a in daccs]} loss={loss.item():.4f}")

    # ─── PART 4: CheckNode vs BitNode ablation ────────────────────────
    log("\n" + "=" * 70)
    log("PART 4: CheckNode vs BitNode Ablation (N=16)")
    log("=" * 70)

    N_abl = 16
    ref = SC_REF[N_abl]
    ku, kv = ref['ku'], ref['kv']
    Au, Av, fu, fv = load_design(N_abl, ku, kv)
    fu_npd, fv_npd = compute_npd_frozen(Au, Av, fu, fv, N_abl)
    n_abl = int(math.log2(N_abl))
    br_abl = bit_reversal_perm(n_abl)

    # Test: Does the gap come from CheckNode or BitNode being wrong?
    # Method: Train with fast_ce, then at decode time:
    # (a) Use analytical CheckNode + learned BitNode
    # (b) Use learned CheckNode + analytical BitNode
    # But we can't easily do analytical operations in the neural framework.
    #
    # Instead: Train TWO models:
    # Model A: freeze CheckNode (random init), train only BitNode + emb2logits
    # Model B: freeze BitNode (random init), train only CheckNode + emb2logits
    # See which one gets closer to SC

    for freeze_what in ['checknode', 'bitnode', 'none']:
        log(f"\n  Ablation: freeze={freeze_what}")
        torch.manual_seed(42)
        model = MAC4ClassDecoder(d=D, hidden=HIDDEN, n_layers=2)

        if freeze_what == 'checknode':
            for p in model.checknode.parameters():
                p.requires_grad = False
        elif freeze_what == 'bitnode':
            for p in model.bitnode_mlp.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"    trainable params: {trainable}")

        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        rng = np.random.default_rng(42)

        model.train()
        for it in range(1, 15001):
            uf = np.zeros((128, N_abl), dtype=int)
            vf = np.zeros((128, N_abl), dtype=int)
            for p in Au: uf[:, p - 1] = rng.integers(0, 2, 128)
            for p in Av: vf[:, p - 1] = rng.integers(0, 2, 128)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            xu_npd = npd_encode(uf)
            xv_npd = npd_encode(vf)
            joint_cw = torch.from_numpy(xu_npd * 2 + xv_npd).long()
            zf_t = torch.from_numpy(zf).float()
            emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br_abl]

            loss, _ = model.fast_ce(emb, joint_cw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Evaluate
        model.eval()
        errs_cw = 0
        eval_rng = np.random.default_rng(999)
        with torch.no_grad():
            for _ in range(2000):
                uf1 = np.zeros((1, N_abl), dtype=int)
                vf1 = np.zeros((1, N_abl), dtype=int)
                for p in Au: uf1[0, p - 1] = eval_rng.integers(0, 2)
                for p in Av: vf1[0, p - 1] = eval_rng.integers(0, 2)
                xf1 = polar_encode_batch(uf1)
                yf1 = polar_encode_batch(vf1)
                zf1 = channel.sample_batch(xf1, yf1)
                zf1_t = torch.from_numpy(zf1).float()
                emb1 = model.z_encoder(zf1_t.unsqueeze(-1))[:, br_abl]
                u_dec, v_dec = model.decode_cw_domain(emb1, fu_npd, fv_npd)
                ue = any(u_dec[0, j].item() != uf1[0, br_abl[j]]
                         for j in range(N_abl) if j not in fu_npd)
                ve = any(v_dec[0, j].item() != vf1[0, br_abl[j]]
                         for j in range(N_abl) if j not in fv_npd)
                if ue or ve:
                    errs_cw += 1
        bler = errs_cw / 2000
        log(f"    BLER_cw={bler:.4f} (SC={sc_blers[N_abl]:.4f}, "
            f"ratio={bler/max(sc_blers[N_abl], 1e-6):.2f}x)")

    # ─── PART 5: Single-user NPD reference ────────────────────────────
    log("\n" + "=" * 70)
    log("PART 5: Single-User NPD fast_ce Reference")
    log("=" * 70)

    for N in [4, 8, 16, 32]:
        ref = SC_REF[N]
        ku = ref['ku']
        # For single user, use u-channel frozen set
        Au, _, fu, _, _, _ = design_gmac(int(math.log2(N)), ku, ku, SIGMA2)
        n_iters = max(5000, min(20000, N * 500))

        log(f"\n  N={N}, ku={ku}, training single-user NPD ({n_iters} iters)...")
        model, bler = train_single_user_npd(
            N, fu, channel, d=D, hidden=HIDDEN, n_layers=2,
            n_iters=n_iters, batch_size=128, lr=3e-4, seed=42, eval_cw=2000
        )
        log(f"  N={N}: single-user NPD BLER = {bler:.4f}")

    # ─── PART 6: Embedding comparison ─────────────────────────────────
    log("\n" + "=" * 70)
    log("PART 6: Embedding Space Analysis (N=8)")
    log("=" * 70)

    N_emb = 8
    ref = SC_REF[N_emb]
    ku, kv = ref['ku'], ref['kv']
    Au, Av, fu, fv = load_design(N_emb, ku, kv)
    fu_npd, fv_npd = compute_npd_frozen(Au, Av, fu, fv, N_emb)
    n_emb = int(math.log2(N_emb))
    br_emb = bit_reversal_perm(n_emb)

    # Generate one fixed test case
    rng_fix = np.random.default_rng(12345)
    uf_fix = np.zeros((1, N_emb), dtype=int)
    vf_fix = np.zeros((1, N_emb), dtype=int)
    for p in Au: uf_fix[0, p - 1] = rng_fix.integers(0, 2)
    for p in Av: vf_fix[0, p - 1] = rng_fix.integers(0, 2)
    xf_fix = polar_encode_batch(uf_fix)
    yf_fix = polar_encode_batch(vf_fix)
    zf_fix = channel.sample_batch(xf_fix, yf_fix)

    # Train two models with different seeds
    log("  Training model A (seed=42) and model B (seed=123)...")
    torch.manual_seed(42)
    modelA = MAC4ClassDecoder(d=D, hidden=HIDDEN, n_layers=2)
    torch.manual_seed(123)
    modelB = MAC4ClassDecoder(d=D, hidden=HIDDEN, n_layers=2)

    for model_tag, model in [('A', modelA), ('B', modelB)]:
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        rng = np.random.default_rng(42 if model_tag == 'A' else 123)
        model.train()
        for it in range(1, 10001):
            uf = np.zeros((128, N_emb), dtype=int)
            vf = np.zeros((128, N_emb), dtype=int)
            for p in Au: uf[:, p - 1] = rng.integers(0, 2, 128)
            for p in Av: vf[:, p - 1] = rng.integers(0, 2, 128)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            xu_npd = npd_encode(uf)
            xv_npd = npd_encode(vf)
            joint_cw = torch.from_numpy(xu_npd * 2 + xv_npd).long()
            zf_t = torch.from_numpy(zf).float()
            emb = model.z_encoder(zf_t.unsqueeze(-1))[:, br_emb]
            loss, _ = model.fast_ce(emb, joint_cw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Compare leaf embeddings from both models on same input
    modelA.eval(); modelB.eval()
    zf_fix_t = torch.from_numpy(zf_fix).float()
    with torch.no_grad():
        embA = modelA.z_encoder(zf_fix_t.unsqueeze(-1))[:, br_emb]
        embB = modelB.z_encoder(zf_fix_t.unsqueeze(-1))[:, br_emb]

    log(f"  Input z: {zf_fix[0, :4].round(3)}...")
    log(f"  EmbA norm per pos: {embA[0].norm(dim=-1).numpy().round(3)}")
    log(f"  EmbB norm per pos: {embB[0].norm(dim=-1).numpy().round(3)}")
    log(f"  Cosine sim (z_encoder output): "
        f"{F.cosine_similarity(embA[0], embB[0], dim=-1).numpy().round(3)}")

    # Now trace through tree and compare
    xu_fix = npd_encode(uf_fix)
    xv_fix = npd_encode(vf_fix)
    joint_fix = torch.from_numpy(xu_fix * 2 + xv_fix).long()

    def trace_tree_embeddings(model, emb, joint_cw):
        """Get embeddings at every tree node."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        node_embs = {'root': emb.clone()}

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

            e_left = model.checknode(torch.cat([E_odd, E_even], -1))
            e_right = model.bitnode(E_odd, E_even, J_left)

            node_embs[f'depth{depth}_left'] = e_left.clone()
            node_embs[f'depth{depth}_right'] = e_right.clone()

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left, cs, 1)
            jr = torch.split(J_even, cs, 1)
            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

        return node_embs

    with torch.no_grad():
        nodes_A = trace_tree_embeddings(modelA, embA, joint_fix)
        nodes_B = trace_tree_embeddings(modelB, embB, joint_fix)

    log("\n  Embedding comparison at each tree level:")
    for key in sorted(nodes_A.keys()):
        eA = nodes_A[key][0]  # (positions, d)
        eB = nodes_B[key][0]
        cos = F.cosine_similarity(eA, eB, dim=-1).mean().item()
        l2_diff = (eA - eB).norm(dim=-1).mean().item()
        log(f"    {key:20s}: cos_sim={cos:.4f}, l2_diff={l2_diff:.4f}")

    # ─── SUMMARY ──────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"\n{'N':>4} | {'SC BLER':>10} | {'fast_ce CW':>12} | {'fast_ce MSG':>12} | {'ratio_cw':>10} | {'ratio_msg':>10}")
    log("-" * 70)
    for N in [4, 8, 16, 32]:
        if N in fastce_results:
            r = fastce_results[N]
            log(f"{N:>4} | {r['sc_bler']:>10.4f} | {r['best_bler_cw']:>12.4f} | {r['best_bler_msg']:>12.4f} | "
                f"{r['ratio_cw']:>10.2f}x | {r['ratio_msg']:>10.2f}x")

    log(f"\nCompleted at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
