#!/usr/bin/env python3
"""
Corrected fast_ce analysis with proper Class B paths.

The v1 script had a BUG: SC evaluation used path_i=N (Class C) with
Class B MC designs. This v2 uses the correct path_i from the design file.

Also trains with more iterations and proper eval.
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, polar_encode, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path, design_gmac
from polar.design_mc import design_from_file
from polar.decoder import decode_batch

LOG_FILE = os.path.join(os.path.dirname(__file__), 'fast_ce_analysis.log')
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')


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


def npd_encode(u):
    N = u.shape[-1]
    if N == 1: return u.copy()
    u_odd = u[..., 0::2]; u_even = u[..., 1::2]
    x_left = npd_encode(u_odd ^ u_even)
    x_right = npd_encode(u_even)
    x = np.empty_like(u)
    x[..., 0::2] = x_left; x[..., 1::2] = x_right
    return x


class MAC4ClassDecoder(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2 * d + 4, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def bitnode(self, e_odd, e_even, uv_left):
        uv_oh = F.one_hot(uv_left.long(), 4).float()
        inp = torch.cat([e_odd, e_even, uv_oh], dim=-1)
        return self.bitnode_mlp(inp) + e_odd + e_even

    def fast_ce(self, emb, joint_cw):
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []
        depth_accs = []

        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 4), joint_cw.reshape(-1)))
        depth_accs.append((logits.argmax(-1) == joint_cw).float().mean().item())

        E_chunks = [emb]; J_chunks = [joint_cw]

        for depth in range(n):
            E_odds, E_evens, J_odds, J_evens = [], [], [], []
            for e, j in zip(E_chunks, J_chunks):
                M = e.shape[1]
                E_odds.append(e.reshape(B, M//2, 2, d)[:,:,0,:])
                E_evens.append(e.reshape(B, M//2, 2, d)[:,:,1,:])
                J_odds.append(j.reshape(B, M//2, 2)[:,:,0])
                J_evens.append(j.reshape(B, M//2, 2)[:,:,1])

            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)
            J_odd = torch.cat(J_odds, 1); J_even = torch.cat(J_evens, 1)

            u_o = J_odd // 2; v_o = J_odd % 2
            u_e = J_even // 2; v_e = J_even % 2
            J_left = (u_o ^ u_e) * 2 + (v_o ^ v_e)
            J_right = J_even

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)

            nc = 2 ** depth; cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1); er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left, cs, 1); jr = torch.split(J_right, cs, 1)

            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]; J_chunks += [c, dd]

            e_all = torch.cat(E_chunks, 1); j_all = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_all)
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4), j_all.reshape(-1)))
            depth_accs.append((logits.argmax(-1) == j_all).float().mean().item())

        return torch.stack(all_losses).mean(), depth_accs

    @torch.no_grad()
    def decode_cw_domain(self, emb, frozen_u, frozen_v):
        B, N, _ = emb.shape
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(e_block):
            bs = e_block.shape[1]
            if bs == 1:
                logits = self.emb2logits(e_block[:, 0, :])
                idx = leaf_idx[0]; leaf_idx[0] += 1
                uf = idx in frozen_u; vf = idx in frozen_v
                if uf and vf: dec = torch.zeros(B, dtype=torch.long)
                elif uf: dec = (logits[:, 1] > logits[:, 0]).long()
                elif vf: dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else: dec = logits.argmax(dim=-1)
                u_hat[:, idx] = dec // 2; v_hat[:, idx] = dec % 2
                return dec.unsqueeze(1)

            e_odd = e_block[:, 0::2, :]; e_even = e_block[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            cw_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, cw_left)
            cw_right = _decode(e_right)

            u_l = cw_left // 2;  v_l = cw_left % 2
            u_r = cw_right // 2; v_r = cw_right % 2
            result = torch.zeros(B, bs, dtype=torch.long)
            result[:, 0::2] = (u_l ^ u_r) * 2 + (v_l ^ v_r)
            result[:, 1::2] = cw_right
            return result

        _decode(emb)
        return u_hat, v_hat


class SingleUserNPD(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.emb2llr = _make_mlp(d, hidden, 1, n_layers)

    def bitnode(self, e_odd, e_even, u_left):
        u_sign = (2.0 * u_left.float() - 1.0).unsqueeze(-1).expand_as(e_odd)
        e_signed = e_odd * u_sign
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce(self, emb, x_cw):
        B, N, d = emb.shape; n = int(math.log2(N))
        all_losses = []
        pred = self.emb2llr(emb).squeeze(-1)
        all_losses.append(F.binary_cross_entropy_with_logits(pred, x_cw.float()))
        V = [x_cw]; E = [emb]
        for depth in range(n):
            V_odds, V_evens, E_odds, E_evens = [], [], [], []
            for v, e in zip(V, E):
                V_odds.append(v[:, 0::2]); V_evens.append(v[:, 1::2])
                E_odds.append(e[:, 0::2, :]); E_evens.append(e[:, 1::2, :])
            V_odd = torch.cat(V_odds, 1); V_even = torch.cat(V_evens, 1)
            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)
            v_xor = V_odd ^ V_even; v_id = V_even
            nc = 2 ** depth; cs = (N // 2) // nc
            v_xor_c = torch.split(v_xor, cs, 1); v_id_c = torch.split(v_id, cs, 1)
            V_new = []
            for vx, vi in zip(v_xor_c, v_id_c): V_new.append(vx); V_new.append(vi)
            V_left = torch.cat(V_new[0::2], 1)
            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, V_left)
            el_c = torch.split(e_left, cs, 1); er_c = torch.split(e_right, cs, 1)
            E_new = []
            for el, er in zip(el_c, er_c): E_new.append(el); E_new.append(er)
            e_all = torch.cat(E_new, 1); v_all = torch.cat(V_new, 1)
            pred = self.emb2llr(e_all).squeeze(-1)
            all_losses.append(F.binary_cross_entropy_with_logits(pred, v_all.float()))
            V = V_new; E = E_new
        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def decode(self, emb, frozen_set):
        B, N, _ = emb.shape
        u_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]
        def _decode(e_block):
            bs = e_block.shape[1]
            if bs == 1:
                llr = self.emb2llr(e_block[:, 0, :]).squeeze(-1)
                idx = leaf_idx[0]; leaf_idx[0] += 1
                dec = torch.zeros(B, dtype=torch.long) if idx in frozen_set else (llr > 0).long()
                u_hat[:, idx] = dec
                return dec.unsqueeze(1)
            e_odd = e_block[:, 0::2, :]; e_even = e_block[:, 1::2, :]
            e_top = self.checknode(torch.cat([e_odd, e_even], -1))
            u1_cw = _decode(e_top)
            e_bot = self.bitnode(e_odd, e_even, u1_cw)
            u2_cw = _decode(e_bot)
            x = torch.zeros(B, bs, dtype=torch.long)
            x[:, 0::2] = u1_cw ^ u2_cw; x[:, 1::2] = u2_cw
            return x
        _decode(emb)
        return u_hat


def load_design_with_path(N, ku, kv):
    n = int(math.log2(N))
    mc = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc):
        Au, Av, fu, fv, _, _, path_i = design_from_file(mc, n, ku, kv)
        return Au, Av, fu, fv, path_i
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv, N  # Class C


def compute_npd_frozen(Au, Av, N):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    Au_0 = {p - 1 for p in Au}; Av_0 = {p - 1 for p in Av}
    fu_npd = set(); fv_npd = set()
    for j in range(N):
        if br[j] not in Au_0: fu_npd.add(j)
        if br[j] not in Av_0: fv_npd.add(j)
    return fu_npd, fv_npd


def eval_sc(N, Au, Av, fu, fv, path_i, channel, n_cw=2000):
    b = make_path(N, path_i)
    rng = np.random.default_rng(999)
    errs = 0; bs = 64
    for start in range(0, n_cw, bs):
        actual = min(bs, n_cw - start)
        uf = np.zeros((actual, N), dtype=int)
        vf = np.zeros((actual, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, actual)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, actual)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        results = decode_batch(N, zf, b, fu, fv, channel)
        for i in range(actual):
            u_dec, v_dec = results[i]
            ue = any(u_dec[p-1] != uf[i, p-1] for p in Au)
            ve = any(v_dec[p-1] != vf[i, p-1] for p in Av)
            if ue or ve: errs += 1
    return errs / n_cw


def eval_nn(model, N, Au, Av, fu_npd, fv_npd, channel, n_cw=2000):
    br = bit_reversal_perm(int(math.log2(N)))
    model.eval()
    errs = 0; rng = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(n_cw):
            uf = np.zeros((1, N), dtype=int)
            vf = np.zeros((1, N), dtype=int)
            for p in Au: uf[0, p - 1] = rng.integers(0, 2)
            for p in Av: vf[0, p - 1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            emb = model.z_encoder(torch.from_numpy(zf).float().unsqueeze(-1))[:, br]
            u_dec, v_dec = model.decode_cw_domain(emb, fu_npd, fv_npd)
            ue = any(u_dec[0, j].item() != uf[0, br[j]] for j in range(N) if j not in fu_npd)
            ve = any(v_dec[0, j].item() != vf[0, br[j]] for j in range(N) if j not in fv_npd)
            if ue or ve: errs += 1
    return errs / n_cw


def train_and_eval_mac(N, Au, Av, fu_npd, fv_npd, channel, d=16, hidden=64,
                       n_iters=20000, batch_size=128, lr=3e-4, seed=42):
    torch.manual_seed(seed)
    model = MAC4ClassDecoder(d=d, hidden=hidden, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    br = bit_reversal_perm(int(math.log2(N)))

    model.train()
    for it in range(1, n_iters + 1):
        uf = np.zeros((batch_size, N), dtype=int)
        vf = np.zeros((batch_size, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, batch_size)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, batch_size)

        xu_npd = npd_encode(uf); xv_npd = npd_encode(vf)
        joint_cw = torch.from_numpy(xu_npd * 2 + xv_npd).long()

        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        emb = model.z_encoder(torch.from_numpy(zf).float().unsqueeze(-1))[:, br]

        loss, daccs = model.fast_ce(emb, joint_cw)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0:
            bler = eval_nn(model, N, Au, Av, fu_npd, fv_npd, channel, n_cw=500)
            log(f'    [{it}/{n_iters}] loss={loss.item():.4f} BLER={bler:.4f} '
                f'daccs={[f"{a:.3f}" for a in daccs]}')
            model.train()

    final_bler = eval_nn(model, N, Au, Av, fu_npd, fv_npd, channel, n_cw=2000)
    return model, final_bler


def train_and_eval_single_user(N, fu_std, channel, d=16, hidden=64,
                               n_iters=15000, batch_size=128, seed=42):
    torch.manual_seed(seed)
    model = SingleUserNPD(d=d, hidden=hidden, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    br = bit_reversal_perm(int(math.log2(N)))

    info_std = set(range(N)) - {p - 1 for p in fu_std}
    frozen_npd = set()
    for j in range(N):
        if br[j] not in info_std: frozen_npd.add(j)

    model.train()
    sigma = math.sqrt(SIGMA2)
    for it in range(1, n_iters + 1):
        uf = np.zeros((batch_size, N), dtype=int)
        for p0 in info_std: uf[:, p0] = rng.integers(0, 2, batch_size)
        xf = npd_encode(uf)
        zf = (1.0 - 2.0 * xf) + rng.normal(0, sigma, xf.shape)
        emb = model.z_encoder(torch.from_numpy(zf).float().unsqueeze(-1))
        loss = model.fast_ce(emb, torch.from_numpy(xf).long())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Eval
    model.eval(); errs = 0; eval_rng = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(2000):
            uf = np.zeros((1, N), dtype=int)
            for p0 in info_std: uf[0, p0] = eval_rng.integers(0, 2)
            xf = npd_encode(uf)
            zf = (1.0 - 2.0 * xf) + eval_rng.normal(0, sigma, xf.shape)
            emb = model.z_encoder(torch.from_numpy(zf).float().unsqueeze(-1))
            u_dec = model.decode(emb, frozen_npd)
            if any(u_dec[0, j].item() != uf[0, br[j]] for j in range(N) if j not in frozen_npd):
                errs += 1
    return errs / 2000


def main():
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n\n{'='*70}\n")
        f.write(f"CORRECTED ANALYSIS v2 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*70}\n\n")

    channel = GaussianMAC(sigma2=SIGMA2)
    D = 16; HIDDEN = 64

    # ─── Correct SC BLER ──────────────────────────────────────────────
    log("PART A: Corrected SC BLER (with proper Class B paths)")
    sc_blers = {}
    for N in [4, 8, 16, 32]:
        ku = {4: 2, 8: 3, 16: 7, 32: 15}[N]
        Au, Av, fu, fv, path_i = load_design_with_path(N, ku, ku)
        bler = eval_sc(N, Au, Av, fu, fv, path_i, channel, 5000)
        sc_blers[N] = bler
        log(f"  N={N:3d}: SC BLER = {bler:.4f} (path_i={path_i}, ku={ku})")

    # ─── fast_ce MAC at each N ────────────────────────────────────────
    log("\nPART B: 4-class MAC fast_ce (codeword-domain decode)")
    nn_blers = {}
    for N in [4, 8, 16, 32]:
        ku = {4: 2, 8: 3, 16: 7, 32: 15}[N]
        Au, Av, fu, fv, path_i = load_design_with_path(N, ku, ku)
        fu_npd, fv_npd = compute_npd_frozen(Au, Av, N)
        n_iters = {4: 10000, 8: 15000, 16: 25000, 32: 40000}[N]

        log(f"\n  N={N}: training {n_iters} iters...")
        t0 = time.time()
        model, bler = train_and_eval_mac(N, Au, Av, fu_npd, fv_npd, channel,
                                         d=D, hidden=HIDDEN, n_iters=n_iters)
        elapsed = time.time() - t0
        nn_blers[N] = bler
        ratio = bler / max(sc_blers[N], 1e-6)
        log(f"  N={N}: NN BLER = {bler:.4f}, SC BLER = {sc_blers[N]:.4f}, "
            f"ratio = {ratio:.2f}x, time = {elapsed:.0f}s")

    # ─── CheckNode vs BitNode ablation at N=16 ───────────────────────
    log("\nPART C: CheckNode vs BitNode Ablation (N=16)")
    N_abl = 16; ku_abl = 7
    Au, Av, fu, fv, path_i = load_design_with_path(N_abl, ku_abl, ku_abl)
    fu_npd, fv_npd = compute_npd_frozen(Au, Av, N_abl)

    for freeze_what in ['none', 'checknode', 'bitnode']:
        torch.manual_seed(42)
        model = MAC4ClassDecoder(d=D, hidden=HIDDEN, n_layers=2)
        if freeze_what == 'checknode':
            for p in model.checknode.parameters(): p.requires_grad = False
        elif freeze_what == 'bitnode':
            for p in model.bitnode_mlp.parameters(): p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  freeze={freeze_what}: {trainable} trainable params")

        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        rng = np.random.default_rng(42)
        br = bit_reversal_perm(int(math.log2(N_abl)))

        model.train()
        for it in range(1, 20001):
            uf = np.zeros((128, N_abl), dtype=int)
            vf = np.zeros((128, N_abl), dtype=int)
            for p in Au: uf[:, p - 1] = rng.integers(0, 2, 128)
            for p in Av: vf[:, p - 1] = rng.integers(0, 2, 128)
            xu = npd_encode(uf); xv = npd_encode(vf)
            joint = torch.from_numpy(xu * 2 + xv).long()
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            emb = model.z_encoder(torch.from_numpy(zf).float().unsqueeze(-1))[:, br]
            loss, _ = model.fast_ce(emb, joint)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        bler = eval_nn(model, N_abl, Au, Av, fu_npd, fv_npd, channel, 2000)
        log(f"    BLER={bler:.4f} (SC={sc_blers[N_abl]:.4f}, ratio={bler/max(sc_blers[N_abl],1e-6):.2f}x)")

    # ─── Single-user reference ────────────────────────────────────────
    log("\nPART D: Single-User NPD fast_ce Reference")
    for N in [4, 8, 16, 32]:
        ku = {4: 2, 8: 3, 16: 7, 32: 15}[N]
        _, _, fu, _, _, _ = design_gmac(int(math.log2(N)), ku, ku, SIGMA2)
        n_iters = {4: 10000, 8: 15000, 16: 20000, 32: 25000}[N]
        bler = train_and_eval_single_user(N, fu, channel, d=D, hidden=HIDDEN,
                                          n_iters=n_iters)
        log(f"  N={N}: single-user NPD BLER = {bler:.4f}")

    # ─── Summary ──────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("FINAL SUMMARY (CORRECTED)")
    log("=" * 70)
    log(f"\n{'N':>4} | {'SC BLER':>10} | {'NN BLER':>10} | {'ratio':>8} | {'Gap?':>10}")
    log("-" * 55)
    for N in [4, 8, 16, 32]:
        sc = sc_blers[N]; nn = nn_blers[N]
        ratio = nn / max(sc, 1e-6)
        gap = "BETTER" if ratio < 1.0 else f"{ratio:.1f}x WORSE"
        log(f"{N:>4} | {sc:>10.4f} | {nn:>10.4f} | {ratio:>8.2f}x | {gap:>10}")

    log(f"\nCompleted at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
