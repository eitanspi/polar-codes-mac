#!/usr/bin/env python3
"""
breakthrough_fix.py — Fixed evaluation with correct frozen set mapping.

KEY BUG FOUND: Previous experiments used standard-order frozen sets
{p-1 for p in fu} with NPD-order SC decode. The correct mapping is
{int(br[p-1]) for p in fu}. Also, the comparison of decoded bits
must account for the NPD reordering.

This completely re-tests the joint 4-class fast_ce approach.
"""
import sys, os, math, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file

D = 16
HIDDEN = 64
N_LAYERS = 2
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
BATCH = 128
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'breakthrough_agent.log')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class ConsistentMACDecoder(nn.Module):
    """4-class MAC NPD decoder with consistent fast_ce and SC decode."""
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d)
        )
        self.checknode = _make_mlp(2 * d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2 * d, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def bitnode(self, e_odd, e_even, uv_left):
        """uv_left: (B, M) integer 0-3."""
        u_left = uv_left // 2
        v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([e_odd[:, :, :h] * u_sign,
                              e_odd[:, :, h:] * v_sign], dim=-1)
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce(self, emb, joint_cw):
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 4),
                                          joint_cw.reshape(-1)))

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

            u_o, v_o = J_odd // 2, J_odd % 2
            u_e, v_e = J_even // 2, J_even % 2
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
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4),
                                              j_all.reshape(-1)))

        return torch.stack(all_losses).mean()

    def fast_ce_ss(self, emb, joint_cw, p_sample=0.0):
        """Fast_ce with scheduled sampling at the BitNode level."""
        B, N, d = emb.shape
        n = int(math.log2(N))
        all_losses = []

        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 4),
                                          joint_cw.reshape(-1)))

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

            u_o, v_o = J_odd // 2, J_odd % 2
            u_e, v_e = J_even // 2, J_even % 2
            J_left_true = (u_o ^ u_e) * 2 + (v_o ^ v_e)
            J_right = J_even

            e_left = self.checknode(torch.cat([E_odd, E_even], -1))

            # Scheduled sampling for BitNode conditioning
            if p_sample > 0 and self.training:
                with torch.no_grad():
                    left_logits = self.emb2logits(e_left)
                    J_left_pred = left_logits.argmax(dim=-1)
                mask = (torch.rand(B, J_left_true.shape[1],
                                   device=e_left.device) < p_sample)
                J_left = torch.where(mask, J_left_pred, J_left_true)
            else:
                J_left = J_left_true

            e_right = self.bitnode(E_odd, E_even, J_left)

            nc = 2 ** depth
            cs = (N // 2) // nc
            el = torch.split(e_left, cs, 1)
            er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left_true, cs, 1)  # targets are always true
            jr = torch.split(J_right, cs, 1)

            E_chunks, J_chunks = [], []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]
                J_chunks += [c, dd]

            e_all = torch.cat(E_chunks, 1)
            j_all = torch.cat(J_chunks, 1)
            logits = self.emb2logits(e_all)
            all_losses.append(F.cross_entropy(logits.reshape(-1, 4),
                                              j_all.reshape(-1)))

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def sc_decode(self, emb, frozen_u_npd, frozen_v_npd):
        """SC decode in NPD order.
        frozen_u_npd, frozen_v_npd: sets of 0-indexed NPD-order positions."""
        B, N = emb.shape[0], emb.shape[1]
        u_hat = torch.zeros(B, N, dtype=torch.long)
        v_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(emb_block):
            block_size = emb_block.shape[1]
            if block_size == 1:
                logits = self.emb2logits(emb_block[:, 0, :])
                idx = leaf_idx[0]
                leaf_idx[0] += 1

                u_frozen = idx in frozen_u_npd
                v_frozen = idx in frozen_v_npd

                if u_frozen and v_frozen:
                    dec = torch.zeros(B, dtype=torch.long)
                elif u_frozen:
                    # u=0, pick v
                    dec = (logits[:, 1] > logits[:, 0]).long()
                elif v_frozen:
                    # v=0, pick u
                    dec = (logits[:, 2] > logits[:, 0]).long() * 2
                else:
                    dec = logits.argmax(dim=-1)

                u_hat[:, idx] = dec // 2
                v_hat[:, idx] = dec % 2
                return dec.unsqueeze(1)

            e_odd = emb_block[:, 0::2, :]
            e_even = emb_block[:, 1::2, :]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            uv_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, uv_left)
            uv_right = _decode(e_right)

            u_l, v_l = uv_left // 2, uv_left % 2
            u_r, v_r = uv_right // 2, uv_right % 2
            joint = torch.zeros(B, block_size, dtype=torch.long)
            joint[:, 0::2] = (u_l ^ u_r) * 2 + (v_l ^ v_r)
            joint[:, 1::2] = u_r * 2 + v_r
            return joint

        _decode(emb)
        return u_hat, v_hat


def correct_evaluate_bler(model, channel, N, Au, Av, fu, fv, n_cw=500):
    """CORRECT BLER evaluation with proper frozen set and bit mapping."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_torch = torch.from_numpy(br).long()

    # Correct frozen sets in NPD order
    fu_npd = {int(br[p - 1]) for p in fu}
    fv_npd = {int(br[p - 1]) for p in fv}

    # Info positions in NPD order
    Au_npd = {int(br[p - 1]) for p in Au}
    Av_npd = {int(br[p - 1]) for p in Av}

    errs_u = 0
    errs_v = 0
    errs_block = 0
    rng = np.random.default_rng(999)
    model.eval()
    with torch.no_grad():
        for _ in range(n_cw):
            # Generate data in standard order
            uf = np.zeros((1, N), dtype=int)
            vf = np.zeros((1, N), dtype=int)
            for p in Au: uf[0, p - 1] = rng.integers(0, 2)
            for p in Av: vf[0, p - 1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            # Embed in NPD order
            emb = model.z_encoder(zf.unsqueeze(-1))[:, br_torch]

            # Decode in NPD order
            u_dec_npd, v_dec_npd = model.sc_decode(emb, fu_npd, fv_npd)

            # Ground truth in NPD order
            uf_npd = uf[0, br]  # message bits in NPD order
            vf_npd = vf[0, br]

            # Compare at info positions (in NPD order)
            ue = any(u_dec_npd[0, p].item() != uf_npd[p] for p in Au_npd)
            ve = any(v_dec_npd[0, p].item() != vf_npd[p] for p in Av_npd)
            if ue:
                errs_u += 1
            if ve:
                errs_v += 1
            if ue or ve:
                errs_block += 1

    model.train()
    return errs_block / n_cw, errs_u / n_cw, errs_v / n_cw


def run_joint_fastce_fixed(N=32, n_iters=20000, lr=3e-4, d=D, hidden=HIDDEN):
    """Joint 4-class fast_ce with CORRECT evaluation."""
    log("=" * 70)
    log(f"JOINT FAST_CE (FIXED EVAL) N={N} d={d} hidden={hidden}")
    log("=" * 70)

    n = int(math.log2(N))
    ku = N // 2 - 1; kv = N // 2 - 1
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    model = ConsistentMACDecoder(d=d, hidden=hidden, n_layers=N_LAYERS)
    log(f"Params: {model.count_parameters():,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(42)
    t0 = time.time()

    best_bler = 1.0
    for it in range(1, n_iters + 1):
        uf = np.zeros((BATCH, N), dtype=int)
        vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        joint = torch.from_numpy(xf * 2 + yf).long()[:, br]

        loss = model.fast_ce(emb, joint)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 2000 == 0 or it == n_iters:
            bler, bler_u, bler_v = correct_evaluate_bler(
                model, channel, N, Au, Av, fu, fv, 500)
            if bler < best_bler:
                best_bler = bler
            log(f"  [{it}/{n_iters}] loss={loss.item():.4f} "
                f"BLER={bler:.4f} (u={bler_u:.4f} v={bler_v:.4f}) "
                f"best={best_bler:.4f} ({time.time()-t0:.0f}s)")

    # Final eval
    bler, bler_u, bler_v = correct_evaluate_bler(
        model, channel, N, Au, Av, fu, fv, 2000)
    log(f"FINAL: BLER={bler:.4f} (u={bler_u:.4f} v={bler_v:.4f}) SC=0.046")
    return model, bler


def run_scheduled_sampling_fixed(N=32, n_iters=30000):
    """Scheduled sampling with correct evaluation."""
    log("=" * 70)
    log(f"SCHEDULED SAMPLING (FIXED EVAL) N={N}")
    log("=" * 70)

    n = int(math.log2(N))
    ku = N // 2 - 1; kv = N // 2 - 1
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    model = ConsistentMACDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, n_iters + 1):
        uf = np.zeros((BATCH, N), dtype=int)
        vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        joint = torch.from_numpy(xf * 2 + yf).long()[:, br]

        p_sample = min(0.5, it / n_iters * 0.5)
        loss = model.fast_ce_ss(emb, joint, p_sample)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 5000 == 0 or it == n_iters:
            bler, _, _ = correct_evaluate_bler(
                model, channel, N, Au, Av, fu, fv, 500)
            log(f"  [{it}/{n_iters}] loss={loss.item():.4f} p={p_sample:.3f} "
                f"BLER={bler:.4f} ({time.time()-t0:.0f}s)")

    bler, bler_u, bler_v = correct_evaluate_bler(
        model, channel, N, Au, Av, fu, fv, 2000)
    log(f"SS FINAL: BLER={bler:.4f} (u={bler_u:.4f} v={bler_v:.4f}) SC=0.046")
    return model, bler


def run_binary_decomp_fixed(N=32, n_iters=20000):
    """Binary decomposition with correct evaluation."""
    log("=" * 70)
    log(f"BINARY DECOMPOSITION (FIXED EVAL) N={N}")
    log("=" * 70)

    n = int(math.log2(N))
    ku = N // 2 - 1; kv = N // 2 - 1
    channel = GaussianMAC(sigma2=SIGMA2)
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    br_np = bit_reversal_perm(n)
    br = torch.from_numpy(br_np).long()
    fu_npd = {int(br_np[p-1]) for p in fu}
    fv_npd = {int(br_np[p-1]) for p in fv}
    Au_npd = {int(br_np[p-1]) for p in Au}
    Av_npd = {int(br_np[p-1]) for p in Av}

    class BinaryDecomp(nn.Module):
        def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
            super().__init__()
            self.d = d
            self.z_encoder = nn.Sequential(nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d))
            self.u_checknode = _make_mlp(2*d, hidden, d, n_layers)
            self.u_bitnode_mlp = _make_mlp(2*d, hidden, d, n_layers)
            self.u_emb2llr = _make_mlp(d, hidden, 1, n_layers)
            self.v_checknode = _make_mlp(2*d, hidden, d, n_layers)
            self.v_bitnode_mlp = _make_mlp(2*d, hidden, d, n_layers)
            self.v_emb2llr = _make_mlp(d, hidden, 1, n_layers)
            self.u_cond = nn.Sequential(nn.Linear(d+1, hidden), nn.ELU(), nn.Linear(hidden, d))

        def u_bitnode(self, eo, ee, uh):
            if uh.dim() == 2: uh = uh.unsqueeze(-1)
            us = (2.0*uh.float()-1.0).expand_as(eo)
            es = eo * us
            return self.u_bitnode_mlp(torch.cat([es, ee], -1)) + es + ee

        def v_bitnode(self, eo, ee, vh):
            if vh.dim() == 2: vh = vh.unsqueeze(-1)
            vs = (2.0*vh.float()-1.0).expand_as(eo)
            es = eo * vs
            return self.v_bitnode_mlp(torch.cat([es, ee], -1)) + es + ee

        def fast_ce_u(self, emb, u_cw):
            B, N, d = emb.shape; n = int(math.log2(N))
            losses = []
            pred = self.u_emb2llr(emb).squeeze(-1)
            losses.append(F.binary_cross_entropy_with_logits(pred, u_cw.float()))
            V, E = [u_cw], [emb]
            for depth in range(n):
                VO, VE, EO, EE = [], [], [], []
                for v, e in zip(V, E):
                    VO.append(v[:, 0::2]); VE.append(v[:, 1::2])
                    EO.append(e[:, 0::2, :]); EE.append(e[:, 1::2, :])
                Vo = torch.cat(VO, 1); Ve = torch.cat(VE, 1)
                Eo = torch.cat(EO, 1); Ee = torch.cat(EE, 1)
                vx = Vo ^ Ve; vi = Ve
                el = self.u_checknode(torch.cat([Eo, Ee], -1))
                er = self.u_bitnode(Eo, Ee, vx)
                nc = 2**depth; cs = (N//2)//nc
                els = torch.split(el, cs, 1); ers = torch.split(er, cs, 1)
                vls = torch.split(vx, cs, 1); vrs = torch.split(vi, cs, 1)
                Vn, En = [], []
                for a, b, c, dd in zip(els, ers, vls, vrs):
                    En += [a, b]; Vn += [c, dd]
                ea = torch.cat(En, 1); va = torch.cat(Vn, 1)
                losses.append(F.binary_cross_entropy_with_logits(
                    self.u_emb2llr(ea).squeeze(-1), va.float()))
                V = Vn; E = En
            return torch.stack(losses).mean()

        def fast_ce_v(self, emb, v_cw, u_cw):
            B, N, d = emb.shape; n = int(math.log2(N))
            losses = []
            emb_v = self.u_cond(torch.cat([emb, u_cw.float().unsqueeze(-1)], -1))
            pred = self.v_emb2llr(emb_v).squeeze(-1)
            losses.append(F.binary_cross_entropy_with_logits(pred, v_cw.float()))
            V, E = [v_cw], [emb_v]
            for depth in range(n):
                VO, VE, EO, EE = [], [], [], []
                for v, e in zip(V, E):
                    VO.append(v[:, 0::2]); VE.append(v[:, 1::2])
                    EO.append(e[:, 0::2, :]); EE.append(e[:, 1::2, :])
                Vo = torch.cat(VO, 1); Ve = torch.cat(VE, 1)
                Eo = torch.cat(EO, 1); Ee = torch.cat(EE, 1)
                vx = Vo ^ Ve; vi = Ve
                el = self.v_checknode(torch.cat([Eo, Ee], -1))
                er = self.v_bitnode(Eo, Ee, vx)
                nc = 2**depth; cs = (N//2)//nc
                els = torch.split(el, cs, 1); ers = torch.split(er, cs, 1)
                vls = torch.split(vx, cs, 1); vrs = torch.split(vi, cs, 1)
                Vn, En = [], []
                for a, b, c, dd in zip(els, ers, vls, vrs):
                    En += [a, b]; Vn += [c, dd]
                ea = torch.cat(En, 1); va = torch.cat(Vn, 1)
                losses.append(F.binary_cross_entropy_with_logits(
                    self.v_emb2llr(ea).squeeze(-1), va.float()))
                V = Vn; E = En
            return torch.stack(losses).mean()

        @torch.no_grad()
        def sc_decode(self, emb, fu_npd, fv_npd):
            B, N = emb.shape[0], emb.shape[1]
            u_hat = torch.zeros(B, N, dtype=torch.long)
            v_hat = torch.zeros(B, N, dtype=torch.long)
            ui = [0]
            def _du(e):
                bs = e.shape[1]
                if bs == 1:
                    llr = self.u_emb2llr(e[:, 0, :]).squeeze(-1)
                    idx = ui[0]; ui[0] += 1
                    dec = torch.zeros(B, dtype=torch.long) if idx in fu_npd else (llr > 0).long()
                    u_hat[:, idx] = dec
                    return dec.unsqueeze(1)
                eo = e[:, 0::2, :]; ee = e[:, 1::2, :]
                el = self.u_checknode(torch.cat([eo, ee], -1))
                u1 = _du(el)
                er = self.u_bitnode(eo, ee, u1)
                u2 = _du(er)
                x = torch.zeros(B, bs, dtype=torch.long)
                x[:, 0::2] = u1 ^ u2; x[:, 1::2] = u2
                return x
            u_cw = _du(emb)
            emb_v = self.u_cond(torch.cat([emb, u_cw.float().unsqueeze(-1)], -1))
            vi = [0]
            def _dv(e):
                bs = e.shape[1]
                if bs == 1:
                    llr = self.v_emb2llr(e[:, 0, :]).squeeze(-1)
                    idx = vi[0]; vi[0] += 1
                    dec = torch.zeros(B, dtype=torch.long) if idx in fv_npd else (llr > 0).long()
                    v_hat[:, idx] = dec
                    return dec.unsqueeze(1)
                eo = e[:, 0::2, :]; ee = e[:, 1::2, :]
                el = self.v_checknode(torch.cat([eo, ee], -1))
                v1 = _dv(el)
                er = self.v_bitnode(eo, ee, v1)
                v2 = _dv(er)
                x = torch.zeros(B, bs, dtype=torch.long)
                x[:, 0::2] = v1 ^ v2; x[:, 1::2] = v2
                return x
            _dv(emb_v)
            return u_hat, v_hat

    model = BinaryDecomp(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Params: {params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, n_iters + 1):
        uf = np.zeros((BATCH, N), dtype=int)
        vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
        emb = model.z_encoder(zf.unsqueeze(-1))[:, br]
        u_cw = torch.from_numpy(xf).long()[:, br]
        v_cw = torch.from_numpy(yf).long()[:, br]

        loss_u = model.fast_ce_u(emb, u_cw)
        loss_v = model.fast_ce_v(emb, v_cw, u_cw)
        loss = loss_u + loss_v
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 2000 == 0 or it == n_iters:
            # Correct eval
            errs = 0; nt = 500
            model.eval()
            rng_t = np.random.default_rng(999)
            with torch.no_grad():
                for _ in range(nt):
                    uf_t = np.zeros((1, N), dtype=int)
                    vf_t = np.zeros((1, N), dtype=int)
                    for p in Au: uf_t[0, p-1] = rng_t.integers(0, 2)
                    for p in Av: vf_t[0, p-1] = rng_t.integers(0, 2)
                    xf_t = polar_encode_batch(uf_t)
                    yf_t = polar_encode_batch(vf_t)
                    zf_t2 = torch.from_numpy(channel.sample_batch(xf_t, yf_t)).float()
                    emb_t = model.z_encoder(zf_t2.unsqueeze(-1))[:, br]
                    u_dec, v_dec = model.sc_decode(emb_t, fu_npd, fv_npd)
                    # Compare in NPD order
                    uf_npd = uf_t[0, br_np]
                    vf_npd = vf_t[0, br_np]
                    ue = any(u_dec[0, p].item() != uf_npd[p] for p in Au_npd)
                    ve = any(v_dec[0, p].item() != vf_npd[p] for p in Av_npd)
                    if ue or ve: errs += 1
            model.train()
            bler = errs / nt
            log(f"  [{it}/{n_iters}] loss_u={loss_u.item():.4f} loss_v={loss_v.item():.4f} "
                f"BLER={bler:.4f} ({time.time()-t0:.0f}s)")

    # Final
    errs = 0; nt = 2000
    model.eval()
    rng_t = np.random.default_rng(999)
    with torch.no_grad():
        for _ in range(nt):
            uf_t = np.zeros((1, N), dtype=int)
            vf_t = np.zeros((1, N), dtype=int)
            for p in Au: uf_t[0, p-1] = rng_t.integers(0, 2)
            for p in Av: vf_t[0, p-1] = rng_t.integers(0, 2)
            xf_t = polar_encode_batch(uf_t)
            yf_t = polar_encode_batch(vf_t)
            zf_t2 = torch.from_numpy(channel.sample_batch(xf_t, yf_t)).float()
            emb_t = model.z_encoder(zf_t2.unsqueeze(-1))[:, br]
            u_dec, v_dec = model.sc_decode(emb_t, fu_npd, fv_npd)
            uf_npd_t = uf_t[0, br_np]
            vf_npd_t = vf_t[0, br_np]
            ue = any(u_dec[0, p].item() != uf_npd_t[p] for p in Au_npd)
            ve = any(v_dec[0, p].item() != vf_npd_t[p] for p in Av_npd)
            if ue or ve: errs += 1
    bler = errs / nt
    log(f"BINARY DECOMP FINAL: BLER={bler:.4f} (SC=0.046)")
    return model, bler


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        return design_from_file(mc_path, n, ku, kv)
    from polar.design import design_gmac
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv, None, None, None


if __name__ == '__main__':
    log("\n" + "=" * 70)
    log("BREAKTHROUGH FIX — CORRECT EVALUATION")
    log("=" * 70)
    log("KEY FIX: frozen sets now mapped to NPD order via bit-reversal")

    results = {}

    # 1. Joint 4-class fast_ce with correct eval
    try:
        model_j, bler_j = run_joint_fastce_fixed(N=32, n_iters=20000)
        results['joint_fastce_N32'] = bler_j
        if bler_j < 0.10:
            # Save and test at N=64
            torch.save(model_j.state_dict(),
                       os.path.join(os.path.dirname(__file__), 'saved_models', 'joint_fastce_N32.pt'))
            log("Saved model. Testing at N=64...")
            model_j64, bler_j64 = run_joint_fastce_fixed(N=64, n_iters=20000)
            results['joint_fastce_N64'] = bler_j64
            if bler_j64 < 0.10:
                model_j128, bler_j128 = run_joint_fastce_fixed(N=128, n_iters=20000)
                results['joint_fastce_N128'] = bler_j128
    except Exception as e:
        log(f"Joint fast_ce FAILED: {e}"); traceback.print_exc()

    # 2. Scheduled sampling
    try:
        model_ss, bler_ss = run_scheduled_sampling_fixed(N=32, n_iters=30000)
        results['scheduled_sampling_N32'] = bler_ss
        if bler_ss < 0.10:
            model_ss64, bler_ss64 = run_scheduled_sampling_fixed(N=64, n_iters=30000)
            results['scheduled_sampling_N64'] = bler_ss64
    except Exception as e:
        log(f"Scheduled sampling FAILED: {e}"); traceback.print_exc()

    # 3. Binary decomposition
    try:
        model_bd, bler_bd = run_binary_decomp_fixed(N=32, n_iters=20000)
        results['binary_decomp_N32'] = bler_bd
        if bler_bd < 0.10:
            model_bd64, bler_bd64 = run_binary_decomp_fixed(N=64, n_iters=20000)
            results['binary_decomp_N64'] = bler_bd64
    except Exception as e:
        log(f"Binary decomp FAILED: {e}"); traceback.print_exc()

    # 4. Larger model if needed
    best = min(results.values()) if results else 1.0
    if best > 0.10:
        log("All approaches > 0.10 BLER. Trying larger model d=32...")
        try:
            model_lg, bler_lg = run_joint_fastce_fixed(N=32, n_iters=30000, d=32, hidden=128)
            results['joint_fastce_large_N32'] = bler_lg
        except Exception as e:
            log(f"Large model FAILED: {e}"); traceback.print_exc()

    log("\n" + "=" * 70)
    log("ALL RESULTS:")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        log(f"  {k}: BLER={v:.4f}")
    log(f"  SC baseline: BLER=0.046")
    log("=" * 70)
