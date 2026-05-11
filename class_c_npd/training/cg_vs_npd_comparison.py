#!/usr/bin/env python3
"""
Head-to-head comparison: CG decoder vs chained NPD on Class C GMAC.

Track 1: CG decoder with NPD-style training (rate 1, MI-guided design)
Track 2: NPD vs CG MI convergence curves

Both run on same channel (GMAC SNR=6dB, Class C path).
Logs MI metrics every eval step for plotting.
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.decoder import decode_batch, _sc_decode_from_llr
from polar.design import make_path
from polar.design_mc import design_from_file

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.eval.chain_eval import wilson_ci

# ─── Config ──────────────────────────────────────────────────────────────────
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def mixture_llr(z, sigma2):
    s2 = sigma2
    def lN(m): return -0.5*(z-m)**2/s2
    return np.logaddexp(lN(2.0), lN(0.0)) - np.logaddexp(lN(0.0), lN(-2.0))

# ─── CG Decoder (simplified for Class C Stage 1) ────────────────────────────
# For Class C, the CG decoder runs on the marginal channel for U.
# We build a simplified CG-style architecture that uses CalcLeft/CalcRight
# with smooth learned re-embedding (logits2emb) instead of sign-flip.

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class CGStyleDecoder(nn.Module):
    """
    CG-style decoder for single-user Class C Stage 1.

    Differences from NPD:
    - CalcLeft/CalcRight take 3d input (parent_first, parent_second, sibling)
      instead of 2d (e_odd, e_even)
    - Uses smooth logits2emb (MLP) for decision re-embedding instead of sign-flip
    - Has a gated residual CalcParent for bottom-up flow
    - Uses emb2logits → 2-class logits (not scalar LLR)

    For fast_ce training, we adapt to the even/odd split structure.
    CalcLeft ≈ CheckNode, CalcRight ≈ BitNode but with SMOOTH re-embedding.
    """
    def __init__(self, d=16, hidden=64, n_layers=2, z_dim=1):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(z_dim, hidden), nn.ELU(), nn.Linear(hidden, d))

        # CalcLeft: like CheckNode but takes 3d input
        self.calc_left = _make_mlp(3*d, hidden, d, n_layers)

        # CalcRight: like BitNode but with SMOOTH re-embedding (no sign-flip)
        self.calc_right_mlp = _make_mlp(3*d, hidden, d, n_layers)

        # Decision: d → 2-class logits
        self.emb2logits = _make_mlp(d, hidden, 2, n_layers)

        # Smooth re-embedding: 2-class log-prob → d (learned, NOT sign-flip)
        self.logits2emb = _make_mlp(2, hidden, d, n_layers)

        # No-info embedding (for positions without side info)
        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_channel(self, z_features):
        if z_features.dim() == 2:
            z_features = z_features.unsqueeze(-1)
        return self.z_encoder(z_features)

    def _make_leaf_emb(self, u_hard, B):
        """Smooth re-embedding of a binary decision."""
        # Convert bit to soft 2-class log-prob
        log_prob = torch.zeros(B, 1, 2)
        log_prob[:, :, 0] = torch.where(u_hard.unsqueeze(-1) == 0, 0.0, -30.0).squeeze(-1)
        log_prob[:, :, 1] = torch.where(u_hard.unsqueeze(-1) == 1, 0.0, -30.0).squeeze(-1)
        return self.logits2emb(log_prob)

    def calc_right(self, e_odd, e_even, u_left):
        """CalcRight with smooth re-embedding (CG style)."""
        # Re-embed the left-child decision smoothly
        B, M = u_left.shape[0], u_left.shape[1]
        # Create soft one-hot from hard bits
        soft = torch.zeros(B, M, 2, device=e_odd.device)
        soft.scatter_(2, u_left.unsqueeze(-1).long(), 1.0)
        left_emb = self.logits2emb(soft)  # (B, M, d) — smooth embedding

        inp = torch.cat([e_odd, e_even, left_emb], dim=-1)
        return self.calc_right_mlp(inp) + e_even  # residual on e_even (not sign-flip!)

    def calc_left(self, e_odd, e_even):
        """CalcLeft (like CheckNode but with no_info_emb as third input)."""
        B, M, d = e_odd.shape
        no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0).expand(B, M, d)
        inp = torch.cat([e_odd, e_even, no_info], dim=-1)
        return self.calc_left(inp)  # Oops, this would recurse. Fix:

    def fast_ce(self, emb, x_cw):
        """Fast CE with CG-style ops."""
        if x_cw.dim() == 3: x_cw = x_cw.squeeze(-1)
        B, N, d = emb.shape; n = int(math.log2(N))
        all_losses = []

        # Depth 0
        logits = self.emb2logits(emb)  # (B, N, 2)
        target = x_cw.long()
        all_losses.append(F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1)))

        V = [x_cw]; E = [emb]
        for depth in range(n):
            V_odds, V_evens, E_odds, E_evens = [], [], [], []
            for v, e in zip(V, E):
                V_odds.append(v[:, 0::2]); V_evens.append(v[:, 1::2])
                E_odds.append(e[:, 0::2, :]); E_evens.append(e[:, 1::2, :])
            V_odd = torch.cat(V_odds, 1); V_even = torch.cat(V_evens, 1)
            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)

            v_top = V_odd ^ V_even; v_bot = V_even
            nc = 2**depth; cs = (N//2)//nc
            vtc = torch.split(v_top, cs, 1); vbc = torch.split(v_bot, cs, 1)
            Vn = []
            for a, b in zip(vtc, vbc): Vn += [a, b]
            Vl = torch.cat(Vn[0::2], 1)

            # CG-style ops
            no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0).expand_as(E_odd)
            e_left = _make_mlp.__self__ if False else None  # placeholder
            # Use actual calc_left and calc_right
            inp_left = torch.cat([E_odd, E_even, no_info], dim=-1)
            e_top = self.calc_left_mlp(inp_left)  # CalcLeft
            e_bot = self.calc_right(E_odd, E_even, Vl)  # CalcRight with smooth re-emb

            etc = torch.split(e_top, cs, 1); ebc = torch.split(e_bot, cs, 1)
            En = []
            for a, b in zip(etc, ebc): En += [a, b]

            e_all = torch.cat(En, 1); v_all = torch.cat(Vn, 1)
            logits = self.emb2logits(e_all)
            all_losses.append(F.cross_entropy(logits.reshape(-1, 2), v_all.reshape(-1).long()))
            V = Vn; E = En

        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def decode(self, emb, frozen_set):
        from polar.encoder import bit_reversal_perm
        B, N, _ = emb.shape; n = int(math.log2(N))
        br = bit_reversal_perm(n)
        u_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]

        def _decode(e_block):
            bs = e_block.shape[1]
            if bs == 1:
                logits = self.emb2logits(e_block[:, 0, :])  # (B, 2)
                idx = leaf_idx[0]; leaf_idx[0] += 1
                nat_idx = int(br[idx])
                if nat_idx in frozen_set:
                    dec = torch.zeros(B, dtype=torch.long)
                else:
                    dec = logits.argmax(dim=-1)
                u_hat[:, nat_idx] = dec
                return dec.unsqueeze(1)

            e_odd = e_block[:, 0::2, :]; e_even = e_block[:, 1::2, :]
            no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0).expand_as(e_odd)
            inp = torch.cat([e_odd, e_even, no_info], dim=-1)
            e_top = self.calc_left_mlp(inp)
            cw_top = _decode(e_top)
            e_bot = self.calc_right(e_odd, e_even, cw_top)
            cw_bot = _decode(e_bot)
            cw = torch.zeros(B, bs, dtype=torch.long)
            cw[:, 0::2] = cw_top ^ cw_bot; cw[:, 1::2] = cw_bot
            return cw

        _decode(emb)
        return u_hat


# Fix the CG class — calc_left was recursive. Rename the MLP.
class CGStyleDecoderFixed(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2, z_dim=1):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(z_dim, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.calc_left_mlp = _make_mlp(3*d, hidden, d, n_layers)
        self.calc_right_mlp = _make_mlp(3*d, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 2, n_layers)
        self.logits2emb = _make_mlp(2, hidden, d, n_layers)
        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_channel(self, z):
        if z.dim() == 2: z = z.unsqueeze(-1)
        return self.z_encoder(z)

    def fast_ce(self, emb, x_cw):
        if x_cw.dim() == 3: x_cw = x_cw.squeeze(-1)
        B, N, d = emb.shape; n = int(math.log2(N))
        all_losses = []
        logits = self.emb2logits(emb)
        all_losses.append(F.cross_entropy(logits.reshape(-1, 2), x_cw.reshape(-1).long()))
        V = [x_cw]; E = [emb]
        for depth in range(n):
            Vo, Ve, Eo, Ee = [], [], [], []
            for v, e in zip(V, E):
                Vo.append(v[:,0::2]); Ve.append(v[:,1::2])
                Eo.append(e[:,0::2,:]); Ee.append(e[:,1::2,:])
            Vo=torch.cat(Vo,1); Ve=torch.cat(Ve,1)
            Eo=torch.cat(Eo,1); Ee=torch.cat(Ee,1)
            vt=Vo^Ve; vb=Ve
            nc=2**depth; cs=(N//2)//nc
            vtc=torch.split(vt,cs,1); vbc=torch.split(vb,cs,1)
            Vn=[]
            for a,b in zip(vtc,vbc): Vn+=[a,b]
            Vl=torch.cat(Vn[0::2],1)

            # CalcLeft: (e_odd, e_even, no_info) → e_top
            no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0).expand_as(Eo)
            e_top = self.calc_left_mlp(torch.cat([Eo, Ee, no_info], -1))

            # CalcRight: (e_odd, e_even, smooth_re_emb(Vl)) → e_bot
            soft = torch.zeros(Vl.shape[0], Vl.shape[1], 2, device=Eo.device)
            soft.scatter_(2, Vl.unsqueeze(-1).long(), 1.0)
            left_emb = self.logits2emb(soft)
            e_bot = self.calc_right_mlp(torch.cat([Eo, Ee, left_emb], -1)) + Ee

            etc=torch.split(e_top,cs,1); ebc=torch.split(e_bot,cs,1)
            En=[]
            for a,b in zip(etc,ebc): En+=[a,b]
            e_all=torch.cat(En,1); v_all=torch.cat(Vn,1)
            logits=self.emb2logits(e_all)
            all_losses.append(F.cross_entropy(logits.reshape(-1,2), v_all.reshape(-1).long()))
            V=Vn; E=En
        return torch.stack(all_losses).mean()

    @torch.no_grad()
    def decode(self, emb, frozen_set):
        B, N, _ = emb.shape; n = int(math.log2(N))
        br = bit_reversal_perm(n)
        u_hat = torch.zeros(B, N, dtype=torch.long)
        leaf_idx = [0]
        def _decode(eb):
            bs = eb.shape[1]
            if bs == 1:
                logits = self.emb2logits(eb[:,0,:])
                idx = leaf_idx[0]; leaf_idx[0] += 1
                nat_idx = int(br[idx])
                if nat_idx in frozen_set: dec = torch.zeros(B, dtype=torch.long)
                else: dec = logits.argmax(dim=-1)
                u_hat[:, nat_idx] = dec
                return dec.unsqueeze(1)
            eo = eb[:,0::2,:]; ee = eb[:,1::2,:]
            no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0).expand_as(eo)
            e_top = self.calc_left_mlp(torch.cat([eo, ee, no_info], -1))
            cw_top = _decode(e_top)
            soft = torch.zeros(cw_top.shape[0], cw_top.shape[1], 2, device=eo.device)
            soft.scatter_(2, cw_top.unsqueeze(-1).long(), 1.0)
            left_emb = self.logits2emb(soft)
            e_bot = self.calc_right_mlp(torch.cat([eo, ee, left_emb], -1)) + ee
            cw_bot = _decode(e_bot)
            cw = torch.zeros(B, bs, dtype=torch.long)
            cw[:,0::2] = cw_top ^ cw_bot; cw[:,1::2] = cw_bot
            return cw
        _decode(emb)
        return u_hat


# ─── Training + MI tracking ─────────────────────────────────────────────────

def train_and_track_mi(model, model_name, N, Av_genie, iters, batch, lr, eval_every=3000):
    """Train a model on the GMAC mixture channel and track MI over iterations."""
    n = int(math.log2(N)); br = bit_reversal_perm(n)
    channel = GaussianMAC(sigma2=SIGMA2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(42)

    mi_log = []
    t0 = time.time()
    print(f'\n  Training {model_name} (params={model.count_parameters():,}, {iters} iters)', flush=True)

    model.train()
    losses = []
    for it in range(1, iters + 1):
        u_msg = rng.integers(0, 2, (batch, N)).astype(np.int8)
        x = polar_encode_batch(u_msg.astype(int))
        v_msg = np.zeros((batch, N), dtype=np.int8)
        for p in Av_genie: v_msg[:, p-1] = rng.integers(0, 2, batch)
        y = polar_encode_batch(v_msg.astype(int))
        z = channel.sample_batch(x.astype(int), y.astype(int))
        emb = model.encode_channel(torch.from_numpy(z[:, br].astype(np.float32)).unsqueeze(-1))
        loss = model.fast_ce(emb, torch.from_numpy(x[:, br]).long())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(loss.item())

        if it % eval_every == 0:
            mi_avg, mi_min, mi_per_pos = measure_leaf_mi(model, model_name, N, Av_genie, br, channel, n_samples=5000)
            elapsed = (time.time() - t0) / 60
            avg_loss = float(np.mean(losses[-500:]))
            print(f'    [{it:>6}] loss={avg_loss:.4f} MI_avg={mi_avg:.4f} MI_min={mi_min:.4f} {elapsed:.1f}min', flush=True)
            mi_log.append({
                'iter': it, 'loss': avg_loss,
                'mi_avg': float(mi_avg), 'mi_min': float(mi_min),
                'mi_per_pos': [float(x) for x in mi_per_pos],
                'elapsed_min': float(elapsed),
            })

    return mi_log


def measure_leaf_mi(model, model_name, N, Av_genie, br, channel, n_samples=5000):
    """Measure per-position leaf-level MI."""
    n = int(math.log2(N))
    model.eval()
    leaf_bce = np.zeros(N); count = 0
    rng = np.random.default_rng(789); np.random.seed(789)
    batch = min(100, n_samples)

    is_npd = isinstance(model, NPDSingleUser)

    with torch.no_grad():
        while count < n_samples:
            actual = min(batch, n_samples - count)
            u = rng.integers(0, 2, (actual, N)).astype(np.int8)
            x = polar_encode_batch(u.astype(int))
            v = np.zeros((actual, N), dtype=np.int8)
            for p in Av_genie: v[:, p-1] = rng.integers(0, 2, actual)
            y = polar_encode_batch(v.astype(int))
            z = channel.sample_batch(x.astype(int), y.astype(int))
            emb = model.encode_channel(torch.from_numpy(z[:, br].astype(np.float32)).unsqueeze(-1))
            B, N_, d = emb.shape

            V = [torch.from_numpy(x[:, br]).long()]; E = [emb]
            for depth in range(n):
                Vo, Ve, Eo, Ee = [], [], [], []
                for vc, ec in zip(V, E):
                    Vo.append(vc[:,0::2]); Ve.append(vc[:,1::2])
                    Eo.append(ec[:,0::2,:]); Ee.append(ec[:,1::2,:])
                Vo=torch.cat(Vo,1); Ve=torch.cat(Ve,1)
                Eo=torch.cat(Eo,1); Ee=torch.cat(Ee,1)
                vt=Vo^Ve; vb=Ve
                nc=2**depth; cs=(N_//2)//nc
                vtc=torch.split(vt,cs,1); vbc=torch.split(vb,cs,1)
                Vn=[]
                for a,b in zip(vtc,vbc): Vn+=[a,b]
                Vl=torch.cat(Vn[0::2],1)

                if is_npd:
                    et = model.checknode(torch.cat([Eo, Ee], -1))
                    eb = model.bitnode(Eo, Ee, Vl)
                else:
                    no_info = model.no_info_emb.unsqueeze(0).unsqueeze(0).expand_as(Eo)
                    et = model.calc_left_mlp(torch.cat([Eo, Ee, no_info], -1))
                    soft = torch.zeros(Vl.shape[0], Vl.shape[1], 2, device=Eo.device)
                    soft.scatter_(2, Vl.unsqueeze(-1).long(), 1.0)
                    left_emb = model.logits2emb(soft)
                    eb = model.calc_right_mlp(torch.cat([Eo, Ee, left_emb], -1)) + Ee

                etc=torch.split(et,cs,1); ebc=torch.split(eb,cs,1)
                En=[]
                for a,b in zip(etc,ebc): En+=[a,b]
                V=Vn; E=En

            e_l = torch.cat(E, 1); v_l = torch.cat(V, 1)
            if is_npd:
                logits = model.emb2llr(e_l).squeeze(-1)
                bce = F.binary_cross_entropy_with_logits(logits, v_l.float(), reduction='none')
            else:
                logits = model.emb2logits(e_l)
                bce = F.cross_entropy(logits.reshape(-1, 2), v_l.reshape(-1).long(), reduction='none')
                bce = bce.reshape(B, N_)

            leaf_bce += bce.sum(0).numpy()
            count += actual

    avg_bce = leaf_bce / count
    bce_nat = np.zeros(N)
    for t in range(N): bce_nat[br[t]] = avg_bce[t]

    if is_npd:
        mi_nat = (np.log(2) - bce_nat) / np.log(2)  # in bits, max 1
    else:
        mi_nat = (np.log(2) - bce_nat) / np.log(2)  # also binary, max 1

    model.train()
    return float(np.mean(mi_nat)), float(np.min(mi_nat)), mi_nat


# ─── Main ────────────────────────────────────────────────────────────────────

def run_comparison(N_list):
    all_results = {}

    for N in N_list:
        n = int(math.log2(N))
        ku = round(0.50 * 0.4645 * N)
        kv = round(0.50 * 0.9119 * N)
        _, Av_genie, _, _, _, _ = design_from_file(
            f'designs/gmac_C_n{n}_snr6dB.npz', n, ku, kv)[0:6]

        # Determine iters based on N
        if N <= 16: iters = 15000; batch = 64; eval_every = 3000
        elif N <= 32: iters = 30000; batch = 64; eval_every = 5000
        elif N <= 64: iters = 50000; batch = 32; eval_every = 5000
        else: iters = 80000; batch = 16; eval_every = 10000

        print(f'\n{"="*70}')
        print(f'N={N}, ku={ku}, kv={kv} — Head-to-head comparison')
        print(f'{"="*70}')

        # Train NPD
        torch.manual_seed(42)
        npd_model = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=1, use_analytical_training=False)
        npd_mi_log = train_and_track_mi(npd_model, 'NPD', N, Av_genie, iters, batch, 3e-4, eval_every)

        # Train CG
        torch.manual_seed(42)
        cg_model = CGStyleDecoderFixed(d=16, hidden=64, n_layers=2, z_dim=1)
        cg_mi_log = train_and_track_mi(cg_model, 'CG', N, Av_genie, iters, batch, 3e-4, eval_every)

        # Measure final per-position MI for both
        br = bit_reversal_perm(n)
        channel = GaussianMAC(sigma2=SIGMA2)
        npd_mi_avg, npd_mi_min, npd_mi_per_pos = measure_leaf_mi(npd_model, 'NPD', N, Av_genie, br, channel, 10000)
        cg_mi_avg, cg_mi_min, cg_mi_per_pos = measure_leaf_mi(cg_model, 'CG', N, Av_genie, br, channel, 10000)

        # NPD-guided design for both
        npd_best = np.argsort(-npd_mi_per_pos)[:ku]
        cg_best = np.argsort(-cg_mi_per_pos)[:ku]
        npd_Au = sorted(int(p)+1 for p in npd_best)
        cg_Au = sorted(int(p)+1 for p in cg_best)
        overlap = len(set(npd_Au) & set(cg_Au))

        print(f'\n  Final MI comparison:')
        print(f'    NPD: MI_avg={npd_mi_avg:.4f} MI_min={npd_mi_min:.4f}')
        print(f'    CG:  MI_avg={cg_mi_avg:.4f} MI_min={cg_mi_min:.4f}')
        print(f'    Design overlap: {overlap}/{ku}')
        print(f'    NPD picks: {npd_Au[:10]}...')
        print(f'    CG picks:  {cg_Au[:10]}...')

        # Evaluate both with their own designs
        npd_frozen = {p-1 for p in range(1, N+1) if p not in npd_Au}
        cg_frozen = {p-1 for p in range(1, N+1) if p not in cg_Au}

        # Quick retrain NPD with its design
        torch.manual_seed(42)
        npd_r = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=1, use_analytical_training=False)
        opt_n = torch.optim.Adam(npd_r.parameters(), lr=3e-4)
        rng = np.random.default_rng(42)
        for it in range(1, iters+1):
            u = np.zeros((batch, N), dtype=np.int8)
            for p in npd_Au: u[:, p-1] = rng.integers(0, 2, batch)
            x = polar_encode_batch(u.astype(int))
            v = np.zeros((batch, N), dtype=np.int8)
            for p in Av_genie: v[:, p-1] = rng.integers(0, 2, batch)
            y = polar_encode_batch(v.astype(int))
            z = channel.sample_batch(x.astype(int), y.astype(int))
            emb = npd_r.encode_channel(torch.from_numpy(z[:, br].astype(np.float32)).unsqueeze(-1))
            loss = npd_r.fast_ce(emb, torch.from_numpy(x[:, br]).long())
            opt_n.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(npd_r.parameters(), 1.0); opt_n.step()

        # Quick retrain CG with its design
        torch.manual_seed(42)
        cg_r = CGStyleDecoderFixed(d=16, hidden=64, n_layers=2, z_dim=1)
        opt_c = torch.optim.Adam(cg_r.parameters(), lr=3e-4)
        rng2 = np.random.default_rng(42)
        for it in range(1, iters+1):
            u = np.zeros((batch, N), dtype=np.int8)
            for p in cg_Au: u[:, p-1] = rng2.integers(0, 2, batch)
            x = polar_encode_batch(u.astype(int))
            v = np.zeros((batch, N), dtype=np.int8)
            for p in Av_genie: v[:, p-1] = rng2.integers(0, 2, batch)
            y = polar_encode_batch(v.astype(int))
            z = channel.sample_batch(x.astype(int), y.astype(int))
            emb = cg_r.encode_channel(torch.from_numpy(z[:, br].astype(np.float32)).unsqueeze(-1))
            loss = cg_r.fast_ce(emb, torch.from_numpy(x[:, br]).long())
            opt_c.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(cg_r.parameters(), 1.0); opt_c.step()

        # Evaluate
        npd_r.eval(); cg_r.eval()
        def eval_model(model, Au, frozen, n_cw=1000):
            errs = 0; rng = np.random.default_rng(999); np.random.seed(999)
            with torch.no_grad():
                for _ in range(n_cw):
                    u = np.zeros((1,N), dtype=np.int8)
                    for p in Au: u[0,p-1] = rng.integers(0,2)
                    x = polar_encode_batch(u.astype(int))
                    v = np.zeros((1,N), dtype=np.int8)
                    for p in Av_genie: v[0,p-1] = rng.integers(0,2)
                    y = polar_encode_batch(v.astype(int))
                    z = channel.sample_batch(x.astype(int), y.astype(int))
                    emb = model.encode_channel(torch.from_numpy(z[:,br].astype(np.float32)).unsqueeze(-1))
                    u_dec = model.decode(emb, frozen)
                    if any(u_dec[0,p-1].item() != u[0,p-1] for p in Au): errs += 1
            return errs / n_cw

        npd_bler = eval_model(npd_r, npd_Au, npd_frozen)
        cg_bler = eval_model(cg_r, cg_Au, cg_frozen)

        # SC reference
        sc_bler = 0.068 if N == 32 else (0.163 if N == 16 else 0.027 if N == 64 else 0.005)

        print(f'\n  BLER results:')
        print(f'    NPD + NPD design: {npd_bler:.4f}')
        print(f'    CG  + CG design:  {cg_bler:.4f}')
        print(f'    SC reference:     {sc_bler:.4f}')

        all_results[N] = {
            'npd_mi_log': npd_mi_log, 'cg_mi_log': cg_mi_log,
            'npd_mi_final': {'avg': npd_mi_avg, 'min': npd_mi_min, 'per_pos': [float(x) for x in npd_mi_per_pos]},
            'cg_mi_final': {'avg': cg_mi_avg, 'min': cg_mi_min, 'per_pos': [float(x) for x in cg_mi_per_pos]},
            'npd_Au': npd_Au, 'cg_Au': cg_Au, 'overlap': overlap,
            'npd_bler': float(npd_bler), 'cg_bler': float(cg_bler), 'sc_bler': float(sc_bler),
        }

        # Save incrementally
        with open(os.path.join(RESULTS_DIR, 'cg_vs_npd_comparison.json'), 'w') as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Final summary
    print(f'\n{"="*70}')
    print('CG vs NPD COMPARISON — FINAL')
    print(f'{"="*70}')
    print(f'{"N":<6}{"NPD BLER":<12}{"CG BLER":<12}{"SC":<10}{"NPD MI_min":<12}{"CG MI_min":<12}{"Overlap":<10}')
    print('-' * 70)
    for N in sorted(all_results.keys()):
        r = all_results[N]
        ku = round(0.50 * 0.4645 * N)
        print(f'{N:<6}{r["npd_bler"]:<12.4f}{r["cg_bler"]:<12.4f}{r["sc_bler"]:<10.4f}'
              f'{r["npd_mi_final"]["min"]:<12.4f}{r["cg_mi_final"]["min"]:<12.4f}{r["overlap"]}/{ku}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_list', type=str, default='16,32')
    args = parser.parse_args()
    N_list = [int(x) for x in args.N_list.split(',')]

    print(f'CG vs NPD comparison on GMAC Class C')
    print(f'N values: {N_list}')
    print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    run_comparison(N_list)
    print(f'\nFinish: {time.strftime("%Y-%m-%d %H:%M:%S")}')
