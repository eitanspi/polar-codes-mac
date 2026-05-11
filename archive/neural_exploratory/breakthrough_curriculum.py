#!/usr/bin/env python3
"""
breakthrough_curriculum.py — Curriculum: train at N=32 with full gradients,
then scale to N=64,128,256 with gradient detaching.

Strategy:
1. Train at N=32 for 20K iters (full gradients) -> BLER ~ 0.05
2. Transfer to N=64, fine-tune with K=32 for 30K iters
3. Transfer to N=128, fine-tune with K=32 for 30K iters
4. Transfer to N=256, fine-tune with K=32 for 30K iters

The key insight: the model learns correct tree operations at N=32,
then adapts them to longer sequences with truncated gradients.
"""
import sys, os, math, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file

D = 16
HIDDEN = 64
N_LAYERS = 2
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'breakthrough_agent.log')
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')

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


class NeuralCalcParent(nn.Module):
    def __init__(self, d, hidden, n_layers=2):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2*d, hidden), nn.ELU(), nn.Linear(hidden, d), nn.Sigmoid()
        )
        self.candidate_net = _make_mlp(2*d, hidden, d, n_layers)

    def forward(self, left, right):
        c = torch.cat([left, right], dim=-1)
        gate = self.gate_net(c)
        cand = self.candidate_net(c)
        res = (left + right) / 2.0
        return gate * cand + (1 - gate) * res


class GmacNeuralDecoder(nn.Module):
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d))
        self.calc_left_nn = _make_mlp(3*d, hidden, d, n_layers)
        self.calc_right_nn = _make_mlp(3*d, hidden, d, n_layers)
        self.calc_parent_nn = NeuralCalcParent(d, hidden, n_layers)
        self.parent_second_nn = nn.Sequential(nn.Linear(d, d))
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)
        self.logits2emb = _make_mlp(4, hidden, d, n_layers)
        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _neural_calc_left(self, beta, edge_data):
        parent = edge_data[beta]; right = edge_data[2*beta+1]
        l = right.shape[1]
        edge_data[2*beta] = self.calc_left_nn(
            torch.cat([parent[:, :l], parent[:, l:], right], dim=-1))

    def _neural_calc_right(self, beta, edge_data):
        parent = edge_data[beta]; left = edge_data[2*beta]
        l = left.shape[1]
        edge_data[2*beta+1] = self.calc_right_nn(
            torch.cat([parent[:, :l], parent[:, l:], left], dim=-1))

    def _calc_parent(self, beta, edge_data):
        left = edge_data[2*beta]; right = edge_data[2*beta+1]
        pf = self.calc_parent_nn(left, right)
        ps = self.parent_second_nn(right)
        edge_data[beta] = torch.cat([pf, ps], dim=1)

    def _step_one(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0: self._neural_calc_left(current, edge_data)
            else: self._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self._calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: {current}->{beta}")

    @staticmethod
    def _get_path(current, target):
        if current == target: return []
        up, down = [], []
        c, t = current, target
        while c != t:
            if c > t: c >>= 1; up.append(c)
            else: down.append(t); t >>= 1
        down.reverse()
        return up + down

    def _step_to(self, current, target, edge_data):
        if current == target: return current
        for beta in self._get_path(current, target):
            current = self._step_one(current, beta, edge_data)
        return current

    def _make_leaf_emb(self, u_val, v_val, batch, device):
        lp = torch.full((batch, 4), -30.0, device=device)
        LH = math.log(0.5)
        if u_val is not None and v_val is not None:
            lp.scatter_(1, (u_val.long() * 2 + v_val.long()).unsqueeze(1), 0.0)
        elif u_val is not None:
            lp.scatter_(1, (u_val.long() * 2).unsqueeze(1), LH)
            lp.scatter_(1, (u_val.long() * 2 + 1).unsqueeze(1), LH)
        elif v_val is not None:
            lp.scatter_(1, v_val.long().unsqueeze(1), LH)
            lp.scatter_(1, (v_val.long() + 2).unsqueeze(1), LH)
        else:
            lp.fill_(math.log(0.25))
        return self.logits2emb(lp)

    def forward(self, z, b, frozen_u, frozen_v, u_true=None, v_true=None,
                detach_every=None):
        B, N = z.shape; device = z.device; d = self.d
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.z_encoder(z.unsqueeze(-1))[:, br]
        teacher = u_true is not None and v_true is not None

        edge_data = [None] * (2*N)
        edge_data[1] = root
        no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0)
        for beta in range(2, 2*N):
            level = beta.bit_length() - 1
            edge_data[beta] = no_info.expand(B, N >> level, d).clone()

        dec_head = 1; u_hat, v_hat = {}, {}
        all_logits, all_targets = [], []
        i_u, i_v = 0, 0

        for step in range(2*N):
            if detach_every and step > 0 and step % detach_every == 0:
                for idx in range(1, 2*N):
                    if edge_data[idx] is not None:
                        edge_data[idx] = edge_data[idx].detach()

            gamma = b[step]
            if gamma == 0: i_u += 1; i_t = i_u; fdict = frozen_u
            else: i_v += 1; i_t = i_v; fdict = frozen_v

            leaf_edge = i_t + N - 1; target_vtx = leaf_edge >> 1
            dec_head = self._step_to(dec_head, target_vtx, edge_data)

            temp = edge_data[leaf_edge][:, 0].clone()
            if leaf_edge & 1 == 0: self._neural_calc_left(target_vtx, edge_data)
            else: self._neural_calc_right(target_vtx, edge_data)
            logits = self.emb2logits(edge_data[leaf_edge][:, 0] + temp)

            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.float32, device=device)
            else:
                all_logits.append(logits)
                if teacher:
                    all_targets.append((u_true[:, i_t-1] * 2 + v_true[:, i_t-1]).long())
                    bit = u_true[:, i_t-1] if gamma == 0 else v_true[:, i_t-1]
                else:
                    with torch.no_grad():
                        if gamma == 0:
                            p0 = torch.logsumexp(logits[:, :2], dim=1)
                            p1 = torch.logsumexp(logits[:, 2:], dim=1)
                        else:
                            p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                            p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                        bit = (p1 > p0).float()

            if gamma == 0: u_hat[i_t] = bit
            else: v_hat[i_t] = bit
            edge_data[leaf_edge] = self._make_leaf_emb(
                u_hat.get(i_t), v_hat.get(i_t), B, device).unsqueeze(1)

        return all_logits, all_targets, u_hat, v_hat


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        return design_from_file(mc_path, n, ku, kv)
    from polar.design import design_gmac
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv, None, None, None


def setup_N(N):
    ku = N//2 - 1; kv = N//2 - 1
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    b = make_path(N, N//2)
    frozen_u = {i: 0 for i in range(1, N+1) if i not in Au}
    frozen_v = {i: 0 for i in range(1, N+1) if i not in Av}
    return Au, Av, fu, fv, b, frozen_u, frozen_v


def evaluate_bler(model, channel, N, Au, Av, b, frozen_u, frozen_v, n_cw=500):
    errs = 0; rng = np.random.default_rng(999)
    model.eval()
    with torch.no_grad():
        for _ in range(n_cw):
            uf = np.zeros((1, N), dtype=int); vf = np.zeros((1, N), dtype=int)
            for p in Au: uf[0, p-1] = rng.integers(0, 2)
            for p in Av: vf[0, p-1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            _, _, u_hat, v_hat = model(zf, b, frozen_u, frozen_v)
            ue = any(u_hat[i].item() != uf[0, i-1] for i in range(1, N+1) if i in Au)
            ve = any(v_hat[i].item() != vf[0, i-1] for i in range(1, N+1) if i in Av)
            if ue or ve: errs += 1
    model.train()
    return errs / n_cw


def gen_batch(Au, Av, N, channel, rng, batch):
    uf = np.zeros((batch, N), dtype=int); vf = np.zeros((batch, N), dtype=int)
    for p in Au: uf[:, p-1] = rng.integers(0, 2, batch)
    for p in Av: vf[:, p-1] = rng.integers(0, 2, batch)
    xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
    return uf, vf, xf, yf, channel.sample_batch(xf, yf)


SC_REF = {32: 0.046, 64: 0.025, 128: 0.016, 256: 0.005}


if __name__ == '__main__':
    log("\n" + "=" * 70)
    log("CURRICULUM TRAINING: N=32 -> N=64 -> N=128 -> N=256")
    log("=" * 70)

    channel = GaussianMAC(sigma2=SIGMA2)
    model = GmacNeuralDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    log(f"Model params: {model.count_parameters():,}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Stage 1: N=32, full gradients, 20K iters
    N = 32
    Au, Av, fu, fv, b, frozen_u, frozen_v = setup_N(N)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(42)
    t0 = time.time()
    best_bler = 1.0

    log(f"\n--- Stage 1: N={N}, full gradients, 20K iters ---")
    for it in range(1, 20001):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng, batch=32)
        u_t = torch.from_numpy(uf).float(); v_t = torch.from_numpy(vf).float()
        zf_t = torch.from_numpy(zf).float()

        all_logits, all_targets, _, _ = model(
            zf_t, b, frozen_u, frozen_v, u_t, v_t, detach_every=None)

        if len(all_logits) > 0:
            loss = F.cross_entropy(torch.cat(all_logits, 0), torch.cat(all_targets, 0))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            loss = torch.tensor(0.0)

        if it % 2000 == 0:
            bler = evaluate_bler(model, channel, N, Au, Av, b, frozen_u, frozen_v, 500)
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'curriculum_N32.pt'))
            log(f"  [{it}/20000] loss={loss.item():.4f} BLER={bler:.4f} "
                f"best={best_bler:.4f} SC={SC_REF[N]} ({time.time()-t0:.0f}s)")

    log(f"Stage 1 complete: N={N} best BLER={best_bler:.4f} (SC={SC_REF[N]})")

    # Stages 2-4: N=64, 128, 256 with detaching
    stages = [
        {'N': 64,  'iters': 30000, 'batch': 16, 'lr': 1e-4, 'K': None},
        {'N': 128, 'iters': 30000, 'batch': 8,  'lr': 5e-5, 'K': None},
        {'N': 256, 'iters': 50000, 'batch': 4,  'lr': 3e-5, 'K': 64},
    ]

    for stage_idx, stage in enumerate(stages, 2):
        N_s = stage['N']
        try:
            Au_s, Av_s, fu_s, fv_s, b_s, fu_d_s, fv_d_s = setup_N(N_s)
        except Exception as e:
            log(f"Stage {stage_idx}: N={N_s} design failed: {e}")
            continue

        K = stage['K']
        K_label = f"K={K}" if K else "full"
        log(f"\n--- Stage {stage_idx}: N={N_s}, {K_label}, {stage['iters']} iters ---")

        opt_s = torch.optim.Adam(model.parameters(), lr=stage['lr'])
        sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_s, stage['iters'], eta_min=stage['lr']/100)
        rng_s = np.random.default_rng(42 + stage_idx)
        t_stage = time.time()
        best_bler_s = 1.0

        # Initial eval
        bler_init = evaluate_bler(model, channel, N_s, Au_s, Av_s, b_s,
                                  fu_d_s, fv_d_s, 300)
        log(f"  Initial BLER: {bler_init:.4f} (SC={SC_REF.get(N_s, '?')})")

        for it in range(1, stage['iters'] + 1):
            uf, vf, xf, yf, zf = gen_batch(Au_s, Av_s, N_s, channel, rng_s,
                                            batch=stage['batch'])
            u_t = torch.from_numpy(uf).float(); v_t = torch.from_numpy(vf).float()
            zf_t = torch.from_numpy(zf).float()

            all_logits, all_targets, _, _ = model(
                zf_t, b_s, fu_d_s, fv_d_s, u_t, v_t, detach_every=K)

            if len(all_logits) > 0:
                loss = F.cross_entropy(torch.cat(all_logits, 0),
                                       torch.cat(all_targets, 0))
                opt_s.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_s.step(); sched_s.step()
            else:
                loss = torch.tensor(0.0)

            eval_every = 5000 if N_s <= 128 else 5000
            if it % eval_every == 0 or it == stage['iters']:
                eval_cw = 500 if N_s <= 128 else 200
                bler = evaluate_bler(model, channel, N_s, Au_s, Av_s, b_s,
                                     fu_d_s, fv_d_s, eval_cw)
                if bler < best_bler_s:
                    best_bler_s = bler
                    torch.save(model.state_dict(),
                               os.path.join(SAVE_DIR, f'curriculum_N{N_s}.pt'))
                log(f"  [{it}/{stage['iters']}] loss={loss.item():.4f} "
                    f"BLER={bler:.4f} best={best_bler_s:.4f} "
                    f"SC={SC_REF.get(N_s, '?')} ({time.time()-t_stage:.0f}s)")

        log(f"Stage {stage_idx} complete: N={N_s} best={best_bler_s:.4f} "
            f"(SC={SC_REF.get(N_s, '?')})")

    log("\n" + "=" * 70)
    log("CURRICULUM TRAINING COMPLETE")
    log("=" * 70)
