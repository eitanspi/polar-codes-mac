#!/usr/bin/env python3
"""
breakthrough_long.py — Extended gradient detaching experiments.

Key finding from short runs: detaching WORKS but needs more iterations.
- Full gradients: BLER=0.112 at 10K iters (phase transition at ~7K)
- K=16: loss=0.24, BLER=0.992 at 10K (almost at transition)
- K=32: loss=0.83 at 10K (same as full gradients pre-transition)

This script runs K=16,32 for 50K+ iterations to see convergence.
Then scales to N=256 with K=32.
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
        return self.gate_net(c) * self.candidate_net(c) + (1 - self.gate_net(c)) * (left + right) / 2


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
            idx = (u_val.long() * 2 + v_val.long()).unsqueeze(1)
            lp.scatter_(1, idx, 0.0)
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
            level = beta.bit_length() - 1; size = N >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

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
            top_down = edge_data[leaf_edge][:, 0]
            logits = self.emb2logits(top_down + temp)

            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.float32, device=device)
            else:
                all_logits.append(logits)
                if teacher:
                    target = (u_true[:, i_t-1] * 2 + v_true[:, i_t-1]).long()
                    all_targets.append(target)
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

            new_emb = self._make_leaf_emb(u_hat.get(i_t), v_hat.get(i_t), B, device)
            edge_data[leaf_edge] = new_emb.unsqueeze(1)

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
    n = int(math.log2(N)); ku = N//2 - 1; kv = N//2 - 1
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


def train(model, N, channel, Au, Av, b, frozen_u, frozen_v,
          n_iters, batch, lr, detach_every=None, rng_seed=42,
          eval_every=5000, eval_cw=500, label="", save_best=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iters, eta_min=lr/100)
    rng = np.random.default_rng(rng_seed)
    t0 = time.time(); best_bler = 1.0; best_state = None

    for it in range(1, n_iters + 1):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng, batch)
        zf_t = torch.from_numpy(zf).float()
        u_t = torch.from_numpy(uf).float()
        v_t = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _ = model(
            zf_t, b, frozen_u, frozen_v, u_t, v_t, detach_every=detach_every)

        if len(all_logits) > 0:
            loss = F.cross_entropy(torch.cat(all_logits, 0), torch.cat(all_targets, 0))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        else:
            loss = torch.tensor(0.0)

        if it % eval_every == 0 or it == n_iters:
            bler = evaluate_bler(model, channel, N, Au, Av, b,
                                 frozen_u, frozen_v, eval_cw)
            if bler < best_bler:
                best_bler = bler
                if save_best:
                    os.makedirs(os.path.dirname(save_best), exist_ok=True)
                    torch.save(model.state_dict(), save_best)
            log(f"  {label}[{it}/{n_iters}] loss={loss.item():.4f} "
                f"BLER={bler:.4f} best={best_bler:.4f} "
                f"lr={sched.get_last_lr()[0]:.1e} ({time.time()-t0:.0f}s)")

    return best_bler


if __name__ == '__main__':
    log("\n" + "=" * 70)
    log("EXTENDED DETACH EXPERIMENTS")
    log("=" * 70)

    channel = GaussianMAC(sigma2=SIGMA2)

    # 1. Longer runs at N=32 with K=16, K=32
    N = 32
    Au, Av, fu, fv, b, frozen_u, frozen_v = setup_N(N)

    for K in [16, 32, None]:
        K_label = f"K={K}" if K else "full"
        log(f"\n--- N={N}, {K_label}, 30K iters ---")
        model = GmacNeuralDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
        save_path = os.path.join(os.path.dirname(__file__), 'saved_models',
                                 f'detach_{K_label}_N{N}.pt')
        bler = train(model, N, channel, Au, Av, b, frozen_u, frozen_v,
                     n_iters=30000, batch=32, lr=3e-4, detach_every=K,
                     eval_every=5000, label=f"{K_label} ", save_best=save_path)
        log(f"N={N} {K_label}: best BLER={bler:.4f} (SC=0.046)")

    # 2. N=64 with K=16, K=32, full
    N = 64
    Au64, Av64, fu64, fv64, b64, fu64_d, fv64_d = setup_N(N)

    for K in [16, 32, None]:
        K_label = f"K={K}" if K else "full"
        log(f"\n--- N={N}, {K_label}, 30K iters ---")
        model = GmacNeuralDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
        save_path = os.path.join(os.path.dirname(__file__), 'saved_models',
                                 f'detach_{K_label}_N{N}.pt')
        bler = train(model, N, channel, Au64, Av64, b64, fu64_d, fv64_d,
                     n_iters=30000, batch=16, lr=3e-4, detach_every=K,
                     eval_every=5000, label=f"{K_label} ", save_best=save_path)
        log(f"N={N} {K_label}: best BLER={bler:.4f} (SC=0.025)")

    # 3. N=256 with K=32 (the real target!)
    N = 256
    try:
        Au256, Av256, fu256, fv256, b256, fu256_d, fv256_d = setup_N(N)
        for K in [32, 64]:
            K_label = f"K={K}"
            log(f"\n--- N={N}, {K_label}, 30K iters ---")
            model = GmacNeuralDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
            save_path = os.path.join(os.path.dirname(__file__), 'saved_models',
                                     f'detach_{K_label}_N{N}.pt')
            bler = train(model, N, channel, Au256, Av256, b256, fu256_d, fv256_d,
                         n_iters=30000, batch=4, lr=3e-4, detach_every=K,
                         eval_every=5000, eval_cw=200, label=f"{K_label} ",
                         save_best=save_path)
            log(f"N={N} {K_label}: best BLER={bler:.4f}")
    except Exception as e:
        log(f"N=256 FAILED: {e}")
        traceback.print_exc()

    log("\n" + "=" * 70)
    log("EXTENDED DETACH EXPERIMENTS COMPLETE")
    log("=" * 70)
