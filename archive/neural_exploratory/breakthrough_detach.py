#!/usr/bin/env python3
"""
breakthrough_detach.py — Gradient detaching for scalable MAC decoder training.

The sequential tree walk decoder works (matches SC at N=32-128) but has
O(N log N) gradient depth, preventing training at N>=256.

Solution: Detach gradients every K steps during sequential training.
This limits gradient depth to O(K) while still training sequentially.

Also test: multi-scale curriculum (train at N=32, grow to N=64, 128, 256).
"""
import sys, os, math, time, traceback, copy
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
        gate = self.gate_net(c)
        cand = self.candidate_net(c)
        res = (left + right) / 2.0
        return gate * cand + (1 - gate) * res


class GmacNeuralDecoder(nn.Module):
    """NCG-style tree walk decoder for GMAC, identical to ncg_gmac.py."""

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
        parent = edge_data[beta]
        right = edge_data[2*beta+1]
        l = right.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], right], dim=-1)
        edge_data[2*beta] = self.calc_left_nn(inp)

    def _neural_calc_right(self, beta, edge_data):
        parent = edge_data[beta]
        left = edge_data[2*beta]
        l = left.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], left], dim=-1)
        edge_data[2*beta+1] = self.calc_right_nn(inp)

    def _calc_parent(self, beta, edge_data):
        left = edge_data[2*beta]
        right = edge_data[2*beta+1]
        pf = self.calc_parent_nn(left, right)
        ps = self.parent_second_nn(right)
        edge_data[beta] = torch.cat([pf, ps], dim=1)

    def _step_one(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0:
                self._neural_calc_left(current, edge_data)
            else:
                self._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self._calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: {current}->{beta}")

    @staticmethod
    def _get_path(current, target):
        if current == target:
            return []
        path_up, path_down = [], []
        c, t = current, target
        while c != t:
            if c > t:
                c >>= 1
                path_up.append(c)
            else:
                path_down.append(t)
                t >>= 1
        path_down.reverse()
        return path_up + path_down

    def _step_to(self, current, target, edge_data):
        if current == target:
            return current
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
        B, N = z.shape
        device = z.device
        d = self.d
        n = int(math.log2(N))
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.z_encoder(z.unsqueeze(-1))[:, br]

        teacher = u_true is not None and v_true is not None

        edge_data = [None] * (2 * N)
        edge_data[1] = root
        no_info = self.no_info_emb.unsqueeze(0).unsqueeze(0)
        for beta in range(2, 2*N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

        dec_head = 1
        u_hat, v_hat = {}, {}
        all_logits, all_targets = [], []
        i_u, i_v = 0, 0

        for step in range(2 * N):
            # Gradient detaching
            if detach_every and step > 0 and step % detach_every == 0:
                for idx in range(1, 2*N):
                    if edge_data[idx] is not None:
                        edge_data[idx] = edge_data[idx].detach()

            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = frozen_v

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1
            dec_head = self._step_to(dec_head, target_vtx, edge_data)

            temp = edge_data[leaf_edge][:, 0].clone()
            if leaf_edge & 1 == 0:
                self._neural_calc_left(target_vtx, edge_data)
            else:
                self._neural_calc_right(target_vtx, edge_data)
            top_down = edge_data[leaf_edge][:, 0]
            combined = top_down + temp
            logits = self.emb2logits(combined)

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

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

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


def evaluate_bler(model, channel, N, Au, Av, b, frozen_u, frozen_v, n_cw=500):
    errs = 0
    rng = np.random.default_rng(999)
    model.eval()
    with torch.no_grad():
        for _ in range(n_cw):
            uf = np.zeros((1, N), dtype=int)
            vf = np.zeros((1, N), dtype=int)
            for p in Au: uf[0, p-1] = rng.integers(0, 2)
            for p in Av: vf[0, p-1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            _, _, u_hat, v_hat = model(zf, b, frozen_u, frozen_v)
            ue = any(u_hat[i_t].item() != uf[0, i_t-1]
                     for i_t in range(1, N+1) if i_t in Au)
            ve = any(v_hat[i_t].item() != vf[0, i_t-1]
                     for i_t in range(1, N+1) if i_t in Av)
            if ue or ve:
                errs += 1
    model.train()
    return errs / n_cw


def gen_batch(Au, Av, N, channel, rng, batch):
    uf = np.zeros((batch, N), dtype=int)
    vf = np.zeros((batch, N), dtype=int)
    for p in Au: uf[:, p-1] = rng.integers(0, 2, batch)
    for p in Av: vf[:, p-1] = rng.integers(0, 2, batch)
    xf = polar_encode_batch(uf)
    yf = polar_encode_batch(vf)
    zf = channel.sample_batch(xf, yf)
    return uf, vf, xf, yf, zf


def setup_N(N):
    """Set up design for given N."""
    n = int(math.log2(N))
    ku = N // 2 - 1
    kv = N // 2 - 1
    result = load_design(N, ku, kv)
    Au, Av, fu, fv = result[0], result[1], result[2], result[3]
    b = make_path(N, N // 2)
    frozen_u = {i: 0 for i in range(1, N+1) if i not in Au}
    frozen_v = {i: 0 for i in range(1, N+1) if i not in Av}
    return Au, Av, fu, fv, b, frozen_u, frozen_v


def train_sequential(model, N, channel, Au, Av, b, frozen_u, frozen_v,
                     n_iters, batch, lr, detach_every=None, rng_seed=42,
                     eval_every=2000, eval_cw=500, label=""):
    """Train sequentially with optional gradient detaching."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(rng_seed)
    t0 = time.time()
    best_bler = 1.0

    for it in range(1, n_iters + 1):
        uf, vf, xf, yf, zf = gen_batch(Au, Av, N, channel, rng, batch)
        zf_t = torch.from_numpy(zf).float()
        u_t = torch.from_numpy(uf).float()
        v_t = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _ = model(
            zf_t, b, frozen_u, frozen_v, u_t, v_t,
            detach_every=detach_every)

        if len(all_logits) > 0:
            loss = F.cross_entropy(torch.cat(all_logits, 0),
                                   torch.cat(all_targets, 0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            loss = torch.tensor(0.0)

        if it % eval_every == 0 or it == n_iters:
            bler = evaluate_bler(model, channel, N, Au, Av, b,
                                 frozen_u, frozen_v, eval_cw)
            if bler < best_bler:
                best_bler = bler
            elapsed = time.time() - t0
            log(f"  {label}[{it}/{n_iters}] loss={loss.item():.4f} "
                f"BLER={bler:.4f} best={best_bler:.4f} ({elapsed:.0f}s)")

    return best_bler


def run_detach_comparison():
    """Compare different detach_every values at N=32."""
    log("=" * 70)
    log("EXPERIMENT: Detach comparison at N=32")
    log("=" * 70)

    N = 32
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv, b, frozen_u, frozen_v = setup_N(N)

    results = {}

    # Baseline: no detaching (full gradients)
    log("--- No detaching (baseline) ---")
    model_full = GmacNeuralDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    bler_full = train_sequential(model_full, N, channel, Au, Av, b,
                                  frozen_u, frozen_v, n_iters=10000,
                                  batch=32, lr=3e-4, detach_every=None,
                                  label="full ")
    results['full'] = bler_full

    # Detach every K steps
    for K in [4, 8, 16, 32]:
        log(f"--- Detach every K={K} ---")
        model_k = GmacNeuralDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
        bler_k = train_sequential(model_k, N, channel, Au, Av, b,
                                   frozen_u, frozen_v, n_iters=10000,
                                   batch=32, lr=3e-4, detach_every=K,
                                   label=f"K={K} ")
        results[f'K={K}'] = bler_k

    log("\nDetach comparison results:")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        log(f"  {k}: BLER={v:.4f}")

    return results


def run_scaling_test():
    """Test gradient detaching at N=64, 128, 256."""
    log("=" * 70)
    log("EXPERIMENT: Scaling test (N=64, 128, 256)")
    log("=" * 70)

    channel = GaussianMAC(sigma2=SIGMA2)
    results = {}

    for N in [64, 128, 256]:
        n = int(math.log2(N))
        ku = N // 2 - 1
        kv = N // 2 - 1

        try:
            Au, Av, fu, fv, b, frozen_u, frozen_v = setup_N(N)
        except Exception as e:
            log(f"N={N}: Failed to load design: {e}")
            continue

        # Determine batch size and iterations based on N
        if N <= 64:
            batch, n_iters = 32, 15000
        elif N <= 128:
            batch, n_iters = 16, 15000
        else:
            batch, n_iters = 8, 15000

        # Use detach_every = n (log2(N))
        K = n

        log(f"--- N={N}, K={K}, batch={batch}, iters={n_iters} ---")
        model = GmacNeuralDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
        log(f"Params: {model.count_parameters():,}")

        bler = train_sequential(model, N, channel, Au, Av, b,
                                frozen_u, frozen_v, n_iters=n_iters,
                                batch=batch, lr=3e-4, detach_every=K,
                                eval_every=3000, eval_cw=300,
                                label=f"N={N} ")
        results[f'N={N}_K={K}'] = bler

        # Also try full gradients for comparison (if N <= 128)
        if N <= 128:
            log(f"--- N={N}, full gradients ---")
            model2 = GmacNeuralDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
            bler2 = train_sequential(model2, N, channel, Au, Av, b,
                                     frozen_u, frozen_v, n_iters=n_iters,
                                     batch=batch, lr=3e-4, detach_every=None,
                                     eval_every=3000, eval_cw=300,
                                     label=f"N={N}_full ")
            results[f'N={N}_full'] = bler2

    log("\nScaling results:")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        log(f"  {k}: BLER={v:.4f}")

    return results


def run_curriculum():
    """Curriculum learning: train at N=32, transfer to N=64, 128, 256."""
    log("=" * 70)
    log("EXPERIMENT: Curriculum learning")
    log("=" * 70)

    channel = GaussianMAC(sigma2=SIGMA2)

    # Phase 1: Train at N=32
    N = 32
    Au, Av, fu, fv, b, frozen_u, frozen_v = setup_N(N)
    model = GmacNeuralDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    log(f"Phase 1: Train at N={N}")
    bler32 = train_sequential(model, N, channel, Au, Av, b,
                               frozen_u, frozen_v, n_iters=10000,
                               batch=32, lr=3e-4, detach_every=None,
                               label="N=32 ")
    log(f"N=32 BLER={bler32:.4f}")

    # Phase 2: Fine-tune at N=64 (same weights)
    for N_next in [64, 128, 256]:
        try:
            Au_n, Av_n, fu_n, fv_n, b_n, frozen_u_n, frozen_v_n = setup_N(N_next)
        except Exception as e:
            log(f"N={N_next}: {e}")
            continue

        if N_next <= 64:
            batch, n_iters = 32, 10000
        elif N_next <= 128:
            batch, n_iters = 16, 10000
        else:
            batch, n_iters = 8, 10000

        K = int(math.log2(N_next))
        log(f"\nPhase: Fine-tune at N={N_next}, K={K}")
        bler = train_sequential(model, N_next, channel, Au_n, Av_n, b_n,
                                frozen_u_n, frozen_v_n, n_iters=n_iters,
                                batch=batch, lr=1e-4, detach_every=K,
                                eval_every=2000, eval_cw=300,
                                label=f"N={N_next}_curriculum ")
        log(f"N={N_next} curriculum BLER={bler:.4f}")

        if bler < 0.10:
            # Save model
            os.makedirs(os.path.join(os.path.dirname(__file__), 'saved_models'), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(os.path.dirname(__file__), 'saved_models',
                                    f'curriculum_N{N_next}.pt'))
            log(f"Saved curriculum model for N={N_next}")


if __name__ == '__main__':
    log("\n" + "=" * 70)
    log("BREAKTHROUGH DETACH EXPERIMENTS")
    log("=" * 70)

    # 1. Compare detach values at N=32
    try:
        detach_results = run_detach_comparison()
    except Exception as e:
        log(f"Detach comparison FAILED: {e}")
        traceback.print_exc()

    # 2. Scaling test
    try:
        scale_results = run_scaling_test()
    except Exception as e:
        log(f"Scaling test FAILED: {e}")
        traceback.print_exc()

    # 3. Curriculum
    try:
        run_curriculum()
    except Exception as e:
        log(f"Curriculum FAILED: {e}")
        traceback.print_exc()

    log("\n" + "=" * 70)
    log("DETACH EXPERIMENTS COMPLETE")
    log("=" * 70)
