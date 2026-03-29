#!/usr/bin/env python3
"""
train_gmac_d32.py — Train bigger GMAC neural decoder (d=32, deeper z_encoder).

Hypothesis: d=16/hidden=64 is under-parameterized for continuous GMAC channel.
Doubling capacity + deeper residual z_encoder should close the N=128 gap.

Architecture changes vs baseline (d=16):
  - d=32, hidden=128 (~100K params vs ~25K)
  - z_encoder: 4-layer residual MLP (vs 2-layer shallow)
  - Same tree operations (CalcLeft/CalcRight/CalcParent weight-shared)

Training:
  - Curriculum: N=32 -> N=64 -> N=128
  - MC design (critical for Class B interleaved path)
  - Cosine annealing with warm restarts
  - Class B, SNR=6dB

Usage:
    cd to_git_v2
    python3 -u neural/train_gmac_d32.py 2>&1 | tee neural/train_gmac_d32.log
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

D = 32
HIDDEN = 128
N_LAYERS = 2
Z_HIDDEN = 64

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'train_gmac_d32_results.json')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')

SC_REF = {
    32:  {'ku': 15,  'kv': 15,  'sc_bler': 0.046},
    64:  {'ku': 31,  'kv': 31,  'sc_bler': 0.025},
    128: {'ku': 62,  'kv': 62,  'sc_bler': 0.016},
}

# Curriculum stages
STAGES = [
    {'N': 32,  'iters': 15000, 'batch': 32, 'lr': 3e-4, 'eval_cw': 3000, 'eval_every': 2000},
    {'N': 64,  'iters': 30000, 'batch': 16, 'lr': 1e-4, 'eval_cw': 3000, 'eval_every': 3000},
    {'N': 128, 'iters': 30000, 'batch': 8,  'lr': 5e-5, 'eval_cw': 2000, 'eval_every': 3000},
]


# ─── Residual z_encoder ────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class DeepZEncoder(nn.Module):
    """4-layer residual z_encoder: project to d, then 2 residual blocks."""
    def __init__(self, d, z_hidden):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(d),
            nn.ELU(),
            ResidualBlock(d),
        )

    def forward(self, x):
        return self.res_blocks(self.project(x))


# ─── MLP helpers ────────────────────────────────────────────────────────────

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
            nn.Linear(2 * d, hidden), nn.ELU(),
            nn.Linear(hidden, d), nn.Sigmoid()
        )
        self.candidate_net = _make_mlp(2 * d, hidden, d, n_layers)

    def forward(self, left_emb, right_emb):
        concat = torch.cat([left_emb, right_emb], dim=-1)
        gate = self.gate_net(concat)
        candidate = self.candidate_net(concat)
        residual = (left_emb + right_emb) / 2.0
        return gate * candidate + (1 - gate) * residual


# ─── Model ──────────────────────────────────────────────────────────────────

class GmacDecoderD32(nn.Module):
    """Bigger GMAC neural decoder with deep residual z_encoder."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_hidden=Z_HIDDEN):
        super().__init__()
        self.d = d

        self.z_encoder = DeepZEncoder(d, z_hidden)

        self.calc_left_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_parent_nn = NeuralCalcParent(d, hidden, n_layers)
        self.parent_second_nn = nn.Sequential(nn.Linear(d, d))

        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)
        self.logits2emb = _make_mlp(4, hidden, d, n_layers)

        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _calc_left(self, beta, edge_data):
        parent = edge_data[beta]
        right = edge_data[2 * beta + 1]
        l = right.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], right], dim=-1)
        edge_data[2 * beta] = self.calc_left_nn(inp)

    def _calc_right(self, beta, edge_data):
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], left], dim=-1)
        edge_data[2 * beta + 1] = self.calc_right_nn(inp)

    def _calc_parent(self, beta, edge_data):
        left = edge_data[2 * beta]
        right = edge_data[2 * beta + 1]
        parent_first = self.calc_parent_nn(left, right)
        parent_second = self.parent_second_nn(right)
        edge_data[beta] = torch.cat([parent_first, parent_second], dim=1)

    def _step_one(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0:
                self._calc_left(current, edge_data)
            else:
                self._calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self._calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: current={current}, target={beta}")

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

    def forward(self, z, b, fu, fv, u_true=None, v_true=None):
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
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

        dec_head = 1
        u_hat, v_hat = {}, {}
        all_logits, all_targets = [], []
        i_u, i_v = 0, 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = fu
            else:
                i_v += 1; i_t = i_v; fdict = fv

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1

            dec_head = self._step_to(dec_head, target_vtx, edge_data)

            temp = edge_data[leaf_edge][:, 0].clone()

            if leaf_edge & 1 == 0:
                self._calc_left(target_vtx, edge_data)
            else:
                self._calc_right(target_vtx, edge_data)
            top_down = edge_data[leaf_edge][:, 0]

            combined = top_down + temp
            logits = self.emb2logits(combined)

            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.float32, device=device)
            else:
                all_logits.append(logits)
                if teacher:
                    target = (u_true[:, i_t - 1] * 2 + v_true[:, i_t - 1]).long()
                    all_targets.append(target)
                    bit = u_true[:, i_t - 1] if gamma == 0 else v_true[:, i_t - 1]
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


# ─── Training helpers ───────────────────────────────────────────────────────

def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
    return Au, Av, fu, fv


def evaluate(model, channel, N, b, Au, Av, fu, fv, n_cw):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    bs = max(2, min(16, 128 // max(1, N // 16)))
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            _, _, uh, vh, = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


def train_stage(model, stage, channel, prev_state=None):
    N = stage['N']
    n_iters = stage['iters']
    batch = stage['batch']
    lr = stage['lr']
    eval_cw = stage['eval_cw']
    eval_every = stage['eval_every']

    ref = SC_REF[N]
    ku, kv = ref['ku'], ref['kv']
    sc_bler = ref['sc_bler']
    b = make_path(N, N // 2)
    Au, Av, fu, fv = load_design(N, ku, kv)

    print(f'\n{"="*60}', flush=True)
    print(f'  N={N}, ku={ku}, kv={kv}, SC_ref={sc_bler}', flush=True)
    print(f'  d={model.d}, params={model.count_parameters():,}', flush=True)
    print(f'  batch={batch}, lr={lr}, iters={n_iters}', flush=True)
    print(f'  {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'{"="*60}', flush=True)

    init_bler = evaluate(model, channel, N, b, Au, Av, fu, fv, min(eval_cw, 1000))
    print(f'  Initial BLER: {init_bler:.4f} (SC={sc_bler})', flush=True)

    best_bler = init_bler
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1000, n_iters // 4), T_mult=2, eta_min=lr * 0.01)

    rng = np.random.default_rng()
    t0 = time.time()
    losses = []

    for it in range(1, n_iters + 1):
        u = np.zeros((batch, N), dtype=int); v = np.zeros((batch, N), dtype=int)
        for p in Au: u[:, p-1] = rng.integers(0, 2, batch)
        for p in Av: v[:, p-1] = rng.integers(0, 2, batch)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(channel.sample_batch(x, y)).float()

        logits, targets, _, _ = model(z, b, fu, fv,
            u_true=torch.from_numpy(u).float(),
            v_true=torch.from_numpy(v).float())

        if logits:
            loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4),
                                   torch.stack(targets).reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            losses.append(loss.item())

        if it % eval_every == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-min(len(losses), 200):])
            bler = evaluate(model, channel, N, b, Au, Av, fu, fv, eval_cw)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(SAVE_DIR, f'ncg_gmac_d32_N{N}.pt'))
                improved = ' *BEST*'

            ratio = bler / max(sc_bler, 1e-8)
            print(f'  [{it:>6}/{n_iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}, SC={sc_bler}, ratio={ratio:.2f}x) '
                  f'{elapsed/60:.0f}min lr={sch.get_last_lr()[0]:.1e}{improved}',
                  flush=True)

    # Final eval
    model.load_state_dict(best_state)
    final_bler = evaluate(model, channel, N, b, Au, Av, fu, fv, eval_cw * 2)
    total_time = time.time() - t0

    print(f'\n  N={N} DONE: best={best_bler:.4f}, final={final_bler:.4f}, '
          f'{total_time/60:.0f}min', flush=True)

    return {
        'N': N, 'ku': ku, 'kv': kv,
        'best_bler': best_bler, 'final_bler': final_bler,
        'sc_bler': sc_bler, 'ratio': best_bler / max(sc_bler, 1e-8),
        'iters': n_iters, 'time_min': total_time / 60,
        'd': D, 'hidden': HIDDEN, 'params': model.count_parameters(),
    }


def main():
    print('=' * 60, flush=True)
    print(f'GMAC Neural Decoder Training — d={D}, hidden={HIDDEN}', flush=True)
    print(f'Deep residual z_encoder, z_hidden={Z_HIDDEN}', flush=True)
    print(f'SNR={SNR_DB}dB, Class B, MC design', flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print('=' * 60, flush=True)

    channel = GaussianMAC(sigma2=SIGMA2)
    model = GmacDecoderD32(d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_hidden=Z_HIDDEN)
    print(f'Model params: {model.count_parameters():,}', flush=True)

    results = {}
    for stage in STAGES:
        result = train_stage(model, stage, channel)
        results[str(stage['N'])] = result
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        gc.collect()

    print(f'\n{"="*60}', flush=True)
    print('TRAINING COMPLETE', flush=True)
    print(f'{"="*60}', flush=True)
    for k, r in sorted(results.items(), key=lambda x: int(x[0])):
        print(f'  N={r["N"]}: best={r["best_bler"]:.4f} '
              f'(SC={r["sc_bler"]}, ratio={r["ratio"]:.2f}x, '
              f'{r["time_min"]:.0f}min)', flush=True)


if __name__ == '__main__':
    main()
