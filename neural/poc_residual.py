#!/usr/bin/env python3
"""
poc_residual.py — Add residual connections to CalcLeft/CalcRight, then
curriculum train from scratch: N=16→32→64→128→256.

The change: output = MLP(input) + p_first (skip connection from parent).
This prevents embedding range collapse and gives the MLP an easier target
(learn the correction, not the full transformation).
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import _make_mlp, NeuralCalcParent

D = 16
HIDDEN = 64
N_LAYERS = 2
Z_HIDDEN = 32
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')

SC_REF = {
    16:  {'ku': 8,   'kv': 8,   'sc_bler': 0.10},
    32:  {'ku': 15,  'kv': 15,  'sc_bler': 0.046},
    64:  {'ku': 31,  'kv': 31,  'sc_bler': 0.025},
    128: {'ku': 62,  'kv': 62,  'sc_bler': 0.016},
    256: {'ku': 123, 'kv': 123, 'sc_bler': 0.005},
}

STAGES = [
    {'N': 16,  'iters': 10000,  'batch': 64, 'lr': 3e-4, 'eval_cw': 2000, 'eval_every': 2000},
    {'N': 32,  'iters': 15000,  'batch': 32, 'lr': 3e-4, 'eval_cw': 2000, 'eval_every': 3000},
    {'N': 64,  'iters': 30000,  'batch': 16, 'lr': 1e-4, 'eval_cw': 2000, 'eval_every': 5000},
    {'N': 128, 'iters': 50000,  'batch': 8,  'lr': 8e-5, 'eval_cw': 2000, 'eval_every': 5000},
    {'N': 256, 'iters': 50000,  'batch': 4,  'lr': 5e-5, 'eval_cw': 1000, 'eval_every': 5000},
]


class ResidualGmacDecoder(nn.Module):
    """Same as GmacNeuralCompGraphDecoder but with residual in CalcLeft/CalcRight."""

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_hidden=Z_HIDDEN):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )
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
        p_first = parent[:, :l]
        p_second = parent[:, l:]
        inp = torch.cat([p_first, p_second, right], dim=-1)
        edge_data[2 * beta] = self.calc_left_nn(inp) + p_first  # RESIDUAL

    def _calc_right(self, beta, edge_data):
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]
        p_first = parent[:, :l]
        p_second = parent[:, l:]
        inp = torch.cat([p_first, p_second, left], dim=-1)
        edge_data[2 * beta + 1] = self.calc_right_nn(inp) + p_first  # RESIDUAL

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
            raise ValueError(f"Invalid step: {current} -> {beta}")

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
        teacher = u_true is not None and v_true is not None

        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.z_encoder(z.unsqueeze(-1))[:, br]

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


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    if os.path.exists(mc_path):
        return design_from_file(mc_path, n, ku, kv)[:4]
    from polar.design import design_gmac
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, SIGMA2)
    return Au, Av, fu, fv


def evaluate(model, channel, N, b, Au, Av, fu, fv, n_cw):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(8, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            _, _, uh, vh = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


def get_lr(it, total, base_lr, warmup=1000):
    if it < warmup:
        return base_lr * it / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    model = ResidualGmacDecoder()
    print(f'Residual GMAC Decoder: {model.count_parameters():,} params', flush=True)
    print(f'SNR={SNR_DB}dB, Class B', flush=True)

    results = {}
    for stage in STAGES:
        N = stage['N']
        ref = SC_REF[N]
        ku, kv = ref['ku'], ref['kv']
        sc_bler = ref['sc_bler']
        b = make_path(N, N // 2)
        Au, Av, fu, fv = load_design(N, ku, kv)

        n_iters = stage['iters']
        batch = stage['batch']
        lr = stage['lr']
        eval_cw = stage['eval_cw']
        eval_every = stage['eval_every']

        print(f'\n{"="*60}', flush=True)
        print(f'N={N}, ku={ku}, kv={kv}, SC={sc_bler}', flush=True)
        print(f'batch={batch}, lr={lr}, iters={n_iters}', flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        rng = np.random.default_rng(42 + N)
        t0 = time.time()
        losses = []
        best_bler = 1.0

        model.train()
        for it in range(1, n_iters + 1):
            lr_now = get_lr(it, n_iters, lr)
            for pg in opt.param_groups:
                pg['lr'] = lr_now

            uf = np.zeros((batch, N), dtype=int); vf = np.zeros((batch, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, batch)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, batch)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            logits_list, targets_list, _, _ = model(zf, b, fu, fv,
                u_true=torch.from_numpy(uf).float(),
                v_true=torch.from_numpy(vf).float())

            if logits_list:
                loss = F.cross_entropy(torch.stack(logits_list).reshape(-1, 4),
                                       torch.stack(targets_list).reshape(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

            if it % eval_every == 0:
                elapsed = time.time() - t0
                avg_loss = np.mean(losses[-500:])
                bler = evaluate(model, channel, N, b, Au, Av, fu, fv, eval_cw)
                improved = ''
                if bler < best_bler:
                    best_bler = bler
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'residual_N{N}.pt'))
                    improved = ' *BEST*'
                ratio = bler / max(sc_bler, 1e-8)
                print(f'  [{it:>6}/{n_iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                      f'(best={best_bler:.4f}, SC={sc_bler}, ratio={ratio:.2f}x) '
                      f'{elapsed/60:.0f}min{improved}', flush=True)

        results[N] = {'best_bler': best_bler, 'sc_bler': sc_bler,
                       'ratio': best_bler / max(sc_bler, 1e-8)}
        print(f'  N={N} DONE: best={best_bler:.4f} (SC={sc_bler}, '
              f'ratio={results[N]["ratio"]:.2f}x)', flush=True)

    print(f'\n{"="*60}', flush=True)
    print(f'FINAL SUMMARY', flush=True)
    for N, r in sorted(results.items()):
        print(f'  N={N:>5}: BLER={r["best_bler"]:.4f} (SC={r["sc_bler"]}, ratio={r["ratio"]:.2f}x)', flush=True)


if __name__ == '__main__':
    main()
