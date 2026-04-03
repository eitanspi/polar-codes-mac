#!/usr/bin/env python3
"""
train_d32_30hr.py — 30-hour d=32 training campaign.

The d=16 model plateaus at ~0.015 BLER regardless of N.
This tests whether d=32 (157K params) can break that ceiling
with enough training budget (300K+ iters per N).

Curriculum: N=32 → N=64 → N=128 → N=256
Each stage gets enough iters to converge before moving on.
Uses C++ accelerated forward pass where beneficial.
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
from neural.ncg_pure_neural import _make_mlp, NeuralCalcParent

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
D = 32
HIDDEN = 128
N_LAYERS = 2
Z_HIDDEN = 64

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'train_d32_30hr.log')
TOTAL_HOURS = 30

SC_REF = {
    32:  {'ku': 15,  'kv': 15,  'sc_bler': 0.046},
    64:  {'ku': 31,  'kv': 31,  'sc_bler': 0.025},
    128: {'ku': 62,  'kv': 62,  'sc_bler': 0.016},
    256: {'ku': 123, 'kv': 123, 'sc_bler': 0.005},
}

# Time allocation per N (approximate)
STAGES = [
    {'N': 32,  'hours': 2,  'batch': 32, 'lr': 3e-4, 'eval_cw': 2000, 'eval_every': 3000},
    {'N': 64,  'hours': 6,  'batch': 16, 'lr': 1e-4, 'eval_cw': 2000, 'eval_every': 5000},
    {'N': 128, 'hours': 10, 'batch': 8,  'lr': 8e-5, 'eval_cw': 2000, 'eval_every': 5000},
    {'N': 256, 'hours': 12, 'batch': 4,  'lr': 5e-5, 'eval_cw': 1000, 'eval_every': 5000},
]


class D32GmacDecoder(nn.Module):
    """d=32 GMAC decoder — same architecture as d=16 but bigger."""

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

    def _neural_calc_left(self, beta, edge_data):
        parent = edge_data[beta]
        right = edge_data[2 * beta + 1]
        l = right.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], right], dim=-1)
        edge_data[2 * beta] = self.calc_left_nn(inp)

    def _neural_calc_right(self, beta, edge_data):
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], left], dim=-1)
        edge_data[2 * beta + 1] = self.calc_right_nn(inp)

    def _pure_neural_calc_parent(self, beta, edge_data):
        left = edge_data[2 * beta]
        right = edge_data[2 * beta + 1]
        parent_first = self.calc_parent_nn(left, right)
        parent_second = self.parent_second_nn(right)
        edge_data[beta] = torch.cat([parent_first, parent_second], dim=1)

    def _step_one(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0:
                self._neural_calc_left(current, edge_data)
            else:
                self._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self._pure_neural_calc_parent(current, edge_data)
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


def get_lr(it, total_iters, base_lr, warmup=1000):
    if it < warmup:
        return base_lr * it / warmup
    progress = (it - warmup) / max(1, total_iters - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    global_start = time.time()
    channel = GaussianMAC(sigma2=SIGMA2)

    model = D32GmacDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_hidden=Z_HIDDEN)
    print(f'{"="*60}', flush=True)
    print(f'd=32 GMAC Decoder — 30-Hour Training Campaign', flush=True)
    print(f'Params: {model.count_parameters():,}', flush=True)
    print(f'd={D}, hidden={HIDDEN}, z_hidden={Z_HIDDEN}', flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'Budget: {TOTAL_HOURS} hours', flush=True)
    print(f'{"="*60}', flush=True)

    # Try to load existing d=32 checkpoint
    ckpt = os.path.join(SAVE_DIR, 'd32_30hr_best.pt')
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False))
        print(f'Resumed from: {ckpt}', flush=True)
    else:
        # Try to load the old d=32 checkpoint
        for old in ['ncg_gmac_d32_N32.pt', 'ncg_gmac_d32_N64.pt', 'ncg_gmac_d32_N128.pt']:
            old_path = os.path.join(SAVE_DIR, old)
            if os.path.exists(old_path):
                try:
                    sd = torch.load(old_path, map_location='cpu', weights_only=False)
                    model.load_state_dict(sd, strict=False)
                    print(f'Loaded old checkpoint: {old_path}', flush=True)
                except:
                    print(f'Failed to load {old_path}, starting fresh', flush=True)
                break
        else:
            print(f'Starting from scratch', flush=True)

    results = {}

    for stage in STAGES:
        N = stage['N']
        elapsed_hr = (time.time() - global_start) / 3600
        if elapsed_hr >= TOTAL_HOURS:
            print(f'\nTime budget exhausted ({elapsed_hr:.1f}hr)', flush=True)
            break

        ref = SC_REF[N]
        ku, kv = ref['ku'], ref['kv']
        sc_bler = ref['sc_bler']
        b = make_path(N, N // 2)
        Au, Av, fu, fv = load_design(N, ku, kv)

        stage_hours = min(stage['hours'], TOTAL_HOURS - elapsed_hr)
        stage_budget = stage_hours * 3600
        batch = stage['batch']
        lr = stage['lr']
        eval_cw = stage['eval_cw']
        eval_every = stage['eval_every']

        # Estimate iters
        # Rough: ~1ms per tree node per batch sample
        # N=32: ~64 nodes, batch=32 → ~2ms/iter → 1.8M iters/hr
        # N=256: ~512 nodes, batch=4 → ~2ms/iter → 1.8M iters/hr... no
        # Better: measure
        print(f'\n{"="*60}', flush=True)
        print(f'Stage: N={N}, ku={ku}, kv={kv}, SC={sc_bler}', flush=True)
        print(f'Budget: {stage_hours:.1f}hr, batch={batch}, lr={lr}', flush=True)

        # Benchmark one iter
        model.train()
        uf = np.zeros((batch, N), dtype=int); vf = np.zeros((batch, N), dtype=int)
        for p in Au: uf[:, p-1] = np.random.randint(0, 2, batch)
        for p in Av: vf[:, p-1] = np.random.randint(0, 2, batch)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        t_bench = time.time()
        logits, targets, _, _ = model(zf, b, fu, fv,
            u_true=torch.from_numpy(uf).float(), v_true=torch.from_numpy(vf).float())
        if logits:
            loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4), torch.stack(targets).reshape(-1))
            loss.backward()
        ms_per_iter = (time.time() - t_bench) * 1000
        total_iters = int(stage_budget / (ms_per_iter / 1000))
        total_iters = max(total_iters, 5000)  # minimum 5K iters

        print(f'{ms_per_iter:.0f} ms/iter → {total_iters} iters in {stage_hours:.1f}hr', flush=True)

        init_bler = evaluate(model, channel, N, b, Au, Av, fu, fv, min(eval_cw, 1000))
        print(f'Initial BLER: {init_bler:.4f} (SC={sc_bler})', flush=True)
        print(f'{"="*60}', flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        rng = np.random.default_rng(42 + N)
        t0 = time.time()
        losses = []
        best_bler = init_bler

        model.train()
        for it in range(1, total_iters + 1):
            # Check time
            if time.time() - global_start > TOTAL_HOURS * 3600:
                print(f'  Total budget exhausted at iter {it}', flush=True)
                break

            lr_now = get_lr(it, total_iters, lr)
            for pg in opt.param_groups:
                pg['lr'] = lr_now

            uf = np.zeros((batch, N), dtype=int); vf = np.zeros((batch, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, batch)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, batch)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            logits, targets, _, _ = model(zf, b, fu, fv,
                u_true=torch.from_numpy(uf).float(),
                v_true=torch.from_numpy(vf).float())

            if logits:
                loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4),
                                       torch.stack(targets).reshape(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

            # Save latest every 10K iters
            if it % 10000 == 0:
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'd32_30hr_latest.pt'))

            if it % eval_every == 0:
                elapsed = time.time() - t0
                total_elapsed = time.time() - global_start
                avg_loss = np.mean(losses[-500:])
                bler = evaluate(model, channel, N, b, Au, Av, fu, fv, eval_cw)

                improved = ''
                if bler < best_bler:
                    best_bler = bler
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'd32_30hr_N{N}_best.pt'))
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'd32_30hr_best.pt'))
                    improved = ' *BEST*'

                ratio = bler / max(sc_bler, 1e-8)
                msg = (f'  [{it:>6}/{total_iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                       f'(best={best_bler:.4f}, SC={sc_bler}, ratio={ratio:.2f}x) '
                       f'{elapsed/60:.0f}min total={total_elapsed/60:.0f}min '
                       f'lr={lr_now:.1e}{improved}')
                print(msg, flush=True)
                with open(LOG_FILE, 'a') as f:
                    f.write(msg + '\n')

        # Stage summary
        stage_time = time.time() - t0
        results[str(N)] = {
            'N': N, 'ku': ku, 'kv': kv,
            'best_bler': best_bler, 'sc_bler': sc_bler,
            'ratio': best_bler / max(sc_bler, 1e-8),
            'iters': it, 'time_hr': stage_time / 3600,
        }
        print(f'\n  N={N} DONE: best={best_bler:.4f} (SC={sc_bler}, '
              f'ratio={results[str(N)]["ratio"]:.2f}x) {stage_time/3600:.1f}hr', flush=True)

        # Save results
        with open(os.path.join(os.path.dirname(__file__), 'train_d32_30hr_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        gc.collect()

    # Final summary
    total_time = time.time() - global_start
    print(f'\n{"="*60}', flush=True)
    print(f'd=32 CAMPAIGN COMPLETE: {total_time/3600:.1f} hours', flush=True)
    print(f'{"="*60}', flush=True)
    for k, r in sorted(results.items(), key=lambda x: int(x[0])):
        print(f'  N={r["N"]:>5}: best={r["best_bler"]:.4f} '
              f'(SC={r["sc_bler"]}, ratio={r["ratio"]:.2f}x, '
              f'{r["iters"]} iters, {r["time_hr"]:.1f}hr)', flush=True)

    # Compare with d=16
    d16_best = {32: 0.046, 64: 0.026, 128: 0.017, 256: 0.015}
    print(f'\n  d=16 vs d=32 comparison:', flush=True)
    for N_cmp in [32, 64, 128, 256]:
        if str(N_cmp) in results:
            d32 = results[str(N_cmp)]['best_bler']
            d16 = d16_best.get(N_cmp, '?')
            sc = SC_REF[N_cmp]['sc_bler']
            print(f'    N={N_cmp}: d16={d16} d32={d32:.4f} SC={sc}', flush=True)


if __name__ == '__main__':
    main()
