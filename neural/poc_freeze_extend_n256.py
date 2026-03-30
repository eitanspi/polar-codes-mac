#!/usr/bin/env python3
"""
poc_freeze_extend_n256.py — Freeze & Extend to N=256.

Takes the N=128 freeze-extend model (levels 1-5 shared frozen + level 6 trained),
freezes everything, adds NEW level-7 CalcLeft/CalcRight, trains only those.

Chain: shared(1-5) → trained_level6 → NEW trainable level7
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
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder, _make_mlp, NeuralCalcParent

D = 16
HIDDEN = 64
N_LAYERS = 2
Z_HIDDEN = 32

N_VAL = 256
KU = 123
KV = 123
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
SC_BLER = 0.005

BATCH = 8
LR = 1e-4
TOTAL_ITERS = 100000
EVAL_EVERY = 5000
EVAL_CW = 2000
WARMUP_ITERS = 1000

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'poc_freeze_extend_n256.log')


class FreezeExtendN256(nn.Module):
    """
    Two-level freeze-extend: shared(frozen) + level6(frozen from N=128 training) + level7(NEW trainable).
    """
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_hidden=Z_HIDDEN):
        super().__init__()
        self.d = d

        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

        # Level 6 MLPs (from N=128 training, will be loaded and frozen)
        self.calc_left_6 = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_6 = _make_mlp(3 * d, hidden, d, n_layers)

        # Level 7 MLPs (NEW, trainable)
        self.calc_left_7 = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_7 = _make_mlp(3 * d, hidden, d, n_layers)

    def freeze_all_except_level7(self):
        for p in self.z_encoder.parameters():
            p.requires_grad = False
        for p in self.tree.parameters():
            p.requires_grad = False
        for p in self.calc_left_6.parameters():
            p.requires_grad = False
        for p in self.calc_right_6.parameters():
            p.requires_grad = False
        for p in self.calc_left_7.parameters():
            p.requires_grad = True
        for p in self.calc_right_7.parameters():
            p.requires_grad = True

    def load_from_n128(self, n128_ckpt_path, shared_ckpt_path):
        """Load shared weights + level-6 weights from N=128 freeze-extend model."""
        # First load the shared base from the original checkpoint
        state = torch.load(shared_ckpt_path, map_location='cpu', weights_only=False)
        own_state = self.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
                loaded += 1
        self.load_state_dict(own_state, strict=False)
        print(f'  Loaded {loaded} shared params from base checkpoint', flush=True)

        # Now load the N=128 freeze-extend checkpoint for level-6 MLPs
        if os.path.exists(n128_ckpt_path):
            n128_state = torch.load(n128_ckpt_path, map_location='cpu', weights_only=False)
            # Map calc_left_new -> calc_left_6, calc_right_new -> calc_right_6
            for k, v in n128_state.items():
                mapped = k.replace('calc_left_new', 'calc_left_6').replace('calc_right_new', 'calc_right_6')
                if mapped in own_state and own_state[mapped].shape == v.shape:
                    own_state[mapped] = v
            self.load_state_dict(own_state, strict=False)
            print(f'  Loaded level-6 MLPs from N=128 checkpoint', flush=True)
        else:
            # If no N=128 checkpoint, init level-6 from shared
            self.calc_left_6.load_state_dict(self.tree.calc_left_nn.state_dict())
            self.calc_right_6.load_state_dict(self.tree.calc_right_nn.state_dict())
            print(f'  No N=128 checkpoint found, init level-6 from shared', flush=True)

        # Initialize level-7 from level-6 (warm start)
        self.calc_left_7.load_state_dict(self.calc_left_6.state_dict())
        self.calc_right_7.load_state_dict(self.calc_right_6.state_dict())
        print(f'  Initialized level-7 MLPs from level-6 weights', flush=True)

    def count_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def _calc_left(self, beta, edge_data):
        level = beta.bit_length() - 1
        parent = edge_data[beta]
        right = edge_data[2 * beta + 1]
        l = right.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], right], dim=-1)
        if level == 7:
            edge_data[2 * beta] = self.calc_left_7(inp)
        elif level == 6:
            edge_data[2 * beta] = self.calc_left_6(inp)
        else:
            edge_data[2 * beta] = self.tree.calc_left_nn(inp)

    def _calc_right(self, beta, edge_data):
        level = beta.bit_length() - 1
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], left], dim=-1)
        if level == 7:
            edge_data[2 * beta + 1] = self.calc_right_7(inp)
        elif level == 6:
            edge_data[2 * beta + 1] = self.calc_right_6(inp)
        else:
            edge_data[2 * beta + 1] = self.tree.calc_right_nn(inp)

    def _calc_parent(self, beta, edge_data):
        left = edge_data[2 * beta]
        right = edge_data[2 * beta + 1]
        parent_first = self.tree.calc_parent_nn(left, right)
        parent_second = self.tree.parent_second_nn(right)
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
        return self.tree.logits2emb(lp)

    def forward(self, z, b, frozen_u, frozen_v, u_true=None, v_true=None):
        B, N_val = z.shape
        device = z.device
        d = self.d
        n = int(math.log2(N_val))

        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.z_encoder(z.unsqueeze(-1))[:, br]

        teacher = u_true is not None and v_true is not None
        edge_data = [None] * (2 * N_val)
        edge_data[1] = root

        no_info = self.tree.no_info_emb.unsqueeze(0).unsqueeze(0)
        for beta in range(2, 2 * N_val):
            level = beta.bit_length() - 1
            size = N_val >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

        dec_head = 1
        u_hat, v_hat = {}, {}
        all_logits, all_targets = [], []
        i_u, i_v = 0, 0

        for step in range(2 * N_val):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = frozen_v

            leaf_edge = i_t + N_val - 1
            target_vtx = leaf_edge >> 1
            dec_head = self._step_to(dec_head, target_vtx, edge_data)

            temp = edge_data[leaf_edge][:, 0].clone()
            if leaf_edge & 1 == 0:
                self._calc_left(target_vtx, edge_data)
            else:
                self._calc_right(target_vtx, edge_data)
            top_down = edge_data[leaf_edge][:, 0]

            combined = top_down + temp
            logits = self.tree.emb2logits(combined)

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

        dummy_loss = torch.tensor(0.0, device=device)
        return all_logits, all_targets, u_hat, v_hat, dummy_loss


def load_design(N, ku, kv):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    return design_from_file(mc_path, n, ku, kv)


def evaluate(model, channel, N, b, Au, Av, fu, fv, n_cw):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    bs = 4
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


def get_lr(it):
    if it < WARMUP_ITERS:
        return LR * it / WARMUP_ITERS
    progress = (it - WARMUP_ITERS) / max(1, TOTAL_ITERS - WARMUP_ITERS)
    return LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design(N_VAL, KU, KV)[:4]
    b = make_path(N_VAL, N_VAL // 2)

    model = FreezeExtendN256()

    # Load: shared base + level-6 from N=128 freeze-extend
    shared_ckpt = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N128.pt')
    n128_ckpt = os.path.join(SAVE_DIR, 'ncg_gmac_freeze_extend_N128.pt')
    model.load_from_n128(n128_ckpt, shared_ckpt)
    model.freeze_all_except_level7()

    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)

    print(f'{"="*60}', flush=True)
    print(f'Freeze & Extend N=256', flush=True)
    print(f'Total params: {total_params:,}', flush=True)
    print(f'Trainable params: {trainable_params:,} (level-7 MLPs only)', flush=True)

    # Initial eval
    init_bler = evaluate(model, channel, N_VAL, b, Au, Av, fu, fv, EVAL_CW)
    print(f'Initial BLER: {init_bler:.4f} (SC={SC_BLER})', flush=True)
    print(f'{"="*60}', flush=True)

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    best_bler = init_bler
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        lr_now = get_lr(it)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        uf = np.zeros((BATCH, N_VAL), dtype=int); vf = np.zeros((BATCH, N_VAL), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        logits, targets, _, _, _ = model(zf, b, fu, fv,
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

        if it % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-min(len(losses), 500):])
            bler = evaluate(model, channel, N_VAL, b, Au, Av, fu, fv, EVAL_CW)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(SAVE_DIR, 'freeze_extend_N256_best.pt'))
                improved = ' *BEST*'

            ratio = bler / max(SC_BLER, 1e-8)
            msg = (f'[{it:>6}/{TOTAL_ITERS}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}, SC={SC_BLER}, ratio={ratio:.1f}x) '
                   f'{elapsed/60:.0f}min lr={lr_now:.1e}{improved}')
            print(msg, flush=True)
            with open(LOG_FILE, 'a') as f:
                f.write(msg + '\n')

    elapsed = time.time() - t0
    print(f'\nDONE: best={best_bler:.4f} (SC={SC_BLER}), {elapsed/60:.0f}min', flush=True)


if __name__ == '__main__':
    main()
