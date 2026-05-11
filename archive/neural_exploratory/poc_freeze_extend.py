#!/usr/bin/env python3
"""
poc_freeze_extend.py — Freeze & Extend per-level training for GMAC neural decoder.

CONCEPT:
  The d=16 shared model works at N=64 (BLER=0.026, nearly matches SC=0.025).
  For N=128, freeze the shared CalcLeft/CalcRight weights and add NEW
  level-7 specific CalcLeft_7 and CalcRight_7 MLPs. Train only these new MLPs.

ARCHITECTURE:
  - z_encoder: shared from checkpoint (frozen)
  - calc_left_nn: shared from checkpoint (FROZEN)
  - calc_right_nn: shared from checkpoint (FROZEN)
  - calc_left_7: NEW MLP(3*d, hidden, d, 2 layers) — trained
  - calc_right_7: NEW MLP(3*d, hidden, d, 2 layers) — trained
  - calc_parent_nn, parent_second_nn: shared from checkpoint (frozen)
  - emb2logits, logits2emb: shared from checkpoint (frozen)

During the tree walk, when beta.bit_length()-1 == 7 (level 7), use
calc_left_7 / calc_right_7 instead of the shared ones.
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


# ─── Config ────────────────────────────────────────────────────────────────

D = 16
HIDDEN = 64
N_LAYERS = 2
Z_HIDDEN = 32

N = 128
KU = 62
KV = 62
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
SC_BLER = 0.016

BATCH = 16
LR = 1e-4
TOTAL_ITERS = 60000
EVAL_EVERY = 5000
EVAL_CW = 3000
WARMUP_ITERS = 1000

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
CKPT_PATH = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N128.pt')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'poc_freeze_extend.log')

# Target level for the new MLPs.
# For N=128 (n=7): vertices at level 6 (indices 64..127) are NEW compared to N=64.
# These vertices are the parents of leaf edges (128..255).
# beta.bit_length()-1 == 6 for beta in [64..127].
TARGET_LEVEL = 6


# ─── Freeze & Extend Model ─────────────────────────────────────────────────

class FreezeExtendGmacDecoder(nn.Module):
    """
    GMAC decoder that uses frozen shared CalcLeft/CalcRight from a pretrained
    checkpoint plus NEW trainable level-specific MLPs for the target level.
    """

    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_hidden=Z_HIDDEN,
                 target_level=TARGET_LEVEL):
        super().__init__()
        self.d = d
        self.target_level = target_level

        # Channel encoder
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )

        # Shared tree ops (will be loaded from checkpoint & FROZEN)
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

        # NEW level-specific MLPs (trainable)
        self.calc_left_new = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_new = _make_mlp(3 * d, hidden, d, n_layers)

    def freeze_shared(self):
        """Freeze all shared weights (z_encoder + tree). Only new MLPs train."""
        for p in self.z_encoder.parameters():
            p.requires_grad = False
        for p in self.tree.parameters():
            p.requires_grad = False
        # New MLPs remain trainable
        for p in self.calc_left_new.parameters():
            p.requires_grad = True
        for p in self.calc_right_new.parameters():
            p.requires_grad = True

    def unfreeze_finetune(self):
        """Optionally unfreeze decision head for joint fine-tuning."""
        for p in self.tree.emb2logits.parameters():
            p.requires_grad = True
        for p in self.tree.logits2emb.parameters():
            p.requires_grad = True

    def count_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def load_pretrained(self, ckpt_path):
        """Load SimpleMLP_Gmac checkpoint (tree.* and z_encoder.*)."""
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        own_state = self.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
                loaded += 1
        self.load_state_dict(own_state)
        print(f'  Loaded {loaded}/{len(state)} params from checkpoint')

        # Initialize new MLPs from shared ones (warm start)
        self.calc_left_new.load_state_dict(self.tree.calc_left_nn.state_dict())
        self.calc_right_new.load_state_dict(self.tree.calc_right_nn.state_dict())
        print(f'  Initialized new level-{self.target_level} MLPs from shared weights')

    # ── Tree operations with level dispatch ──────────────────────────────

    def _calc_left(self, beta, edge_data):
        level = beta.bit_length() - 1
        parent = edge_data[beta]
        right = edge_data[2 * beta + 1]
        l = right.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], right], dim=-1)
        if level == self.target_level:
            edge_data[2 * beta] = self.calc_left_new(inp)
        else:
            edge_data[2 * beta] = self.tree.calc_left_nn(inp)

    def _calc_right(self, beta, edge_data):
        level = beta.bit_length() - 1
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]
        inp = torch.cat([parent[:, :l], parent[:, l:], left], dim=-1)
        if level == self.target_level:
            edge_data[2 * beta + 1] = self.calc_right_new(inp)
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

    def forward(self, z, b, fu, fv, u_true=None, v_true=None):
        B, NN = z.shape
        device = z.device
        d = self.d
        n = int(math.log2(NN))

        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.z_encoder(z.unsqueeze(-1))[:, br]

        teacher = u_true is not None and v_true is not None

        edge_data = [None] * (2 * NN)
        edge_data[1] = root

        no_info = self.tree.no_info_emb.unsqueeze(0).unsqueeze(0)
        for beta in range(2, 2 * NN):
            level = beta.bit_length() - 1
            size = NN >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

        dec_head = 1
        u_hat, v_hat = {}, {}
        all_logits, all_targets = [], []
        i_u, i_v = 0, 0

        for step in range(2 * NN):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = fu
            else:
                i_v += 1; i_t = i_v; fdict = fv

            leaf_edge = i_t + NN - 1
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

        return all_logits, all_targets, u_hat, v_hat


# ─── Helpers ────────────────────────────────────────────────────────────────

def load_design_n128():
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, KU, KV)
    return Au, Av, fu, fv


def evaluate(model, channel, b, Au, Av, fu, fv, n_cw):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    bs = 8
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
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


def get_cosine_lr(it, total, base_lr, warmup=1000):
    """Cosine LR with linear warmup (no restarts)."""
    if it < warmup:
        return base_lr * it / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    log = open(LOG_FILE, 'w')
    def plog(msg):
        print(msg, flush=True)
        log.write(msg + '\n'); log.flush()

    plog('=' * 60)
    plog('Freeze & Extend POC — GMAC Neural Decoder')
    plog(f'N={N}, ku={KU}, kv={KV}, SNR={SNR_DB}dB, SC_BLER={SC_BLER}')
    plog(f'd={D}, hidden={HIDDEN}, target_level={TARGET_LEVEL}')
    plog(f'batch={BATCH}, lr={LR}, iters={TOTAL_ITERS}')
    plog(f'Strategy: freeze shared, train level-{TARGET_LEVEL} MLPs only')
    plog('=' * 60)

    channel = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)
    Au, Av, fu, fv = load_design_n128()

    # Build model
    model = FreezeExtendGmacDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS,
                                     z_hidden=Z_HIDDEN, target_level=TARGET_LEVEL)

    # Try to resume from saved freeze-extend checkpoint, else load base
    fe_ckpt = os.path.join(SAVE_DIR, 'ncg_gmac_freeze_extend_N128.pt')
    if os.path.exists(fe_ckpt):
        state = torch.load(fe_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(state)
        plog(f'  Resumed from freeze-extend checkpoint')
    else:
        model.load_pretrained(CKPT_PATH)
    model.freeze_shared()

    total_params = model.count_parameters(trainable_only=False)
    train_params = model.count_parameters(trainable_only=True)
    plog(f'Total params: {total_params:,}')
    plog(f'Trainable params: {train_params:,} (new level-{TARGET_LEVEL} MLPs)')

    # Initial eval with frozen-only model (shared weights at all levels)
    init_bler = evaluate(model, channel, b, Au, Av, fu, fv, min(EVAL_CW, 2000))
    plog(f'Initial BLER (shared weights at all levels): {init_bler:.4f}')
    plog(f'SC reference: {SC_BLER}')
    plog(f'Current best NN: 0.019')

    # Optimizer: only trainable params
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR, weight_decay=1e-5)

    rng = np.random.default_rng()
    best_bler = init_bler
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    losses = []
    t0 = time.time()

    # Phase 1: Train only new level-7 MLPs
    PHASE1_ITERS = TOTAL_ITERS * 2 // 3  # 2/3 of budget for phase 1

    plog(f'\n--- Phase 1: Train level-{TARGET_LEVEL} MLPs only ({PHASE1_ITERS} iters) ---')

    for it in range(1, TOTAL_ITERS + 1):
        # Phase transition: unfreeze decision head for fine-tuning
        if it == PHASE1_ITERS + 1:
            model.unfreeze_finetune()
            new_train = model.count_parameters(trainable_only=True)
            plog(f'\n--- Phase 2: Unfreeze decision head ({TOTAL_ITERS - PHASE1_ITERS} iters) ---')
            plog(f'  Trainable params now: {new_train:,}')
            # Reset optimizer with all trainable params
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=LR * 0.3, weight_decay=1e-5)

        # LR schedule
        if it <= PHASE1_ITERS:
            lr_now = get_cosine_lr(it, PHASE1_ITERS, LR, warmup=WARMUP_ITERS)
        else:
            lr_now = get_cosine_lr(it - PHASE1_ITERS, TOTAL_ITERS - PHASE1_ITERS,
                                    LR * 0.3, warmup=500)
        for pg in opt.param_groups:
            pg['lr'] = lr_now

        # Generate batch
        u = np.zeros((BATCH, N), dtype=int)
        v = np.zeros((BATCH, N), dtype=int)
        for p in Au: u[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: v[:, p-1] = rng.integers(0, 2, BATCH)
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
            losses.append(loss.item())

        if it % EVAL_EVERY == 0 or it == 1:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-min(len(losses), 500):])
            bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                torch.save(best_state, os.path.join(SAVE_DIR, 'ncg_gmac_freeze_extend_N128.pt'))
                improved = ' *BEST*'

            ratio = bler / max(SC_BLER, 1e-8)
            phase = 'P1' if it <= PHASE1_ITERS else 'P2'
            plog(f'  [{phase} {it:>6}/{TOTAL_ITERS}] loss={avg_loss:.4f} BLER={bler:.4f} '
                 f'(best={best_bler:.4f}, SC={SC_BLER}, ratio={ratio:.2f}x) '
                 f'{elapsed/60:.1f}min lr={lr_now:.1e}{improved}')

    # Final eval
    model.load_state_dict(best_state)
    final_bler = evaluate(model, channel, b, Au, Av, fu, fv, EVAL_CW * 2)
    total_time = time.time() - t0

    plog(f'\n{"="*60}')
    plog(f'RESULTS: Freeze & Extend at N={N}')
    plog(f'  Initial BLER: {init_bler:.4f}')
    plog(f'  Best BLER:    {best_bler:.4f}')
    plog(f'  Final BLER:   {final_bler:.4f}')
    plog(f'  SC BLER:      {SC_BLER}')
    plog(f'  Ratio:        {best_bler/SC_BLER:.3f}x')
    plog(f'  Prev best NN: 0.019 (1.19x SC)')
    plog(f'  Time:         {total_time/60:.1f} min')
    plog(f'  Trainable:    {train_params:,} params (of {total_params:,} total)')
    plog(f'{"="*60}')

    log.close()


if __name__ == '__main__':
    main()
