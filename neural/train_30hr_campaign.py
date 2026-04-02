#!/usr/bin/env python3
"""
train_30hr_campaign.py — 30-hour training campaign.

Phase 1 (first ~10hr): Scheduled sampling at N=256 (80% CPU)
  + Regular training at N=512 (20% CPU, separate process)

If N=256 reaches BLER < 0.010:
  Phase 2: Switch to scheduled sampling at N=512

Saves checkpoints every 5K iters. Logs progress.
"""
import sys, os, math, time, json, signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# Compile C++ extension
print('Compiling C++ extension...', flush=True)
fast_walk = load(name='fast_tree_walk',
                 sources=[os.path.join(os.path.dirname(__file__), 'csrc', 'fast_tree_walk.cpp')],
                 extra_cflags=['-O3', '-std=c++17'], verbose=False)
print('C++ extension ready.', flush=True)

SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
TOTAL_HOURS = 30


class SimpleMLP_Gmac(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = 16
        self.z_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 32), torch.nn.ELU(), torch.nn.Linear(32, 16))
        self.tree = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)


def build_schedule_flat(N, b, fu, fv):
    schedule = []; dec_head = 1; i_u = 0; i_v = 0
    for step in range(2*N):
        gamma = b[step]
        if gamma == 0: i_u += 1; i_t = i_u; is_frozen = int(i_t in fu)
        else: i_v += 1; i_t = i_v; is_frozen = int(i_t in fv)
        leaf_edge = i_t + N - 1; target_vtx = leaf_edge >> 1
        pu, pd = [], []
        c, t = dec_head, target_vtx
        while c != t:
            if c > t: c >>= 1; pu.append(c)
            else: pd.append(t); t >>= 1
        pd.reverse()
        for beta in pu: schedule.extend([2, dec_head]); dec_head = beta
        for beta in pd:
            parent = beta >> 1
            if beta & 1 == 0: schedule.extend([0, parent])
            else: schedule.extend([1, parent])
            dec_head = beta
        op_type = 3 if (leaf_edge & 1 == 0) else 4
        schedule.extend([op_type, target_vtx, leaf_edge, i_t, gamma, is_frozen])
    return schedule


def get_mlp_weights(mlp):
    return [mlp[0].weight, mlp[0].bias, mlp[2].weight, mlp[2].bias, mlp[4].weight, mlp[4].bias]


def get_all_weights(model):
    t = model.tree
    return (get_mlp_weights(t.calc_left_nn), get_mlp_weights(t.calc_right_nn),
            [t.calc_parent_nn.gate_net[0].weight, t.calc_parent_nn.gate_net[0].bias,
             t.calc_parent_nn.gate_net[2].weight, t.calc_parent_nn.gate_net[2].bias],
            get_mlp_weights(t.calc_parent_nn.candidate_net),
            [t.parent_second_nn[0].weight, t.parent_second_nn[0].bias],
            get_mlp_weights(t.emb2logits), get_mlp_weights(t.logits2emb))


def get_lr(it, total_iters, base_lr):
    warmup = 1000
    if it < warmup: return base_lr * it / warmup
    progress = (it - warmup) / max(1, total_iters - warmup)
    return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate(model, channel, N, b, Au, Av, fu, fv, n_cw):
    model.eval()
    n = int(math.log2(N))
    br = torch.from_numpy(bit_reversal_perm(n)).long()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(4, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            root = model.z_encoder(zf.unsqueeze(-1))[:, br]
            _, _, uh, vh, _ = model.tree(z=None, b=b, frozen_u=fu, frozen_v=fv, root_emb=root)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


def forward_scheduled(model, z, b, fu, fv, u_true, v_true, sample_rate, br):
    """Forward with scheduled sampling."""
    B, N = z.shape; d = 16
    root = model.z_encoder(z.unsqueeze(-1))[:, br]

    edge_data = [None]*(2*N); edge_data[1] = root
    no_info = model.tree.no_info_emb.unsqueeze(0).unsqueeze(0)
    for beta in range(2, 2*N):
        lev = beta.bit_length()-1; size = N >> lev
        edge_data[beta] = no_info.expand(B, size, d).clone()

    dec_head = 1; u_hat = {}; v_hat = {}; i_u = 0; i_v = 0
    all_logits = []; all_targets = []
    LH = math.log(0.5); LQ = math.log(0.25)

    for step in range(2*N):
        gamma = b[step]
        if gamma == 0: i_u += 1; i_t = i_u; fdict = fu
        else: i_v += 1; i_t = i_v; fdict = fv
        leaf_edge = i_t+N-1; target_vtx = leaf_edge >> 1

        pu, pd = [], []
        c, t2 = dec_head, target_vtx
        while c != t2:
            if c > t2: c >>= 1; pu.append(c)
            else: pd.append(t2); t2 >>= 1
        pd.reverse()

        for beta in pu:
            curr = dec_head
            left = edge_data[2*curr]; right = edge_data[2*curr+1]
            pf = model.tree.calc_parent_nn(left, right)
            ps = model.tree.parent_second_nn(right)
            edge_data[curr] = torch.cat([pf, ps], dim=1)
            dec_head = beta

        for beta in pd:
            parent = beta >> 1
            if beta & 1 == 0:
                p = edge_data[parent]; r = edge_data[2*parent+1]; l = r.shape[1]
                inp = torch.cat([p[:,:l], p[:,l:], r], dim=-1)
                edge_data[2*parent] = model.tree.calc_left_nn(inp)
            else:
                p = edge_data[parent]; le = edge_data[2*parent]; l = le.shape[1]
                inp = torch.cat([p[:,:l], p[:,l:], le], dim=-1)
                edge_data[2*parent+1] = model.tree.calc_right_nn(inp)
            dec_head = beta

        temp = edge_data[leaf_edge][:, 0].clone()
        if leaf_edge & 1 == 0:
            p = edge_data[target_vtx]; r = edge_data[2*target_vtx+1]; l = r.shape[1]
            inp = torch.cat([p[:,:l], p[:,l:], r], dim=-1)
            edge_data[2*target_vtx] = model.tree.calc_left_nn(inp)
        else:
            p = edge_data[target_vtx]; le = edge_data[2*target_vtx]; l = le.shape[1]
            inp = torch.cat([p[:,:l], p[:,l:], le], dim=-1)
            edge_data[2*target_vtx+1] = model.tree.calc_right_nn(inp)

        top_down = edge_data[leaf_edge][:, 0]
        combined = top_down + temp
        logits = model.tree.emb2logits(combined)

        if i_t in fdict:
            bit = torch.zeros(B)
        else:
            all_logits.append(logits)
            target = (u_true[:, i_t-1]*2 + v_true[:, i_t-1]).long()
            all_targets.append(target)
            if np.random.random() < sample_rate:
                with torch.no_grad():
                    if gamma == 0:
                        p0 = torch.logsumexp(logits[:,:2], dim=1)
                        p1 = torch.logsumexp(logits[:,2:], dim=1)
                    else:
                        p0 = torch.logsumexp(logits[:,[0,2]], dim=1)
                        p1 = torch.logsumexp(logits[:,[1,3]], dim=1)
                    bit = (p1 > p0).float()
            else:
                bit = u_true[:, i_t-1] if gamma == 0 else v_true[:, i_t-1]

        if gamma == 0: u_hat[i_t] = bit
        else: v_hat[i_t] = bit

        lp = torch.full((B, 4), -30.0)
        uv = u_hat.get(i_t), v_hat.get(i_t)
        if uv[0] is not None and uv[1] is not None:
            idx = (uv[0].long()*2+uv[1].long()).unsqueeze(1); lp.scatter_(1, idx, 0.0)
        elif uv[0] is not None:
            lp.scatter_(1, (uv[0].long()*2).unsqueeze(1), LH)
            lp.scatter_(1, (uv[0].long()*2+1).unsqueeze(1), LH)
        elif uv[1] is not None:
            lp.scatter_(1, uv[1].long().unsqueeze(1), LH)
            lp.scatter_(1, (uv[1].long()+2).unsqueeze(1), LH)
        else: lp.fill_(LQ)
        edge_data[leaf_edge] = model.tree.logits2emb(lp).unsqueeze(1)

    return all_logits, all_targets


def forward_cpp(model, z, schedule_flat, frozen_u_mask, frozen_v_mask, u_true, v_true, br):
    """Regular forward with C++ acceleration."""
    root = model.z_encoder(z.unsqueeze(-1))[:, br]
    cl_w, cr_w, pg_w, pc_w, ps_w, el_w, le_w = get_all_weights(model)
    result = fast_walk.tree_walk_forward(root, schedule_flat, model.tree.no_info_emb,
        cl_w, cr_w, pg_w, pc_w, ps_w, el_w, le_w, u_true, v_true, frozen_u_mask, frozen_v_mask)
    return result[0], result[1]


def train_phase(model, channel, N, ku, kv, mode, total_iters, lr, batch,
                eval_cw, log_file, ckpt_best, ckpt_latest, sample_rate_max=0.3):
    """
    Train one phase.
    mode: 'regular' or 'scheduled'
    Returns best_bler.
    """
    n = int(math.log2(N))
    Au, Av, fu, fv, _, _, _ = design_from_file(
        os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz'), n, ku, kv)
    b = make_path(N, N // 2)
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    schedule_flat = build_schedule_flat(N, b, fu, fv)
    frozen_u_mask = torch.tensor([int(i+1 in fu) for i in range(N)], dtype=torch.bool)
    frozen_v_mask = torch.tensor([int(i+1 in fv) for i in range(N)], dtype=torch.bool)

    sc_bler = {256: 0.005, 512: 0.001}.get(N, 0.01)

    init_bler = evaluate(model, channel, N, b, Au, Av, fu, fv, eval_cw)
    print(f'\n{"="*60}', flush=True)
    print(f'Phase: N={N}, mode={mode}, iters={total_iters}', flush=True)
    print(f'Initial BLER: {init_bler:.4f} (SC={sc_bler})', flush=True)
    print(f'batch={batch}, lr={lr}, sample_rate_max={sample_rate_max}', flush=True)
    print(f'{"="*60}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    best_bler = init_bler

    model.train()
    for it in range(1, total_iters + 1):
        lr_now = get_lr(it, total_iters, lr)
        for pg in opt.param_groups: pg['lr'] = lr_now

        uf = np.zeros((batch, N), dtype=int); vf = np.zeros((batch, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, batch)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, batch)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
        ut = torch.from_numpy(uf).float(); vt = torch.from_numpy(vf).float()

        if mode == 'scheduled':
            sample_rate = sample_rate_max * min(1.0, it / 10000)
            logits_list, targets_list = forward_scheduled(
                model, zf, b, fu, fv, ut, vt, sample_rate, br)
            if logits_list:
                loss = F.cross_entropy(torch.stack(logits_list).reshape(-1, 4),
                                       torch.stack(targets_list).reshape(-1))
            else:
                continue
        else:
            logits_all, targets_all = forward_cpp(
                model, zf, schedule_flat, frozen_u_mask, frozen_v_mask, ut, vt, br)
            if logits_all.numel() > 0:
                loss = F.cross_entropy(logits_all.reshape(-1, 4), targets_all.reshape(-1))
            else:
                continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % 5000 == 0:
            torch.save(model.state_dict(), ckpt_latest)

        if it % 5000 == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-500:])
            bler = evaluate(model, channel, N, b, Au, Av, fu, fv, eval_cw)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), ckpt_best)
                improved = ' *BEST*'

            ratio = bler / max(sc_bler, 1e-8)
            sr = f' sr={sample_rate:.2f}' if mode == 'scheduled' else ''
            msg = (f'[{it:>6}/{total_iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}, SC={sc_bler}, ratio={ratio:.1f}x) '
                   f'{elapsed/60:.0f}min lr={lr_now:.1e}{sr}{improved}')
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

            # Check time budget
            total_elapsed = time.time() - GLOBAL_START
            if total_elapsed > TOTAL_HOURS * 3600:
                print(f'Time budget exhausted ({total_elapsed/3600:.1f}hr)', flush=True)
                break

    return best_bler


GLOBAL_START = time.time()


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    log_file = os.path.join(os.path.dirname(__file__), 'train_30hr_campaign.log')

    print(f'30-Hour Training Campaign', flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'Budget: {TOTAL_HOURS} hours', flush=True)

    # ── Phase 1: Scheduled sampling at N=256 ──
    model_256 = SimpleMLP_Gmac()
    ckpt_256 = os.path.join(SAVE_DIR, 'n256_long_best.pt')
    if not os.path.exists(ckpt_256):
        ckpt_256 = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N256.pt')
    sd = torch.load(ckpt_256, map_location='cpu', weights_only=False)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model_256.load_state_dict(fixed, strict=False)
    print(f'Loaded N=256 checkpoint: {ckpt_256}', flush=True)

    best_256 = train_phase(
        model_256, channel, N=256, ku=123, kv=123,
        mode='scheduled', total_iters=50000, lr=1e-5, batch=4,
        eval_cw=1000,
        log_file=log_file,
        ckpt_best=os.path.join(SAVE_DIR, 'campaign_n256_sched_best.pt'),
        ckpt_latest=os.path.join(SAVE_DIR, 'campaign_n256_sched_latest.pt'),
        sample_rate_max=0.3
    )

    elapsed = time.time() - GLOBAL_START
    print(f'\nPhase 1 done: N=256 scheduled best={best_256:.4f}, {elapsed/3600:.1f}hr', flush=True)

    # ── Phase 2: If N=256 improved, do scheduled at N=512 ──
    # Otherwise do regular training at N=512
    if elapsed < TOTAL_HOURS * 3600:
        model_512 = SimpleMLP_Gmac()
        ckpt_512 = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N512.pt')
        if os.path.exists(os.path.join(SAVE_DIR, 'n512_long_best.pt')):
            ckpt_512 = os.path.join(SAVE_DIR, 'n512_long_best.pt')
        sd512 = torch.load(ckpt_512, map_location='cpu', weights_only=False)
        fixed512 = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd512.items()}
        model_512.load_state_dict(fixed512, strict=False)
        print(f'Loaded N=512 checkpoint: {ckpt_512}', flush=True)

        remaining_hours = TOTAL_HOURS - elapsed/3600
        remaining_iters = int(remaining_hours * 3600 / 1.5)  # ~1.5s per iter at N=512
        remaining_iters = min(remaining_iters, 50000)

        if best_256 < 0.010:
            print(f'\nN=256 reached {best_256:.4f} < 0.010 — using scheduled sampling for N=512', flush=True)
            mode_512 = 'scheduled'
        else:
            print(f'\nN=256 at {best_256:.4f} >= 0.010 — using regular training for N=512', flush=True)
            mode_512 = 'regular'

        best_512 = train_phase(
            model_512, channel, N=512, ku=246, kv=246,
            mode=mode_512, total_iters=remaining_iters, lr=3e-5, batch=2,
            eval_cw=500,
            log_file=log_file,
            ckpt_best=os.path.join(SAVE_DIR, 'campaign_n512_best.pt'),
            ckpt_latest=os.path.join(SAVE_DIR, 'campaign_n512_latest.pt'),
            sample_rate_max=0.3
        )

        print(f'\nPhase 2 done: N=512 best={best_512:.4f}', flush=True)

    # Final summary
    total_time = time.time() - GLOBAL_START
    print(f'\n{"="*60}', flush=True)
    print(f'CAMPAIGN COMPLETE: {total_time/3600:.1f} hours', flush=True)
    print(f'N=256 scheduled best: {best_256:.4f} (SC=0.005)', flush=True)
    if 'best_512' in dir():
        print(f'N=512 best: {best_512:.4f} (SC=0.001)', flush=True)
    print(f'{"="*60}', flush=True)


if __name__ == '__main__':
    main()
