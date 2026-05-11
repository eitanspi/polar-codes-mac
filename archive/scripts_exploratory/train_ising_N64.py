#!/usr/bin/env python3
"""
train_ising_N64.py
==================
Priority 2: Extend Ising MAC NPD to N=64 with d=16 h=100 BiGRU.
Warm-start from N=32 checkpoint if available.
Channel: IsingMAC(sigma2=0.251, p_flip=0.1)
Design: GMAC_C proxy at SNR=6dB.
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
import torch

torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels_memory_new import IsingMAC
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from polar.decoder import decode_single as gmac_decode_single
from neural.npd_memory_mac import ChainedNPD_MAC

# Config
SIGMA2 = 0.251
P_FLIP = 0.1
SNR_DB = 6
N = 64
KU = 15
KV = 29
D = 16
HIDDEN = 100
N_LAYERS = 2
GRU_LAYERS = 1
ENCODER_TYPE = 'bigru'
LR = 1e-3
BATCH = 16
ITERS_S1 = 200_000
ITERS_S2 = 50_000
EVAL_EVERY = 5000
CKPT_EVERY = 20000
EVAL_CW = 300
SEED = 42

RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_ising_mac')
os.makedirs(RESULTS_DIR, exist_ok=True)

WARMSTART_S1 = os.path.join(RESULTS_DIR, 'ising_d16_h100_s1_N32_best.pt')
WARMSTART_S2 = os.path.join(RESULTS_DIR, 'ising_d16_h100_s2_N32_best.pt')


def make_channel():
    return IsingMAC(sigma2=SIGMA2, p_flip=P_FLIP)


def load_design():
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, _, _, _ = design_from_file(path, n, KU, KV)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    fu_set = {p-1 for p in range(1, N+1) if p not in Au}
    fv_set = {p-1 for p in range(1, N+1) if p not in Av}
    fu_dict = {p: 0 for p in range(1, N+1) if p not in Au}
    fv_dict = {p: 0 for p in range(1, N+1) if p not in Av}
    return Au, Av, fu_dict, fv_dict, fu_set, fv_set


def make_batch(channel, Au, Av, batch, rng):
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au: u_msg[:, p-1] = rng.integers(0, 2, batch)
    for p in Av: v_msg[:, p-1] = rng.integers(0, 2, batch)
    x_phys = polar_encode_batch(u_msg.astype(int))
    y_phys = polar_encode_batch(v_msg.astype(int))
    z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
    return u_msg, v_msg, np.asarray(z, dtype=np.float32), x_phys, y_phys


def wilson_ci(errs, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = errs / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, centre-margin), min(1, centre+margin))


def eval_stage(model, stage_num, channel, Au, Av, frozen_set, n_cw=300, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    stage = model.stage1 if stage_num == 1 else model.stage2
    stage.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    info_pos = Au if stage_num == 1 else Av
    with torch.no_grad():
        while total < n_cw:
            actual = min(32, n_cw - total)
            u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            if stage_num == 1:
                emb = stage.encode_channel(z_t)
                target_msg = u_msg
            else:
                side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)
                emb = stage.encode_channel(z_t, side=side)
                target_msg = v_msg
            emb_npd = emb[:, br_t, :]
            hat = stage.tree.decode(emb_npd, frozen_set)
            for i in range(actual):
                if any(int(hat[i, p-1].item()) != int(target_msg[i, p-1]) for p in info_pos):
                    errs += 1
            total += actual
    stage.train()
    return errs / n_cw


def eval_chained(model, channel, Au, Av, fu_set, fv_set, n_cw=5000, batch=32, seed=777):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage1.eval()
    model.stage2.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, v_msg, z, x_phys, _ = make_batch(channel, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            emb1 = model.stage1.encode_channel(z_t)
            emb1_npd = emb1[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb1_npd, fu_set)
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)
            side = torch.from_numpy((1.0 - 2.0 * x_hat.astype(np.float32))).unsqueeze(-1)
            emb2 = model.stage2.encode_channel(z_t, side=side)
            emb2_npd = emb2[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb2_npd, fv_set)
            for i in range(actual):
                u_wrong = any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au)
                v_wrong = any(int(v_hat[i, p-1].item()) != int(v_msg[i, p-1]) for p in Av)
                if u_wrong: errs_u += 1
                if v_wrong: errs_v += 1
                if u_wrong or v_wrong: errs_total += 1
            total += actual
            if total % 1000 == 0:
                print(f'  chained [{total}/{n_cw}] errs={errs_total}', flush=True)
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u/n_cw, 'bler_v': errs_v/n_cw, 'bler_total': errs_total/n_cw,
    }


def eval_memoryless_sc(channel, Au, Av, fu_dict, fv_dict, n_cw=3000, seed=555):
    gmac = GaussianMAC(sigma2=channel.sigma2)
    b = make_path(N, N)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    for cw_idx in range(n_cw):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for p in Au: u_msg[p-1] = rng.integers(0, 2)
        for p in Av: v_msg[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u_msg[None, :])[0]
        y = polar_encode_batch(v_msg[None, :])[0]
        z = channel.sample_batch(x[None, :].astype(int), y[None, :].astype(int))
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 2: z = z[0]
        u_dec, v_dec = gmac_decode_single(N, z.tolist(), b, fu_dict, fv_dict, gmac, log_domain=True)
        u_wrong = any(u_dec[p-1] != u_msg[p-1] for p in Au)
        v_wrong = any(v_dec[p-1] != v_msg[p-1] for p in Av)
        if u_wrong: errs_u += 1
        if v_wrong: errs_v += 1
        if u_wrong or v_wrong: errs_total += 1
        if (cw_idx + 1) % 500 == 0:
            print(f'  SC [{cw_idx+1}/{n_cw}] errs={errs_total}', flush=True)
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u/n_cw, 'bler_v': errs_v/n_cw, 'bler_total': errs_total/n_cw,
    }


def try_warmstart(model, stage_num):
    """Try to warm-start from N=32 checkpoint. Loads only compatible params."""
    ckpt_path = WARMSTART_S1 if stage_num == 1 else WARMSTART_S2
    if not os.path.exists(ckpt_path):
        print(f'  No warm-start checkpoint at {ckpt_path}')
        return False
    try:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        stage = model.stage1 if stage_num == 1 else model.stage2
        # Only load tree params (architecture-independent)
        # GRU params may differ in hidden state count if N changes, but
        # for BiGRU the GRU itself is N-agnostic (processes sequences).
        # Tree MLPs are also N-agnostic. So full load should work.
        stage.load_state_dict(ckpt['state_dict'])
        print(f'  Warm-started stage {stage_num} from {ckpt_path}')
        return True
    except Exception as e:
        print(f'  Warm-start failed: {e}')
        # Try partial load
        try:
            stage = model.stage1 if stage_num == 1 else model.stage2
            sd_new = stage.state_dict()
            sd_old = ckpt['state_dict']
            loaded = 0
            for k in sd_new:
                if k in sd_old and sd_new[k].shape == sd_old[k].shape:
                    sd_new[k] = sd_old[k]
                    loaded += 1
            stage.load_state_dict(sd_new)
            print(f'  Partial warm-start: loaded {loaded}/{len(sd_new)} params')
            return True
        except Exception as e2:
            print(f'  Partial warm-start also failed: {e2}')
            return False


def main():
    t_total = time.time()
    print(f'{"="*60}')
    print(f'Ising MAC N={N}, d={D}, h={HIDDEN}, BiGRU')
    print(f'sigma2={SIGMA2}, p_flip={P_FLIP}')
    print(f'ku={KU}, kv={KV}')
    print(f'{"="*60}')

    channel = make_channel()
    Au, Av, fu_dict, fv_dict, fu_set, fv_set = load_design()

    torch.manual_seed(SEED)
    model = ChainedNPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS,
                           encoder_type=ENCODER_TYPE, gru_layers=GRU_LAYERS)
    print(f'Params: {model.count_parameters():,}')

    # Try warm-start
    try_warmstart(model, 1)

    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    log_file = os.path.join(RESULTS_DIR, f'ising_d{D}_h{HIDDEN}_N{N}.log')
    tag_s1 = f'ising_d{D}_h{HIDDEN}_s1_N{N}'
    tag_s2 = f'ising_d{D}_h{HIDDEN}_s2_N{N}'

    # Stage 1
    print(f'\n--- Stage 1 training ({ITERS_S1} iters) ---')
    opt = torch.optim.AdamW(model.stage1.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    best_bler = 1.0
    losses = []
    t0 = time.time()
    model.stage1.train()

    for it in range(1, ITERS_S1 + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, Au, Av, BATCH, rng)
        z_t = torch.from_numpy(z)
        emb = model.stage1.encode_channel(z_t)
        emb_npd = emb[:, br_t, :]
        cw_npd = torch.from_numpy(x_phys[:, br]).long()
        loss = model.stage1.tree.fast_ce(emb_npd, cw_npd)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % CKPT_EVERY == 0:
            torch.save({'state_dict': model.stage1.state_dict(), 'N': N, 'Au': Au, 'Av': Av, 'iter': it},
                       os.path.join(RESULTS_DIR, f'{tag_s1}_iter{it}.pt'))

        if it % EVAL_EVERY == 0 or it == ITERS_S1:
            bler = eval_stage(model, 1, channel, Au, Av, fu_set, n_cw=EVAL_CW, seed=999)
            avg_loss = float(np.mean(losses[-min(500, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                torch.save({'state_dict': model.stage1.state_dict(), 'N': N, 'Au': Au, 'Av': Av, 'iter': it,
                            'best_bler': best_bler},
                           os.path.join(RESULTS_DIR, f'{tag_s1}_best.pt'))
                marker = ' *BEST*'
            msg = f'  [S1 {it:>7}/{ITERS_S1}] loss={avg_loss:.4f} BLER={bler:.4f} (best={best_bler:.4f}) {elapsed:.1f}min{marker}'
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

    s1_time = (time.time() - t0) / 60
    print(f'\nStage 1 done: best BLER={best_bler:.4f} ({s1_time:.1f} min)')

    # Reload best S1
    s1_ckpt = os.path.join(RESULTS_DIR, f'{tag_s1}_best.pt')
    if os.path.exists(s1_ckpt):
        sd = torch.load(s1_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])

    # Stage 2
    try_warmstart(model, 2)
    print(f'\n--- Stage 2 training ({ITERS_S2} iters) ---')
    opt2 = torch.optim.AdamW(model.stage2.parameters(), lr=LR, weight_decay=1e-5)
    rng2 = np.random.default_rng(SEED + 1)
    best_bler2 = 1.0
    losses2 = []
    t1 = time.time()
    model.stage1.eval()
    model.stage2.train()

    for it in range(1, ITERS_S2 + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, Au, Av, BATCH, rng2)
        z_t = torch.from_numpy(z)
        side = torch.from_numpy((1.0 - 2.0 * x_phys.astype(np.float32))).unsqueeze(-1)
        emb = model.stage2.encode_channel(z_t, side=side)
        emb_npd = emb[:, br_t, :]
        cw_npd = torch.from_numpy(y_phys[:, br]).long()
        loss = model.stage2.tree.fast_ce(emb_npd, cw_npd)
        opt2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.stage2.parameters(), 1.0)
        opt2.step()
        losses2.append(loss.item())

        if it % 10000 == 0:
            torch.save({'state_dict': model.stage2.state_dict(), 'N': N, 'Au': Au, 'Av': Av, 'iter': it},
                       os.path.join(RESULTS_DIR, f'{tag_s2}_iter{it}.pt'))

        if it % 2500 == 0 or it == ITERS_S2:
            bler = eval_stage(model, 2, channel, Au, Av, fv_set, n_cw=EVAL_CW, seed=999)
            avg_loss = float(np.mean(losses2[-min(200, len(losses2)):]))
            elapsed = (time.time() - t1) / 60
            marker = ''
            if bler < best_bler2:
                best_bler2 = bler
                torch.save({'state_dict': model.stage2.state_dict(), 'N': N, 'Au': Au, 'Av': Av, 'iter': it},
                           os.path.join(RESULTS_DIR, f'{tag_s2}_best.pt'))
                marker = ' *BEST*'
            msg = f'  [S2 {it:>6}/{ITERS_S2}] loss={avg_loss:.4f} BLER={bler:.4f} (best={best_bler2:.4f}) {elapsed:.1f}min{marker}'
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

    s2_time = (time.time() - t1) / 60
    print(f'\nStage 2 done: best BLER(V|trueU)={best_bler2:.4f} ({s2_time:.1f} min)')

    # Reload best S2
    s2_ckpt = os.path.join(RESULTS_DIR, f'{tag_s2}_best.pt')
    if os.path.exists(s2_ckpt):
        sd = torch.load(s2_ckpt, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd['state_dict'])

    # Chained eval (5000 CW)
    print(f'\nChained inference (5000 CW)...')
    chained = eval_chained(model, channel, Au, Av, fu_set, fv_set, n_cw=5000, seed=777)
    ch_ci = wilson_ci(chained['errs_total'], 5000)
    print(f'Chained BLER={chained["bler_total"]:.4f} CI=[{ch_ci[0]:.4f},{ch_ci[1]:.4f}]')
    print(f'  U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f}')

    # Memoryless SC baseline (3000 CW)
    print(f'\nMemoryless SC baseline (3000 CW)...')
    t_sc = time.time()
    baseline = eval_memoryless_sc(channel, Au, Av, fu_dict, fv_dict, n_cw=3000, seed=555)
    sc_time = (time.time() - t_sc) / 60
    print(f'SC BLER={baseline["bler_total"]:.4f} ({sc_time:.1f} min)')

    total_min = (time.time() - t_total) / 60
    ratio = chained['bler_total'] / max(baseline['bler_total'], 1e-6)
    print(f'\n{"="*60}')
    print(f'Ising MAC N={N} RESULTS')
    print(f'NPD chained: {chained["bler_total"]:.4f}')
    print(f'Memoryless SC: {baseline["bler_total"]:.4f}')
    print(f'Ratio (NPD/SC): {ratio:.3f}')
    print(f'Total time: {total_min:.1f} min')
    print(f'{"="*60}')

    # Save results
    result = {
        'channel': 'ising_mac', 'sigma2': SIGMA2, 'p_flip': P_FLIP,
        'N': N, 'ku': KU, 'kv': KV, 'd': D, 'hidden': HIDDEN,
        'stage1_best_bler': float(best_bler),
        'stage2_best_bler_true_x': float(best_bler2),
        'chained': {k: float(v) for k, v in chained.items()},
        'chained_ci': list(ch_ci),
        'memoryless_sc': {k: float(v) for k, v in baseline.items()},
        's1_time_min': s1_time, 's2_time_min': s2_time,
        'total_time_min': total_min,
    }

    # Update the full results file
    out_json = os.path.join(RESULTS_DIR, 'ising_mac_d16h100_results.json')
    try:
        with open(out_json) as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}
    all_results[str(N)] = result
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    main()
