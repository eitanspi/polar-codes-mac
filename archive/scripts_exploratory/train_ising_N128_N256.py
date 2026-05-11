#!/usr/bin/env python3
"""
train_ising_N128_N256.py
========================
Extend Ising MAC NPD to N=128, N=256 with d=16 h=100 BiGRU on CPU.
Warm-starts from previous N checkpoint if available.

Channel: IsingMAC(sigma2=0.251, p_flip=0.1)
Design: GMAC_C proxy at SNR=6dB.

First finishes N=64 if needed (continues from existing checkpoint),
then trains N=128 (500K iters), then N=256 (300K iters).
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
D = 16
HIDDEN = 100
N_LAYERS = 2
GRU_LAYERS = 1
ENCODER_TYPE = 'bigru'
LR = 1e-3
SEED = 42

CONFIGS = {
    64:  {'ku': 15, 'kv': 29, 'batch': 16, 'iters_s1': 200_000, 'iters_s2': 50_000},
    128: {'ku': 30, 'kv': 58, 'batch': 8,  'iters_s1': 500_000, 'iters_s2': 50_000},
    256: {'ku': 59, 'kv': 117, 'batch': 4, 'iters_s1': 300_000, 'iters_s2': 50_000},
}

RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_ising_mac')
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_channel():
    return IsingMAC(sigma2=SIGMA2, p_flip=P_FLIP)


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    fu_dict_out = {p: 0 for p in range(1, N+1) if p not in Au}
    fv_dict_out = {p: 0 for p in range(1, N+1) if p not in Av}
    fu_set = {p-1 for p in fu_dict_out}
    fv_set = {p-1 for p in fv_dict_out}
    return Au, Av, fu_dict_out, fv_dict_out, fu_set, fv_set


def make_batch(channel, N, Au, Av, batch, rng):
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au:
        u_msg[:, p-1] = rng.integers(0, 2, batch)
    for p in Av:
        v_msg[:, p-1] = rng.integers(0, 2, batch)
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


def eval_stage(model, stage_num, channel, N, Au, Av, frozen_set, n_cw=300, seed=999):
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
            u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, actual, rng)
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


def eval_chained(model, channel, N, Au, Av, fu_set, fv_set, n_cw=5000, batch=32, seed=777):
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
            u_msg, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, actual, rng)
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
                print(f'    chained [{total}/{n_cw}] errs={errs_total}', flush=True)
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u/n_cw, 'bler_v': errs_v/n_cw, 'bler_total': errs_total/n_cw,
    }


def eval_memoryless_sc(channel, N, Au, Av, fu_dict, fv_dict, n_cw=3000, seed=555):
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
            print(f'    SC [{cw_idx+1}/{n_cw}] errs={errs_total}', flush=True)
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u/n_cw, 'bler_v': errs_v/n_cw, 'bler_total': errs_total/n_cw,
    }


def try_warmstart(model, stage_num, from_N):
    """Try to warm-start from a previous N checkpoint."""
    tag = f'ising_d{D}_h{HIDDEN}_s{stage_num}_N{from_N}'
    ckpt_path = os.path.join(RESULTS_DIR, f'{tag}_best.pt')
    if not os.path.exists(ckpt_path):
        print(f'  No warm-start checkpoint at {ckpt_path}')
        return False
    try:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        stage = model.stage1 if stage_num == 1 else model.stage2
        stage.load_state_dict(ckpt['state_dict'])
        print(f'  Warm-started stage {stage_num} from N={from_N}')
        return True
    except Exception as e:
        print(f'  Warm-start failed: {e}')
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


def run_one_N(N, warm_from_N=None):
    cfg = CONFIGS[N]
    ku, kv = cfg['ku'], cfg['kv']
    batch = cfg['batch']
    iters_s1 = cfg['iters_s1']
    iters_s2 = cfg['iters_s2']

    channel = make_channel()
    Au, Av, fu_dict, fv_dict, fu_set, fv_set = load_design(N, ku, kv)

    tag_base = f'ising_d{D}_h{HIDDEN}'
    s1_tag = f'{tag_base}_s1_N{N}'
    s2_tag = f'{tag_base}_s2_N{N}'
    log_file = os.path.join(RESULTS_DIR, f'{tag_base}_N{N}.log')

    with open(log_file, 'a') as lf:
        lf.write(f'\n=== Ising MAC N={N} d={D} h={HIDDEN} started {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
        lf.write(f'sigma2={SIGMA2} p_flip={P_FLIP} ku={ku} kv={kv}\n')

    torch.manual_seed(SEED)
    model = ChainedNPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS,
                           encoder_type=ENCODER_TYPE, gru_layers=GRU_LAYERS)

    # Warm-start
    if warm_from_N:
        try_warmstart(model, 1, warm_from_N)

    print(f'\n{"="*60}')
    print(f'Ising MAC N={N} d={D} h={HIDDEN} params={model.count_parameters():,}')
    print(f'ku={ku} kv={kv} batch={batch} iters_s1={iters_s1} iters_s2={iters_s2}')
    print(f'{"="*60}')

    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    eval_every = max(5000, iters_s1 // 20)
    ckpt_every = max(20000, iters_s1 // 10)

    # Stage 1
    print(f'\n--- Stage 1 training ({iters_s1} iters) ---')
    opt = torch.optim.AdamW(model.stage1.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    best_bler = 1.0
    losses = []
    t0 = time.time()
    model.stage1.train()

    for it in range(1, iters_s1 + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, batch, rng)
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

        if it % ckpt_every == 0:
            torch.save({'state_dict': model.stage1.state_dict(), 'N': N, 'Au': Au, 'Av': Av, 'iter': it},
                       os.path.join(RESULTS_DIR, f'{s1_tag}_iter{it}.pt'))

        if it % eval_every == 0 or it == iters_s1:
            bler = eval_stage(model, 1, channel, N, Au, Av, fu_set, n_cw=300, seed=999)
            avg_loss = float(np.mean(losses[-min(500, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                torch.save({'state_dict': model.stage1.state_dict(), 'N': N, 'Au': Au, 'Av': Av, 'iter': it,
                            'best_bler': best_bler},
                           os.path.join(RESULTS_DIR, f'{s1_tag}_best.pt'))
                marker = ' *BEST*'
            msg = f'  [S1 {it:>7}/{iters_s1}] loss={avg_loss:.4f} BLER={bler:.4f} (best={best_bler:.4f}) {elapsed:.1f}min{marker}'
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

    s1_time = (time.time() - t0) / 60
    print(f'\nStage 1 done: best BLER={best_bler:.4f} ({s1_time:.1f} min)')

    # Reload best S1
    s1_ckpt = os.path.join(RESULTS_DIR, f'{s1_tag}_best.pt')
    if os.path.exists(s1_ckpt):
        sd = torch.load(s1_ckpt, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd['state_dict'])

    # Stage 2
    if warm_from_N:
        try_warmstart(model, 2, warm_from_N)

    print(f'\n--- Stage 2 training ({iters_s2} iters) ---')
    opt2 = torch.optim.AdamW(model.stage2.parameters(), lr=LR, weight_decay=1e-5)
    rng2 = np.random.default_rng(SEED + 1)
    best_bler2 = 1.0
    losses2 = []
    t1 = time.time()
    model.stage1.eval()
    model.stage2.train()

    for it in range(1, iters_s2 + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, batch, rng2)
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
                       os.path.join(RESULTS_DIR, f'{s2_tag}_iter{it}.pt'))

        if it % 2500 == 0 or it == iters_s2:
            bler = eval_stage(model, 2, channel, N, Au, Av, fv_set, n_cw=300, seed=999)
            avg_loss = float(np.mean(losses2[-min(200, len(losses2)):]))
            elapsed = (time.time() - t1) / 60
            marker = ''
            if bler < best_bler2:
                best_bler2 = bler
                torch.save({'state_dict': model.stage2.state_dict(), 'N': N, 'Au': Au, 'Av': Av, 'iter': it},
                           os.path.join(RESULTS_DIR, f'{s2_tag}_best.pt'))
                marker = ' *BEST*'
            msg = f'  [S2 {it:>6}/{iters_s2}] loss={avg_loss:.4f} BLER={bler:.4f} (best={best_bler2:.4f}) {elapsed:.1f}min{marker}'
            print(msg, flush=True)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')

    s2_time = (time.time() - t1) / 60
    print(f'\nStage 2 done: best BLER(V|trueU)={best_bler2:.4f} ({s2_time:.1f} min)')

    # Reload best S2
    s2_ckpt = os.path.join(RESULTS_DIR, f'{s2_tag}_best.pt')
    if os.path.exists(s2_ckpt):
        sd = torch.load(s2_ckpt, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd['state_dict'])

    # Chained eval
    print(f'\n  Chained inference (5000 CW)...')
    chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set, n_cw=5000, seed=777)
    ch_ci = wilson_ci(chained['errs_total'], 5000)
    print(f'  Chained BLER={chained["bler_total"]:.4f} CI=[{ch_ci[0]:.4f},{ch_ci[1]:.4f}]')

    # Skip memoryless SC baseline for N>=128 (too slow with O(N^2) decoder)
    if N <= 64:
        print(f'\n  Memoryless SC baseline (3000 CW)...')
        t_sc = time.time()
        baseline = eval_memoryless_sc(channel, N, Au, Av, fu_dict, fv_dict, n_cw=3000, seed=555)
        print(f'  SC BLER={baseline["bler_total"]:.4f} ({(time.time()-t_sc)/60:.1f} min)')
    else:
        # Use precomputed baseline from paper tables if available
        baseline = None
        print(f'  Skipping memoryless SC (too slow for N={N})')

    result = {
        'channel': 'ising_mac', 'sigma2': SIGMA2, 'p_flip': P_FLIP,
        'N': N, 'ku': ku, 'kv': kv, 'd': D, 'hidden': HIDDEN,
        'stage1_best_bler': float(best_bler),
        'stage2_best_bler_true_x': float(best_bler2),
        'chained': {k: float(v) for k, v in chained.items()},
        'chained_ci': list(ch_ci),
        's1_time_min': s1_time, 's2_time_min': s2_time,
    }
    if baseline:
        result['memoryless_sc'] = {k: float(v) for k, v in baseline.items()}

    return result


def main():
    all_results = {}
    out_json = os.path.join(RESULTS_DIR, 'ising_mac_d16h100_results.json')

    # Load existing results
    if os.path.exists(out_json):
        with open(out_json) as f:
            all_results = json.load(f)

    t_total = time.time()

    # Train N=128 warm from N=64
    for target_N, warm_N in [(128, 64), (256, 128)]:
        print(f'\n{"#"*60}')
        print(f'# Training Ising MAC N={target_N}, warm from N={warm_N}')
        print(f'{"#"*60}')

        res = run_one_N(target_N, warm_from_N=warm_N)
        all_results[str(target_N)] = res
        with open(out_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'\n  Saved: {out_json}')

    total_min = (time.time() - t_total) / 60
    print(f'\n{"="*60}')
    print(f'All done in {total_min:.1f} min')
    print(f'\n{"N":<6}{"S1 BLER":<12}{"Chained":<12}')
    for Ns, r in sorted(all_results.items(), key=lambda x: int(x[0])):
        ch = r['chained']['bler_total']
        print(f'{Ns:<6}{r["stage1_best_bler"]:<12.4f}{ch:<12.4f}')


if __name__ == '__main__':
    main()
