#!/usr/bin/env python3
"""
eval_reliable_isi_mac.py
========================
Standardized 5000 CW evaluations for ISI-MAC NPD thesis table.
Also runs trellis SC at 10K CW for N=64.
"""
import json
import math
import os
import sys
import time
import numpy as np
import torch

torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from polar.decoder_trellis import decode_single as trellis_decode_single
from polar.design import make_path
from neural.npd_memory_mac import ChainedNPD_MAC, MemoryStageNPD

# ---------- Config ----------
SNR_DB = 6.0
ISI_H = 0.3
RATES = {
    16:  (4, 7),
    32:  (7, 15),
    64:  (15, 29),
    128: (30, 58),
    256: (59, 117),
    512: (119, 233),
}
RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')
OUT_DIR = os.path.join(_ROOT, 'results', 'reliable_evals')
os.makedirs(OUT_DIR, exist_ok=True)


def make_channel():
    return ISIMAC.from_snr_db(SNR_DB, h=ISI_H)


def load_design_for_N(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, _path_i = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_dict = {p: 0 for p in range(1, N+1) if p not in Au}
    frozen_v_dict = {p: 0 for p in range(1, N+1) if p not in Av}
    frozen_u_set = {p-1 for p in frozen_u_dict.keys()}
    frozen_v_set = {p-1 for p in frozen_v_dict.keys()}
    return Au, Av, frozen_u_dict, frozen_v_dict, frozen_u_set, frozen_v_set


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
    return u_msg, v_msg, z.astype(np.float32), x_phys, y_phys


def eval_stage1(model_stage1, channel, N, Au, Av, frozen_u_set, n_cw=5000, batch=32, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model_stage1.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, _, z, _, _ = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            emb = model_stage1.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            u_hat = model_stage1.tree.decode(emb_npd, frozen_u_set)
            for i in range(actual):
                if any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au):
                    errs += 1
            total += actual
            if total % 1000 == 0:
                print(f'    S1 eval: {total}/{n_cw}, errs={errs}', flush=True)
    return errs, total


def eval_chained(model, channel, N, Au, Av, frozen_u_set, frozen_v_set,
                 n_cw=5000, batch=32, seed=777):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model.stage1.eval()
    model.stage2.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    total = 0
    first_error_positions_u = []  # for Task 3
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            # Stage 1
            emb1 = model.stage1.encode_channel(z_t)
            emb1_npd = emb1[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb1_npd, frozen_u_set)
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)
            # Stage 2
            side = torch.from_numpy((1.0 - 2.0 * x_hat.astype(np.float32))).unsqueeze(-1)
            emb2 = model.stage2.encode_channel(z_t, side=side)
            emb2_npd = emb2[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb2_npd, frozen_v_set)
            for i in range(actual):
                u_wrong = any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au)
                v_wrong = any(int(v_hat[i, p-1].item()) != int(v_msg[i, p-1]) for p in Av)
                if u_wrong:
                    errs_u += 1
                    # Record first error position (for Task 3)
                    for idx, p in enumerate(sorted(Au)):
                        if int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]):
                            first_error_positions_u.append(p)
                            break
                if v_wrong:
                    errs_v += 1
                if u_wrong or v_wrong:
                    errs_total += 1
            total += actual
            if total % 1000 == 0:
                print(f'    Chained eval: {total}/{n_cw}, errs_total={errs_total}', flush=True)
    return {
        'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw, 'bler_total': errs_total / n_cw,
        'first_error_positions_u': first_error_positions_u,
    }


def eval_trellis_sc(channel, N, Au, Av, fu_dict, fv_dict, n_cw=10000, seed=555):
    b = make_path(N, N)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs_u = errs_v = errs_total = 0
    for i in range(n_cw):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for p in Au:
            u_msg[p-1] = rng.integers(0, 2)
        for p in Av:
            v_msg[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u_msg[None, :])[0]
        y = polar_encode_batch(v_msg[None, :])[0]
        z = channel.sample_batch(x[None, :].astype(int), y[None, :].astype(int))[0]
        try:
            u_dec, v_dec = trellis_decode_single(
                N, z.tolist(), b, fu_dict, fv_dict, channel, log_domain=True)
        except Exception as e:
            print(f'  trellis eval iter {i} raised: {e}')
            continue
        u_wrong = any(u_dec[p-1] != u_msg[p-1] for p in Au)
        v_wrong = any(v_dec[p-1] != v_msg[p-1] for p in Av)
        if u_wrong: errs_u += 1
        if v_wrong: errs_v += 1
        if u_wrong or v_wrong: errs_total += 1
        if (i+1) % 2000 == 0:
            print(f'    Trellis SC: {i+1}/{n_cw}, errs={errs_total}', flush=True)
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw, 'bler_total': errs_total / n_cw,
    }


def load_stage_ckpt(stage_model, ckpt_path, wrapped=True):
    """Load checkpoint -- handle both wrapped (dict with 'state_dict') and raw formats."""
    sd = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    if wrapped and 'state_dict' in sd:
        stage_model.load_state_dict(sd['state_dict'])
    else:
        stage_model.load_state_dict(sd)
    print(f'  Loaded: {os.path.basename(ckpt_path)}')


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval."""
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    spread = z * math.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


# ===== EVALUATION CONFIGURATIONS =====
EVAL_CONFIGS = [
    # (N, d, hidden, enc_type, window, gru_layers, s1_ckpt, s2_ckpt, s1_wrapped, s2_wrapped, label)
    (16, 16, 64, 'bigru', 1, 1,
     'isi_mac_bigru_L1_s1_N16_best.pt',
     'isi_mac_bigru_L1_s2_N16_best.pt',
     True, True, 'N16_d16_bigru'),
    (32, 16, 64, 'window', 2, 1,
     'isi_mac_window_w2_s1_N32_best.pt',
     'isi_mac_window_w2_s2_N32_best.pt',
     True, True, 'N32_d16_window'),
    (64, 16, 64, 'bigru', 1, 1,
     'isi_mac_bigru_L1_s1_N64_best.pt',
     'isi_mac_bigru_L1_s2_N64_best.pt',
     True, True, 'N64_d16_bigru'),
    (128, 64, 128, 'bigru', 1, 1,
     'd64_lr1e3_N128_final.pt',  # raw state_dict
     'd64_s2_N128_best.pt',
     False, True, 'N128_d64_bigru'),
    (256, 64, 128, 'bigru', 1, 1,
     'd64_s1_N256_300k.pt',  # need to check this name
     'd64_s2_N256_best.pt',
     False, True, 'N256_d64_bigru'),
]


def main():
    channel = make_channel()
    all_results = {}

    # Check if the N=256 s1 checkpoint exists under the right name
    n256_s1_path = os.path.join(RESULTS_DIR, 'd64_s1_N256_300k.pt')
    if not os.path.exists(n256_s1_path):
        # Try copying from /tmp
        import shutil
        src = '/tmp/d64_N256_300k.pt'
        if os.path.exists(src):
            shutil.copy2(src, n256_s1_path)
            print(f'Copied {src} -> {n256_s1_path}')
        else:
            print(f'WARNING: N=256 S1 checkpoint not found at {src}')

    for N, d, hidden, enc_type, window, gru_layers, s1_name, s2_name, s1_wrapped, s2_wrapped, label in EVAL_CONFIGS:
        print(f'\n{"="*60}')
        print(f'Evaluating {label}: N={N}, d={d}, enc={enc_type}')
        print(f'{"="*60}')

        ku, kv = RATES[N]
        Au, Av, fu_dict, fv_dict, fu_set, fv_set = load_design_for_N(N, ku, kv)
        print(f'  ku={ku}, kv={kv}, |Au|={len(Au)}, |Av|={len(Av)}')

        s1_path = os.path.join(RESULTS_DIR, s1_name)
        s2_path = os.path.join(RESULTS_DIR, s2_name)

        if not os.path.exists(s1_path):
            print(f'  SKIP: S1 checkpoint not found: {s1_path}')
            continue
        if not os.path.exists(s2_path):
            print(f'  SKIP: S2 checkpoint not found: {s2_path}')
            continue

        # Build model
        model = ChainedNPD_MAC(d=d, hidden=hidden, n_layers=2,
                               encoder_type=enc_type, window_size=window,
                               gru_layers=gru_layers)
        load_stage_ckpt(model.stage1, s1_path, wrapped=s1_wrapped)
        load_stage_ckpt(model.stage2, s2_path, wrapped=s2_wrapped)

        # Stage 1 only eval
        t0 = time.time()
        s1_errs, s1_total = eval_stage1(model.stage1, channel, N, Au, Av, fu_set,
                                         n_cw=5000, batch=32, seed=999)
        s1_time = time.time() - t0
        s1_bler = s1_errs / s1_total
        s1_ci_lo, s1_ci_hi = wilson_ci(s1_errs, s1_total)
        print(f'  S1 BLER={s1_bler:.4f} ({s1_errs}/{s1_total}) '
              f'CI=[{s1_ci_lo:.4f}, {s1_ci_hi:.4f}] ({s1_time:.0f}s)')

        # Chained eval
        t0 = time.time()
        chained = eval_chained(model, channel, N, Au, Av, fu_set, fv_set,
                               n_cw=5000, batch=32, seed=777)
        ch_time = time.time() - t0
        ch_ci_lo, ch_ci_hi = wilson_ci(chained['errs_total'], chained['n_cw'])
        print(f'  Chained BLER={chained["bler_total"]:.4f} '
              f'({chained["errs_total"]}/{chained["n_cw"]}) '
              f'CI=[{ch_ci_lo:.4f}, {ch_ci_hi:.4f}] '
              f'U={chained["bler_u"]:.4f} V={chained["bler_v"]:.4f} ({ch_time:.0f}s)')

        all_results[label] = {
            'N': N, 'd': d, 'hidden': hidden, 'encoder': enc_type,
            'ku': ku, 'kv': kv,
            's1_bler': s1_bler, 's1_errs': s1_errs, 's1_total': s1_total,
            's1_ci': [s1_ci_lo, s1_ci_hi],
            'chained_bler': chained['bler_total'],
            'chained_errs': chained['errs_total'],
            'chained_n_cw': chained['n_cw'],
            'chained_ci': [ch_ci_lo, ch_ci_hi],
            'bler_u': chained['bler_u'], 'bler_v': chained['bler_v'],
            'errs_u': chained['errs_u'], 'errs_v': chained['errs_v'],
            'first_error_positions_u': chained['first_error_positions_u'],
            's1_ckpt': s1_name, 's2_ckpt': s2_name,
        }

        # Save incrementally
        out_path = os.path.join(OUT_DIR, 'isi_mac_npd_reliable.json')
        # Convert first_error_positions to list for JSON
        save_results = {}
        for k, v in all_results.items():
            save_results[k] = {kk: (vv if not isinstance(vv, np.integer) else int(vv))
                               for kk, vv in v.items()}
        with open(out_path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f'  Saved to {out_path}')

    # Trellis SC at 10K CW for N=64
    print(f'\n{"="*60}')
    print(f'Trellis SC reference: N=64, 10K CW')
    print(f'{"="*60}')
    N = 64
    ku, kv = RATES[N]
    Au, Av, fu_dict, fv_dict, fu_set, fv_set = load_design_for_N(N, ku, kv)
    t0 = time.time()
    trellis_result = eval_trellis_sc(channel, N, Au, Av, fu_dict, fv_dict,
                                      n_cw=10000, seed=555)
    tr_time = time.time() - t0
    tr_ci_lo, tr_ci_hi = wilson_ci(trellis_result['errs_total'], trellis_result['n_cw'])
    print(f'  Trellis SC BLER={trellis_result["bler_total"]:.4f} '
          f'({trellis_result["errs_total"]}/{trellis_result["n_cw"]}) '
          f'CI=[{tr_ci_lo:.4f}, {tr_ci_hi:.4f}] ({tr_time:.0f}s)')

    all_results['trellis_sc_N64_10k'] = {
        'N': 64, 'decoder': 'trellis_sc',
        **{k: (int(v) if isinstance(v, (np.integer, np.int64)) else v)
           for k, v in trellis_result.items()},
        'ci': [tr_ci_lo, tr_ci_hi],
    }

    # Final save
    out_path = os.path.join(OUT_DIR, 'isi_mac_npd_reliable.json')
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: (int(vv) if isinstance(vv, (np.integer, np.int64)) else vv)
                           for kk, vv in v.items()}
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f'\nAll results saved to {out_path}')

    # Print summary table
    print(f'\n{"="*60}')
    print('SUMMARY TABLE')
    print(f'{"="*60}')
    for label, res in all_results.items():
        if 'trellis' in label:
            print(f'{label}: BLER={res["bler_total"]:.4f} '
                  f'({res["errs_total"]}/{res["n_cw"]})')
        else:
            print(f'{label}: S1={res["s1_bler"]:.4f} '
                  f'Chained={res["chained_bler"]:.4f} '
                  f'({res["chained_errs"]}/{res["chained_n_cw"]})')


if __name__ == '__main__':
    main()
