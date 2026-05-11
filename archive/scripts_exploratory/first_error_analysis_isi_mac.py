#!/usr/bin/env python3
"""
first_error_analysis_isi_mac.py
===============================
First-error position analysis for ISI-MAC NPD at N=64 and N=128.
Compares error distribution between the two sizes to see if errors
cluster differently at larger N.
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
from polar.design import make_path
from neural.npd_memory_mac import ChainedNPD_MAC

SNR_DB = 6.0
ISI_H = 0.3
RATES = {64: (15, 29), 128: (30, 58), 512: (119, 233)}
RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')


def make_channel():
    return ISIMAC.from_snr_db(SNR_DB, h=ISI_H)


def load_design_for_N(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, _path_i = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_set = {p-1 for p in range(1, N+1) if p not in Au}
    frozen_v_set = {p-1 for p in range(1, N+1) if p not in Av}
    frozen_u_dict = {p: 0 for p in range(1, N+1) if p not in Au}
    frozen_v_dict = {p: 0 for p in range(1, N+1) if p not in Av}
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


def first_error_analysis(model, channel, N, Au, Av, frozen_u_set, frozen_v_set,
                          n_cw=5000, batch=32, seed=777):
    """Chained eval + first-error position analysis."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    b = make_path(N, N)  # Class C corner: all U first, then all V

    model.stage1.eval()
    model.stage2.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Storage
    first_err_step_u = []   # step in path where first U error occurs (1..2N)
    first_err_pos_u = []    # position within Au sorted list
    first_err_quartile_u = []  # which quartile of Au (0-3)
    per_pos_errs_u = np.zeros(N, dtype=int)  # per-position error counts for U
    per_pos_errs_v = np.zeros(N, dtype=int)
    errs_u = errs_v = errs_total = 0
    total = 0

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
            v_hat_np = v_hat.numpy().astype(int)

            for i in range(actual):
                # Per-position errors
                for p in Au:
                    if u_hat_np[i, p-1] != u_msg[i, p-1]:
                        per_pos_errs_u[p-1] += 1
                for p in Av:
                    if v_hat_np[i, p-1] != v_msg[i, p-1]:
                        per_pos_errs_v[p-1] += 1

                u_wrong = any(u_hat_np[i, p-1] != u_msg[i, p-1] for p in Au)
                v_wrong = any(v_hat_np[i, p-1] != v_msg[i, p-1] for p in Av)
                if u_wrong: errs_u += 1
                if v_wrong: errs_v += 1
                if u_wrong or v_wrong: errs_total += 1

                # First-error position in U (path order)
                if u_wrong:
                    Au_sorted = sorted(Au)
                    for idx, p in enumerate(Au_sorted):
                        if u_hat_np[i, p-1] != u_msg[i, p-1]:
                            first_err_pos_u.append(p)
                            q = int(idx * 4 / len(Au_sorted))
                            first_err_quartile_u.append(min(q, 3))
                            # Path step: in Class C, U positions are first N steps
                            first_err_step_u.append(p)  # for corner rate, step == position
                            break

            total += actual
            if total % 1000 == 0:
                print(f'  {total}/{n_cw}, errs_total={errs_total}', flush=True)

    # Quartile analysis
    quartile_counts = [0, 0, 0, 0]
    for q in first_err_quartile_u:
        quartile_counts[q] += 1

    return {
        'N': N, 'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw, 'bler_total': errs_total / n_cw,
        'first_err_pos_u': [int(x) for x in first_err_pos_u],
        'first_err_quartile_u': [int(x) for x in first_err_quartile_u],
        'quartile_counts': quartile_counts,
        'quartile_fracs': [c / max(len(first_err_quartile_u), 1) for c in quartile_counts],
        'per_pos_errs_u': per_pos_errs_u.tolist(),
        'per_pos_errs_v': per_pos_errs_v.tolist(),
        'n_u_failures': errs_u,
    }


def load_stage_ckpt(stage_model, ckpt_path, wrapped=True):
    sd = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    if wrapped and 'state_dict' in sd:
        stage_model.load_state_dict(sd['state_dict'])
    else:
        stage_model.load_state_dict(sd)
    print(f'  Loaded: {os.path.basename(ckpt_path)}')


def main():
    channel = make_channel()
    all_results = {}

    # N=64 d=16 BiGRU
    configs = [
        (64, 16, 64, 'bigru', 1, 1,
         'isi_mac_bigru_L1_s1_N64_best.pt',
         'isi_mac_bigru_L1_s2_N64_best.pt',
         True, True, 'N64_d16_bigru'),
        (128, 64, 128, 'bigru', 1, 1,
         'd64_lr1e3_N128_final.pt',
         'd64_s2_N128_best.pt',
         False, True, 'N128_d64_bigru'),
    ]

    for N, d, hidden, enc_type, window, gru_layers, s1_name, s2_name, s1_wrapped, s2_wrapped, label in configs:
        print(f'\n{"="*60}')
        print(f'First-error analysis: {label}')
        print(f'{"="*60}')

        ku, kv = RATES[N]
        Au, Av, fu_dict, fv_dict, fu_set, fv_set = load_design_for_N(N, ku, kv)

        model = ChainedNPD_MAC(d=d, hidden=hidden, n_layers=2,
                               encoder_type=enc_type, window_size=window,
                               gru_layers=gru_layers)
        load_stage_ckpt(model.stage1, os.path.join(RESULTS_DIR, s1_name), wrapped=s1_wrapped)
        load_stage_ckpt(model.stage2, os.path.join(RESULTS_DIR, s2_name), wrapped=s2_wrapped)

        t0 = time.time()
        result = first_error_analysis(model, channel, N, Au, Av, fu_set, fv_set,
                                       n_cw=5000, batch=32, seed=777)
        elapsed = time.time() - t0

        print(f'\n  BLER: total={result["bler_total"]:.4f} U={result["bler_u"]:.4f} V={result["bler_v"]:.4f}')
        print(f'  First-error quartile distribution (among {result["n_u_failures"]} U failures):')
        for q in range(4):
            frac = result['quartile_fracs'][q]
            cnt = result['quartile_counts'][q]
            pct = frac * 100
            print(f'    Q{q+1}: {cnt} ({pct:.1f}%)')
        print(f'  Time: {elapsed:.0f}s')

        all_results[label] = result

    # Summary comparison
    print(f'\n{"="*60}')
    print('COMPARISON: First-error distribution')
    print(f'{"="*60}')
    print(f'{"":>20s}  {"Q1":>8s}  {"Q2":>8s}  {"Q3":>8s}  {"Q4":>8s}  {"BLER":>8s}')
    for label, res in all_results.items():
        qf = res['quartile_fracs']
        print(f'{label:>20s}  {qf[0]:>7.1%}  {qf[1]:>7.1%}  {qf[2]:>7.1%}  {qf[3]:>7.1%}  {res["bler_total"]:>8.4f}')

    # Save
    out_path = os.path.join(_ROOT, 'results', 'reliable_evals', 'isi_mac_first_error_analysis.json')
    # Convert numpy to native types
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items()}
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
