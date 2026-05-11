#!/usr/bin/env python3
"""
Evaluate d=16 h=100 ISI-MAC chained NPD at N=64 and N=128 with 5000 CW.
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
from neural.npd_memory_mac import ChainedNPD_MAC

SNR_DB = 6.0
ISI_H = 0.3
D = 16
HIDDEN = 100
N_LAYERS = 2
GRU_LAYERS = 1

CONFIGS = {
    64:  {'ku': 15, 'kv': 29},
    128: {'ku': 30, 'kv': 58},
}

CKPT_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')


def make_channel():
    return ISIMAC.from_snr_db(SNR_DB, h=ISI_H)


def load_design(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    fu_set = {p-1 for p in range(1, N+1) if p not in Au}
    fv_set = {p-1 for p in range(1, N+1) if p not in Av}
    return Au, Av, fu_set, fv_set


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
    return {
        'n_cw': n_cw, 'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw, 'bler_total': errs_total / n_cw,
    }


def main():
    channel = make_channel()
    results = {}

    for N in [64, 128]:
        cfg = CONFIGS[N]
        Au, Av, fu_set, fv_set = load_design(N, cfg['ku'], cfg['kv'])

        # Load model
        torch.manual_seed(42)
        model = ChainedNPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS,
                               encoder_type='bigru', window_size=2,
                               gru_layers=GRU_LAYERS)

        s1_path = os.path.join(CKPT_DIR, f'd16_h100_standalone_s1_N{N}_best.pt')
        s2_path = os.path.join(CKPT_DIR, f'd16_h100_standalone_s2_N{N}_best.pt')

        if not os.path.exists(s1_path):
            print(f'  N={N}: S1 checkpoint not found: {s1_path}')
            continue
        if not os.path.exists(s2_path):
            print(f'  N={N}: S2 checkpoint not found: {s2_path}')
            continue

        sd1 = torch.load(s1_path, weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(sd1['state_dict'])
        sd2 = torch.load(s2_path, weights_only=False, map_location='cpu')
        model.stage2.load_state_dict(sd2['state_dict'])

        print(f'\nN={N} d={D} h={HIDDEN}: chained eval (5000 CW)...')
        t0 = time.time()
        r = eval_chained(model, channel, N, Au, Av, fu_set, fv_set, n_cw=5000, seed=777)
        elapsed = time.time() - t0
        print(f'  BLER={r["bler_total"]:.4f} (U={r["bler_u"]:.4f}, V={r["bler_v"]:.4f}) '
              f'({elapsed:.1f}s)')
        results[str(N)] = r

    out_json = os.path.join(_ROOT, 'results', 'reliable_evals', 'isi_mac_d16h100_chained.json')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_json}')


if __name__ == '__main__':
    main()
