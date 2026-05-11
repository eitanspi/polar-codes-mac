#!/usr/bin/env python3
"""Audit: settle chained NPD vs chained trellis SC discrepancy at ISI-MAC h=0.3, SNR=6 dB.

Two prior measurements disagreed at N=32:
  - class_c_npd/results/npd_memory_mac_results.md: NPD=0.078 (window W=2) and 0.1115 (BiGRU).
  - results/snr_sweep/isi_mac_h_sweep_N32.json: NPD=0.118 (BiGRU).

This script evaluates each checkpoint pair at 10k codewords with Wilson 95% CIs
for N in {16, 32, 64} at h=0.3, SNR=6 dB, and also freshly re-runs chained trellis
SC as ground truth at each N.
"""
from __future__ import annotations
import os, sys, math, json, time
import numpy as np
import torch

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from polar.decoder_trellis_mac_chained import (
    decode_stage1_u as trellis_decode_stage1,
    decode_stage2_v as trellis_decode_stage2,
)
from neural.npd_memory_mac import ChainedNPD_MAC


SNR_DB = 6.0
H = 0.3
N_CW = 10000
SEED_BASE = 20260416

CKPT_DIR = os.path.join(_ROOT, 'class_c_npd/results/npd_memory_mac')

# (n_log, N, ku, kv)
SIZES = [(4, 16, 4, 7), (5, 32, 7, 15), (6, 64, 15, 29)]


def wilson_ci(errs, total, z=1.96):
    if total == 0:
        return (0.0, 1.0)
    p = errs / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_design(n_log, ku, kv):
    dp = os.path.join(_ROOT, 'designs', f'gmac_C_n{n_log}_snr6dB.npz')
    Au, Av, fu_1idx, fv_1idx, _, _, _ = design_from_file(dp, n_log, ku, kv)
    return sorted(Au), sorted(Av), fu_1idx, fv_1idx


def build_model(encoder_type):
    kwargs = dict(d=16, hidden=64, n_layers=2)
    if encoder_type == 'bigru':
        return ChainedNPD_MAC(encoder_type='bigru', gru_layers=1, **kwargs)
    elif encoder_type == 'window':
        return ChainedNPD_MAC(encoder_type='window', window_size=2, **kwargs)
    else:
        raise ValueError(encoder_type)


def load_pair(encoder_type, N, n_log):
    if encoder_type == 'bigru':
        s1n = f'isi_mac_bigru_L1_s1_N{N}_best.pt'
        s2n = f'isi_mac_bigru_L1_s2_N{N}_best.pt'
    elif encoder_type == 'window':
        s1n = f'isi_mac_window_w2_s1_N{N}_best.pt'
        s2n = f'isi_mac_window_w2_s2_N{N}_best.pt'
    else:
        raise ValueError(encoder_type)
    s1p = os.path.join(CKPT_DIR, s1n)
    s2p = os.path.join(CKPT_DIR, s2n)
    assert os.path.exists(s1p), f'missing {s1p}'
    assert os.path.exists(s2p), f'missing {s2p}'
    model = build_model(encoder_type)
    s1 = torch.load(s1p, weights_only=False, map_location='cpu')
    s2 = torch.load(s2p, weights_only=False, map_location='cpu')
    model.stage1.load_state_dict(s1['state_dict'])
    model.stage2.load_state_dict(s2['state_dict'])
    model.stage1.eval()
    model.stage2.eval()
    return model, s1n, s2n


def eval_npd(model, channel, N, n_log, Au, Av, n_cw, seed_off=0):
    br = bit_reversal_perm(n_log)
    br_t = torch.from_numpy(br.copy()).long()
    fu_set = {p - 1 for p in range(1, N + 1) if p not in Au}
    fv_set = {p - 1 for p in range(1, N + 1) if p not in Av}
    rng = np.random.default_rng(SEED_BASE + seed_off)
    errs = 0
    errs_u = 0
    errs_v = 0
    total = 0
    batch = 32
    t0 = time.time()
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg = np.zeros((actual, N), dtype=np.int8)
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Au:
                u_msg[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                v_msg[:, p - 1] = rng.integers(0, 2, actual)
            x = polar_encode_batch(u_msg.astype(int))
            y = polar_encode_batch(v_msg.astype(int))
            z = channel.sample_batch(x, y).astype(np.float32)
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
                u_wrong = any(int(u_hat[i, p - 1].item()) != int(u_msg[i, p - 1]) for p in Au)
                v_wrong = any(int(v_hat[i, p - 1].item()) != int(v_msg[i, p - 1]) for p in Av)
                if u_wrong: errs_u += 1
                if v_wrong: errs_v += 1
                if u_wrong or v_wrong: errs += 1
            total += actual
    t = time.time() - t0
    return {
        'bler': errs / total, 'errs_total': errs,
        'errs_u': errs_u, 'errs_v': errs_v,
        'n_cw': total, 'time_s': round(t, 2),
    }


def eval_trellis_sc(channel, N, Au, Av, fu_1idx, fv_1idx, n_cw, seed_off=0):
    rng = np.random.default_rng(SEED_BASE + seed_off + 7)
    errs = 0
    errs_u = 0
    errs_v = 0
    t0 = time.time()
    for _ in range(n_cw):
        u_msg = np.zeros(N, dtype=int)
        v_msg = np.zeros(N, dtype=int)
        for p in Au:
            u_msg[p - 1] = rng.integers(0, 2)
        for p in Av:
            v_msg[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u_msg.reshape(1, -1))[0]
        y = polar_encode_batch(v_msg.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        u_hat = trellis_decode_stage1(z, N, fu_1idx, channel)
        v_hat = trellis_decode_stage2(z, u_hat, N, fv_1idx, channel)
        u_wrong = any(int(u_hat[p - 1]) != int(u_msg[p - 1]) for p in Au)
        v_wrong = any(int(v_hat[p - 1]) != int(v_msg[p - 1]) for p in Av)
        if u_wrong: errs_u += 1
        if v_wrong: errs_v += 1
        if u_wrong or v_wrong: errs += 1
    t = time.time() - t0
    return {
        'bler': errs / n_cw, 'errs_total': errs,
        'errs_u': errs_u, 'errs_v': errs_v,
        'n_cw': n_cw, 'time_s': round(t, 2),
    }


def main():
    out = {
        'config': {
            'snr_db': SNR_DB,
            'h': H,
            'n_cw': N_CW,
            'seed_base': SEED_BASE,
            'channel': f'ISIMAC(h={H}) at SNR={SNR_DB} dB',
        },
        'results': {},
    }
    sigma2 = 10 ** (-SNR_DB / 10)
    channel = ISIMAC(sigma2=sigma2, h=H)

    out_path = os.path.join(_ROOT, 'results', 'snr_sweep', 'isi_mac_audit_10kcw.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _save():
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)

    for (n_log, N, ku, kv) in SIZES:
        print(f'\n=== N={N} (n_log={n_log}, ku={ku}, kv={kv}) ===')
        Au, Av, fu_1idx, fv_1idx = load_design(n_log, ku, kv)
        slot = {'Au': Au, 'Av': Av, 'ku': ku, 'kv': kv}

        # Trellis SC (ground truth).
        print(f'  Trellis SC, {N_CW} codewords...', flush=True)
        r_sc = eval_trellis_sc(channel, N, Au, Av, fu_1idx, fv_1idx,
                                n_cw=N_CW, seed_off=n_log * 100)
        ci = wilson_ci(r_sc['errs_total'], r_sc['n_cw'])
        r_sc['ci_lo'] = ci[0]; r_sc['ci_hi'] = ci[1]
        slot['chained_trellis_sc'] = r_sc
        print(f'    BLER={r_sc["bler"]:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
              f'u={r_sc["errs_u"]} v={r_sc["errs_v"]} t={r_sc["time_s"]}s')
        out['results'][f'N={N}'] = slot
        _save()

        # Window encoder.
        print(f'  NPD window(W=2), {N_CW} codewords...', flush=True)
        model, s1n, s2n = load_pair('window', N, n_log)
        r = eval_npd(model, channel, N, n_log, Au, Av,
                      n_cw=N_CW, seed_off=n_log * 100 + 1)
        ci = wilson_ci(r['errs_total'], r['n_cw'])
        r['ci_lo'] = ci[0]; r['ci_hi'] = ci[1]
        r['s1_ckpt'] = s1n; r['s2_ckpt'] = s2n
        slot['npd_window_w2'] = r
        print(f'    BLER={r["bler"]:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
              f'u={r["errs_u"]} v={r["errs_v"]} t={r["time_s"]}s')
        _save()

        # BiGRU encoder.
        print(f'  NPD BiGRU(L=1), {N_CW} codewords...', flush=True)
        model, s1n, s2n = load_pair('bigru', N, n_log)
        r = eval_npd(model, channel, N, n_log, Au, Av,
                      n_cw=N_CW, seed_off=n_log * 100 + 2)
        ci = wilson_ci(r['errs_total'], r['n_cw'])
        r['ci_lo'] = ci[0]; r['ci_hi'] = ci[1]
        r['s1_ckpt'] = s1n; r['s2_ckpt'] = s2n
        slot['npd_bigru_L1'] = r
        print(f'    BLER={r["bler"]:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
              f'u={r["errs_u"]} v={r["errs_v"]} t={r["time_s"]}s')
        _save()

    print(f'\nSaved {out_path}')
    return out


if __name__ == '__main__':
    main()
