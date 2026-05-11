#!/usr/bin/env python3
"""
Task 2: ISI-MAC h-sweep at fixed SNR=6 dB, N=32.

For each h in {0.2, 0.3, 0.5, 0.7} evaluate:
  (a) Chained NPD BiGRU (trained at h=0.3) -- generalization test
  (b) Chained trellis SC (handles any h analytically)
  (c) Memoryless SC (treats ISI-MAC as regular GaussianMAC)

1000 codewords, Wilson CIs.

Output:
  results/snr_sweep/isi_mac_h_sweep_N32.json
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
from polar.decoder import decode_single
from polar.channels import GaussianMAC
from polar.channels_memory import ISIMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder_trellis_mac_chained import (
    decode_stage1_u as trellis_decode_stage1,
    decode_stage2_v as trellis_decode_stage2,
)
from neural.npd_memory_mac import ChainedNPD_MAC


SNR_DB = 6.0
N = 32
n_log = 5
H_LIST = [0.2, 0.3, 0.5, 0.7]
N_CW = 1000
SEED_BASE = 20260416


def wilson_ci(errs, total, z=1.96):
    if total == 0:
        return (0.0, 1.0)
    p = errs / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_checkpoints_for_N32():
    """Load the BiGRU chained NPD trained at h=0.3."""
    s1p = os.path.join(_ROOT, 'class_c_npd/results/npd_memory_mac',
                       'isi_mac_bigru_L1_s1_N32_best.pt')
    s2p = os.path.join(_ROOT, 'class_c_npd/results/npd_memory_mac',
                       'isi_mac_bigru_L1_s2_N32_best.pt')
    model = ChainedNPD_MAC(d=16, hidden=64, n_layers=2,
                           encoder_type='bigru', gru_layers=1)
    s1 = torch.load(s1p, weights_only=False, map_location='cpu')
    s2 = torch.load(s2p, weights_only=False, map_location='cpu')
    model.stage1.load_state_dict(s1['state_dict'])
    model.stage2.load_state_dict(s2['state_dict'])
    model.stage1.eval()
    model.stage2.eval()
    return model, os.path.basename(s1p), os.path.basename(s2p)


def load_design_N32():
    """Load the Class C design used for NPD training (GMAC proxy @ 6 dB)."""
    # The training used gmac_C_n5_snr6dB.npz as the design proxy.
    n = n_log
    dp = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr6dB.npz')
    Au, Av, fu_1idx, fv_1idx, _, _, _ = design_from_file(dp, n, 7, 15)
    return sorted(Au), sorted(Av), fu_1idx, fv_1idx


def eval_npd(model, channel, Au, Av, n_cw=N_CW, seed_off=0):
    """Chained NPD BiGRU on ISI-MAC."""
    br = bit_reversal_perm(n_log)
    br_t = torch.from_numpy(br.copy()).long()
    fu_set = {p - 1 for p in range(1, N + 1) if p not in Au}
    fv_set = {p - 1 for p in range(1, N + 1) if p not in Av}
    rng = np.random.default_rng(SEED_BASE + seed_off)
    errs = 0
    errs_u = 0
    errs_v = 0
    total = 0
    batch = 16
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
            # Stage 1
            emb1 = model.stage1.encode_channel(z_t)
            emb1_npd = emb1[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb1_npd, fu_set)
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)
            # Stage 2
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


def eval_trellis_sc(channel, Au, Av, fu_1idx, fv_1idx, n_cw=N_CW, seed_off=0):
    """Chained trellis SC on ISI-MAC."""
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


def eval_memoryless_sc(isi_channel, Au, Av, fu_1idx, fv_1idx, n_cw=N_CW, seed_off=0):
    """Memoryless SC: encode with ISI channel but decode as if GaussianMAC.

    The sampler uses the real ISI-MAC (with memory), but the decoder treats
    each z[i] as an independent GaussianMAC observation with the SAME sigma2.
    This is the naive baseline: how much does ignoring ISI cost?
    """
    sigma2 = isi_channel.sigma2
    gmac = GaussianMAC(sigma2=sigma2)
    b = make_path(N, N)  # Class C (chained)

    rng = np.random.default_rng(SEED_BASE + seed_off + 42)
    errs = 0
    errs_u = 0
    errs_v = 0
    t0 = time.time()
    for _ in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p - 1] = rng.integers(0, 2)
        for p in Av: vf[p - 1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        # Sample via ISI channel (sample_batch needs 2D inputs)
        x_arr = np.array(x, dtype=int).reshape(1, -1)
        y_arr = np.array(y, dtype=int).reshape(1, -1)
        z = isi_channel.sample_batch(x_arr, y_arr)[0]  # (N,) 1D
        # Decode via memoryless GaussianMAC SC (Class C chained)
        u_dec, v_dec = decode_single(N, z.tolist(), b, fu_1idx, fv_1idx, gmac,
                                      log_domain=True)
        u_wrong = any(int(u_dec[p - 1]) != int(uf[p - 1]) for p in Au)
        v_wrong = any(int(v_dec[p - 1]) != int(vf[p - 1]) for p in Av)
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
            'N': N, 'n_log': n_log, 'snr_db': SNR_DB,
            'h_list': H_LIST, 'n_cw': N_CW,
            'channel': 'ISIMAC h-sweep, SNR=6 dB',
            'npd_training_h': 0.3,
            'note': 'NPD BiGRU was trained at h=0.3 only; other h values test generalization.',
        },
        'results': {},
    }
    Au, Av, fu_1idx, fv_1idx = load_design_N32()
    out['config']['Au'] = Au
    out['config']['Av'] = Av
    out['config']['ku'] = len(Au)
    out['config']['kv'] = len(Av)
    print(f'Au={Au}')
    print(f'Av={Av}')

    model, s1_name, s2_name = load_checkpoints_for_N32()
    out['config']['npd_s1_ckpt'] = s1_name
    out['config']['npd_s2_ckpt'] = s2_name

    out_path = os.path.join(_ROOT, 'results', 'snr_sweep', 'isi_mac_h_sweep_N32.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _save():
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)

    sigma2 = 10 ** (-SNR_DB / 10)
    for i, h in enumerate(H_LIST):
        print(f'\n=== h = {h} ===')
        channel = ISIMAC(sigma2=sigma2, h=h)
        slot = {}
        # NPD
        r_npd = eval_npd(model, channel, Au, Av, n_cw=N_CW, seed_off=i * 1000)
        ci = wilson_ci(r_npd['errs_total'], r_npd['n_cw'])
        r_npd['ci_lo'] = ci[0]; r_npd['ci_hi'] = ci[1]
        slot['chained_npd_bigru'] = r_npd
        print(f'  NPD       BLER={r_npd["bler"]:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
              f'u={r_npd["errs_u"]} v={r_npd["errs_v"]} t={r_npd["time_s"]}s')
        out['results'][f'h={h}'] = slot
        _save()

        # Chained trellis SC
        r_sc = eval_trellis_sc(channel, Au, Av, fu_1idx, fv_1idx,
                                n_cw=N_CW, seed_off=i * 1000)
        ci = wilson_ci(r_sc['errs_total'], r_sc['n_cw'])
        r_sc['ci_lo'] = ci[0]; r_sc['ci_hi'] = ci[1]
        slot['chained_trellis_sc'] = r_sc
        print(f'  TrellisSC BLER={r_sc["bler"]:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
              f'u={r_sc["errs_u"]} v={r_sc["errs_v"]} t={r_sc["time_s"]}s')
        _save()

        # Memoryless SC (decoder ignores ISI)
        r_ml = eval_memoryless_sc(channel, Au, Av, fu_1idx, fv_1idx,
                                   n_cw=N_CW, seed_off=i * 1000)
        ci = wilson_ci(r_ml['errs_total'], r_ml['n_cw'])
        r_ml['ci_lo'] = ci[0]; r_ml['ci_hi'] = ci[1]
        slot['memoryless_sc'] = r_ml
        print(f'  MemlessSC BLER={r_ml["bler"]:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
              f'u={r_ml["errs_u"]} v={r_ml["errs_v"]} t={r_ml["time_s"]}s')
        _save()

    print(f'\nSaved to {out_path}')
    return out


if __name__ == '__main__':
    main()
