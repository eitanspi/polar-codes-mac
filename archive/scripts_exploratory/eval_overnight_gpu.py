#!/usr/bin/env python3
"""
eval_overnight_gpu.py
=====================
Priority 1: Eval the overnight GPU checkpoint (d=64, h=128 BiGRU, N=512).
ISI-MAC, ku=119, kv=233, 2000 CW.
Compare to previous best (0.108).
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
from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from neural.npd_memory_mac import ChainedNPD_MAC

# Config
SNR_DB = 6.0
ISI_H = 0.3
N = 512
KU = 119
KV = 233
D = 64
HIDDEN = 128
N_LAYERS = 2
GRU_LAYERS = 1
N_CW = 2000
BATCH = 16

CKPT_PATH = '/tmp/overnight_250k.pt'
RESULTS_DIR = os.path.join(_ROOT, 'results', 'reliable_evals')
os.makedirs(RESULTS_DIR, exist_ok=True)


def wilson_ci(errs, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = errs / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0, centre-margin), min(1, centre+margin))


def make_channel():
    return ISIMAC.from_snr_db(SNR_DB, h=ISI_H)


def load_design():
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, _path_i = design_from_file(path, n, KU, KV)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_set = {p-1 for p in range(1, N+1) if p not in Au}
    frozen_v_set = {p-1 for p in range(1, N+1) if p not in Av}
    return Au, Av, frozen_u_set, frozen_v_set


def make_batch(channel, Au, Av, batch, rng):
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au: u_msg[:, p-1] = rng.integers(0, 2, batch)
    for p in Av: v_msg[:, p-1] = rng.integers(0, 2, batch)
    x_phys = polar_encode_batch(u_msg.astype(int))
    y_phys = polar_encode_batch(v_msg.astype(int))
    z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
    return u_msg, v_msg, np.asarray(z, dtype=np.float32), x_phys, y_phys


def eval_stage1(model_s1, channel, Au, Av, frozen_u_set, n_cw, batch, seed=999):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()
    model_s1.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    errs = 0
    total = 0
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, _, z, _, _ = make_batch(channel, Au, Av, actual, rng)
            z_t = torch.from_numpy(z)
            emb = model_s1.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            u_hat = model_s1.tree.decode(emb_npd, frozen_u_set)
            for i in range(actual):
                if any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au):
                    errs += 1
            total += actual
            if total % 500 == 0:
                print(f'  [{total}/{n_cw}] errs={errs}', flush=True)
    return errs, total


def main():
    print(f'Evaluating overnight GPU checkpoint: {CKPT_PATH}')
    print(f'N={N}, ku={KU}, kv={KV}, d={D}, h={HIDDEN}')
    print(f'ISI-MAC SNR={SNR_DB}dB, h={ISI_H}')
    print(f'Codewords: {N_CW}')

    if not os.path.exists(CKPT_PATH):
        print(f'ERROR: Checkpoint not found at {CKPT_PATH}')
        return

    channel = make_channel()
    Au, Av, fu_set, fv_set = load_design()
    print(f'|Au|={len(Au)}, |Av|={len(Av)}')

    # Load model
    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS,
                           encoder_type='bigru', gru_layers=GRU_LAYERS)
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location='cpu')
    print(f'Checkpoint keys: {list(ckpt.keys())}')

    # The GPU checkpoint may have different formats
    if 'state_dict' in ckpt:
        try:
            model.stage1.load_state_dict(ckpt['state_dict'])
            print('Loaded state_dict into stage1')
        except Exception as e:
            print(f'state_dict load failed: {e}')
            return
    elif 'z_encoder' in ckpt and 'tree' in ckpt:
        # Split format: z_encoder and tree state dicts
        try:
            model.stage1.z_encoder.load_state_dict(ckpt['z_encoder'])
            model.stage1.tree.load_state_dict(ckpt['tree'])
            print('Loaded z_encoder + tree into stage1')
        except Exception as e:
            print(f'Split load failed: {e}')
            # Try with prefix mapping
            try:
                sd = {}
                for k, v in ckpt['z_encoder'].items():
                    sd[f'z_encoder.{k}'] = v
                for k, v in ckpt['tree'].items():
                    sd[f'tree.{k}'] = v
                model.stage1.load_state_dict(sd)
                print('Loaded with prefix mapping into stage1')
            except Exception as e2:
                print(f'Prefix mapping also failed: {e2}')
                return
    else:
        # Try as full state dict
        try:
            model.stage1.load_state_dict(ckpt)
            print('Loaded raw dict into stage1')
        except Exception as e:
            print(f'Raw load failed: {e}')
            return

    print(f'Params: {model.count_parameters():,}')

    # Eval stage 1
    t0 = time.time()
    errs, total = eval_stage1(model.stage1, channel, Au, Av, fu_set, N_CW, BATCH)
    elapsed = time.time() - t0
    bler = errs / total
    ci_lo, ci_hi = wilson_ci(errs, total)

    print(f'\n{"="*60}')
    print(f'Stage 1 BLER = {bler:.4f} ({errs}/{total})')
    print(f'95% CI: [{ci_lo:.5f}, {ci_hi:.5f}]')
    print(f'Previous best: 0.108')
    print(f'Improvement: {"YES" if bler < 0.108 else "NO"} ({bler/0.108:.2f}x)')
    print(f'Time: {elapsed:.1f}s')
    print(f'{"="*60}')

    result = {
        'model': 'overnight_gpu_d64_h128_N512',
        'checkpoint': os.path.basename(CKPT_PATH),
        'N': N, 'ku': KU, 'kv': KV,
        'd': D, 'hidden': HIDDEN,
        'channel': 'isi_mac',
        'snr_db': SNR_DB,
        'n_cw': N_CW,
        'errs': errs,
        'bler': float(bler),
        'ci_low': float(ci_lo),
        'ci_high': float(ci_hi),
        'previous_best': 0.108,
        'time_sec': float(elapsed),
        'iter': ckpt.get('iter', 'unknown'),
    }

    out_path = os.path.join(RESULTS_DIR, 'overnight_gpu_N512_eval.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
