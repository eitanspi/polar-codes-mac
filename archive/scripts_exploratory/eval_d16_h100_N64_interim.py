#!/usr/bin/env python3
"""
eval_d16_h100_N64_interim.py
=============================
Evaluate the current best d=16 h=100 model at N=64 with 5000 CW
to get a reliable BLER estimate.
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
from neural.npd_memory_mac import MemoryStageNPD

N = 64
KU, KV = 15, 29
SNR_DB = 6.0
ISI_H = 0.3
D = 16
HIDDEN = 100

CKPT_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')
CKPT = os.path.join(CKPT_DIR, 'd16_h100_standalone_s1_N64_best.pt')

def wilson_ci(k, n, z=1.96):
    p = k / n
    d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    w = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / d
    return max(0, c-w), min(1, c+w)


def main():
    if not os.path.exists(CKPT):
        print(f"Checkpoint not found: {CKPT}")
        return

    channel = ISIMAC.from_snr_db(SNR_DB, h=ISI_H)

    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, KU, KV)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_set = {p-1 for p in range(1, N+1) if p not in Au}

    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()

    # Build model
    model = MemoryStageNPD(d=D, hidden=HIDDEN, n_layers=2,
                           encoder_type='bigru', extra_dim=0, gru_layers=1)

    # Load checkpoint
    sd = torch.load(CKPT, weights_only=False, map_location='cpu')
    model.load_state_dict(sd['state_dict'])
    print(f"Loaded checkpoint from iter {sd.get('iter', '?')}, "
          f"reported BLER={sd.get('best_bler', '?')}")

    model.eval()
    rng = np.random.default_rng(999)
    np.random.seed(999)

    n_cw = 5000
    batch = 32
    errs = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg = np.zeros((actual, N), dtype=np.int8)
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Au:
                u_msg[:, p-1] = rng.integers(0, 2, actual)
            for p in Av:
                v_msg[:, p-1] = rng.integers(0, 2, actual)
            x_phys = polar_encode_batch(u_msg.astype(int))
            y_phys = polar_encode_batch(v_msg.astype(int))
            z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
            z_t = torch.from_numpy(z.astype(np.float32))

            emb = model.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            u_hat = model.tree.decode(emb_npd, frozen_u_set)

            for i in range(actual):
                if any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au):
                    errs += 1
            total += actual
            if total % 1000 == 0:
                print(f'  {total}/{n_cw}, errs={errs}', flush=True)

    elapsed = time.time() - t0
    bler = errs / total
    ci_lo, ci_hi = wilson_ci(errs, total)

    print(f'\nN={N} d={D} h={HIDDEN} STANDALONE S1 eval:')
    print(f'  BLER = {bler:.4f} ({errs}/{total})')
    print(f'  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]')
    print(f'  Time: {elapsed:.1f}s')
    print(f'\n  For comparison:')
    print(f'  d=16 h=64 (old):   0.0460 (229/5000)')
    print(f'  Chained trellis SC: 0.0407 (407/10000)')
    print(f'  Joint trellis SC:   0.0260 (262/10000)')

    # Save result
    out_dir = os.path.join(_ROOT, 'results', 'reliable_evals')
    out_path = os.path.join(out_dir, 'd16_h100_N64_interim.json')
    result = {
        'N': N, 'd': D, 'hidden': HIDDEN, 'encoder': 'bigru',
        'bler': bler, 'errs': errs, 'n_cw': total,
        'ci': [ci_lo, ci_hi],
        'ckpt_iter': sd.get('iter'),
        'ckpt_reported_bler': sd.get('best_bler'),
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved: {out_path}')


if __name__ == '__main__':
    main()
