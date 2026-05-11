#!/usr/bin/env python3
"""
SC BLER baselines for 3 MAC channels with memory, N=16 to 1024.

Strategy: use GMAC Class C frozen set designs (which give good BLER curves),
transmit over memory channels, decode with trellis SC decoder.

The trellis decoder handles the memory optimally, so BLER should be
comparable to GMAC SC but with the memory affecting the effective SNR.

Channels:
  1. ISI-MAC h=0.5 — moderate intersymbol interference
  2. ISI-MAC h=0.8 — strong ISI
  3. Gilbert-Elliott MAC — bursty AWGN with good/bad states
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from polar.encoder import polar_encode_batch
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder_trellis import decode_single
from polar.channels_memory import ISIMAC
from polar.channels_memory_new import GilbertElliottMAC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd', 'results')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

# GMAC Class C rates from existing project
RATES = {
    16:  {'ku': 4,   'kv': 7},
    32:  {'ku': 7,   'kv': 15},
    64:  {'ku': 15,  'kv': 29},
    128: {'ku': 30,  'kv': 58},
    256: {'ku': 59,  'kv': 117},
    512: {'ku': 119, 'kv': 233},
    1024: {'ku': 238, 'kv': 467},
}

CHANNELS = {
    'isi_h05': {
        'instance': ISIMAC(sigma2=SIGMA2, h=0.5),
        'label': 'ISI-MAC (h=0.5)',
    },
    'isi_h08': {
        'instance': ISIMAC(sigma2=SIGMA2, h=0.8),
        'label': 'ISI-MAC (h=0.8)',
    },
    'ge_mac': {
        'instance': GilbertElliottMAC(p_gb=0.08, p_bg=0.4,
                                       sigma2_good=SIGMA2 * 0.8,
                                       sigma2_bad=SIGMA2 * 5.0),
        'label': 'Gilbert-Elliott MAC',
    },
}


def evaluate_bler(channel, N, b, fu, fv, Au, Av, n_cw, seed=999):
    rng = np.random.default_rng(seed)
    errs = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au:
            u[p - 1] = rng.integers(0, 2)
        for p in Av:
            v[p - 1] = rng.integers(0, 2)
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        u_d, v_d = decode_single(N, z, b, fu, fv, channel)
        ue = any(u_d[p - 1] != u[p - 1] for p in Au)
        ve = any(v_d[p - 1] != v[p - 1] for p in Av)
        if ue or ve:
            errs += 1
    return errs / n_cw


def main():
    all_results = {}

    for ch_name, ch_cfg in CHANNELS.items():
        channel = ch_cfg['instance']
        print(f'\n{"#" * 60}')
        print(f'# {ch_cfg["label"]}')
        print(f'{"#" * 60}')

        ch_results = {}

        for N in [16, 32, 64, 128, 256, 512, 1024]:
            n = int(np.log2(N))
            b = make_path(N, N)  # Class C

            # Load GMAC design
            rates = RATES[N]
            ku, kv = rates['ku'], rates['kv']
            design_path = os.path.join(DESIGNS_DIR, f'gmac_C_n{n}_snr{SNR_DB:.0f}dB.npz')
            if not os.path.exists(design_path):
                print(f'  N={N}: no design file, skipping')
                continue
            Au, Av, fu, fv, _, _, _ = design_from_file(design_path, n, ku, kv)

            # Evaluate BLER
            n_cw = max(200, min(5000, 20000 // max(1, N // 16)))
            print(f'  N={N} (ku={ku}, kv={kv}, {n_cw} cw)...', end=' ', flush=True)
            t0 = time.time()
            bler = evaluate_bler(channel, N, b, fu, fv, Au, Av, n_cw)
            elapsed = time.time() - t0

            print(f'BLER={bler:.4f} ({elapsed:.0f}s)', flush=True)

            ch_results[N] = {
                'N': N, 'ku': ku, 'kv': kv,
                'bler': bler, 'n_cw': n_cw,
                'time_s': elapsed,
            }

            # Save incrementally
            all_results[ch_name] = {
                'label': ch_cfg['label'],
                'results': ch_results,
            }
            with open(os.path.join(RESULTS_DIR, 'memory_channel_baselines.json'), 'w') as f:
                json.dump(all_results, f, indent=2)

        print(f'\n  {ch_cfg["label"]} summary:')
        for N in sorted(ch_results.keys()):
            r = ch_results[N]
            print(f'    N={N:4d}: BLER={r["bler"]:.4f}')

    print(f'\n{"=" * 60}')
    print('ALL DONE')
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
