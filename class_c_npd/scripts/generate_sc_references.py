"""
Generate SC reference BLER tables for Class C across channels and N values.

For each channel:
  - Compute analytical SC BLER on the chained Class C decoder
  - Use ku, kv chosen as 50% of capacity per user
  - Use 5000 codewords per (channel, N) point
  - Output a JSON table

This gives us all the reference numbers needed to compare NPD against SC
for the multi-channel evaluation.
"""
from __future__ import annotations
import os
import sys
import time
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC, BEMAC, ABNMAC
from polar.decoder import decode_batch
from polar.design import make_path
from polar.design_mc import design_from_file


# ─── Per-channel capacity values ─────────────────────────────────────────────

CHANNEL_CAPACITIES = {
    # GMAC at SNR=6dB
    ('gmac', 6.0): {'I_X_Z': 0.4645, 'I_Y_ZX': 0.9119, 'I_X_ZY': 0.9119, 'I_Y_Z': 0.4645,
                    'sigma2': 10**(-6/10)},
    # BEMAC: Z = X + Y, |Z|=3
    # I(X;Z) = ?  For X,Y uniform, Z is {0,1,2} with probs {1/4, 1/2, 1/4}
    # H(Z) = -1/4*log(1/4) - 1/2*log(1/2) - 1/4*log(1/4) = 1/2 + 1/2 = 1.5 bits
    # H(Z|X) = (1/2)[H(Z|X=0) + H(Z|X=1)]
    #        = (1/2)[H(Y) + H(Y+1)] = (1/2)[1 + 1] = 1
    # I(X;Z) = 1.5 - 1 = 0.5
    # I(Y;Z|X) = H(Z|X) - H(Z|X,Y) = 1 - 0 = 1
    # I(X,Y;Z) = H(Z) - H(Z|X,Y) = 1.5 - 0 = 1.5
    ('bemac', None): {'I_X_Z': 0.5, 'I_Y_ZX': 1.0, 'I_X_ZY': 1.0, 'I_Y_Z': 0.5},
}


def get_design_path(channel_name, n, snr_db=None):
    """Find the .npz file for this channel/n."""
    designs_dir = os.path.join(_ROOT, 'designs')
    if channel_name == 'gmac':
        return os.path.join(designs_dir, f'gmac_C_n{n}_snr{int(round(snr_db))}dB.npz')
    elif channel_name == 'bemac':
        return os.path.join(designs_dir, f'bemac_C_n{n}.npz')
    elif channel_name == 'abnmac':
        return os.path.join(designs_dir, f'abnmac_C_n{n}.npz')
    raise ValueError(f"Unknown channel: {channel_name}")


def make_channel(channel_name, snr_db=None):
    if channel_name == 'gmac':
        return GaussianMAC(sigma2=10 ** (-snr_db / 10))
    elif channel_name == 'bemac':
        return BEMAC()
    elif channel_name == 'abnmac':
        return ABNMAC(p_x=0.1, p_y=0.1)
    raise ValueError(f"Unknown channel: {channel_name}")


def compute_sc_bler(channel_name, n, snr_db, frac_capacity, n_cw=5000):
    """Compute analytical SC chained Class C BLER for one (channel, N, frac) point."""
    N = 1 << n

    # Get capacities for this channel
    cap_key = (channel_name, snr_db) if channel_name == 'gmac' else (channel_name, None)
    if cap_key not in CHANNEL_CAPACITIES:
        return None
    caps = CHANNEL_CAPACITIES[cap_key]

    ku = round(frac_capacity * caps['I_X_Z'] * N)
    kv = round(frac_capacity * caps['I_Y_ZX'] * N)
    if ku < 1 or kv < 1:
        return None

    # Load frozen set design
    design_path = get_design_path(channel_name, n, snr_db)
    if not os.path.exists(design_path):
        return None

    Au, Av, fu, fv, pe_u, pe_v, _ = design_from_file(design_path, n, ku, kv)
    b = make_path(N, N)
    frozen_u = {i: 0 for i in range(1, N + 1) if i not in Au}
    frozen_v = {i: 0 for i in range(1, N + 1) if i not in Av}

    channel = make_channel(channel_name, snr_db)
    rng = np.random.default_rng(42)

    errs = 0
    bs = 200
    for bstart in range(0, n_cw, bs):
        actual = min(bs, n_cw - bstart)
        uf = np.zeros((actual, N), dtype=int)
        vf = np.zeros((actual, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, actual)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, actual)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        z = channel.sample_batch(xf, yf)
        try:
            res = decode_batch(N, z, b, frozen_u, frozen_v, channel)
        except Exception as e:
            print(f'    decode failed: {e}')
            return None
        for i, (u_dec, v_dec) in enumerate(res):
            ue = any(u_dec[p - 1] != uf[i, p - 1] for p in Au)
            ve = any(v_dec[p - 1] != vf[i, p - 1] for p in Av)
            if ue or ve:
                errs += 1

    return {
        'n': n, 'N': N, 'ku': ku, 'kv': kv,
        'R_u': ku / N, 'R_v': kv / N,
        'sc_bler': errs / n_cw, 'n_cw': n_cw,
    }


def main():
    out = {}
    fracs = [0.50, 0.65, 0.75]

    # GMAC at SNR=6dB
    for n in [4, 5, 6, 7, 8, 9, 10]:
        N = 1 << n
        for frac in fracs:
            key = f'gmac_snr6.0_frac{frac}_N{N}'
            print(f'  {key}...', flush=True)
            r = compute_sc_bler('gmac', n, 6.0, frac, n_cw=2000)
            if r is not None:
                out[key] = {'channel': 'gmac', 'snr_db': 6.0, 'frac_capacity': frac, **r}
                print(f'    ku={r["ku"]} kv={r["kv"]} BLER={r["sc_bler"]:.5f}', flush=True)

    # BEMAC
    for n in [4, 5, 6, 7, 8, 9]:
        N = 1 << n
        for frac in fracs:
            key = f'bemac_frac{frac}_N{N}'
            print(f'  {key}...', flush=True)
            r = compute_sc_bler('bemac', n, None, frac, n_cw=2000)
            if r is not None:
                out[key] = {'channel': 'bemac', **r}
                print(f'    ku={r["ku"]} kv={r["kv"]} BLER={r["sc_bler"]:.5f}', flush=True)

    out_path = os.path.join(_ROOT, 'class_c_npd', 'results', 'sc_references_multi_channel.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
