#!/usr/bin/env python3
"""Quick trellis SC eval at N=512 for ISI-MAC with 2000 CW."""
import os, sys, time, json, math
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from polar.encoder import polar_encode_batch
from polar.decoder_trellis_mac_chained import decode_chained

N = 512
n = int(math.log2(N))
SNR_DB = 6.0
ku, kv = 119, 233

ch = ISIMAC.from_snr_db(SNR_DB, h=0.3)
path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
Au_list, Av_list, _, _, _, _, _ = design_from_file(path, n, ku, kv)
Au = sorted(Au_list)
Av = sorted(Av_list)
fu = {p: 0 for p in range(1, N+1) if p not in Au}
fv = {p: 0 for p in range(1, N+1) if p not in Av}

n_cw = 2000
rng = np.random.default_rng(1001)
np.random.seed(1001)

errs_u = errs_v = errs_total = 0
t0 = time.time()
for i in range(n_cw):
    u = np.zeros(N, dtype=int)
    v = np.zeros(N, dtype=int)
    for p in Au: u[p-1] = rng.integers(0, 2)
    for p in Av: v[p-1] = rng.integers(0, 2)
    x = polar_encode_batch(u[None, :])[0]
    y = polar_encode_batch(v[None, :])[0]
    z = ch.sample_batch(x[None, :].astype(int), y[None, :].astype(int))[0]
    u_hat, v_hat = decode_chained(z, N, fu, fv, ch)
    ue = any(int(u_hat[p-1]) != int(u[p-1]) for p in Au)
    ve = any(int(v_hat[p-1]) != int(v[p-1]) for p in Av)
    if ue: errs_u += 1
    if ve: errs_v += 1
    if ue or ve: errs_total += 1
    if (i+1) % 200 == 0:
        elapsed = time.time() - t0
        print(f'  [{i+1}/{n_cw}] errs={errs_total} bler={errs_total/(i+1):.4f} ({elapsed:.1f}s)')

total_time = time.time() - t0
print(f'\nN=512 Chained Trellis SC: BLER={errs_total/n_cw:.4f} '
      f'(U={errs_u/n_cw:.4f}, V={errs_v/n_cw:.4f}) '
      f'errs={errs_total}/{n_cw} ({total_time:.1f}s)')

result = {
    'N': 512, 'n_cw': n_cw, 'ku': ku, 'kv': kv,
    'bler_total': errs_total/n_cw, 'bler_u': errs_u/n_cw, 'bler_v': errs_v/n_cw,
    'errs_total': errs_total, 'errs_u': errs_u, 'errs_v': errs_v,
}
out_json = os.path.join(_ROOT, 'results', 'reliable_evals', 'isi_mac_trellis_n512.json')
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f'Saved: {out_json}')
