#!/usr/bin/env python3
"""ISI-MAC trellis SC at N=512 with 10K CW for reliable error count."""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch; torch.set_num_threads(4)

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode_batch
from polar.design_mc import design_from_file
from polar.decoder_trellis_mac_chained import decode_chained

N = 512; KU = 119; KV = 233; SNR_DB = 6.0; ISI_H = 0.3; N_CW = 10000
BASE = os.path.join(os.path.dirname(__file__), '..')

n = int(math.log2(N))
ch = ISIMAC.from_snr_db(SNR_DB, h=ISI_H)
Au, Av, _, _, _, _, _ = design_from_file(os.path.join(BASE, f'designs/gmac_C_n{n}_snr6dB.npz'), n, KU, KV)
Au = sorted(Au); Av = sorted(Av)
fu = {p: 0 for p in range(1, N+1) if p not in Au}
fv = {p: 0 for p in range(1, N+1) if p not in Av}

rng = np.random.default_rng(555)
np.random.seed(555)
errs_u = errs_v = errs_total = 0
t0 = time.time()
for i in range(N_CW):
    u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
    for p in Au: u[p-1] = rng.integers(0, 2)
    for p in Av: v[p-1] = rng.integers(0, 2)
    x = polar_encode_batch(u[None,:])[0]; y = polar_encode_batch(v[None,:])[0]
    z = ch.sample_batch(x[None,:].astype(int), y[None,:].astype(int))[0]
    u_dec, v_dec = decode_chained(z, N, fu, fv, ch)
    uw = any(int(u_dec[p-1]) != u[p-1] for p in Au)
    vw = any(int(v_dec[p-1]) != v[p-1] for p in Av)
    if uw: errs_u += 1
    if vw: errs_v += 1
    if uw or vw: errs_total += 1
    if (i+1) % 2000 == 0:
        print(f'  [{i+1}/{N_CW}] errs={errs_total} ({(time.time()-t0)/60:.1f}min)', flush=True)

elapsed = (time.time() - t0) / 60
bler = errs_total / N_CW
print(f'\nN={N}: BLER={bler:.5f} ({errs_total}/{N_CW}) U={errs_u/N_CW:.4f} V={errs_v/N_CW:.4f} ({elapsed:.1f}min)')

result = {'N': N, 'bler': bler, 'errs_total': errs_total, 'errs_u': errs_u, 'errs_v': errs_v,
          'n_cw': N_CW, 'time_min': elapsed}
out = os.path.join(BASE, 'results', 'reliable_evals', 'isi_mac_sc_N512_10kcw.json')
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f'Saved: {out}')
