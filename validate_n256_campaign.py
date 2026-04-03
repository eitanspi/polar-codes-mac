#!/usr/bin/env python3
"""
Validate N=256 GMAC campaign result: NN-SC vs SC decoder, 5000 codewords.
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.eval import MACEval

# ── Import model architecture ──
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

class SimpleMLP_Gmac(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = 16
        self.z_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 32), torch.nn.ELU(), torch.nn.Linear(32, 16))
        self.tree = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)

# ── Parameters ──
N = 256
n = int(math.log2(N))  # 8
ku, kv = 123, 123
path_i = 128  # Class B
SNR_DB = 6.0
SIGMA2 = 10**(-SNR_DB / 10)
N_CW = 5000

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), 'designs')
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')

# ── Setup ──
channel = GaussianMAC(sigma2=SIGMA2)
Au, Av, fu, fv, _, _, _ = design_from_file(
    os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz'), n, ku, kv)
b = make_path(N, path_i)
br = torch.from_numpy(bit_reversal_perm(n)).long()

print(f"N={N}, ku={ku}, kv={kv}, path_i={path_i}, SNR={SNR_DB}dB")
print(f"|Au|={len(Au)}, |Av|={len(Av)}, |fu|={len(fu)}, |fv|={len(fv)}")
print(f"Evaluating {N_CW} codewords\n")

# ── Load NN model ──
ckpt_path = os.path.join(SAVE_DIR, 'campaign_n256_sched_best.pt')
print(f"Loading checkpoint: {ckpt_path}")
model = SimpleMLP_Gmac()
sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
model.load_state_dict(fixed, strict=False)
model.eval()
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ── NN-SC Evaluation ──
print("=" * 60)
print("NN-SC Decoder Evaluation")
print("=" * 60)

t0 = time.time()
rng = np.random.default_rng(42)
nn_errors = 0
nn_total = 0
batch_size = 4

while nn_total < N_CW:
    actual = min(batch_size, N_CW - nn_total)
    uf = np.zeros((actual, N), dtype=int)
    vf = np.zeros((actual, N), dtype=int)
    for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
    for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
    xf = polar_encode_batch(uf)
    yf = polar_encode_batch(vf)
    zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

    with torch.no_grad():
        root = model.z_encoder(zf.unsqueeze(-1))[:, br]
        _, _, uh, vh, _ = model.tree(z=None, b=b, frozen_u=fu, frozen_v=fv, root_emb=root)

    for i in range(actual):
        err = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
              any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
        if err:
            nn_errors += 1
    nn_total += actual

    if nn_total % 500 == 0:
        print(f"  NN-SC: {nn_total}/{N_CW} done, errors={nn_errors}, "
              f"BLER={nn_errors/nn_total:.4f}")

nn_bler = nn_errors / nn_total
nn_time = time.time() - t0
print(f"\nNN-SC Result: BLER = {nn_bler:.6f} ({nn_errors}/{nn_total}), "
      f"time={nn_time:.1f}s\n")

# ── SC Decoder Evaluation ──
print("=" * 60)
print("SC Decoder Evaluation (interleaved backend)")
print("=" * 60)

t0 = time.time()
sc_rng = np.random.default_rng(42)  # same seed for fair comparison
evaluator = MACEval(channel, log_domain=True, backend='interleaved', rng=sc_rng)
ber_u, ber_v, sc_bler = evaluator.run(N, b, Au, Av, fu, fv,
                                       n_codewords=N_CW, batch_size=25,
                                       verbose=True)
sc_time = time.time() - t0
print(f"\nSC Result: BLER = {sc_bler:.6f}, ber_u={ber_u:.6e}, ber_v={ber_v:.6e}, "
      f"time={sc_time:.1f}s\n")

# ── Summary ──
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  N={N}, ku={ku}, kv={kv}, Class B (path_i={path_i}), SNR={SNR_DB}dB")
print(f"  Codewords: {N_CW}")
print(f"  NN-SC BLER: {nn_bler:.6f} ({nn_errors} errors)")
print(f"  SC    BLER: {sc_bler:.6f}")
if sc_bler > 0:
    print(f"  Ratio NN/SC: {nn_bler/sc_bler:.3f}")
print(f"  NN time: {nn_time:.1f}s, SC time: {sc_time:.1f}s")
print("=" * 60)
