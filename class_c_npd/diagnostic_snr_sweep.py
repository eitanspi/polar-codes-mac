"""
Diagnostic: train Stage 1 NPD on the mixture channel at multiple SNRs.

Hypothesis: maybe the NPD plateau on the mixture channel is operating-point
sensitive — at higher SNR, individual positions are more reliable, and the
NPD has more "approximation slack". If high SNR works, the issue is fragility
near marginal operating points, not architecture.

Test:
  N=32, ku=7
  Train Stage 1 at SNR = 6, 8, 10, 12 dB
  Compare NPD BLER vs SC BLER at each SNR
  Look for a regime where the gap closes
"""
import os
import sys
import time
import json
import math
import torch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.decoder import decode_batch
from polar.design import make_path
from polar.design_mc import design_from_file

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.training.train_stage import generate_stage1_batch, evaluate_stage
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design


N = 32
n = 5
KU, KV = 7, 15
ITERS = 30000
BATCH = 64
LR = 3e-4
SNR_LIST = [6.0, 8.0, 10.0]


def get_sc_stage1_bler(snr_db, n_cw=2000):
    """Compute SC BLER for Stage 1 (U on mixture channel) at this SNR."""
    sigma2 = 10 ** (-snr_db / 10)
    channel = GaussianMAC(sigma2=sigma2)

    # Load design at this SNR (use n5_snr{snr}dB)
    snr_int = int(round(snr_db))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{snr_int}dB.npz')
    if not os.path.exists(path):
        return None

    Au, Av, fu, fv, _, _, _ = design_from_file(path, n, KU, KV)
    b = make_path(N, N)
    frozen_u = {i: 0 for i in range(1, N+1) if i not in Au}
    frozen_v = {i: 0 for i in range(1, N+1) if i not in Av}

    errs_u = 0
    rng = np.random.default_rng(42)
    for bstart in range(0, n_cw, 200):
        actual = min(200, n_cw - bstart)
        uf = np.zeros((actual, N), dtype=int)
        vf = np.zeros((actual, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        z = channel.sample_batch(xf, yf)
        res = decode_batch(N, z, b, frozen_u, frozen_v, channel)
        for i, (u_dec, v_dec) in enumerate(res):
            if any(u_dec[p-1] != uf[i, p-1] for p in Au):
                errs_u += 1
    return errs_u / n_cw


def train_stage1_at_snr(snr_db):
    print(f'\n=== Stage 1 @ SNR = {snr_db} dB ===')
    sigma2 = 10 ** (-snr_db / 10)
    channel = build_channel('gmac', sigma2=sigma2)

    snr_int = int(round(snr_db))
    Au, Av, fu, fv, pe_u, pe_v = load_class_c_design(
        'gmac', n, snr_db=snr_int, ku=KU, kv=KV,
    )

    max_pe_u = max(pe_u[p-1] for p in Au)
    print(f'  N={N}, ku={KU}, max U Pe = {max_pe_u:.5f}')

    torch.manual_seed(42)
    model = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    best_bler = 1.0

    model.train()
    for it in range(1, ITERS + 1):
        _, features_npd, cw_npd = generate_stage1_batch(channel, N, Au, BATCH, rng, Av)
        ft = torch.from_numpy(features_npd).float()
        if ft.dim() == 2:
            ft = ft.unsqueeze(-1)
        cw = torch.from_numpy(cw_npd).long()
        emb = model.encode_channel(ft)
        loss = model.fast_ce(emb, cw)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % 5000 == 0:
            bler = evaluate_stage(model, channel, N, Au, fu,
                                   generate_stage1_batch, 500, seed=999, other_info=Av)
            avg_loss = float(np.mean(losses[-500:]))
            elapsed = (time.time() - t0) / 60
            if bler < best_bler:
                best_bler = bler
            print(f'  [{it:>5}/{ITERS}] loss={avg_loss:.4f} BLER={bler:.4f} (best={best_bler:.4f}) {elapsed:.1f}min')

    return best_bler


def main():
    print(f'SNR sweep diagnostic: N={N}, ku={KU}, Stage 1 only')
    print(f'Hypothesis: NPD gap to SC narrows at higher SNR')
    print(f'Iters per SNR: {ITERS}')

    results = {}
    for snr_db in SNR_LIST:
        # SC reference at this SNR
        sc_bler = get_sc_stage1_bler(snr_db, n_cw=2000)
        print(f'\n[{snr_db} dB] SC Stage 1 BLER = {sc_bler:.5f}')

        # Train NPD
        npd_bler = train_stage1_at_snr(snr_db)
        ratio = npd_bler / sc_bler if sc_bler > 0 else float('inf')
        print(f'  NPD BLER = {npd_bler:.4f}  ratio = {ratio:.2f}x')

        results[snr_db] = {
            'sc_bler': float(sc_bler),
            'npd_bler': float(npd_bler),
            'ratio': float(ratio),
        }

    # Summary
    print('\n' + '=' * 60)
    print('SNR SWEEP RESULTS')
    print('=' * 60)
    print(f'{"SNR(dB)":<12}{"SC":<12}{"NPD":<12}{"ratio":<10}')
    print('-' * 60)
    for snr_db, r in results.items():
        print(f'{snr_db:<12}{r["sc_bler"]:<12.5f}{r["npd_bler"]:<12.5f}{r["ratio"]:<10.2f}')

    with open(os.path.join(_HERE, 'results', 'snr_sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to class_c_npd/results/snr_sweep_results.json')


if __name__ == '__main__':
    main()
