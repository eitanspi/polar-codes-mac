"""
Error analysis: where does the NPD differ from SC on Stage 1?

For each codeword in a test set:
  1. Run analytical SC decoder → record per-info-position errors
  2. Run NPD decoder → record per-info-position errors
  3. Tabulate:
     - SC-only errors (NPD got it right but SC didn't)
     - NPD-only errors (NPD got it wrong but SC didn't)
     - Both wrong
     - Both right

Goal: characterize the "extra" errors NPD makes vs SC, by:
  - Per-position frequency (which positions does NPD struggle with?)
  - First-error position distribution
  - Correlation with z value at that position (z near 0 = ambiguous?)
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.decoder import decode_batch
from polar.design import make_path
from polar.design_mc import design_from_file

from class_c_npd.models.npd_single_user import NPDSingleUser


def error_analysis(npd_ckpt_path, n=5, snr_db=6.0, ku=7, kv=15, n_cw=2000):
    N = 1 << n
    sigma2 = 10 ** (-snr_db / 10)
    channel = GaussianMAC(sigma2=sigma2)

    # Load design
    design_path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(round(snr_db))}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(design_path, n, ku, kv)
    b = make_path(N, N)
    frozen_u_dict = {i: 0 for i in range(1, N + 1) if i not in Au}
    frozen_v_dict = {i: 0 for i in range(1, N + 1) if i not in Av}

    # Load NPD
    ckpt = torch.load(npd_ckpt_path, weights_only=False, map_location='cpu')
    model = NPDSingleUser(d=ckpt['d'], hidden=ckpt['hidden'],
                          n_layers=ckpt['n_layers'], z_dim=ckpt['z_dim'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'Loaded NPD from {npd_ckpt_path}')

    br = bit_reversal_perm(n)
    fu_natural = {p - 1 for p in range(1, N + 1) if p not in Au}

    # Counters
    sc_errs_per_pos = {p: 0 for p in Au}
    npd_errs_per_pos = {p: 0 for p in Au}
    both_wrong_per_pos = {p: 0 for p in Au}
    sc_only_per_pos = {p: 0 for p in Au}
    npd_only_per_pos = {p: 0 for p in Au}

    sc_block_errs = 0
    npd_block_errs = 0
    sc_first_err_dist = []
    npd_first_err_dist = []

    rng = np.random.default_rng(42)
    print(f'Running error analysis on {n_cw} codewords...')

    with torch.no_grad():
        for trial in range(n_cw):
            u_msg = np.zeros((1, N), dtype=int)
            v_msg = np.zeros((1, N), dtype=int)
            for p in Au: u_msg[0, p - 1] = rng.integers(0, 2)
            for p in Av: v_msg[0, p - 1] = rng.integers(0, 2)
            x = polar_encode_batch(u_msg)
            y = polar_encode_batch(v_msg)
            z = channel.sample_batch(x, y)

            # SC decode
            sc_results = decode_batch(N, z, b, frozen_u_dict, frozen_v_dict, channel)
            u_sc = sc_results[0][0]  # numpy array length N

            # NPD decode (Stage 1 only)
            z_npd = z[:, br]
            ft = torch.from_numpy(z_npd).float().unsqueeze(-1)
            emb = model.encode_channel(ft)
            u_npd_t = model.decode(emb, fu_natural)
            u_npd = u_npd_t[0].numpy()

            # Compare per info position
            sc_block_wrong = False
            npd_block_wrong = False
            sc_first_err = -1
            npd_first_err = -1

            for idx, p in enumerate(Au):
                truth = u_msg[0, p - 1]
                sc_wrong = (u_sc[p - 1] != truth)
                npd_wrong = (u_npd[p - 1] != truth)
                if sc_wrong:
                    sc_errs_per_pos[p] += 1
                    sc_block_wrong = True
                    if sc_first_err < 0:
                        sc_first_err = idx
                if npd_wrong:
                    npd_errs_per_pos[p] += 1
                    npd_block_wrong = True
                    if npd_first_err < 0:
                        npd_first_err = idx
                if sc_wrong and npd_wrong:
                    both_wrong_per_pos[p] += 1
                if sc_wrong and not npd_wrong:
                    sc_only_per_pos[p] += 1
                if npd_wrong and not sc_wrong:
                    npd_only_per_pos[p] += 1

            if sc_block_wrong:
                sc_block_errs += 1
                sc_first_err_dist.append(sc_first_err)
            if npd_block_wrong:
                npd_block_errs += 1
                npd_first_err_dist.append(npd_first_err)

    # Summary
    print()
    print('=' * 70)
    print(f'Error analysis: N={N}, ku={len(Au)}, n_cw={n_cw}')
    print('=' * 70)
    print(f'SC block error rate:  {sc_block_errs / n_cw:.4f}  ({sc_block_errs}/{n_cw})')
    print(f'NPD block error rate: {npd_block_errs / n_cw:.4f}  ({npd_block_errs}/{n_cw})')
    print(f'Ratio NPD/SC: {npd_block_errs / max(sc_block_errs, 1):.2f}x')
    print()

    print('Per-info-position error counts:')
    print(f'{"Au pos":<8}{"SC errs":<10}{"NPD errs":<10}{"Both":<8}{"SC only":<10}{"NPD only":<10}')
    print('-' * 60)
    for p in sorted(Au):
        sc = sc_errs_per_pos[p]
        npd = npd_errs_per_pos[p]
        both = both_wrong_per_pos[p]
        so = sc_only_per_pos[p]
        no = npd_only_per_pos[p]
        print(f'{p:<8}{sc:<10}{npd:<10}{both:<8}{so:<10}{no:<10}')

    total_sc = sum(sc_errs_per_pos.values())
    total_npd = sum(npd_errs_per_pos.values())
    print('-' * 60)
    print(f'{"TOTAL":<8}{total_sc:<10}{total_npd:<10}'
          f'{sum(both_wrong_per_pos.values()):<8}'
          f'{sum(sc_only_per_pos.values()):<10}'
          f'{sum(npd_only_per_pos.values()):<10}')
    print()

    print('First-error info-index distribution:')
    print(f'  SC  first errors: {dict(zip(*np.unique(sc_first_err_dist, return_counts=True)))}')
    print(f'  NPD first errors: {dict(zip(*np.unique(npd_first_err_dist, return_counts=True)))}')
    print()

    extra_npd_errs = total_npd - total_sc
    print(f'NPD has {extra_npd_errs} extra position errors over SC ({extra_npd_errs/n_cw:.4f}/codeword)')
    print(f'  ({sum(npd_only_per_pos.values())} were "NPD only", '
          f'{sum(sc_only_per_pos.values())} were "SC only" — NPD better)')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--snr', type=float, default=6.0)
    parser.add_argument('--ku', type=int, default=7)
    parser.add_argument('--kv', type=int, default=15)
    parser.add_argument('--n_cw', type=int, default=2000)
    args = parser.parse_args()
    error_analysis(args.ckpt, args.n, args.snr, args.ku, args.kv, args.n_cw)


if __name__ == '__main__':
    main()
