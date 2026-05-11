#!/usr/bin/env python3
"""
Measure both BLER and BER for NCG (ncg_gmac_mlp_N256.pt) vs analytical SC
on GMAC Class B, N=256, ku=kv=123, SNR 6 dB, 10000 codewords.

Report:
 - BLER (block-failure rate)
 - BER  (bit-error rate over info positions)
 - bits/failed block
 - ratios NCG/SC
 - histogram of bit errors per failed block
"""

import os
import sys
import math
import time
import numpy as np
import torch

torch.set_num_threads(2)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode, polar_encode_batch
from polar.decoder import decode_single
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac


SNR_DB = 6
SIGMA2 = 10 ** (-SNR_DB / 10)
N = 256
KU = KV = 123
N_CW = 10000
SEED = 42 + N + KU
BATCH_SIZE = 50


def load_design(N, ku, kv, snr_db=SNR_DB):
    n = int(math.log2(N))
    dp = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr{snr_db}dB.npz')
    d = np.load(dp)
    if 'u_error_rates' in d:
        su = np.argsort(d['u_error_rates'])
        sv = np.argsort(d['v_error_rates'])
    else:
        su = d['sorted_u']
        sv = d['sorted_v']
    Au = sorted([int(i + 1) for i in su[:ku]])
    Av = sorted([int(i + 1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_gmac_model(N):
    ckpt = os.path.join(BASE, 'saved_models', f'ncg_gmac_mlp_N{N}.pt')
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    fixed = {}
    for k, v in sd.items():
        nk = k.replace('z_enc.', 'z_encoder.') if k.startswith('z_enc.') else k
        fixed[nk] = v
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def bin_bit_errors(counts):
    """Bin a list of per-block bit-error counts (failed blocks only) into
    bins 1, 2-3, 4-7, 8-15, 16+."""
    bins = {'1': 0, '2-3': 0, '4-7': 0, '8-15': 0, '16+': 0}
    for c in counts:
        if c <= 0:
            continue
        if c == 1:
            bins['1'] += 1
        elif c <= 3:
            bins['2-3'] += 1
        elif c <= 7:
            bins['4-7'] += 1
        elif c <= 15:
            bins['8-15'] += 1
        else:
            bins['16+'] += 1
    return bins


def main():
    print(f"N={N}, ku=kv={KU}, SNR={SNR_DB} dB, sigma2={SIGMA2:.6f}, n_cw={N_CW}", flush=True)

    channel = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)
    Au, Av, fu, fv = load_design(N, KU, KV)
    info_u = np.array(Au, dtype=int) - 1  # 0-indexed positions
    info_v = np.array(Av, dtype=int) - 1
    k_info_total = KU + KV  # 246

    print(f"|Au|={len(Au)}, |Av|={len(Av)}, info bits/cw = {k_info_total}", flush=True)

    model = load_gmac_model(N)
    if model is None:
        print("ERROR: no NCG model", flush=True)
        return

    rng_u = np.random.default_rng(SEED)
    rng_v = np.random.default_rng(SEED + 1)

    ncg_block_errs = 0
    sc_block_errs = 0
    ncg_bit_errs = 0
    sc_bit_errs = 0
    ncg_failed_counts = []
    sc_failed_counts = []

    t0 = time.perf_counter()
    total = 0
    with torch.no_grad():
        while total < N_CW:
            actual = min(BATCH_SIZE, N_CW - total)

            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            uf[:, info_u] = rng_u.integers(0, 2, size=(actual, len(info_u)))
            vf[:, info_v] = rng_v.integers(0, 2, size=(actual, len(info_v)))

            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf).astype(np.float32)

            # NCG decode (batched)
            zt = torch.from_numpy(zf).float()
            _, _, uh, vh, _ = model(zt, b, fu, fv)

            uhat_ncg = np.zeros((actual, N), dtype=int)
            vhat_ncg = np.zeros((actual, N), dtype=int)
            for p in Au:
                if p in uh:
                    uhat_ncg[:, p - 1] = uh[p].detach().cpu().numpy().astype(int)
            for p in Av:
                if p in vh:
                    vhat_ncg[:, p - 1] = vh[p].detach().cpu().numpy().astype(int)

            # Per-codeword SC + error accounting
            for i in range(actual):
                z_list = zf[i].tolist()

                # SC decode
                u_sc, v_sc = decode_single(N, z_list, b, fu, fv, channel, log_domain=True)
                u_sc_arr = np.asarray(u_sc, dtype=int)
                v_sc_arr = np.asarray(v_sc, dtype=int)

                # Bit errors on info positions only
                ncg_u_mismatch = int(np.sum(uhat_ncg[i, info_u] != uf[i, info_u]))
                ncg_v_mismatch = int(np.sum(vhat_ncg[i, info_v] != vf[i, info_v]))
                ncg_total = ncg_u_mismatch + ncg_v_mismatch

                sc_u_mismatch = int(np.sum(u_sc_arr[info_u] != uf[i, info_u]))
                sc_v_mismatch = int(np.sum(v_sc_arr[info_v] != vf[i, info_v]))
                sc_total = sc_u_mismatch + sc_v_mismatch

                ncg_bit_errs += ncg_total
                sc_bit_errs += sc_total

                if ncg_total > 0:
                    ncg_block_errs += 1
                    ncg_failed_counts.append(ncg_total)
                if sc_total > 0:
                    sc_block_errs += 1
                    sc_failed_counts.append(sc_total)

            total += actual
            if total % 500 == 0 or total == N_CW:
                el = time.perf_counter() - t0
                rate = total / el if el > 0 else 0.0
                eta = (N_CW - total) / rate if rate > 0 else 0.0
                print(
                    f"  {total}/{N_CW}  NCG_bler={ncg_block_errs/total:.3e}  "
                    f"SC_bler={sc_block_errs/total:.3e}  "
                    f"NCG_ber={ncg_bit_errs/(total*k_info_total):.3e}  "
                    f"SC_ber={sc_bit_errs/(total*k_info_total):.3e}  "
                    f"eta={eta:.0f}s",
                    flush=True,
                )

    total_bits = N_CW * k_info_total

    ncg_bler = ncg_block_errs / N_CW
    sc_bler = sc_block_errs / N_CW
    ncg_ber = ncg_bit_errs / total_bits
    sc_ber = sc_bit_errs / total_bits

    ncg_bits_per_failed = (ncg_bit_errs / ncg_block_errs) if ncg_block_errs > 0 else float('nan')
    sc_bits_per_failed = (sc_bit_errs / sc_block_errs) if sc_block_errs > 0 else float('nan')

    bler_ratio = (ncg_bler / sc_bler) if sc_bler > 0 else float('nan')
    ber_ratio = (ncg_ber / sc_ber) if sc_ber > 0 else float('nan')

    ncg_hist = bin_bit_errors(ncg_failed_counts)
    sc_hist = bin_bit_errors(sc_failed_counts)

    print("\n" + "=" * 72, flush=True)
    print(f"RESULTS  N={N}, ku=kv={KU}, SNR={SNR_DB} dB, {N_CW} codewords", flush=True)
    print("=" * 72, flush=True)
    print(f"                             NCG                     SC", flush=True)
    print(f"  BLER                 {ncg_bler:.4f} ({ncg_block_errs}/{N_CW})     {sc_bler:.4f} ({sc_block_errs}/{N_CW})", flush=True)
    print(f"  BER                  {ncg_ber:.3e} ({ncg_bit_errs}/{total_bits})   {sc_ber:.3e} ({sc_bit_errs}/{total_bits})", flush=True)
    print(f"  bits/failed block    {ncg_bits_per_failed:.2f}                   {sc_bits_per_failed:.2f}", flush=True)
    print(f"  BLER ratio NCG/SC    {bler_ratio:.3f}x", flush=True)
    print(f"  BER  ratio NCG/SC    {ber_ratio:.3f}x", flush=True)
    print("", flush=True)
    print("  Bit-errors per failed block (histogram):", flush=True)
    print(f"    bin     NCG      SC", flush=True)
    for k in ['1', '2-3', '4-7', '8-15', '16+']:
        print(f"    {k:>5}  {ncg_hist[k]:>5}   {sc_hist[k]:>5}", flush=True)
    print("", flush=True)

    total_time = time.perf_counter() - t0
    print(f"Elapsed: {total_time:.0f}s", flush=True)


if __name__ == '__main__':
    main()
