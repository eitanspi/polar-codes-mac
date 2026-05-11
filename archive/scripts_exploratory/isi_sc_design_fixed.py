#!/usr/bin/env python3
"""
Fixed genie-aided SC design for chained trellis SC on ISI-MAC.

Bug in previous version: generated random bits only at INFO positions (frozen=0).
This biased the Pe measurement. Fix: use rate-1 (ALL positions random).

The standard Tal-Vardy MC design:
1. Generate random u at ALL N positions (rate-1)
2. Encode x = polar_encode(u)
3. Channel z = ISI(x, y)
4. Genie-aided SC: check per-position decision, feed TRUE bits
5. Rank by Pe → best k = info
"""
import sys, os, time, math
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode, polar_encode_batch
from polar.channels_memory import ISIMAC
from polar.decoder_trellis_mac_chained import (
    _log_W_stage1, _log_W_stage2, _forward_backward_2state, _marg_to_llr,
    bler_chained
)
from polar.decoder import _SCNode, _f_llr, _g_llr
from polar.design_mc import design_from_file, _argsort_with_polar_tiebreak

SNR_DB = 6.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)
ISI_H = 0.3


def genie_sc_from_llr(leaf_llr, true_bits):
    """Genie-aided SC: check decision, feed true bit. Returns per-position errors."""
    N = len(leaf_llr)
    true_bits = np.asarray(true_bits, dtype=np.int8)
    errors = np.zeros(N, dtype=np.int32)
    node = _SCNode(np.asarray(leaf_llr, dtype=np.float64))
    for i in range(N):
        L = node.get_llr(i)
        decision = 0 if L >= 0 else 1
        if decision != true_bits[i]:
            errors[i] = 1
        node.feed(i, true_bits[i])
    return errors


def rate1_genie_design(channel, N, n_trials, seed=42):
    """
    RATE-1 genie-aided design: ALL positions have random bits.
    This is the standard Tal-Vardy approach.
    """
    rng = np.random.default_rng(seed)
    pe_u = np.zeros(N, dtype=np.float64)
    pe_v = np.zeros(N, dtype=np.float64)

    t0 = time.time()
    for trial in range(n_trials):
        # ALL positions random (rate-1)
        u = rng.integers(0, 2, N).astype(np.int8)
        v = rng.integers(0, 2, N).astype(np.int8)
        x = np.array(polar_encode(list(u)), dtype=np.int8)
        y = np.array(polar_encode(list(v)), dtype=np.int8)
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]

        # Stage 1: U (marginalise V)
        log_W1 = _log_W_stage1(z, channel)
        log_marg1 = _forward_backward_2state(log_W1)
        llr_u = _marg_to_llr(log_marg1)
        pe_u += genie_sc_from_llr(llr_u, u)

        # Stage 2: V given true X
        log_W2 = _log_W_stage2(z, x, channel)
        log_marg2 = _forward_backward_2state(log_W2)
        llr_v = _marg_to_llr(log_marg2)
        pe_v += genie_sc_from_llr(llr_v, v)

        if (trial + 1) % max(1, n_trials // 5) == 0:
            elapsed = time.time() - t0
            print(f'  {trial+1}/{n_trials} ({elapsed:.1f}s)', flush=True)

    pe_u /= n_trials
    pe_v /= n_trials
    return pe_u, pe_v


def main():
    channel = ISIMAC(sigma2=SIGMA2, h=ISI_H)
    ku_kv = {4: (4, 7), 5: (7, 15), 6: (15, 29), 7: (30, 58), 8: (59, 117)}

    for n in [4, 5, 6, 7, 8]:
        N = 1 << n
        ku, kv = ku_kv[n]
        n_trials = 20000 if N <= 64 else 10000

        print(f'\n{"="*60}')
        print(f'Rate-1 genie design: N={N}, ku={ku}, kv={kv}, trials={n_trials}')
        print(f'{"="*60}')

        pe_u, pe_v = rate1_genie_design(channel, N, n_trials)

        # Select info sets
        sorted_u = _argsort_with_polar_tiebreak(pe_u)
        sorted_v = _argsort_with_polar_tiebreak(pe_v)
        Au = sorted([int(i + 1) for i in sorted_u[:ku]])
        Av = sorted([int(i + 1) for i in sorted_v[:kv]])

        # Compare with GMAC proxy
        gmac_path = os.path.join(_ROOT, f'designs/gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
        Au_gmac, Av_gmac, _, _, _, _, _ = design_from_file(gmac_path, n, ku, kv)

        overlap_u = len(set(Au) & set(Au_gmac))
        overlap_v = len(set(Av) & set(Av_gmac))

        print(f'\n  Pe_u: min={pe_u.min():.4f} max={pe_u.max():.4f}')
        print(f'  Pe_v: min={pe_v.min():.4f} max={pe_v.max():.4f}')
        print(f'  Au_isi: {Au}')
        print(f'  Au_gmac: {list(Au_gmac)}')
        print(f'  U overlap: {overlap_u}/{ku}')

        # Eval SC with ISI design
        fu_isi = {p: 0 for p in range(1, N + 1) if p not in Au}
        fv_isi = {p: 0 for p in range(1, N + 1) if p not in Av}
        n_cw = 10000 if N <= 64 else 5000
        r_isi = bler_chained(channel, N, fu_isi, fv_isi, Au, Av, n_cw, seed=0)

        # Eval SC with GMAC design
        fu_gmac = {p: 0 for p in range(1, N + 1) if p not in Au_gmac}
        fv_gmac = {p: 0 for p in range(1, N + 1) if p not in Av_gmac}
        r_gmac = bler_chained(channel, N, fu_gmac, fv_gmac, Au_gmac, Av_gmac, n_cw, seed=0)

        ratio = r_isi['chained_bler'] / max(r_gmac['chained_bler'], 1e-9)
        print(f'\n  SC + ISI_design: BLER={r_isi["chained_bler"]:.4f}')
        print(f'  SC + GMAC_proxy: BLER={r_gmac["chained_bler"]:.4f}')
        print(f'  Ratio: {ratio:.3f}x')

        if ratio > 1.0:
            print(f'  WARNING: ISI design worse than GMAC proxy!')
            # Debug: show positions that differ
            isi_only = set(Au) - set(Au_gmac)
            gmac_only = set(Au_gmac) - set(Au)
            for p in sorted(isi_only):
                print(f'    ISI-only pos {p}: Pe_u={pe_u[p-1]:.4f}')
            for p in sorted(gmac_only):
                print(f'    GMAC-only pos {p}: Pe_u={pe_u[p-1]:.4f}')

        # Save
        out_path = os.path.join(_ROOT, f'designs/isi_mac_rate1_C_n{n}_snr6dB_h0.3.npz')
        np.savez(out_path, u_error_rates=pe_u, v_error_rates=pe_v,
                 path_i=N, n_trials=n_trials, sigma2=SIGMA2, snr_db=SNR_DB)
        print(f'  Saved: {out_path}')


if __name__ == '__main__':
    main()
