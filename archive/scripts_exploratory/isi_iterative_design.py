#!/usr/bin/env python3
"""
Iterative frozen set design for chained trellis SC on ISI-MAC.

The problem: rate-1 design fails because FB marginals are correlated and
SC f/g operations assume independence. Without frozen anchors, all positions
get Pe ~ 0.25-0.50.

The fix: design iteratively starting from GMAC proxy frozen set. With most
positions frozen, the correlation issue is suppressed. Genie-aided SC
measures per-position error rates, then we swap worst info positions with
best frozen positions. Repeat until stable.
"""
import sys, os, time, math
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode, polar_encode_batch
from polar.channels_memory import ISIMAC
from polar.decoder_trellis_mac_chained import (
    _log_W_stage1, _log_W_stage2, _forward_backward_2state, _marg_to_llr,
    decode_stage1_u, bler_chained
)
from polar.decoder import _SCNode, _f_llr, _g_llr
from polar.design_mc import design_from_file, _argsort_with_polar_tiebreak

SNR_DB = 6.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)
ISI_H = 0.3


def genie_sc_from_llr(leaf_llr, true_bits):
    """
    Genie-aided SC decode: at each leaf, check if decision matches truth,
    but always feed the TRUE bit. Returns per-position error array (0 or 1).
    """
    N = len(leaf_llr)
    true_bits = np.asarray(true_bits, dtype=np.int8)
    errors = np.zeros(N, dtype=np.int32)

    node = _SCNode(np.asarray(leaf_llr, dtype=np.float64))

    for i in range(N):
        L = node.get_llr(i)
        decision = 0 if L >= 0 else 1
        if decision != true_bits[i]:
            errors[i] = 1
        node.feed(i, true_bits[i])  # genie: feed true bit

    return errors


def genie_design_stage1(channel, N, Au, fu_1idx, n_trials, seed=42):
    """
    Genie-aided per-position error rates for Stage 1 (U decoder).

    Frozen positions use their known values. Info positions are checked
    for errors. ALL positions feed true bits (genie).

    Returns pe_all (N,) — error rate at every position (frozen positions
    will have Pe~0 since they're always correct via genie).
    """
    rng = np.random.default_rng(seed)
    Au_set = set(Au)  # 1-indexed
    Av_dummy = list(range(1, N + 1))  # V is random

    pe = np.zeros(N, dtype=np.float64)

    for trial in range(n_trials):
        # Random message
        u = np.zeros(N, dtype=np.int8)
        v = np.zeros(N, dtype=np.int8)
        for p in Au:
            u[p - 1] = rng.integers(0, 2)
        for p in Av_dummy:
            v[p - 1] = rng.integers(0, 2)

        x = np.array(polar_encode(list(u)), dtype=np.int8)
        y = np.array(polar_encode(list(v)), dtype=np.int8)
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]

        # FB → LLRs
        log_W1 = _log_W_stage1(z, channel)
        log_marg = _forward_backward_2state(log_W1)
        llr = _marg_to_llr(log_marg)

        # Genie-aided SC
        errors = genie_sc_from_llr(llr, u)
        pe += errors

    pe /= n_trials
    return pe


def genie_design_stage2(channel, N, Av, n_trials, seed=43):
    """Genie-aided per-position error rates for Stage 2 (V decoder, given true X)."""
    rng = np.random.default_rng(seed)
    Au_all = list(range(1, N + 1))

    pe = np.zeros(N, dtype=np.float64)

    for trial in range(n_trials):
        u = rng.integers(0, 2, N).astype(np.int8)
        v = np.zeros(N, dtype=np.int8)
        for p in Av:
            v[p - 1] = rng.integers(0, 2)

        x = np.array(polar_encode(list(u)), dtype=np.int8)
        y = np.array(polar_encode(list(v)), dtype=np.int8)
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]

        log_W2 = _log_W_stage2(z, x, channel)
        log_marg = _forward_backward_2state(log_W2)
        llr = _marg_to_llr(log_marg)

        errors = genie_sc_from_llr(llr, v)
        pe += errors

    pe /= n_trials
    return pe


def select_info(pe, k):
    """Select best k positions (lowest Pe). Returns 1-indexed sorted list."""
    sorted_idx = _argsort_with_polar_tiebreak(pe)
    info_0 = sorted(sorted_idx[:k].tolist())
    info_1 = [i + 1 for i in info_0]
    return info_1


def make_frozen_dicts(N, Au, Av):
    """Build frozen dicts (1-indexed) for SC decoder."""
    fu = {p: 0 for p in range(1, N + 1) if p not in Au}
    fv = {p: 0 for p in range(1, N + 1) if p not in Av}
    return fu, fv


def iterative_design(N, ku, kv, n_trials=5000, max_iters=10, seed=42):
    """
    Iterative frozen set design for chained trellis SC on ISI-MAC.

    Starts from GMAC proxy, runs genie-aided SC to measure per-position
    error rates, re-selects frozen set, repeats until convergence.
    """
    n = int(math.log2(N))
    channel = ISIMAC(sigma2=SIGMA2, h=ISI_H)

    # Load GMAC proxy as starting point
    gmac_path = os.path.join(_ROOT, f'designs/gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_gmac, Av_gmac, _, _, _, _, _ = design_from_file(gmac_path, n, ku, kv)

    Au = list(Au_gmac)
    Av = list(Av_gmac)

    print(f'\n{"="*60}')
    print(f'Iterative ISI design: N={N}, ku={ku}, kv={kv}')
    print(f'Starting from GMAC proxy: Au={Au}')
    print(f'{"="*60}')

    # Eval GMAC proxy baseline
    fu, fv = make_frozen_dicts(N, Au, Av)
    r0 = bler_chained(channel, N, fu, fv, Au, Av, n_cw=n_trials, seed=0)
    print(f'\nGMAC proxy SC BLER: {r0["chained_bler"]:.4f}')

    history = []

    for iteration in range(max_iters):
        t0 = time.time()
        fu_1idx = {p: 0 for p in range(1, N + 1) if p not in Au}

        # Genie-aided design for U
        pe_u = genie_design_stage1(channel, N, Au, fu_1idx, n_trials, seed=seed + iteration)

        # Genie-aided design for V
        pe_v = genie_design_stage2(channel, N, Av, n_trials, seed=seed + iteration + 100)

        # Select new info sets
        Au_new = select_info(pe_u, ku)
        Av_new = select_info(pe_v, kv)

        # Check convergence
        u_changed = set(Au_new) != set(Au)
        v_changed = set(Av_new) != set(Av)
        u_overlap = len(set(Au_new) & set(Au))
        v_overlap = len(set(Av_new) & set(Av))

        elapsed = time.time() - t0

        # Eval new design
        fu_new, fv_new = make_frozen_dicts(N, Au_new, Av_new)
        r = bler_chained(channel, N, fu_new, fv_new, Au_new, Av_new, n_cw=n_trials, seed=0)

        pe_info_u = [pe_u[p - 1] for p in Au_new]
        pe_info_v = [pe_v[p - 1] for p in Av_new]

        print(f'\nIter {iteration + 1}: ({elapsed:.1f}s)')
        print(f'  U: overlap={u_overlap}/{ku}, changed={u_changed}')
        print(f'     info Pe: min={min(pe_info_u):.4f} max={max(pe_info_u):.4f} mean={np.mean(pe_info_u):.4f}')
        print(f'  V: overlap={v_overlap}/{kv}, changed={v_changed}')
        print(f'     info Pe: min={min(pe_info_v):.4f} max={max(pe_info_v):.4f} mean={np.mean(pe_info_v):.4f}')
        print(f'  SC BLER: {r["chained_bler"]:.4f} (U={r["u_err_rate"]:.4f} V={r["v_err_rate"]:.4f})')
        print(f'  Au: {Au_new}')

        history.append({
            'Au': Au_new, 'Av': Av_new,
            'bler': r['chained_bler'],
            'u_overlap': u_overlap, 'v_overlap': v_overlap,
        })

        Au = Au_new
        Av = Av_new

        if not u_changed and not v_changed:
            print(f'\n  CONVERGED at iteration {iteration + 1}!')
            break

    # Final comparison
    print(f'\n{"="*60}')
    print(f'FINAL COMPARISON (N={N}):')
    fu_final, fv_final = make_frozen_dicts(N, Au, Av)
    r_final = bler_chained(channel, N, fu_final, fv_final, Au, Av, n_cw=max(5000, n_trials), seed=0)
    print(f'  ISI iterative: BLER={r_final["chained_bler"]:.4f} (U={r_final["u_err_rate"]:.4f} V={r_final["v_err_rate"]:.4f})')
    print(f'  GMAC proxy:    BLER={r0["chained_bler"]:.4f} (U={r0["u_err_rate"]:.4f} V={r0["v_err_rate"]:.4f})')
    ratio = r_final['chained_bler'] / max(r0['chained_bler'], 1e-9)
    print(f'  Ratio: {ratio:.3f}x')
    print(f'  Au overlap with GMAC: {len(set(Au) & set(Au_gmac))}/{ku}')
    print(f'{"="*60}')

    # Save
    out = {
        'N': N, 'ku': ku, 'kv': kv,
        'Au_final': Au, 'Av_final': Av,
        'Au_gmac': list(Au_gmac), 'Av_gmac': list(Av_gmac),
        'bler_isi': r_final['chained_bler'],
        'bler_gmac': r0['chained_bler'],
        'ratio': ratio,
        'iterations': len(history),
        'pe_u_final': pe_u.tolist(),
        'pe_v_final': pe_v.tolist(),
    }

    # Save design file compatible with design_from_file
    out_path = os.path.join(_ROOT, f'designs/isi_mac_iterative_C_n{n}_snr6dB_h0.3.npz')
    np.savez(out_path,
             u_error_rates=pe_u, v_error_rates=pe_v,
             path_i=N, n_trials=n_trials, sigma2=SIGMA2, snr_db=SNR_DB)
    print(f'  Saved design: {out_path}')

    return out


def main():
    ku_kv = {4: (4, 7), 5: (7, 15), 6: (15, 29), 7: (30, 58), 8: (59, 117)}

    for n in [4, 5, 6, 7, 8]:
        N = 1 << n
        ku, kv = ku_kv[n]
        n_trials = 10000 if N <= 64 else 5000
        iterative_design(N, ku, kv, n_trials=n_trials, max_iters=10)


if __name__ == '__main__':
    main()
