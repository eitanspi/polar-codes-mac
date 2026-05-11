"""
ISI-MAC proper MC design using chained trellis SC decoder.

For each N, runs genie-aided chained SC on the actual ISI-MAC channel
to get per-position error rates. Saves in the same format as gmac_C designs
so design_from_file() can load them directly.
"""
import os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode
from polar.channels_memory import ISIMAC
from polar.decoder_trellis_mac_chained import (
    _log_W_stage1, _log_W_stage2, _forward_backward_2state, _marg_to_llr
)
from polar.decoder import _sc_decode_from_llr


def genie_chained_sc(z, u_true, v_true, channel):
    """
    Genie-aided chained SC on ISI-MAC. Returns per-position errors for U and V.

    Stage 1: decode U with trellis FB LLRs, genie-aided SC (feed true bits).
    Stage 2: decode V given TRUE X, with trellis FB LLRs, genie-aided SC.
    """
    N = len(z)
    z = np.asarray(z, dtype=np.float64)

    # --- Stage 1: U ---
    log_W1 = _log_W_stage1(z, channel)
    log_marg1 = _forward_backward_2state(log_W1)
    llr_u = _marg_to_llr(log_marg1)

    # Genie-aided SC: at each leaf, check if LLR-based decision matches truth,
    # but always feed the TRUE bit for subsequent operations.
    u_errors = _genie_sc_from_llr(llr_u, u_true)

    # --- Stage 2: V given true X ---
    x_true = np.array(polar_encode(list(u_true)), dtype=np.int64)
    log_W2 = _log_W_stage2(z, x_true, channel)
    log_marg2 = _forward_backward_2state(log_W2)
    llr_v = _marg_to_llr(log_marg2)

    v_errors = _genie_sc_from_llr(llr_v, v_true)

    return u_errors, v_errors


def _genie_sc_from_llr(llr, true_bits):
    """
    Genie-aided SC decoder operating on LLRs.
    Returns per-position error indicator (0 or 1).

    Standard Arikan SC tree traversal, but at each leaf we:
    1. Check if the LLR-based decision matches the true bit
    2. Always feed the TRUE bit (genie) for subsequent f/g operations
    """
    N = len(llr)
    true_bits = np.asarray(true_bits, dtype=np.int64)
    errors = np.zeros(N, dtype=np.int32)

    # Recursive SC with genie
    def _recurse(llr_vec, depth, offset):
        n = len(llr_vec)
        if n == 1:
            pos = offset  # 0-indexed position in Arikan order
            decision = 1 if llr_vec[0] < 0 else 0
            if decision != true_bits[pos]:
                errors[pos] = 1
            return np.array([true_bits[pos]])  # genie: return true bit

        half = n // 2
        # f-operation (CalcLeft): min-sum approximation
        l_top = llr_vec[:half]
        l_bot = llr_vec[half:]
        f_llr = np.sign(l_top) * np.sign(l_bot) * np.minimum(np.abs(l_top), np.abs(l_bot))

        # Decode left half (genie)
        cw_left = _recurse(f_llr, depth + 1, offset)

        # g-operation (CalcRight)
        g_llr = l_bot + (1 - 2 * cw_left) * l_top

        # Decode right half (genie)
        cw_right = _recurse(g_llr, depth + 1, offset + half)

        # Combine
        cw = np.zeros(n, dtype=np.int64)
        cw[:half] = cw_left ^ cw_right
        cw[half:] = cw_right
        return cw

    _recurse(llr, 0, 0)
    return errors


def design_isi_mc(n, sigma2, h, n_trials, seed=42):
    """Run MC design for ISI-MAC at block length N=2^n."""
    N = 1 << n
    channel = ISIMAC(sigma2=sigma2, h=h)
    rng = np.random.default_rng(seed)

    u_err_counts = np.zeros(N, dtype=np.float64)
    v_err_counts = np.zeros(N, dtype=np.float64)

    t0 = time.time()
    for trial in range(n_trials):
        u = rng.integers(0, 2, size=N).tolist()
        v = rng.integers(0, 2, size=N).tolist()

        x = polar_encode(u)
        y = polar_encode(v)

        z = channel.sample_batch(
            np.array(x).reshape(1, -1),
            np.array(y).reshape(1, -1)
        )[0]

        u_err, v_err = genie_chained_sc(z, u, v, channel)
        u_err_counts += u_err
        v_err_counts += v_err

        if (trial + 1) % max(1, n_trials // 10) == 0:
            elapsed = time.time() - t0
            print(f'  N={N}: {trial+1}/{n_trials} trials ({elapsed:.1f}s)')

    pe_u = u_err_counts / n_trials
    pe_v = v_err_counts / n_trials

    elapsed = time.time() - t0
    print(f'  N={N} done: {elapsed:.1f}s')
    print(f'  U: {(pe_u < 0.01).sum()} positions with Pe<0.01')
    print(f'  V: {(pe_v < 0.01).sum()} positions with Pe<0.01')

    return pe_u, pe_v


def main():
    sigma2 = 10 ** (-6.0 / 10)  # SNR = 6 dB
    h = 0.3
    snr_db = 6.0

    out_dir = os.path.join(_ROOT, 'designs')
    os.makedirs(out_dir, exist_ok=True)

    configs = [
        (4,  50000),  # N=16
        (5,  50000),  # N=32
        (6,  50000),  # N=64
        (7,  50000),  # N=128
        (8,  20000),  # N=256
        (9,  10000),  # N=512
    ]

    for n, n_trials in configs:
        N = 1 << n
        out_path = os.path.join(out_dir, f'isi_mac_C_n{n}_snr6dB_h0.3.npz')

        print(f'\n{"="*50}')
        print(f'N={N} (n={n}), {n_trials} trials')
        print(f'{"="*50}')

        pe_u, pe_v = design_isi_mc(n, sigma2, h, n_trials, seed=42)

        # Save in the format design_from_file expects
        np.savez(
            out_path,
            u_error_rates=pe_u,
            v_error_rates=pe_v,
            path_i=N,
            n_trials=n_trials,
            seed=42,
            sigma2=sigma2,
            snr_db=snr_db,
        )
        print(f'  Saved: {out_path}')

        # Compare with GMAC proxy
        gmac_path = os.path.join(out_dir, f'gmac_C_n{n}_snr6dB.npz')
        if os.path.exists(gmac_path):
            from polar.design_mc import _argsort_with_polar_tiebreak
            gmac = np.load(gmac_path)
            gmac_pe = gmac['u_error_rates']

            # Get ku from GMAC design
            ku = int((gmac_pe < 0.5).sum())  # rough
            # Better: count positions with Pe significantly below 0.5
            gmac_sorted = _argsort_with_polar_tiebreak(gmac_pe)
            isi_sorted = _argsort_with_polar_tiebreak(pe_u)

            # Compare top-ku positions
            for ku_test in [N // 4, N // 3]:
                gmac_info = set(gmac_sorted[:ku_test].tolist())
                isi_info = set(isi_sorted[:ku_test].tolist())
                overlap = len(gmac_info & isi_info)
                print(f'  ku={ku_test}: overlap={overlap}/{ku_test} '
                      f'({100*overlap/ku_test:.0f}%)')


if __name__ == '__main__':
    main()
