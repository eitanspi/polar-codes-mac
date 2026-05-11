#!/usr/bin/env python3
"""
run_all_baselines.py
====================
Produce all analytical baselines for paper-style comparison tables.

Tasks:
  1. ISI-MAC trellis SC (chained) at N=16..1024 with 10K CW
  2. ISI-MAC memoryless SC at N=16..1024 with 10K CW
  3. Ising MAC trellis SC at N=16..64 with 5K CW
  4. Ising MAC memoryless SC at N=16..64 with 5K CW
  5. MA-AGN memoryless SC at N=16..128 with 5K CW

Saves results incrementally to results/paper_style/.
"""
import sys, os, time, json, math
import numpy as np

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# Torch thread limit
try:
    import torch
    torch.set_num_threads(4)
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.encoder import polar_encode_batch
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import build_log_W_leaf, _sc_decode_from_llr
from polar.channels_memory import ISIMAC
from polar.channels_memory_new import IsingMAC, MAAGNMAC
from polar.decoder_trellis_mac_chained import decode_chained as isi_decode_chained
from polar.decoder_trellis_ising_chained import decode_chained as ising_decode_chained

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'paper_style')
os.makedirs(RESULTS_DIR, exist_ok=True)

# GMAC Class C rates (from existing project)
RATES = {
    16:  {'ku': 4,   'kv': 7},
    32:  {'ku': 7,   'kv': 15},
    64:  {'ku': 15,  'kv': 29},
    128: {'ku': 30,  'kv': 58},
    256: {'ku': 59,  'kv': 117},
    512: {'ku': 119, 'kv': 233},
    1024: {'ku': 238, 'kv': 467},
}


def wilson_ci(n_err, n_total, z=1.96):
    """Wilson score 95% CI for binomial proportion."""
    if n_total == 0:
        return (0.0, 1.0)
    p_hat = n_err / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def load_design(N):
    """Load GMAC Class C design for given N."""
    n = int(math.log2(N))
    rates = RATES[N]
    ku, kv = rates['ku'], rates['kv']
    design_path = os.path.join(DESIGNS_DIR, f'gmac_C_n{n}_snr6dB.npz')
    if not os.path.exists(design_path):
        raise FileNotFoundError(f'No design for N={N}: {design_path}')
    Au, Av, fu, fv, _, _, _ = design_from_file(design_path, n, ku, kv)
    return Au, Av, fu, fv, ku, kv


def build_gmac_llrs(z_vec, sigma2):
    """
    Build memoryless GMAC LLRs for a single observation vector.
    GMAC: Z = (1-2X) + (1-2Y) + W, W ~ N(0, sigma2).
    Leaf LLR[t] = log P(z|x=0)/P(z|x=1) marginalizing y as uniform.

    For user U (X), marginalizing Y:
        P(z|X=x) = 0.5 * N(z; (1-2x)+1, sigma2) + 0.5 * N(z; (1-2x)-1, sigma2)
    LLR_U[t] = log P(z|X=0) / P(z|X=1)

    For Class C path (all U first, then all V), we use the standard GMAC SC approach.
    Actually for memoryless SC on a Class C path, we use the standard decoder with
    GMAC leaf probs.
    """
    from polar.channels import GaussianMAC
    ch_gmac = GaussianMAC(sigma2=sigma2)
    log_W = build_log_W_leaf(list(z_vec), ch_gmac)  # (N, 2, 2)
    return log_W, ch_gmac


def eval_memoryless_sc(z_vec, fu, fv, Au, Av, sigma2):
    """
    Decode using memoryless GMAC SC decoder (ignoring memory).
    Returns (u_err, v_err) booleans.
    """
    from polar.channels import GaussianMAC
    from polar.decoder import decode_single

    N = len(z_vec)
    b = make_path(N, N)  # Class C
    ch_gmac = GaussianMAC(sigma2=sigma2)

    u_hat, v_hat = decode_single(N, list(z_vec), b, fu, fv, ch_gmac, log_domain=True)

    u_err = any(int(u_hat[p-1]) != 0 for p in fu) is False and \
            any(int(u_hat[p-1]) for p in Au if int(u_hat[p-1]) != int(u_hat[p-1]))
    # Actually this is wrong - we need to compare against the TRUE u/v.
    # This function should take the true u, v as inputs.
    raise NotImplementedError("Use eval_memoryless_sc_full instead")


def run_bler_eval(channel, N, n_cw, decode_fn, seed=42):
    """
    Generic BLER evaluation.

    decode_fn(z_vec, fu, fv) -> (u_hat, v_hat)

    Returns dict with BLER stats and Wilson CI.
    """
    Au, Av, fu, fv, ku, kv = load_design(N)
    rng = np.random.default_rng(seed)

    errs_u = 0
    errs_v = 0
    errs_total = 0

    t0 = time.time()
    for cw_idx in range(n_cw):
        # Generate random info bits
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au:
            u[p - 1] = rng.integers(0, 2)
        for p in Av:
            v[p - 1] = rng.integers(0, 2)

        # Encode
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]

        # Transmit through channel
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))
        if z.ndim == 2:
            z = z[0]
        z_vec = np.asarray(z, dtype=np.float64)

        # Decode
        u_hat, v_hat = decode_fn(z_vec, fu, fv)

        # Check errors
        ue = any(int(u_hat[p - 1]) != int(u[p - 1]) for p in Au)
        ve = any(int(v_hat[p - 1]) != int(v[p - 1]) for p in Av)
        if ue: errs_u += 1
        if ve: errs_v += 1
        if ue or ve: errs_total += 1

        # Progress
        if (cw_idx + 1) % max(1, n_cw // 10) == 0:
            elapsed = time.time() - t0
            print(f'    {cw_idx+1}/{n_cw} errs={errs_total} '
                  f'({elapsed:.0f}s)', flush=True)

    elapsed = time.time() - t0
    ci = wilson_ci(errs_total, n_cw)

    return {
        'N': N, 'ku': ku, 'kv': kv, 'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw,
        'bler_v': errs_v / n_cw,
        'bler_total': errs_total / n_cw,
        'wilson_95_ci': list(ci),
        'time_s': elapsed,
        'time_min': elapsed / 60.0,
    }


def save_results(results, filename):
    """Save results dict to JSON."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  Saved: {path}')


# ============================================================================
#  Task 1: ISI-MAC trellis SC baselines
# ============================================================================

def run_isi_mac_trellis_sc():
    """ISI-MAC chained trellis SC at all N values."""
    print('=' * 60)
    print('TASK 1: ISI-MAC Trellis SC Baselines')
    print('=' * 60)

    ch = ISIMAC(sigma2=10**(-0.6), h=0.3)
    print(f'Channel: ISI-MAC, SNR={ch.snr_db:.1f}dB, h={ch.h}')

    # Load existing results
    existing_file = os.path.join(os.path.dirname(__file__), '..',
                                  'results', 'reliable_evals', 'isi_mac_sc_10kcw.json')
    existing = {}
    if os.path.exists(existing_file):
        existing = json.load(open(existing_file))
        print(f'  Loaded existing results: {sorted(existing.keys())}')

    results = {}

    for N in [16, 32, 64, 128, 256, 512, 1024]:
        key = f'N{N}'

        # Check if we already have 10K CW results
        if key in existing and existing[key].get('n_cw', 0) >= 10000:
            r = existing[key]
            # Add Wilson CI if missing
            if 'wilson_95_ci' not in r:
                r['wilson_95_ci'] = list(wilson_ci(r['errs_total'], r['n_cw']))
            results[key] = r
            print(f'  N={N}: REUSING existing (bler={r["bler_total"]:.4f}, '
                  f'CI={r["wilson_95_ci"]})')
            continue

        print(f'\n  N={N}: Running trellis SC ({10000} CW)...')

        def decode_fn(z_vec, fu, fv, ch=ch, N=N):
            return isi_decode_chained(z_vec, N, fu, fv, ch)

        r = run_bler_eval(ch, N, 10000, decode_fn, seed=1001)
        results[key] = r
        print(f'  N={N}: bler={r["bler_total"]:.4f} '
              f'CI=[{r["wilson_95_ci"][0]:.4f}, {r["wilson_95_ci"][1]:.4f}] '
              f'({r["time_min"]:.1f} min)')

        # Save incrementally
        save_results(results, 'isi_mac_sc_baselines.json')

    save_results(results, 'isi_mac_sc_baselines.json')
    return results


# ============================================================================
#  Task 1b: ISI-MAC memoryless SC baselines
# ============================================================================

def run_isi_mac_memoryless_sc():
    """ISI-MAC decoded with memoryless GMAC SC (ignoring memory)."""
    print('\n' + '=' * 60)
    print('TASK 1b: ISI-MAC Memoryless SC Baselines')
    print('=' * 60)

    ch_isi = ISIMAC(sigma2=10**(-0.6), h=0.3)
    sigma2 = 10**(-0.6)  # same sigma2 for GMAC decoder

    from polar.channels import GaussianMAC
    from polar.decoder import decode_single

    ch_gmac = GaussianMAC(sigma2=sigma2)

    results = {}

    for N in [16, 32, 64, 128, 256, 512, 1024]:
        key = f'N{N}'
        b = make_path(N, N)

        print(f'\n  N={N}: Running memoryless SC ({10000} CW)...')

        def decode_fn(z_vec, fu, fv, ch_g=ch_gmac, b=b, N=N):
            u_hat, v_hat = decode_single(N, list(z_vec), b, fu, fv, ch_g, log_domain=True)
            return u_hat, v_hat

        r = run_bler_eval(ch_isi, N, 10000, decode_fn, seed=1001)
        results[key] = r
        print(f'  N={N}: bler={r["bler_total"]:.4f} '
              f'CI=[{r["wilson_95_ci"][0]:.4f}, {r["wilson_95_ci"][1]:.4f}] '
              f'({r["time_min"]:.1f} min)')

        save_results(results, 'isi_mac_memoryless_sc_baselines.json')

    save_results(results, 'isi_mac_memoryless_sc_baselines.json')
    return results


# ============================================================================
#  Task 2: Ising MAC baselines
# ============================================================================

def run_ising_mac_baselines():
    """Ising MAC baselines: trellis SC + memoryless SC."""
    print('\n' + '=' * 60)
    print('TASK 2: Ising MAC Baselines')
    print('=' * 60)

    ch = IsingMAC(sigma2=0.251, p_flip=0.1)
    sigma2_gmac = 0.251  # for memoryless decoder

    from polar.channels import GaussianMAC
    from polar.decoder import decode_single
    ch_gmac = GaussianMAC(sigma2=sigma2_gmac)

    results = {'trellis_sc': {}, 'memoryless_sc': {}}

    for N in [16, 32, 64]:
        b = make_path(N, N)
        n_cw = 5000

        # Trellis SC
        print(f'\n  N={N}: Ising trellis SC ({n_cw} CW)...')

        def decode_trellis(z_vec, fu, fv, ch=ch, N=N):
            return ising_decode_chained(z_vec, N, fu, fv, ch)

        r = run_bler_eval(ch, N, n_cw, decode_trellis, seed=2001)
        results['trellis_sc'][f'N{N}'] = r
        print(f'  N={N} trellis: bler={r["bler_total"]:.4f} '
              f'CI=[{r["wilson_95_ci"][0]:.4f}, {r["wilson_95_ci"][1]:.4f}]')

        # Memoryless SC
        print(f'  N={N}: Ising memoryless SC ({n_cw} CW)...')

        def decode_memless(z_vec, fu, fv, ch_g=ch_gmac, b=b, N=N):
            return decode_single(N, list(z_vec), b, fu, fv, ch_g, log_domain=True)

        r = run_bler_eval(ch, N, n_cw, decode_memless, seed=2001)
        results['memoryless_sc'][f'N{N}'] = r
        print(f'  N={N} memoryless: bler={r["bler_total"]:.4f} '
              f'CI=[{r["wilson_95_ci"][0]:.4f}, {r["wilson_95_ci"][1]:.4f}]')

        save_results(results, 'ising_mac_baselines.json')

    save_results(results, 'ising_mac_baselines.json')
    return results


# ============================================================================
#  Task 3: MA-AGN baselines
# ============================================================================

def run_maagn_mac_baselines():
    """MA-AGN MAC decoded with memoryless GMAC SC."""
    print('\n' + '=' * 60)
    print('TASK 3: MA-AGN MAC Baselines')
    print('=' * 60)

    ch = MAAGNMAC(sigma2=10**(-0.6), alpha=0.3)
    sigma2_gmac = 10**(-0.6)

    from polar.channels import GaussianMAC
    from polar.decoder import decode_single
    ch_gmac = GaussianMAC(sigma2=sigma2_gmac)

    results = {}

    for N in [16, 32, 64, 128]:
        b = make_path(N, N)
        n_cw = 5000

        print(f'\n  N={N}: MA-AGN memoryless SC ({n_cw} CW)...')

        def decode_fn(z_vec, fu, fv, ch_g=ch_gmac, b=b, N=N):
            return decode_single(N, list(z_vec), b, fu, fv, ch_g, log_domain=True)

        r = run_bler_eval(ch, N, n_cw, decode_fn, seed=3001)
        results[f'N{N}'] = r
        print(f'  N={N}: bler={r["bler_total"]:.4f} '
              f'CI=[{r["wilson_95_ci"][0]:.4f}, {r["wilson_95_ci"][1]:.4f}]')

        save_results(results, 'maagn_mac_baselines.json')

    save_results(results, 'maagn_mac_baselines.json')
    return results


# ============================================================================
#  Main
# ============================================================================

if __name__ == '__main__':
    print(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Threads: 4')

    # Task 1: ISI-MAC trellis SC
    isi_trellis = run_isi_mac_trellis_sc()

    # Task 1b: ISI-MAC memoryless SC
    isi_memless = run_isi_mac_memoryless_sc()

    # Task 2: Ising MAC
    ising = run_ising_mac_baselines()

    # Task 3: MA-AGN
    maagn = run_maagn_mac_baselines()

    print(f'\n{"=" * 60}')
    print(f'ALL BASELINES COMPLETE')
    print(f'End time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Results in: {RESULTS_DIR}')
