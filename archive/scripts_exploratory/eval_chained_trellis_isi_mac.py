"""
eval_chained_trellis_isi_mac.py
===============================
Evaluate the chained (two-stage) trellis SC decoder on ISI-MAC at SNR 6 dB
and compare to:
    (1) Joint trellis SC decoder (the 4-state MAC trellis baseline)
    (2) Memoryless SC decoder (ignores ISI, treats channel as AWGN GMAC)

This directly answers the question "does SCT also do the chained thing?" by
providing in-code numerics: the chained trellis SC output is the "ceiling"
that a chained neural NPD is expected to approach.

Channel: ISIMAC(h=0.3)
SNR:     6 dB
Path:    Class C (path_i = N)
N:       16, 32, 64 (optional: 128)
Rates:   GMAC Class C frozen sets, ku/kv matching existing NPD evaluation.
"""
from __future__ import annotations
import os
import sys
import time
import json

# Pin threads per project spec (PID 92903 using ~4 cores).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.channels_memory import ISIMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.encoder import polar_encode_batch
from polar.decoder_trellis import decode_single as decode_joint_trellis
from polar.decoder_trellis_mac_chained import decode_chained
from polar.decoder import (
    _sc_decode_from_llr, _u_marginal_llr, _v_conditional_llr,
)


SNR_DB = 6.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)
H_TAP = 0.3

# (ku, kv) per the existing NPD evaluation / GMAC Class C rates.
RATES = {
    16: (4, 7),
    32: (7, 15),
    64: (15, 29),
    128: (30, 58),
}


# -------------------------------------------------------------------------
#  Memoryless SC baseline: pretend there is no ISI — treat as GMAC
# -------------------------------------------------------------------------

def _memoryless_log_W_leaf(z, sigma2):
    """
    Build (N, 2, 2) leaf log-prob tensor treating the channel as memoryless
    GMAC Z = (1-2X) + (1-2Y) + W.  (Ignores the ISI term.)
    """
    z = np.asarray(z, dtype=np.float64)
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma2)
    N = z.shape[0]
    log_W = np.empty((N, 2, 2), dtype=np.float64)
    for x in range(2):
        for y in range(2):
            mu = (1.0 - 2.0 * x) + (1.0 - 2.0 * y)
            log_W[:, x, y] = log_norm - (z - mu) ** 2 / (2.0 * sigma2)
    return log_W


def decode_memoryless_sc(z, N, fu, fv):
    """
    Memoryless SC (Class C corner) decoder: ignores ISI, decodes U from the
    Y-marginal LLR, then V given the re-encoded U.
    """
    log_W = _memoryless_log_W_leaf(z, SIGMA2)
    u_hat = _sc_decode_from_llr(_u_marginal_llr(log_W), fu)
    from polar.encoder import polar_encode
    x_hat = np.array(polar_encode(u_hat.tolist()), dtype=np.int64)
    v_hat = _sc_decode_from_llr(_v_conditional_llr(log_W, x_hat), fv)
    return u_hat, v_hat


# -------------------------------------------------------------------------
#  BLER evaluation helpers
# -------------------------------------------------------------------------

def _sample_messages(N, Au, Av, rng):
    u = np.zeros(N, dtype=int)
    v = np.zeros(N, dtype=int)
    for p in Au:
        u[p - 1] = rng.integers(0, 2)
    for p in Av:
        v[p - 1] = rng.integers(0, 2)
    return u, v


def evaluate_all(channel, N, fu, fv, Au, Av, b_path, n_cw, seed=2026):
    """
    For each codeword, run all decoders and tally errors per decoder.
    A shared rng ensures the same (u,v,z) is used for every decoder, giving
    a coupled (same-noise) comparison.
    """
    rng = np.random.default_rng(seed)

    n_chain_joint = 0
    n_chain_u = 0
    n_chain_v = 0
    n_stage1_only = 0
    n_stage2_given_trueu = 0
    n_joint_trellis = 0
    n_memoryless = 0

    # For stage1-only and stage2-given-true-u we use the chained decoder's
    # internal helpers, but here we can just count partial errors from the
    # chained run:
    from polar.decoder_trellis_mac_chained import (
        decode_stage1_u, decode_stage2_v,
    )

    t_chain = 0.0
    t_joint = 0.0
    t_mem = 0.0

    for _ in range(n_cw):
        u, v = _sample_messages(N, Au, Av, rng)
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]

        # Chained trellis SC (reuse stage1 decode for stage1-only metric)
        t0 = time.time()
        u_hat = decode_stage1_u(z, N, fu, channel)
        v_hat = decode_stage2_v(z, u_hat, N, fv, channel)
        t_chain += time.time() - t0

        ue = any(int(u_hat[p - 1]) != int(u[p - 1]) for p in Au)
        ve = any(int(v_hat[p - 1]) != int(v[p - 1]) for p in Av)
        if ue:
            n_chain_u += 1
        if ve:
            n_chain_v += 1
        if ue or ve:
            n_chain_joint += 1
        if ue:
            n_stage1_only += 1

        # Stage 2 conditioned on the TRUE U codeword (no stage-1 error
        # propagation). Shows how reliable V|X is.
        v_hat_given_trueu = decode_stage2_v(z, u.astype(np.int8), N, fv, channel)
        if any(int(v_hat_given_trueu[p - 1]) != int(v[p - 1]) for p in Av):
            n_stage2_given_trueu += 1

        # Joint 4-state trellis SC (existing decoder)
        t0 = time.time()
        u_j, v_j = decode_joint_trellis(N, z, b_path, fu, fv, channel)
        t_joint += time.time() - t0
        uj_err = any(int(u_j[p - 1]) != int(u[p - 1]) for p in Au)
        vj_err = any(int(v_j[p - 1]) != int(v[p - 1]) for p in Av)
        if uj_err or vj_err:
            n_joint_trellis += 1

        # Memoryless SC baseline (ignores ISI)
        t0 = time.time()
        u_m, v_m = decode_memoryless_sc(z, N, fu, fv)
        t_mem += time.time() - t0
        um_err = any(int(u_m[p - 1]) != int(u[p - 1]) for p in Au)
        vm_err = any(int(v_m[p - 1]) != int(v[p - 1]) for p in Av)
        if um_err or vm_err:
            n_memoryless += 1

    return {
        "n_cw": n_cw,
        "chained_bler": n_chain_joint / n_cw,
        "chained_u_err": n_chain_u / n_cw,
        "chained_v_err": n_chain_v / n_cw,
        "stage1_u_only_bler": n_stage1_only / n_cw,
        "stage2_v_given_true_u_bler": n_stage2_given_trueu / n_cw,
        "joint_trellis_bler": n_joint_trellis / n_cw,
        "memoryless_sc_bler": n_memoryless / n_cw,
        "time_chained_s": t_chain,
        "time_joint_trellis_s": t_joint,
        "time_memoryless_s": t_mem,
    }


# -------------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------------

def _load_design(N, ku, kv):
    """Load frozen sets from the pre-computed GMAC Class C design."""
    n = int(np.log2(N))
    path = os.path.join(
        _ROOT, "designs", f"gmac_C_n{n}_snr{int(SNR_DB)}dB.npz")
    Au, Av, fu, fv, _, _, _ = design_from_file(path, n, ku, kv)
    return Au, Av, fu, fv


def main(n_cw_by_N=None, save_path=None):
    if n_cw_by_N is None:
        # Default: 3000 for small N, 2000 for N=64, 1000 for N=128.
        n_cw_by_N = {16: 3000, 32: 3000, 64: 2000, 128: 1000}

    if save_path is None:
        save_path = os.path.join(
            _ROOT, "class_c_npd", "results",
            "chained_trellis_sc_isi_mac.json")

    channel = ISIMAC(sigma2=SIGMA2, h=H_TAP)
    print(f"Channel: ISI-MAC h={H_TAP}  SNR={SNR_DB} dB  sigma^2={SIGMA2:.4f}")

    all_results = {
        "channel": "ISI-MAC",
        "h": H_TAP,
        "snr_db": SNR_DB,
        "sigma2": SIGMA2,
        "path": "Class C (path_i = N)",
        "by_N": {},
    }

    Ns = sorted(k for k in n_cw_by_N if k in RATES and n_cw_by_N[k] > 0)
    for N in Ns:
        n = int(np.log2(N))
        ku, kv = RATES[N]
        b_path = make_path(N, N)
        Au, Av, fu, fv = _load_design(N, ku, kv)
        n_cw = n_cw_by_N[N]

        print(f"\n=== N={N}  ku={ku}  kv={kv}  n_cw={n_cw} ===")
        t0 = time.time()
        res = evaluate_all(channel, N, fu, fv, Au, Av, b_path, n_cw)
        wall = time.time() - t0
        res["ku"] = ku
        res["kv"] = kv
        res["N"] = N
        res["wall_s"] = wall

        print(f"  Chained trellis BLER:       {res['chained_bler']:.4f}")
        print(f"    Stage-1 (U-only):         {res['stage1_u_only_bler']:.4f}")
        print(f"    Stage-2 (V|trueU):        {res['stage2_v_given_true_u_bler']:.4f}")
        print(f"  Joint 4-state trellis BLER: {res['joint_trellis_bler']:.4f}")
        print(f"  Memoryless SC baseline:     {res['memoryless_sc_bler']:.4f}")
        print(f"  wall: {wall:.1f}s "
              f"(chained={res['time_chained_s']:.1f}s, "
              f"joint={res['time_joint_trellis_s']:.1f}s, "
              f"memoryless={res['time_memoryless_s']:.1f}s)")

        all_results["by_N"][str(N)] = res
        # Incremental save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {save_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n16", type=int, default=3000)
    p.add_argument("--n32", type=int, default=3000)
    p.add_argument("--n64", type=int, default=2000)
    p.add_argument("--n128", type=int, default=0)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    n_cw = {16: args.n16, 32: args.n32, 64: args.n64, 128: args.n128}
    main(n_cw_by_N=n_cw, save_path=args.save)
