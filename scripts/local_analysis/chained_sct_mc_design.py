"""
chained_sct_mc_design.py
========================
MC genie-aided design for the chained trellis SC decoder on ISI-MAC.

For each of n_trials random (u, v) drawings:
  * Sample z = ISIMAC(x, y)
  * Stage-1: FB on 2-state X-trellis (Y marginalized as uniform), → length-N LLR.
             Run genie SC (feed back true u at every step) and record which
             positions SC would have decoded wrongly.
  * Stage-2: With TRUE x (genie), FB on 2-state Y-trellis with X known, →
             length-N LLR. Genie SC on v.

Aggregate per-position error rates → sort → top-k make the info set.

Eval phase then runs decode_chained with the new (Au, Av) and counts BLER.

Output: JSON with design info-sets, design wall time, and eval BLER per N.
Runs N=128 and N=256 by default.

Run:
  python scripts/local_analysis/chained_sct_mc_design.py
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
from multiprocessing import Pool

# Pin numpy threads — we parallelize at the trial level instead
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode
from polar.decoder import _SCNode
from polar.decoder_trellis_mac_chained import (
    _log_W_stage1, _log_W_stage2, _forward_backward_2state, decode_chained,
)
from polar.design_mc import _argsort_with_polar_tiebreak


SNR_DB = 6.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)
H_TAP = 0.3

RATES = {
    16:  (4, 7),
    32:  (7, 15),
    64:  (15, 29),
    128: (30, 58),
    256: (59, 117),
    512: (119, 233),
    1024:(239, 467),
}


def _genie_sc_per_position(leaf_llr: np.ndarray, true_bits: np.ndarray) -> np.ndarray:
    """Run Arikan SC with genie feedback. Return per-position error indicator."""
    N = len(leaf_llr)
    node = _SCNode(np.asarray(leaf_llr, dtype=np.float64))
    err = np.zeros(N, dtype=np.int32)
    for i in range(N):
        L = node.get_llr(i)
        decided = 0 if L >= 0 else 1
        true_val = int(true_bits[i])
        if decided != true_val:
            err[i] = 1
        node.feed(i, true_val)
    return err


def _mc_trial_chunk(args):
    """Run a chunk of MC trials. Returns (u_err_counts, v_err_counts, n_done)."""
    N, sigma2, h, seed, n_trials = args
    ch = ISIMAC(sigma2=sigma2, h=h)
    rng = np.random.default_rng(seed)

    u_err = np.zeros(N, dtype=np.int64)
    v_err = np.zeros(N, dtype=np.int64)

    for _ in range(n_trials):
        u = rng.integers(0, 2, size=N)
        v = rng.integers(0, 2, size=N)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = ch.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]

        # Stage 1
        log_W1 = _log_W_stage1(z, ch)
        log_marg1 = _forward_backward_2state(log_W1)
        llr_u = log_marg1[:, 0] - log_marg1[:, 1]
        u_err += _genie_sc_per_position(llr_u, u)

        # Stage 2 — genie aid: use TRUE x
        log_W2 = _log_W_stage2(z, x, ch)
        log_marg2 = _forward_backward_2state(log_W2)
        llr_v = log_marg2[:, 0] - log_marg2[:, 1]
        v_err += _genie_sc_per_position(llr_v, v)

    return u_err, v_err, n_trials


def mc_design_chained_sct(N: int, n_trials: int, sigma2: float, h: float,
                          base_seed: int = 1234, n_workers: int = 1):
    """Parallel MC design. Returns (Pe_u, Pe_v) arrays of shape (N,)."""
    t0 = time.time()
    chunk = max(1, n_trials // n_workers)
    jobs = []
    started = 0
    for w in range(n_workers):
        nt = chunk if w < n_workers - 1 else (n_trials - started)
        jobs.append((N, sigma2, h, base_seed + w * 9001, nt))
        started += nt

    if n_workers == 1:
        results = [_mc_trial_chunk(jobs[0])]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_mc_trial_chunk, jobs)

    u_err = np.zeros(N, dtype=np.int64)
    v_err = np.zeros(N, dtype=np.int64)
    total = 0
    for ue, ve, nt in results:
        u_err += ue
        v_err += ve
        total += nt

    Pe_u = u_err / total
    Pe_v = v_err / total
    elapsed = time.time() - t0
    return Pe_u, Pe_v, total, elapsed


def pick_info_set(Pe: np.ndarray, k: int) -> list[int]:
    """1-indexed positions of the k channels with lowest Pe.

    Ties broken by polar bit-reversal weight (descending = more reliable first),
    matching the convention in polar.design_mc — critical when many MC error
    rates are 0 with finite trials.
    """
    order = _argsort_with_polar_tiebreak(np.asarray(Pe))
    info_0idx = sorted(int(i) for i in order[:k])
    return [i + 1 for i in info_0idx]


def _eval_chunk(args):
    """Eval BLER chunk for chained-SCT. Returns (errs, n_cw)."""
    N, sigma2, h, fu, fv, Au, Av, seed, n_cw = args
    ch = ISIMAC(sigma2=sigma2, h=h)
    rng = np.random.default_rng(seed)
    Au_set = set(int(p) for p in Au)
    Av_set = set(int(p) for p in Av)

    errs = 0
    u_errs = 0
    v_errs = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for p in Au_set:
            u[p - 1] = rng.integers(0, 2)
        for p in Av_set:
            v[p - 1] = rng.integers(0, 2)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = ch.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        u_hat, v_hat = decode_chained(z, N, fu, fv, ch)

        ue = any(int(u_hat[p - 1]) != int(u[p - 1]) for p in Au_set)
        ve = any(int(v_hat[p - 1]) != int(v[p - 1]) for p in Av_set)
        if ue:
            u_errs += 1
        if ve:
            v_errs += 1
        if ue or ve:
            errs += 1
    return errs, u_errs, v_errs, n_cw


def eval_chained_sct(N: int, n_cw: int, sigma2: float, h: float,
                     Au: list[int], Av: list[int],
                     base_seed: int = 5678, n_workers: int = 1):
    fu = {p: 0 for p in range(1, N + 1) if p not in set(Au)}
    fv = {p: 0 for p in range(1, N + 1) if p not in set(Av)}

    chunk = max(1, n_cw // n_workers)
    jobs = []
    started = 0
    for w in range(n_workers):
        nt = chunk if w < n_workers - 1 else (n_cw - started)
        jobs.append((N, sigma2, h, fu, fv, Au, Av, base_seed + w * 7777, nt))
        started += nt

    t0 = time.time()
    if n_workers == 1:
        results = [_eval_chunk(jobs[0])]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_eval_chunk, jobs)

    errs = sum(r[0] for r in results)
    u_errs = sum(r[1] for r in results)
    v_errs = sum(r[2] for r in results)
    total = sum(r[3] for r in results)
    elapsed = time.time() - t0
    return {
        "bler_chained": errs / total,
        "bler_u": u_errs / total,
        "bler_v": v_errs / total,
        "errs_chained": int(errs),
        "errs_u": int(u_errs),
        "errs_v": int(v_errs),
        "n_cw": total,
        "elapsed_s": elapsed,
    }


def run_one(N: int, n_design: int, n_eval: int, n_workers: int, out: dict):
    ku, kv = RATES[N]
    print(f"\n{'='*60}\nN={N}  ku={ku}  kv={kv}", flush=True)

    print(f"  MC design: {n_design} trials, {n_workers} workers...", flush=True)
    Pe_u, Pe_v, n_done, t_design = mc_design_chained_sct(
        N, n_design, SIGMA2, H_TAP, base_seed=2026 * N, n_workers=n_workers)
    print(f"  Design done: {n_done} trials in {t_design:.1f}s", flush=True)
    print(f"  Pe_u: best 5 = {sorted(Pe_u)[:5]}, worst 5 = {sorted(Pe_u, reverse=True)[:5]}", flush=True)
    print(f"  Pe_v: best 5 = {sorted(Pe_v)[:5]}, worst 5 = {sorted(Pe_v, reverse=True)[:5]}", flush=True)

    Au = pick_info_set(Pe_u, ku)
    Av = pick_info_set(Pe_v, kv)

    print(f"  Eval: {n_eval} CW, {n_workers} workers...", flush=True)
    r = eval_chained_sct(N, n_eval, SIGMA2, H_TAP, Au, Av,
                        base_seed=99 * N, n_workers=n_workers)
    print(f"  BLER chained = {r['bler_chained']:.5f} ({r['errs_chained']}/{r['n_cw']})", flush=True)
    print(f"  BLER U-only  = {r['bler_u']:.5f} ({r['errs_u']}/{r['n_cw']})", flush=True)
    print(f"  BLER V-only  = {r['bler_v']:.5f} ({r['errs_v']}/{r['n_cw']})", flush=True)
    print(f"  Eval time: {r['elapsed_s']:.1f}s", flush=True)

    out[str(N)] = {
        "N": N, "ku": ku, "kv": kv,
        "h": H_TAP, "snr_db": SNR_DB,
        "n_design_trials": n_done,
        "design_time_s": t_design,
        "Au": Au, "Av": Av,
        "Pe_u_top10": [float(Pe_u[p - 1]) for p in Au[:10]],
        "Pe_v_top10": [float(Pe_v[p - 1]) for p in Av[:10]],
        "Pe_u": [float(x) for x in Pe_u],
        "Pe_v": [float(x) for x in Pe_v],
        "eval": r,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ns", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--n-design", type=int, default=50000)
    parser.add_argument("--n-eval", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--out", type=str,
                        default=os.path.join(_HERE, "chained_sct_owndesign.json"))
    args = parser.parse_args()

    print(f"Chained SCT MC design + eval | h={H_TAP} SNR={SNR_DB}dB", flush=True)
    print(f"Ns={args.Ns} n_design={args.n_design} n_eval={args.n_eval} workers={args.workers}", flush=True)
    print(f"Output: {args.out}", flush=True)

    out = {
        "channel": "ISI-MAC",
        "h": H_TAP, "snr_db": SNR_DB, "sigma2": SIGMA2,
        "path": "Class C corner-rate (chained SU SCT)",
        "n_design_trials_per_N": args.n_design,
        "n_eval_per_N": args.n_eval,
        "n_workers": args.workers,
    }
    out["results"] = {}

    t_total = time.time()
    for N in args.Ns:
        run_one(N, args.n_design, args.n_eval, args.workers, out["results"])
        # Save after each N
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved to {args.out}", flush=True)

    out["total_time_s"] = time.time() - t_total
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDONE in {out['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
