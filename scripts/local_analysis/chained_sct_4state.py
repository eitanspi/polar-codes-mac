"""4-state chained SCT for corner-rate ISI-MAC.

The fix for the per-position double-counting bug in chained SCT:
use a 4-state lattice (X_{t-1}, Y_{t-1}) and properly marginalize Y
across positions. For corner-rate (Class C), this should match the joint
MAC SCT.

Stage 1: 4-state FB → (N, 2, 2) joint marginal → marginalize Y per
position → scalar LLR → Arikan SC for U.

Stage 2: same as the old chained SCT — 2-state Y-FB given known X.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode
from polar.decoder import _SCNode, _sc_decode_from_llr
from polar.decoder_trellis import _forward_backward_joint
from polar.decoder_trellis_mac_chained import _log_W_stage2, _forward_backward_2state, _marg_to_llr
from polar.design_mc import _argsort_with_polar_tiebreak

SNR_DB = 6.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)
H_TAP = 0.3

RATES = {
    16: (4, 7), 32: (7, 15), 64: (15, 29), 128: (30, 58),
    256: (59, 117), 512: (119, 233), 1024: (239, 467),
}


def stage1_u_llr_4state(z, ch):
    """4-state FB → marginalize V → scalar LLR for X_t."""
    log_W = ch.build_leaf_tensors(z)              # (N, 2, 2, 4, 4)
    log_marg = _forward_backward_joint(log_W, len(z), ch.num_states)  # (N, 2, 2)
    # Marginalize Y per position: log P(z | X_t=x) = LSE_y log_marg[t, x, y]
    log_pu0 = np.logaddexp(log_marg[:, 0, 0], log_marg[:, 0, 1])
    log_pu1 = np.logaddexp(log_marg[:, 1, 0], log_marg[:, 1, 1])
    return log_pu0 - log_pu1                       # LLR


def decode_4state_chained(z, N, fu, fv, ch):
    """Chained decoder using proper 4-state stage-1 FB."""
    llr_u = stage1_u_llr_4state(z, ch)
    u_hat = _sc_decode_from_llr(llr_u, fu)
    # Stage 2: identical to existing chained SCT (Y-state-2 FB, X known)
    x_hat = np.array(polar_encode(list(map(int, u_hat))), dtype=np.int64)
    log_W2 = _log_W_stage2(z, x_hat, ch)
    log_marg2 = _forward_backward_2state(log_W2)
    llr_v = _marg_to_llr(log_marg2)
    v_hat = _sc_decode_from_llr(llr_v, fv)
    return u_hat, v_hat


# ------------------ MC design (proper 4-state stage 1) ------------------

def _mc_chunk(args):
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

        # Stage 1: 4-state FB
        llr_u = stage1_u_llr_4state(z, ch)
        node = _SCNode(llr_u.astype(np.float64))
        for i in range(N):
            L = node.get_llr(i)
            decided = 0 if L >= 0 else 1
            true_val = int(u[i])
            if decided != true_val:
                u_err[i] += 1
            node.feed(i, true_val)

        # Stage 2: with TRUE x (genie)
        log_W2 = _log_W_stage2(z, x, ch)
        log_marg2 = _forward_backward_2state(log_W2)
        llr_v = _marg_to_llr(log_marg2)
        node = _SCNode(llr_v.astype(np.float64))
        for i in range(N):
            L = node.get_llr(i)
            decided = 0 if L >= 0 else 1
            true_val = int(v[i])
            if decided != true_val:
                v_err[i] += 1
            node.feed(i, true_val)

    return u_err, v_err, n_trials


def mc_design(N, n_trials, workers, base_seed):
    chunk = max(1, n_trials // workers)
    jobs, started = [], 0
    for w in range(workers):
        nt = chunk if w < workers - 1 else (n_trials - started)
        jobs.append((N, SIGMA2, H_TAP, base_seed + w * 9001, nt))
        started += nt
    with Pool(workers) as p:
        results = p.map(_mc_chunk, jobs)
    u_err = sum(r[0] for r in results)
    v_err = sum(r[1] for r in results)
    total = sum(r[2] for r in results)
    return u_err / total, v_err / total, total


def _eval_chunk(args):
    N, sigma2, h, fu, fv, Au, Av, seed, n_cw = args
    ch = ISIMAC(sigma2=sigma2, h=h)
    rng = np.random.default_rng(seed)
    Au_s, Av_s = set(int(p) for p in Au), set(int(p) for p in Av)
    errs = u_errs = v_errs = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
        for p in Au_s: u[p - 1] = rng.integers(0, 2)
        for p in Av_s: v[p - 1] = rng.integers(0, 2)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = ch.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        u_hat, v_hat = decode_4state_chained(z, N, fu, fv, ch)
        ue = any(int(u_hat[p - 1]) != int(u[p - 1]) for p in Au_s)
        ve = any(int(v_hat[p - 1]) != int(v[p - 1]) for p in Av_s)
        if ue: u_errs += 1
        if ve: v_errs += 1
        if ue or ve: errs += 1
    return errs, u_errs, v_errs, n_cw


def eval_at(N, n_cw, Au, Av, workers, base_seed):
    fu = {p: 0 for p in range(1, N + 1) if p not in set(Au)}
    fv = {p: 0 for p in range(1, N + 1) if p not in set(Av)}
    chunk = max(1, n_cw // workers)
    jobs, started = [], 0
    for w in range(workers):
        nt = chunk if w < workers - 1 else (n_cw - started)
        jobs.append((N, SIGMA2, H_TAP, fu, fv, Au, Av, base_seed + w * 7777, nt))
        started += nt
    with Pool(workers) as p:
        results = p.map(_eval_chunk, jobs)
    errs = sum(r[0] for r in results)
    u_errs = sum(r[1] for r in results)
    v_errs = sum(r[2] for r in results)
    total = sum(r[3] for r in results)
    return dict(bler=errs/total, errs=errs, bler_u=u_errs/total, errs_u=u_errs,
                bler_v=v_errs/total, errs_v=v_errs, n_cw=total)


def pick(Pe, k):
    order = _argsort_with_polar_tiebreak(np.asarray(Pe))
    return [int(i) + 1 for i in sorted(order[:k].tolist())]


def main():
    Ns = [128, 256]
    N_DESIGN = 30000   # 4-state is slower; trim
    N_EVAL = 10000
    WORKERS = 7
    out = {"channel": "ISI-MAC", "h": H_TAP, "snr_db": SNR_DB,
           "decoder": "4-state chained SCT (corner-rate)", "results": {}}
    t_all = time.time()
    for N in Ns:
        ku, kv = RATES[N]
        print(f"\n=== N={N} ku={ku} kv={kv} ===", flush=True)
        t0 = time.time()
        Pe_u, Pe_v, n_done = mc_design(N, N_DESIGN, WORKERS, base_seed=2027 * N)
        td = time.time() - t0
        Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
        print(f"  design ({n_done} trials, {td:.1f}s):", flush=True)
        print(f"    Pe_u min={min(Pe_u):.5f} max={max(Pe_u):.5f}", flush=True)
        print(f"    Pe_v min={min(Pe_v):.5f} max={max(Pe_v):.5f}", flush=True)
        t0 = time.time()
        r = eval_at(N, N_EVAL, Au, Av, WORKERS, base_seed=99 * N)
        te = time.time() - t0
        print(f"  eval ({N_EVAL} CW, {te:.1f}s): BLER={r['bler']:.5f} ({r['errs']}/{r['n_cw']})", flush=True)
        print(f"    bler_u={r['bler_u']:.5f}  bler_v={r['bler_v']:.5f}", flush=True)
        out["results"][str(N)] = {
            "N": N, "ku": ku, "kv": kv, "Au": Au, "Av": Av,
            "n_design": n_done, "design_time_s": td, "eval_time_s": te,
            "eval": r,
        }
        with open(os.path.join(_HERE, "chained_sct_4state.json"), "w") as f:
            json.dump(out, f, indent=2)
    out["total_time_s"] = time.time() - t_all
    with open(os.path.join(_HERE, "chained_sct_4state.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDONE in {out['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
