"""Overnight sweep v2 — fixed pickling bugs from v1.

Each chunk function is at MODULE LEVEL so multiprocessing.Pool can pickle it.

Phases:
  A2: 4-state chained SCT at N=1024 with HIGHER design count (50K trials)
  B2: 4-state chained SCT at smaller N=16, 32, 64 (complete the data series)
  D2: Joint MAC SCT N=1024 push to 200K CW (tighten the upper bound)
  E2: MA-AGN whitened-SCT at N=128, 256, 512 (α=0.5)
  F2: 4-state chained SCT SNR sweep at N=128 (3, 4, 5, 6, 7, 8 dB)

Outputs in scripts/local_analysis/overnight2_*.json. Log /tmp/overnight2/log.txt.
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

LOG_DIR = "/tmp/overnight2"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "log.txt")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


WORKERS = 7
SNR_DB = 6.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)
H_TAP = 0.3
RATES = {16:(4,7),32:(7,15),64:(15,29),128:(30,58),256:(59,117),512:(119,233),1024:(239,467)}


# =========================== Module-level chunks ===========================
# All multiprocessing chunks defined at module level so Pool can pickle them.

def _jt_eval_chunk_n1024(args):
    """Joint MAC SCT eval chunk at N=1024 for Phase D2."""
    from polar.channels_memory import ISIMAC
    from polar.encoder import polar_encode
    from polar.decoder_trellis import decode_single
    from polar.design import make_path
    seed, Au, Av, nt = args
    N = 1024
    Au_s, Av_s = set(int(p) for p in Au), set(int(p) for p in Av)
    fu = {p: 0 for p in range(1, N + 1) if p not in Au_s}
    fv = {p: 0 for p in range(1, N + 1) if p not in Av_s}
    b_path = make_path(N, N)
    ch = ISIMAC(sigma2=SIGMA2, h=H_TAP)
    rng = np.random.default_rng(seed)
    errs = 0
    for _ in range(nt):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
        for p in Au_s: u[p-1] = rng.integers(0, 2)
        for p in Av_s: v[p-1] = rng.integers(0, 2)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = ch.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        u_hat, v_hat = decode_single(N, z, b_path, fu, fv, ch)
        ue = any(int(u_hat[p-1]) != int(u[p-1]) for p in Au_s)
        ve = any(int(v_hat[p-1]) != int(v[p-1]) for p in Av_s)
        if ue or ve:
            errs += 1
    return errs, nt


# MA-AGN whitened-SCT module-level chunks
def _maagn_whiten(z, alpha):
    zp = np.empty_like(z)
    zp[0] = z[0]
    zp[1:] = z[1:] - alpha * z[:-1]
    return zp


def _maagn_design_chunk(args):
    from polar.channels_memory import ISIMAC
    from polar.channels_memory_new import MAAGNMAC
    from polar.encoder import polar_encode
    from polar.decoder import _SCNode
    from polar.decoder_trellis import _forward_backward_joint
    from polar.decoder_trellis_mac_chained import _log_W_stage2, _forward_backward_2state
    N, alpha, sigma2_eff, sigma2_stat, h_eff, seed, n_trials = args
    isi_eff = ISIMAC(sigma2=sigma2_eff, h=h_eff)
    maagn = MAAGNMAC(sigma2=sigma2_stat, alpha=alpha)
    rng = np.random.default_rng(seed)
    u_err = np.zeros(N, dtype=np.int64)
    v_err = np.zeros(N, dtype=np.int64)
    for _ in range(n_trials):
        u = rng.integers(0, 2, size=N)
        v = rng.integers(0, 2, size=N)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = maagn.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        zp = _maagn_whiten(z, alpha)
        log_W = isi_eff.build_leaf_tensors(zp)
        log_marg = _forward_backward_joint(log_W, N, isi_eff.num_states)
        log_pu0 = np.logaddexp(log_marg[:, 0, 0], log_marg[:, 0, 1])
        log_pu1 = np.logaddexp(log_marg[:, 1, 0], log_marg[:, 1, 1])
        llr_u = log_pu0 - log_pu1
        node = _SCNode(llr_u.astype(np.float64))
        for i in range(N):
            L = node.get_llr(i)
            decided = 0 if L >= 0 else 1
            true_val = int(u[i])
            if decided != true_val:
                u_err[i] += 1
            node.feed(i, true_val)
        log_W2 = _log_W_stage2(zp, x, isi_eff)
        log_marg2 = _forward_backward_2state(log_W2)
        llr_v = log_marg2[:, 0] - log_marg2[:, 1]
        node = _SCNode(llr_v.astype(np.float64))
        for i in range(N):
            L = node.get_llr(i)
            decided = 0 if L >= 0 else 1
            true_val = int(v[i])
            if decided != true_val:
                v_err[i] += 1
            node.feed(i, true_val)
    return u_err, v_err, n_trials


def _maagn_eval_chunk(args):
    from polar.channels_memory import ISIMAC
    from polar.channels_memory_new import MAAGNMAC
    from polar.encoder import polar_encode
    from polar.decoder_trellis import decode_single
    from polar.design import make_path
    N, alpha, sigma2_eff, sigma2_stat, h_eff, fu, fv, Au, Av, seed, n_cw = args
    isi_eff = ISIMAC(sigma2=sigma2_eff, h=h_eff)
    maagn = MAAGNMAC(sigma2=sigma2_stat, alpha=alpha)
    rng = np.random.default_rng(seed)
    Au_s, Av_s = set(int(p) for p in Au), set(int(p) for p in Av)
    b_path = make_path(N, N)
    errs = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
        for p in Au_s: u[p-1] = rng.integers(0, 2)
        for p in Av_s: v[p-1] = rng.integers(0, 2)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = maagn.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        zp = _maagn_whiten(z, alpha)
        u_hat, v_hat = decode_single(N, zp, b_path, fu, fv, isi_eff)
        ue = any(int(u_hat[p-1]) != int(u[p-1]) for p in Au_s)
        ve = any(int(v_hat[p-1]) != int(v[p-1]) for p in Av_s)
        if ue or ve:
            errs += 1
    return errs, n_cw


# ================================ Phases =================================

def phase_A2_4state_n1024_redo(out_path):
    log("=" * 60); log("PHASE A2: 4-state chained SCT N=1024 with 50K design trials")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    from chained_sct_4state import mc_design, eval_at, pick
    N = 1024; ku, kv = RATES[N]
    log(f"  N={N} ku={ku} kv={kv}")
    t0 = time.time()
    Pe_u, Pe_v, n_done = mc_design(N, 50000, WORKERS, base_seed=4001)
    td = time.time() - t0
    log(f"    design {n_done} trials in {td:.1f}s")
    Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
    t0 = time.time()
    r = eval_at(N, 10000, Au, Av, WORKERS, base_seed=4002)
    te = time.time() - t0
    log(f"    eval {r['n_cw']} CW: BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']}), {te:.1f}s")
    out = {"phase":"A2","decoder":"4-state chained SCT (50K design)",
           "channel":"ISI-MAC","h":H_TAP,"snr_db":SNR_DB,
           "results":{"1024":{"N":N,"ku":ku,"kv":kv,
                              "n_design":n_done,"design_time_s":td,
                              "Au":Au,"Av":Av,"eval":r}}}
    with open(out_path, "w") as f: json.dump(out, f, indent=2)


def phase_B2_4state_small_n(out_path):
    log("=" * 60); log("PHASE B2: 4-state chained SCT at N=16, 32, 64")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    from chained_sct_4state import mc_design, eval_at, pick
    out = {"phase":"B2","decoder":"4-state chained SCT","channel":"ISI-MAC",
           "h":H_TAP,"snr_db":SNR_DB,"results":{}}
    for N in [16, 32, 64]:
        ku, kv = RATES[N]
        log(f"  N={N} ku={ku} kv={kv}")
        t0 = time.time()
        Pe_u, Pe_v, n_done = mc_design(N, 50000, WORKERS, base_seed=5001+N)
        td = time.time() - t0
        log(f"    design {n_done} in {td:.1f}s")
        Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
        t0 = time.time()
        r = eval_at(N, 30000, Au, Av, WORKERS, base_seed=5002+N)
        te = time.time() - t0
        log(f"    eval {r['n_cw']} CW: BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']}), {te:.1f}s")
        out["results"][str(N)] = {"N":N,"ku":ku,"kv":kv,
                                   "n_design":n_done,"design_time_s":td,
                                   "Au":Au,"Av":Av,"eval":r}
        with open(out_path, "w") as f: json.dump(out, f, indent=2)


def phase_D2_jt_n1024_push(out_path):
    log("=" * 60); log("PHASE D2: Joint MAC SCT at N=1024, push to 200K CW")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    # Get Au, Av from Phase A's 4-state design (best available) or fall back
    src = None
    for path in [os.path.join(_HERE, "overnight2_A2_4state_n1024.json"),
                 os.path.join(_HERE, "overnight_A_4state_largeN.json")]:
        if os.path.exists(path):
            src = path; break
    if src is None:
        log("  no source for N=1024 design; SKIP"); return
    d = json.load(open(src))
    res = d.get("results", {})
    if "1024" not in res:
        log("  source has no N=1024; SKIP"); return
    Au = res["1024"]["Au"]; Av = res["1024"]["Av"]
    log(f"  using design from {src}")

    target_n_cw = 200000
    cw_per = max(1, target_n_cw // WORKERS)
    jobs = []; started = 0
    for w in range(WORKERS):
        nt = cw_per if w < WORKERS - 1 else (target_n_cw - started)
        jobs.append((30000 + w, Au, Av, nt))
        started += nt
    t0 = time.time()
    log(f"  launching {WORKERS} workers, ~{jobs[0][3]} CW each")
    with Pool(WORKERS) as pool:
        results = pool.map(_jt_eval_chunk_n1024, jobs)
    total_errs = sum(r[0] for r in results)
    total_cw = sum(r[1] for r in results)
    elapsed = time.time() - t0
    bler = total_errs / total_cw
    ucl = 3.0 / total_cw if total_errs == 0 else (total_errs + 1.96*np.sqrt(total_errs)) / total_cw
    log(f"  N=1024 joint MAC SCT: {total_errs}/{total_cw} BLER={bler:.3e} UCL95={ucl:.3e}  in {elapsed:.1f}s")
    out = {"phase":"D2","decoder":"Joint MAC SCT N=1024",
           "channel":"ISI-MAC","h":H_TAP,"snr_db":SNR_DB,
           "N":1024,"Au":Au,"Av":Av,
           "n_cw":total_cw,"errs":int(total_errs),"bler":bler,"ucl_95":ucl,
           "elapsed_s":elapsed,"design_source":os.path.basename(src)}
    with open(out_path, "w") as f: json.dump(out, f, indent=2)


def phase_E2_maagn_whitened(out_path):
    log("=" * 60); log("PHASE E2: MA-AGN whitened-SCT at N=128, 256, 512 (α=0.5)")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    alpha = 0.5
    sigma2_stat = SIGMA2
    sigma2_eff = sigma2_stat * (1 - alpha ** 2)
    h_eff = -alpha
    from polar.design_mc import _argsort_with_polar_tiebreak
    log(f"  MA-AGN α={alpha} σ²_stat={sigma2_stat:.4f} → ISI h={h_eff} σ²_eff={sigma2_eff:.4f}")
    out = {"phase":"E2","decoder":"MA-AGN whitened-SCT",
           "channel":"MA-AGN","alpha":alpha,"snr_db":SNR_DB,
           "h_equiv":h_eff,"sigma2_eff":sigma2_eff,"results":{}}
    designs = {128: 30000, 256: 20000, 512: 10000}
    evals = {128: 10000, 256: 10000, 512: 5000}
    for N in [128, 256, 512]:
        ku, kv = RATES[N]
        log(f"  N={N} ku={ku} kv={kv}: design")
        # Design
        n_trials = designs[N]
        chunk_size = max(1, n_trials // WORKERS)
        jobs = []; started = 0
        for w in range(WORKERS):
            nt = chunk_size if w < WORKERS - 1 else (n_trials - started)
            jobs.append((N, alpha, sigma2_eff, sigma2_stat, h_eff,
                          70000+w*9001+N, nt))
            started += nt
        t0 = time.time()
        with Pool(WORKERS) as pool:
            results = pool.map(_maagn_design_chunk, jobs)
        u_err = sum(r[0] for r in results)
        v_err = sum(r[1] for r in results)
        n_done = sum(r[2] for r in results)
        td = time.time() - t0
        Pe_u = u_err / n_done; Pe_v = v_err / n_done
        order_u = _argsort_with_polar_tiebreak(np.asarray(Pe_u))
        order_v = _argsort_with_polar_tiebreak(np.asarray(Pe_v))
        Au = [int(i)+1 for i in sorted(order_u[:ku].tolist())]
        Av = [int(i)+1 for i in sorted(order_v[:kv].tolist())]
        log(f"    design {n_done} in {td:.1f}s")
        # Eval
        n_cw = evals[N]
        fu = {p: 0 for p in range(1, N + 1) if p not in set(Au)}
        fv = {p: 0 for p in range(1, N + 1) if p not in set(Av)}
        chunk_size = max(1, n_cw // WORKERS)
        jobs = []; started = 0
        for w in range(WORKERS):
            nt = chunk_size if w < WORKERS - 1 else (n_cw - started)
            jobs.append((N, alpha, sigma2_eff, sigma2_stat, h_eff,
                          fu, fv, Au, Av, 80000+w+N, nt))
            started += nt
        t0 = time.time()
        with Pool(WORKERS) as pool:
            results = pool.map(_maagn_eval_chunk, jobs)
        errs = sum(r[0] for r in results)
        total_cw = sum(r[1] for r in results)
        te = time.time() - t0
        bler = errs / total_cw
        log(f"    eval {total_cw} CW: BLER={bler:.6f} ({errs}/{total_cw}), {te:.1f}s")
        out["results"][str(N)] = {"N":N,"ku":ku,"kv":kv,
                                   "n_design":n_done,"design_time_s":td,
                                   "Au":Au,"Av":Av,
                                   "eval":{"n_cw":total_cw,"errs":int(errs),
                                            "bler":bler,"elapsed_s":te}}
        with open(out_path, "w") as f: json.dump(out, f, indent=2)


def phase_F2_chained_snr_sweep(out_path):
    log("=" * 60); log("PHASE F2: 4-state chained SCT SNR sweep at N=128")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    from polar.channels_memory import ISIMAC
    from chained_sct_4state import mc_design, eval_at, pick
    from chained_sct_4state import H_TAP as _H, SIGMA2 as _S2  # check existing constants
    # Override SIGMA2 module global via a workaround: build per-SNR ch and pass.
    # The module's mc_design uses module-level SIGMA2 — so we monkey-patch.
    import chained_sct_4state as cs4
    out = {"phase":"F2","decoder":"4-state chained SCT SNR sweep",
           "channel":"ISI-MAC","h":H_TAP,"N":128,"results":{}}
    N = 128
    ku, kv = RATES[N]
    for snr_db in [3, 4, 5, 6, 7, 8]:
        sigma2 = 10 ** (-snr_db / 10)
        log(f"  SNR={snr_db}dB σ²={sigma2:.4f}")
        cs4.SIGMA2 = sigma2  # monkey-patch
        t0 = time.time()
        Pe_u, Pe_v, n_done = mc_design(N, 30000, WORKERS, base_seed=6000+snr_db)
        td = time.time() - t0
        Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
        t0 = time.time()
        r = eval_at(N, 10000, Au, Av, WORKERS, base_seed=6100+snr_db)
        te = time.time() - t0
        log(f"    BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']})  design {td:.0f}s eval {te:.0f}s")
        out["results"][str(snr_db)] = {"snr_db":snr_db,"sigma2":sigma2,
                                        "N":N,"ku":ku,"kv":kv,
                                        "n_design":n_done,
                                        "Au":Au,"Av":Av,"eval":r}
        with open(out_path, "w") as f: json.dump(out, f, indent=2)
    cs4.SIGMA2 = 10 ** (-0.6)  # restore


# ================================ Driver =================================
def main():
    log("################################################")
    log("OVERNIGHT v2 starting")
    log("################################################")
    t_all = time.time()
    # Run in priority order; A2 first because D2 depends on it
    try: phase_A2_4state_n1024_redo(os.path.join(_HERE, "overnight2_A2_4state_n1024.json"))
    except Exception as e: log(f"PHASE A2 FAILED: {e}")
    try: phase_B2_4state_small_n(os.path.join(_HERE, "overnight2_B2_4state_smallN.json"))
    except Exception as e: log(f"PHASE B2 FAILED: {e}")
    try: phase_E2_maagn_whitened(os.path.join(_HERE, "overnight2_E2_maagn_whitened.json"))
    except Exception as e: log(f"PHASE E2 FAILED: {e}")
    try: phase_F2_chained_snr_sweep(os.path.join(_HERE, "overnight2_F2_snr_sweep.json"))
    except Exception as e: log(f"PHASE F2 FAILED: {e}")
    try: phase_D2_jt_n1024_push(os.path.join(_HERE, "overnight2_D2_jt_n1024.json"))
    except Exception as e: log(f"PHASE D2 FAILED: {e}")
    log(f"ALL DONE in {(time.time()-t_all)/3600:.2f} hr")


if __name__ == "__main__":
    main()
