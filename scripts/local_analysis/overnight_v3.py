"""Overnight v3 (Shabbat run, ~12 hr budget).

Phases:
  G3: MA-AGN whitened-SCT α sweep — α ∈ {0.3, 0.7, 0.9} at N=128, 256
       (we already have α=0.5 from v2)
  H3: 4-state chained SCT SNR sweep at N=256 (extend the N=128 sweep)
  I3: 4-state chained SCT at N=128 with even more design trials (100K)
       to get tight final numbers

All chunks at module level so multiprocessing.Pool can pickle them.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
if _HERE not in sys.path: sys.path.insert(0, _HERE)

LOG_DIR = "/tmp/overnight3"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "log.txt")

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

WORKERS = 7
SNR_DB = 6.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)
H_TAP = 0.3
RATES = {16:(4,7),32:(7,15),64:(15,29),128:(30,58),256:(59,117),512:(119,233),1024:(239,467)}


# =================== MA-AGN whitened-SCT chunks (module-level) ===================
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
        u = rng.integers(0, 2, size=N); v = rng.integers(0, 2, size=N)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = maagn.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
        zp = _maagn_whiten(z, alpha)
        log_W = isi_eff.build_leaf_tensors(zp)
        log_marg = _forward_backward_joint(log_W, N, isi_eff.num_states)
        log_pu0 = np.logaddexp(log_marg[:,0,0], log_marg[:,0,1])
        log_pu1 = np.logaddexp(log_marg[:,1,0], log_marg[:,1,1])
        llr_u = log_pu0 - log_pu1
        node = _SCNode(llr_u.astype(np.float64))
        for i in range(N):
            L = node.get_llr(i); decided = 0 if L >= 0 else 1
            tv = int(u[i])
            if decided != tv: u_err[i] += 1
            node.feed(i, tv)
        log_W2 = _log_W_stage2(zp, x, isi_eff)
        log_marg2 = _forward_backward_2state(log_W2)
        llr_v = log_marg2[:,0] - log_marg2[:,1]
        node = _SCNode(llr_v.astype(np.float64))
        for i in range(N):
            L = node.get_llr(i); decided = 0 if L >= 0 else 1
            tv = int(v[i])
            if decided != tv: v_err[i] += 1
            node.feed(i, tv)
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
    Au_s = set(int(p) for p in Au); Av_s = set(int(p) for p in Av)
    b_path = make_path(N, N)
    errs = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
        for p in Au_s: u[p-1] = rng.integers(0, 2)
        for p in Av_s: v[p-1] = rng.integers(0, 2)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = maagn.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
        zp = _maagn_whiten(z, alpha)
        u_hat, v_hat = decode_single(N, zp, b_path, fu, fv, isi_eff)
        ue = any(int(u_hat[p-1]) != int(u[p-1]) for p in Au_s)
        ve = any(int(v_hat[p-1]) != int(v[p-1]) for p in Av_s)
        if ue or ve: errs += 1
    return errs, n_cw


# =================== Phases ===================
def phase_G3_maagn_alpha_sweep(out_path):
    log("="*60); log("PHASE G3: MA-AGN whitened-SCT α sweep — α ∈ {0.3, 0.7, 0.9}")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    from polar.design_mc import _argsort_with_polar_tiebreak
    out = {"phase":"G3","decoder":"MA-AGN whitened-SCT α sweep",
           "channel":"MA-AGN","snr_db":SNR_DB,"results":{}}
    designs = {128: 30000, 256: 20000}
    evals = {128: 10000, 256: 10000}
    for alpha in [0.3, 0.7, 0.9]:
        sigma2_stat = SIGMA2
        sigma2_eff = sigma2_stat * (1 - alpha**2)
        h_eff = -alpha
        log(f"  α={alpha} σ²_eff={sigma2_eff:.4f} h={h_eff}")
        out["results"][str(alpha)] = {"alpha":alpha,"h_equiv":h_eff,
                                       "sigma2_eff":sigma2_eff,"by_N":{}}
        for N in [128, 256]:
            ku, kv = RATES[N]
            log(f"    N={N} ku={ku} kv={kv}: design")
            n_trials = designs[N]
            chunk = max(1, n_trials // WORKERS)
            jobs, started = [], 0
            for w in range(WORKERS):
                nt = chunk if w < WORKERS-1 else (n_trials - started)
                jobs.append((N, alpha, sigma2_eff, sigma2_stat, h_eff,
                              90000+w*9001+N+int(alpha*100), nt))
                started += nt
            t0 = time.time()
            with Pool(WORKERS) as p:
                results = p.map(_maagn_design_chunk, jobs)
            u_err = sum(r[0] for r in results)
            v_err = sum(r[1] for r in results)
            n_done = sum(r[2] for r in results)
            td = time.time() - t0
            Pe_u = u_err / n_done; Pe_v = v_err / n_done
            order_u = _argsort_with_polar_tiebreak(np.asarray(Pe_u))
            order_v = _argsort_with_polar_tiebreak(np.asarray(Pe_v))
            Au = [int(i)+1 for i in sorted(order_u[:ku].tolist())]
            Av = [int(i)+1 for i in sorted(order_v[:kv].tolist())]
            log(f"      design {n_done} in {td:.1f}s")
            n_cw = evals[N]
            fu = {p:0 for p in range(1,N+1) if p not in set(Au)}
            fv = {p:0 for p in range(1,N+1) if p not in set(Av)}
            chunk = max(1, n_cw // WORKERS)
            jobs, started = [], 0
            for w in range(WORKERS):
                nt = chunk if w < WORKERS-1 else (n_cw - started)
                jobs.append((N, alpha, sigma2_eff, sigma2_stat, h_eff,
                              fu, fv, Au, Av, 95000+w+N+int(alpha*100), nt))
                started += nt
            t0 = time.time()
            with Pool(WORKERS) as p:
                results = p.map(_maagn_eval_chunk, jobs)
            errs = sum(r[0] for r in results); tc = sum(r[1] for r in results)
            te = time.time() - t0
            bler = errs/tc
            log(f"      eval {tc} CW: BLER={bler:.6f} ({errs}/{tc}), {te:.1f}s")
            out["results"][str(alpha)]["by_N"][str(N)] = {
                "N":N,"ku":ku,"kv":kv,"n_design":n_done,
                "design_time_s":td,"Au":Au,"Av":Av,
                "eval":{"n_cw":tc,"errs":int(errs),"bler":bler,"elapsed_s":te},
            }
            with open(out_path, "w") as f: json.dump(out, f, indent=2)


def phase_H3_snr_sweep_n256(out_path):
    log("="*60); log("PHASE H3: 4-state chained SCT SNR sweep at N=256")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    from chained_sct_4state import mc_design, eval_at, pick
    import chained_sct_4state as cs4
    N = 256; ku, kv = RATES[N]
    out = {"phase":"H3","decoder":"4-state chained SCT SNR sweep N=256",
           "channel":"ISI-MAC","h":H_TAP,"N":N,"results":{}}
    for snr_db in [3, 4, 5, 6, 7, 8]:
        sigma2 = 10**(-snr_db/10)
        log(f"  SNR={snr_db} σ²={sigma2:.4f}")
        cs4.SIGMA2 = sigma2
        t0 = time.time()
        Pe_u, Pe_v, n_done = mc_design(N, 20000, WORKERS, base_seed=110000+snr_db)
        td = time.time() - t0
        Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
        t0 = time.time()
        r = eval_at(N, 10000, Au, Av, WORKERS, base_seed=120000+snr_db)
        te = time.time() - t0
        log(f"    BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']})  design {td:.0f}s eval {te:.0f}s")
        out["results"][str(snr_db)] = {"snr_db":snr_db,"sigma2":sigma2,
                                         "N":N,"ku":ku,"kv":kv,
                                         "n_design":n_done,"Au":Au,"Av":Av,"eval":r}
        with open(out_path, "w") as f: json.dump(out, f, indent=2)
    cs4.SIGMA2 = 10**(-0.6)


def phase_I3_4state_n128_tight(out_path):
    log("="*60); log("PHASE I3: 4-state chained SCT N=128 with 100K design (tight)")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    from chained_sct_4state import mc_design, eval_at, pick
    N = 128; ku, kv = RATES[N]
    log(f"  N={N} ku={ku} kv={kv}")
    t0 = time.time()
    Pe_u, Pe_v, n_done = mc_design(N, 100000, WORKERS, base_seed=130000)
    td = time.time() - t0
    Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
    log(f"  design {n_done} in {td:.1f}s")
    t0 = time.time()
    r = eval_at(N, 30000, Au, Av, WORKERS, base_seed=131000)
    te = time.time() - t0
    log(f"  eval {r['n_cw']} CW: BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']}), {te:.1f}s")
    out = {"phase":"I3","decoder":"4-state chained SCT N=128 (100K design)",
           "channel":"ISI-MAC","h":H_TAP,"snr_db":SNR_DB,
           "N":N,"ku":ku,"kv":kv,
           "n_design":n_done,"design_time_s":td,"Au":Au,"Av":Av,"eval":r}
    with open(out_path, "w") as f: json.dump(out, f, indent=2)


def main():
    log("################################################")
    log("OVERNIGHT v3 starting")
    log("################################################")
    t_all = time.time()
    # G3 is the most valuable result: completes the MA-AGN α sweep with proper analytical baseline
    try: phase_G3_maagn_alpha_sweep(os.path.join(_HERE, "overnight3_G3_maagn_alpha.json"))
    except Exception as e: log(f"PHASE G3 FAILED: {e}")
    try: phase_H3_snr_sweep_n256(os.path.join(_HERE, "overnight3_H3_snr_sweep_n256.json"))
    except Exception as e: log(f"PHASE H3 FAILED: {e}")
    try: phase_I3_4state_n128_tight(os.path.join(_HERE, "overnight3_I3_n128_tight.json"))
    except Exception as e: log(f"PHASE I3 FAILED: {e}")
    log(f"ALL DONE in {(time.time()-t_all)/3600:.2f} hr")


if __name__ == "__main__":
    main()
