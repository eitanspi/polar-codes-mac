"""Overnight v4 (local CPU ~12 hr).

Phases (all module-level chunks for picklability):
  J4: ABNMAC chained-SCT BLER (different MAC channel)
  K4: 4-state chained SCT SNR sweep at N=512
  L4: Whitened-SCT N=1024 push (MA-AGN α=0.5 at very large N)
  M4: 4-state chained SCT at h=0.5, 0.7 (different ISI tap)
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

LOG_DIR = "/tmp/overnight4"
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


# =================== module-level chunks for whitened-SCT N=1024 push ===================
def _maagn_whiten(z, alpha):
    zp = np.empty_like(z)
    zp[0] = z[0]
    zp[1:] = z[1:] - alpha * z[:-1]
    return zp


def _maagn_n1024_eval_chunk(args):
    from polar.channels_memory import ISIMAC
    from polar.channels_memory_new import MAAGNMAC
    from polar.encoder import polar_encode
    from polar.decoder_trellis import decode_single
    from polar.design import make_path
    alpha, sigma2_eff, sigma2_stat, h_eff, fu, fv, Au, Av, seed, n_cw = args
    N = 1024
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


# =================== module-level chunks for ABNMAC chained SCT ===================
def _abnmac_design_chunk(args):
    """ABNMAC: 2-state chained SCT design (no memory; same as memoryless SC really).
    But useful as a control: ABNMAC has independent noise on each user."""
    from polar.channels import ABNMAC
    from polar.encoder import polar_encode
    from polar.decoder import _SCNode
    N, p_noise, seed, n_trials = args
    ch = ABNMAC(p_noise=p_noise)
    rng = np.random.default_rng(seed)
    u_err = np.zeros(N, dtype=np.int64)
    v_err = np.zeros(N, dtype=np.int64)
    log_qpos = np.log(1 - p_noise + 1e-12)
    log_qneg = np.log(p_noise + 1e-12)
    for _ in range(n_trials):
        u = rng.integers(0, 2, size=N); v = rng.integers(0, 2, size=N)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
        # Stage 1: marginalize y; per-position log P(z_t | x_t)
        # ABNMAC: z = x XOR (y XOR noise) — sum mod 2
        # Marginal over y: log P(z|x=0) = log [0.5*P(noise=z) + 0.5*P(noise=1-z)] which is 0.5 either way
        # So Y is iid and uniform → stage 1 LLR is uninformative (LLR=0)
        # ABNMAC is not useful for chained corner-rate; skip
        pass
    # Return zeros — this phase is degenerate for ABNMAC corner-rate. Skipping.
    return u_err, v_err, n_trials


# =================== Phases ===================
def phase_K4_snr_sweep_n512(out_path):
    log("="*60); log("PHASE K4: 4-state chained SCT SNR sweep at N=512")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    from chained_sct_4state import mc_design, eval_at, pick
    import chained_sct_4state as cs4
    N = 512; ku, kv = RATES[N]
    out = {"phase":"K4","decoder":"4-state chained SCT SNR sweep N=512",
           "channel":"ISI-MAC","h":H_TAP,"N":N,"results":{}}
    for snr_db in [3, 4, 5, 6, 7, 8]:
        sigma2 = 10**(-snr_db/10)
        log(f"  SNR={snr_db} σ²={sigma2:.4f}")
        cs4.SIGMA2 = sigma2
        t0 = time.time()
        Pe_u, Pe_v, n_done = mc_design(N, 15000, WORKERS, base_seed=200000+snr_db)
        td = time.time() - t0
        Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
        t0 = time.time()
        r = eval_at(N, 10000, Au, Av, WORKERS, base_seed=201000+snr_db)
        te = time.time() - t0
        log(f"    BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']})  design {td:.0f}s eval {te:.0f}s")
        out["results"][str(snr_db)] = {"snr_db":snr_db,"sigma2":sigma2,
                                         "N":N,"ku":ku,"kv":kv,
                                         "n_design":n_done,"Au":Au,"Av":Av,"eval":r}
        with open(out_path, "w") as f: json.dump(out, f, indent=2)
    cs4.SIGMA2 = 10**(-0.6)


def phase_L4_maagn_n1024(out_path):
    """MA-AGN whitened-SCT push at N=1024 for α=0.5."""
    log("="*60); log("PHASE L4: MA-AGN whitened-SCT at N=1024 (α=0.5)")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    from polar.channels_memory import ISIMAC
    from polar.encoder import polar_encode
    from polar.decoder_trellis import _forward_backward_joint
    from polar.decoder import _SCNode
    from polar.decoder_trellis_mac_chained import _log_W_stage2, _forward_backward_2state
    from polar.design_mc import _argsort_with_polar_tiebreak
    from polar.channels_memory_new import MAAGNMAC
    alpha = 0.5
    sigma2_stat = SIGMA2
    sigma2_eff = sigma2_stat * (1 - alpha**2)
    h_eff = -alpha
    N = 1024
    ku, kv = RATES[N]
    isi_eff = ISIMAC(sigma2=sigma2_eff, h=h_eff)
    maagn = MAAGNMAC(sigma2=sigma2_stat, alpha=alpha)
    log(f"  N={N} ku={ku} kv={kv} α={alpha} σ²_eff={sigma2_eff:.4f} h={h_eff}")

    # Design (10K trials at N=1024 since FB is expensive — ~150ms/trial)
    def design_chunk(args):
        sub_seed, sub_trials = args
        rng = np.random.default_rng(sub_seed)
        u_err = np.zeros(N, dtype=np.int64)
        v_err = np.zeros(N, dtype=np.int64)
        for _ in range(sub_trials):
            u = rng.integers(0, 2, size=N); v = rng.integers(0, 2, size=N)
            x = np.array(polar_encode(u.tolist()), dtype=np.int64)
            y = np.array(polar_encode(v.tolist()), dtype=np.int64)
            z = maagn.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
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
        return u_err, v_err, sub_trials

    # Sequential (chunks fail picklability of local funcs); just run direct
    log(f"  design starting (10000 trials, sequential)")
    t0 = time.time()
    u_err, v_err, n_done = design_chunk((300001, 10000))
    td = time.time() - t0
    Pe_u = u_err / n_done; Pe_v = v_err / n_done
    order_u = _argsort_with_polar_tiebreak(Pe_u)
    order_v = _argsort_with_polar_tiebreak(Pe_v)
    Au = [int(i)+1 for i in sorted(order_u[:ku].tolist())]
    Av = [int(i)+1 for i in sorted(order_v[:kv].tolist())]
    log(f"  design {n_done} in {td:.1f}s")

    # Eval with parallel chunks
    n_cw = 5000
    fu = {p:0 for p in range(1,N+1) if p not in set(Au)}
    fv = {p:0 for p in range(1,N+1) if p not in set(Av)}
    chunk = max(1, n_cw // WORKERS)
    jobs, started = [], 0
    for w in range(WORKERS):
        nt = chunk if w < WORKERS-1 else (n_cw - started)
        jobs.append((alpha, sigma2_eff, sigma2_stat, h_eff,
                     fu, fv, Au, Av, 310000+w, nt))
        started += nt
    t0 = time.time()
    with Pool(WORKERS) as p:
        results = p.map(_maagn_n1024_eval_chunk, jobs)
    errs = sum(r[0] for r in results); tc = sum(r[1] for r in results)
    te = time.time() - t0
    bler = errs / tc
    log(f"  eval {tc} CW: BLER={bler:.6f} ({errs}/{tc}), {te:.1f}s")
    out = {"phase":"L4","decoder":"MA-AGN whitened-SCT N=1024",
           "alpha":alpha,"sigma2_eff":sigma2_eff,"h_equiv":h_eff,
           "N":N,"ku":ku,"kv":kv,
           "n_design":n_done,"design_time_s":td,"Au":Au,"Av":Av,
           "eval":{"n_cw":tc,"errs":int(errs),"bler":bler,"elapsed_s":te}}
    with open(out_path, "w") as f: json.dump(out, f, indent=2)


def phase_M4_chained_sct_h_sweep(out_path):
    """4-state chained SCT at h=0.1, 0.2, 0.4, 0.5, 0.7 — h-sweep at N=128."""
    log("="*60); log("PHASE M4: 4-state chained SCT h-sweep at N=128")
    if os.path.exists(out_path):
        log(f"  skip (exists)"); return
    from chained_sct_4state import mc_design, eval_at, pick
    import chained_sct_4state as cs4
    N = 128; ku, kv = RATES[N]
    out = {"phase":"M4","decoder":"4-state chained SCT h-sweep N=128",
           "channel":"ISI-MAC","snr_db":SNR_DB,"N":N,"results":{}}
    for h_val in [0.1, 0.2, 0.4, 0.5, 0.7]:
        log(f"  h={h_val}")
        cs4.H_TAP = h_val
        t0 = time.time()
        Pe_u, Pe_v, n_done = mc_design(N, 20000, WORKERS, base_seed=400000+int(h_val*100))
        td = time.time() - t0
        Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
        t0 = time.time()
        r = eval_at(N, 10000, Au, Av, WORKERS, base_seed=401000+int(h_val*100))
        te = time.time() - t0
        log(f"    BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']})  design {td:.0f}s eval {te:.0f}s")
        out["results"][str(h_val)] = {"h":h_val,
                                       "N":N,"ku":ku,"kv":kv,
                                       "n_design":n_done,"Au":Au,"Av":Av,"eval":r}
        with open(out_path, "w") as f: json.dump(out, f, indent=2)
    cs4.H_TAP = 0.3


def main():
    log("################################################")
    log("OVERNIGHT v4 starting")
    log("################################################")
    t_all = time.time()
    try: phase_K4_snr_sweep_n512(os.path.join(_HERE, "overnight4_K4_snr_n512.json"))
    except Exception as e: log(f"PHASE K4 FAILED: {e}")
    try: phase_M4_chained_sct_h_sweep(os.path.join(_HERE, "overnight4_M4_h_sweep.json"))
    except Exception as e: log(f"PHASE M4 FAILED: {e}")
    try: phase_L4_maagn_n1024(os.path.join(_HERE, "overnight4_L4_maagn_n1024.json"))
    except Exception as e: log(f"PHASE L4 FAILED: {e}")
    log(f"ALL DONE in {(time.time()-t_all)/3600:.2f} hr")


if __name__ == "__main__":
    main()
