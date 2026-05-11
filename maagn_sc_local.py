"""MA-AGN memoryless SC sweep: α ∈ {0.3, 0.5, 0.7, 0.9} × N ∈ {16..1024}.
Runs locally on Mac CPU. Memoryless SC + own BCE-genie MC design."""
import sys, os, json, time, argparse
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np

from polar.channels_memory_new import MAAGNMAC
from polar.encoder import polar_encode_batch
from polar.decoder import _SCNode, _sc_decode_from_llr

OUT = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results_local/maagn_sc_local"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt"); RES = os.path.join(OUT, "results.json")
SIGMA2 = 10**(-0.6)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")

def llr_u(z, sigma2):
    s = sigma2
    def lN(z_, m): return -0.5 * np.log(2*np.pi*s) - 0.5 * (z_ - m)**2 / s
    return (np.logaddexp(lN(z, 2.0), lN(z, 0.0)) + np.log(0.5)) - (np.logaddexp(lN(z, 0.0), lN(z, -2.0)) + np.log(0.5))
def llr_v(z, x_hat, sigma2):
    bx = 1.0 - 2.0 * x_hat; m0 = bx + 1.0; m1 = bx - 1.0; s = sigma2
    return (-0.5 * (z - m0)**2 + 0.5 * (z - m1)**2) / s

def design_sc(N, ku, kv, ch, n_mc, seed):
    rng = np.random.default_rng(seed); bce_u = np.zeros(N); bce_v = np.zeros(N)
    for cw in range(n_mc):
        u = rng.integers(0, 2, N).astype(np.int8); v = rng.integers(0, 2, N).astype(np.int8)
        x = polar_encode_batch(u.reshape(1,-1))[0]; y = polar_encode_batch(v.reshape(1,-1))[0]
        z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
        for stg, llr_fn, true_b in [(0, lambda z,s: llr_u(z, s), u), (1, lambda z,s: llr_v(z, x.astype(np.float64), s), v)]:
            llr = llr_fn(z, ch.sigma2)
            node = _SCNode(np.asarray(llr, dtype=np.float64))
            for i in range(N):
                L = float(node.get_llr(i)); b = int(true_b[i])
                xx = -L if b == 0 else L
                bce_val = xx if xx > 50 else (0.0 if xx < -50 else float(np.log1p(np.exp(xx))))
                if stg == 0: bce_u[i] += bce_val
                else: bce_v[i] += bce_val
                node.feed(i, b)
    Au = sorted([int(i+1) for i in np.argsort(bce_u/n_mc)[:ku]])
    Av = sorted([int(i+1) for i in np.argsort(bce_v/n_mc)[:kv]])
    return Au, Av

def eval_sc(N, ku, kv, Au, Av, ch, n_cw, seed, target_errs=30):
    fu = {p: 0 for p in (set(range(1, N+1)) - set(Au))}
    fv = {p: 0 for p in (set(range(1, N+1)) - set(Av))}
    rng = np.random.default_rng(seed); errs = 0
    t0 = time.time(); done = 0
    while done < n_cw:
        u = np.zeros(N, dtype=np.int8); v = np.zeros(N, dtype=np.int8)
        for p in Au: u[p-1] = rng.integers(0, 2)
        for p in Av: v[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u.reshape(1,-1))[0]; y = polar_encode_batch(v.reshape(1,-1))[0]
        z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
        u_hat = _sc_decode_from_llr(llr_u(z, ch.sigma2), fu)
        x_hat = polar_encode_batch(u_hat.reshape(1,-1))[0]
        v_hat = _sc_decode_from_llr(llr_v(z, x_hat.astype(np.float64), ch.sigma2), fv)
        ue = any(int(u_hat[p-1]) != int(u[p-1]) for p in Au)
        ve = any(int(v_hat[p-1]) != int(v[p-1]) for p in Av)
        if ue or ve: errs += 1
        done += 1
        if done % max(1, n_cw // 20) == 0:
            log(f"      eval {done}/{n_cw} {errs} ({(time.time()-t0)/60:.1f}min)")
    return errs/done, errs, done

# (α, N, ku, kv, n, n_design, n_eval)
JOBS = []
for alpha in [0.3, 0.7, 0.9]:
    JOBS += [
        (alpha,  16,   4,   7, 4,  5_000,  20_000),
        (alpha,  32,   7,  15, 5,  5_000,  20_000),
        (alpha,  64,  15,  29, 6,  5_000,  20_000),
        (alpha, 128,  30,  58, 7,  5_000,  30_000),
        (alpha, 256,  59, 117, 8,  3_000,  50_000),
        (alpha, 512, 119, 233, 9,  2_000, 100_000),
        (alpha,1024, 239, 467,10,  1_000, 150_000),
    ]

def main():
    results = json.load(open(RES)) if os.path.exists(RES) else {}
    for (alpha, N, ku, kv, n, nmc, ncw) in JOBS:
        key = f"a{alpha}_N{N}"
        if key in results and results[key].get("done"):
            log(f"  {key} done, skip"); continue
        log(f"=== α={alpha} N={N} ===")
        ch = MAAGNMAC(sigma2=SIGMA2, alpha=alpha)
        log(f"  design ({nmc} CW)")
        Au, Av = design_sc(N, ku, kv, ch, nmc, seed=70000+N+int(alpha*10))
        log(f"  eval ({ncw} CW)")
        bler, errs, ncw_actual = eval_sc(N, ku, kv, Au, Av, ch, ncw, seed=80000+N+int(alpha*10))
        log(f"  α={alpha} N={N} SC: {errs}/{ncw_actual} BLER={bler:.6f}")
        results[key] = dict(alpha=alpha, N=N, ku=ku, kv=kv, Au=Au, Av=Av,
                           bler=bler, errs=errs, n_cw=ncw_actual, done=True)
        json.dump(results, open(RES, "w"), indent=2)
    log("DONE")

if __name__ == "__main__":
    main()
