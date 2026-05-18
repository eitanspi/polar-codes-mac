"""Chained SCT at N=1024 with 200K design trials + 50K CW eval.

Tests whether the gap to Joint MAC SCT at N=1024 is due to design under-sampling.
ETA ~1.5-2 hr on 7 cores.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
if _HERE not in sys.path: sys.path.insert(0, _HERE)

LOG_DIR = "/tmp/chained_sct_n1024_200k"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "log.txt")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

WORKERS = 7
OUT = os.path.join(_HERE, "chained_sct_n1024_200k.json")


def main():
    from chained_sct_4state import mc_design, eval_at, pick, RATES
    out = {}
    if os.path.exists(OUT):
        try: out = json.load(open(OUT))
        except: out = {}

    log("============================================")
    log("Chained SCT N=1024: 200K design + 50K CW eval")
    log("============================================")

    N = 1024
    ku, kv = RATES[N]
    log(f"N={N} ku={ku} kv={kv}")

    if "design_done" not in out:
        log(f"  design (200K trials)...")
        t0 = time.time()
        Pe_u, Pe_v, n_done = mc_design(N, 200000, WORKERS, base_seed=700001)
        td = time.time() - t0
        log(f"    design done {n_done} in {td/60:.1f}min")
        Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
        out["design_done"] = True
        out["n_design"] = n_done
        out["design_time_s"] = td
        out["Au"] = Au
        out["Av"] = Av
        out["Pe_u_top10"] = [float(Pe_u[p-1]) for p in Au[:10]]
        out["Pe_v_top10"] = [float(Pe_v[p-1]) for p in Av[:10]]
        with open(OUT, "w") as f: json.dump(out, f, indent=2)
    else:
        log(f"  design already done ({out['n_design']} trials), skip")
        Au = out["Au"]; Av = out["Av"]

    log(f"  eval (50K CW)...")
    t0 = time.time()
    r = eval_at(N, 50000, Au, Av, WORKERS, base_seed=700999)
    te = time.time() - t0
    log(f"    BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']}) in {te/60:.1f}min")
    out["eval"] = r
    out["done"] = True
    with open(OUT, "w") as f: json.dump(out, f, indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
