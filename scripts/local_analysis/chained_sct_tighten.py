"""Tighten the 4-state chained SCT BLER at N=512 and N=1024.

Run 50K design trials + 30K eval CW per N. Total ~3-4 hr on 7 cores.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
if _HERE not in sys.path: sys.path.insert(0, _HERE)

LOG = "/tmp/chained_sct_tighten/log.txt"
os.makedirs(os.path.dirname(LOG), exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

from chained_sct_4state import mc_design, eval_at, pick, RATES

WORKERS = 7
OUT = os.path.join(_HERE, "chained_sct_tight_large_N.json")

def main():
    out = {}
    if os.path.exists(OUT):
        try: out = json.load(open(OUT))
        except: out = {}

    log("============================================")
    log("4-state chained SCT tightening at N=512, 1024")
    log("50K design trials + 30K eval CW each")
    log("============================================")

    for N in [512, 1024]:
        key = str(N)
        if key in out and out[key].get("done"):
            log(f"  N={N} already done, skip"); continue
        ku, kv = RATES[N]
        log(f"--- N={N} ku={ku} kv={kv} ---")
        # Design 50K trials
        log(f"  design (50K trials)...")
        t0 = time.time()
        Pe_u, Pe_v, n_done = mc_design(N, 50000, WORKERS, base_seed=500001 + N)
        td = time.time() - t0
        log(f"    design done {n_done} in {td/60:.1f}min")
        Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
        # Eval 30K
        log(f"  eval (30K CW)...")
        t0 = time.time()
        r = eval_at(N, 30000, Au, Av, WORKERS, base_seed=500999 + N)
        te = time.time() - t0
        log(f"    BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']}) in {te/60:.1f}min")
        out[key] = {"N":N,"ku":ku,"kv":kv,"n_design":n_done,"design_time_s":td,
                    "Au":Au,"Av":Av,"eval":r,"done":True}
        with open(OUT, "w") as f: json.dump(out, f, indent=2)

    log("ALL DONE")


if __name__ == "__main__":
    main()
