"""Path search for equal-rate at N=64, ISI r=1 h=0.3.

Replicates equal_rate_path_search.py for N=64. ~6x slower per trial than
N=32 (O(N^2) genie SC), so use 1500 trials per candidate with fine
sweep around N/2 = 32.
"""
import sys, os, json, time
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np

from polar.channels_memory import ISIMAC
from polar.design_mc import mc_design

SIGMA2 = 10**(-0.6); H = 0.3
ch = ISIMAC(sigma2=SIGMA2, h=H)

N = 64
n = int(np.log2(N))
MC_TRIALS = 1500
SEED_BASE = 9000

# coarse + fine around N/2; expected a* ~ N/2 + 6 based on N=32 pattern (a*=19, N/2=16)
CANDIDATES = sorted(set(
    [0, N // 4, N]
    + list(range(max(0, N // 2 - 2), min(N, N // 2 + 12) + 1, 2))   # 30,32,...,44
))

OUT = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis/equal_rate_search_n64.json"


def h2(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mi_from_perr(p):
    return 1.0 - h2(p)


def main():
    print(f"=== Equal-rate search ISI r=1 N={N}, h={H}, SNR=6 dB ===")
    print(f"Candidates a: {CANDIDATES}")
    print(f"MC trials per candidate: {MC_TRIALS}\n")

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    results.setdefault("N", N); results.setdefault("mc_trials", MC_TRIALS)
    results.setdefault("candidates", {})

    for a in CANDIDATES:
        key = str(a)
        if key in results["candidates"] and results["candidates"][key].get("done"):
            d = results["candidates"][key]
            print(f"  a={a:>3}  (cached)  R_U={d['R_U']:.3f}  R_V={d['R_V']:.3f}  "
                  f"|dR|={abs(d['R_U']-d['R_V']):.3f}", flush=True)
            continue
        t0 = time.time()
        peU, peV, _, _ = mc_design(n, ch, mc_trials=MC_TRIALS,
                                    seed=SEED_BASE + a, verbose=False,
                                    path_i=a)
        R_U = float(mi_from_perr(peU).sum())
        R_V = float(mi_from_perr(peV).sum())
        elapsed = time.time() - t0
        print(f"  a={a:>3}  R_U={R_U:.3f}  R_V={R_V:.3f}  "
              f"|dR|={abs(R_U-R_V):.3f}  R_tot={R_U+R_V:.3f}  ({elapsed:.0f}s)",
              flush=True)
        results["candidates"][key] = dict(a=a, R_U=R_U, R_V=R_V,
                                          R_tot=R_U + R_V,
                                          abs_diff=abs(R_U - R_V),
                                          peU=peU.tolist(), peV=peV.tolist(),
                                          done=True)
        json.dump(results, open(OUT, "w"), indent=2)

    cand = results["candidates"]
    items = sorted([(int(k), v) for k, v in cand.items()], key=lambda x: x[0])
    a_star = min(items, key=lambda x: x[1]["abs_diff"])
    print(f"\nOPTIMAL EQUAL-RATE a* = {a_star[0]}  "
          f"(R_U={a_star[1]['R_U']:.3f}, R_V={a_star[1]['R_V']:.3f}, "
          f"R_tot={a_star[1]['R_tot']:.3f})")
    results["a_star"] = a_star[0]; results["done"] = True
    json.dump(results, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
