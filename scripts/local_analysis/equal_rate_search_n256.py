"""Path search at N=256, ISI r=1 h=0.3 — to enable N=256 NCG and SCT eval.

Per-candidate cost: ~120s (vs 25s at N=128, scaling O(N^2.5) roughly).
Strategy: only search near predicted a*(256) = 152 (from 0.594*N rule).
"""
import sys, os, json, time
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np

from polar.channels_memory import ISIMAC
from polar.design_mc import mc_design

SIGMA2 = 10**(-0.6); H = 0.3
ch = ISIMAC(sigma2=SIGMA2, h=H)
N = 256; n = 8
MC_TRIALS = 600
SEED_BASE = 30000
# fine sweep around predicted a* = 152
CANDIDATES = sorted(set([0, N // 4, N] + list(range(146, 162, 2))))

OUT = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis/equal_rate_search_n256.json"


def h2(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main():
    print(f"=== Path search N={N}, ISI h={H} ===")
    print(f"Candidates: {CANDIDATES}\n")
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    results.setdefault("candidates", {})
    for a in CANDIDATES:
        key = str(a)
        if key in results['candidates'] and results['candidates'][key].get('done'):
            d = results['candidates'][key]
            print(f"  a={a:>4}  cached  R_U={d['R_U']:.2f} R_V={d['R_V']:.2f} "
                  f"|dR|={abs(d['R_U']-d['R_V']):.2f}")
            continue
        t0 = time.time()
        peU, peV, _, _ = mc_design(n, ch, mc_trials=MC_TRIALS,
                                    seed=SEED_BASE + a, verbose=False, path_i=a)
        R_U = float((1 - h2(peU)).sum())
        R_V = float((1 - h2(peV)).sum())
        print(f"  a={a:>4}  R_U={R_U:.2f} R_V={R_V:.2f} |dR|={abs(R_U-R_V):.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        results['candidates'][key] = dict(a=a, R_U=R_U, R_V=R_V,
                                          abs_diff=abs(R_U-R_V),
                                          peU=peU.tolist(), peV=peV.tolist(),
                                          done=True)
        json.dump(results, open(OUT, "w"), indent=2)

    items = sorted([(int(k), v) for k, v in results['candidates'].items()],
                   key=lambda x: x[0])
    a_star, dstar = min(items, key=lambda x: x[1]['abs_diff'])
    print(f"\nOPTIMAL a*(256) = {a_star}  (predicted 0.594*256 = 152)")
    print(f"  R_U={dstar['R_U']:.3f}  R_V={dstar['R_V']:.3f}  R_tot={dstar['R_U']+dstar['R_V']:.3f}")
    results['a_star'] = a_star
    json.dump(results, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
