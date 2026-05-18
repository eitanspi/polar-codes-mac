"""Equal-rate path search for ISI-MAC.

For each candidate `a` in the staircase family b = 0^a 1^N 0^(N-a):
  - Run MC genie SC to estimate per-channel P_e
  - Convert to per-channel MI bound: I_i = 1 - h2(P_e_i)
  - R_U(a) = sum of I over U-steps; R_V(a) = sum over V-steps
Find a* minimizing |R_U(a) - R_V(a)|.

Sanity check: R_U(a) + R_V(a) should be ~constant across a (invariant
sum-capacity I(X^N,Y^N;Z^N)).
"""
import sys, os, json, time
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np

from polar.channels_memory import ISIMAC
from polar.design_mc import mc_design

SIGMA2 = 10**(-0.6)
H = 0.3
ch = ISIMAC(sigma2=SIGMA2, h=H)

N = 32                # small N for tractable CPU genie-SC
n = int(np.log2(N))
MC_TRIALS = 4000      # per candidate (~20s each at N=32)
SEED_BASE = 7000

# coarse for overview + fine 1-step sweep across [N/2 - 4 .. N/2 + 4]
CANDIDATES = sorted(set(
    [0, N // 4, N]                                          # endpoints + sanity
    + list(range(max(0, N // 2 - 4), min(N, N // 2 + 4) + 1))  # fine around N/2
))

OUT = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis/equal_rate_search_results.json"


def h2(p, eps=1e-12):
    """Binary entropy in bits, clipped to [eps, 1-eps]."""
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mi_from_perr(p_err):
    """Each synthesized channel is a binary-input channel; lower bound MI by
    1 - h2(P_e). This is exact for BSC; for general binary-input channels it
    is a Fano-type lower bound that is tight when the channel is close to
    symmetric, which is what polar synthesis tends toward."""
    return 1.0 - h2(p_err)


def main():
    print(f"=== Equal-rate path search: ISI-MAC h={H}, N={N}, "
          f"sigma2={SIGMA2:.4f} ===", flush=True)
    print(f"Candidates a: {CANDIDATES}", flush=True)
    print(f"MC trials per candidate: {MC_TRIALS}", flush=True)
    print()

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    results.setdefault("channel", dict(name="ISIMAC", h=H, sigma2=SIGMA2))
    results.setdefault("N", N)
    results.setdefault("mc_trials", MC_TRIALS)
    results.setdefault("candidates", {})

    for a in CANDIDATES:
        key = str(a)
        if key in results["candidates"] and results["candidates"][key].get("done"):
            d = results["candidates"][key]
            print(f"  a={a:>3}  (cached)  R_U={d['R_U']:.3f}  R_V={d['R_V']:.3f}"
                  f"  |dR|={abs(d['R_U']-d['R_V']):.3f}  R_tot={d['R_tot']:.3f}",
                  flush=True)
            continue

        t0 = time.time()
        peU, peV, _, _ = mc_design(n, ch, mc_trials=MC_TRIALS,
                                    seed=SEED_BASE + a, verbose=False,
                                    path_i=a)
        miU = mi_from_perr(peU); miV = mi_from_perr(peV)
        R_U = float(miU.sum()); R_V = float(miV.sum())
        R_tot = R_U + R_V
        elapsed = time.time() - t0
        print(f"  a={a:>3}  R_U={R_U:.3f}  R_V={R_V:.3f}  "
              f"|dR|={abs(R_U - R_V):.3f}  R_tot={R_tot:.3f}  ({elapsed:.1f}s)",
              flush=True)

        results["candidates"][key] = dict(
            a=a, R_U=R_U, R_V=R_V, R_tot=R_tot,
            abs_diff=abs(R_U - R_V),
            peU=peU.tolist(), peV=peV.tolist(),
            elapsed=elapsed, done=True,
        )
        json.dump(results, open(OUT, "w"), indent=2)

    # ── find a* ──────────────────────────────────────────────────────────
    cand = results["candidates"]
    items = sorted([(int(k), v) for k, v in cand.items()], key=lambda x: x[0])
    a_star = min(items, key=lambda x: x[1]["abs_diff"])
    print()
    print("=== Summary ===", flush=True)
    print(f"{'a':>4} {'R_U':>7} {'R_V':>7} {'|dR|':>7} {'R_tot':>7}")
    for a, d in items:
        marker = "  <-- a*" if a == a_star[0] else ""
        print(f"{a:>4} {d['R_U']:>7.3f} {d['R_V']:>7.3f} "
              f"{d['abs_diff']:>7.3f} {d['R_tot']:>7.3f}{marker}")
    print()
    print(f"OPTIMAL EQUAL-RATE a* = {a_star[0]}  "
          f"(R_U={a_star[1]['R_U']:.3f}, R_V={a_star[1]['R_V']:.3f}, "
          f"R_tot={a_star[1]['R_tot']:.3f})")
    print(f"  per-user rate at a*: R = {a_star[1]['R_U']/N:.4f}, "
          f"{a_star[1]['R_V']/N:.4f}  (info bits / channel use)")

    # symmetry check
    R_tots = [d["R_tot"] for _, d in items]
    print(f"\nR_tot across a: min={min(R_tots):.3f}, max={max(R_tots):.3f}, "
          f"spread={max(R_tots)-min(R_tots):.3f} bits "
          f"(should be ~0 modulo MC noise — invariant sum-capacity)")

    results["a_star"] = a_star[0]
    results["done"] = True
    json.dump(results, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
