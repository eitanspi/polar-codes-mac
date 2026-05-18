"""SCT-only equal-rate eval at N=128 (no NCG training — too slow on CPU).

Steps:
  1. Path search at N=128 (fewer trials, finer sweep around expected a*=76)
  2. SCT BLER at a*(128) with k_U=k_V=36 (per-user rate 0.281, matches N=32/64)
"""
import sys, os, time, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np

from polar.channels_memory import ISIMAC
from polar.design_mc import mc_design
from polar.encoder import polar_encode_batch
from polar.design import make_path
from polar.decoder_trellis import decode_single

SIGMA2 = 10**(-0.6); H = 0.3
ch = ISIMAC(sigma2=SIGMA2, h=H)
N = 128; n = 7
K_U = 36; K_V = 36

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
RES_FILE = os.path.join(OUT_DIR, "equal_rate_validate_n128.json")
SEARCH_FILE = os.path.join(OUT_DIR, "equal_rate_search_n128.json")

MC_TRIALS = 800
SEED_BASE = 11000
# expected a* ~ N*0.594 = 76 from N=32 (a*=19), N=64 (a*=38) pattern
CANDIDATES = sorted(set([0, N // 4, N] + list(range(70, 84, 2))))

N_EVAL = 2000


def h2(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mi(p): return 1.0 - h2(p)


def select_info_set(p_err, k):
    sorted_p = sorted([(p, i + 1) for i, p in enumerate(p_err)])
    info = sorted([pos for _, pos in sorted_p[:k]])
    frozen = {i: 0 for i in range(1, len(p_err) + 1) if i not in info}
    return info, frozen


def main():
    print(f"=== Equal-rate SCT-only N=128 ===")
    # ── path search ────────────────────────────────────────────────────
    search = json.load(open(SEARCH_FILE)) if os.path.exists(SEARCH_FILE) else {}
    search.setdefault("candidates", {})
    print(f"Path search: candidates {CANDIDATES}  ({MC_TRIALS} trials each)")
    for a in CANDIDATES:
        if str(a) in search["candidates"] and search["candidates"][str(a)].get("done"):
            d = search["candidates"][str(a)]
            print(f"  a={a:>3}  cached  R_U={d['R_U']:.2f} R_V={d['R_V']:.2f} "
                  f"|dR|={abs(d['R_U']-d['R_V']):.2f}")
            continue
        t0 = time.time()
        peU, peV, _, _ = mc_design(n, ch, mc_trials=MC_TRIALS,
                                    seed=SEED_BASE + a, verbose=False, path_i=a)
        R_U = float(mi(peU).sum()); R_V = float(mi(peV).sum())
        print(f"  a={a:>3}  R_U={R_U:.2f} R_V={R_V:.2f} |dR|={abs(R_U-R_V):.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        search["candidates"][str(a)] = dict(a=a, R_U=R_U, R_V=R_V,
                                            abs_diff=abs(R_U-R_V),
                                            peU=peU.tolist(), peV=peV.tolist(),
                                            done=True)
        json.dump(search, open(SEARCH_FILE, "w"), indent=2)

    items = sorted([(int(k), v) for k, v in search['candidates'].items()],
                   key=lambda x: x[0])
    a_star, dstar = min(items, key=lambda x: x[1]['abs_diff'])
    print(f"\nOPTIMAL a*({N}) = {a_star}  (R_U={dstar['R_U']:.2f}, R_V={dstar['R_V']:.2f})")

    # ── SCT eval at a* ────────────────────────────────────────────────
    peU = np.array(dstar['peU']); peV = np.array(dstar['peV'])
    Au, fu = select_info_set(peU, K_U)
    Av, fv = select_info_set(peV, K_V)
    print(f"max Pe in A_U = {max(peU[a-1] for a in Au):.4f}")
    print(f"max Pe in A_V = {max(peV[a-1] for a in Av):.4f}")
    b = make_path(N, a_star)

    results = json.load(open(RES_FILE)) if os.path.exists(RES_FILE) else {}
    if 'sct' not in results or not results['sct'].get('done'):
        print(f"\n--- SCT eval ({N_EVAL} CW) ---")
        rng = np.random.default_rng(11)
        errs = 0; t0 = time.time()
        for cw in range(N_EVAL):
            u = np.zeros(N, dtype=np.int32); v = np.zeros(N, dtype=np.int32)
            for p in Au: u[p-1] = rng.integers(0, 2)
            for p in Av: v[p-1] = rng.integers(0, 2)
            x = polar_encode_batch(u.reshape(1,-1))[0]
            y = polar_encode_batch(v.reshape(1,-1))[0]
            z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
            u_hat, v_hat = decode_single(N, list(z), b, fu, fv, ch)
            ue = any(int(u_hat[p-1]) != int(u[p-1]) for p in Au)
            ve = any(int(v_hat[p-1]) != int(v[p-1]) for p in Av)
            if ue or ve: errs += 1
            if (cw + 1) % max(1, N_EVAL // 10) == 0:
                print(f"  SCT {cw+1}/{N_EVAL} errs={errs} "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)
        bler = errs / N_EVAL
        print(f"\nSCT BLER = {bler:.4f} ({errs}/{N_EVAL})")
        results['sct'] = dict(bler=bler, errs=int(errs), n_cw=int(N_EVAL),
                              N=N, a_star=a_star, k_u=K_U, k_v=K_V, done=True)
        json.dump(results, open(RES_FILE, "w"), indent=2)


if __name__ == "__main__":
    main()
