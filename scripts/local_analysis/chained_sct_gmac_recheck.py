"""Re-eval chained-SCT with GMAC design at high CW; compare info-sets with own design."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from chained_sct_mc_design import _eval_chunk, RATES, SIGMA2, H_TAP

WORKERS = 7
N_CW = 10000

def eval_for(Au, Av, N, base_seed, label):
    fu = {p: 0 for p in range(1, N + 1) if p not in set(Au)}
    fv = {p: 0 for p in range(1, N + 1) if p not in set(Av)}
    chunk = max(1, N_CW // WORKERS)
    jobs = []
    started = 0
    for w in range(WORKERS):
        nt = chunk if w < WORKERS - 1 else (N_CW - started)
        jobs.append((N, SIGMA2, H_TAP, fu, fv, Au, Av, base_seed + w * 7777, nt))
        started += nt
    t0 = time.time()
    with Pool(WORKERS) as pool:
        results = pool.map(_eval_chunk, jobs)
    errs = sum(r[0] for r in results)
    u_errs = sum(r[1] for r in results)
    v_errs = sum(r[2] for r in results)
    total = sum(r[3] for r in results)
    return dict(label=label, bler_chained=errs/total, errs=errs,
                bler_u=u_errs/total, errs_u=u_errs,
                bler_v=v_errs/total, errs_v=v_errs,
                n_cw=total, elapsed_s=time.time()-t0)

def main():
    # Load own-design results (already produced) to compare Au/Av
    own = json.load(open(os.path.join(_HERE, "chained_sct_owndesign.json")))

    out = {}
    for N in [128, 256]:
        n = int(np.log2(N))
        ku, kv = RATES[N]
        # GMAC design
        gmac_path = os.path.join(_ROOT, "designs", f"gmac_C_n{n}_snr6dB.npz")
        Au_g, Av_g, fu_g, fv_g, _, _, _ = design_from_file(gmac_path, n, ku, kv)
        # Own design (1-indexed)
        Au_o = own["results"][str(N)]["Au"]
        Av_o = own["results"][str(N)]["Av"]

        # Set diffs
        Au_g_s, Au_o_s = set(int(x) for x in Au_g), set(int(x) for x in Au_o)
        Av_g_s, Av_o_s = set(int(x) for x in Av_g), set(int(x) for x in Av_o)

        # Eval GMAC design with the SAME pipeline at 10K CW
        r_gmac = eval_for(list(Au_g), list(Av_g), N, base_seed=33*N + 1, label="GMAC")

        out[str(N)] = {
            "N": N, "ku": ku, "kv": kv,
            "info_set_diff_U": {
                "common": len(Au_g_s & Au_o_s),
                "only_in_gmac": sorted(Au_g_s - Au_o_s),
                "only_in_own":  sorted(Au_o_s - Au_g_s),
            },
            "info_set_diff_V": {
                "common": len(Av_g_s & Av_o_s),
                "only_in_gmac": sorted(Av_g_s - Av_o_s),
                "only_in_own":  sorted(Av_o_s - Av_g_s),
            },
            "gmac_design_10K": r_gmac,
            "own_design_10K": own["results"][str(N)]["eval"],
        }
        print(f"N={N}: GMAC 10K BLER = {r_gmac['bler_chained']:.4f} ({r_gmac['errs']}/{r_gmac['n_cw']})", flush=True)
        print(f"  own 10K BLER       = {own['results'][str(N)]['eval']['bler_chained']:.4f}", flush=True)
        print(f"  U info-set: common={out[str(N)]['info_set_diff_U']['common']}/{ku}, "
              f"GMAC-only={len(out[str(N)]['info_set_diff_U']['only_in_gmac'])}, "
              f"own-only={len(out[str(N)]['info_set_diff_U']['only_in_own'])}", flush=True)
        print(f"  V info-set: common={out[str(N)]['info_set_diff_V']['common']}/{kv}, "
              f"GMAC-only={len(out[str(N)]['info_set_diff_V']['only_in_gmac'])}, "
              f"own-only={len(out[str(N)]['info_set_diff_V']['only_in_own'])}", flush=True)

    with open(os.path.join(_HERE, "chained_sct_recheck.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved chained_sct_recheck.json", flush=True)

if __name__ == "__main__":
    main()
