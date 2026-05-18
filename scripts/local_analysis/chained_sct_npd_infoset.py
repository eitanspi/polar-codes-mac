"""Run chained SCT eval using the NPD's MI-designed info set.

If chained-SCT-with-NPD-info-set ≈ chained-SCT-with-own-design, the gap to NPD
is fundamentally in the decoder (chained's iid-Y assumption). If chained-SCT
gets much worse with NPD's info set, that means NPD's "low-index" choices
require Y-polar-structure that chained SCT can't exploit.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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
    d_isi = json.load(open(os.path.join(_ROOT, "class_c_npd/results/isi_campaign/results.json")))
    d_jt = json.load(open(os.path.join(_ROOT, "class_c_npd/results/joint_trellis_batched/results.json")))
    own = json.load(open(os.path.join(_HERE, "chained_sct_owndesign.json")))

    out = {}
    for N in [128, 256]:
        ku, kv = RATES[N]
        Au_npd = sorted(int(p) for p in d_isi[str(N)]['Au_mi'])
        Av_npd = sorted(int(p) for p in d_isi[str(N)]['Av_mi'])
        Au_jt  = sorted(int(p) for p in d_jt[str(N)]['Au_jt'])
        Av_jt  = sorted(int(p) for p in d_jt[str(N)]['Av_jt'])
        Au_own = sorted(own['results'][str(N)]['Au'])
        Av_own = sorted(own['results'][str(N)]['Av'])

        print(f"=== N={N} ===", flush=True)
        # Chained SCT with NPD info set
        r_npd_infoset = eval_for(Au_npd, Av_npd, N, base_seed=77*N + 1,
                                  label="chained SCT, NPD info set")
        # Chained SCT with joint-trellis SCT info set
        r_jt_infoset  = eval_for(Au_jt, Av_jt, N, base_seed=77*N + 2,
                                  label="chained SCT, JT info set")
        # For reference: chained SCT with own info set (already have it)
        r_own = own['results'][str(N)]['eval']

        out[str(N)] = {
            "chained_sct_with_NPD_infoset": r_npd_infoset,
            "chained_sct_with_JT_infoset":  r_jt_infoset,
            "chained_sct_with_own_infoset": r_own,
            "NPD_bler":  d_isi[str(N)]['npd_mi']['bler_chained'],
            "JT_bler":   d_jt[str(N)]['jt_eval']['bler_chained'],
        }
        print(f"  chained SCT + NPD info set:  {r_npd_infoset['bler_chained']:.5f} "
              f"({r_npd_infoset['errs']}/{r_npd_infoset['n_cw']})", flush=True)
        print(f"  chained SCT + JT info set :  {r_jt_infoset['bler_chained']:.5f} "
              f"({r_jt_infoset['errs']}/{r_jt_infoset['n_cw']})", flush=True)
        print(f"  chained SCT + own info set:  {r_own['bler_chained']:.5f} "
              f"({r_own['errs_chained']}/{r_own['n_cw']})", flush=True)
        print(f"  (ref) NPD with NPD info set: {d_isi[str(N)]['npd_mi']['bler_chained']:.5f}", flush=True)
        print(f"  (ref) Joint SCT  + JT info set: {d_jt[str(N)]['jt_eval']['bler_chained']:.5f}", flush=True)

    with open(os.path.join(_HERE, "chained_sct_infoset_swap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved chained_sct_infoset_swap.json", flush=True)

if __name__ == "__main__":
    main()
