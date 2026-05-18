"""Corner-rate SCT BLER for ISI-MAC at N=64, 128, 256 — control for the
staircase finite-N anomaly seen at equal-rate.

Path: a=N (Class C: all U first, then all V). Designs from MC at corner path.
Same per-user rate budget as the existing chained-NPD campaigns.
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
OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
RES_FILE = os.path.join(OUT_DIR, "corner_rate_sct_isi.json")

CONFIGS = [
    # (N, ku, kv, n_cw)
    (32,    7,  15, 5000),
    (64,   15,  29, 3000),
    (128,  30,  58, 2000),
    (256,  59, 117, 1000),
]
MC_TRIALS = 3000


def select_info_set(p_err, k):
    sorted_p = sorted([(p, i + 1) for i, p in enumerate(p_err)])
    info = sorted([pos for _, pos in sorted_p[:k]])
    frozen = {i: 0 for i in range(1, len(p_err) + 1) if i not in info}
    return info, frozen


def main():
    print("=== Corner-rate SCT BLER (path a=N) — ISI-MAC h=0.3, SNR=6 dB ===\n")
    results = json.load(open(RES_FILE)) if os.path.exists(RES_FILE) else {}
    for (N, ku, kv, ncw) in CONFIGS:
        key = str(N)
        if key in results and results[key].get('done'):
            d = results[key]
            print(f"N={N} cached: BLER = {d['bler']:.4f} ({d['errs']}/{d['n_cw']})")
            continue
        n = int(np.log2(N))
        b = make_path(N, N)            # corner path
        print(f"N={N} ku={ku} kv={kv} — designing...")
        t0 = time.time()
        peU, peV, _, _ = mc_design(n, ch, mc_trials=MC_TRIALS,
                                    seed=31000 + N, verbose=False, path_i=N)
        Au, fu = select_info_set(peU, ku)
        Av, fv = select_info_set(peV, kv)
        print(f"  design {time.time()-t0:.0f}s  "
              f"max Pe U={max(peU[a-1] for a in Au):.4f}  "
              f"max Pe V={max(peV[a-1] for a in Av):.4f}")

        print(f"  SCT eval ({ncw} CW)...")
        rng = np.random.default_rng(311 + N)
        errs = 0; teval0 = time.time()
        for cw in range(ncw):
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
            if (cw + 1) % max(1, ncw // 5) == 0:
                print(f"    {cw+1}/{ncw} errs={errs} "
                      f"BLER~{errs/(cw+1):.4f} "
                      f"({(time.time()-teval0)/60:.1f}min)", flush=True)
        bler = errs / ncw
        print(f"  N={N} corner-rate SCT BLER = {bler:.4f} ({errs}/{ncw})\n")
        results[key] = dict(N=N, ku=ku, kv=kv, path='corner',
                            bler=bler, errs=int(errs), n_cw=int(ncw),
                            max_pe_u=float(max(peU[a-1] for a in Au)),
                            max_pe_v=float(max(peV[a-1] for a in Av)),
                            done=True)
        json.dump(results, open(RES_FILE, "w"), indent=2)
    print("=== ALL DONE ===")
    print(f"{'N':>4} {'rate_U':>7} {'rate_V':>7} {'BLER':>8}")
    for k in sorted(results.keys(), key=int):
        d = results[k]
        print(f"{d['N']:>4} {d['ku']/d['N']:>7.4f} {d['kv']/d['N']:>7.4f}  {d['bler']:.4f}")


if __name__ == "__main__":
    main()
