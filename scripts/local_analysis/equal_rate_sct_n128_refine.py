"""Re-do N=128 a=76 MC design with more trials, then SCT eval."""
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
import sys as _sys
N = 128; n = 7; A_STAR = 76
K_U = int(_sys.argv[1]) if len(_sys.argv) > 1 else 36
K_V = int(_sys.argv[2]) if len(_sys.argv) > 2 else 36
MC_TRIALS = 4000
SEED = 21000
N_EVAL = 2500

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
RES_FILE = os.path.join(OUT_DIR, f"equal_rate_validate_n128_refined_k{K_U}.json")


def select_info_set(p_err, k):
    sorted_p = sorted([(p, i + 1) for i, p in enumerate(p_err)])
    info = sorted([pos for _, pos in sorted_p[:k]])
    frozen = {i: 0 for i in range(1, len(p_err) + 1) if i not in info}
    return info, frozen


def h2(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main():
    print(f"=== N=128 a={A_STAR} refined design ({MC_TRIALS} MC trials) ===")
    t0 = time.time()
    peU, peV, _, _ = mc_design(n, ch, mc_trials=MC_TRIALS,
                                seed=SEED, verbose=False, path_i=A_STAR)
    R_U = float((1 - h2(peU)).sum()); R_V = float((1 - h2(peV)).sum())
    print(f"  design: R_U={R_U:.3f} R_V={R_V:.3f} R_tot={R_U+R_V:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    Au, fu = select_info_set(peU, K_U)
    Av, fv = select_info_set(peV, K_V)
    print(f"  worst Pe in A_U = {max(peU[a-1] for a in Au):.5f}")
    print(f"  worst Pe in A_V = {max(peV[a-1] for a in Av):.5f}")
    b = make_path(N, A_STAR)

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
            print(f"  {cw+1}/{N_EVAL} errs={errs} BLER~{errs/(cw+1):.4f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)
    bler = errs / N_EVAL
    print(f"\nSCT BLER = {bler:.4f} ({errs}/{N_EVAL})")
    json.dump(dict(N=N, a_star=A_STAR, k_u=K_U, k_v=K_V,
                   mc_trials=MC_TRIALS,
                   peU=peU.tolist(), peV=peV.tolist(),
                   bler=bler, errs=int(errs), n_cw=int(N_EVAL)),
              open(RES_FILE, 'w'), indent=2)
    print(f"saved {RES_FILE}")


if __name__ == "__main__":
    main()
