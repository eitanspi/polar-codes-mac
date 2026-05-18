"""Test 1-by-1 interleaved path (0101...) at N=128 equal-rate.

Compares against the staircase result (a*=76, BLER=0.20). If interleaved
gives much lower BLER, the staircase has a structural penalty.
"""
import sys, os, time, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np

from polar.channels_memory import ISIMAC
from polar.design_mc import mc_design
from polar.encoder import polar_encode_batch
from polar.decoder_trellis import decode_single

SIGMA2 = 10**(-0.6); H = 0.3
ch = ISIMAC(sigma2=SIGMA2, h=H)
N = 128; n = 7
K = 36   # same k as staircase test
MC_TRIALS = 4000
N_EVAL = 2500

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
RES_FILE = os.path.join(OUT_DIR, "interleaved_path_test_n128.json")


def select_info_set(p_err, k):
    sorted_p = sorted([(p, i + 1) for i, p in enumerate(p_err)])
    info = sorted([pos for _, pos in sorted_p[:k]])
    frozen = {i: 0 for i in range(1, len(p_err) + 1) if i not in info}
    return info, frozen


def main():
    # 1-by-1 alternation: 010101010101... length 2N
    # equal counts of 0 and 1 -> equal-rate
    b = [0, 1] * N   # length 2N, N zeros, N ones
    assert b.count(0) == N and b.count(1) == N
    print(f"=== N={N} interleaved path (b = 0101...0101) ===")
    print(f"path length = {len(b)}, U-steps = {b.count(0)}, V-steps = {b.count(1)}")

    print(f"Designing ({MC_TRIALS} MC trials)...")
    t0 = time.time()
    peU, peV, _, _ = mc_design(n, ch, mc_trials=MC_TRIALS,
                                seed=42001, verbose=False, b=b)
    print(f"  design {time.time()-t0:.0f}s")
    Au, fu = select_info_set(peU, K)
    Av, fv = select_info_set(peV, K)
    print(f"  worst Pe U = {max(peU[a-1] for a in Au):.5f}")
    print(f"  worst Pe V = {max(peV[a-1] for a in Av):.5f}")

    # MI sanity
    def h2(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(p*np.log2(p) + (1-p)*np.log2(1-p))
    R_U = float((1 - h2(peU)).sum())
    R_V = float((1 - h2(peV)).sum())
    print(f"  MI bound R_U={R_U:.3f} R_V={R_V:.3f} R_tot={R_U+R_V:.3f}")

    print(f"\n--- SCT eval ({N_EVAL} CW) ---")
    rng = np.random.default_rng(11)
    errs = 0; teval0 = time.time()
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
                  f"({(time.time()-teval0)/60:.1f}min)", flush=True)
    bler = errs / N_EVAL
    print(f"\nINTERLEAVED 0101... BLER = {bler:.4f} ({errs}/{N_EVAL})")
    print(f"  compare staircase a*=76 same k=36 BLER = 0.1996")

    json.dump(dict(N=N, path='interleaved_0101', k_u=K, k_v=K,
                   bler=bler, errs=int(errs), n_cw=int(N_EVAL),
                   max_pe_u=float(max(peU[a-1] for a in Au)),
                   max_pe_v=float(max(peV[a-1] for a in Av)),
                   R_U=R_U, R_V=R_V, R_tot=R_U+R_V),
              open(RES_FILE, 'w'), indent=2)
    print(f"saved {RES_FILE}")


if __name__ == "__main__":
    main()
