"""
sim_scl_large_N.py — SCL L=4 at N=512, 1024 for Class B GMAC.

Extends the SCL L=4 baselines to large N. Uses decode_single (reliable).
"""
import os, sys, time, json, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.channels import GaussianMAC
from polar.encoder import polar_encode
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder_scl import decode_single_list


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    snr_db = 6.0
    sigma2 = 10.0 ** (-snr_db / 10.0)
    channel = GaussianMAC(sigma2=sigma2)
    L = 4
    seed = 42
    designs_dir = os.path.join(os.path.dirname(__file__), "..", "designs")

    configs = [
        (512, 246, 246, 500),
        (1024, 492, 492, 200),
    ]

    results = {}
    log(f"SCL L={L} large N baselines, SNR={snr_db}dB, Class B")

    for N, ku, kv, n_cw in configs:
        n = int(np.log2(N))
        pi = N // 2
        b = make_path(N, pi)

        mc_path = os.path.join(designs_dir, f"gmac_B_n{n}_snr{snr_db:.0f}dB.npz")
        if not os.path.exists(mc_path):
            log(f"  SKIP N={N}: no MC design at {mc_path}")
            continue

        Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
        u_idx = [p - 1 for p in Au]
        v_idx = [p - 1 for p in Av]

        log(f"=== N={N} ku={ku} kv={kv} L={L} n_cw={n_cw} ===")

        rng = np.random.default_rng(seed + n * 100000)
        block_errors = 0
        t0 = time.time()

        for i in range(n_cw):
            info_u = rng.integers(0, 2, ku)
            info_v = rng.integers(0, 2, kv)
            u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
            for j, p in enumerate(Au): u[p-1] = info_u[j]
            for j, p in enumerate(Av): v[p-1] = info_v[j]
            x = np.array(polar_encode(u.tolist()), dtype=float)
            y = np.array(polar_encode(v.tolist()), dtype=float)
            z = channel.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]

            u_dec, v_dec = decode_single_list(N, z.tolist(), b, fu, fv, channel, L=L)
            u_dec = np.array(u_dec); v_dec = np.array(v_dec)
            if np.any(u_dec[u_idx] != info_u) or np.any(v_dec[v_idx] != info_v):
                block_errors += 1

            if (i+1) % 50 == 0:
                elapsed = time.time() - t0
                bler = block_errors / (i+1)
                eta = elapsed / (i+1) * (n_cw - i - 1)
                log(f"  {i+1}/{n_cw} BLER={bler:.4f} ({block_errors} err) "
                    f"ETA={eta/60:.0f}min")

        elapsed = time.time() - t0
        bler = block_errors / n_cw
        results[N] = {"N": N, "ku": ku, "kv": kv, "L": L,
                       "bler": bler, "block_errors": block_errors,
                       "n_codewords": n_cw, "time_s": round(elapsed, 1)}
        log(f"  DONE: BLER={bler:.4f} ({block_errors}/{n_cw}) {elapsed/60:.1f}min")

        out = {"description": "SCL L=4 large N baselines, Class B, SNR=6dB",
               "results": results,
               "timestamp": datetime.now().isoformat()}
        out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                                "gmac_snr6dB", "scl_L4_classB_large_N.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
