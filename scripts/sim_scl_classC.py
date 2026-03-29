"""
sim_scl_classC.py — SCL L=4 for Class C GMAC at SNR=6dB.

Class C uses extreme path 0^N 1^N, so GA design is valid.
Fills gap: SC and SCL32 exist but no SCL L=4.
"""
import os, sys, time, json, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.channels import GaussianMAC
from polar.encoder import polar_encode
from polar.design import design_gmac, make_path
from polar.decoder_scl import decode_single_list


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    snr_db = 6.0
    sigma2 = 10.0 ** (-snr_db / 10.0)
    channel = GaussianMAC(sigma2=sigma2)
    L = 4
    seed = 42

    # Two rate points matching existing SC/SCL32 results
    rate_configs = [
        ("Ru23_Rv46", 0.23, 0.46),
        ("Ru33_Rv64", 0.33, 0.64),
    ]

    N_values = [32, 64, 128, 256, 512]
    cw_budget = {32: 5000, 64: 5000, 128: 3000, 256: 2000, 512: 500}

    for tag, ru_frac, rv_frac in rate_configs:
        log(f"\n{'='*60}")
        log(f"Class C, {tag}, SCL L={L}, SNR={snr_db}dB")
        log(f"{'='*60}")

        results = {}
        for N in N_values:
            n = int(np.log2(N))
            ku = max(1, round(ru_frac * N))
            kv = max(1, round(rv_frac * N))
            n_cw = cw_budget[N]
            pi = N  # Class C: extreme path 0^N 1^N
            b = make_path(N, pi)

            Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, sigma2)

            log(f"  N={N} ku={ku} kv={kv} L={L} n_cw={n_cw}")

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
                u_idx = [p-1 for p in Au]; v_idx = [p-1 for p in Av]
                if np.any(u_dec[u_idx] != info_u) or np.any(v_dec[v_idx] != info_v):
                    block_errors += 1

                if (i+1) % 500 == 0:
                    log(f"    {i+1}/{n_cw} BLER={block_errors/(i+1):.4f}")

            elapsed = time.time() - t0
            bler = block_errors / n_cw
            results[N] = {"N": N, "ku": ku, "kv": kv, "L": L,
                          "bler": bler, "block_errors": block_errors,
                          "n_codewords": n_cw, "time_s": round(elapsed, 1)}
            log(f"    BLER={bler:.4f} ({block_errors}/{n_cw}) {elapsed:.0f}s")

            # Checkpoint
            out = {"description": f"GMAC 6dB Class C SCL L={L} {tag}",
                   "results": results,
                   "timestamp": datetime.now().isoformat()}
            out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                                    "gmac_snr6dB", f"scl_L4_classC_{tag}.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

        log(f"\n  SUMMARY ({tag}):")
        for N, r in sorted(results.items()):
            log(f"    N={N:>5} BLER={r['bler']:.4f} ({r['block_errors']}/{r['n_codewords']})")


if __name__ == "__main__":
    main()
