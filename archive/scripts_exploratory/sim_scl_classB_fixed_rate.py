"""
sim_scl_classB_fixed_rate.py — SCL L=4 simulation at fixed Ru≈0.48, Rv≈0.48

Fills the gap: existing SCL L=4 results have only 3000 codewords and stop at N=128.
This extends to N=256 with more codewords for reliable BLER estimates.
"""
import os, sys, time, json, numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polar.channels import GaussianMAC
from polar.encoder import polar_encode, polar_encode_batch, build_message, build_message_batch
from polar.design import design_gmac, make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_single
from polar.decoder_scl import decode_single_list


def log(msg=""):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sim_single(N, ku, kv, Au, Av, frozen_u, frozen_v, b, L, n_cw, channel, seed):
    """Run simulation one codeword at a time (decode_single is reliable for GMAC)."""
    rng = np.random.default_rng(seed)
    block_errors = 0
    u_bit_errors = 0
    v_bit_errors = 0

    u_idx = np.array([p - 1 for p in Au])
    v_idx = np.array([p - 1 for p in Av])

    for i in range(n_cw):
        info_u = rng.integers(0, 2, ku)
        info_v = rng.integers(0, 2, kv)

        u = np.zeros(N, dtype=int)
        v = np.zeros(N, dtype=int)
        for j, p in enumerate(Au):
            u[p - 1] = info_u[j]
        for j, p in enumerate(Av):
            v[p - 1] = info_v[j]

        x = np.array(polar_encode(u.tolist()), dtype=float)
        y = np.array(polar_encode(v.tolist()), dtype=float)
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]

        if L == 1:
            u_dec, v_dec = decode_single(N, z.tolist(), b, frozen_u, frozen_v, channel)
        else:
            u_dec, v_dec = decode_single_list(N, z.tolist(), b, frozen_u, frozen_v, channel, L=L)

        u_dec = np.array(u_dec)
        v_dec = np.array(v_dec)
        ue = int(np.sum(u_dec[u_idx] != info_u))
        ve = int(np.sum(v_dec[v_idx] != info_v))
        u_bit_errors += ue
        v_bit_errors += ve
        if ue > 0 or ve > 0:
            block_errors += 1

        if (i + 1) % 500 == 0:
            bler = block_errors / (i + 1)
            log(f"    progress: {i+1}/{n_cw}  BLER={bler:.4f}  ({block_errors} errors)")

    return block_errors, u_bit_errors, v_bit_errors


def main():
    snr_db = 6.0
    sigma2 = 10.0 ** (-snr_db / 10.0)
    channel = GaussianMAC(sigma2=sigma2)
    L = 4
    seed = 42

    # Fixed rate: Ru≈0.48, Rv≈0.48 (same as existing results)
    # Uses MC design (GA design is wrong for Class B interleaved path)
    configs = [
        (32,  15, 15, 5000),
        (64,  31, 31, 5000),
        (128, 62, 62, 5000),
        (256, 123, 123, 2000),
        (512, 246, 246, 1000),
        (1024, 492, 492, 500),
    ]

    results = {}
    total_time = 0

    log(f"SCL L={L} Class B fixed-rate simulation, SNR={snr_db}dB")
    log(f"sigma2={sigma2:.6f}")
    log()

    for N, ku, kv, n_cw in configs:
        n = int(np.log2(N))
        pi = N // 2  # Class B
        b = make_path(N, pi)
        mc_path = os.path.join(os.path.dirname(__file__), "..", "designs",
                               f"gmac_B_n{n}_snr{snr_db:.0f}dB.npz")
        Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)

        Ru = ku / N
        Rv = kv / N
        log(f"=== N={N}  ku={ku} (Ru={Ru:.4f})  kv={kv} (Rv={Rv:.4f})  L={L}  n_cw={n_cw} ===")

        t0 = time.time()
        be, ube, vbe = sim_single(N, ku, kv, Au, Av, fu, fv, b, L, n_cw, channel, seed + n * 100000)
        elapsed = time.time() - t0
        total_time += elapsed

        bler = be / n_cw
        ber_u = ube / max(1, n_cw * ku)
        ber_v = vbe / max(1, n_cw * kv)

        results[N] = {
            "N": N, "ku": ku, "kv": kv, "Ru": round(Ru, 4), "Rv": round(Rv, 4),
            "L": L, "path_i": pi,
            "bler": bler, "ber_u": ber_u, "ber_v": ber_v,
            "block_errors": be, "n_codewords": n_cw,
            "time_s": round(elapsed, 1),
        }

        log(f"  BLER={bler:.4f}  BER_u={ber_u:.4e}  BER_v={ber_v:.4e}  "
            f"({be}/{n_cw})  {elapsed:.1f}s")
        log()

        # Checkpoint
        output = {
            "description": f"GMAC 6dB Class B SCL(L={L}) fixed-rate",
            "channel": "Gaussian MAC",
            "snr_db": snr_db, "sigma2": sigma2,
            "class": "B", "L": L, "path_i_frac": 0.5,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }
        out_path = os.path.join(os.path.dirname(__file__), "..", "results",
                                "gmac_snr6dB", "scl_L4_classB_fixed_rate.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for N, r in sorted(results.items()):
        log(f"  N={N:>5}  BLER={r['bler']:.4f}  ({r['block_errors']}/{r['n_codewords']})  {r['time_s']:.0f}s")
    log(f"Total: {total_time:.0f}s ({total_time/3600:.2f}h)")


if __name__ == "__main__":
    main()
