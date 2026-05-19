"""30-second smoke test: imports work, channel + decoders + design + eval roundtrip.

Run after `pip install -r requirements.txt`.  Expected output:
    Encoder roundtrip   : OK
    Channel sample      : OK
    Joint MAC SCT decode: OK
    Chained SCT N=16    : BLER 0.10-0.20 (random noise)

If any line errors, the corresponding module is broken.
"""
import sys, os, time
import numpy as np

# Make `polar/`, `neural/`, etc. importable from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)


def main():
    print("=== Polar Codes MAC — smoke test ===\n")

    # 1. Encoder
    from polar.encoder import polar_encode, polar_encode_batch
    u = [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]
    x = polar_encode(u)
    xb = polar_encode_batch(np.array(u, dtype=np.int32).reshape(1, -1))[0]
    assert list(xb) == x, "polar_encode and polar_encode_batch disagree"
    print(f"Encoder roundtrip    : OK  (N={len(u)}, x[:5] = {x[:5]})")

    # 2. Channel
    from polar.channels_memory import ISIMAC
    ch = ISIMAC(sigma2=10 ** (-0.6), h=0.3)
    X = np.random.randint(0, 2, (1, 16))
    Y = np.random.randint(0, 2, (1, 16))
    z = ch.sample_batch(X, Y)
    assert z.shape == (1, 16) and np.isfinite(z).all()
    print(f"Channel sample       : OK  (ISI-MAC h=0.3, |z|_max = {np.abs(z).max():.3f})")

    # 3. Joint MAC SCT decode at N=16
    from polar.decoder_trellis import decode_single
    from polar.design import make_path
    N = 16
    fu = {1: 0, 2: 0, 3: 0, 5: 0, 6: 0, 7: 0, 9: 0, 10: 0,
          11: 0, 13: 0}                      # keep info: 4, 8, 12, 14, 15, 16  (ku=6 for demo)
    fv = {1: 0, 2: 0, 3: 0, 5: 0, 6: 0, 9: 0, 10: 0, 11: 0, 13: 0}  # ku=7
    b = make_path(N, N)  # corner-rate Class C
    z1 = ch.sample_batch(X, Y)[0]
    u_hat, v_hat = decode_single(N, z1, b, fu, fv, ch)
    assert len(u_hat) == N and len(v_hat) == N
    print(f"Joint MAC SCT decode : OK  (u_hat[:5] = {u_hat[:5]})")

    # 4. Quick MC design + eval at N=16 with the 4-state chained SCT
    sys.path.insert(0, os.path.join(_ROOT, "scripts", "local_analysis"))
    from chained_sct_4state import mc_design, eval_at, pick, RATES
    N = 16
    ku, kv = RATES[N]
    t0 = time.time()
    Pe_u, Pe_v, _ = mc_design(N, 500, 1, base_seed=0)   # 500 trials, single thread
    Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
    r = eval_at(N, 500, Au, Av, 1, base_seed=1)
    print(f"Chained SCT N=16     : BLER {r['bler']:.3f} ({r['errs']}/{r['n_cw']}) "
          f"in {time.time() - t0:.1f}s")

    print("\nAll modules working.  See GETTING_STARTED.md for next steps.")


if __name__ == "__main__":
    main()
