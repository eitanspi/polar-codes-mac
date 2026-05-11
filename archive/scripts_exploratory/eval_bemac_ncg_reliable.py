#!/usr/bin/env python3
"""
eval_bemac_ncg_reliable.py - Reliable eval of BEMAC NCG models for Class B.
Target: >= 100 errors per N for N=32,64. N=128 partial.
"""
import os, sys, json, math, time, numpy as np, torch
torch.set_num_threads(4)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode_batch
from polar.channels import BEMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder


def wilson_ci(p, n, z=1.96):
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    spread = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, centre - spread), min(1, centre + spread)


def load_model(ckpt_path, d=16, hidden=64, n_layers=2):
    model = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers, vocab_size=3)
    sd = torch.load(ckpt_path, weights_only=True, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def get_design_class_b(N, ku, kv):
    n = int(math.log2(N))
    design_file = os.path.join(BASE, 'designs', f'bemac_B_n{n}.npz')
    if not os.path.exists(design_file):
        return None
    Au_list, Av_list, fu, fv, pe_u, pe_v, path_i = design_from_file(design_file, n, ku, kv)
    b = make_path(N, path_i)
    return sorted(Au_list), sorted(Av_list), fu, fv, b


RATES_B = {
    32: (16, 22), 64: (32, 45), 128: (64, 90),
}


def main():
    channel = BEMAC()
    results = {}
    t_global = time.time()

    for N in [32, 64, 128]:
        ku, kv = RATES_B[N]
        design = get_design_class_b(N, ku, kv)
        if design is None:
            print(f"N={N}: no design file, skip")
            continue
        Au, Av, fu, fv, b = design

        ckpt = os.path.join(BASE, 'saved_models', f'ncg_pure_neural_N{N}.pt')
        if not os.path.exists(ckpt):
            print(f"N={N}: no checkpoint, skip")
            continue

        model = load_model(ckpt)

        # CW targets based on expected BLER
        target_cw = {32: 15000, 64: 50000, 128: 50000}
        n_cw = target_cw[N]
        batch = max(2, min(16, 256 // N))

        print(f"\nN={N}, ku={ku}, kv={kv}, target CW={n_cw}, batch={batch}")
        t0 = time.time()

        rng = np.random.default_rng(42)
        errs = total = 0

        with torch.no_grad():
            while total < n_cw:
                actual = min(batch, n_cw - total)
                uf = np.zeros((actual, N), dtype=int)
                vf = np.zeros((actual, N), dtype=int)
                for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
                for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
                xf = polar_encode_batch(uf)
                yf = polar_encode_batch(vf)
                zf = channel.sample_batch(xf, yf)
                zt = torch.from_numpy(zf).long()
                _, _, uh, vh, _ = model(zt, b, fu, fv)
                for i in range(actual):
                    e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                        any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                    if e:
                        errs += 1
                total += actual

                if total % 2000 == 0:
                    elapsed = time.time() - t0
                    bler_now = errs / total
                    print(f"  {total}/{n_cw}  BLER={bler_now:.6f}  errs={errs}  "
                          f"time={elapsed:.1f}s", flush=True)
                    # Early stop if we have enough errors and have done >= 5000 CW
                    if errs >= 150 and total >= 5000:
                        print(f"  Early stop: {errs} errors >= 150 at {total} CW")
                        break

        elapsed = time.time() - t0
        bler = errs / total
        ci_lo, ci_hi = wilson_ci(bler, total)
        print(f"  FINAL: BLER={bler:.6f} ({errs}/{total}) CI=[{ci_lo:.6f}, {ci_hi:.6f}] "
              f"time={elapsed:.1f}s")

        results[str(N)] = {
            'N': N, 'ku': ku, 'kv': kv,
            'bler': bler, 'errs': errs, 'total': total,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'time_s': round(elapsed, 1),
            'ckpt': f'ncg_pure_neural_N{N}.pt',
        }

        out = os.path.join(BASE, 'results', 'bemac', 'bemac_classB_ncg_reliable.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved incrementally to {out}")

    print(f"\nTotal time: {(time.time()-t_global)/60:.1f} min")


if __name__ == '__main__':
    main()
