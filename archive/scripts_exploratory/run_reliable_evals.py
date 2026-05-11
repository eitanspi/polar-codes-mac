#!/usr/bin/env python3
"""
Reliable BLER evaluations for:
  - GMAC Class B NCG at N=32, 64, 512, 1024
  - GMAC Class C NCG at N=32, 64 (if checkpoints found)
  - BEMAC Class C SC at N=1024
"""

import sys, os, json, time, math
import numpy as np
import torch

torch.set_num_threads(4)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC, BEMAC
from polar.design import make_path, design_bemac
from polar.design_mc import design_from_file
from polar.eval import MACEval, build_message_batch
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder
from neural.neural_scl import SimpleMLP_Gmac

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'reliable_evals')

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)


def wilson_ci(n_errors, n_total, z=1.96):
    if n_total == 0:
        return 0, 0, 0
    p_hat = n_errors / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return p_hat, max(0, center - half), min(1, center + half)


def load_ncg_model(ckpt_path):
    m = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    fixed = {}
    for k, v in sd.items():
        if k.startswith('z_enc.'):
            fixed[k.replace('z_enc.', 'z_encoder.')] = v
        else:
            fixed[k] = v
    m.load_state_dict(fixed, strict=False)
    m.eval()
    return m


def load_gmac_design(N, ku, kv, cls='B'):
    n = int(math.log2(N))
    mc_path = os.path.join(DESIGNS_DIR, f'gmac_{cls}_n{n}_snr{SNR_DB:.0f}dB.npz')
    Au, Av, fu, fv, _, _, path_i = design_from_file(mc_path, n, ku, kv)
    return Au, Av, fu, fv


def eval_ncg(model, channel, N, b, Au, Av, fu, fv, n_cw, batch_size=None, time_limit=10800):
    model.eval()
    if batch_size is None:
        if N <= 64:
            batch_size = 16
        elif N <= 128:
            batch_size = 8
        elif N <= 256:
            batch_size = 4
        elif N <= 512:
            batch_size = 2
        else:
            batch_size = 1

    errs = 0
    total = 0
    rng = np.random.default_rng(999)
    t0 = time.time()

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au:
                uf[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                vf[:, p - 1] = rng.integers(0, 2, actual)

            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            _, _, uh, vh, _ = model(zf, b, fu, fv)

            for i in range(actual):
                err = False
                for p in Au:
                    if p in uh and int(uh[p][i].item()) != uf[i, p - 1]:
                        err = True
                        break
                if not err:
                    for p in Av:
                        if p in vh and int(vh[p][i].item()) != vf[i, p - 1]:
                            err = True
                            break
                if err:
                    errs += 1

            total += actual
            elapsed = time.time() - t0
            if total % max(100, batch_size * 10) == 0 or total == n_cw:
                bler_now = errs / total if total > 0 else 0
                print(f'    [{total:>6d}/{n_cw}] errors={errs} BLER={bler_now:.4f} ({elapsed:.1f}s)', flush=True)

            if elapsed > time_limit:
                print(f'  TIME LIMIT ({time_limit}s) reached at {total} CW, stopping.')
                break

    return errs, total, time.time() - t0


def save_result(result, filename):
    path = os.path.join(RESULTS_DIR, filename)
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.append(result)
    with open(path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'  Saved to {path}')


def run_gmac_ncg(N, ku, kv, n_cw, ckpt_name, cls='B', path_i=None, time_limit=10800):
    print(f'\n{"="*60}')
    print(f'GMAC Class {cls} NCG: N={N}, ku={ku}, kv={kv}, n_cw={n_cw}')
    print(f'{"="*60}')

    ckpt_path = os.path.join(SAVE_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f'  SKIP: checkpoint not found: {ckpt_path}')
        return None

    model = load_ncg_model(ckpt_path)
    print(f'  Loaded {ckpt_path} ({model.count_parameters()} params)')

    Au, Av, fu, fv = load_gmac_design(N, ku, kv, cls=cls)
    channel = GaussianMAC(sigma2=SIGMA2)
    if path_i is None:
        path_i = N // 2 if cls == 'B' else N
    b = make_path(N, path_i)

    errs, total, elapsed = eval_ncg(model, channel, N, b, Au, Av, fu, fv, n_cw, time_limit=time_limit)

    bler, ci_lo, ci_hi = wilson_ci(errs, total)
    print(f'\n  RESULT: BLER={bler:.6f} ({errs}/{total}), CI=[{ci_lo:.6f}, {ci_hi:.6f}], {elapsed:.1f}s')
    print(f'  Reliable: {errs >= 100}')

    result = {
        'channel': 'GMAC', 'class': cls, 'N': N,
        'decoder': 'NCG', 'ku': ku, 'kv': kv,
        'sigma2': SIGMA2, 'snr_db': SNR_DB,
        'bler': bler, 'n_cw': total, 'n_errors': errs,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'elapsed_s': round(elapsed, 1),
        'reliable': errs >= 100,
        'checkpoint': ckpt_name,
        'path_i': path_i,
    }
    save_result(result, f'gmac_{cls}_ncg_reliable.json')
    return result


def run_bemac_classC_sc(N, ku, kv, n_cw, time_limit=10800):
    print(f'\n{"="*60}')
    print(f'BEMAC Class C SC: N={N}, ku={ku}, kv={kv}, n_cw={n_cw}')
    print(f'{"="*60}')

    n = int(math.log2(N))
    channel = BEMAC()
    b = make_path(N, N)  # Class C

    Au, Av, fu, fv, zu, zv = design_bemac(n, ku, kv)

    evaluator = MACEval(
        channel=channel, log_domain=False,
        backend='interleaved'
    )

    errs = 0
    total = 0
    t0 = time.time()
    batch_size = 50

    rng = np.random.default_rng(42)

    while total < n_cw:
        bs = min(batch_size, n_cw - total)

        U_info = rng.integers(0, 2, size=(bs, ku), dtype=np.int32)
        V_info = rng.integers(0, 2, size=(bs, kv), dtype=np.int32)

        U_msg = build_message_batch(N, U_info, Au)
        V_msg = build_message_batch(N, V_info, Av)

        X = polar_encode_batch(U_msg)
        Y = polar_encode_batch(V_msg)

        Z_list = []
        for k in range(bs):
            z_k = channel.sample_batch(X[k], Y[k])
            z_k = z_k.tolist() if hasattr(z_k, 'tolist') else list(z_k)
            Z_list.append(z_k)

        decoded = evaluator._decode_batch(N, Z_list, b, fu, fv)

        for k, (u_dec, v_dec) in enumerate(decoded):
            u_errs = sum(u_dec[p - 1] != U_msg[k, p - 1] for p in Au)
            v_errs = sum(v_dec[p - 1] != V_msg[k, p - 1] for p in Av)
            if u_errs > 0 or v_errs > 0:
                errs += 1

        total += bs
        elapsed = time.time() - t0
        if total % 1000 == 0 or total == n_cw:
            bler_now = errs / total if total > 0 else 0
            print(f'    [{total:>6d}/{n_cw}] errors={errs} BLER={bler_now:.6f} ({elapsed:.1f}s)', flush=True)

        if elapsed > time_limit:
            print(f'  TIME LIMIT ({time_limit}s) reached at {total} CW, stopping.')
            break

    elapsed = time.time() - t0
    bler, ci_lo, ci_hi = wilson_ci(errs, total)
    print(f'\n  RESULT: BLER={bler:.6f} ({errs}/{total}), CI=[{ci_lo:.6f}, {ci_hi:.6f}], {elapsed:.1f}s')
    print(f'  Reliable: {errs >= 100}')

    result = {
        'channel': 'BEMAC', 'class': 'C', 'N': N,
        'decoder': 'SC', 'ku': ku, 'kv': kv,
        'bler': bler, 'n_cw': total, 'n_errors': errs,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'elapsed_s': round(elapsed, 1),
        'reliable': errs >= 100,
    }
    save_result(result, 'bemac_C_sc_reliable.json')
    return result


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []

    # ── Item 1: GMAC Class B NCG at N=32, N=64 (~30 min) ──
    r = run_gmac_ncg(N=32, ku=15, kv=15, n_cw=3000,
                     ckpt_name='ncg_gmac_mlp_N32.pt', cls='B')
    if r: all_results.append(r)

    r = run_gmac_ncg(N=64, ku=31, kv=31, n_cw=5000,
                     ckpt_name='ncg_gmac_mlp_N64.pt', cls='B')
    if r: all_results.append(r)

    # ── Item 2: GMAC Class B NCG at N=512 (~2-4h) ──
    r = run_gmac_ncg(N=512, ku=246, kv=246, n_cw=10000,
                     ckpt_name='ncg_gmac_mlp_N512.pt', cls='B', time_limit=10800)
    if r: all_results.append(r)

    # ── Item 3: GMAC Class B NCG at N=1024 (~2-4h) ──
    r = run_gmac_ncg(N=1024, ku=492, kv=492, n_cw=2000,
                     ckpt_name='ncg_gmac_mlp_N1024.pt', cls='B', time_limit=10800)
    if r: all_results.append(r)

    # ── Item 5: BEMAC Class C SC at N=1024 (100K CW, ~1-2h) ──
    r = run_bemac_classC_sc(N=1024, ku=307, kv=614, n_cw=100000, time_limit=7200)
    if r: all_results.append(r)

    print(f'\n\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    for r in all_results:
        tag = f"{r['channel']} {r['class']} {r['decoder']} N={r['N']}"
        print(f"  {tag:40s} BLER={r['bler']:.6f} ({r['n_errors']}/{r['n_cw']}) "
              f"{'RELIABLE' if r['reliable'] else 'partial'}")
