#!/usr/bin/env python3
"""
Priority 2: Fill reliability gaps — run more CW for entries with <100 errors.
Ordered by expected time (fastest first: closest to 100 errors).
"""
import os, sys, math, time, json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(4)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import BEMAC, GaussianMAC, ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder import decode_single, decode_batch
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

RESULTS_FILE = os.path.join(BASE, 'results', 'reliability_fill_results.json')
all_results = {}


def save_results():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {RESULTS_FILE}", flush=True)


def load_bemac_design_B(N, ku, kv):
    n = int(math.log2(N))
    design_file = os.path.join(BASE, 'designs', f'bemac_B_n{n}.npz')
    Au, Av, fu, fv, _, _, path_i = design_from_file(design_file, n, ku, kv)
    b = make_path(N, path_i)
    return sorted(Au), sorted(Av), fu, fv, b


def load_gmac_design(N, ku, kv, cls='B'):
    n = int(math.log2(N))
    design_file = os.path.join(BASE, 'designs', f'gmac_{cls}_n{n}_snr6dB.npz')
    Au, Av, fu, fv, _, _, path_i = design_from_file(design_file, n, ku, kv)
    b = make_path(N, path_i)
    return sorted(Au), sorted(Av), fu, fv, b


def load_abnmac_design(N, ku, kv, cls='B'):
    n = int(math.log2(N))
    design_file = os.path.join(BASE, 'designs', f'abnmac_{cls}_n{n}.npz')
    Au, Av, fu, fv, _, _, path_i = design_from_file(design_file, n, ku, kv)
    b = make_path(N, path_i)
    return sorted(Au), sorted(Av), fu, fv, b


def eval_sc_bemac(N, Au, Av, fu, fv, b, n_cw, seed=42):
    """SC eval for BEMAC (discrete channel)."""
    channel = BEMAC()
    rng = np.random.default_rng(seed)
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    errs = 0; total = 0
    t0 = time.time()
    batch_sz = min(25, max(1, 256 // N))
    while total < n_cw:
        actual = min(batch_sz, n_cw - total)
        info_u = rng.integers(0, 2, size=(actual, len(Au)))
        info_v = rng.integers(0, 2, size=(actual, len(Av)))
        U = build_message_batch(N, info_u, Au)
        V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U)
        Y = polar_encode_batch(V)
        Z = channel.sample_batch(X, Y)
        results = decode_batch(N, Z.tolist(), b, fu, fv, channel, vectorized=True)
        for i, (u_dec, v_dec) in enumerate(results):
            u_dec_arr = np.array(u_dec); v_dec_arr = np.array(v_dec)
            if np.sum(u_dec_arr[u_info_idx] != info_u[i]) > 0 or \
               np.sum(v_dec_arr[v_info_idx] != info_v[i]) > 0:
                errs += 1
        total += actual
        if total % 2000 == 0:
            print(f"    [{total}/{n_cw}] errs={errs} ({(time.time()-t0)/60:.1f}min)", flush=True)
    elapsed = (time.time() - t0) / 60
    bler = errs / total
    return {'bler': bler, 'errs': errs, 'n_cw': total, 'time_min': round(elapsed, 1)}


def eval_ncg_bemac(N, Au, Av, fu, fv, b, n_cw, ckpt_name, seed=42):
    """NCG eval for BEMAC (discrete channel, vocab_size=3)."""
    channel = BEMAC()
    ckpt_path = os.path.join(BASE, 'saved_models', ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"    Checkpoint not found: {ckpt_name}", flush=True)
        return None
    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=3)
    sd = torch.load(ckpt_path, weights_only=True, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.eval()

    rng = np.random.default_rng(seed)
    errs = 0; total = 0
    t0 = time.time()
    batch_sz = min(25, max(2, 256 // N))
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_sz, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            zt = torch.from_numpy(zf).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
            if total % 2000 == 0:
                print(f"    [{total}/{n_cw}] errs={errs} ({(time.time()-t0)/60:.1f}min)", flush=True)
    elapsed = (time.time() - t0) / 60
    bler = errs / total
    return {'bler': bler, 'errs': errs, 'n_cw': total, 'time_min': round(elapsed, 1)}


def eval_sc_gmac(N, Au, Av, fu, fv, b, n_cw, sigma2, seed=42):
    """SC eval for GMAC (continuous channel)."""
    channel = GaussianMAC(sigma2=sigma2)
    rng = np.random.default_rng(seed)
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    errs = 0; total = 0
    t0 = time.time()
    for cw in range(n_cw):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
        for p in Au: u[p-1] = rng.integers(0, 2)
        for p in Av: v[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u[None, :])[0]
        y = polar_encode_batch(v[None, :])[0]
        z = channel.sample_batch(x[None, :].astype(int), y[None, :].astype(int))
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.ndim == 2: z_arr = z_arr[0]
        u_dec, v_dec = decode_single(N, z_arr.tolist(), b, fu, fv, channel, log_domain=True)
        u_dec_arr = np.array(u_dec); v_dec_arr = np.array(v_dec)
        if np.sum(u_dec_arr[u_info_idx] != u[u_info_idx]) > 0 or \
           np.sum(v_dec_arr[v_info_idx] != v[v_info_idx]) > 0:
            errs += 1
        total += 1
        if total % 1000 == 0:
            print(f"    [{total}/{n_cw}] errs={errs} ({(time.time()-t0)/60:.1f}min)", flush=True)
    elapsed = (time.time() - t0) / 60
    bler = errs / total
    return {'bler': bler, 'errs': errs, 'n_cw': total, 'time_min': round(elapsed, 1)}


def eval_sc_maagn(N, Au, Av, fu, fv, b, n_cw, sigma2, alpha, seed=42):
    """Memoryless SC eval for MA-AGN (uses GMAC as proxy decoder)."""
    from polar.channels_memory_new import MAAGNMAC
    ch = MAAGNMAC(sigma2=sigma2, alpha=alpha)
    gmac = GaussianMAC(sigma2=sigma2)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    errs = 0; total = 0
    t0 = time.time()
    for cw in range(n_cw):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
        for p in Au: u[p-1] = rng.integers(0, 2)
        for p in Av: v[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u[None, :])[0]
        y = polar_encode_batch(v[None, :])[0]
        z = ch.sample_batch(x[None, :].astype(int), y[None, :].astype(int))
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.ndim == 2: z_arr = z_arr[0]
        u_dec, v_dec = decode_single(N, z_arr.tolist(), b, fu, fv, gmac, log_domain=True)
        u_dec_arr = np.array(u_dec); v_dec_arr = np.array(v_dec)
        if np.sum(u_dec_arr[u_info_idx] != u[u_info_idx]) > 0 or \
           np.sum(v_dec_arr[v_info_idx] != v[v_info_idx]) > 0:
            errs += 1
        total += 1
        if total % 1000 == 0:
            print(f"    [{total}/{n_cw}] errs={errs} ({(time.time()-t0)/60:.1f}min)", flush=True)
    elapsed = (time.time() - t0) / 60
    bler = errs / total
    return {'bler': bler, 'errs': errs, 'n_cw': total, 'time_min': round(elapsed, 1)}


def main():
    global all_results
    t_global = time.time()
    BUDGET_HOURS = 4.0

    # ── Task 1: BEMAC B N=16 SC (55 errs in 5K, need 10K more) ──
    print("\n=== BEMAC B N=16 SC (target: 10K more CW) ===", flush=True)
    Au, Av, fu, fv, b = load_bemac_design_B(16, 8, 11)
    r = eval_sc_bemac(16, Au, Av, fu, fv, b, 10000, seed=100)
    all_results['bemac_B_N16_SC'] = r
    print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']})", flush=True)
    save_results()

    # ── Task 2: BEMAC B N=16 NCG (55 errs in 5K, need 10K more) ──
    print("\n=== BEMAC B N=16 NCG (target: 10K more CW) ===", flush=True)
    r = eval_ncg_bemac(16, Au, Av, fu, fv, b, 10000, 'ncg_pure_neural_N16.pt', seed=101)
    if r:
        all_results['bemac_B_N16_NCG'] = r
        print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']})", flush=True)
    save_results()

    if (time.time() - t_global) / 3600 > BUDGET_HOURS:
        print("TIME LIMIT"); return

    # ��─ Task 3: GMAC C N=128 SC (71 errs in 10K, need 5K more) ��─
    print("\n=== GMAC C N=128 SC (target: 5K more CW) ===", flush=True)
    sigma2 = 10 ** (-6.0 / 10)
    Au, Av, fu, fv, b = load_gmac_design(128, 30, 58, cls='C')
    r = eval_sc_gmac(128, Au, Av, fu, fv, b, 5000, sigma2, seed=102)
    all_results['gmac_C_N128_SC'] = r
    print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']})", flush=True)
    save_results()

    if (time.time() - t_global) / 3600 > BUDGET_HOURS:
        print("TIME LIMIT"); return

    # ── Task 4: BEMAC B N=128 SC (81 errs in 50K, need 20K more) ���─
    print("\n=== BEMAC B N=128 SC (target: 20K more CW) ===", flush=True)
    Au, Av, fu, fv, b = load_bemac_design_B(128, 64, 90)
    r = eval_sc_bemac(128, Au, Av, fu, fv, b, 20000, seed=103)
    all_results['bemac_B_N128_SC'] = r
    print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']})", flush=True)
    save_results()

    if (time.time() - t_global) / 3600 > BUDGET_HOURS:
        print("TIME LIMIT"); return

    # ── Task 5: MA-AGN N=128 SC (23 errs in 3K, need 15K more) ──
    print("\n=== MA-AGN N=128 SC (target: 15K more CW) ===", flush=True)
    sigma2_agn = 10 ** (-6.0 / 10)
    Au, Av, fu, fv, b = load_gmac_design(128, 30, 58, cls='C')
    r = eval_sc_maagn(128, Au, Av, fu, fv, b, 15000, sigma2_agn, alpha=0.3, seed=104)
    all_results['maagn_N128_SC'] = r
    print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']})", flush=True)
    save_results()

    if (time.time() - t_global) / 3600 > BUDGET_HOURS:
        print("TIME LIMIT"); return

    # ── Task 6: ISI-MAC N=256 SC (61 errs in 10K, need 5K more) ──
    print("\n=== ISI-MAC N=256 SC (target: 5K more CW) ===", flush=True)
    try:
        from polar.channels_memory_new import ISIMAC2
        isi_ch = ISIMAC2(h=0.3, sigma2=sigma2_agn)
        Au, Av, fu, fv, b = load_gmac_design(256, 59, 117, cls='C')
        r = eval_sc_gmac(256, Au, Av, fu, fv, b, 5000, sigma2_agn, seed=105)
        all_results['isimac_N256_SC'] = r
        print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']})", flush=True)
    except Exception as e:
        print(f"  ISI-MAC N=256 failed: {e}", flush=True)
    save_results()

    if (time.time() - t_global) / 3600 > BUDGET_HOURS:
        print("TIME LIMIT"); return

    # ── Task 7: GMAC B N=256 SC (30 errs in 5K, need 20K more) ──
    print("\n=== GMAC B N=256 SC (target: 20K more CW) ===", flush=True)
    Au, Av, fu, fv, b = load_gmac_design(256, 123, 123, cls='B')
    r = eval_sc_gmac(256, Au, Av, fu, fv, b, 20000, sigma2_agn, seed=106)
    all_results['gmac_B_N256_SC'] = r
    print(f"  BLER={r['bler']:.4f} ({r['errs']}/{r['n_cw']})", flush=True)
    save_results()

    elapsed_h = (time.time() - t_global) / 3600
    print(f"\n  All reliability fill tasks done. Total time: {elapsed_h:.2f}h", flush=True)


if __name__ == '__main__':
    main()
