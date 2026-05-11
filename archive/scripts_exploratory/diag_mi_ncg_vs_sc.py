#!/usr/bin/env python3
"""
Diagnostic: per-position MI under teacher forcing for NCG vs SC on Class B GMAC.
- POC-0: N=32 SC MI (3000 samples) vs saved NCG MI (already in JSON).
- POC-1: N=256 NCG MI + SC MI (10K samples).
Numbers only, no plots, no commits.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch

torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from class_c_npd.plots.compute_mi import measure_mi_soft
from neural.ncg_gmac import GmacNeuralCompGraphDecoder
from neural.neural_scl import SimpleMLP_Gmac

SIGMA2 = 10 ** (-6.0 / 10)
CAP_XZ = 0.465
CAP_YZ_X = 0.912
CAP_XYZ = 1.376


def binary_entropy(p):
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def measure_ncg_mi(model, ch, b, N, n_samples, batch, device, seed=12345):
    """Teacher-forcing per-position MI for an NCG model whose forward matches."""
    rng = np.random.default_rng(seed)
    h_u = np.zeros(N); cnt_u = np.zeros(N)
    h_v = np.zeros(N); cnt_v = np.zeros(N)
    done = 0
    while done < n_samples:
        B = min(batch, n_samples - done)
        u = rng.integers(0, 2, (B, N)).astype(int)
        v = rng.integers(0, 2, (B, N)).astype(int)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float().to(device)
        ut = torch.from_numpy(u).float().to(device)
        vt = torch.from_numpy(v).float().to(device)
        with torch.no_grad():
            L, _, _, _, _ = model(z, b, {}, {}, u_true=ut, v_true=vt)
        idx_u = idx_v = 0
        for step in range(2 * N):
            gamma = b[step]
            lg = L[step]
            if gamma == 0:
                lp0 = torch.logsumexp(lg[:, :2], 1)
                lp1 = torch.logsumexp(lg[:, 2:], 1)
            else:
                lp0 = torch.logsumexp(lg[:, [0, 2]], 1)
                lp1 = torch.logsumexp(lg[:, [1, 3]], 1)
            p0 = torch.sigmoid(lp0 - lp1).cpu().numpy()
            h = binary_entropy(p0).sum()
            if gamma == 0:
                idx_u += 1
                h_u[idx_u - 1] += h; cnt_u[idx_u - 1] += B
            else:
                idx_v += 1
                h_v[idx_v - 1] += h; cnt_v[idx_v - 1] += B
        done += B
    mi_u = 1.0 - h_u / np.maximum(cnt_u, 1)
    mi_v = 1.0 - h_v / np.maximum(cnt_v, 1)
    return mi_u, mi_v


def hist_counts(mi):
    return {
        'gt_0.99': int((mi > 0.99).sum()),
        'gt_0.9': int((mi > 0.9).sum()),
        'gt_0.5': int((mi > 0.5).sum()),
        'lt_0.5': int((mi < 0.5).sum()),
        'lt_0.1': int((mi < 0.1).sum()),
        'lt_0.01': int((mi < 0.01).sum()),
    }


def cdf_at(mi, xs):
    return {f'{x:.1f}': float((mi <= x).mean()) for x in xs}


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def topk_overlap(mi_a, mi_b, k):
    a = set(np.argsort(-mi_a)[:k].tolist())
    b = set(np.argsort(-mi_b)[:k].tolist())
    return len(a & b)


def worst_info_mi(mi, k):
    """MI at the k-th-best position (1-indexed)."""
    srt = np.sort(mi)[::-1]  # descending
    return float(srt[k - 1])


# ─── Part 1: POC-0 (N=32) ──────────────────────────────────────────────────
def poc0():
    print('=' * 72)
    print('Part 1 — POC-0 sanity at N=32')
    print('=' * 72)
    N = 32
    b = make_path(N, N // 2)
    ch = GaussianMAC(sigma2=SIGMA2)

    # SC MI via compute_mi.measure_mi_soft (3000 samples)
    t0 = time.time()
    sc_mi_u, sc_mi_v = measure_mi_soft(N, ch, b, n_samples=3000, seed=789)
    print(f'SC N=32 (3000 samples, {time.time()-t0:.1f}s):')
    print(f'  avg MI_U = {sc_mi_u.mean():.4f}')
    print(f'  avg MI_V = {sc_mi_v.mean():.4f}')
    print(f'  sum      = {(sc_mi_u+sc_mi_v).mean():.4f}')

    # NCG from saved JSON (truncated) — parse mi_u_mean / mi_v_mean
    path = os.path.join(os.path.dirname(__file__), '..',
                       'class_c_npd', 'results', 'ncg_r1_32',
                       'mi_per_pos_reliable.json')
    with open(path) as f:
        text = f.read()
    idx = text.find('"ncg_info_u_top13"')
    if idx > 0:
        text = text[:idx].rstrip().rstrip(',') + '\n}'
    d = json.loads(text)
    ncg_mi_u = np.array(d['mi_u_mean'])
    ncg_mi_v = np.array(d['mi_v_mean'])
    print(f'NCG N=32 (saved, mean over 3 seeds x 200K):')
    print(f'  avg MI_U = {ncg_mi_u.mean():.4f}')
    print(f'  avg MI_V = {ncg_mi_v.mean():.4f}')
    print(f'  sum      = {(ncg_mi_u+ncg_mi_v).mean():.4f}')

    # Consistency check
    corr_u = pearson(sc_mi_u, ncg_mi_u)
    corr_v = pearson(sc_mi_v, ncg_mi_v)
    print(f'\nPer-position Pearson corr: MI_U={corr_u:.4f}  MI_V={corr_v:.4f}')
    # top-13 U and top-25 V overlap
    o13 = topk_overlap(sc_mi_u, ncg_mi_u, 13)
    o25 = topk_overlap(sc_mi_v, ncg_mi_v, 25)
    print(f'Top-13 U overlap: {o13}/13   Top-25 V overlap: {o25}/25')

    return {
        'sc_mi_u': sc_mi_u, 'sc_mi_v': sc_mi_v,
        'ncg_mi_u': ncg_mi_u, 'ncg_mi_v': ncg_mi_v,
    }


# ─── Part 2: POC-1 (N=256) ─────────────────────────────────────────────────
def poc1():
    print()
    print('=' * 72)
    print('Part 2 — POC-1 at N=256')
    print('=' * 72)

    N = 256
    b = make_path(N, N // 2)
    ch = GaussianMAC(sigma2=SIGMA2)
    device = torch.device('cpu')

    # Load NCG N=256
    ckpt = os.path.join(os.path.dirname(__file__), '..',
                        'saved_models', 'ncg_gmac_mlp_N256.pt')
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    fixed = {(k.replace('z_enc.', 'z_encoder.') if k.startswith('z_enc.') else k): v
             for k, v in sd.items()}
    r = model.load_state_dict(fixed, strict=False)
    print(f'NCG N=256 model loaded. missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}')
    if r.missing_keys:
        print('  missing:', r.missing_keys[:5])
    if r.unexpected_keys:
        print('  unexpected:', r.unexpected_keys[:5])
    model.eval()

    N_SAMPLES = 10000
    BATCH = 64

    t0 = time.time()
    ncg_mi_u, ncg_mi_v = measure_ncg_mi(model, ch, b, N, N_SAMPLES, BATCH, device, seed=12345)
    print(f'NCG N=256 (10K samples, {time.time()-t0:.1f}s):')
    print(f'  avg MI_U = {ncg_mi_u.mean():.4f}  sum MI_U = {ncg_mi_u.sum():.2f}')
    print(f'  avg MI_V = {ncg_mi_v.mean():.4f}  sum MI_V = {ncg_mi_v.sum():.2f}')

    t0 = time.time()
    sc_mi_u, sc_mi_v = measure_mi_soft(N, ch, b, n_samples=N_SAMPLES, seed=789)
    print(f'SC N=256 (10K samples, {time.time()-t0:.1f}s):')
    print(f'  avg MI_U = {sc_mi_u.mean():.4f}  sum MI_U = {sc_mi_u.sum():.2f}')
    print(f'  avg MI_V = {sc_mi_v.mean():.4f}  sum MI_V = {sc_mi_v.sum():.2f}')

    # Insight 1 — histogram counts
    print('\n[Insight 1] Histogram counts (out of N=256 positions per user):')
    print(f'{"":>15} {"NCG_U":>8} {"SC_U":>8} {"NCG_V":>8} {"SC_V":>8}')
    bins = ['gt_0.99', 'gt_0.9', 'gt_0.5', 'lt_0.5', 'lt_0.1', 'lt_0.01']
    hu_n = hist_counts(ncg_mi_u); hu_s = hist_counts(sc_mi_u)
    hv_n = hist_counts(ncg_mi_v); hv_s = hist_counts(sc_mi_v)
    for bk in bins:
        print(f'{bk:>15} {hu_n[bk]:>8} {hu_s[bk]:>8} {hv_n[bk]:>8} {hv_s[bk]:>8}')

    # Insight 2 — scatter representative + correlation
    # Sort by SC_MI descending, pick positions spanning the full SC MI range
    print('\n[Insight 2] Scatter NCG vs SC — representative positions:')
    for user, ncg, sc in [('U', ncg_mi_u, sc_mi_u), ('V', ncg_mi_v, sc_mi_v)]:
        order = np.argsort(-sc)
        picks_idx = [0, N // 10, N // 4, N // 2, 3 * N // 4, 9 * N // 10, N - 1]
        print(f'  {user} — (pos, SC_MI, NCG_MI):')
        for pi in picks_idx:
            p = order[pi]
            print(f'    pos {p+1:>4d}: SC={sc[p]:.4f}  NCG={ncg[p]:.4f}')
        pr = pearson(sc, ncg)
        print(f'  {user} Pearson corr: {pr:.4f}')

    # Insight 3 — top-K ranking overlap
    print('\n[Insight 3] Top-K ranking overlap (NCG vs SC):')
    print(f'{"K":>5} {"U_overlap":>12} {"V_overlap":>12}')
    for K in (100, 123, 140):
        ou = topk_overlap(ncg_mi_u, sc_mi_u, K)
        ov = topk_overlap(ncg_mi_v, sc_mi_v, K)
        print(f'{K:>5} {ou:>7}/{K:<4} {ov:>7}/{K:<4}')

    # Insight 4 — sum MI per user
    print('\n[Insight 4] Sum MI per user  (expected N*0.688 = 176.13 per user):')
    print(f'  sum MI_U  NCG={ncg_mi_u.sum():.2f}   SC={sc_mi_u.sum():.2f}   '
          f'N*I(X;Z)={N*CAP_XZ:.2f}')
    print(f'  sum MI_V  NCG={ncg_mi_v.sum():.2f}   SC={sc_mi_v.sum():.2f}   '
          f'N*I(Y;Z|X)={N*CAP_YZ_X:.2f}')
    print(f'  sum total NCG={(ncg_mi_u+ncg_mi_v).sum():.2f}   '
          f'SC={(sc_mi_u+sc_mi_v).sum():.2f}   N*I(X,Y;Z)={N*CAP_XYZ:.2f}')

    # Insight 5 — MI at 123rd-best position
    print('\n[Insight 5] 123rd-best MI (worst info position, 1-indexed):')
    for user, ncg, sc in [('U', ncg_mi_u, sc_mi_u), ('V', ncg_mi_v, sc_mi_v)]:
        nw = worst_info_mi(ncg, 123)
        sw = worst_info_mi(sc, 123)
        print(f'  {user}: NCG={nw:.4f}   SC={sw:.4f}')

    # Insight 6 — CDF
    print('\n[Insight 6] CDF: fraction of positions with MI <= x:')
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f'{"x":>5} {"NCG_U":>8} {"SC_U":>8} {"NCG_V":>8} {"SC_V":>8}')
    cu_n = cdf_at(ncg_mi_u, xs); cu_s = cdf_at(sc_mi_u, xs)
    cv_n = cdf_at(ncg_mi_v, xs); cv_s = cdf_at(sc_mi_v, xs)
    for x in xs:
        key = f'{x:.1f}'
        print(f'{x:>5.1f} {cu_n[key]:>8.4f} {cu_s[key]:>8.4f} {cv_n[key]:>8.4f} {cv_s[key]:>8.4f}')

    # Save raw arrays
    out = os.path.join(os.path.dirname(__file__), '..', 'results',
                       'diag_mi_ncg_vs_sc_N256.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'ncg_mi_u': ncg_mi_u.tolist(),
            'ncg_mi_v': ncg_mi_v.tolist(),
            'sc_mi_u': sc_mi_u.tolist(),
            'sc_mi_v': sc_mi_v.tolist(),
            'n_samples': N_SAMPLES,
        }, f, indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    poc0()
    poc1()
