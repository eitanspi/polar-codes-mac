#!/usr/bin/env python3
"""
Task 1 verification: GMAC Class C corner NPD discrepancy.

Evaluates every plausible Class C N=64 NPD checkpoint under two modes:
  (a) Stage 1 only (U bits) -- matches old "serious_npd_vs_sc" eval style
  (b) Chained (Stage 1 + Stage 2) -- matches the new SNR sweep style

Target config: GMAC Class C, N=64, SNR=6 dB, ku=15, kv=29, 2000 codewords.

Also re-runs analytical SC at the same config.

Writes to results/snr_sweep/task1_gmac_corner_npd_verification.json.
"""
from __future__ import annotations
import os, sys, math, json, time
import numpy as np
import torch

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.decoder import decode_single
from polar.channels import GaussianMAC
from polar.design import make_path
from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.channels.mac_channel import build_channel as build_c_channel
from class_c_npd.channels.frozen_sets import load_class_c_design


SNR_DB = 6.0
N = 64
n_log = 6
N_CW = 2000
SEED = 20260416 + int(SNR_DB * 100) + N


def wilson_ci(errs: int, total: int, z: float = 1.96):
    if total == 0:
        return (0.0, 1.0)
    p = errs / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_npd_checkpoint(path: str) -> dict:
    ck = torch.load(path, weights_only=False, map_location='cpu')
    info = {
        'path': path,
        'basename': os.path.basename(path),
        'd': ck.get('d', 16),
        'hidden': ck.get('hidden', 64),
        'n_layers': ck.get('n_layers', 2),
        'z_dim': ck.get('z_dim', 1),
        'N': ck.get('N', N),
        'Au': sorted(ck.get('Au', [])),
        'Av': sorted(ck.get('Av', [])),
        'stage': ck.get('stage'),
        'channel': ck.get('channel'),
        'state_dict': ck['state_dict'],
    }
    return info


def build_model(info: dict) -> NPDSingleUser:
    model = NPDSingleUser(
        d=info['d'], hidden=info['hidden'],
        n_layers=info['n_layers'], z_dim=info['z_dim'],
    )
    model.load_state_dict(info['state_dict'])
    model.eval()
    return model


# ─── Eval modes ────────────────────────────────────────────────────────────

def eval_stage1_only(model_s1, Au, Av, n_cw=N_CW, seed=SEED):
    """Stage 1 only: measure U bit block error on the marginal channel."""
    sigma2 = 10 ** (-SNR_DB / 10)
    channel = build_c_channel('gmac', sigma2=sigma2)
    br = bit_reversal_perm(n_log)
    fu_set = {p - 1 for p in range(1, N + 1) if p not in Au}

    errs = 0
    total = 0
    batch = 16
    rng = np.random.default_rng(seed)

    t0 = time.time()
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg = np.zeros((actual, N), dtype=np.int8)
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Au:
                u_msg[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                v_msg[:, p - 1] = rng.integers(0, 2, actual)
            x = polar_encode_batch(u_msg.astype(int))
            y = polar_encode_batch(v_msg.astype(int))
            z = channel.sample_z(x.astype(int), y.astype(int))
            features1 = channel.stage1_features(z)
            features1_npd = (features1[..., br] if features1.ndim == 2
                             else features1[:, br, :])
            ft1 = torch.from_numpy(features1_npd).float()
            if ft1.dim() == 2:
                ft1 = ft1.unsqueeze(-1)
            emb1 = model_s1.encode_channel(ft1)
            u_hat = model_s1.decode(emb1, fu_set)
            for i in range(actual):
                u_wrong = any(int(u_hat[i, p - 1].item()) != int(u_msg[i, p - 1]) for p in Au)
                if u_wrong:
                    errs += 1
            total += actual
    t = time.time() - t0
    return errs, total, round(t, 2)


def eval_chained(model_s1, model_s2, Au, Av, n_cw=N_CW, seed=SEED):
    """Chained: U bit on marginal + V bit on clean channel given U_hat."""
    sigma2 = 10 ** (-SNR_DB / 10)
    channel = build_c_channel('gmac', sigma2=sigma2)
    br = bit_reversal_perm(n_log)
    fu_set = {p - 1 for p in range(1, N + 1) if p not in Au}
    fv_set = {p - 1 for p in range(1, N + 1) if p not in Av}

    errs = 0
    errs_u = 0
    errs_v = 0
    errs_v_given_u_ok = 0
    total = 0
    batch = 16
    rng = np.random.default_rng(seed)
    t0 = time.time()
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg = np.zeros((actual, N), dtype=np.int8)
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Au:
                u_msg[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                v_msg[:, p - 1] = rng.integers(0, 2, actual)
            x_phys = polar_encode_batch(u_msg.astype(int))
            y_phys = polar_encode_batch(v_msg.astype(int))
            z = channel.sample_z(x_phys.astype(int), y_phys.astype(int))

            # Stage 1
            features1 = channel.stage1_features(z)
            features1_npd = (features1[..., br] if features1.ndim == 2
                             else features1[:, br, :])
            ft1 = torch.from_numpy(features1_npd).float()
            if ft1.dim() == 2:
                ft1 = ft1.unsqueeze(-1)
            emb1 = model_s1.encode_channel(ft1)
            u_hat = model_s1.decode(emb1, fu_set)
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)

            # Stage 2
            features2 = channel.stage2_features(z, x_hat.astype(int))
            features2_npd = (features2[..., br] if features2.ndim == 2
                             else features2[:, br, :])
            ft2 = torch.from_numpy(features2_npd).float()
            if ft2.dim() == 2:
                ft2 = ft2.unsqueeze(-1)
            emb2 = model_s2.encode_channel(ft2)
            v_hat = model_s2.decode(emb2, fv_set)

            for i in range(actual):
                u_wrong = any(int(u_hat[i, p - 1].item()) != int(u_msg[i, p - 1]) for p in Au)
                v_wrong = any(int(v_hat[i, p - 1].item()) != int(v_msg[i, p - 1]) for p in Av)
                if u_wrong:
                    errs_u += 1
                if v_wrong:
                    errs_v += 1
                if (not u_wrong) and v_wrong:
                    errs_v_given_u_ok += 1
                if u_wrong or v_wrong:
                    errs += 1
            total += actual
    t = time.time() - t0
    return {
        'errs_total': errs,
        'errs_u': errs_u,
        'errs_v': errs_v,
        'errs_v_given_u_ok': errs_v_given_u_ok,
        'n_cw': total,
        'time_s': round(t, 2),
    }


def eval_sc(Au, Av, n_cw=N_CW, seed=SEED + 7):
    """Analytical chained SC on Class C path (path_i = N)."""
    sigma2 = 10 ** (-SNR_DB / 10)
    channel = GaussianMAC(sigma2=sigma2)
    fu = {p: 0 for p in range(1, N + 1) if p not in Au}
    fv = {p: 0 for p in range(1, N + 1) if p not in Av}
    b = make_path(N, N)  # Class C corner
    rng = np.random.default_rng(seed)
    errs = 0
    errs_u = 0
    errs_v = 0
    t0 = time.time()
    for _ in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au:
            uf[p - 1] = rng.integers(0, 2)
        for p in Av:
            vf[p - 1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int),
                                 np.array(y, dtype=int)).tolist()
        u_dec, v_dec = decode_single(N, z, b, fu, fv, channel, log_domain=True)
        u_wrong = any(u_dec[p - 1] != uf[p - 1] for p in Au)
        v_wrong = any(v_dec[p - 1] != vf[p - 1] for p in Av)
        if u_wrong:
            errs_u += 1
        if v_wrong:
            errs_v += 1
        if u_wrong or v_wrong:
            errs += 1
    t = time.time() - t0
    return {
        'errs_total': errs, 'errs_u': errs_u, 'errs_v': errs_v,
        'n_cw': n_cw, 'time_s': round(t, 2),
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    out_path = os.path.join(_ROOT, 'results', 'snr_sweep',
                            'task1_gmac_corner_npd_verification.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Checkpoints to test at N=64 for GMAC Class C (stage 1)
    CKPTS = {
        'curriculum_gmac_c_s1_N64_best.pt':
            os.path.join(_ROOT, 'class_c_npd/results/curriculum_gmac_c_s1_N64_best.pt'),
        'npd_design_p1_N64_best.pt':
            os.path.join(_ROOT, 'class_c_npd/results/npd_design_p1_N64_best.pt'),
        'npd_design_p3_N64_best.pt':
            os.path.join(_ROOT, 'class_c_npd/results/npd_design_p3_N64_best.pt'),
    }
    # Stage 2 checkpoints
    S2_CKPTS = {
        'curriculum_gmac_c_s2_N64_best.pt':
            os.path.join(_ROOT, 'class_c_npd/results/curriculum_gmac_c_s2_N64_best.pt'),
    }

    results = {
        'config': {
            'N': N, 'snr_db': SNR_DB, 'ku': 15, 'kv': 29,
            'channel': 'GaussianMAC (Class C corner, path_i=N)',
            'n_cw': N_CW,
        },
        'checkpoints_tested': {},
        'stage1_only': {},
        'chained': {},
        'sc_baseline': {},
        'notes': [],
    }

    # Inspect each checkpoint
    for name, p in CKPTS.items():
        if not os.path.exists(p):
            print(f'[skip] {name}: file not found')
            continue
        info = load_npd_checkpoint(p)
        results['checkpoints_tested'][name] = {
            'Au': info['Au'], 'len_Au': len(info['Au']),
            'Av': info['Av'], 'len_Av': len(info['Av']),
            'd': info['d'], 'hidden': info['hidden'],
            'n_layers': info['n_layers'], 'z_dim': info['z_dim'],
            'stage': info.get('stage'), 'channel': info.get('channel'),
        }
    for name, p in S2_CKPTS.items():
        if not os.path.exists(p):
            print(f'[skip-s2] {name}: file not found')
            continue
        info = load_npd_checkpoint(p)
        results['checkpoints_tested'][name] = {
            'Au': info['Au'], 'len_Au': len(info['Au']),
            'Av': info['Av'], 'len_Av': len(info['Av']),
            'd': info['d'], 'hidden': info['hidden'],
            'n_layers': info['n_layers'], 'z_dim': info['z_dim'],
            'stage': info.get('stage'), 'channel': info.get('channel'),
        }

    def _save():
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # ─── SC baseline at target config (ku=15, kv=29, Class C design) ──────
    print('\n=== SC baseline (Class C, ku=15, kv=29) ===')
    Au_sc, Av_sc, _, _, _, _ = load_class_c_design('gmac', n_log, snr_db=SNR_DB, ku=15, kv=29)
    Au_sc = sorted(Au_sc); Av_sc = sorted(Av_sc)
    print(f'  Au_sc: {Au_sc}')
    print(f'  Av_sc: {Av_sc}')
    r_sc = eval_sc(Au_sc, Av_sc, n_cw=N_CW)
    ci = wilson_ci(r_sc['errs_total'], r_sc['n_cw'])
    r_sc['bler'] = r_sc['errs_total'] / r_sc['n_cw']
    r_sc['ci_lo'] = ci[0]; r_sc['ci_hi'] = ci[1]
    r_sc['Au'] = Au_sc; r_sc['Av'] = Av_sc
    results['sc_baseline']['analytical_sc_classC_genie_design'] = r_sc
    print(f'  SC BLER = {r_sc["bler"]:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
          f'u={r_sc["errs_u"]} v={r_sc["errs_v"]} tot={r_sc["errs_total"]}/{r_sc["n_cw"]}')
    _save()

    # ─── Stage 1 only evaluations ─────────────────────────────────────────
    print('\n=== Stage 1 only evaluations ===')
    for name, p in CKPTS.items():
        if not os.path.exists(p):
            continue
        info = load_npd_checkpoint(p)
        m = build_model(info)
        Au = info['Au']; Av = info['Av']
        # Some checkpoints store their own Av; for stage 1 only, Av doesn't affect U decoding
        # but we use the ckpt's Av for message generation consistency
        if len(Av) < 10:
            # Fall back to SC-picked Av (stage1 tests U, so Av barely matters, but pad it)
            _, Av_sc_full, _, _, _, _ = load_class_c_design('gmac', n_log, snr_db=SNR_DB, ku=15, kv=29)
            Av = sorted(Av_sc_full)
        errs, total, t = eval_stage1_only(m, Au, Av, n_cw=N_CW)
        bler = errs / total
        ci = wilson_ci(errs, total)
        results['stage1_only'][name] = {
            'ckpt': name, 'Au': Au, 'len_Au': len(Au),
            'errs_u': errs, 'n_cw': total, 'bler': bler,
            'ci_lo': ci[0], 'ci_hi': ci[1], 'time_s': t,
        }
        print(f'  [{name}] U-BLER = {bler:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
              f'({errs}/{total}) t={t}s')
        _save()

    # ─── Chained evaluations ──────────────────────────────────────────────
    print('\n=== Chained evaluations ===')
    # Find all valid (s1, s2) pairings at N=64.
    # The canonical pair is curriculum s1+s2.
    # The older p3/p1 checkpoints don't have a matched s2.
    s1_info = load_npd_checkpoint(CKPTS['curriculum_gmac_c_s1_N64_best.pt'])
    s2_info = load_npd_checkpoint(S2_CKPTS['curriculum_gmac_c_s2_N64_best.pt'])
    m1 = build_model(s1_info); m2 = build_model(s2_info)
    Au = s1_info['Au']; Av = s2_info['Av']
    r = eval_chained(m1, m2, Au, Av, n_cw=N_CW)
    bler = r['errs_total'] / r['n_cw']
    ci = wilson_ci(r['errs_total'], r['n_cw'])
    r['bler'] = bler; r['ci_lo'] = ci[0]; r['ci_hi'] = ci[1]
    r['s1_ckpt'] = 'curriculum_gmac_c_s1_N64_best.pt'
    r['s2_ckpt'] = 'curriculum_gmac_c_s2_N64_best.pt'
    r['Au'] = Au; r['Av'] = Av
    results['chained']['curriculum_s1_s2'] = r
    print(f'  [curriculum s1+s2] chained BLER = {bler:.4f} '
          f'[{ci[0]:.4f},{ci[1]:.4f}]  u_err={r["errs_u"]} v_err={r["errs_v"]} '
          f'v_given_u_ok={r["errs_v_given_u_ok"]}')
    _save()

    # Also test: stage 1 only using curriculum checkpoint
    # (already done above, but re-save for clarity)

    # Also test: Old-style claim reconstruction
    # Run "npd_design_p3_N64" stage1 only on its own Au, and compare to SC
    # on that same Au rate (15).
    print('\n=== Reconstruction of old "0.017" claim ===')
    p = CKPTS['npd_design_p3_N64_best.pt']
    if os.path.exists(p):
        info = load_npd_checkpoint(p)
        m = build_model(info)
        Au = info['Au']; Av = info['Av']
        # NPD-picked Av is identical to SC design's Av here
        errs, total, t = eval_stage1_only(m, Au, Av, n_cw=N_CW)
        bler = errs / total
        ci = wilson_ci(errs, total)
        results['notes'].append(
            f'npd_design_p3_N64 stage1-only U-BLER on its NPD-chosen frozen set '
            f'(Au={len(Au)} bits): {bler:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
            f'({errs}/{total}). This is the "0.017 number" from old eval.'
        )
        print(f'  npd_design_p3 (stage1 only, NPD design) BLER={bler:.4f}')
        _save()

    # Final save
    _save()
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
