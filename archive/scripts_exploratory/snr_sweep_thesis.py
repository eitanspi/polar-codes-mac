#!/usr/bin/env python3
"""
snr_sweep_thesis.py — BLER-vs-SNR sweeps for thesis curves.

Runs three config sweeps at SNR ∈ {4, 5, 6, 7, 8} dB:
  (1) Chained NPD (BiGRU) on ISI-MAC h=0.3  + chained trellis SC baseline
      at N ∈ {16, 32, 64}
  (2) NCG on GMAC Class B + analytical SC baseline
      at N ∈ {32, 64, 128}
  (3) Chained NPD on GMAC Class C (curriculum checkpoints) + analytical SC
      at N ∈ {64, 128}

No training. All checkpoints must exist. Outputs one JSON per config to
results/snr_sweep/.
"""
from __future__ import annotations
import os
import sys
import math
import json
import time
import argparse
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import torch  # noqa: E402
torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SNR_LIST = [4.0, 5.0, 6.0, 7.0, 8.0]
OUT_DIR = os.path.join(_ROOT, 'results', 'snr_sweep')
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Wilson CI helper ──────────────────────────────────────────────────────

def wilson_ci(errs: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = errs / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def compute_n_cw(expected_bler_hint: float) -> int:
    """Pick codeword count based on expected BLER (more for lower)."""
    if expected_bler_hint is None:
        return 1000
    if expected_bler_hint > 0.05:
        return 1000
    if expected_bler_hint > 0.01:
        return 1500
    return 2000


# ─── (1) ISI-MAC — chained NPD + chained trellis SC ────────────────────────

def run_isi_mac_sweep(Ns=(16, 32, 64), snrs=SNR_LIST, h=0.3,
                     max_cw=1500, min_cw=500):
    """Chained NPD on ISI-MAC, plus chained trellis SC baseline."""
    from polar.channels_memory import ISIMAC
    from polar.encoder import polar_encode_batch, bit_reversal_perm
    from polar.design_mc import design_from_file
    from polar.decoder_trellis_mac_chained import (
        decode_stage1_u, decode_stage2_v,
    )
    from neural.npd_memory_mac import ChainedNPD_MAC

    # (ku, kv) per training runs
    RATES = {16: (4, 7), 32: (7, 15), 64: (15, 29)}
    SNR_DB_DESIGN = 6.0

    out = {}  # keyed by (decoder, N)

    # For each N, load model, then sweep SNRs
    for N in Ns:
        ku, kv = RATES[N]
        n = int(math.log2(N))
        # Design: GMAC_C @ 6 dB (same proxy the training used)
        design_path = os.path.join(
            _ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB_DESIGN)}dB.npz')
        Au_list, Av_list, fu_1idx, fv_1idx, _, _, _ = design_from_file(
            design_path, n, ku, kv)
        Au = sorted(Au_list)
        Av = sorted(Av_list)
        fu_set = {p - 1 for p in fu_1idx.keys()}
        fv_set = {p - 1 for p in fv_1idx.keys()}

        # Load BiGRU chained NPD
        model = ChainedNPD_MAC(d=16, hidden=64, n_layers=2,
                               encoder_type='bigru', gru_layers=1)
        s1 = torch.load(os.path.join(
            _ROOT, 'class_c_npd/results/npd_memory_mac',
            f'isi_mac_bigru_L1_s1_N{N}_best.pt'),
            weights_only=False, map_location='cpu')
        s2 = torch.load(os.path.join(
            _ROOT, 'class_c_npd/results/npd_memory_mac',
            f'isi_mac_bigru_L1_s2_N{N}_best.pt'),
            weights_only=False, map_location='cpu')
        model.stage1.load_state_dict(s1['state_dict'])
        model.stage2.load_state_dict(s2['state_dict'])
        model.stage1.eval()
        model.stage2.eval()
        br = bit_reversal_perm(n)
        br_t = torch.from_numpy(br.copy()).long()

        npd_rec = {
            'config': f'chained_npd_isi_mac_N{N}',
            'N': N, 'channel': f'ISIMAC h={h}',
            'path': 'class_c', 'decoder': 'chained_npd_bigru',
            'ku': ku, 'kv': kv, 'sweep': {}
        }
        sc_rec = {
            'config': f'chained_trellis_sc_isi_mac_N{N}',
            'N': N, 'channel': f'ISIMAC h={h}',
            'path': 'class_c', 'decoder': 'chained_trellis_sc',
            'ku': ku, 'kv': kv, 'sweep': {}
        }

        # Time-budget-aware cw count. Chained trellis SC on N=64 is ~5 ms/cw,
        # on N=16 ~1 ms. NPD should be similar or faster.
        sc_cw_per_N = {16: 1500, 32: 1500, 64: 1000}
        npd_cw_per_N = {16: 1500, 32: 1500, 64: 1000}
        sc_cw = sc_cw_per_N.get(N, 800)
        npd_cw = npd_cw_per_N.get(N, 800)

        for snr_db in snrs:
            sigma2 = 10.0 ** (-snr_db / 10.0)
            channel = ISIMAC(sigma2=sigma2, h=h)

            # --- Chained NPD ---
            t0 = time.time()
            rng = np.random.default_rng(20260416 + int(snr_db * 100) + N)
            errs = 0
            total = 0
            batch = 16
            with torch.no_grad():
                while total < npd_cw:
                    actual = min(batch, npd_cw - total)
                    # Generate messages
                    u_msg = np.zeros((actual, N), dtype=np.int8)
                    v_msg = np.zeros((actual, N), dtype=np.int8)
                    for p in Au:
                        u_msg[:, p - 1] = rng.integers(0, 2, actual)
                    for p in Av:
                        v_msg[:, p - 1] = rng.integers(0, 2, actual)
                    x = polar_encode_batch(u_msg.astype(int))
                    y = polar_encode_batch(v_msg.astype(int))
                    z = channel.sample_batch(x, y).astype(np.float32)
                    z_t = torch.from_numpy(z)
                    # Stage 1
                    emb1 = model.stage1.encode_channel(z_t)
                    emb1_npd = emb1[:, br_t, :]
                    u_hat = model.stage1.tree.decode(emb1_npd, fu_set)
                    u_hat_np = u_hat.numpy().astype(int)
                    x_hat = polar_encode_batch(u_hat_np)
                    # Stage 2
                    side = torch.from_numpy((1.0 - 2.0 * x_hat.astype(np.float32))).unsqueeze(-1)
                    emb2 = model.stage2.encode_channel(z_t, side=side)
                    emb2_npd = emb2[:, br_t, :]
                    v_hat = model.stage2.tree.decode(emb2_npd, fv_set)
                    for i in range(actual):
                        u_wrong = any(int(u_hat[i, p - 1].item()) != int(u_msg[i, p - 1]) for p in Au)
                        v_wrong = any(int(v_hat[i, p - 1].item()) != int(v_msg[i, p - 1]) for p in Av)
                        if u_wrong or v_wrong:
                            errs += 1
                    total += actual
            t_npd = time.time() - t0
            bler_npd = errs / total
            ci_lo, ci_hi = wilson_ci(errs, total)
            npd_rec['sweep'][str(int(snr_db))] = {
                'bler': bler_npd, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'n_cw': total, 'errs': errs, 'time_s': round(t_npd, 2),
                'sigma2': sigma2,
            }
            print(f'  [ISI N={N} NPD SNR={snr_db}] BLER={bler_npd:.4f} '
                  f'[{ci_lo:.4f},{ci_hi:.4f}] n={total} t={t_npd:.1f}s')

            # --- Chained trellis SC baseline ---
            t0 = time.time()
            rng = np.random.default_rng(20260416 + int(snr_db * 100) + N + 7)
            errs = 0
            for _ in range(sc_cw):
                u_msg = np.zeros(N, dtype=int)
                v_msg = np.zeros(N, dtype=int)
                for p in Au:
                    u_msg[p - 1] = rng.integers(0, 2)
                for p in Av:
                    v_msg[p - 1] = rng.integers(0, 2)
                x = polar_encode_batch(u_msg.reshape(1, -1))[0]
                y = polar_encode_batch(v_msg.reshape(1, -1))[0]
                z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
                u_hat = decode_stage1_u(z, N, fu_1idx, channel)
                v_hat = decode_stage2_v(z, u_hat, N, fv_1idx, channel)
                u_wrong = any(int(u_hat[p - 1]) != int(u_msg[p - 1]) for p in Au)
                v_wrong = any(int(v_hat[p - 1]) != int(v_msg[p - 1]) for p in Av)
                if u_wrong or v_wrong:
                    errs += 1
            t_sc = time.time() - t0
            bler_sc = errs / sc_cw
            ci_lo_sc, ci_hi_sc = wilson_ci(errs, sc_cw)
            sc_rec['sweep'][str(int(snr_db))] = {
                'bler': bler_sc, 'ci_lo': ci_lo_sc, 'ci_hi': ci_hi_sc,
                'n_cw': sc_cw, 'errs': errs, 'time_s': round(t_sc, 2),
                'sigma2': sigma2,
            }
            print(f'  [ISI N={N} SC  SNR={snr_db}] BLER={bler_sc:.4f} '
                  f'[{ci_lo_sc:.4f},{ci_hi_sc:.4f}] n={sc_cw} t={t_sc:.1f}s')

            # Incremental save
            _save(npd_rec)
            _save(sc_rec)

        out[f'npd_N{N}'] = npd_rec
        out[f'sc_N{N}'] = sc_rec
    return out


# ─── (2) GMAC Class B — NCG vs analytical SC ───────────────────────────────

def run_gmac_classB_sweep(Ns=(32, 64, 128), snrs=SNR_LIST):
    from polar.encoder import polar_encode, polar_encode_batch
    from polar.decoder import decode_single
    from polar.channels import GaussianMAC
    from polar.design import make_path
    from neural.neural_scl import SimpleMLP_Gmac

    # Historical per-N rates used in nn_scl_full_comparison (matches NCG training)
    GMAC_B_RATES = {32: (15, 15), 64: (31, 31), 128: (62, 62)}
    out = {}
    for N in Ns:
        n = int(math.log2(N))
        ku, kv = GMAC_B_RATES.get(N, (N // 2 - 1, N // 2 - 1))
        # Design @ SNR=6dB (the checkpoint's training SNR)
        dp = os.path.join(_ROOT, 'designs', f'gmac_B_n{n}_snr6dB.npz')
        d = np.load(dp)
        su = np.argsort(d['u_error_rates'])
        sv = np.argsort(d['v_error_rates'])
        Au = sorted([int(i + 1) for i in su[:ku]])
        Av = sorted([int(i + 1) for i in sv[:kv]])
        all_pos = set(range(1, N + 1))
        fu = {p: 0 for p in sorted(all_pos - set(Au))}
        fv = {p: 0 for p in sorted(all_pos - set(Av))}
        b = make_path(N, N // 2)  # Class B

        # Load NCG
        model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
        sd = torch.load(os.path.join(_ROOT, 'saved_models',
                                     f'ncg_gmac_mlp_N{N}.pt'),
                        map_location='cpu', weights_only=True)
        fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
        model.load_state_dict(fixed, strict=False)
        model.eval()

        ncg_rec = {
            'config': f'ncg_gmac_classB_N{N}',
            'N': N, 'channel': 'GaussianMAC',
            'path': 'class_b', 'decoder': 'ncg_mlp',
            'ku': ku, 'kv': kv, 'sweep': {}
        }
        sc_rec = {
            'config': f'sc_gmac_classB_N{N}',
            'N': N, 'channel': 'GaussianMAC',
            'path': 'class_b', 'decoder': 'analytical_sc',
            'ku': ku, 'kv': kv, 'sweep': {}
        }

        # Codeword counts. Scale down a bit for larger N.
        ncg_cw_per_N = {32: 2000, 64: 2000, 128: 1500}
        sc_cw_per_N = {32: 2000, 64: 2000, 128: 1500}

        for snr_db in snrs:
            sigma2 = 10.0 ** (-snr_db / 10.0)
            channel = GaussianMAC(sigma2=sigma2)

            # --- NCG ---
            t0 = time.time()
            rng = np.random.default_rng(20260416 + int(snr_db * 100) + N)
            errs = 0
            total = 0
            bs = max(8, min(64, 256 // (N // 16)))
            n_cw = ncg_cw_per_N.get(N, 1500)
            with torch.no_grad():
                while total < n_cw:
                    actual = min(bs, n_cw - total)
                    uf = np.zeros((actual, N), dtype=int)
                    vf = np.zeros((actual, N), dtype=int)
                    for p in Au:
                        uf[:, p - 1] = rng.integers(0, 2, actual)
                    for p in Av:
                        vf[:, p - 1] = rng.integers(0, 2, actual)
                    xf = polar_encode_batch(uf)
                    yf = polar_encode_batch(vf)
                    zf = channel.sample_batch(xf, yf)
                    zt = torch.from_numpy(zf.astype(np.float32)).float()
                    _, _, uh, vh, _ = model(zt, b, fu, fv)
                    for i in range(actual):
                        e = any(int(uh[p][i].item()) != uf[i, p - 1] for p in Au if p in uh) or \
                            any(int(vh[p][i].item()) != vf[i, p - 1] for p in Av if p in vh)
                        if e:
                            errs += 1
                    total += actual
            t_ncg = time.time() - t0
            bler_ncg = errs / total
            ci_lo, ci_hi = wilson_ci(errs, total)
            ncg_rec['sweep'][str(int(snr_db))] = {
                'bler': bler_ncg, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'n_cw': total, 'errs': errs, 'time_s': round(t_ncg, 2),
                'sigma2': sigma2,
            }
            print(f'  [GMAC-B N={N} NCG SNR={snr_db}] BLER={bler_ncg:.4f} '
                  f'[{ci_lo:.4f},{ci_hi:.4f}] n={total} t={t_ncg:.1f}s')

            # --- Analytical SC ---
            t0 = time.time()
            rng = np.random.default_rng(20260416 + int(snr_db * 100) + N + 7)
            errs = 0
            n_cw_sc = sc_cw_per_N.get(N, 1500)
            for _ in range(n_cw_sc):
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
                u_dec, v_dec = decode_single(N, z, b, fu, fv, channel,
                                              log_domain=True)
                if any(u_dec[p - 1] != uf[p - 1] for p in Au) or \
                   any(v_dec[p - 1] != vf[p - 1] for p in Av):
                    errs += 1
            t_sc = time.time() - t0
            bler_sc = errs / n_cw_sc
            ci_lo_s, ci_hi_s = wilson_ci(errs, n_cw_sc)
            sc_rec['sweep'][str(int(snr_db))] = {
                'bler': bler_sc, 'ci_lo': ci_lo_s, 'ci_hi': ci_hi_s,
                'n_cw': n_cw_sc, 'errs': errs, 'time_s': round(t_sc, 2),
                'sigma2': sigma2,
            }
            print(f'  [GMAC-B N={N} SC  SNR={snr_db}] BLER={bler_sc:.4f} '
                  f'[{ci_lo_s:.4f},{ci_hi_s:.4f}] n={n_cw_sc} t={t_sc:.1f}s')
            _save(ncg_rec)
            _save(sc_rec)

        out[f'ncg_N{N}'] = ncg_rec
        out[f'sc_N{N}'] = sc_rec
    return out


# ─── (3) GMAC Class C — chained NPD vs analytical SC ───────────────────────

def run_gmac_classC_sweep(Ns=(64, 128), snrs=SNR_LIST):
    """Chained NPD (NPDSingleUser, curriculum checkpoints) on GMAC Class C.

    Uses the Class C corner-rate path. Baseline is analytical SC on Class C.
    """
    from polar.encoder import polar_encode_batch, bit_reversal_perm, polar_encode
    from polar.decoder import decode_single
    from polar.channels import GaussianMAC
    from polar.design import make_path
    from class_c_npd.models.npd_single_user import NPDSingleUser
    from class_c_npd.channels.frozen_sets import load_class_c_design
    from class_c_npd.channels.mac_channel import build_channel as build_c_channel

    out = {}
    for N in Ns:
        n = int(math.log2(N))
        # Load stage 1 + stage 2
        s1p = os.path.join(_ROOT, 'class_c_npd/results',
                          f'curriculum_gmac_c_s1_N{N}_best.pt')
        s2p = os.path.join(_ROOT, 'class_c_npd/results',
                          f'curriculum_gmac_c_s2_N{N}_best.pt')
        if not (os.path.exists(s1p) and os.path.exists(s2p)):
            print(f'  [GMAC-C N={N}] checkpoints missing, skipping')
            continue
        ck1 = torch.load(s1p, weights_only=False, map_location='cpu')
        ck2 = torch.load(s2p, weights_only=False, map_location='cpu')
        # Use the checkpoint's own Au/Av (these were fixed during training)
        Au = sorted(ck1['Au'])
        Av = sorted(ck2['Av'])
        ku = len(Au); kv = len(Av)
        fu_set = {p - 1 for p in range(1, N + 1) if p not in Au}
        fv_set = {p - 1 for p in range(1, N + 1) if p not in Av}
        all_pos = set(range(1, N + 1))
        fu_1idx = {p: 0 for p in sorted(all_pos - set(Au))}
        fv_1idx = {p: 0 for p in sorted(all_pos - set(Av))}
        b = make_path(N, N)  # Class C corner
        m1 = NPDSingleUser(d=ck1['d'], hidden=ck1['hidden'],
                           n_layers=ck1['n_layers'], z_dim=ck1['z_dim'])
        m1.load_state_dict(ck1['state_dict']); m1.eval()
        m2 = NPDSingleUser(d=ck2['d'], hidden=ck2['hidden'],
                           n_layers=ck2['n_layers'], z_dim=ck2['z_dim'])
        m2.load_state_dict(ck2['state_dict']); m2.eval()
        br = bit_reversal_perm(n)
        br_t = torch.from_numpy(br.copy()).long()

        npd_rec = {
            'config': f'chained_npd_gmac_classC_N{N}',
            'N': N, 'channel': 'GaussianMAC',
            'path': 'class_c', 'decoder': 'chained_npd_mlp',
            'ku': ku, 'kv': kv, 'sweep': {}
        }
        sc_rec = {
            'config': f'sc_gmac_classC_N{N}',
            'N': N, 'channel': 'GaussianMAC',
            'path': 'class_c', 'decoder': 'analytical_sc',
            'ku': ku, 'kv': kv, 'sweep': {}
        }

        npd_cw_per_N = {64: 2000, 128: 1500}
        sc_cw_per_N = {64: 2000, 128: 1500}

        for snr_db in snrs:
            sigma2 = 10.0 ** (-snr_db / 10.0)
            channel_c = build_c_channel('gmac', sigma2=sigma2)
            channel = GaussianMAC(sigma2=sigma2)

            # --- NPD chained ---
            t0 = time.time()
            rng = np.random.default_rng(20260416 + int(snr_db * 100) + N)
            errs = 0
            total = 0
            batch = 16
            n_cw = npd_cw_per_N.get(N, 1500)
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
                    z = channel_c.sample_z(x_phys.astype(int), y_phys.astype(int))

                    # Stage 1: decode U on marginal
                    features1 = channel_c.stage1_features(z)
                    features1_npd = (features1[..., br] if features1.ndim == 2
                                     else features1[:, br, :])
                    ft1 = torch.from_numpy(features1_npd).float()
                    if ft1.dim() == 2:
                        ft1 = ft1.unsqueeze(-1)
                    emb1 = m1.encode_channel(ft1)
                    u_hat = m1.decode(emb1, fu_set)
                    u_hat_np = u_hat.numpy().astype(int)
                    x_hat = polar_encode_batch(u_hat_np)

                    features2 = channel_c.stage2_features(z, x_hat.astype(int))
                    features2_npd = (features2[..., br] if features2.ndim == 2
                                     else features2[:, br, :])
                    ft2 = torch.from_numpy(features2_npd).float()
                    if ft2.dim() == 2:
                        ft2 = ft2.unsqueeze(-1)
                    emb2 = m2.encode_channel(ft2)
                    v_hat = m2.decode(emb2, fv_set)
                    for i in range(actual):
                        u_wrong = any(int(u_hat[i, p - 1].item()) != int(u_msg[i, p - 1]) for p in Au)
                        v_wrong = any(int(v_hat[i, p - 1].item()) != int(v_msg[i, p - 1]) for p in Av)
                        if u_wrong or v_wrong:
                            errs += 1
                    total += actual
            t_npd = time.time() - t0
            bler_npd = errs / total
            ci_lo, ci_hi = wilson_ci(errs, total)
            npd_rec['sweep'][str(int(snr_db))] = {
                'bler': bler_npd, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'n_cw': total, 'errs': errs, 'time_s': round(t_npd, 2),
                'sigma2': sigma2,
            }
            print(f'  [GMAC-C N={N} NPD SNR={snr_db}] BLER={bler_npd:.4f} '
                  f'[{ci_lo:.4f},{ci_hi:.4f}] n={total} t={t_npd:.1f}s')

            # --- Analytical SC baseline ---
            t0 = time.time()
            rng = np.random.default_rng(20260416 + int(snr_db * 100) + N + 7)
            errs = 0
            n_cw_sc = sc_cw_per_N.get(N, 1500)
            for _ in range(n_cw_sc):
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
                u_dec, v_dec = decode_single(N, z, b, fu_1idx, fv_1idx,
                                             channel, log_domain=True)
                if any(u_dec[p - 1] != uf[p - 1] for p in Au) or \
                   any(v_dec[p - 1] != vf[p - 1] for p in Av):
                    errs += 1
            t_sc = time.time() - t0
            bler_sc = errs / n_cw_sc
            ci_lo_s, ci_hi_s = wilson_ci(errs, n_cw_sc)
            sc_rec['sweep'][str(int(snr_db))] = {
                'bler': bler_sc, 'ci_lo': ci_lo_s, 'ci_hi': ci_hi_s,
                'n_cw': n_cw_sc, 'errs': errs, 'time_s': round(t_sc, 2),
                'sigma2': sigma2,
            }
            print(f'  [GMAC-C N={N} SC  SNR={snr_db}] BLER={bler_sc:.4f} '
                  f'[{ci_lo_s:.4f},{ci_hi_s:.4f}] n={n_cw_sc} t={t_sc:.1f}s')
            _save(npd_rec)
            _save(sc_rec)

        out[f'npd_N{N}'] = npd_rec
        out[f'sc_N{N}'] = sc_rec
    return out


# ─── IO ────────────────────────────────────────────────────────────────────

def _save(rec):
    fp = os.path.join(OUT_DIR, rec['config'] + '.json')
    with open(fp, 'w') as f:
        json.dump(rec, f, indent=2)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', nargs='+', default=None,
                   choices=['isi', 'gmacB', 'gmacC', 'all'])
    p.add_argument('--Ns_isi', nargs='+', type=int, default=[16, 32, 64])
    p.add_argument('--Ns_gmacB', nargs='+', type=int, default=[32, 64, 128])
    p.add_argument('--Ns_gmacC', nargs='+', type=int, default=[64, 128])
    p.add_argument('--snrs', nargs='+', type=float, default=SNR_LIST)
    args = p.parse_args()

    only = set(args.only) if args.only else {'isi', 'gmacB', 'gmacC'}
    if 'all' in only:
        only = {'isi', 'gmacB', 'gmacC'}

    t_total = time.time()
    if 'isi' in only:
        print('\n=== ISI-MAC sweep ===')
        run_isi_mac_sweep(args.Ns_isi, args.snrs)
    if 'gmacB' in only:
        print('\n=== GMAC Class B sweep ===')
        run_gmac_classB_sweep(args.Ns_gmacB, args.snrs)
    if 'gmacC' in only:
        print('\n=== GMAC Class C sweep ===')
        run_gmac_classC_sweep(args.Ns_gmacC, args.snrs)
    print(f'\nTotal wall time: {(time.time() - t_total) / 60:.1f} min')
    print(f'Results in {OUT_DIR}')


if __name__ == '__main__':
    main()
