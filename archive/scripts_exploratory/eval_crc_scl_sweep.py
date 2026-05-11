#!/usr/bin/env python3
"""
eval_crc_scl_sweep.py — Traditional (analytical) CRC-aided SCL L=4 sweep.

For every (channel, class, N) config, evaluate:
  1. SC (greedy) BLER
  2. SCL L=4 BLER (analytical, best-path-metric)
  3. CRC-SCL L=4 BLER (analytical, CRC-8 on U info bits)

Uses the analytical decoder_scl.py and decoder.py, not neural models.

Output: results/crc_scl_sweep/{channel}_{class}_crc_scl.json
"""

import os, sys, json, time, math
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import torch
torch.set_num_threads(4)

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.decoder import decode_single, build_log_W_leaf, _detect_path_i
from polar.decoder import _circ_conv_batch, _norm_prod_batch
from polar.decoder_scl import (
    _BatchCompGraph, _set_leaves_batched,
    _decode_extreme_u_first, _decode_extreme_v_first,
    build_log_W_leaf_batch,
)
from polar.channels import GaussianMAC, BEMAC, ABNMAC
from polar.design import make_path

_NEG_INF = -np.inf
_LOG_HALF = np.log(0.5)
_LOG_QUARTER = np.log(0.25)

# ─── CRC-8 ────────────────────────────────────────────────────────────────────

CRC_POLY = 0x107
CRC_BITS = 8

def compute_crc8(message_bits):
    if len(message_bits) == 0:
        return [0] * CRC_BITS
    msg = int(''.join(str(int(b)) for b in message_bits), 2) << CRC_BITS
    for i in range(len(message_bits)):
        if msg & (1 << (len(message_bits) + CRC_BITS - 1 - i)):
            msg ^= (CRC_POLY << (len(message_bits) - 1 - i))
    crc = msg & ((1 << CRC_BITS) - 1)
    return [(crc >> (CRC_BITS - 1 - i)) & 1 for i in range(CRC_BITS)]


def wilson_ci(k, n, z=1.96):
    """Wilson 95% confidence interval."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - halfwidth), min(1, center + halfwidth))


# ─── Channel setup ─────────────────────────────────────────────────────────────

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

def make_channel(name):
    if name == 'gmac':
        return GaussianMAC(sigma2=SIGMA2)
    elif name == 'bemac':
        return BEMAC()
    elif name == 'abnmac':
        return ABNMAC()
    raise ValueError(name)


# ─── Design loading ────────────────────────────────────────────────────────────

def load_design(channel_name, cls, N, ku, kv):
    n = int(math.log2(N))
    dp = os.path.join(BASE, 'designs', f'{channel_name}_{cls}_n{n}')
    if channel_name == 'gmac':
        dp += f'_snr{int(SNR_DB)}dB'
    dp += '.npz'
    if not os.path.exists(dp):
        raise FileNotFoundError(f"Design file not found: {dp}")
    # Use design_from_file which has proper polar tiebreak rule
    from polar.design_mc import design_from_file
    Au, Av, fu, fv, pe_u, pe_v, path_i = design_from_file(dp, n, ku, kv)
    return Au, Av, fu, fv


# ─── SCL returning all candidates ──────────────────────────────────────────────

def _decode_tensor_scl_all_paths(N, log_W, b, frozen_u, frozen_v, L):
    """
    Tensor-based SCL decoder (O(L * N log N)) for arbitrary paths.
    Returns ALL L active paths sorted by metric (not just the best).
    """
    n = N.bit_length() - 1
    max_paths = 2 * L

    graph = _BatchCompGraph(n, log_W, max_paths)
    PM = np.full(max_paths, _NEG_INF, dtype=np.float64)
    PM[0] = 0.0
    active = np.zeros(max_paths, dtype=bool)
    active[0] = True
    u_bits = np.zeros((max_paths, N + 1), dtype=np.int8)
    v_bits = np.zeros((max_paths, N + 1), dtype=np.int8)
    u_decided = np.zeros((max_paths, N + 1), dtype=bool)
    v_decided = np.zeros((max_paths, N + 1), dtype=bool)

    i_u = 0
    i_v = 0

    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u; frozen_dict = frozen_u
        else:
            i_v += 1; i_t = i_v; frozen_dict = frozen_v

        leaf_edge = i_t + N - 1
        target_vertex = leaf_edge >> 1
        is_left = (leaf_edge & 1 == 0)

        graph.step_to(target_vertex)

        leaf_view = graph._e(leaf_edge)
        temp = leaf_view[:, 0].copy()
        if is_left:
            graph.calc_left(target_vertex)
        else:
            graph.calc_right(target_vertex)
        leaf_view = graph._e(leaf_edge)
        top_down = leaf_view[:, 0]
        combined = _norm_prod_batch(top_down, temp)

        is_frozen = i_t in frozen_dict

        if is_frozen:
            bit = frozen_dict[i_t]
            aidx = np.where(active)[0]
            if gamma == 0:
                pb = np.logaddexp(combined[aidx, bit, 0], combined[aidx, bit, 1])
            else:
                pb = np.logaddexp(combined[aidx, 0, bit], combined[aidx, 1, bit])
            PM[aidx] += pb
            if gamma == 0:
                u_bits[aidx, i_t] = bit; u_decided[aidx, i_t] = True
            else:
                v_bits[aidx, i_t] = bit; v_decided[aidx, i_t] = True
            _set_leaves_batched(graph, leaf_edge, i_t, aidx,
                                u_bits, v_bits, u_decided, v_decided)
        else:
            aidx = np.where(active)[0]
            if gamma == 0:
                pb0 = np.logaddexp(combined[aidx, 0, 0], combined[aidx, 0, 1])
                pb1 = np.logaddexp(combined[aidx, 1, 0], combined[aidx, 1, 1])
            else:
                pb0 = np.logaddexp(combined[aidx, 0, 0], combined[aidx, 1, 0])
                pb1 = np.logaddexp(combined[aidx, 0, 1], combined[aidx, 1, 1])

            met0 = PM[aidx] + pb0
            met1 = PM[aidx] + pb1
            n_active = len(aidx)
            all_mets = np.empty(2 * n_active, dtype=np.float64)
            all_srcs = np.empty(2 * n_active, dtype=np.intp)
            all_bits_arr = np.empty(2 * n_active, dtype=np.int8)
            all_mets[0::2] = met0; all_mets[1::2] = met1
            all_srcs[0::2] = aidx; all_srcs[1::2] = aidx
            all_bits_arr[0::2] = 0; all_bits_arr[1::2] = 1
            n_keep = min(len(all_mets), L)
            order = np.argsort(-all_mets, kind='stable')
            top_idx = order[:n_keep]

            keep_mets = all_mets[top_idx]
            keep_srcs = all_srcs[top_idx]
            keep_bits = all_bits_arr[top_idx]

            src_used = set()
            inactive_iter = iter(list(np.where(~active)[0]))
            assignments = []
            for k in range(n_keep):
                src = int(keep_srcs[k])
                if src not in src_used:
                    src_used.add(src)
                    assignments.append((src, src, keep_mets[k], keep_bits[k], False))
                else:
                    dst = next(inactive_iter)
                    assignments.append((dst, src, keep_mets[k], keep_bits[k], True))

            for dst, src, _, _, need_copy in assignments:
                if need_copy:
                    graph.copy_path(dst, src)
                    u_bits[dst] = u_bits[src]
                    v_bits[dst] = v_bits[src]
                    u_decided[dst] = u_decided[src]
                    v_decided[dst] = v_decided[src]

            active[:] = False
            for dst, _, met, bit, _ in assignments:
                PM[dst] = met; active[dst] = True
                if gamma == 0:
                    u_bits[dst, i_t] = bit; u_decided[dst, i_t] = True
                else:
                    v_bits[dst, i_t] = bit; v_decided[dst, i_t] = True

            surv = np.array([a[0] for a in assignments], dtype=np.intp)
            _set_leaves_batched(graph, leaf_edge, i_t, surv,
                                u_bits, v_bits, u_decided, v_decided)

        aidx = np.where(active)[0]
        if len(aidx) > 0:
            max_pm = np.max(PM[aidx])
            if np.isfinite(max_pm):
                PM[aidx] -= max_pm

    # Return ALL active paths sorted by metric
    aidx = np.where(active)[0]
    order = np.argsort(-PM[aidx])
    sorted_idx = aidx[order]
    paths = []
    for l in sorted_idx:
        paths.append({
            'u_dec': u_bits[l, 1:N+1].tolist(),
            'v_dec': v_bits[l, 1:N+1].tolist(),
            'metric': float(PM[l]),
        })
    return paths


def _decode_extreme_u_first_all_paths(N, m, L, log_W, frozen_u, frozen_v):
    """U-first SCL returning all L paths sorted by metric."""
    from polar.decoder_scl import _scl_decode_phase, _v_conditional_logprob, _u_marginal_logprob
    from polar.encoder import polar_encode

    frozen_u_0 = {k - 1: v for k, v in frozen_u.items()}
    frozen_v_0 = {k - 1: v for k, v in frozen_v.items()}
    max_paths = 2 * L

    lp0, lp1 = _u_marginal_logprob(log_W)
    P0, P1, C, PM, u_bits_phase1, active = _scl_decode_phase(
        N, m, L, lp0, lp1, frozen_u_0)

    aidx = np.where(active)[0]
    for l in aidx:
        x_l = np.array(polar_encode(u_bits_phase1[l].tolist()), dtype=np.int8)
        vlp0, vlp1 = _v_conditional_logprob(log_W, x_l)
        P0[l, 0, :N] = vlp0
        P1[l, 0, :N] = vlp1
    P0[:, 1:, :] = _NEG_INF
    P1[:, 1:, :] = _NEG_INF
    C[:, :, :, :] = 0

    from polar.decoder_scl import _calc_P, _update_C, _fork_and_prune

    v_bits = np.zeros((max_paths, N), dtype=np.int8)
    for phi in range(N):
        _calc_P(m, phi, P0, P1, C, m)
        is_frozen = phi in frozen_v_0
        if is_frozen:
            fval = frozen_v_0[phi]
            aidx = np.where(active)[0]
            if fval == 0:
                PM[aidx] += P0[aidx, m, 0]
            else:
                PM[aidx] += P1[aidx, m, 0]
            v_bits[aidx, phi] = fval
            C[aidx, m, 0, phi % 2] = fval
        else:
            _fork_and_prune(phi, P0, P1, C, PM, v_bits, active, L, m,
                            extra_bits_list=[u_bits_phase1])
        _update_C(m, phi, C, m)
        aidx = np.where(active)[0]
        if len(aidx) > 0:
            max_pm = np.max(PM[aidx])
            if max_pm != _NEG_INF:
                PM[aidx] -= max_pm

    aidx = np.where(active)[0]
    order = np.argsort(-PM[aidx])
    sorted_idx = aidx[order]
    paths = []
    for l in sorted_idx:
        paths.append({
            'u_dec': u_bits_phase1[l].tolist(),
            'v_dec': v_bits[l].tolist(),
            'metric': float(PM[l]),
        })
    return paths


def _decode_extreme_v_first_all_paths(N, m, L, log_W, frozen_u, frozen_v):
    """V-first SCL returning all L paths sorted by metric."""
    from polar.decoder_scl import _scl_decode_phase, _u_conditional_logprob, _v_marginal_logprob
    from polar.encoder import polar_encode

    frozen_u_0 = {k - 1: v for k, v in frozen_u.items()}
    frozen_v_0 = {k - 1: v for k, v in frozen_v.items()}
    max_paths = 2 * L

    lp0, lp1 = _v_marginal_logprob(log_W)
    P0, P1, C, PM, v_bits_phase1, active = _scl_decode_phase(
        N, m, L, lp0, lp1, frozen_v_0)

    aidx = np.where(active)[0]
    for l in aidx:
        y_l = np.array(polar_encode(v_bits_phase1[l].tolist()), dtype=np.int8)
        ulp0, ulp1 = _u_conditional_logprob(log_W, y_l)
        P0[l, 0, :N] = ulp0
        P1[l, 0, :N] = ulp1
    P0[:, 1:, :] = _NEG_INF
    P1[:, 1:, :] = _NEG_INF
    C[:, :, :, :] = 0

    from polar.decoder_scl import _calc_P, _update_C, _fork_and_prune

    u_bits = np.zeros((max_paths, N), dtype=np.int8)
    for phi in range(N):
        _calc_P(m, phi, P0, P1, C, m)
        is_frozen = phi in frozen_u_0
        if is_frozen:
            fval = frozen_u_0[phi]
            aidx = np.where(active)[0]
            if fval == 0:
                PM[aidx] += P0[aidx, m, 0]
            else:
                PM[aidx] += P1[aidx, m, 0]
            u_bits[aidx, phi] = fval
            C[aidx, m, 0, phi % 2] = fval
        else:
            _fork_and_prune(phi, P0, P1, C, PM, u_bits, active, L, m,
                            extra_bits_list=[v_bits_phase1])
        _update_C(m, phi, C, m)
        aidx = np.where(active)[0]
        if len(aidx) > 0:
            max_pm = np.max(PM[aidx])
            if max_pm != _NEG_INF:
                PM[aidx] -= max_pm

    aidx = np.where(active)[0]
    order = np.argsort(-PM[aidx])
    sorted_idx = aidx[order]
    paths = []
    for l in sorted_idx:
        paths.append({
            'u_dec': u_bits[l].tolist(),
            'v_dec': v_bits_phase1[l].tolist(),
            'metric': float(PM[l]),
        })
    return paths


def decode_all_paths(N, z, b, frozen_u, frozen_v, channel, L=4):
    """SCL decode returning all L candidates, sorted by metric."""
    log_W = build_log_W_leaf(z, channel)
    path_i = _detect_path_i(N, b)
    m = N.bit_length() - 1
    if path_i == N:
        return _decode_extreme_u_first_all_paths(N, m, L, log_W, frozen_u, frozen_v)
    elif path_i == 0:
        return _decode_extreme_v_first_all_paths(N, m, L, log_W, frozen_u, frozen_v)
    else:
        return _decode_tensor_scl_all_paths(N, log_W, b, frozen_u, frozen_v, L)


# ─── Evaluation ────────────────────────────────────────────────────────────────

def eval_config(channel_name, cls, N, ku, kv, n_cw, seed=42, L=4):
    """Evaluate SC, SCL L=4, CRC-SCL L=4 for one config."""
    channel = make_channel(channel_name)

    # Determine path
    if cls == 'B':
        path_i = N // 2
    elif cls == 'C':
        path_i = N  # U first, then V
    elif cls == 'A':
        path_i = 0  # V first, then U
    else:
        raise ValueError(f"Unknown class: {cls}")
    b = make_path(N, path_i)

    Au, Av, fu, fv = load_design(channel_name, cls, N, ku, kv)

    # CRC positions: last 8 of Au (if ku > 8)
    if ku > CRC_BITS:
        crc_positions = Au[-CRC_BITS:]
        msg_positions = [p for p in Au if p not in crc_positions]
    else:
        crc_positions = []
        msg_positions = list(Au)

    errs_sc = 0
    errs_scl = 0
    errs_crc_scl = 0

    rng = np.random.default_rng(seed)
    t0 = time.time()

    for i in range(n_cw):
        # Generate random info bits
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au:
            uf[p-1] = rng.integers(0, 2)
        for p in Av:
            vf[p-1] = rng.integers(0, 2)

        # Set CRC bits on U
        if crc_positions:
            msg_bits = [uf[p-1] for p in msg_positions]
            crc_vals = compute_crc8(msg_bits)
            for cp, cv in zip(crc_positions, crc_vals):
                uf[cp-1] = cv

        # Encode and transmit
        xf = polar_encode_batch(uf.reshape(1, N))
        yf = polar_encode_batch(vf.reshape(1, N))
        z = channel.sample_batch(xf, yf)[0]
        if hasattr(z, 'tolist'):
            z = z.tolist()

        # SC decode
        u_sc, v_sc = decode_single(N, z, b, fu, fv, channel, log_domain=True)
        err_sc = any(u_sc[p-1] != uf[p-1] for p in Au) or \
                 any(v_sc[p-1] != vf[p-1] for p in Av)
        if err_sc:
            errs_sc += 1

        # SCL L=4 — all candidates
        candidates = decode_all_paths(N, z, b, fu, fv, channel, L=L)

        # Best path (SCL)
        best = candidates[0]
        # For extreme paths, u_dec/v_dec are 0-indexed
        u_scl = best['u_dec']
        v_scl = best['v_dec']
        err_scl = any(u_scl[p-1] != uf[p-1] for p in Au) or \
                  any(v_scl[p-1] != vf[p-1] for p in Av)
        if err_scl:
            errs_scl += 1

        # CRC-SCL: pick first candidate passing CRC on U
        if crc_positions:
            picked = None
            for cand in candidates:
                u_c = cand['u_dec']
                m_bits = [u_c[p-1] for p in msg_positions]
                c_bits = [u_c[p-1] for p in crc_positions]
                if compute_crc8(m_bits) == c_bits:
                    picked = cand
                    break
            if picked is None:
                picked = candidates[0]
            u_crc = picked['u_dec']
            v_crc = picked['v_dec']
        else:
            u_crc = u_scl
            v_crc = v_scl

        err_crc = any(u_crc[p-1] != uf[p-1] for p in Au) or \
                  any(v_crc[p-1] != vf[p-1] for p in Av)
        if err_crc:
            errs_crc_scl += 1

        if (i+1) % 200 == 0 or i+1 == n_cw:
            elapsed = time.time() - t0
            rate = elapsed / (i+1)
            print(f"    {i+1}/{n_cw}  SC={errs_sc/(i+1):.4f}  "
                  f"SCL={errs_scl/(i+1):.4f}  CRC-SCL={errs_crc_scl/(i+1):.4f}  "
                  f"({rate:.2f}s/cw)", flush=True)

    t_total = time.time() - t0
    bler_sc = errs_sc / n_cw
    bler_scl = errs_scl / n_cw
    bler_crc = errs_crc_scl / n_cw

    ci_sc = wilson_ci(errs_sc, n_cw)
    ci_scl = wilson_ci(errs_scl, n_cw)
    ci_crc = wilson_ci(errs_crc_scl, n_cw)

    result = {
        'channel': channel_name, 'class': cls,
        'N': N, 'ku': ku, 'kv': kv, 'L': L,
        'n_cw': n_cw, 'time_s': round(t_total, 1),
        'sc_bler': bler_sc, 'sc_errs': errs_sc,
        'sc_ci': list(ci_sc),
        'scl_bler': bler_scl, 'scl_errs': errs_scl,
        'scl_ci': list(ci_scl),
        'crc_scl_bler': bler_crc, 'crc_scl_errs': errs_crc_scl,
        'crc_scl_ci': list(ci_crc),
        'crc_positions': crc_positions,
        'seed': seed,
    }
    return result


# ─── Configuration ─────────────────────────────────────────────────────────────

CONFIGS = [
    # GMAC Class B
    {'channel': 'gmac', 'cls': 'B', 'N': 32,  'ku': 15, 'kv': 15, 'n_cw': 3000},
    {'channel': 'gmac', 'cls': 'B', 'N': 64,  'ku': 31, 'kv': 31, 'n_cw': 3000},
    {'channel': 'gmac', 'cls': 'B', 'N': 128, 'ku': 62, 'kv': 62, 'n_cw': 3000},
    {'channel': 'gmac', 'cls': 'B', 'N': 256, 'ku': 123,'kv': 123,'n_cw': 2000},

    # GMAC Class C
    {'channel': 'gmac', 'cls': 'C', 'N': 16,  'ku': 4,  'kv': 7,  'n_cw': 5000},
    {'channel': 'gmac', 'cls': 'C', 'N': 32,  'ku': 7,  'kv': 15, 'n_cw': 5000},
    {'channel': 'gmac', 'cls': 'C', 'N': 64,  'ku': 15, 'kv': 29, 'n_cw': 5000},
    {'channel': 'gmac', 'cls': 'C', 'N': 128, 'ku': 30, 'kv': 58, 'n_cw': 3000},

    # BEMAC Class B
    {'channel': 'bemac', 'cls': 'B', 'N': 32,  'ku': 16, 'kv': 22, 'n_cw': 5000},
    {'channel': 'bemac', 'cls': 'B', 'N': 64,  'ku': 32, 'kv': 44, 'n_cw': 5000},
    {'channel': 'bemac', 'cls': 'B', 'N': 128, 'ku': 64, 'kv': 89, 'n_cw': 5000},

    # BEMAC Class C
    {'channel': 'bemac', 'cls': 'C', 'N': 16,  'ku': 5,  'kv': 10, 'n_cw': 5000},
    {'channel': 'bemac', 'cls': 'C', 'N': 32,  'ku': 10, 'kv': 19, 'n_cw': 5000},
    {'channel': 'bemac', 'cls': 'C', 'N': 64,  'ku': 19, 'kv': 38, 'n_cw': 5000},
    {'channel': 'bemac', 'cls': 'C', 'N': 128, 'ku': 38, 'kv': 77, 'n_cw': 3000},

    # ABNMAC Class B
    {'channel': 'abnmac', 'cls': 'B', 'N': 8,   'ku': 3,  'kv': 3,  'n_cw': 5000},
    {'channel': 'abnmac', 'cls': 'B', 'N': 16,  'ku': 5,  'kv': 5,  'n_cw': 5000},
    {'channel': 'abnmac', 'cls': 'B', 'N': 32,  'ku': 10, 'kv': 10, 'n_cw': 5000},
    {'channel': 'abnmac', 'cls': 'B', 'N': 64,  'ku': 22, 'kv': 22, 'n_cw': 3000},

    # ABNMAC Class C
    {'channel': 'abnmac', 'cls': 'C', 'N': 16,  'ku': 3,  'kv': 6,  'n_cw': 5000},
    {'channel': 'abnmac', 'cls': 'C', 'N': 32,  'ku': 6,  'kv': 13, 'n_cw': 5000},
    {'channel': 'abnmac', 'cls': 'C', 'N': 64,  'ku': 13, 'kv': 26, 'n_cw': 3000},
]


def main():
    out_dir = os.path.join(BASE, 'results', 'crc_scl_sweep')
    os.makedirs(out_dir, exist_ok=True)

    # Group configs by (channel, class)
    groups = {}
    for c in CONFIGS:
        key = (c['channel'], c['cls'])
        groups.setdefault(key, []).append(c)

    for (channel, cls), cfgs in groups.items():
        out_path = os.path.join(out_dir, f'{channel}_{cls}_crc_scl.json')

        # Load existing
        all_results = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                all_results = json.load(f)

        print(f"\n{'='*78}")
        print(f"  Traditional CRC-SCL(L=4) — {channel} Class {cls}")
        print(f"{'='*78}")

        for cfg in cfgs:
            N = cfg['N']
            ku = cfg['ku']
            kv = cfg['kv']
            n_cw = cfg['n_cw']
            key = str(N)

            if key in all_results and all_results[key].get('n_cw', 0) >= n_cw:
                print(f"  [skip] N={N} already done: "
                      f"SC={all_results[key].get('sc_bler'):.4f}  "
                      f"SCL={all_results[key].get('scl_bler'):.4f}  "
                      f"CRC-SCL={all_results[key].get('crc_scl_bler'):.4f}")
                continue

            # Skip configs where CRC won't help (ku <= 8)
            if ku <= CRC_BITS:
                print(f"  [skip] N={N} ku={ku} <= CRC_BITS={CRC_BITS}, CRC not applicable")
                all_results[key] = {
                    'channel': channel, 'class': cls,
                    'N': N, 'ku': ku, 'kv': kv,
                    'skip_reason': f'ku={ku} <= CRC_BITS={CRC_BITS}',
                }
                continue

            print(f"\n  N={N}, ku={ku}, kv={kv}, n_cw={n_cw}")
            try:
                result = eval_config(channel, cls, N, ku, kv, n_cw, L=4)
                all_results[key] = result
                print(f"    DONE  SC={result['sc_bler']:.4f}  "
                      f"SCL={result['scl_bler']:.4f}  "
                      f"CRC-SCL={result['crc_scl_bler']:.4f}  "
                      f"[{result['time_s']:.0f}s]")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"    FAILED: {e}")
                all_results[key] = {'error': str(e), 'N': N, 'ku': ku, 'kv': kv}

            # Save after each
            with open(out_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    print(f"\n  All results saved to: {out_dir}")


if __name__ == '__main__':
    main()
