#!/usr/bin/env python3
"""
eval_temperature_n256.py — Temperature scaling diagnostic for GMAC B N=256 NCG.

The project notes miscalibrated probabilities at N=256 (docs/comprehensive_report.md
§8). We try T in {1.0, 1.5, 2.0, 3.0}: divide logits by T before taking argmax
(for plain NCG-SC) or before softmax (for SCL / CRC-SCL). T > 1 flattens the
distribution and might let SCL keep the correct path in the list longer.

Outputs: results/crc_scl_expansion/gmac_N256_temperature.json
"""

import os, sys, json, time, math
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

torch.set_num_threads(2)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
N = 256
KU = 123
KV = 123


# ─── CRC-8 ────────────────────────────────────────────────────────────────

CRC_POLY = 0x107
CRC_BITS = 8

def compute_crc8(message_bits):
    if len(message_bits) == 0:
        return [0]*CRC_BITS
    msg = int(''.join(str(int(b)) for b in message_bits), 2) << CRC_BITS
    for i in range(len(message_bits)):
        if msg & (1 << (len(message_bits) + CRC_BITS - 1 - i)):
            msg ^= (CRC_POLY << (len(message_bits) - 1 - i))
    crc = msg & ((1 << CRC_BITS) - 1)
    return [(crc >> (CRC_BITS - 1 - i)) & 1 for i in range(CRC_BITS)]


def load_design():
    n = int(math.log2(N))
    dp = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz')
    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:KU]])
    Av = sorted([int(i+1) for i in sv[:KV]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_model():
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    ckpt = os.path.join(BASE, 'saved_models', f'ncg_gmac_mlp_N{N}.pt')
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def decode_ncg_sc_temp(model, z_single, b, fu, fv, T=1.0):
    """Greedy NCG-SC decode with logit temperature T."""
    tree = model.tree
    n_ = N.bit_length() - 1
    d = tree.d

    br = torch.from_numpy(bit_reversal_perm(n_)).long()
    root = model.z_encoder(z_single.unsqueeze(-1).unsqueeze(0))[:, br]
    no_info = tree.no_info_emb.unsqueeze(0).unsqueeze(0)

    edge_data = [None] * (2 * N)
    edge_data[1] = root.clone()
    for beta in range(2, 2 * N):
        level = beta.bit_length() - 1
        size = N >> level
        edge_data[beta] = no_info.expand(1, size, d).clone()

    dec_head = 1
    uh, vh = {}, {}
    i_u, i_v = 0, 0

    def step_to(current, target, ed):
        if current == target:
            return current
        path = tree._get_path(current, target)
        for beta in path:
            if current == beta >> 1:
                if beta & 1 == 0:
                    tree._neural_calc_left(current, ed)
                else:
                    tree._neural_calc_right(current, ed)
                current = beta
            elif beta == current >> 1:
                tree._pure_neural_calc_parent(current, ed)
                current = beta
        return current

    with torch.no_grad():
        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = fu
            else:
                i_v += 1; i_t = i_v; fdict = fv
            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1

            dec_head = step_to(dec_head, target_vtx, edge_data)
            temp = edge_data[leaf_edge][:, 0].clone()
            if leaf_edge & 1 == 0:
                tree._neural_calc_left(target_vtx, edge_data)
            else:
                tree._neural_calc_right(target_vtx, edge_data)
            top_down = edge_data[leaf_edge][:, 0]
            if tree.use_combine_nn:
                combined = tree.combine_nn(torch.cat([top_down, temp], dim=-1))
            else:
                combined = top_down + temp
            logits = tree.emb2logits(combined) / T  # <-- temperature
            if i_t in fdict:
                bit = fdict[i_t]
            else:
                if gamma == 0:
                    p0 = torch.logsumexp(logits[:, :2], dim=1)
                    p1 = torch.logsumexp(logits[:, 2:], dim=1)
                else:
                    p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                    p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                bit = int((p1 > p0).item())

            if gamma == 0:
                uh[i_t] = bit
            else:
                vh[i_t] = bit

            u_t = torch.tensor([float(uh.get(i_t, -1))]) if i_t in uh else None
            v_t = torch.tensor([float(vh.get(i_t, -1))]) if i_t in vh else None
            new_emb = tree._make_leaf_emb(u_t, v_t, 1, 'cpu')
            edge_data[leaf_edge] = new_emb.unsqueeze(1)

    return uh, vh


def decode_crc_scl_temp(model, z_single, b, fu, fv, Au,
                        L=4, T=1.0, crc_positions=None):
    """SCL with temperature, optional CRC check on U bits."""
    tree = model.tree
    n_ = N.bit_length() - 1
    d = tree.d

    br = torch.from_numpy(bit_reversal_perm(n_)).long()
    root = model.z_encoder(z_single.unsqueeze(-1).unsqueeze(0))[:, br]
    no_info = tree.no_info_emb.unsqueeze(0).unsqueeze(0)

    def _make_ed():
        ed = [None] * (2 * N)
        ed[1] = root.clone()
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            ed[beta] = no_info.expand(1, size, d).clone()
        return ed

    def _clone(ed):
        return [e.clone() if e is not None else None for e in ed]

    def _step(current, target, ed):
        if current == target:
            return current
        path = tree._get_path(current, target)
        for beta in path:
            if current == beta >> 1:
                if beta & 1 == 0:
                    tree._neural_calc_left(current, ed)
                else:
                    tree._neural_calc_right(current, ed)
                current = beta
            elif beta == current >> 1:
                tree._pure_neural_calc_parent(current, ed)
                current = beta
        return current

    paths = [{'ed': _make_ed(), 'dh': 1, 'uh': {}, 'vh': {}, 'pm': 0.0}]
    i_u, i_v = 0, 0

    with torch.no_grad():
        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = fu
            else:
                i_v += 1; i_t = i_v; fdict = fv
            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1
            is_frozen = i_t in fdict

            path_logits = []
            for p in paths:
                p['dh'] = _step(p['dh'], target_vtx, p['ed'])
                temp = p['ed'][leaf_edge][:, 0].clone()
                if leaf_edge & 1 == 0:
                    tree._neural_calc_left(target_vtx, p['ed'])
                else:
                    tree._neural_calc_right(target_vtx, p['ed'])
                top_down = p['ed'][leaf_edge][:, 0]
                if tree.use_combine_nn:
                    combined = tree.combine_nn(torch.cat([top_down, temp], dim=-1))
                else:
                    combined = top_down + temp
                logits = tree.emb2logits(combined) / T  # <-- temperature
                lp = F.log_softmax(logits, dim=-1)
                path_logits.append(lp[0])

            if is_frozen:
                fb = fdict[i_t]
                for pi, p in enumerate(paths):
                    lp = path_logits[pi]
                    if gamma == 0:
                        p['uh'][i_t] = int(fb)
                        p['pm'] += torch.logsumexp(lp[fb*2:fb*2+2], dim=0).item()
                    else:
                        p['vh'][i_t] = int(fb)
                        p['pm'] += torch.logsumexp(lp[[fb, fb+2]], dim=0).item()
                    u_t = torch.tensor([float(p['uh'][i_t])]) if i_t in p['uh'] else None
                    v_t = torch.tensor([float(p['vh'][i_t])]) if i_t in p['vh'] else None
                    new_emb = tree._make_leaf_emb(u_t, v_t, 1, 'cpu')
                    p['ed'][leaf_edge] = new_emb.unsqueeze(1)
            else:
                cands = []
                for pi, p in enumerate(paths):
                    lp = path_logits[pi]
                    if gamma == 0:
                        options = [(0, torch.logsumexp(lp[:2], dim=0).item()),
                                   (1, torch.logsumexp(lp[2:], dim=0).item())]
                    else:
                        options = [(0, torch.logsumexp(lp[[0,2]], dim=0).item()),
                                   (1, torch.logsumexp(lp[[1,3]], dim=0).item())]
                    for bv, lpv in options:
                        cands.append((p['pm'] + lpv, pi, bv))
                cands.sort(key=lambda x: x[0], reverse=True)
                cands = cands[:L]
                new_paths = []
                for pm, pi, bv in cands:
                    op = paths[pi]
                    np_ = {'ed': _clone(op['ed']), 'dh': op['dh'],
                           'uh': dict(op['uh']), 'vh': dict(op['vh']), 'pm': pm}
                    if gamma == 0:
                        np_['uh'][i_t] = bv
                    else:
                        np_['vh'][i_t] = bv
                    u_t = torch.tensor([float(np_['uh'][i_t])]) if i_t in np_['uh'] else None
                    v_t = torch.tensor([float(np_['vh'][i_t])]) if i_t in np_['vh'] else None
                    new_emb = tree._make_leaf_emb(u_t, v_t, 1, 'cpu')
                    np_['ed'][leaf_edge] = new_emb.unsqueeze(1)
                    new_paths.append(np_)
                paths = new_paths

    paths.sort(key=lambda p: p['pm'], reverse=True)

    if crc_positions:
        msg_pos = [p for p in Au if p not in crc_positions]
        for cand in paths:
            uh = cand['uh']
            msg_bits = [uh.get(p, 0) for p in msg_pos]
            crc_dec = [uh.get(p, 0) for p in crc_positions]
            if compute_crc8(msg_bits) == crc_dec:
                return cand['uh'], cand['vh'], 'crc_pass'
        return paths[0]['uh'], paths[0]['vh'], 'crc_fail'

    return paths[0]['uh'], paths[0]['vh'], 'no_crc'


def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design()
    b = make_path(N, N // 2)
    model = load_model()

    print(f"\n{'='*78}")
    print(f"  Temperature Scaling Diagnostic — GMAC B N={N}")
    print(f"  SC ref BLER ≈ 0.005, NCG baseline ≈ 0.015 (T=1)")
    print(f"{'='*78}")

    results = {'N': N, 'ku': KU, 'kv': KV, 'snr_db': SNR_DB}

    # Phase 1: NCG-SC at different T, 500 cw — quick scan
    print(f"\n--- Phase 1: NCG-SC vs temperature (500 cw each) ---")
    for T in [1.0, 1.2, 1.5, 2.0, 3.0]:
        errs = 0
        n_cw = 500
        rng = np.random.default_rng(42)
        t0 = time.time()
        with torch.no_grad():
            for i in range(n_cw):
                uf = np.zeros(N, dtype=int)
                vf = np.zeros(N, dtype=int)
                for p in Au: uf[p-1] = rng.integers(0, 2)
                for p in Av: vf[p-1] = rng.integers(0, 2)
                xf = polar_encode_batch(uf.reshape(1, N))
                yf = polar_encode_batch(vf.reshape(1, N))
                z_np = channel.sample_batch(xf, yf).astype(np.float32)
                z_t = torch.from_numpy(z_np[0]).float()
                uh, vh = decode_ncg_sc_temp(model, z_t, b, fu, fv, T=T)
                err = any(uh.get(p, 0) != uf[p-1] for p in Au) or \
                      any(vh.get(p, 0) != vf[p-1] for p in Av)
                if err: errs += 1
                if (i+1) % 100 == 0:
                    print(f"    T={T}: {i+1}/{n_cw}  BLER={errs/(i+1):.4f}", flush=True)
        bler = errs / n_cw
        t = time.time() - t0
        print(f"  T={T:.1f}: BLER={bler:.4f}  [{t:.0f}s]")
        results[f'ncg_sc_T{T}'] = {'bler': bler, 'n_cw': n_cw, 'time_s': round(t,1)}

    # Phase 2: best-T  × L=4 × CRC, 200 cw
    best_T = min([(T, results[f'ncg_sc_T{T}']['bler']) for T in [1.0, 1.2, 1.5, 2.0, 3.0]],
                 key=lambda x: x[1])[0]
    print(f"\n--- Phase 2: CRC-SCL with best T={best_T} (200 cw, L=4) ---")
    crc_positions = Au[-CRC_BITS:]
    msg_positions = [p for p in Au if p not in crc_positions]

    for T in [1.0, best_T]:
        errs_scl = 0
        errs_crc = 0
        errs_crc_status = {'crc_pass': 0, 'crc_fail': 0}
        n_cw = 200
        rng = np.random.default_rng(42)
        t0 = time.time()
        with torch.no_grad():
            for i in range(n_cw):
                uf = np.zeros(N, dtype=int)
                vf = np.zeros(N, dtype=int)
                for p in Au: uf[p-1] = rng.integers(0, 2)
                for p in Av: vf[p-1] = rng.integers(0, 2)
                msg_bits = [uf[p-1] for p in msg_positions]
                crc_vals = compute_crc8(msg_bits)
                for cp, cv in zip(crc_positions, crc_vals):
                    uf[cp-1] = cv
                xf = polar_encode_batch(uf.reshape(1, N))
                yf = polar_encode_batch(vf.reshape(1, N))
                z_np = channel.sample_batch(xf, yf).astype(np.float32)
                z_t = torch.from_numpy(z_np[0]).float()

                # CRC-aided
                uh_crc, vh_crc, status = decode_crc_scl_temp(
                    model, z_t, b, fu, fv, Au, L=4, T=T,
                    crc_positions=crc_positions)
                errs_crc_status[status] = errs_crc_status.get(status, 0) + 1
                err_crc = any(uh_crc.get(p, 0) != uf[p-1] for p in Au) or \
                          any(vh_crc.get(p, 0) != vf[p-1] for p in Av)
                if err_crc: errs_crc += 1

                # Plain SCL (no CRC selection — just best)
                uh_scl, vh_scl, _ = decode_crc_scl_temp(
                    model, z_t, b, fu, fv, Au, L=4, T=T,
                    crc_positions=None)
                err_scl = any(uh_scl.get(p, 0) != uf[p-1] for p in Au) or \
                          any(vh_scl.get(p, 0) != vf[p-1] for p in Av)
                if err_scl: errs_scl += 1

                if (i+1) % 50 == 0:
                    print(f"    T={T}: {i+1}/{n_cw} SCL={errs_scl/(i+1):.4f} "
                          f"CRC-SCL={errs_crc/(i+1):.4f}", flush=True)

        t = time.time() - t0
        results[f'crc_scl_T{T}'] = {
            'bler_scl': errs_scl / n_cw,
            'bler_crc_scl': errs_crc / n_cw,
            'crc_stats': errs_crc_status,
            'n_cw': n_cw, 'time_s': round(t, 1),
        }
        print(f"  T={T:.1f}: SCL={errs_scl/n_cw:.4f}  CRC-SCL={errs_crc/n_cw:.4f}  "
              f"{errs_crc_status}  [{t:.0f}s]")

    out_path = os.path.join(BASE, 'results', 'crc_scl_expansion',
                             'gmac_N256_temperature.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
