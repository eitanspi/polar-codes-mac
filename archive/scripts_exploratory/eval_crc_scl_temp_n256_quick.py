#!/usr/bin/env python3
"""
eval_crc_scl_temp_n256_quick.py — N=256 CRC-SCL with temperature scaling.

Tests whether temperature improves CRC-SCL L=4 at N=256 (where it normally barely
beats SC). Tests T ∈ {1.0, 1.5, 3.0} each at 150 cw.
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
KU = KV = 123

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
    d = np.load(os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz'))
    su = np.argsort(d['u_error_rates']); sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:KU]]); Av = sorted([int(i+1) for i in sv[:KV]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_model():
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(os.path.join(BASE, 'saved_models', f'ncg_gmac_mlp_N{N}.pt'),
                    map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def decode_crc_scl_temp(model, z_single, b, fu, fv, Au,
                        L=4, T=1.0, crc_positions=None):
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
                logits = tree.emb2logits(combined) / T
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

    crc_positions = Au[-CRC_BITS:]
    msg_positions = [p for p in Au if p not in crc_positions]

    print(f"\n{'='*78}")
    print(f"  N={N} CRC-SCL+Temperature  (L=4)")
    print(f"{'='*78}")

    results = {'N': N, 'ku': KU, 'kv': KV, 'snr_db': SNR_DB, 'L': 4}

    N_CW = 100  # keep it tight — CRC-SCL only (not plain SCL, save time)
    for T in [1.0, 1.5, 3.0]:
        errs_crc = 0
        stats = {'crc_pass': 0, 'crc_fail': 0}
        rng = np.random.default_rng(42)
        t0 = time.time()
        for i in range(N_CW):
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

            uh_crc, vh_crc, status = decode_crc_scl_temp(
                model, z_t, b, fu, fv, Au, L=4, T=T,
                crc_positions=crc_positions)
            stats[status] = stats.get(status, 0) + 1
            if any(uh_crc.get(p, 0) != uf[p-1] for p in Au) or \
               any(vh_crc.get(p, 0) != vf[p-1] for p in Av):
                errs_crc += 1

            if (i+1) % 20 == 0:
                print(f"  T={T}: {i+1}/{N_CW}  "
                      f"CRC-SCL={errs_crc/(i+1):.4f}  {stats}", flush=True)
        t = time.time() - t0
        results[f'T_{T}'] = {
            'bler_crc_scl': errs_crc/N_CW,
            'n_cw': N_CW,
            'time_s': round(t, 1),
            'crc_stats': stats,
        }
        print(f"  Final T={T}: CRC-SCL={errs_crc/N_CW:.4f}  [{t:.0f}s]")

    out_path = os.path.join(BASE, 'results', 'crc_scl_expansion',
                            'gmac_N256_crc_scl_temperature.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
