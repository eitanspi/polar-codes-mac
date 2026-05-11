#!/usr/bin/env python3
"""
eval_crc_scl_validation.py — CRC-SCL validation at N=256/512 with tight CI.

Parameterized: --model, --N, --L, --n_cw, --T, --out.
Uses the same decoding loop as eval_crc_scl_temp_n256_quick.py, but:
  - Supports any N for which a design file gmac_B_n{log2(N)}_snr6dB.npz exists
  - Supports any GMAC NCG checkpoint with d=16, hidden=64, n_layers=2, z_hidden=32
  - Writes per-run JSON + progress logs

Usage:
    python eval_crc_scl_validation.py --model ncg_gmac_mlp_N256.pt --N 256 \
        --L 4 --n_cw 2000 --T 1.0 --ku 123 --kv 123 \
        --out results/crc_scl_expansion/validation_N256_L4_2000cw.json
"""
import os, sys, json, time, math, argparse
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


def wilson_ci(errors, n, z=1.96):
    """Wilson score 95% CI. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = errors / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    rad = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))


def load_design(N, ku, kv):
    n = int(math.log2(N))
    d = np.load(os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz'))
    su = np.argsort(d['u_error_rates']); sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:ku]]); Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_model(model_filename):
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    sd = torch.load(os.path.join(BASE, 'saved_models', model_filename),
                    map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def decode_crc_scl_temp(model, z_single, b, fu, fv, Au, N,
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help='Filename of model checkpoint in saved_models/')
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--T', type=float, default=1.0)
    ap.add_argument('--n_cw', type=int, default=2000)
    ap.add_argument('--ku', type=int, default=None,
                    help='Defaults to N//2 - 5 (matches prior work: N=256 -> 123)')
    ap.add_argument('--kv', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--progress_every', type=int, default=100)
    args = ap.parse_args()

    N = args.N
    if args.ku is None:
        args.ku = N // 2 - 5
    if args.kv is None:
        args.kv = N // 2 - 5

    channel = GaussianMAC(sigma2=SIGMA2)
    Au, Av, fu, fv = load_design(N, args.ku, args.kv)
    b = make_path(N, N // 2)
    model = load_model(args.model)

    crc_positions = Au[-CRC_BITS:]
    msg_positions = [p for p in Au if p not in crc_positions]

    print(f"\n{'='*78}")
    print(f"  N={N}  model={args.model}  L={args.L}  T={args.T}  n_cw={args.n_cw}")
    print(f"  ku={args.ku}  kv={args.kv}  crc_positions(last8 of Au)={crc_positions[-3:]}...")
    print(f"{'='*78}", flush=True)

    errs_crc = 0
    stats = {'crc_pass': 0, 'crc_fail': 0}
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    per_cw_times = []

    for i in range(args.n_cw):
        ti = time.time()
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
            model, z_t, b, fu, fv, Au, N, L=args.L, T=args.T,
            crc_positions=crc_positions)
        stats[status] = stats.get(status, 0) + 1
        if any(uh_crc.get(p, 0) != uf[p-1] for p in Au) or \
           any(vh_crc.get(p, 0) != vf[p-1] for p in Av):
            errs_crc += 1
        per_cw_times.append(time.time() - ti)

        if (i+1) % args.progress_every == 0:
            elapsed = time.time() - t0
            avg_t = elapsed / (i+1)
            rem = avg_t * (args.n_cw - (i+1))
            lo, hi = wilson_ci(errs_crc, i+1)
            print(f"  {i+1}/{args.n_cw}  errs={errs_crc}  "
                  f"BLER={errs_crc/(i+1):.4f}  "
                  f"CI95=[{lo:.4f},{hi:.4f}]  "
                  f"avg={avg_t:.2f}s/cw  ETA={rem/60:.1f}min", flush=True)

    t_elapsed = time.time() - t0
    bler = errs_crc / args.n_cw
    lo, hi = wilson_ci(errs_crc, args.n_cw)

    result = {
        'model': args.model,
        'N': N,
        'L': args.L,
        'T': args.T,
        'n_cw': args.n_cw,
        'ku': args.ku,
        'kv': args.kv,
        'seed': args.seed,
        'snr_db': SNR_DB,
        'errors': errs_crc,
        'bler': bler,
        'ci95_lo': lo,
        'ci95_hi': hi,
        'crc_stats': stats,
        'time_s': round(t_elapsed, 1),
        'avg_s_per_cw': round(t_elapsed/args.n_cw, 3),
    }

    print(f"\n  Final: errs={errs_crc}/{args.n_cw}  "
          f"BLER={bler:.4f}  CI95=[{lo:.4f},{hi:.4f}]  "
          f"time={t_elapsed:.0f}s  ({t_elapsed/args.n_cw:.2f}s/cw)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {args.out}", flush=True)


if __name__ == '__main__':
    main()
