#!/usr/bin/env python3
"""
eval_crc_scl_unified.py — CRC-aided Neural SCL across channels.

Runs CRC-8-aided Neural SCL on GMAC, BEMAC, and ABNMAC for Class B at
N=32, 64, 128, with L in {4, 8, 16}. Extends the existing GMAC-only
result (scripts/eval_crc_aided_nn_scl.py) to all three channels.

Each run:
  * builds a CRC-8 on the first len(Au)-8 info positions of U
  * places the 8 CRC bits at the last 8 positions of Au
  * runs Neural SCL, keeps all L candidates, checks CRC on each
  * picks highest-metric candidate that passes CRC; if none, best metric

Outputs: results/crc_scl_expansion/{channel}_classB_crc_scl.json
"""

import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'neural'))

torch.set_num_threads(2)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC, BEMAC, ABNMAC
from polar.design import make_path
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder
from neural.neural_scl import SimpleMLP_Gmac

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

# ─── CRC-8 ──────────────────────────────────────────────────────────────────

CRC_POLY = 0x107  # x^8 + x^2 + x + 1
CRC_BITS = 8

def compute_crc8(message_bits):
    """Compute CRC-8 for message bits, return CRC as list of bits."""
    if len(message_bits) == 0:
        return [0]*CRC_BITS
    msg = int(''.join(str(int(b)) for b in message_bits), 2) << CRC_BITS
    for i in range(len(message_bits)):
        if msg & (1 << (len(message_bits) + CRC_BITS - 1 - i)):
            msg ^= (CRC_POLY << (len(message_bits) - 1 - i))
    crc = msg & ((1 << CRC_BITS) - 1)
    return [(crc >> (CRC_BITS - 1 - i)) & 1 for i in range(CRC_BITS)]


# ─── Channel setup ──────────────────────────────────────────────────────────

def make_channel(channel_name):
    if channel_name == 'gmac':
        return GaussianMAC(sigma2=SIGMA2)
    elif channel_name == 'bemac':
        return BEMAC()
    elif channel_name == 'abnmac':
        return ABNMAC()
    else:
        raise ValueError(f'Unknown channel: {channel_name}')


def channel_to_z_tensor(channel_name, z_batch):
    """Convert channel output to tensor suitable for the model."""
    if channel_name == 'gmac':
        return torch.from_numpy(z_batch.astype(np.float32)).float()
    elif channel_name == 'bemac':
        return torch.from_numpy(z_batch.astype(np.int64)).long()
    elif channel_name == 'abnmac':
        # z is a (batch, N) object ndarray of tuples (zx,zy) → encode as 2*zx+zy
        out = np.empty(z_batch.shape, dtype=np.int64)
        for idx in np.ndindex(z_batch.shape):
            zx, zy = z_batch[idx]
            out[idx] = 2*int(zx) + int(zy)
        return torch.from_numpy(out).long()
    else:
        raise ValueError(channel_name)


# ─── Model loading ──────────────────────────────────────────────────────────

def load_model(channel_name, N, device='cpu'):
    save_dir = os.path.join(BASE, 'saved_models')
    if channel_name == 'gmac':
        model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
        ckpt = os.path.join(save_dir, f'ncg_gmac_mlp_N{N}.pt')
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
        model.load_state_dict(fixed, strict=False)
        model.eval()
        return model, 'gmac'  # tree accessed via model.tree
    elif channel_name == 'bemac':
        model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=3)
        ckpt = os.path.join(save_dir, f'ncg_pure_neural_N{N}.pt')
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model, 'bemac'
    elif channel_name == 'abnmac':
        model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=4)
        ckpt = os.path.join(save_dir, f'ncg_abnmac_N{N}.pt')
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model, 'abnmac'
    else:
        raise ValueError(channel_name)


# ─── Design loading ─────────────────────────────────────────────────────────

def load_design(channel_name, N, ku, kv):
    n = int(math.log2(N))
    if channel_name == 'gmac':
        dp = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz')
    elif channel_name == 'bemac':
        dp = os.path.join(BASE, 'designs', f'bemac_B_n{n}.npz')
    elif channel_name == 'abnmac':
        dp = os.path.join(BASE, 'designs', f'abnmac_B_n{n}.npz')
    else:
        raise ValueError(channel_name)

    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:ku]])
    Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


# ─── Channel-agnostic SCL with full candidate list ──────────────────────────

class UnifiedSCL:
    """SCL decoder that works for GMAC (SimpleMLP_Gmac) or BEMAC/ABNMAC
    (PureNeuralCompGraphDecoder). Returns all L candidates sorted by metric."""

    def __init__(self, model, kind, L=4):
        self.model = model
        self.kind = kind
        self.L = L
        # Access tree
        if kind == 'gmac':
            self.tree = model.tree
            self.z_encoder = model.z_encoder
        else:
            self.tree = model
            self.z_encoder = None  # Use embedding_z for discrete

    def _root_emb(self, z_single, n):
        br = torch.from_numpy(bit_reversal_perm(n)).long()
        if self.kind == 'gmac':
            return self.z_encoder(z_single.unsqueeze(-1).unsqueeze(0))[:, br]
        else:
            return self.model.embedding_z(z_single.unsqueeze(0))[:, br]

    @torch.no_grad()
    def decode_list(self, z_single, b, frozen_u, frozen_v):
        self.model.eval()
        device = z_single.device if z_single.is_floating_point() else 'cpu'
        N = z_single.shape[0]
        n = N.bit_length() - 1
        d = self.tree.d
        L = self.L

        root = self._root_emb(z_single, n)
        no_info = self.tree.no_info_emb.unsqueeze(0).unsqueeze(0)

        def _make_edge_data():
            ed = [None] * (2 * N)
            ed[1] = root.clone()
            for beta in range(2, 2 * N):
                level = beta.bit_length() - 1
                size = N >> level
                ed[beta] = no_info.expand(1, size, d).clone()
            return ed

        def _clone_edge_data(ed):
            return [e.clone() if e is not None else None for e in ed]

        paths = [{
            'ed': _make_edge_data(),
            'dh': 1, 'uh': {}, 'vh': {}, 'pm': 0.0,
        }]

        i_u, i_v = 0, 0
        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = frozen_v

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1
            is_frozen = i_t in fdict

            path_logits = []
            for p in paths:
                ed = p['ed']
                dh = p['dh']
                dh = self._step_to_path(dh, target_vtx, ed)
                p['dh'] = dh
                temp = ed[leaf_edge][:, 0].clone()
                if leaf_edge & 1 == 0:
                    self.tree._neural_calc_left(target_vtx, ed)
                else:
                    self.tree._neural_calc_right(target_vtx, ed)
                top_down = ed[leaf_edge][:, 0]
                if self.tree.use_combine_nn:
                    combined = self.tree.combine_nn(
                        torch.cat([top_down, temp], dim=-1))
                else:
                    combined = top_down + temp
                logits = self.tree.emb2logits(combined)
                log_probs = F.log_softmax(logits, dim=-1)
                path_logits.append(log_probs[0])

            if is_frozen:
                frozen_bit = fdict[i_t]
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]
                    if gamma == 0:
                        p['uh'][i_t] = int(frozen_bit)
                        u_val = frozen_bit
                        if i_t in frozen_v:
                            v_val = frozen_v[i_t]
                            p['pm'] += lp[u_val * 2 + v_val].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[u_val*2:u_val*2+2], dim=0).item()
                    else:
                        p['vh'][i_t] = int(frozen_bit)
                        v_val = frozen_bit
                        if i_t in frozen_u:
                            u_val = frozen_u[i_t]
                            p['pm'] += lp[u_val * 2 + v_val].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[[v_val, v_val + 2]], dim=0).item()

                    u_val_t = torch.tensor([float(p['uh'][i_t])]) if i_t in p['uh'] else None
                    v_val_t = torch.tensor([float(p['vh'][i_t])]) if i_t in p['vh'] else None
                    new_emb = self.tree._make_leaf_emb(u_val_t, v_val_t, 1, device)
                    p['ed'][leaf_edge] = new_emb.unsqueeze(1)
            else:
                candidates = []
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]
                    if gamma == 0:
                        options = [
                            (0, torch.logsumexp(lp[:2], dim=0).item()),
                            (1, torch.logsumexp(lp[2:], dim=0).item()),
                        ]
                    else:
                        options = [
                            (0, torch.logsumexp(lp[[0, 2]], dim=0).item()),
                            (1, torch.logsumexp(lp[[1, 3]], dim=0).item()),
                        ]
                    for bit_val, log_p in options:
                        candidates.append((p['pm'] + log_p, pidx, bit_val))

                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:L]

                new_paths = []
                for new_pm, pidx, bit_val in candidates:
                    old_p = paths[pidx]
                    new_p = {
                        'ed': _clone_edge_data(old_p['ed']),
                        'dh': old_p['dh'],
                        'uh': dict(old_p['uh']),
                        'vh': dict(old_p['vh']),
                        'pm': new_pm,
                    }
                    if gamma == 0:
                        new_p['uh'][i_t] = bit_val
                    else:
                        new_p['vh'][i_t] = bit_val

                    u_val_t = torch.tensor([float(new_p['uh'][i_t])]) if i_t in new_p['uh'] else None
                    v_val_t = torch.tensor([float(new_p['vh'][i_t])]) if i_t in new_p['vh'] else None
                    new_emb = self.tree._make_leaf_emb(u_val_t, v_val_t, 1, device)
                    new_p['ed'][leaf_edge] = new_emb.unsqueeze(1)
                    new_paths.append(new_p)

                paths = new_paths

        paths.sort(key=lambda p: p['pm'], reverse=True)
        return paths

    def _step_to_path(self, current, target, edge_data):
        if current == target:
            return current
        path = self.tree._get_path(current, target)
        for beta in path:
            current = self._step_one_path(current, beta, edge_data)
        return current

    def _step_one_path(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0:
                self.tree._neural_calc_left(current, edge_data)
            else:
                self.tree._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self.tree._pure_neural_calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: current={current}, target={beta}")


# ─── Evaluation ─────────────────────────────────────────────────────────────

def run_one(channel_name, N, ku, kv, L, n_cw, seed=42, use_crc=True):
    """Evaluate SCL (+ optionally CRC-aided) on a single config."""
    channel = make_channel(channel_name)
    Au, Av, fu, fv = load_design(channel_name, N, ku, kv)
    b = make_path(N, N // 2)
    model, kind = load_model(channel_name, N)
    decoder = UnifiedSCL(model, kind, L=L)

    crc_positions = Au[-CRC_BITS:] if use_crc and len(Au) > CRC_BITS else []
    msg_positions = [p for p in Au if p not in crc_positions]

    errs_scl = 0
    errs_crc = 0
    rng = np.random.default_rng(seed)
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_cw):
            uf = np.zeros(N, dtype=int)
            vf = np.zeros(N, dtype=int)
            for p in Au: uf[p-1] = rng.integers(0, 2)
            for p in Av: vf[p-1] = rng.integers(0, 2)
            if crc_positions:
                msg_bits = [uf[p-1] for p in msg_positions]
                crc_vals = compute_crc8(msg_bits)
                for cp, cv in zip(crc_positions, crc_vals):
                    uf[cp-1] = cv

            xf = polar_encode_batch(uf.reshape(1, N))
            yf = polar_encode_batch(vf.reshape(1, N))
            z_np = channel.sample_batch(xf, yf)
            z_t = channel_to_z_tensor(channel_name, z_np)[0]

            paths = decoder.decode_list(z_t, b, fu, fv)

            # Plain SCL: best by metric
            best = paths[0]
            uh_scl = best['uh']; vh_scl = best['vh']
            err_scl = any(uh_scl.get(p, 0) != uf[p-1] for p in Au) or \
                      any(vh_scl.get(p, 0) != vf[p-1] for p in Av)
            if err_scl:
                errs_scl += 1

            # CRC-aided: first candidate passing CRC
            if use_crc and crc_positions:
                picked = None
                for cand in paths:
                    uh = cand['uh']
                    msg_bits = [uh.get(p, 0) for p in msg_positions]
                    crc_dec = [uh.get(p, 0) for p in crc_positions]
                    if compute_crc8(msg_bits) == crc_dec:
                        picked = cand
                        break
                if picked is None:
                    picked = paths[0]
                uh_crc = picked['uh']; vh_crc = picked['vh']
                err_crc = any(uh_crc.get(p, 0) != uf[p-1] for p in Au) or \
                          any(vh_crc.get(p, 0) != vf[p-1] for p in Av)
                if err_crc:
                    errs_crc += 1

            if (i+1) % 200 == 0 or i+1 == n_cw:
                msg = f"      {i+1}/{n_cw}  SCL BLER={errs_scl/(i+1):.4f}"
                if use_crc and crc_positions:
                    msg += f"  CRC-SCL BLER={errs_crc/(i+1):.4f}"
                msg += f"  ({(time.time()-t0)/(i+1):.2f}s/cw)"
                print(msg, flush=True)

    t_total = time.time() - t0
    result = {
        'channel': channel_name, 'N': N, 'ku': ku, 'kv': kv, 'L': L,
        'n_cw': n_cw, 'time_s': round(t_total, 1),
        'bler_scl': errs_scl / n_cw,
        'crc_positions': crc_positions if use_crc else [],
    }
    if use_crc and crc_positions:
        result['bler_crc_scl'] = errs_crc / n_cw
    return result


# ─── Main ───────────────────────────────────────────────────────────────────

# Rates per channel (Class B). Match existing training rates where known.
RATES = {
    'gmac': {
        32:  {'ku': 15, 'kv': 15},
        64:  {'ku': 31, 'kv': 31},
        128: {'ku': 62, 'kv': 62},
        256: {'ku': 123, 'kv': 123},
    },
    'bemac': {
        32:  {'ku': 16, 'kv': 22},  # Ru=0.5, Rv=~0.7
        64:  {'ku': 32, 'kv': 44},
        128: {'ku': 64, 'kv': 89},
        256: {'ku': 128, 'kv': 178},
        512: {'ku': 256, 'kv': 358},
        1024: {'ku': 512, 'kv': 716},
    },
    'abnmac': {
        8:   {'ku': 3,  'kv': 3},
        16:  {'ku': 5,  'kv': 5},
        32:  {'ku': 10, 'kv': 10},
        64:  {'ku': 22, 'kv': 22},
        128: {'ku': 45, 'kv': 45},
    },
}

# Codeword budgets per (N, L) — balance cost vs statistical confidence
CW_BUDGET = {
    (32, 4): 1500, (32, 8): 1000, (32, 16): 700,
    (64, 4): 1000, (64, 8): 600,  (64, 16): 400,
    (128, 4): 500, (128, 8): 300, (128, 16): 200,
    (256, 4): 300, (256, 8): 200, (256, 16): 150,
    (512, 4): 150, (512, 8): 100, (512, 16): 50,
    (1024, 4): 100,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channel', required=True, choices=['gmac', 'bemac', 'abnmac'])
    ap.add_argument('--Ns', type=int, nargs='+', default=[32, 64, 128])
    ap.add_argument('--Ls', type=int, nargs='+', default=[4, 8, 16])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        BASE, 'results', 'crc_scl_expansion',
        f'{args.channel}_classB_crc_scl.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Resume logic: load existing results
    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)

    print(f"\n{'='*78}")
    print(f"  CRC-aided Neural SCL — channel={args.channel}, Class B, SNR={SNR_DB} dB")
    print(f"{'='*78}")

    for N in args.Ns:
        rates = RATES[args.channel][N]
        ku, kv = rates['ku'], rates['kv']
        key_N = str(N)
        if key_N not in all_results:
            all_results[key_N] = {'N': N, 'ku': ku, 'kv': kv}

        for L in args.Ls:
            key_L = f'L{L}'
            if key_L in all_results[key_N]:
                print(f"  [skip] N={N}, L={L} already done: "
                      f"BLER-SCL={all_results[key_N][key_L].get('bler_scl')}")
                continue

            n_cw = CW_BUDGET.get((N, L), 300)
            print(f"\n  N={N}  L={L}  n_cw={n_cw} ... ", flush=True)
            try:
                res = run_one(args.channel, N, ku, kv, L, n_cw, seed=args.seed)
                all_results[key_N][key_L] = res
                print(f"    DONE  SCL={res['bler_scl']:.4f} "
                      f"{'CRC-SCL=' + format(res.get('bler_crc_scl', -1), '.4f') if 'bler_crc_scl' in res else ''}"
                      f" [{res['time_s']:.0f}s]")
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"    FAILED: {e}")
                all_results[key_N][key_L] = {'error': str(e)}

            # Save after each config
            with open(out_path, 'w') as f:
                json.dump(all_results, f, indent=2)

    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
