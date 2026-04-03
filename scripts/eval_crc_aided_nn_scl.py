#!/usr/bin/env python3
"""
eval_crc_aided_nn_scl.py — CRC-Aided Neural SCL evaluation for paper.

Evaluates NN-CA-SCL(L=4) at N=32, 64, 128.
Also N=128 with L=4, 8, 16.

CRC-aided: After SCL produces L candidates, check CRC-8 on User U's message.
Pick the candidate that passes CRC. If none pass, pick the best path metric.
"""

import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.decoder import decode_single
from polar.decoder_scl import decode_single_list
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.neural_scl import SimpleMLP_Gmac, NeuralSCLDecoder

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)

RATES = {
    32:  {'ku': 15, 'kv': 15},
    64:  {'ku': 31, 'kv': 31},
    128: {'ku': 62, 'kv': 62},
}


# ─── CRC-8 ────────────────────────────────────────────────────────────────

CRC_POLY = 0x107  # x^8 + x^2 + x + 1 (CRC-8)
CRC_BITS = 8

def crc8(bits):
    """Compute CRC-8 of a binary list/array."""
    crc = 0
    for b in bits:
        crc ^= (int(b) << CRC_BITS)
        for _ in range(1):
            if crc & (1 << CRC_BITS):
                crc ^= CRC_POLY
    return crc

def crc8_check(message_bits, crc_bits):
    """Check if CRC matches."""
    full = list(message_bits) + list(crc_bits)
    return crc8(full) == 0

def compute_crc8(message_bits):
    """Compute CRC-8 for message bits, return CRC as list of bits."""
    # Shift message left by CRC_BITS
    msg = int(''.join(str(int(b)) for b in message_bits), 2) << CRC_BITS
    # Divide by polynomial
    for i in range(len(message_bits)):
        if msg & (1 << (len(message_bits) + CRC_BITS - 1 - i)):
            msg ^= (CRC_POLY << (len(message_bits) - 1 - i))
    # Remainder is CRC
    crc = msg & ((1 << CRC_BITS) - 1)
    return [(crc >> (CRC_BITS - 1 - i)) & 1 for i in range(CRC_BITS)]


# ─── Design loading ───────────────────────────────────────────────────────

def load_design(N, ku, kv):
    n = int(math.log2(N))
    dp = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz')
    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:ku]])
    Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def load_model(N):
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    ckpt = os.path.join(BASE, 'saved_models', f'ncg_gmac_mlp_N{N}.pt')
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


# ─── CRC-Aided NN-SCL ─────────────────────────────────────────────────────

class CRCAidedNeuralSCLDecoder(NeuralSCLDecoder):
    """
    CRC-Aided Neural SCL decoder.

    After SCL produces L candidate paths, check CRC-8 on each path's
    User U message. Return the highest-metric path that passes CRC.
    If no path passes, return the highest-metric path.
    """

    def __init__(self, model, L=4, Au=None, crc_positions=None):
        super().__init__(model, L=L)
        self.Au = Au  # 1-indexed U info positions
        self.crc_positions = crc_positions  # which Au positions carry CRC bits

    def decode_crc(self, z_single, b, fu, fv, Au, crc_positions=None):
        """
        Decode with CRC check.

        If crc_positions is given, those positions in Au carry CRC bits.
        The CRC is checked on the remaining positions.
        """
        # Run standard SCL decode but return ALL L candidates
        # instead of just the best one
        candidates = self._decode_all_candidates(z_single, b, fu, fv)

        if crc_positions is None or len(crc_positions) == 0:
            # No CRC — return best path
            best = candidates[0]
            return best['uh'], best['vh']

        # Extract message bits and CRC bits for each candidate
        msg_positions = [p for p in Au if p not in crc_positions]

        for cand in candidates:
            uh = cand['uh']
            msg_bits = [uh.get(p, 0) for p in msg_positions]
            crc_bits_decoded = [uh.get(p, 0) for p in crc_positions]
            crc_expected = compute_crc8(msg_bits)

            if crc_bits_decoded == crc_expected:
                return uh, cand['vh']

        # No candidate passes CRC — return best path metric
        return candidates[0]['uh'], candidates[0]['vh']

    def _decode_all_candidates(self, z_single, b, fu, fv):
        """Run SCL and return all L candidate paths sorted by metric."""
        self.model.eval()
        device = z_single.device
        N = z_single.shape[0]
        n = N.bit_length() - 1
        d = self.tree.d
        L = self.L

        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.model.z_encoder(z_single.unsqueeze(-1).unsqueeze(0))[:, br]

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
            'dh': 1,
            'uh': {},
            'vh': {},
            'pm': 0.0,
        }]

        i_u, i_v = 0, 0
        frozen_u = fu
        frozen_v = fv

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
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                path_logits.append(log_probs[0])

            if is_frozen:
                frozen_bit = fdict[i_t]
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]
                    if gamma == 0:
                        p['uh'][i_t] = frozen_bit
                        u_val = frozen_bit
                        if i_t in frozen_v:
                            v_val = frozen_v[i_t]
                            p['pm'] += lp[u_val * 2 + v_val].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[u_val*2:u_val*2+2], dim=0).item()
                    else:
                        p['vh'][i_t] = frozen_bit
                        v_val = frozen_bit
                        if i_t in frozen_u:
                            u_val = frozen_u[i_t]
                            p['pm'] += lp[u_val * 2 + v_val].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[[v_val, v_val + 2]], dim=0).item()

                    new_emb = self.tree._make_leaf_emb(
                        torch.tensor([float(p['uh'].get(i_t, -1))]) if i_t in p.get('uh', {}) else None,
                        torch.tensor([float(p['vh'].get(i_t, -1))]) if i_t in p.get('vh', {}) else None,
                        1, device)
                    # Simplified: just rebuild
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

        # Sort all candidates by path metric
        paths.sort(key=lambda p: p['pm'], reverse=True)
        return paths


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    channel = GaussianMAC(sigma2=SIGMA2)
    results = {}

    print(f"\n{'='*80}")
    print(f"  CRC-Aided Neural SCL Evaluation — GMAC Class B, SNR={SNR_DB}dB")
    print(f"{'='*80}")

    for N in [32, 64, 128]:
        rates = RATES[N]
        ku, kv = rates['ku'], rates['kv']
        Au, Av, fu, fv = load_design(N, ku, kv)
        b = make_path(N, N // 2)

        model = load_model(N)
        n = int(math.log2(N))

        print(f"\n{'─'*80}")
        print(f"  N={N}, ku={ku}, kv={kv}")
        print(f"{'─'*80}")

        N_results = {}

        # For CRC-aided: last CRC_BITS positions of Au carry CRC
        # This reduces effective ku by CRC_BITS
        crc_positions = Au[-CRC_BITS:] if len(Au) > CRC_BITS else []
        effective_ku = ku - CRC_BITS if len(Au) > CRC_BITS else ku
        print(f"  CRC-8 positions: {crc_positions}")
        print(f"  Effective ku (after CRC): {effective_ku}")

        for L in [4, 8, 16]:
            # Skip large L for small N
            if N <= 32 and L > 4:
                continue

            n_cw = {4: 1000, 8: 500, 16: 300}[L]
            if N >= 128:
                n_cw = min(n_cw, {4: 500, 8: 300, 16: 200}[L])

            print(f"\n  NN-SCL(L={L}) ({n_cw} cw)...", flush=True)
            decoder = NeuralSCLDecoder(model, L=L)

            # Standard NN-SCL (no CRC)
            errs_no_crc = 0
            rng = np.random.default_rng(42)
            t0 = time.time()
            with torch.no_grad():
                for i in range(n_cw):
                    uf = np.zeros(N, dtype=int)
                    vf = np.zeros(N, dtype=int)
                    for p in Au: uf[p-1] = rng.integers(0, 2)
                    for p in Av: vf[p-1] = rng.integers(0, 2)

                    # Compute CRC and set CRC bits
                    msg_positions = [p for p in Au if p not in crc_positions]
                    msg_bits = [uf[p-1] for p in msg_positions]
                    crc_vals = compute_crc8(msg_bits)
                    for cp, cv in zip(crc_positions, crc_vals):
                        uf[cp-1] = cv

                    xf = polar_encode_batch(uf.reshape(1, N))
                    yf = polar_encode_batch(vf.reshape(1, N))
                    zf = channel.sample_batch(xf, yf).astype(np.float32)
                    z_t = torch.from_numpy(zf[0]).float()

                    uh, vh = decoder.decode(z_t, b, fu, fv)

                    if any(uh.get(p, 0) != uf[p-1] for p in Au) or \
                       any(vh.get(p, 0) != vf[p-1] for p in Av):
                        errs_no_crc += 1

                    if (i+1) % 200 == 0:
                        print(f"    {i+1}/{n_cw}  BLER={errs_no_crc/(i+1):.4f}", flush=True)

            bler_no_crc = errs_no_crc / n_cw
            t_elapsed = time.time() - t0
            print(f"  NN-SCL(L={L}): BLER={bler_no_crc:.4f}  [{t_elapsed:.0f}s]")

            N_results[f'NN_SCL_L{L}'] = {
                'bler': bler_no_crc, 'n_cw': n_cw,
                'time_s': round(t_elapsed, 1)
            }

            # CRC-aided: use same decoder but check CRC on candidates
            # For this, we need the full candidate list — use CRCAidedNeuralSCLDecoder
            print(f"  NN-CA-SCL(L={L}) ({n_cw} cw)...", flush=True)
            ca_decoder = CRCAidedNeuralSCLDecoder(model, L=L)

            errs_crc = 0
            rng = np.random.default_rng(42)
            t0 = time.time()
            with torch.no_grad():
                for i in range(n_cw):
                    uf = np.zeros(N, dtype=int)
                    vf = np.zeros(N, dtype=int)
                    for p in Au: uf[p-1] = rng.integers(0, 2)
                    for p in Av: vf[p-1] = rng.integers(0, 2)

                    # Set CRC bits
                    msg_positions = [p for p in Au if p not in crc_positions]
                    msg_bits = [uf[p-1] for p in msg_positions]
                    crc_vals = compute_crc8(msg_bits)
                    for cp, cv in zip(crc_positions, crc_vals):
                        uf[cp-1] = cv

                    xf = polar_encode_batch(uf.reshape(1, N))
                    yf = polar_encode_batch(vf.reshape(1, N))
                    zf = channel.sample_batch(xf, yf).astype(np.float32)
                    z_t = torch.from_numpy(zf[0]).float()

                    uh, vh = ca_decoder.decode_crc(z_t, b, fu, fv, Au, crc_positions)

                    if any(uh.get(p, 0) != uf[p-1] for p in Au) or \
                       any(vh.get(p, 0) != vf[p-1] for p in Av):
                        errs_crc += 1

                    if (i+1) % 200 == 0:
                        print(f"    {i+1}/{n_cw}  BLER={errs_crc/(i+1):.4f}", flush=True)

            bler_crc = errs_crc / n_cw
            t_elapsed = time.time() - t0
            print(f"  NN-CA-SCL(L={L}): BLER={bler_crc:.4f}  [{t_elapsed:.0f}s]")

            N_results[f'NN_CA_SCL_L{L}'] = {
                'bler': bler_crc, 'n_cw': n_cw,
                'time_s': round(t_elapsed, 1)
            }

        results[str(N)] = {'N': N, 'ku': ku, 'kv': kv, **N_results}

    # Save
    out_path = os.path.join(BASE, 'results', 'gmac_snr6dB', 'crc_aided_nn_scl.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  Summary: CRC-Aided NN-SCL")
    print(f"{'='*80}")
    for N_s, r in results.items():
        print(f"\n  N={r['N']}, ku={r['ku']}, kv={r['kv']}:")
        for k, v in r.items():
            if isinstance(v, dict) and 'bler' in v:
                print(f"    {k:<20s}  BLER={v['bler']:.4f}")


if __name__ == '__main__':
    main()
