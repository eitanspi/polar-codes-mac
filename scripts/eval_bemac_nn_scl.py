#!/usr/bin/env python3
"""
eval_bemac_nn_scl.py — Evaluate Neural SCL decoder on BEMAC (Z=X+Y).

Tests the PureNeuralCompGraphDecoder with SCL (list decoding) on BEMAC
Class B at Ru=0.50, Rv=0.70.

Uses the pre-trained ncg_pure_neural_N{N}.pt models.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'nn_mac'))

import math
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import BEMAC
from polar.design import make_path
from ncg_pure_neural import PureNeuralCompGraphDecoder


# ─── BEMAC Neural SCL Decoder ─────────────────────────────────────────────

class BemacNeuralSCLDecoder:
    """
    Neural SC List decoder for BEMAC polar codes.

    Uses PureNeuralCompGraphDecoder with nn.Embedding(3, d) front-end.
    Maintains L candidate paths; at each non-frozen leaf, forks into
    up to 4 candidates and keeps the best L.
    """

    def __init__(self, model, L=4):
        self.model = model
        self.L = L

    @torch.no_grad()
    def decode(self, z_single, b, frozen_u, frozen_v):
        """
        Decode a SINGLE codeword with SCL.

        Parameters
        ----------
        z_single : (N,) LongTensor — single channel output (values in {0,1,2})
        b        : list of 2N ints — path vector
        frozen_u : dict {pos: 0} — frozen U positions (1-indexed)
        frozen_v : dict {pos: 0} — frozen V positions (1-indexed)

        Returns
        -------
        u_hat : dict {pos: int}
        v_hat : dict {pos: int}
        """
        self.model.eval()
        device = z_single.device
        N = z_single.shape[0]
        n = N.bit_length() - 1 if isinstance(N, int) else int(math.log2(N))
        N = int(N)
        assert (1 << n) == N
        d = self.model.d
        L = self.L

        # Compute root embedding once: (1, N, d)
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.model.embedding_z(z_single.unsqueeze(0))[:, br]  # (1, N, d)

        # No-info embedding
        no_info = self.model.no_info_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, d)

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

        # Each path: (edge_data, dec_head, u_hat, v_hat, path_metric)
        paths = [{
            'ed': _make_edge_data(),
            'dh': 1,
            'uh': {},
            'vh': {},
            'pm': 0.0,
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

            # Step 1: navigate each path to target_vtx and compute logits
            path_logits = []
            for p in paths:
                ed = p['ed']
                dh = p['dh']

                # Navigate to target vertex
                dh = self._step_to_path(dh, target_vtx, ed)
                p['dh'] = dh

                # Save bottom-up message
                temp = ed[leaf_edge][:, 0].clone()  # (1, d)

                # Compute top-down
                if leaf_edge & 1 == 0:
                    self.model._neural_calc_left(target_vtx, ed)
                else:
                    self.model._neural_calc_right(target_vtx, ed)
                top_down = ed[leaf_edge][:, 0]  # (1, d)

                # Combine
                if self.model.use_combine_nn:
                    combined = self.model.combine_nn(
                        torch.cat([top_down, temp], dim=-1))
                else:
                    combined = top_down + temp

                logits = self.model.emb2logits(combined)  # (1, 4)
                log_probs = F.log_softmax(logits, dim=-1)  # (1, 4)
                path_logits.append(log_probs[0])  # (4,)

            if is_frozen:
                frozen_bit = fdict[i_t]
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]

                    if gamma == 0:
                        p['uh'][i_t] = torch.tensor([float(frozen_bit)], device=device)
                    else:
                        p['vh'][i_t] = torch.tensor([float(frozen_bit)], device=device)

                    # Path metric update for frozen leaf
                    if gamma == 0:
                        u_val = frozen_bit
                        if i_t in frozen_v:
                            v_val = frozen_v[i_t]
                            idx = u_val * 2 + v_val
                            p['pm'] += lp[idx].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[u_val*2 : u_val*2+2], dim=0).item()
                    else:
                        v_val = frozen_bit
                        if i_t in frozen_u:
                            u_val = frozen_u[i_t]
                            idx = u_val * 2 + v_val
                            p['pm'] += lp[idx].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[[v_val, v_val + 2]], dim=0).item()

                    # Set leaf embedding
                    new_emb = self.model._make_leaf_emb(
                        p['uh'].get(i_t), p['vh'].get(i_t), 1, device)
                    p['ed'][leaf_edge] = new_emb.unsqueeze(1)

            else:
                # Non-frozen leaf: fork
                candidates = []
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]

                    if gamma == 0:
                        options = [
                            (0.0, torch.logsumexp(lp[:2], dim=0).item()),
                            (1.0, torch.logsumexp(lp[2:], dim=0).item()),
                        ]
                    else:
                        options = [
                            (0.0, torch.logsumexp(lp[[0, 2]], dim=0).item()),
                            (1.0, torch.logsumexp(lp[[1, 3]], dim=0).item()),
                        ]

                    for bit_val, log_p in options:
                        new_pm = p['pm'] + log_p
                        candidates.append((new_pm, pidx, bit_val))

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
                    bit_tensor = torch.tensor([bit_val], device=device)
                    if gamma == 0:
                        new_p['uh'][i_t] = bit_tensor
                    else:
                        new_p['vh'][i_t] = bit_tensor

                    new_emb = self.model._make_leaf_emb(
                        new_p['uh'].get(i_t), new_p['vh'].get(i_t), 1, device)
                    new_p['ed'][leaf_edge] = new_emb.unsqueeze(1)
                    new_paths.append(new_p)

                paths = new_paths

        # Return best path
        best = max(paths, key=lambda p: p['pm'])
        u_hat = {k: int(v[0].item()) for k, v in best['uh'].items()}
        v_hat = {k: int(v[0].item()) for k, v in best['vh'].items()}
        return u_hat, v_hat

    def _step_to_path(self, current, target, edge_data):
        if current == target:
            return current
        path = self.model._get_path(current, target)
        for beta in path:
            current = self._step_one_path(current, beta, edge_data)
        return current

    def _step_one_path(self, current, beta, edge_data):
        if current == beta >> 1:
            if beta & 1 == 0:
                self.model._neural_calc_left(current, edge_data)
            else:
                self.model._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            self.model._pure_neural_calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: current={current}, target={beta}")


# ─── Evaluation helpers ─────────────────────────────────────────────────────

def load_bemac_model(N, device='cpu'):
    """Load PureNeuralCompGraphDecoder for BEMAC."""
    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=3)
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    ckpt = os.path.join(save_dir, f'ncg_pure_neural_N{N}.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint: {ckpt}")
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def load_bemac_design(N, Ru=0.50, Rv=0.70):
    """Load BEMAC Class B design from npz."""
    n = int(math.log2(N))
    ku = int(N * Ru)
    kv = int(N * Rv)
    dp = os.path.join(os.path.dirname(__file__), '..', 'designs', f'bemac_B_n{n}.npz')
    if os.path.exists(dp):
        d = np.load(dp)
        su = np.argsort(d['u_error_rates'])
        sv = np.argsort(d['v_error_rates'])
        Au = sorted([int(i+1) for i in su[:ku]])
        Av = sorted([int(i+1) for i in sv[:kv]])
    else:
        raise FileNotFoundError(f"Design file not found: {dp}")
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv, ku, kv


def evaluate_nn_sc_bemac(model, N, b, Au, Av, fu, fv, n_cw, batch_size=50):
    """Evaluate NN-SC (greedy, L=1) on BEMAC."""
    model.eval()
    errs = 0
    total = 0
    rng = np.random.default_rng(999)

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy((xf + yf).astype(np.int64)).long()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e:
                    errs += 1
            total += actual
    return errs / total


def evaluate_nn_scl_bemac(model, N, b, Au, Av, fu, fv, n_cw, L=4):
    """Evaluate NN-SCL on BEMAC (one codeword at a time)."""
    decoder = BemacNeuralSCLDecoder(model, L=L)
    errs = 0
    total = 0
    rng = np.random.default_rng(999)

    with torch.no_grad():
        for _ in range(n_cw):
            uf = np.zeros(N, dtype=int)
            vf = np.zeros(N, dtype=int)
            for p in Au: uf[p-1] = rng.integers(0, 2)
            for p in Av: vf[p-1] = rng.integers(0, 2)
            xf = polar_encode_batch(uf.reshape(1, N))
            yf = polar_encode_batch(vf.reshape(1, N))
            zf = torch.from_numpy((xf + yf).astype(np.int64)).long()
            z_single = zf[0]

            uh, vh = decoder.decode(z_single, b, fu, fv)

            e = any(uh.get(p, 0) != uf[p-1] for p in Au) or \
                any(vh.get(p, 0) != vf[p-1] for p in Av)
            if e:
                errs += 1
            total += 1

            if total % 200 == 0:
                print(f"    [{total}/{n_cw}] errs={errs} BLER={errs/total:.4f}")

    return errs / total


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    RU, RV = 0.50, 0.70
    N_CW = 3000
    PATH_CLASS = 'B'

    # Reference SC BLERs from bemac_nn_vs_sc_complete.json
    SC_REF = {32: 0.008, 64: 0.0056, 128: 0.002}
    NN_SC_REF = {32: 0.0088, 64: 0.003, 128: 0.0012}

    results = {}

    print(f"\n{'='*72}")
    print(f"BEMAC NN-SCL Evaluation — Class {PATH_CLASS}, Ru={RU}, Rv={RV}")
    print(f"{'='*72}")

    for N in [32, 64, 128]:
        n = int(math.log2(N))

        # Load design
        Au, Av, fu, fv, ku, kv = load_bemac_design(N, RU, RV)
        b = make_path(N, N // 2)  # Class B

        # Load model
        try:
            model = load_bemac_model(N)
        except FileNotFoundError as e:
            print(f"\n  N={N}: {e} -- SKIPPING")
            continue

        print(f"\n{'~'*72}")
        print(f"  N={N}, ku={ku}, kv={kv}")
        print(f"  Model params: {model.count_parameters():,}")
        print(f"  SC ref BLER:    {SC_REF.get(N, '?')}")
        print(f"  NN-SC ref BLER: {NN_SC_REF.get(N, '?')}")

        res = {'N': N, 'ku': ku, 'kv': kv,
               'sc_ref': SC_REF.get(N), 'nn_sc_ref': NN_SC_REF.get(N)}

        # NN-SC (greedy, L=1) — verify we match reference
        t0 = time.time()
        bs = min(50, max(4, 200 // (N // 16)))
        bler_sc = evaluate_nn_sc_bemac(model, N, b, Au, Av, fu, fv, N_CW, batch_size=bs)
        t_sc = time.time() - t0
        print(f"  NN-SC  (L=1):  BLER={bler_sc:.4f}  [{t_sc:.1f}s]")
        res['nn_sc_bler'] = bler_sc
        res['nn_sc_time'] = round(t_sc, 1)

        # NN-SCL with L=4
        n_cw_l4 = N_CW
        if N >= 128:
            n_cw_l4 = min(N_CW, 1500)  # slower, reduce count

        t0 = time.time()
        bler_scl4 = evaluate_nn_scl_bemac(model, N, b, Au, Av, fu, fv, n_cw_l4, L=4)
        t_scl4 = time.time() - t0
        print(f"  NN-SCL (L=4):  BLER={bler_scl4:.4f}  [{t_scl4:.1f}s, {n_cw_l4} cw]")
        res['nn_scl4_bler'] = bler_scl4
        res['nn_scl4_cw'] = n_cw_l4
        res['nn_scl4_time'] = round(t_scl4, 1)

        # Compute ratios
        sc_ref = SC_REF.get(N, 0)
        if sc_ref > 0:
            res['nn_sc_vs_sc'] = round(bler_sc / sc_ref, 3)
            res['nn_scl4_vs_sc'] = round(bler_scl4 / sc_ref, 3) if bler_scl4 > 0 else 0
            res['nn_scl4_vs_nn_sc'] = round(bler_scl4 / bler_sc, 3) if bler_sc > 0 else 0

        # Summary
        print(f"\n  Summary for N={N}:")
        print(f"    {'Decoder':<18s}  {'BLER':>8s}  {'vs SC':>8s}  {'vs NN-SC':>10s}")
        print(f"    {'SC (analytical)':<18s}  {sc_ref:>8.4f}  {'1.00x':>8s}  {'':>10s}")
        print(f"    {'NN-SC (L=1)':<18s}  {bler_sc:>8.4f}  {bler_sc/sc_ref:>7.2f}x  {'1.00x':>10s}" if sc_ref else "")
        nn_sc_str = f"{bler_scl4/bler_sc:>9.2f}x" if bler_sc > 0 else "N/A"
        print(f"    {'NN-SCL (L=4)':<18s}  {bler_scl4:>8.4f}  {bler_scl4/sc_ref:>7.2f}x  {nn_sc_str}" if sc_ref else "")

        results[str(N)] = res

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'bemac',
                           'bemac_classB_Ru50_Rv70_nn_scl')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'bemac_nn_scl_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*72}")
    print("Done.")
    return results


if __name__ == '__main__':
    main()
