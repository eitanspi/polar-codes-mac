#!/usr/bin/env python3
"""
neural_scl.py — Neural SC List (SCL) decoder for MAC polar codes.

Uses a trained PureNeuralCompGraphDecoder (wrapped in SimpleMLP_Gmac) to produce
4-class log-probabilities at each leaf.  Instead of greedy argmax, maintains L
candidate paths and prunes by cumulative log-probability.

Key idea: maintain L copies of the tree state (edge_data).  At each non-frozen
leaf, fork each path into up to 4 candidates (one per (u,v) outcome), score them,
and keep the top L.

For frozen leaves only one outcome is possible, so no forking occurs.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from encoder import polar_encode_batch, bit_reversal_perm
from channels import GaussianMAC
from design import make_path, design_gmac
from ncg_pure_neural import PureNeuralCompGraphDecoder


# ─── SimpleMLP_Gmac (same as train_gmac_48hr.py) ────────────────────────────

class SimpleMLP_Gmac(nn.Module):
    def __init__(self, d=16, hidden=64, n_layers=2, z_hidden=32):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden), nn.ELU(), nn.Linear(z_hidden, d),
        )
        self.tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers)

    def forward(self, z, b, fu, fv, u_true=None, v_true=None):
        n = z.shape[1].bit_length() - 1
        br = torch.from_numpy(bit_reversal_perm(n)).long()
        root = self.z_encoder(z.unsqueeze(-1))[:, br]
        return self.tree(z=None, b=b, frozen_u=fu, frozen_v=fv,
                         u_true=u_true, v_true=v_true, root_emb=root)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Neural SCL Decoder ─────────────────────────────────────────────────────

class NeuralSCLDecoder:
    """
    Neural SC List decoder for MAC polar codes.

    Maintains L candidate paths through the sequential tree walk.
    At each non-frozen leaf the 4 possible (u,v) outcomes are enumerated,
    giving up to 4*L candidate paths; the best L are kept.

    Parameters
    ----------
    model : SimpleMLP_Gmac — trained model (used in eval mode)
    L     : int            — list size
    """

    def __init__(self, model, L=4):
        self.model = model
        self.L = L
        self.tree = model.tree   # the PureNeuralCompGraphDecoder

    @torch.no_grad()
    def decode(self, z_single, b, frozen_u, frozen_v):
        """
        Decode a SINGLE codeword with SCL.

        Parameters
        ----------
        z_single : (N,) tensor — single channel output
        b        : list of 2N ints — path vector
        frozen_u : dict {pos: 0} — frozen U positions (1-indexed)
        frozen_v : dict {pos: 0} — frozen V positions (1-indexed)

        Returns
        -------
        u_hat : dict {pos: int} — decoded U bits (1-indexed)
        v_hat : dict {pos: int} — decoded V bits (1-indexed)
        """
        self.model.eval()
        device = z_single.device
        N = z_single.shape[0]
        n = N.bit_length() - 1
        assert (1 << n) == N
        d = self.tree.d
        L = self.L

        # Compute root embedding once: (1, N, d)
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.model.z_encoder(z_single.unsqueeze(-1).unsqueeze(0))[:, br]  # (1, N, d)

        # Initialize L paths (start with 1 path, expand as needed)
        no_info = self.tree.no_info_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, d)

        def _make_edge_data():
            """Create fresh edge_data for one path, batch=1."""
            ed = [None] * (2 * N)
            ed[1] = root.clone()  # (1, N, d)
            for beta in range(2, 2 * N):
                level = beta.bit_length() - 1
                size = N >> level
                ed[beta] = no_info.expand(1, size, d).clone()
            return ed

        def _clone_edge_data(ed):
            """Deep-clone edge_data list."""
            return [e.clone() if e is not None else None for e in ed]

        # Each path: (edge_data, dec_head, u_hat, v_hat, path_metric)
        paths = [{
            'ed': _make_edge_data(),
            'dh': 1,        # dec_head (current vertex)
            'uh': {},       # u_hat decisions
            'vh': {},       # v_hat decisions
            'pm': 0.0,     # path metric (cumulative log-prob)
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
                    self.tree._neural_calc_left(target_vtx, ed)
                else:
                    self.tree._neural_calc_right(target_vtx, ed)
                top_down = ed[leaf_edge][:, 0]  # (1, d)

                # Combine
                if self.tree.use_combine_nn:
                    combined = self.tree.combine_nn(
                        torch.cat([top_down, temp], dim=-1))
                else:
                    combined = top_down + temp

                logits = self.tree.emb2logits(combined)  # (1, 4)
                log_probs = F.log_softmax(logits, dim=-1)  # (1, 4)
                path_logits.append(log_probs[0])  # (4,)

            if is_frozen:
                # Frozen leaf: only one option (frozen value = 0)
                frozen_bit = fdict[i_t]
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]

                    # Store as (1,) tensor for compatibility with _make_leaf_emb
                    if gamma == 0:
                        p['uh'][i_t] = torch.tensor([float(frozen_bit)], device=device)
                    else:
                        p['vh'][i_t] = torch.tensor([float(frozen_bit)], device=device)

                    # Compute log-prob for frozen outcome
                    # Marginalise over the other user's bit
                    if gamma == 0:
                        # u is frozen, v unknown at this leaf
                        u_val = frozen_bit
                        if i_t in frozen_v:
                            v_val = frozen_v[i_t]
                            idx = u_val * 2 + v_val
                            p['pm'] += lp[idx].item()
                        else:
                            # marginalise over v
                            p['pm'] += torch.logsumexp(lp[u_val*2 : u_val*2+2], dim=0).item()
                    else:
                        # v is frozen, u unknown at this leaf
                        v_val = frozen_bit
                        if i_t in frozen_u:
                            u_val = frozen_u[i_t]
                            idx = u_val * 2 + v_val
                            p['pm'] += lp[idx].item()
                        else:
                            # marginalise over u
                            p['pm'] += torch.logsumexp(lp[[v_val, v_val + 2]], dim=0).item()

                    # Set leaf embedding
                    new_emb = self.tree._make_leaf_emb(
                        p['uh'].get(i_t), p['vh'].get(i_t), 1, device)
                    p['ed'][leaf_edge] = new_emb.unsqueeze(1)

            else:
                # Non-frozen leaf: fork each path into candidates
                candidates = []

                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]  # (4,) log-probs over (00, 01, 10, 11)

                    if gamma == 0:
                        # Deciding u bit. Marginalise over v.
                        # Option u=0: log P(u=0) = logsumexp(lp[0], lp[1])
                        # Option u=1: log P(u=1) = logsumexp(lp[2], lp[3])
                        options = [
                            (0.0, torch.logsumexp(lp[:2], dim=0).item()),
                            (1.0, torch.logsumexp(lp[2:], dim=0).item()),
                        ]
                    else:
                        # Deciding v bit. Marginalise over u.
                        # Option v=0: logsumexp(lp[0], lp[2])
                        # Option v=1: logsumexp(lp[1], lp[3])
                        options = [
                            (0.0, torch.logsumexp(lp[[0, 2]], dim=0).item()),
                            (1.0, torch.logsumexp(lp[[1, 3]], dim=0).item()),
                        ]

                    for bit_val, log_p in options:
                        new_pm = p['pm'] + log_p
                        candidates.append((new_pm, pidx, bit_val))

                # Sort candidates by path metric (descending = best first)
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:L]

                # Build new paths
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

                    # Set leaf embedding
                    new_emb = self.tree._make_leaf_emb(
                        new_p['uh'].get(i_t), new_p['vh'].get(i_t), 1, device)
                    new_p['ed'][leaf_edge] = new_emb.unsqueeze(1)
                    new_paths.append(new_p)

                paths = new_paths

        # Return best path
        best = max(paths, key=lambda p: p['pm'])
        # Convert to int dicts (values are (1,) tensors)
        u_hat = {k: int(v[0].item()) for k, v in best['uh'].items()}
        v_hat = {k: int(v[0].item()) for k, v in best['vh'].items()}
        return u_hat, v_hat

    def _step_to_path(self, current, target, edge_data):
        """Navigate from current to target vertex, updating edge_data."""
        if current == target:
            return current
        path = self.tree._get_path(current, target)
        for beta in path:
            current = self._step_one_path(current, beta, edge_data)
        return current

    def _step_one_path(self, current, beta, edge_data):
        """Single navigation step (no distillation)."""
        if current == beta >> 1:
            # Going DOWN
            if beta & 1 == 0:
                self.tree._neural_calc_left(current, edge_data)
            else:
                self.tree._neural_calc_right(current, edge_data)
            return beta
        elif beta == current >> 1:
            # Going UP
            self.tree._pure_neural_calc_parent(current, edge_data)
            return beta
        else:
            raise ValueError(f"Invalid step: current={current}, target={beta}")


# ─── Batch evaluation helpers ───────────────────────────────────────────────

def load_model(N, device='cpu'):
    """Load trained SimpleMLP_Gmac checkpoint for given N."""
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)

    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    ckpt = os.path.join(save_dir, f'ncg_gmac_mlp_N{N}.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint: {ckpt}")

    sd = torch.load(ckpt, map_location=device, weights_only=True)
    # Handle z_enc vs z_encoder key mismatch
    fixed = {}
    for k, v in sd.items():
        nk = k.replace('z_enc.', 'z_encoder.') if k.startswith('z_enc.') else k
        fixed[nk] = v
    model.load_state_dict(fixed, strict=False)
    model.to(device)
    model.eval()
    return model


def load_design_npz(N, ku, kv):
    """Load GMAC design from npz files (matches training scripts)."""
    n = int(math.log2(N))
    dp = os.path.join(os.path.dirname(__file__), '..', 'to_git', 'designs',
                      f'gmac_B_n{n}_snr6dB.npz')
    if os.path.exists(dp):
        d = np.load(dp)
        su = np.argsort(d['u_error_rates'])
        sv = np.argsort(d['v_error_rates'])
        Au = sorted([int(i+1) for i in su[:ku]])
        Av = sorted([int(i+1) for i in sv[:kv]])
    else:
        # Fallback to GA design
        sigma2 = 10 ** (-6.0 / 10)
        Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, sigma2, method='ga')
        return Au, Av, fu, fv
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


def evaluate_nn_sc(model, channel, N, b, Au, Av, fu, fv, n_cw, batch_size=25):
    """Evaluate NN-SC (greedy, L=1) decoder."""
    model.eval()
    errs = 0
    total = 0
    rng = np.random.default_rng(999)
    bs = min(batch_size, n_cw)
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            _, _, uh, vh, _ = model(zf, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e:
                    errs += 1
            total += actual
    return errs / total


def evaluate_nn_scl(model, channel, N, b, Au, Av, fu, fv, n_cw, L=4):
    """Evaluate NN-SCL decoder (processes one codeword at a time)."""
    decoder = NeuralSCLDecoder(model, L=L)
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
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()
            z_single = zf[0]  # (N,)

            uh, vh = decoder.decode(z_single, b, fu, fv)

            e = any(uh.get(p, 0) != uf[p-1] for p in Au) or \
                any(vh.get(p, 0) != vf[p-1] for p in Av)
            if e:
                errs += 1
            total += 1
    return errs / total


# ─── Main test ──────────────────────────────────────────────────────────────

def main():
    import time

    SNR_DB = 6.0
    SIGMA2 = 10 ** (-SNR_DB / 10)
    N_CW = 3000

    # SC reference BLERs at 6dB (from train_gmac_48hr campaign)
    SC_REF = {
        32:  0.042,
        64:  0.025,
        128: 0.015,
    }

    # Rates from the training scripts (train_gmac_48hr.py)
    # N=32 was trained with different ku/kv (from run_gmac_mlp scripts)
    RATES = {
        32:  {'ku': 7,  'kv': 15},
        64:  {'ku': 31, 'kv': 31},
        128: {'ku': 62, 'kv': 62},
    }

    channel = GaussianMAC(sigma2=SIGMA2)

    print(f"\n{'='*72}")
    print(f"Neural SCL Decoder Evaluation — GMAC SNR={SNR_DB}dB, {N_CW} codewords")
    print(f"{'='*72}")

    for N in [32, 64, 128]:
        n = int(math.log2(N))
        ku = RATES[N]['ku']
        kv = RATES[N]['kv']

        # Load design from npz (same as training)
        Au, Av, fu, fv = load_design_npz(N, ku, kv)
        b = make_path(N, N // 2)  # Class B

        # Load model
        try:
            model = load_model(N)
        except FileNotFoundError as e:
            print(f"\n  N={N}: {e} -- SKIPPING")
            continue

        print(f"\n{'─'*72}")
        print(f"  N={N}, ku={ku}, kv={kv}, SC ref BLER={SC_REF.get(N, '?')}")
        print(f"  Model params: {model.count_parameters():,}")

        # NN-SC (greedy, L=1)
        t0 = time.time()
        bler_sc = evaluate_nn_sc(model, channel, N, b, Au, Av, fu, fv, N_CW,
                                 batch_size=min(50, max(2, 200 // (N // 16))))
        t_sc = time.time() - t0
        print(f"  NN-SC  (L=1):  BLER={bler_sc:.4f}  [{t_sc:.1f}s]")

        # NN-SCL with L=4
        n_cw_scl = min(N_CW, 1000) if N >= 128 else N_CW
        t0 = time.time()
        bler_scl4 = evaluate_nn_scl(model, channel, N, b, Au, Av, fu, fv,
                                     n_cw_scl, L=4)
        t_scl4 = time.time() - t0
        print(f"  NN-SCL (L=4):  BLER={bler_scl4:.4f}  [{t_scl4:.1f}s, {n_cw_scl} cw]")

        # NN-SCL with L=8
        n_cw_scl8 = min(N_CW, 500) if N >= 128 else min(N_CW, 1000)
        t0 = time.time()
        bler_scl8 = evaluate_nn_scl(model, channel, N, b, Au, Av, fu, fv,
                                     n_cw_scl8, L=8)
        t_scl8 = time.time() - t0
        print(f"  NN-SCL (L=8):  BLER={bler_scl8:.4f}  [{t_scl8:.1f}s, {n_cw_scl8} cw]")

        # Summary
        sc_ref = SC_REF.get(N, None)
        print(f"\n  Summary for N={N}:")
        print(f"    {'Decoder':<16s}  {'BLER':>8s}  {'vs SC':>8s}")
        if sc_ref:
            print(f"    {'Analytical SC':<16s}  {sc_ref:>8.4f}  {'1.00x':>8s}")
        print(f"    {'NN-SC (L=1)':<16s}  {bler_sc:>8.4f}  "
              f"{bler_sc/sc_ref:>7.2f}x" if sc_ref else "")
        print(f"    {'NN-SCL (L=4)':<16s}  {bler_scl4:>8.4f}  "
              f"{bler_scl4/sc_ref:>7.2f}x" if sc_ref else "")
        print(f"    {'NN-SCL (L=8)':<16s}  {bler_scl8:>8.4f}  "
              f"{bler_scl8/sc_ref:>7.2f}x" if sc_ref else "")

    print(f"\n{'='*72}")
    print("Done.")


if __name__ == '__main__':
    main()
