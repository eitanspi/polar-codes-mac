"""
test_residual_collapse.py — Test whether residual connections in CalcLeft/CalcRight
prevent embedding range collapse across tree levels.

Creates two models (baseline vs residual variant) with random weights,
walks down the leftmost CalcLeft chain at N=256, and compares embedding stats.
"""

import sys
import os
import copy
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from polar.encoder import bit_reversal_perm
from polar.channels import GaussianMAC

# ─── MLP helper (from ncg_gmac.py) ─────────────────────────────────────────

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


# ─── Minimal model for this test ────────────────────────────────────────────

class CalcLeftTestModel(nn.Module):
    """Minimal model with z_encoder + calc_left_nn for testing embedding propagation."""

    def __init__(self, d=16, hidden=64, n_layers=2, z_hidden=32, use_residual=False):
        super().__init__()
        self.d = d
        self.use_residual = use_residual

        # Continuous channel embedding
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_hidden),
            nn.ELU(),
            nn.Linear(z_hidden, d),
        )

        # CalcLeft and CalcRight
        self.calc_left_nn = _make_mlp(3 * d, hidden, d, n_layers)
        self.calc_right_nn = _make_mlp(3 * d, hidden, d, n_layers)

        # No-info embedding
        self.no_info_emb = nn.Parameter(torch.randn(d) * 0.01)

    def calc_left(self, beta, edge_data):
        parent = edge_data[beta]
        right = edge_data[2 * beta + 1]
        l = right.shape[1]
        p_first = parent[:, :l]
        p_second = parent[:, l:]
        inp = torch.cat([p_first, p_second, right], dim=-1)
        result = self.calc_left_nn(inp)
        if self.use_residual:
            result = result + p_first  # Skip connection from parent first half
        edge_data[2 * beta] = result

    def calc_right(self, beta, edge_data):
        parent = edge_data[beta]
        left = edge_data[2 * beta]
        l = left.shape[1]
        p_first = parent[:, :l]
        p_second = parent[:, l:]
        inp = torch.cat([p_first, p_second, left], dim=-1)
        result = self.calc_right_nn(inp)
        if self.use_residual:
            result = result + p_first  # Skip connection from parent first half
        edge_data[2 * beta + 1] = result


def run_test():
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    n = int(np.log2(N))
    B = 32  # batch size
    d = 16
    hidden = 64
    sigma2 = 10 ** (-6.0 / 10.0)

    # Generate channel output
    channel = GaussianMAC(sigma2=sigma2)
    X = np.random.randint(0, 2, (B, N))
    Y = np.random.randint(0, 2, (B, N))
    Z = channel.sample_batch(X, Y)

    z_tensor = torch.tensor(Z, dtype=torch.float32)
    br = torch.from_numpy(bit_reversal_perm(n)).long()

    # Create baseline model
    torch.manual_seed(123)
    baseline = CalcLeftTestModel(d=d, hidden=hidden, use_residual=False)

    # Create residual model with SAME initial weights
    residual = CalcLeftTestModel(d=d, hidden=hidden, use_residual=True)
    residual.load_state_dict(baseline.state_dict())

    # Leftmost chain: vertex 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128
    # These are vertices. Edge for vertex beta's parent-edge is beta itself.
    # Edge 1 = root. CalcLeft at vertex beta writes edge 2*beta.
    # So walking left: edge 1 -> calcLeft(1) -> edge 2 -> calcLeft(2) -> edge 4 -> ...

    models = {'Baseline': baseline, 'Residual': residual}
    results = {}

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            # Initialize root embedding
            root = model.z_encoder(z_tensor.unsqueeze(-1))[:, br]  # (B, N, d)

            # Initialize edge_data
            edge_data = [None] * (2 * N)
            edge_data[1] = root  # (B, N, d)

            no_info = model.no_info_emb.unsqueeze(0).unsqueeze(0)
            for beta in range(2, 2 * N):
                level = beta.bit_length() - 1
                size = N >> level
                edge_data[beta] = no_info.expand(B, size, d).clone()

            # Record stats at root
            stats = []
            emb = edge_data[1]  # root: (B, N, d)
            vals = emb.flatten()
            stats.append({
                'level': 0,
                'vertex': 'root',
                'edge': 1,
                'shape': tuple(emb.shape),
                'min': vals.min().item(),
                'max': vals.max().item(),
                'mean': vals.mean().item(),
                'std': vals.std().item(),
            })

            # Walk down leftmost CalcLeft chain
            # CalcLeft at vertex 1 -> writes edge 2 (size N/2)
            # CalcLeft at vertex 2 -> writes edge 4 (size N/4)
            # ...
            # CalcLeft at vertex 128 -> writes edge 256 (size 1, leaf)
            vertices = [1, 2, 4, 8, 16, 32, 64, 128]
            for i, vtx in enumerate(vertices):
                model.calc_left(vtx, edge_data)
                child_edge = 2 * vtx
                emb = edge_data[child_edge]
                vals = emb.flatten()
                stats.append({
                    'level': i + 1,
                    'vertex': vtx,
                    'edge': child_edge,
                    'shape': tuple(emb.shape),
                    'min': vals.min().item(),
                    'max': vals.max().item(),
                    'mean': vals.mean().item(),
                    'std': vals.std().item(),
                })

            results[name] = stats

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"Embedding Range Collapse Test — N={N}, d={d}, hidden={hidden}, batch={B}")
    print(f"GaussianMAC sigma2={sigma2:.4f} (SNR=6dB), random weights (no training)")
    print(f"Walking leftmost CalcLeft chain: root -> vertex 1 -> 2 -> 4 -> ... -> 128")
    print(f"{'='*100}\n")

    header = f"{'Level':>5}  {'Edge':>5}  {'Shape':>14}  |  {'Min':>9} {'Max':>9} {'Mean':>9} {'Std':>9}  |  {'Min':>9} {'Max':>9} {'Mean':>9} {'Std':>9}"
    print(f"{'':>5}  {'':>5}  {'':>14}  |  {'--- Baseline ---':^39}  |  {'--- Residual ---':^39}")
    print(header)
    print('-' * 120)

    for i in range(len(results['Baseline'])):
        b = results['Baseline'][i]
        r = results['Residual'][i]
        print(f"{b['level']:>5}  {b['edge']:>5}  {str(b['shape']):>14}  |  "
              f"{b['min']:>9.4f} {b['max']:>9.4f} {b['mean']:>9.4f} {b['std']:>9.4f}  |  "
              f"{r['min']:>9.4f} {r['max']:>9.4f} {r['mean']:>9.4f} {r['std']:>9.4f}")

    print()

    # Summary: ratio of leaf std to root std
    b_root_std = results['Baseline'][0]['std']
    b_leaf_std = results['Baseline'][-1]['std']
    r_root_std = results['Residual'][0]['std']
    r_leaf_std = results['Residual'][-1]['std']

    print(f"Std retention (leaf/root):  Baseline = {b_leaf_std/b_root_std:.4f}   "
          f"Residual = {r_leaf_std/r_root_std:.4f}")

    b_root_range = results['Baseline'][0]['max'] - results['Baseline'][0]['min']
    b_leaf_range = results['Baseline'][-1]['max'] - results['Baseline'][-1]['min']
    r_root_range = results['Residual'][0]['max'] - results['Residual'][0]['min']
    r_leaf_range = results['Residual'][-1]['max'] - results['Residual'][-1]['min']

    print(f"Range retention (leaf/root): Baseline = {b_leaf_range/b_root_range:.4f}   "
          f"Residual = {r_leaf_range/r_root_range:.4f}")
    print()


if __name__ == '__main__':
    run_test()
