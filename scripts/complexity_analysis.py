#!/usr/bin/env python3
"""
complexity_analysis.py — Measure computational complexity of all decoder variants.

Produces:
  - FLOPs per codeword (NN-SC vs analytical SC)
  - Inference time (ms/codeword)
  - Model size (parameters, memory)
  - Training time to reach within 10% of SC

Output: docs/complexity_analysis.md + results/complexity_analysis.json
"""

import os, sys, json, time, math
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.decoder import decode_single
from polar.decoder_scl import decode_single_list
from polar.channels import BEMAC, GaussianMAC
from polar.design import design_bemac, make_path, design_gmac
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder
from neural.ncg_gmac import GmacNeuralCompGraphDecoder


def count_flops_nn(model):
    """Count FLOPs for a single forward pass through the NN model (approximate).

    For each Linear(in, out): 2*in*out FLOPs (multiply + add).
    """
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total += 2 * module.in_features * module.out_features
    return total


def count_tree_operations(N, b):
    """Count the number of CalcLeft, CalcRight, CalcParent operations in a tree walk.

    Simulates the navigation path through the binary tree.
    """
    n = int(math.log2(N))

    calc_left = 0
    calc_right = 0
    calc_parent = 0

    dec_head = 1
    i_u, i_v = 0, 0

    def get_path(current, target):
        if current == target:
            return []
        path_up, path_down = [], []
        c, t = current, target
        while c != t:
            if c > t:
                c >>= 1
                path_up.append(c)
            else:
                path_down.append(t)
                t >>= 1
        path_down.reverse()
        return path_up + path_down

    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u
        else:
            i_v += 1; i_t = i_v

        leaf_edge = i_t + N - 1
        target_vtx = leaf_edge >> 1

        # Navigate to target
        path = get_path(dec_head, target_vtx)
        for beta in path:
            if dec_head == beta >> 1:
                if beta & 1 == 0:
                    calc_left += 1
                else:
                    calc_right += 1
                dec_head = beta
            elif beta == dec_head >> 1:
                calc_parent += 1
                dec_head = beta

        # Final CalcLeft/CalcRight at leaf
        if leaf_edge & 1 == 0:
            calc_left += 1
        else:
            calc_right += 1

        dec_head = target_vtx

    return calc_left, calc_right, calc_parent


def estimate_nn_flops_per_codeword(N, d=16, hidden=64, b=None):
    """Estimate total FLOPs for NN-SC decoder on one codeword."""
    if b is None:
        b = make_path(N, N//2)

    n_cl, n_cr, n_cp = count_tree_operations(N, b)

    # FLOPs per operation:
    # CalcLeft/Right MLP: Linear(3d, hidden) + Linear(hidden, hidden) + Linear(hidden, d)
    #   = 2*(3d*hidden + hidden*hidden + hidden*d)
    flops_cl_cr = 2 * (3*d*hidden + hidden*hidden + hidden*d)

    # CalcParent: gate_net (Linear(2d,hidden) + Linear(hidden,d)) + candidate_net (same as CalcLeft)
    #   + parent_second_nn: Linear(d,d)
    flops_gate = 2 * (2*d*hidden + hidden*d)
    flops_candidate = 2 * (2*d*hidden + hidden*hidden + hidden*d)
    flops_second = 2 * d * d
    flops_cp = flops_gate + flops_candidate + flops_second

    # emb2logits: MLP(d, hidden, hidden, 4) = 2*(d*hidden + hidden*hidden + hidden*4)
    flops_emb2logits = 2 * (d*hidden + hidden*hidden + hidden*4)

    # logits2emb: MLP(4, hidden, hidden, d) = 2*(4*hidden + hidden*hidden + hidden*d)
    flops_logits2emb = 2 * (4*hidden + hidden*hidden + hidden*d)

    # z_encoder (GMAC): Linear(1,32) + Linear(32,d) = 2*(32 + 32*d) per position
    flops_z_encoder = N * 2 * (32 + 32*d)

    # Embedding (BEMAC): lookup, negligible
    flops_embedding_bemac = 0

    # Per-leaf operations: emb2logits + logits2emb = per non-frozen leaf
    # Assume ~N non-frozen leaves (info bits for both users)
    n_leaves = 2 * N  # total leaves visited

    total = (n_cl * flops_cl_cr +
             n_cr * flops_cl_cr +
             n_cp * flops_cp +
             n_leaves * (flops_emb2logits + flops_logits2emb))

    return {
        'total_flops': total,
        'n_calc_left': n_cl,
        'n_calc_right': n_cr,
        'n_calc_parent': n_cp,
        'total_mlp_calls': n_cl + n_cr + n_cp + 2 * n_leaves,
        'flops_per_calc_left_right': flops_cl_cr,
        'flops_per_calc_parent': flops_cp,
    }


def estimate_sc_flops_per_codeword(N, b=None):
    """Estimate FLOPs for analytical SC decoder on one codeword."""
    if b is None:
        b = make_path(N, N//2)

    n_cl, n_cr, n_cp = count_tree_operations(N, b)

    # Each analytical operation works on 2x2 tensors (log-probabilities)
    # CalcLeft (circular convolution): ~32 multiplications + 16 additions + 4 logsumexp
    # Each logsumexp of 4 terms: ~12 ops (exp, add, log)
    flops_cl = 32 + 16 + 4 * 12  # ~96 per 2x2 tensor pair

    # CalcRight (normalized product): ~8 multiplications + 4 additions + normalization
    flops_cr = 8 + 4 + 8  # ~20 per 2x2 tensor pair

    # CalcParent (inverse circ conv): same as CalcLeft
    flops_cp = 96

    # But: operations work on edge_data which contains N/level tensors
    # At level l, each tensor pair has N >> l elements
    # Total work = sum over all operations of (N >> level) * flops_per_tensor

    # Simplified: at level l, tensor size = N >> l
    # CalcLeft/Right at each level: one call processes all N >> l tensors
    # But SC visits specific nodes, not full levels

    # More accurate: each individual 2x2 tensor operation is ~96 FLOPs (circ_conv)
    # or ~20 FLOPs (norm_prod)
    # Total = n_cl * avg_tensor_count * 96 + n_cr * avg_tensor_count * 20 + ...

    # For SC MAC, each operation processes SIZE tensors where SIZE = N >> level
    # Average across the tree walk, each CalcLeft/Right processes ~1 tensor at leaf level
    # Actually for the comp graph decoder, each call processes edge_data[beta]
    # which has N >> (level of beta) entries

    # Simplified estimate: each individual tensor op is ~100 FLOPs for circ_conv
    # For SC with path 0^{N/2} 1^N 0^{N/2}:
    # Total operations ~O(N log N) at the tensor level

    # More practical: count multiply-adds for the entire tree walk
    # Each position in the tree has O(1) tensor operations
    # Total tensor operations = n_cl + n_cr + n_cp (each on varying sizes)

    # Conservative estimate: average tensor count per call = 1 (leaf-level dominated)
    total = n_cl * 96 + n_cr * 20 + n_cp * 96

    return {
        'total_flops': total,
        'n_calc_left': n_cl,
        'n_calc_right': n_cr,
        'n_calc_parent': n_cp,
    }


def measure_inference_time(N, n_trials=50):
    """Measure actual inference time for SC, NN-SC, and SCL."""
    channel_bemac = BEMAC()
    n = int(math.log2(N))

    # Class B design
    design_file = os.path.join(BASE, 'designs', f'bemac_B_n{n}.npz')
    if os.path.exists(design_file):
        ku = BEMAC_CLASS_B_RATES[N]['ku']
        kv = BEMAC_CLASS_B_RATES[N]['kv']
        Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, ku, kv)
        b = make_path(N, N//2)
    else:
        ku = N // 2
        kv = N
        Au, Av, fu, fv, _, _ = design_bemac(n, ku, kv)
        b = make_path(N, N)

    # Generate test data
    rng = np.random.default_rng(42)
    uf = np.zeros(N, dtype=int)
    vf = np.zeros(N, dtype=int)
    for p in Au: uf[p-1] = rng.integers(0, 2)
    for p in Av: vf[p-1] = rng.integers(0, 2)
    x = polar_encode(uf.tolist())
    y = polar_encode(vf.tolist())
    z = channel_bemac.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int)).tolist()

    results = {}

    # SC timing
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        decode_single(N, z, b, fu, fv, channel_bemac, log_domain=False)
        times.append(time.perf_counter() - t0)
    results['SC'] = {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'median_ms': np.median(times) * 1000,
    }

    # SCL(L=4) timing
    if N <= 256:
        n_scl = min(n_trials, 20)
        times = []
        for _ in range(n_scl):
            t0 = time.perf_counter()
            decode_single_list(N, z, b, fu, fv, channel_bemac, log_domain=True, L=4)
            times.append(time.perf_counter() - t0)
        results['SCL_L4'] = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'median_ms': np.median(times) * 1000,
        }

    # NN-SC timing
    ckpt = os.path.join(BASE, 'saved_models', f'ncg_pure_neural_N{N}.pt')
    if os.path.exists(ckpt):
        model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=3)
        sd = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(sd, strict=False)
        model.eval()

        zf = channel_bemac.sample_batch(
            np.array(x, dtype=int).reshape(1, N),
            np.array(y, dtype=int).reshape(1, N))
        zt = torch.from_numpy(zf).long()

        # Warmup
        with torch.no_grad():
            model(zt, b, fu, fv)

        times = []
        with torch.no_grad():
            for _ in range(n_trials):
                t0 = time.perf_counter()
                model(zt, b, fu, fv)
                times.append(time.perf_counter() - t0)
        results['NN_SC'] = {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'median_ms': np.median(times) * 1000,
        }

    return results


BEMAC_CLASS_B_RATES = {
    16:  {'ku': 8,   'kv': 11},
    32:  {'ku': 16,  'kv': 22},
    64:  {'ku': 32,  'kv': 45},
    128: {'ku': 64,  'kv': 90},
    256: {'ku': 128, 'kv': 179},
    512: {'ku': 256, 'kv': 358},
    1024:{'ku': 512, 'kv': 716},
}


def main():
    print("="*80)
    print("  Computational Complexity Analysis")
    print("="*80)

    all_results = {}

    # ── 1. Model size ────────────────────────────────────────────────────────
    print("\n--- Model Size ---")

    # BEMAC model
    model_bemac = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=3)
    n_params_bemac = sum(p.numel() for p in model_bemac.parameters())
    n_trainable_bemac = sum(p.numel() for p in model_bemac.parameters() if p.requires_grad)
    mem_bytes_bemac = sum(p.numel() * p.element_size() for p in model_bemac.parameters())
    print(f"  BEMAC model: {n_params_bemac:,} params ({n_trainable_bemac:,} trainable), "
          f"{mem_bytes_bemac/1024:.1f} KB")

    # GMAC model
    model_gmac = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    n_params_gmac = sum(p.numel() for p in model_gmac.parameters())
    mem_bytes_gmac = sum(p.numel() * p.element_size() for p in model_gmac.parameters())
    print(f"  GMAC model:  {n_params_gmac:,} params, {mem_bytes_gmac/1024:.1f} KB")

    # d=32 model
    model_d32 = GmacNeuralCompGraphDecoder(d=32, hidden=128, n_layers=2, z_hidden=64)
    n_params_d32 = sum(p.numel() for p in model_d32.parameters())
    print(f"  GMAC d=32:   {n_params_d32:,} params, {sum(p.numel()*p.element_size() for p in model_d32.parameters())/1024:.1f} KB")

    all_results['model_size'] = {
        'bemac_d16': {'params': n_params_bemac, 'memory_KB': round(mem_bytes_bemac/1024, 1)},
        'gmac_d16': {'params': n_params_gmac, 'memory_KB': round(mem_bytes_gmac/1024, 1)},
        'gmac_d32': {'params': n_params_d32},
    }

    # ── 2. Tree operation counts ─────────────────────────────────────────────
    print("\n--- Tree Operation Counts ---")
    print(f"  {'N':>5s}  {'Path':>12s}  {'CalcLeft':>10s}  {'CalcRight':>10s}  {'CalcParent':>11s}  {'Total':>8s}")
    print(f"  {'-'*62}")

    ops_data = {}
    for N in [32, 64, 128, 256, 512, 1024]:
        for path_name, path_i in [('Class B', N//2), ('Class C', N)]:
            b = make_path(N, path_i)
            n_cl, n_cr, n_cp = count_tree_operations(N, b)
            total = n_cl + n_cr + n_cp
            print(f"  {N:>5d}  {path_name:>12s}  {n_cl:>10d}  {n_cr:>10d}  {n_cp:>11d}  {total:>8d}")
            ops_data[f'N{N}_{path_name.replace(" ","")}'] = {
                'N': N, 'path': path_name,
                'calc_left': n_cl, 'calc_right': n_cr,
                'calc_parent': n_cp, 'total': total
            }
    all_results['tree_operations'] = ops_data

    # ── 3. FLOPs comparison ─────────────────────────────────────────────────
    print("\n--- FLOPs per Codeword ---")
    print(f"  {'N':>5s}  {'SC FLOPs':>12s}  {'NN FLOPs':>12s}  {'Ratio':>8s}")
    print(f"  {'-'*42}")

    flops_data = {}
    for N in [32, 64, 128, 256, 512, 1024]:
        b = make_path(N, N//2)
        sc_info = estimate_sc_flops_per_codeword(N, b)
        nn_info = estimate_nn_flops_per_codeword(N, d=16, hidden=64, b=b)
        ratio = nn_info['total_flops'] / max(1, sc_info['total_flops'])
        print(f"  {N:>5d}  {sc_info['total_flops']:>12,}  {nn_info['total_flops']:>12,}  {ratio:>7.1f}x")
        flops_data[str(N)] = {
            'sc_flops': sc_info['total_flops'],
            'nn_flops': nn_info['total_flops'],
            'ratio': round(ratio, 1),
            'nn_total_mlp_calls': nn_info['total_mlp_calls'],
        }
    all_results['flops'] = flops_data

    # ── 4. Inference time ───────────────────────────────────────────────────
    print("\n--- Inference Time (ms/codeword) ---")
    print(f"  {'N':>5s}  {'SC':>10s}  {'NN-SC':>10s}  {'SCL(4)':>10s}  {'NN/SC':>8s}")
    print(f"  {'-'*50}")

    timing_data = {}
    for N in [32, 64, 128, 256, 512, 1024]:
        n_trials = max(10, 100 // max(1, N // 32))
        timings = measure_inference_time(N, n_trials)

        sc_ms = timings.get('SC', {}).get('mean_ms', None)
        nn_ms = timings.get('NN_SC', {}).get('mean_ms', None)
        scl_ms = timings.get('SCL_L4', {}).get('mean_ms', None)
        ratio = f"{nn_ms/sc_ms:.1f}x" if sc_ms and nn_ms else "-"

        sc_s = f"{sc_ms:.1f}" if sc_ms else "-"
        nn_s = f"{nn_ms:.1f}" if nn_ms else "-"
        scl_s = f"{scl_ms:.1f}" if scl_ms else "-"
        print(f"  {N:>5d}  {sc_s:>10s}  {nn_s:>10s}  {scl_s:>10s}  {ratio:>8s}")

        timing_data[str(N)] = timings
    all_results['inference_time'] = timing_data

    # ── 5. Training time (from report) ───────────────────────────────────────
    print("\n--- Training Time to Match SC (from report) ---")
    training_data = {
        '32':  {'iters': 15000,  'wall_time_hr': 0.33, 'best_bler': 0.046},
        '64':  {'iters': 80000,  'wall_time_hr': 12,   'best_bler': 0.026},
        '128': {'iters': 135000, 'wall_time_hr': 28,   'best_bler': 0.017},
        '256': {'iters': 100000, 'wall_time_hr': 16,   'best_bler': 0.015},
        '512': {'iters': 45000,  'wall_time_hr': 28,   'best_bler': 0.012},
    }
    print(f"  {'N':>5s}  {'Iters':>8s}  {'Time':>8s}  {'Best BLER':>10s}")
    print(f"  {'-'*36}")
    for N_s, d in training_data.items():
        print(f"  {N_s:>5s}  {d['iters']:>8,}  {d['wall_time_hr']:>6.1f}hr  {d['best_bler']:>10.4f}")
    all_results['training_time'] = training_data

    # ── Save ────────────────────────────────────────────────────────────────
    out_json = os.path.join(BASE, 'results', 'complexity_analysis.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_json}")

    # ── Write markdown report ───────────────────────────────────────────────
    md_path = os.path.join(BASE, 'docs', 'complexity_analysis.md')
    with open(md_path, 'w') as f:
        f.write("# Computational Complexity Analysis\n\n")
        f.write("## Model Size\n\n")
        f.write("| Model | Parameters | Memory |\n")
        f.write("|-------|-----------|--------|\n")
        f.write(f"| BEMAC (d=16) | {n_params_bemac:,} | {mem_bytes_bemac/1024:.1f} KB |\n")
        f.write(f"| GMAC (d=16) | {n_params_gmac:,} | {mem_bytes_gmac/1024:.1f} KB |\n")
        f.write(f"| GMAC (d=32) | {n_params_d32:,} | {sum(p.numel()*p.element_size() for p in model_d32.parameters())/1024:.1f} KB |\n\n")

        f.write("## FLOPs per Codeword\n\n")
        f.write("| N | SC FLOPs | NN-SC FLOPs | Ratio |\n")
        f.write("|---|---------|-------------|-------|\n")
        for N_s, d in flops_data.items():
            f.write(f"| {N_s} | {d['sc_flops']:,} | {d['nn_flops']:,} | {d['ratio']}x |\n")

        f.write("\n## Inference Time (ms/codeword)\n\n")
        f.write("| N | SC | NN-SC | SCL(L=4) | NN/SC Ratio |\n")
        f.write("|---|---|-------|----------|-------------|\n")
        for N_s, t in timing_data.items():
            sc = t.get('SC', {}).get('mean_ms')
            nn = t.get('NN_SC', {}).get('mean_ms')
            scl = t.get('SCL_L4', {}).get('mean_ms')
            ratio = f"{nn/sc:.1f}x" if sc and nn else "-"
            nn_s = f"{nn:.1f}" if nn else "-"
            scl_s = f"{scl:.1f}" if scl else "-"
            f.write(f"| {N_s} | {sc:.1f} | {nn_s} | {scl_s} | {ratio} |\n")

        f.write("\n## Training Time (GMAC, reported)\n\n")
        f.write("| N | Iterations | Wall Time | Best BLER |\n")
        f.write("|---|-----------|-----------|----------|\n")
        for N_s, d in training_data.items():
            f.write(f"| {N_s} | {d['iters']:,} | {d['wall_time_hr']}hr | {d['best_bler']} |\n")

    print(f"  Report saved to: {md_path}")


if __name__ == '__main__':
    main()
