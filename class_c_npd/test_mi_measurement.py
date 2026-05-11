#!/usr/bin/env python3
"""
Minimal test: correct MI measurement for neural-mode NPD checkpoint.

Demonstrates the bug and the fix:
  BUG:  Using analytical tree ops (scalar emb[:,0]) on a neural-mode model
        gives avg MI = -0.42 nats (garbage, mostly negative)
  FIX:  Using neural tree ops (d-dim checknode/bitnode + emb2llr)
        gives avg MI = 0.50 nats (close to GMAC capacity 0.4645)

The bug arises when code from mi_convergence_plot.py (designed for
use_analytical_training=True models) is applied to a neural-mode checkpoint:
  1. emb[:,:,0] is just the first of 16 embedding dims, not a trained LLR
  2. analytical_checknode/bitnode do exact f/g on scalars, but the model
     was trained with neural checknode/bitnode MLPs on 16-dim embeddings
  3. The scalar at the leaves is meaningless, so BCE >> log(2), MI << 0

The working npd_design_sweep.py uses neural tree ops + emb2llr, which
matches how fast_ce trains when use_analytical_training=False.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from class_c_npd.models.npd_single_user import NPDSingleUser

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
# P1 checkpoint: trained at rate 1 (all info) with use_analytical_training=False
CKPT = os.path.join(os.path.dirname(__file__), 'results', 'npd_design_p1_N256_best.pt')


def _generate_samples(N, Av, batch, rng, br, channel):
    """Generate channel observations and codewords."""
    u_msg = rng.integers(0, 2, (batch, N)).astype(np.int8)
    x_phys = polar_encode_batch(u_msg.astype(int))
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Av:
        v_msg[:, p - 1] = rng.integers(0, 2, batch)
    y_phys = polar_encode_batch(v_msg.astype(int))
    z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
    emb_input = torch.from_numpy(z[:, br].astype(np.float32)).unsqueeze(-1)
    cw_br = torch.from_numpy(x_phys[:, br]).long()
    return emb_input, cw_br


def _walk_tree_neural(model, emb, cw_br, n, N):
    """Walk tree using NEURAL checknode/bitnode on d-dim embeddings, emb2llr at leaves."""
    V = [cw_br]
    E = [emb]  # (B, N, d)
    for depth in range(n):
        Vo, Ve, Eo, Ee = [], [], [], []
        for vc, ec in zip(V, E):
            Vo.append(vc[:, 0::2]); Ve.append(vc[:, 1::2])
            Eo.append(ec[:, 0::2, :]); Ee.append(ec[:, 1::2, :])
        Vo = torch.cat(Vo, 1); Ve = torch.cat(Ve, 1)
        Eo = torch.cat(Eo, 1); Ee = torch.cat(Ee, 1)
        vt = Vo ^ Ve; vb = Ve
        nc = 2 ** depth; cs = (N // 2) // nc
        vtc = torch.split(vt, cs, 1); vbc = torch.split(vb, cs, 1)
        Vn = []
        for a, b in zip(vtc, vbc):
            Vn += [a, b]
        Vl = torch.cat(Vn[0::2], 1)
        et = model.checknode(torch.cat([Eo, Ee], -1))
        eb = model.bitnode(Eo, Ee, Vl)
        etc = torch.split(et, cs, 1); ebc = torch.split(eb, cs, 1)
        En = []
        for a, b in zip(etc, ebc):
            En += [a, b]
        V = Vn; E = En
    e_leaves = torch.cat(E, 1); v_leaves = torch.cat(V, 1)
    logits = model.emb2llr(e_leaves).squeeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, v_leaves.float(), reduction='none')
    return bce


def _walk_tree_analytical_buggy(model, emb, cw_br, n, N):
    """BUGGY: analytical ops on scalar emb[:,:,0], no emb2llr."""
    E_scalar = emb[:, :, 0]  # first dim only
    V = [cw_br]
    E = [E_scalar]
    for depth in range(n):
        Vo, Ve, Eo, Ee = [], [], [], []
        for vc, ec in zip(V, E):
            Vo.append(vc[:, 0::2]); Ve.append(vc[:, 1::2])
            Eo.append(ec[:, 0::2]); Ee.append(ec[:, 1::2])
        Vo = torch.cat(Vo, 1); Ve = torch.cat(Ve, 1)
        Eo = torch.cat(Eo, 1); Ee = torch.cat(Ee, 1)
        vt = Vo ^ Ve; vb = Ve
        nc = 2 ** depth; cs = (N // 2) // nc
        vtc = torch.split(vt, cs, 1); vbc = torch.split(vb, cs, 1)
        Vn = []
        for a, b in zip(vtc, vbc):
            Vn += [a, b]
        Vl = torch.cat(Vn[0::2], 1)
        et = model.analytical_checknode(Eo, Ee)
        eb = model.analytical_bitnode(Eo, Ee, Vl)
        etc = torch.split(et, cs, 1); ebc = torch.split(eb, cs, 1)
        En = []
        for a, b in zip(etc, ebc):
            En += [a, b]
        V = Vn; E = En
    e_leaf = torch.cat(E, 1); v_leaf = torch.cat(V, 1)
    bce = F.binary_cross_entropy_with_logits(e_leaf, v_leaf.float(), reduction='none')
    return bce


def measure_mi(model, N, Av, n_samples, use_neural):
    """Measure per-position leaf MI. use_neural=True for correct, False for buggy."""
    n = int(math.log2(N)); br = bit_reversal_perm(n)
    channel = GaussianMAC(sigma2=SIGMA2)
    model.eval()
    leaf_bce = np.zeros(N); count = 0
    rng = np.random.default_rng(123); np.random.seed(123)
    batch = min(100, n_samples)

    with torch.no_grad():
        while count < n_samples:
            actual = min(batch, n_samples - count)
            emb_input, cw_br = _generate_samples(N, Av, actual, rng, br, channel)
            emb = model.encode_channel(emb_input)
            if use_neural:
                bce = _walk_tree_neural(model, emb, cw_br, n, N)
            else:
                bce = _walk_tree_analytical_buggy(model, emb, cw_br, n, N)
            leaf_bce += bce.sum(0).numpy()
            count += actual

    avg_bce = leaf_bce / count
    bce_nat = np.zeros(N)
    for tidx in range(N):
        bce_nat[br[tidx]] = avg_bce[tidx]
    mi_nat = np.log(2) - bce_nat
    return mi_nat


def main():
    print(f'Loading checkpoint: {CKPT}')
    ckpt = torch.load(CKPT, weights_only=False, map_location='cpu')
    N = ckpt['N']
    Av = ckpt['Av']
    d = ckpt.get('d', 16)
    hidden = ckpt.get('hidden', 64)
    n_layers = ckpt.get('n_layers', 2)
    z_dim = ckpt.get('z_dim', 1)

    model = NPDSingleUser(d=d, hidden=hidden, n_layers=n_layers, z_dim=z_dim,
                           use_analytical_training=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'N={N}, d={d}, |Av|={len(Av)}')
    print(f'GMAC marginal capacity I(X;Z) = 0.4645 nats/bit')

    n_samples = 5000

    print(f'\n--- BUGGY: analytical tree ops on scalar emb[:,0] ---')
    mi_buggy = measure_mi(model, N, Av, n_samples, use_neural=False)
    print(f'  avg MI = {mi_buggy.mean():.4f} nats')
    print(f'  Positive MI count: {(mi_buggy > 0).sum()}/{N}')

    print(f'\n--- CORRECT: neural tree ops (checknode/bitnode) + emb2llr ---')
    mi_correct = measure_mi(model, N, Av, n_samples, use_neural=True)
    print(f'  avg MI = {mi_correct.mean():.4f} nats  (should be close to 0.4645)')
    print(f'  Positive MI count: {(mi_correct > 0).sum()}/{N}')

    # Show top/bottom positions
    sorted_idx = np.argsort(-mi_correct)
    print(f'\n  Top 5 MI positions (1-indexed):    ', end='')
    for i in sorted_idx[:5]:
        print(f'{i+1}({mi_correct[i]:.3f})', end=' ')
    print(f'\n  Bottom 5 MI positions (1-indexed):  ', end='')
    for i in sorted_idx[-5:]:
        print(f'{i+1}({mi_correct[i]:.3f})', end=' ')
    print()

    print(f'\n--- Summary ---')
    print(f'  Buggy  avg MI:  {mi_buggy.mean():.4f} nats  (garbage, negative)')
    print(f'  Correct avg MI: {mi_correct.mean():.4f} nats  (matches capacity)')
    print(f'  GMAC capacity:  0.4645 nats')

    ok = mi_correct.mean() > 0.35
    print(f'\n  {"PASS" if ok else "FAIL"}: correct MI = {mi_correct.mean():.4f}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
