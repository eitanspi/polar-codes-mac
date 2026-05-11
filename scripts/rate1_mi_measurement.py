#!/usr/bin/env python3
"""
rate1_mi_measurement.py
=======================
Measure per-position mutual information at N=512 using the d=64 N=256
checkpoint (no frozen positions, rate-1) under teacher forcing.

This tells us which positions the model finds hard at longer sequences
and whether they match the GMAC proxy design ordering.
"""
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels_memory import ISIMAC
from polar.design_mc import design_from_file
from neural.npd_memory_mac import MemoryStageNPD

SNR_DB = 6.0
ISI_H = 0.3
RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')


def make_channel():
    return ISIMAC.from_snr_db(SNR_DB, h=ISI_H)


def load_design_for_N(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, _path_i = design_from_file(path, n, ku, kv)
    return sorted(Au_list), sorted(Av_list)


def measure_per_position_mi(stage1_model, channel, N, n_batches=200, batch=32, seed=42):
    """
    Measure per-position mutual information using the trained model
    under teacher forcing (fast_ce) at rate 1 (all positions are info).

    MI(U_i; Z^N) approx 1 - H(U_i | Z^N) where H is estimated from
    the model's binary CE at each leaf.

    Returns per-position CE (in nats), from which MI = ln(2) - CE.
    """
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()

    stage1_model.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # Accumulate per-position CE at the LEAF level
    # We need to run the tree forward with teacher forcing and
    # collect per-leaf CE, not the depth-averaged fast_ce.
    per_pos_ce = np.zeros(N, dtype=np.float64)  # in nats
    per_pos_count = np.zeros(N, dtype=np.float64)

    with torch.no_grad():
        for bi in range(n_batches):
            # Generate rate-1 data: ALL positions are info
            u_msg = rng.integers(0, 2, (batch, N)).astype(np.int8)
            x_phys = polar_encode_batch(u_msg.astype(int))
            # For ISI-MAC, we need Y too. Generate random V.
            v_msg = rng.integers(0, 2, (batch, N)).astype(np.int8)
            y_phys = polar_encode_batch(v_msg.astype(int))
            z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
            z_t = torch.from_numpy(z.astype(np.float32))

            emb = stage1_model.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            x_cw_npd = torch.from_numpy(x_phys[:, br]).long()

            # Run the tree forward to get per-leaf logits
            # We use the existing fast_ce infrastructure but extract per-position
            leaf_ce = compute_leaf_ce(stage1_model.tree, emb_npd, x_cw_npd)
            # leaf_ce is (B, N) in NPD tree order

            # Convert back to natural order
            inv_br = np.zeros(N, dtype=int)
            inv_br[br] = np.arange(N)
            leaf_ce_nat = leaf_ce[:, inv_br]

            per_pos_ce += leaf_ce_nat.sum(axis=0)
            per_pos_count += batch

            if (bi + 1) % 50 == 0:
                avg_ce = per_pos_ce / per_pos_count
                avg_mi = np.log(2) - avg_ce
                print(f'  batch {bi+1}/{n_batches}: mean CE={avg_ce.mean():.4f} '
                      f'mean MI={avg_mi.mean():.4f} nats', flush=True)

    avg_ce = per_pos_ce / per_pos_count  # per-position CE in nats
    mi = np.log(2) - avg_ce  # per-position MI in nats
    mi_bits = mi / np.log(2)  # MI in bits

    return avg_ce, mi, mi_bits


def compute_leaf_ce(tree, emb, x_cw):
    """
    Run teacher-forced SC tree and return per-leaf binary CE.

    emb: (B, N, d) in NPD tree order
    x_cw: (B, N) true codeword bits in NPD tree order

    Returns: (B, N) per-leaf CE in nats (numpy)
    """
    B, N, d = emb.shape
    n = int(math.log2(N))

    leaf_ce = np.zeros((B, N), dtype=np.float64)
    leaf_idx = [0]

    def _recurse(e_block, v_block):
        block_size = e_block.shape[1]
        if block_size == 1:
            logit = tree.emb2llr(e_block[:, 0, :]).squeeze(-1)  # (B,)
            target = v_block[:, 0].float()
            ce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')  # (B,)
            idx = leaf_idx[0]
            leaf_ce[:, idx] = ce.numpy()
            leaf_idx[0] += 1
            return v_block  # return true bits for teacher forcing

        e_odd = e_block[:, 0::2, :]
        e_even = e_block[:, 1::2, :]
        v_odd = v_block[:, 0::2]
        v_even = v_block[:, 1::2]

        # Top (check node)
        v_top = v_odd ^ v_even
        e_top = tree.checknode(torch.cat([e_odd, e_even], dim=-1))
        _recurse(e_top, v_top)

        # Bottom (bit node), teacher-forced with true top
        e_bot = tree.bitnode(e_odd, e_even, v_top)
        _recurse(e_bot, v_even)

    _recurse(emb, x_cw)
    return leaf_ce


def main():
    channel = make_channel()

    # Load N=256 d=64 checkpoint
    s1_path = os.path.join(RESULTS_DIR, 'd64_s1_N256_300k.pt')
    if not os.path.exists(s1_path):
        s1_path = '/tmp/d64_N256_300k.pt'

    print(f'Loading checkpoint: {s1_path}')
    stage1 = MemoryStageNPD(d=64, hidden=128, n_layers=2,
                             encoder_type='bigru', extra_dim=0, gru_layers=1)
    sd = torch.load(s1_path, weights_only=False, map_location='cpu')
    if 'state_dict' in sd:
        stage1.load_state_dict(sd['state_dict'])
    else:
        stage1.load_state_dict(sd)
    print('Loaded successfully')

    # Measure MI at N=256 first (sanity check)
    for target_N in [256, 512]:
        print(f'\n{"="*60}')
        print(f'MI measurement at N={target_N}')
        print(f'{"="*60}')

        t0 = time.time()
        n_batches = 200 if target_N <= 256 else 100
        batch_size = 32 if target_N <= 256 else 16

        avg_ce, mi_nats, mi_bits = measure_per_position_mi(
            stage1, channel, target_N,
            n_batches=n_batches, batch=batch_size, seed=42)
        elapsed = time.time() - t0

        print(f'\nResults (N={target_N}):')
        print(f'  Mean MI: {mi_bits.mean():.4f} bits')
        print(f'  Min MI:  {mi_bits.min():.4f} bits (position {mi_bits.argmin()+1})')
        print(f'  Max MI:  {mi_bits.max():.4f} bits (position {mi_bits.argmax()+1})')
        print(f'  Sum MI:  {mi_bits.sum():.1f} bits')
        print(f'  Time: {elapsed:.0f}s')

        # Compare with GMAC design ordering
        if target_N in {256, 512}:
            ku_ref = {256: 59, 512: 119}[target_N]
            kv_ref = {256: 117, 512: 233}[target_N]
            try:
                Au_ref, Av_ref = load_design_for_N(target_N, ku_ref, kv_ref)
                # Au is the set of info positions for U
                Au_set = set(Au_ref)
                info_mi = [mi_bits[p-1] for p in Au_ref]
                frozen_mi = [mi_bits[p-1] for p in range(1, target_N+1) if p not in Au_set]
                print(f'\n  GMAC proxy design comparison:')
                print(f'    Info positions ({len(Au_ref)}): mean MI = {np.mean(info_mi):.4f} bits')
                print(f'    Frozen positions ({target_N - len(Au_ref)}): mean MI = {np.mean(frozen_mi):.4f} bits')

                # Check: does model MI ordering match design ordering?
                design_ranking = np.argsort(-mi_bits)  # positions ranked by MI (best first), 0-indexed
                n_agree = sum(1 for p in design_ranking[:ku_ref] if (p+1) in Au_set)
                print(f'    Top-{ku_ref} by model MI overlap with Au: {n_agree}/{ku_ref} ({n_agree/ku_ref:.1%})')

                # Quartile analysis of MI
                sorted_mi = np.sort(mi_bits)
                q_size = target_N // 4
                for qi in range(4):
                    q_start = qi * q_size
                    q_end = (qi + 1) * q_size
                    q_mi = sorted_mi[q_start:q_end]
                    print(f'    Q{qi+1} MI: [{q_mi.min():.4f}, {q_mi.max():.4f}], mean={q_mi.mean():.4f}')
            except Exception as e:
                print(f'  Could not load design: {e}')

        # Save
        out_path = os.path.join(_ROOT, 'results', 'reliable_evals',
                                f'isi_mac_rate1_mi_N{target_N}.json')
        result = {
            'N': target_N,
            'n_batches': n_batches,
            'batch_size': batch_size,
            'mean_mi_bits': float(mi_bits.mean()),
            'min_mi_bits': float(mi_bits.min()),
            'max_mi_bits': float(mi_bits.max()),
            'sum_mi_bits': float(mi_bits.sum()),
            'per_position_mi_bits': mi_bits.tolist(),
            'per_position_ce_nats': avg_ce.tolist(),
        }
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  Saved to {out_path}')


if __name__ == '__main__':
    main()
