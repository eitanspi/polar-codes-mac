#!/usr/bin/env python3
"""
rate1_mi_measurement_v2.py
==========================
Measure per-position MI at N=256 and N=512 using the d=64 checkpoint.

Instead of rate-1 (which the model was never trained on), we measure
MI at the actual operating rate -- frozen positions set to 0, info
positions random. This measures how well the model estimates each
info position's bit given the channel observations.

We also compare the MI ranking to the GMAC proxy design ranking.
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
RATES = {
    256: (59, 117),
    512: (119, 233),
}


def make_channel():
    return ISIMAC.from_snr_db(SNR_DB, h=ISI_H)


def load_design_for_N(N, ku, kv):
    n = int(math.log2(N))
    path = os.path.join(_ROOT, 'designs', f'gmac_C_n{n}_snr{int(SNR_DB)}dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, pe_u, pe_v, _path_i = design_from_file(path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_set = {p-1 for p in range(1, N+1) if p not in Au}
    return Au, Av, frozen_u_set


def compute_leaf_ce(tree, emb, x_cw, frozen_set_0idx):
    """
    Run teacher-forced SC tree and return per-leaf binary CE.
    Only computes CE for INFO positions (frozen positions get CE=0).

    emb: (B, N, d) in NPD tree order
    x_cw: (B, N) true codeword bits in NPD tree order

    Returns: (B, N) per-leaf CE in nats (numpy), in NPD tree order
    """
    from polar.encoder import bit_reversal_perm
    B, N, d = emb.shape
    n = int(math.log2(N))
    br = bit_reversal_perm(n)

    leaf_ce = np.zeros((B, N), dtype=np.float64)
    leaf_idx = [0]

    def _recurse(e_block, v_block):
        block_size = e_block.shape[1]
        if block_size == 1:
            logit = tree.emb2llr(e_block[:, 0, :]).squeeze(-1)  # (B,)
            target = v_block[:, 0].float()
            idx = leaf_idx[0]
            leaf_idx[0] += 1
            # Map tree-order leaf to natural position
            nat_idx = int(br[idx])
            if nat_idx in frozen_set_0idx:
                # Frozen position: CE = 0 (we know it's 0)
                return v_block
            ce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')  # (B,)
            leaf_ce[:, idx] = ce.numpy()
            return v_block

        e_odd = e_block[:, 0::2, :]
        e_even = e_block[:, 1::2, :]
        v_odd = v_block[:, 0::2]
        v_even = v_block[:, 1::2]

        v_top = v_odd ^ v_even
        e_top = tree.checknode(torch.cat([e_odd, e_even], dim=-1))
        _recurse(e_top, v_top)

        e_bot = tree.bitnode(e_odd, e_even, v_top)
        _recurse(e_bot, v_even)

    _recurse(emb, x_cw)
    return leaf_ce


def measure_per_position_mi(stage1, channel, N, Au, Av, frozen_u_set,
                              n_batches=200, batch=32, seed=42):
    """
    Measure per-position MI for info positions using the trained model.
    """
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.from_numpy(br.copy()).long()

    stage1.eval()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    per_pos_ce = np.zeros(N, dtype=np.float64)  # in tree order
    count = 0

    with torch.no_grad():
        for bi in range(n_batches):
            # Generate data at actual operating rate
            u_msg = np.zeros((batch, N), dtype=np.int8)
            for p in Au:
                u_msg[:, p-1] = rng.integers(0, 2, batch)
            x_phys = polar_encode_batch(u_msg.astype(int))

            # Random V
            v_msg = np.zeros((batch, N), dtype=np.int8)
            for p in Av:
                v_msg[:, p-1] = rng.integers(0, 2, batch)
            y_phys = polar_encode_batch(v_msg.astype(int))

            z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
            z_t = torch.from_numpy(z.astype(np.float32))

            emb = stage1.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            x_cw_npd = torch.from_numpy(x_phys[:, br]).long()

            leaf_ce = compute_leaf_ce(stage1.tree, emb_npd, x_cw_npd, frozen_u_set)
            per_pos_ce += leaf_ce.sum(axis=0)  # sum over batch, still in tree order
            count += batch

            if (bi + 1) % 50 == 0:
                avg_ce = per_pos_ce / count
                # Convert tree order to natural order
                inv_br = np.zeros(N, dtype=int)
                inv_br[br] = np.arange(N)
                avg_ce_nat = avg_ce[inv_br]
                info_ce = [avg_ce_nat[p-1] for p in Au]
                avg_info_ce = np.mean(info_ce)
                avg_info_mi = np.log(2) - avg_info_ce
                print(f'  batch {bi+1}/{n_batches}: mean info CE={avg_info_ce:.4f} '
                      f'MI={avg_info_mi:.4f} nats ({avg_info_mi/np.log(2):.4f} bits)',
                      flush=True)

    avg_ce_tree = per_pos_ce / count
    # Convert to natural order
    inv_br = np.zeros(N, dtype=int)
    inv_br[br] = np.arange(N)
    avg_ce_nat = avg_ce_tree[inv_br]

    # MI per position (only meaningful for info positions)
    mi_nats = np.log(2) - avg_ce_nat
    mi_bits = mi_nats / np.log(2)

    return avg_ce_nat, mi_nats, mi_bits


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

    for target_N in [256, 512]:
        print(f'\n{"="*60}')
        print(f'MI measurement at N={target_N} (operating rate)')
        print(f'{"="*60}')

        ku, kv = RATES[target_N]
        Au, Av, frozen_u_set = load_design_for_N(target_N, ku, kv)
        Au_set = set(Au)
        print(f'  ku={ku}, kv={kv}')

        t0 = time.time()
        n_batches = 200 if target_N <= 256 else 100
        batch_size = 32 if target_N <= 256 else 16

        avg_ce, mi_nats, mi_bits = measure_per_position_mi(
            stage1, channel, target_N, Au, Av, frozen_u_set,
            n_batches=n_batches, batch=batch_size, seed=42)
        elapsed = time.time() - t0

        # Extract info-position MIs
        info_mi = np.array([mi_bits[p-1] for p in Au])
        frozen_mi = np.array([mi_bits[p-1] for p in range(1, target_N+1) if p not in Au_set])

        print(f'\nResults (N={target_N}):')
        print(f'  Info positions ({len(Au)}):')
        print(f'    Mean MI: {info_mi.mean():.4f} bits')
        print(f'    Min MI:  {info_mi.min():.4f} bits (pos {Au[info_mi.argmin()]})')
        print(f'    Max MI:  {info_mi.max():.4f} bits (pos {Au[info_mi.argmax()]})')
        print(f'    Std MI:  {info_mi.std():.4f} bits')
        print(f'  Frozen positions ({target_N - len(Au)}):')
        print(f'    Mean MI: {frozen_mi.mean():.6f} bits (should be ~0)')

        # MI ranking vs design ranking
        all_mi = mi_bits.copy()
        # Set frozen positions MI to -inf for ranking purposes
        for p in range(1, target_N+1):
            if p not in Au_set:
                all_mi[p-1] = -np.inf
        model_ranking = np.argsort(-all_mi)  # 0-indexed, best first
        n_top_agree = sum(1 for p in model_ranking[:ku] if (p+1) in Au_set)
        print(f'\n  Model MI ranking vs GMAC proxy design:')
        print(f'    Top-{ku} overlap: {n_top_agree}/{ku} ({n_top_agree/ku:.1%})')

        # Quartile analysis of info positions' MI
        sorted_info_mi = np.sort(info_mi)
        q_size = max(len(info_mi) // 4, 1)
        print(f'\n  Info-position MI quartiles:')
        for qi in range(4):
            q_start = qi * q_size
            q_end = min((qi + 1) * q_size, len(info_mi))
            if q_start >= len(info_mi):
                break
            q_mi = sorted_info_mi[q_start:q_end]
            print(f'    Q{qi+1} ({q_start}-{q_end}): [{q_mi.min():.4f}, {q_mi.max():.4f}], mean={q_mi.mean():.4f} bits')

        # Find weakest positions (lowest MI among info)
        weak_order = np.argsort(info_mi)
        print(f'\n  5 weakest info positions:')
        for k in range(min(5, len(weak_order))):
            idx = weak_order[k]
            p = Au[idx]
            print(f'    pos {p}: MI={info_mi[idx]:.4f} bits, CE={avg_ce[p-1]:.4f} nats')

        print(f'\n  5 strongest info positions:')
        for k in range(min(5, len(weak_order))):
            idx = weak_order[-(k+1)]
            p = Au[idx]
            print(f'    pos {p}: MI={info_mi[idx]:.4f} bits, CE={avg_ce[p-1]:.4f} nats')

        print(f'\n  Time: {elapsed:.0f}s')

        # Save
        out_path = os.path.join(_ROOT, 'results', 'reliable_evals',
                                f'isi_mac_mi_N{target_N}.json')
        result = {
            'N': target_N, 'ku': ku, 'kv': kv,
            'n_batches': n_batches, 'batch_size': batch_size,
            'info_mean_mi_bits': float(info_mi.mean()),
            'info_min_mi_bits': float(info_mi.min()),
            'info_max_mi_bits': float(info_mi.max()),
            'info_std_mi_bits': float(info_mi.std()),
            'per_position_mi_bits': mi_bits.tolist(),
            'per_position_ce_nats': avg_ce.tolist(),
            'Au': Au,
        }
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  Saved to {out_path}')


if __name__ == '__main__':
    main()
