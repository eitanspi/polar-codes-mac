#!/usr/bin/env python3
"""
NPD-guided design for N=512: Phase 2 (MI measurement) + Phase 3 (retrain).

Loads Phase 1 checkpoint (rate-1, neural-mode), measures per-position MI
using CORRECT neural tree ops (checknode/bitnode MLPs + emb2llr), selects
top ku=119 positions, and retrains on those.

Bug fix: previous code used analytical_checknode (scalar ops) for MI measurement
on neural-mode models, giving MI=0. This script uses the neural tree ops that
match how the model was trained with use_analytical_training=False.
"""
import sys, os, math, time, json
import numpy as np
import torch
import torch.nn.functional as F

# Setup paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design
from class_c_npd.training.train_stage import generate_stage1_batch, evaluate_stage

# ─── Configuration ───────────────────────────────────────────────────────────
N = 512
n = 9
SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
D = 16; HIDDEN = 64; N_LAYERS = 2; Z_DIM = 1

# Rate design: ku = round(0.50 * I(X;Z) * N) = round(0.50 * 0.4645 * 512) = 119
KU = 119

# Phase 2 config
MI_SAMPLES = 20000
MI_BATCH = 100

# Phase 3 config
P3_ITERS = 100000
P3_BATCH = 16
P3_LR = 1e-4
P3_EVAL_EVERY = 10000

# Paths
P1_CKPT = os.path.expanduser('~/polar_project/class_c_npd/results/npd_design_p1_N512_best.pt')
RESULTS_DIR = os.path.expanduser('~/polar_project/class_c_npd/results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


def measure_leaf_mi_neural(model, N, Av, n_samples=20000, batch=100):
    """
    Measure per-position leaf-level MI using NEURAL tree ops.

    This is the FIXED version: uses model.checknode() (MLP on 2*d dims),
    model.bitnode() (MLP with residual), and model.emb2llr() at leaves.

    The BUGGY version used analytical_checknode (scalar LLR f/g) which
    gives MI=0 for neural-mode models.
    """
    from polar.channels import GaussianMAC
    nn_ = int(math.log2(N))
    br = bit_reversal_perm(nn_)
    channel = GaussianMAC(sigma2=SIGMA2)
    model.eval()

    leaf_bce = np.zeros(N)
    count = 0
    rng = np.random.default_rng(123)
    np.random.seed(123)

    with torch.no_grad():
        while count < n_samples:
            actual = min(batch, n_samples - count)
            u_msg = rng.integers(0, 2, (actual, N)).astype(np.int8)
            x_phys = polar_encode_batch(u_msg.astype(int))
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Av:
                v_msg[:, p - 1] = rng.integers(0, 2, actual)
            y_phys = polar_encode_batch(v_msg.astype(int))
            z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))

            emb_in = torch.from_numpy(z[:, br].astype(np.float32)).unsqueeze(-1).to(DEVICE)
            emb = model.encode_channel(emb_in)
            B, N_, d = emb.shape

            cw_br = torch.from_numpy(x_phys[:, br]).long().to(DEVICE)

            # Walk tree using NEURAL ops (the fix)
            V = [cw_br]
            E = [emb]
            for depth in range(nn_):
                Vo, Ve, Eo, Ee = [], [], [], []
                for vc, ec in zip(V, E):
                    Vo.append(vc[:, 0::2]); Ve.append(vc[:, 1::2])
                    Eo.append(ec[:, 0::2, :]); Ee.append(ec[:, 1::2, :])
                Vo = torch.cat(Vo, 1); Ve = torch.cat(Ve, 1)
                Eo = torch.cat(Eo, 1); Ee = torch.cat(Ee, 1)
                vt = Vo ^ Ve; vb = Ve
                nc = 2 ** depth; cs = (N_ // 2) // nc
                vtc = torch.split(vt, cs, 1); vbc = torch.split(vb, cs, 1)
                Vn = []
                for a, b in zip(vtc, vbc):
                    Vn += [a, b]
                Vl = torch.cat(Vn[0::2], 1)
                # NEURAL tree ops (correct for use_analytical_training=False)
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
            leaf_bce += bce.sum(0).cpu().numpy()
            count += actual

    avg_bce = leaf_bce / count
    # Map tree order -> natural order
    bce_nat = np.zeros(N)
    for tidx in range(N):
        bce_nat[br[tidx]] = avg_bce[tidx]
    mi_nat = np.log(2) - bce_nat
    model.train()
    return mi_nat, bce_nat


def main():
    t_start = time.time()
    print(f'NPD-guided design N={N}, Phase 2+3')
    print(f'SNR={SNR_DB}dB, sigma2={SIGMA2:.4f}')
    print(f'ku={KU}, P3 iters={P3_ITERS}, batch={P3_BATCH}')
    print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'P1 checkpoint: {P1_CKPT}')

    # Load Av from genie design (V frozen set)
    Au_genie, Av, frozen_u_genie, frozen_v, pe_u, pe_v = load_class_c_design(
        'gmac', n, snr_db=SNR_DB, ku=None, kv=None)
    print(f'Av (V info positions): {len(Av)} positions')

    # ─── Load Phase 1 checkpoint ─────────────────────────────────────────────
    print(f'\nLoading P1 checkpoint...')
    ckpt = torch.load(P1_CKPT, weights_only=False, map_location='cpu')
    model_p1 = NPDSingleUser(d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_dim=Z_DIM,
                              use_analytical_training=False)
    model_p1.load_state_dict(ckpt['state_dict'])
    model_p1.to(DEVICE)
    print(f'  Loaded. Model params: {model_p1.count_parameters():,}')

    # ─── Phase 2: Measure MI with NEURAL tree ops ────────────────────────────
    print(f'\n{"="*60}')
    print(f'Phase 2: Measuring per-position MI ({MI_SAMPLES} samples)')
    print(f'{"="*60}')
    t2 = time.time()
    mi_nat, bce_nat = measure_leaf_mi_neural(model_p1, N, Av, MI_SAMPLES, MI_BATCH)
    print(f'  avg MI = {mi_nat.mean():.4f} nats (GMAC capacity = 0.4645)')
    print(f'  Positive MI count: {(mi_nat > 0).sum()}/{N}')
    print(f'  MI min/max: {mi_nat.min():.4f} / {mi_nat.max():.4f}')
    print(f'  Phase 2 time: {(time.time()-t2)/60:.1f} min')

    # Select top ku positions by MI
    sorted_pos = np.argsort(-mi_nat)
    npd_Au = sorted(int(sorted_pos[i]) + 1 for i in range(KU))  # 1-indexed
    npd_frozen = {p - 1 for p in range(1, N + 1) if p not in npd_Au}

    # Compare to genie design
    overlap = set(npd_Au) & set(Au_genie)
    print(f'\n  NPD-optimal Au ({len(npd_Au)} positions): {npd_Au[:10]}...')
    print(f'  Genie Au ({len(Au_genie)} positions):       {sorted(Au_genie)[:10]}...')
    print(f'  Overlap: {len(overlap)}/{KU}')
    print(f'  NPD avoids: {sorted(set(Au_genie) - set(npd_Au))[:20]}...')
    print(f'  NPD picks:  {sorted(set(npd_Au) - set(Au_genie))[:20]}...')

    # Save MI data
    mi_path = os.path.join(RESULTS_DIR, 'mi_per_pos_N512.json')
    with open(mi_path, 'w') as f:
        json.dump({'mi_nat': mi_nat.tolist(), 'npd_Au': npd_Au,
                   'genie_Au': sorted(Au_genie), 'overlap': len(overlap)}, f, indent=2)
    print(f'  MI data saved to {mi_path}')

    # Free P1 model
    del model_p1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ─── Phase 3: Retrain with NPD-optimal frozen set ────────────────────────
    print(f'\n{"="*60}')
    print(f'Phase 3: Retrain with NPD-optimal design ({P3_ITERS} iters)')
    print(f'{"="*60}')

    channel = build_channel('gmac', sigma2=SIGMA2)
    torch.manual_seed(42)
    model_p3 = NPDSingleUser(d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_dim=Z_DIM,
                              use_analytical_training=False)
    # Warm start from P1 checkpoint
    ckpt = torch.load(P1_CKPT, weights_only=False, map_location='cpu')
    model_p3.load_state_dict(ckpt['state_dict'])
    model_p3.to(DEVICE)
    print(f'  Warm-started from P1 checkpoint')

    opt = torch.optim.Adam(model_p3.parameters(), lr=P3_LR)
    rng = np.random.default_rng(42)
    t3 = time.time()
    best_bler = 1.0
    p3_ckpt_path = os.path.join(RESULTS_DIR, f'npd_design_p3_N{N}_best.pt')

    model_p3.train()
    for it in range(1, P3_ITERS + 1):
        # Generate training batch
        _, features_npd, cw_npd = generate_stage1_batch(
            channel, N, npd_Au, P3_BATCH, rng, Av)
        ft = torch.from_numpy(features_npd).float().unsqueeze(-1).to(DEVICE)
        cw = torch.from_numpy(cw_npd).long().to(DEVICE)

        emb = model_p3.encode_channel(ft)
        loss = model_p3.fast_ce(emb, cw)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_p3.parameters(), 1.0)
        opt.step()

        if it % P3_EVAL_EVERY == 0:
            # Move to CPU for eval (avoids GPU memory issues with sequential decode)
            model_p3.cpu()
            bler = evaluate_stage(
                model_p3, channel, N, npd_Au, npd_frozen,
                generate_stage1_batch, n_cw=500, seed=999,
                other_info=Av)
            marker = ''
            if bler < best_bler:
                best_bler = bler
                marker = ' *BEST*'
            # Always save (in case best doesn't improve but we want latest)
            torch.save({
                'state_dict': model_p3.state_dict(),
                'd': D, 'hidden': HIDDEN, 'n_layers': N_LAYERS, 'z_dim': Z_DIM,
                'N': N, 'Au': npd_Au, 'Av': Av,
                'npd_frozen': sorted(npd_frozen),
                'mi_avg': float(mi_nat.mean()),
                'overlap_with_genie': len(overlap),
                'best_bler': float(best_bler),
            }, p3_ckpt_path)
            elapsed = (time.time() - t3) / 60
            print(f'  P3 [{it:>6}/{P3_ITERS}] loss={loss.item():.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}', flush=True)
            # Move back to GPU
            model_p3.to(DEVICE)

    # Final eval
    model_p3.cpu()
    final_bler = evaluate_stage(
        model_p3, channel, N, npd_Au, npd_frozen,
        generate_stage1_batch, n_cw=2000, seed=999,
        other_info=Av)
    torch.save({
        'state_dict': model_p3.state_dict(),
        'd': D, 'hidden': HIDDEN, 'n_layers': N_LAYERS, 'z_dim': Z_DIM,
        'N': N, 'Au': npd_Au, 'Av': Av,
        'npd_frozen': sorted(npd_frozen),
        'mi_avg': float(mi_nat.mean()),
        'overlap_with_genie': len(overlap),
        'final_bler': float(final_bler),
        'best_bler': float(best_bler),
    }, p3_ckpt_path)

    total_min = (time.time() - t_start) / 60
    print(f'\n{"="*60}')
    print(f'RESULTS N={N}')
    print(f'{"="*60}')
    print(f'  MI avg:          {mi_nat.mean():.4f} nats')
    print(f'  Genie overlap:   {len(overlap)}/{KU}')
    print(f'  NPD best BLER:   {best_bler:.4f}')
    print(f'  NPD final BLER:  {final_bler:.4f}')
    print(f'  Checkpoint:      {p3_ckpt_path}')
    print(f'  Total wall time: {total_min:.1f} min')
    print(f'  Finish: {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
