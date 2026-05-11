#!/usr/bin/env python3
"""
gpu_isi_npd_design_retrain.py
=============================
Key experiment: retrain ISI-MAC NPD with NPD-optimal frozen sets.

The hypothesis (from frozen set analysis):
  GMAC proxy frozen set has only 47% overlap with what the NPD can actually
  learn on ISI-MAC at N=128. Retraining with NPD-optimal frozen sets should
  significantly reduce BLER.

Steps per N:
  1. Load existing GMAC-trained NPD model (best checkpoint)
  2. Run rate-1 decode to get per-position error rates (NPD-specific design)
  3. Select new frozen set from NPD ranking (same ku/kv as before)
  4. Train a FRESH NPD from scratch with the new frozen set (1M iters)
  5. Eval and compare to:
     a. NPD + GMAC proxy (existing result)
     b. Chained trellis SC + GMAC proxy (analytical baseline)

Run on GPU:
  python scripts/gpu_isi_npd_design_retrain.py
"""
from __future__ import annotations
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels_memory import ISIMAC
from polar.design_mc import _argsort_with_polar_tiebreak
from neural.npd_memory_mac import ChainedNPD_MAC, MemoryStageNPD

# ─── Config ──────────────────────────────────────────────────────────────────

SNR_DB = 6.0
ISI_H = 0.3
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Same ku/kv as GMAC proxy (preserves rate for fair comparison)
KU_KV = {
    16:  (4, 7),
    32:  (7, 15),
    64:  (15, 29),
    128: (30, 58),
    256: (59, 117),
}

# Training config per N
TRAIN_CFG = {
    64:  dict(iters=500_000,  batch=64,  d=16, hidden=100, lr=1e-3, eval_every=10_000),
    128: dict(iters=1_000_000, batch=32,  d=16, hidden=100, lr=1e-3, eval_every=20_000),
    256: dict(iters=1_000_000, batch=16,  d=16, hidden=100, lr=1e-3, eval_every=20_000),
}

RESULTS_DIR = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_isi_design')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Checkpoint loading (GMAC-trained models) ────────────────────────────────

def find_gmac_checkpoint(N, stage='s1', d=16, hidden=100):
    """Find best existing GMAC-trained checkpoint for a given N."""
    base = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')
    candidates = [
        # Standalone models (d=16 h=100)
        os.path.join(base, f'd16_h100_standalone_{stage}_N{N}_best.pt'),
        os.path.join(base, f'd16_h100_standalone_{stage}_N{N}_iter200000.pt'),
        # BiGRU models
        os.path.join(base, f'isi_mac_bigru_L1_{stage}_N{N}_best.pt'),
        # GPU curriculum models
        os.path.join(base, f'gpu_curriculum_{stage}_N{N}_final.pt'),
        # Other patterns
        os.path.join(base, f'd64_lr1e3_N{N}_final.pt') if stage == 's1' else '',
    ]
    # Also check /tmp/isi_ckpts/
    if stage == 's1':
        candidates.extend([
            f'/tmp/isi_ckpts/isi_N{N}_final.pt',
            f'/tmp/isi_ckpts/isi_N{N}_iter1000000.pt',
        ])

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_stage_model(ckpt_path, d=16, hidden=100, extra_dim=0, gru_layers=1):
    """Load a MemoryStageNPD from checkpoint."""
    stage = MemoryStageNPD(d=d, hidden=hidden, n_layers=2,
                           encoder_type='bigru', extra_dim=extra_dim,
                           gru_layers=gru_layers)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        stage.load_state_dict(ckpt['state_dict'])
    else:
        stage.load_state_dict(ckpt)
    return stage


# ─── Step 1: NPD-based frozen set design ──────────────────────────────────────

def npd_rate1_design(stage, N, n_cw=20000, batch_size=100):
    """
    Measure per-position decode error rate using the NPD at rate-1 (all info).
    Returns pe_per_pos (N,) array.
    """
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.tensor(br, dtype=torch.long)
    channel = ISIMAC(sigma2=SIGMA2, h=ISI_H)

    stage.eval()
    pe = np.zeros(N)
    total = 0
    rng = np.random.default_rng(123)

    # Rate-1: all positions are info, no frozen
    Au_all = list(range(1, N + 1))
    Av_all = list(range(1, N + 1))  # V side also rate-1 for channel generation
    empty_frozen = set()

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)

            u_msg = rng.integers(0, 2, (actual, N)).astype(np.int32)
            v_msg = rng.integers(0, 2, (actual, N)).astype(np.int32)

            x = polar_encode_batch(u_msg)
            y = polar_encode_batch(v_msg)
            z = channel.sample_batch(x, y).astype(np.float32)
            zt = torch.from_numpy(z).to(DEVICE)

            emb = stage.encode_channel(zt)
            emb_br = emb[:, br_t, :]
            u_hat = stage.tree.decode(emb_br, empty_frozen)

            for pos in range(N):
                pe[pos] += (u_hat[:, pos].cpu().numpy() != u_msg[:, pos]).sum()

            total += actual
            if total % 5000 == 0:
                print(f'    design: {total}/{n_cw}', flush=True)

    pe /= n_cw
    return pe


def build_frozen_set(pe, k):
    """Build info set (1-indexed) and frozen set (0-indexed) from per-position Pe."""
    N = len(pe)
    sorted_idx = _argsort_with_polar_tiebreak(pe)
    info_0idx = sorted(sorted_idx[:k].tolist())
    info_1idx = [i + 1 for i in info_0idx]
    frozen_0idx = set(range(N)) - set(info_0idx)
    frozen_1idx = {p: 0 for p in range(1, N + 1) if p not in info_1idx}
    return info_1idx, frozen_0idx, frozen_1idx


# ─── Step 2: Training ────────────────────────────────────────────────────────

def make_batch(channel, N, Au, Av, batch, rng):
    """Generate training batch."""
    u_msg = np.zeros((batch, N), dtype=np.int8)
    v_msg = np.zeros((batch, N), dtype=np.int8)
    for p in Au:
        u_msg[:, p - 1] = rng.integers(0, 2, batch)
    for p in Av:
        v_msg[:, p - 1] = rng.integers(0, 2, batch)
    x = polar_encode_batch(u_msg.astype(int))
    y = polar_encode_batch(v_msg.astype(int))
    z = channel.sample_batch(x.astype(int), y.astype(int))
    return u_msg, v_msg, z.astype(np.float32), x, y


def train_stage(stage, channel, N, Au, Av, frozen_set, iters, batch, lr,
                tag, is_stage2=False, eval_every=10000, eval_cw=500, seed=42):
    """Train one stage with fast_ce. Constant lr (paper recipe)."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.tensor(br, dtype=torch.long)

    stage.to(DEVICE)
    opt = torch.optim.AdamW(stage.parameters(), lr=lr, weight_decay=1e-5)
    # Constant lr (no decay — paper recipe, avoids the lr decay bug)
    rng = np.random.default_rng(seed)

    best_bler = 1.0
    best_state = None
    losses = []
    t0 = time.time()

    log_path = os.path.join(RESULTS_DIR, f'{tag}.log')

    stage.train()
    for it in range(1, iters + 1):
        u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, batch, rng)
        z_t = torch.from_numpy(z).to(DEVICE)

        if is_stage2:
            # Stage 2: teacher-forced with TRUE x
            side = torch.from_numpy(
                (1.0 - 2.0 * x_phys.astype(np.float32))
            ).unsqueeze(-1).to(DEVICE)
            emb = stage.encode_channel(z_t, side=side)
            emb_npd = emb[:, br_t, :]
            target = torch.from_numpy(y_phys[:, br]).long().to(DEVICE)
        else:
            emb = stage.encode_channel(z_t)
            emb_npd = emb[:, br_t, :]
            target = torch.from_numpy(x_phys[:, br]).long().to(DEVICE)

        loss = stage.tree.fast_ce(emb_npd, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

        if it % eval_every == 0 or it == iters:
            bler = eval_stage(stage, channel, N, Au, Av, frozen_set,
                              is_stage2=is_stage2, n_cw=eval_cw, seed=999)
            avg_loss = float(np.mean(losses[-min(500, len(losses)):]))
            elapsed = (time.time() - t0) / 60
            marker = ''
            if bler < best_bler:
                best_bler = bler
                best_state = {k: v.cpu().clone() for k, v in stage.state_dict().items()}
                marker = ' *BEST*'

            # Save periodic checkpoint
            if it % 100_000 == 0 or it == iters:
                ckpt_path = os.path.join(RESULTS_DIR, f'{tag}_iter{it}.pt')
                torch.save({'state_dict': stage.state_dict(), 'N': N,
                            'Au': Au, 'Av': Av, 'iter': it}, ckpt_path)

            msg = (f'  [{tag} {it:>7}/{iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}) {elapsed:.1f}min{marker}')
            print(msg, flush=True)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

    # Save best
    if best_state is not None:
        ckpt_path = os.path.join(RESULTS_DIR, f'{tag}_best.pt')
        torch.save({'state_dict': best_state, 'N': N, 'Au': Au, 'Av': Av},
                   ckpt_path)
        # Reload best for subsequent use
        stage.load_state_dict(best_state)

    return best_bler


def eval_stage(stage, channel, N, Au, Av, frozen_set, is_stage2=False,
               n_cw=500, batch=32, seed=999):
    """Eval one stage BLER."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.tensor(br, dtype=torch.long)
    stage.eval()
    rng = np.random.default_rng(seed)
    errs = 0
    total = 0
    info_positions = Au if not is_stage2 else Av

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, v_msg, z, x_phys, y_phys = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z).to(DEVICE)

            if is_stage2:
                side = torch.from_numpy(
                    (1.0 - 2.0 * x_phys.astype(np.float32))
                ).unsqueeze(-1).to(DEVICE)
                emb = stage.encode_channel(z_t, side=side)
                msg = v_msg
            else:
                emb = stage.encode_channel(z_t)
                msg = u_msg

            emb_npd = emb[:, br_t, :]
            hat = stage.tree.decode(emb_npd, frozen_set)

            for i in range(actual):
                if any(int(hat[i, p - 1].item()) != int(msg[i, p - 1])
                       for p in info_positions):
                    errs += 1
            total += actual
    stage.train()
    return errs / n_cw


def eval_chained(model, channel, N, Au, Av, fu_set, fv_set, n_cw=2000,
                 batch=32, seed=777):
    """Full chained eval: Stage 1 → Stage 2."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.tensor(br, dtype=torch.long)
    model.stage1.eval()
    model.stage2.eval()
    rng = np.random.default_rng(seed)
    errs_u = errs_v = errs_total = 0
    total = 0

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg, v_msg, z, x_phys, _ = make_batch(channel, N, Au, Av, actual, rng)
            z_t = torch.from_numpy(z).to(DEVICE)

            # Stage 1
            emb1 = model.stage1.encode_channel(z_t)
            emb1_npd = emb1[:, br_t, :]
            u_hat = model.stage1.tree.decode(emb1_npd, fu_set)
            u_hat_np = u_hat.cpu().numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)

            # Stage 2
            side = torch.from_numpy(
                (1.0 - 2.0 * x_hat.astype(np.float32))
            ).unsqueeze(-1).to(DEVICE)
            emb2 = model.stage2.encode_channel(z_t, side=side)
            emb2_npd = emb2[:, br_t, :]
            v_hat = model.stage2.tree.decode(emb2_npd, fv_set)

            for i in range(actual):
                ue = any(int(u_hat[i, p-1].item()) != int(u_msg[i, p-1]) for p in Au)
                ve = any(int(v_hat[i, p-1].item()) != int(v_msg[i, p-1]) for p in Av)
                if ue: errs_u += 1
                if ve: errs_v += 1
                if ue or ve: errs_total += 1
            total += actual

    return {
        'bler_u': errs_u / n_cw,
        'bler_v': errs_v / n_cw,
        'bler_total': errs_total / n_cw,
        'n_cw': n_cw,
    }


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_one_N(N):
    """Full pipeline for one N."""
    ku, kv = KU_KV[N]
    n = int(math.log2(N))
    cfg = TRAIN_CFG[N]
    channel = ISIMAC(sigma2=SIGMA2, h=ISI_H)

    print(f'\n{"="*70}')
    print(f' N={N}, ku={ku}, kv={kv}')
    print(f' Config: {cfg}')
    print(f' Device: {DEVICE}')
    print(f'{"="*70}')

    results = {'N': N, 'ku': ku, 'kv': kv}

    # ── Step 1: NPD-based design using existing GMAC-trained model ──
    print(f'\n--- Step 1: NPD rate-1 design ---')

    # Try to load existing GMAC-trained checkpoint
    s1_ckpt = find_gmac_checkpoint(N, 's1', d=cfg['d'], hidden=cfg['hidden'])
    if s1_ckpt is None:
        print(f'  ERROR: No GMAC-trained checkpoint found for N={N}. '
              f'Cannot do NPD design. Skipping.')
        return None

    print(f'  Loading GMAC-trained S1 from: {os.path.basename(s1_ckpt)}')
    stage1_gmac = load_stage_model(s1_ckpt, d=cfg['d'], hidden=cfg['hidden'],
                                   extra_dim=0)
    stage1_gmac.to(DEVICE)

    n_design = 20000
    print(f'  Running rate-1 design ({n_design} CW)...')
    pe_u = npd_rate1_design(stage1_gmac, N, n_cw=n_design)

    # For V: also load GMAC-trained S2 and do rate-1 design
    s2_ckpt = find_gmac_checkpoint(N, 's2', d=cfg['d'], hidden=cfg['hidden'])
    if s2_ckpt:
        print(f'  Loading GMAC-trained S2 from: {os.path.basename(s2_ckpt)}')
        stage2_gmac = load_stage_model(s2_ckpt, d=cfg['d'], hidden=cfg['hidden'],
                                       extra_dim=1)
        stage2_gmac.to(DEVICE)
        # V design: rate-1 decode with true X side info
        # For simplicity, use GMAC proxy for V (stage 2 is easier, less wall)
        pe_v = None
    else:
        pe_v = None

    del stage1_gmac
    if 's2_ckpt' in dir() and s2_ckpt:
        del stage2_gmac
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    # Build NPD-optimal frozen sets for U
    Au_npd, fu_npd_set, fu_npd_dict = build_frozen_set(pe_u, ku)
    results['Au_npd'] = Au_npd
    results['pe_u_rate1'] = pe_u.tolist()

    # For V: use GMAC proxy (stage 2 rarely walls)
    gmac_design = np.load(os.path.join(_ROOT, f'designs/gmac_C_n{n}_snr{int(SNR_DB)}dB.npz'))
    pe_v_gmac = gmac_design['v_error_rates']
    Av_gmac_sorted = _argsort_with_polar_tiebreak(pe_v_gmac)
    Av_0 = sorted(Av_gmac_sorted[:kv].tolist())
    Av = [i + 1 for i in Av_0]
    fv_set = set(range(N)) - set(Av_0)
    fv_dict = {p: 0 for p in range(1, N + 1) if p not in Av}

    results['Av'] = Av

    # Compare with GMAC proxy U design
    pe_u_gmac = gmac_design['u_error_rates']
    Au_gmac_sorted = _argsort_with_polar_tiebreak(pe_u_gmac)
    Au_gmac_0 = sorted(Au_gmac_sorted[:ku].tolist())
    Au_gmac = [i + 1 for i in Au_gmac_0]

    overlap = len(set(Au_npd) & set(Au_gmac))
    print(f'\n  NPD-design vs GMAC-proxy U overlap: {overlap}/{ku} ({100*overlap/ku:.0f}%)')
    print(f'  Au_npd (0-idx): {[p-1 for p in Au_npd]}')
    print(f'  Au_gmac (0-idx): {Au_gmac_0}')
    results['overlap_u'] = overlap
    results['Au_gmac'] = Au_gmac

    # ── Step 2: Train fresh NPD with NPD-optimal frozen set ──
    print(f'\n--- Step 2: Train NPD with NPD-optimal frozen set ---')
    print(f'  iters={cfg["iters"]}, batch={cfg["batch"]}, lr={cfg["lr"]}, '
          f'd={cfg["d"]}, hidden={cfg["hidden"]}')

    torch.manual_seed(42)
    model = ChainedNPD_MAC(d=cfg['d'], hidden=cfg['hidden'], n_layers=2,
                           encoder_type='bigru', gru_layers=1)
    model.to(DEVICE)
    print(f'  Parameters: {model.count_parameters():,}')

    # Train Stage 1
    t0 = time.time()
    s1_bler = train_stage(
        model.stage1, channel, N, Au_npd, Av, fu_npd_set,
        iters=cfg['iters'], batch=cfg['batch'], lr=cfg['lr'],
        tag=f'isi_npd_design_s1_N{N}',
        eval_every=cfg['eval_every'], eval_cw=500, seed=42,
    )
    s1_time = (time.time() - t0) / 60
    print(f'\n  Stage 1 best BLER: {s1_bler:.4f} ({s1_time:.1f} min)')
    results['s1_bler'] = float(s1_bler)
    results['s1_time_min'] = float(s1_time)

    # Train Stage 2
    t0 = time.time()
    s2_bler = train_stage(
        model.stage2, channel, N, Au_npd, Av, fv_set,
        iters=cfg['iters'] // 2,  # Stage 2 needs less training
        batch=cfg['batch'], lr=cfg['lr'],
        tag=f'isi_npd_design_s2_N{N}',
        is_stage2=True,
        eval_every=cfg['eval_every'], eval_cw=500, seed=43,
    )
    s2_time = (time.time() - t0) / 60
    print(f'\n  Stage 2 best BLER: {s2_bler:.4f} ({s2_time:.1f} min)')
    results['s2_bler'] = float(s2_bler)
    results['s2_time_min'] = float(s2_time)

    # ── Step 3: Eval ──
    print(f'\n--- Step 3: Chained eval ---')
    chained = eval_chained(model, channel, N, Au_npd, Av,
                           fu_npd_set, fv_set, n_cw=5000, seed=777)
    print(f'  NPD+NPD_design: BLER={chained["bler_total"]:.4f} '
          f'(U={chained["bler_u"]:.4f}, V={chained["bler_v"]:.4f})')
    results['chained_npd_design'] = chained

    # Compare with chained SC + GMAC design
    from polar.decoder_trellis_mac_chained import bler_chained
    fu_gmac_dict = {p: 0 for p in range(1, N + 1) if p not in Au_gmac}
    fv_gmac_dict = {p: 0 for p in range(1, N + 1) if p not in Av}

    print(f'  Running chained SC + GMAC design ({min(5000, 2000)} CW)...')
    sc_n_cw = 5000 if N <= 128 else 2000
    r_sc = bler_chained(channel, N, fu_gmac_dict, fv_gmac_dict,
                        Au_gmac, Av, sc_n_cw, seed=0)
    print(f'  SC+GMAC: BLER={r_sc["chained_bler"]:.4f}')
    results['sc_gmac_bler'] = r_sc['chained_bler']

    # Summary
    print(f'\n{"="*50}')
    print(f'  N={N} SUMMARY:')
    print(f'    NPD + NPD_design:  {chained["bler_total"]:.4f}')
    print(f'    SC  + GMAC_design: {r_sc["chained_bler"]:.4f}')
    print(f'    Ratio: {chained["bler_total"]/max(r_sc["chained_bler"], 1e-9):.2f}x')
    print(f'{"="*50}')
    results['ratio_vs_sc'] = chained['bler_total'] / max(r_sc['chained_bler'], 1e-9)

    # Save results
    out_path = os.path.join(RESULTS_DIR, f'isi_npd_design_N{N}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'  Saved: {out_path}')

    return results


def main():
    print(f'Device: {DEVICE}')
    print(f'ISI-MAC: SNR={SNR_DB}dB, h={ISI_H}, sigma2={SIGMA2:.4f}')

    all_results = {}

    # Start with N=128 (the main wall case)
    for N in [128, 64, 256]:
        if N not in TRAIN_CFG:
            print(f'Skipping N={N} (no training config)')
            continue

        result = run_one_N(N)
        if result is not None:
            all_results[str(N)] = result

        # Save all results incrementally
        out_path = os.path.join(RESULTS_DIR, 'all_results.json')
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print(f'\n\n{"="*70}')
    print('FINAL SUMMARY: NPD+NPD_design vs SC+GMAC_design')
    print(f'{"="*70}')
    print(f'{"N":<6} {"NPD+NPD_design":<18} {"SC+GMAC":<12} {"Ratio":<10}')
    for Ns, r in sorted(all_results.items(), key=lambda x: int(x[0])):
        npd = r['chained_npd_design']['bler_total']
        sc = r['sc_gmac_bler']
        ratio = npd / max(sc, 1e-9)
        print(f'{Ns:<6} {npd:<18.4f} {sc:<12.4f} {ratio:<10.2f}x')


if __name__ == '__main__':
    main()
