#!/usr/bin/env python3
"""
frozen_set_analysis.py — Comprehensive analysis of ISI-MAC NPD wall.

Analysis 1: Frozen set comparison (NPD-MI vs GMAC proxy)
Analysis 2: MI trajectory during training (N=128 checkpoints)
Analysis 4: ISI-MAC MC design comparison (decode error rates)

Outputs:
  - results/frozen_set_analysis/*.json (raw data)
  - docs/meeting/*.pdf (figures)
  - project_summary/FROZEN_SET_ANALYSIS.md (summary)
"""
import sys, os, math, json, time
import numpy as np
import torch
import torch.nn.functional as F

# Setup
torch.set_num_threads(4)
BASE = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2'
sys.path.insert(0, BASE)

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.channels_memory import ISIMAC
from polar.design import make_path
from neural.npd_memory_mac import ChainedNPD_MAC, MemoryStageNPD

RESULTS_DIR = os.path.join(BASE, 'results', 'frozen_set_analysis')
PLOTS_DIR = os.path.join(BASE, 'docs', 'meeting')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SNR_DB = 6.0
ISI_H = 0.3
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)

# N -> (ku, kv) from eval_gpu_checkpoints.py
N_CONFIGS = {
    16:   (4, 7),
    32:   (7, 15),
    64:   (15, 29),
    128:  (30, 58),
    256:  (59, 117),
}


# ─── Design loading ─────────────────────────────────────────────────────────

def load_design(N):
    """Load GMAC proxy design. Returns Au, Av (1-indexed), fu_set, fv_set (0-indexed), pe_u, pe_v."""
    ku, kv = N_CONFIGS[N]
    n = int(math.log2(N))
    dfile = np.load(os.path.join(BASE, f'designs/gmac_C_n{n}_snr{int(SNR_DB)}dB.npz'))
    pe_u = dfile['u_error_rates']
    pe_v = dfile['v_error_rates']
    Au_0 = sorted(np.argsort(pe_u)[:ku].tolist())
    Av_0 = sorted(np.argsort(pe_v)[:kv].tolist())
    Au = [i + 1 for i in Au_0]
    Av = [i + 1 for i in Av_0]
    fu_set = {p - 1 for p in range(1, N + 1) if p not in Au}
    fv_set = {p - 1 for p in range(1, N + 1) if p not in Av}
    return Au, Av, fu_set, fv_set, pe_u, pe_v


# ─── Checkpoint loading ─────────────────────────────────────────────────────

def load_stage1_model(N, d=16, hidden=100, gru_layers=1):
    """Load the best Stage 1 model for a given N."""
    stage = MemoryStageNPD(d=d, hidden=hidden, n_layers=2,
                           encoder_type='bigru', extra_dim=0,
                           gru_layers=gru_layers)
    ckpt_paths = []
    if N == 64:
        ckpt_paths = ['/tmp/isi_ckpts/isi_N64_final.pt']
    elif N == 128:
        ckpt_paths = ['/tmp/isi_ckpts/isi_N128_iter1000000.pt']
    elif N == 256:
        ckpt_paths = ['/tmp/isi_ckpts/isi_N256_v2_final.pt']
    elif N == 16:
        ckpt_paths = [os.path.join(BASE, 'class_c_npd/results/npd_memory_mac/isi_mac_bigru_L1_s1_N16_best.pt')]
    elif N == 32:
        ckpt_paths = [os.path.join(BASE, 'class_c_npd/results/npd_memory_mac/isi_mac_bigru_L1_s1_N32_best.pt')]

    for path in ckpt_paths:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                stage.load_state_dict(ckpt['state_dict'])
            else:
                stage.load_state_dict(ckpt)
            print(f"  Loaded N={N} from {os.path.basename(path)}")
            return stage
    raise FileNotFoundError(f"No checkpoint found for N={N}")


def load_stage1_from_path(path, d=16, hidden=100, gru_layers=1):
    """Load a specific checkpoint."""
    stage = MemoryStageNPD(d=d, hidden=hidden, n_layers=2,
                           encoder_type='bigru', extra_dim=0,
                           gru_layers=gru_layers)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        stage.load_state_dict(ckpt['state_dict'])
    else:
        stage.load_state_dict(ckpt)
    return stage


# ─── Genie-aided MI measurement ─────────────────────────────────────────────

def measure_genie_mi(stage, N, Au, fu_set, n_cw=5000, batch_size=100):
    """
    Genie-aided per-position MI measurement.

    Uses sequential decode with TRUE prior bits (genie). At each leaf,
    records the logit and the true value. Computes BCE -> MI.

    Returns: mi_per_pos (N,) array, bce_per_pos (N,) array.
    Only info positions (Au, 1-indexed) have meaningful MI; frozen positions
    always have trivial targets.
    """
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    channel = ISIMAC(sigma2=SIGMA2, h=ISI_H)
    ku, kv = N_CONFIGS[N]
    Au_set = set(Au)
    Av_0 = sorted(np.argsort(np.load(os.path.join(BASE, f'designs/gmac_C_n{n}_snr{int(SNR_DB)}dB.npz'))['v_error_rates'])[:kv].tolist())
    Av = [i + 1 for i in Av_0]

    stage.eval()

    # Accumulate per-position stats
    logit_sums = {pos: [] for pos in range(N)}
    true_sums = {pos: [] for pos in range(N)}
    total = 0
    rng = np.random.default_rng(42)

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)

            u_msg = np.zeros((actual, N), dtype=np.int32)
            v_msg = np.zeros((actual, N), dtype=np.int32)
            for p in Au:
                u_msg[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                v_msg[:, p - 1] = rng.integers(0, 2, actual)

            x = polar_encode_batch(u_msg)
            y = polar_encode_batch(v_msg)
            z = channel.sample_batch(x, y).astype(np.float32)
            zt = torch.from_numpy(z)

            emb = stage.encode_channel(zt)
            emb_br = emb[:, br, :]
            x_br = torch.from_numpy(x[:, br]).long()

            tree = stage.tree
            leaf_idx_box = [0]
            batch_logits = {}
            batch_targets = {}

            def _genie_decode(e_block, true_cw):
                bsz = e_block.shape[0]
                block_size = e_block.shape[1]
                if block_size == 1:
                    logit = tree.emb2llr(e_block[:, 0, :]).squeeze(-1)
                    idx = leaf_idx_box[0]
                    leaf_idx_box[0] += 1
                    nat_idx = int(br[idx])
                    batch_logits[nat_idx] = logit.numpy().copy()
                    batch_targets[nat_idx] = true_cw[:, 0].numpy().copy()
                    return true_cw

                e_odd = e_block[:, 0::2, :]
                e_even = e_block[:, 1::2, :]
                t_odd = true_cw[:, 0::2]
                t_even = true_cw[:, 1::2]
                t_top = t_odd ^ t_even
                t_bot = t_even

                e_top = tree.checknode(torch.cat([e_odd, e_even], dim=-1))
                cw_top = _genie_decode(e_top, t_top)

                e_bot = tree.bitnode(e_odd, e_even, cw_top)
                cw_bot = _genie_decode(e_bot, t_bot)

                cw = torch.zeros(bsz, block_size, dtype=torch.long)
                cw[:, 0::2] = cw_top ^ cw_bot
                cw[:, 1::2] = cw_bot
                return cw

            _genie_decode(emb_br, x_br)

            for pos in range(N):
                logit_sums[pos].append(batch_logits[pos])
                true_sums[pos].append(batch_targets[pos])

            total += actual
            if total % 1000 == 0:
                print(f"    MI: {total}/{n_cw}", flush=True)

    # Compute per-position BCE and MI
    mi = np.zeros(N)
    bce = np.zeros(N)
    for pos in range(N):
        all_logits = torch.from_numpy(np.concatenate(logit_sums[pos])).float()
        all_targets = torch.from_numpy(np.concatenate(true_sums[pos])).float()
        pos_bce = F.binary_cross_entropy_with_logits(all_logits, all_targets, reduction='mean').item()
        bce[pos] = pos_bce
        mi[pos] = max(0.0, (math.log(2) - pos_bce) / math.log(2))

    return mi, bce


def measure_decode_errors(stage, N, Au, fu_set, n_cw=5000, batch_size=100):
    """
    Measure per-position decode error rate (actual SC decode, not genie).

    Returns: pe_per_pos (N,) — error rate at each position.
    """
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    channel = ISIMAC(sigma2=SIGMA2, h=ISI_H)
    ku, kv = N_CONFIGS[N]
    Av_0 = sorted(np.argsort(np.load(os.path.join(BASE, f'designs/gmac_C_n{n}_snr{int(SNR_DB)}dB.npz'))['v_error_rates'])[:kv].tolist())
    Av = [i + 1 for i in Av_0]

    stage.eval()
    pe = np.zeros(N)
    total = 0
    rng = np.random.default_rng(123)

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)

            u_msg = np.zeros((actual, N), dtype=np.int32)
            v_msg = np.zeros((actual, N), dtype=np.int32)
            for p in Au:
                u_msg[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                v_msg[:, p - 1] = rng.integers(0, 2, actual)

            x = polar_encode_batch(u_msg)
            y = polar_encode_batch(v_msg)
            z = channel.sample_batch(x, y).astype(np.float32)
            zt = torch.from_numpy(z)

            emb = stage.encode_channel(zt)
            emb_br = emb[:, br, :]
            u_hat = stage.tree.decode(emb_br, fu_set)

            for pos in range(N):
                pe[pos] += (u_hat[:, pos].numpy() != u_msg[:, pos]).sum()

            total += actual
            if total % 2000 == 0:
                print(f"    decode errors: {total}/{n_cw}", flush=True)

    pe /= n_cw
    return pe


# ─── Analysis 1: Frozen Set Comparison ───────────────────────────────────────

def analysis1():
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Frozen Set Comparison (NPD-MI vs GMAC proxy)")
    print("=" * 70)

    results = {}

    for N in [16, 32, 64, 128, 256]:
        n = int(math.log2(N))
        print(f"\n--- N={N} ---")

        # Model params: N=16,32 used d=16 h=64; N=64,128,256 use d=16 h=100
        if N <= 32:
            d, hidden = 16, 64
        else:
            d, hidden = 16, 100

        try:
            stage = load_stage1_model(N, d=d, hidden=hidden)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        Au, Av, fu_set, fv_set, pe_u, pe_v = load_design(N)
        ku = N_CONFIGS[N][0]
        Au_0indexed = set(p - 1 for p in Au)

        # Measure genie MI
        n_cw = 10000 if N <= 64 else 5000
        print(f"  Measuring genie MI ({n_cw} CW)...")
        mi, bce = measure_genie_mi(stage, N, Au, fu_set, n_cw=n_cw, batch_size=100)

        # Measure decode errors
        n_cw_err = 10000 if N <= 64 else 5000
        print(f"  Measuring decode errors ({n_cw_err} CW)...")
        pe_decode = measure_decode_errors(stage, N, Au, fu_set, n_cw=n_cw_err, batch_size=100)

        # Compare rankings
        # NPD-MI ranking: sort all positions by MI descending
        npd_ranking_desc = np.argsort(-mi)
        npd_info_0idx = set(npd_ranking_desc[:ku].tolist())

        # GMAC proxy ranking: sort by pe_u ascending
        gmac_ranking_asc = np.argsort(pe_u)
        gmac_info_0idx = Au_0indexed  # these are the actual GMAC info positions used

        overlap = len(npd_info_0idx & gmac_info_0idx)
        overlap_pct = 100 * overlap / ku

        print(f"  ku={ku}, Overlap: {overlap}/{ku} ({overlap_pct:.1f}%)")

        # Info-position MI stats
        info_mi = [mi[p - 1] for p in Au]
        info_pe = [pe_decode[p - 1] for p in Au]

        # Which GMAC info positions have low MI?
        low_mi_info = [(p, mi[p - 1], pe_decode[p - 1]) for p in Au if mi[p - 1] < 0.5]
        print(f"  Info positions with MI<0.5: {len(low_mi_info)}")
        for p, m, e in low_mi_info:
            print(f"    pos {p}: MI={m:.4f}, decode_pe={e:.4f}")

        # Correlation between MI and GMAC proxy for ALL positions
        # (frozen positions have MI~0 trivially, so focus on a comparison metric)
        # For a fair comparison: rank correlation among info positions
        info_mi_arr = np.array([mi[p - 1] for p in Au])
        info_gmac_pe = np.array([pe_u[p - 1] for p in Au])

        # Spearman rank correlation
        from scipy.stats import spearmanr
        try:
            corr, pval = spearmanr(-info_mi_arr, info_gmac_pe)
        except:
            corr = float('nan')

        print(f"  Spearman corr(-MI, GMAC_Pe) among info pos: {corr:.4f}")
        print(f"  Info MI: mean={np.mean(info_mi):.4f}, min={np.min(info_mi):.4f}, max={np.max(info_mi):.4f}")
        print(f"  Info decode Pe: mean={np.mean(info_pe):.4f}, max={np.max(info_pe):.4f}")

        # Positions NPD would pick as info but GMAC doesn't
        npd_only = npd_info_0idx - gmac_info_0idx
        gmac_only = gmac_info_0idx - npd_info_0idx
        print(f"  NPD-only info (0-idx): {sorted(npd_only)[:10]}")
        print(f"  GMAC-only info (0-idx): {sorted(gmac_only)[:10]}")

        # NPD-only positions: what are their GMAC Pe values?
        if npd_only:
            print(f"  NPD-only positions GMAC Pe: {[pe_u[p] for p in sorted(npd_only)]}")
            print(f"  GMAC-only positions MI: {[mi[p] for p in sorted(gmac_only)]}")

        results[N] = {
            'mi': mi.tolist(),
            'bce': bce.tolist(),
            'pe_decode': pe_decode.tolist(),
            'gmac_pe_u': pe_u.tolist(),
            'Au': Au,
            'ku': ku,
            'overlap': overlap,
            'overlap_pct': float(overlap_pct),
            'spearman_corr': float(corr) if not np.isnan(corr) else None,
            'info_mi_mean': float(np.mean(info_mi)),
            'info_mi_min': float(np.min(info_mi)),
            'info_mi_max': float(np.max(info_mi)),
            'info_pe_mean': float(np.mean(info_pe)),
            'info_pe_max': float(np.max(info_pe)),
            'n_low_mi_info': len(low_mi_info),
            'low_mi_info_positions': [(int(p), float(m), float(e)) for p, m, e in low_mi_info],
            'npd_only_0idx': sorted(list(npd_only)),
            'gmac_only_0idx': sorted(list(gmac_only)),
        }

        with open(os.path.join(RESULTS_DIR, 'analysis1_frozen_set.json'), 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    return results


# ─── Analysis 2: MI Trajectory ───────────────────────────────────────────────

def analysis2():
    print("\n" + "=" * 70)
    print("ANALYSIS 2: MI Trajectory During Training (N=128)")
    print("=" * 70)

    N = 128
    n = int(math.log2(N))
    Au, Av, fu_set, fv_set, pe_u, pe_v = load_design(N)
    ku = N_CONFIGS[N][0]

    checkpoints = [
        ('/tmp/isi_ckpts/isi_N128_iter200000.pt', 200000),
        ('/tmp/isi_ckpts/isi_N128_iter400000.pt', 400000),
        ('/tmp/isi_ckpts/isi_N128_iter600000.pt', 600000),
        ('/tmp/isi_ckpts/isi_N128_iter800000.pt', 800000),
        ('/tmp/isi_ckpts/isi_N128_iter1000000.pt', 1000000),
    ]

    trajectory = {}
    n_cw = 3000  # fewer CW for trajectory (5 checkpoints)

    for ckpt_path, iteration in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {ckpt_path} (not found)")
            continue

        print(f"\n  Iteration {iteration}:")
        stage = load_stage1_from_path(ckpt_path, d=16, hidden=100)
        mi, bce = measure_genie_mi(stage, N, Au, fu_set, n_cw=n_cw, batch_size=100)

        # Info MI stats
        info_mi = np.array([mi[p - 1] for p in Au])
        avg_info_mi = float(info_mi.mean())
        weakest_info_mi = float(info_mi.min())
        weakest_info_pos = Au[int(info_mi.argmin())]

        # Polarization
        n_high = int((info_mi > 0.9).sum())
        n_low = int((info_mi < 0.1).sum())
        n_mid = ku - n_high - n_low

        print(f"    Info MI: mean={avg_info_mi:.4f}, min={weakest_info_mi:.4f} (pos {weakest_info_pos})")
        print(f"    Polarized: {n_high} high, {n_low} low, {n_mid} mid")

        trajectory[iteration] = {
            'mi': mi.tolist(),
            'info_mi': info_mi.tolist(),
            'avg_info_mi': avg_info_mi,
            'weakest_info_mi': weakest_info_mi,
            'weakest_info_pos': int(weakest_info_pos),
            'n_high': n_high,
            'n_low': n_low,
            'n_mid': n_mid,
        }

        with open(os.path.join(RESULTS_DIR, 'analysis2_mi_trajectory.json'), 'w') as f:
            json.dump({str(k): v for k, v in trajectory.items()}, f, indent=2)

    # N=64 comparison
    print("\n  N=64 comparison:")
    N64 = 64
    Au64, _, fu64, _, pe_u64, _ = load_design(N64)
    ku64 = N_CONFIGS[N64][0]
    try:
        stage64 = load_stage1_model(N64, d=16, hidden=100)
        mi64, _ = measure_genie_mi(stage64, N64, Au64, fu64, n_cw=n_cw, batch_size=100)
        info_mi64 = np.array([mi64[p - 1] for p in Au64])
        trajectory['N64_comparison'] = {
            'mi': mi64.tolist(),
            'info_mi': info_mi64.tolist(),
            'avg_info_mi': float(info_mi64.mean()),
            'weakest_info_mi': float(info_mi64.min()),
            'n_high': int((info_mi64 > 0.9).sum()),
        }
        print(f"    N=64 Info MI: mean={info_mi64.mean():.4f}, min={info_mi64.min():.4f}")
    except Exception as e:
        print(f"    Failed: {e}")

    with open(os.path.join(RESULTS_DIR, 'analysis2_mi_trajectory.json'), 'w') as f:
        json.dump({str(k): v for k, v in trajectory.items()}, f, indent=2)

    return trajectory


# ─── Analysis 4: ISI-MC Design ───────────────────────────────────────────────

def analysis4():
    """Compare ISI-specific MC error rates with GMAC proxy."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: ISI-MAC MC Design (NPD decode error rates)")
    print("=" * 70)

    results = {}

    for N in [64, 128]:
        n = int(math.log2(N))
        print(f"\n--- N={N} ---")

        d, hidden = 16, 100
        try:
            stage = load_stage1_model(N, d=d, hidden=hidden)
        except:
            continue

        Au, Av, fu_set, fv_set, pe_u, pe_v = load_design(N)
        ku = N_CONFIGS[N][0]

        # Rate-1 decode to get per-position error rates (ISI-specific ranking)
        # Use empty frozen set for rate-1
        n_trials = 20000 if N == 64 else 10000
        print(f"  Rate-1 decode ({n_trials} CW)...")
        pe_rate1 = measure_decode_errors(stage, N, list(range(1, N + 1)),
                                          set(), n_cw=n_trials, batch_size=100)

        # Also get actual-rate error rates
        n_trials2 = 20000 if N == 64 else 10000
        print(f"  Actual-rate decode ({n_trials2} CW)...")
        pe_actual = measure_decode_errors(stage, N, Au, fu_set, n_cw=n_trials2, batch_size=100)

        # ISI-optimal info set from rate-1 Pe
        isi_ranking = np.argsort(pe_rate1)
        isi_info_set = set(isi_ranking[:ku].tolist())

        # GMAC info set
        gmac_info_set = set(p - 1 for p in Au)

        overlap = len(isi_info_set & gmac_info_set)
        overlap_pct = 100 * overlap / ku

        # Correlation
        from scipy.stats import spearmanr
        try:
            corr, _ = spearmanr(pe_rate1, pe_u)
        except:
            corr = float('nan')

        print(f"  ISI vs GMAC overlap: {overlap}/{ku} ({overlap_pct:.1f}%)")
        print(f"  Spearman corr(ISI_Pe, GMAC_Pe): {corr:.4f}")

        # Positions where they disagree
        isi_only = isi_info_set - gmac_info_set
        gmac_only = gmac_info_set - isi_info_set
        if isi_only:
            print(f"  ISI-only info: {sorted(isi_only)}")
            print(f"  GMAC-only info: {sorted(gmac_only)}")
            for p in sorted(isi_only):
                print(f"    ISI-only pos {p}: ISI_Pe={pe_rate1[p]:.4f}, GMAC_Pe={pe_u[p]:.4f}")
            for p in sorted(gmac_only):
                print(f"    GMAC-only pos {p}: ISI_Pe={pe_rate1[p]:.4f}, GMAC_Pe={pe_u[p]:.4f}")

        results[N] = {
            'pe_rate1': pe_rate1.tolist(),
            'pe_actual': pe_actual.tolist(),
            'gmac_pe_u': pe_u.tolist(),
            'ku': ku,
            'overlap': overlap,
            'overlap_pct': float(overlap_pct),
            'spearman_corr': float(corr) if not np.isnan(corr) else None,
            'isi_only_0idx': sorted(list(isi_only)),
            'gmac_only_0idx': sorted(list(gmac_only)),
        }

        with open(os.path.join(RESULTS_DIR, 'analysis4_isi_mc_design.json'), 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    return results


# ─── Plotting ────────────────────────────────────────────────────────────────

def make_plots(a1_results, a2_results, a4_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot 1: Overlap vs N
    print("\nPlot: frozen overlap vs N...")
    Ns = sorted([int(k) for k in a1_results.keys()])
    overlaps = [a1_results[N]['overlap_pct'] for N in Ns]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ns, overlaps, 'bo-', markersize=8, linewidth=2)
    ax.set_xlabel('Block length N')
    ax.set_ylabel('Overlap (%)')
    ax.set_title('NPD-MI vs GMAC Proxy Info Set Overlap')
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    for n_val, ov in zip(Ns, overlaps):
        ax.annotate(f'{ov:.0f}%', (n_val, ov), textcoords="offset points",
                    xytext=(0, 10), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_frozen_overlap_vs_N.pdf'))
    plt.close()

    # Plot 2: MI scatter (info positions only) for N=64, N=128
    for N in [64, 128]:
        if N not in a1_results:
            continue
        print(f"Plot: MI scatter N={N}...")
        mi = np.array(a1_results[N]['mi'])
        pe_u = np.array(a1_results[N]['gmac_pe_u'])
        Au = a1_results[N]['Au']
        pe_decode = np.array(a1_results[N]['pe_decode'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: all positions - MI vs GMAC Pe
        info_mask = np.zeros(N, dtype=bool)
        for p in Au:
            info_mask[p - 1] = True

        ax1.scatter(pe_u[info_mask], mi[info_mask], c='tab:blue', alpha=0.6, s=40, label='Info pos')
        ax1.scatter(pe_u[~info_mask], mi[~info_mask], c='tab:red', alpha=0.3, s=15, label='Frozen pos')
        ax1.set_xlabel('GMAC Proxy Pe')
        ax1.set_ylabel('NPD Genie MI')
        ax1.set_title(f'N={N}: NPD MI vs GMAC Proxy Pe')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: info positions - MI vs actual decode error rate
        ax2.scatter([pe_decode[p - 1] for p in Au],
                    [mi[p - 1] for p in Au], c='tab:blue', alpha=0.6, s=40)
        ax2.set_xlabel('NPD Decode Error Rate')
        ax2.set_ylabel('NPD Genie MI')
        ax2.set_title(f'N={N}: MI vs Decode Error at Info Positions')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'plot_mi_scatter_N{N}.pdf'))
        plt.close()

    # Plot 3: MI trajectory N=128
    if a2_results:
        def _get(d, k):
            if k in d: return d[k]
            if str(k) in d: return d[str(k)]
            return d[int(k)]
        iters_list = sorted([int(k) for k in a2_results.keys() if str(k).isdigit()])
        if iters_list:
            print("Plot: MI trajectory N=128...")
            avg_mis = [_get(a2_results, it)['avg_info_mi'] for it in iters_list]
            weakest = [_get(a2_results, it)['weakest_info_mi'] for it in iters_list]
            n_high = [_get(a2_results, it)['n_high'] for it in iters_list]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            ax1.plot([it / 1000 for it in iters_list], avg_mis, 'bo-', label='N=128 avg info MI', linewidth=2)
            if 'N64_comparison' in a2_results:
                ax1.axhline(y=a2_results['N64_comparison']['avg_info_mi'], color='g',
                            linestyle='--', label=f'N=64 final')
            ax1.set_xlabel('Iteration (K)')
            ax1.set_ylabel('Average Info-Position MI')
            ax1.set_title('N=128: Info MI vs Training Iteration')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot([it / 1000 for it in iters_list], n_high, 'ro-', label='MI > 0.9', linewidth=2)
            ku = N_CONFIGS[128][0]
            ax2.axhline(y=ku, color='gray', linestyle=':', label=f'ku={ku}', alpha=0.5)
            if 'N64_comparison' in a2_results:
                ax2.axhline(y=a2_results['N64_comparison']['n_high'], color='g',
                            linestyle='--', label='N=64 final')
            ax2.set_xlabel('Iteration (K)')
            ax2.set_ylabel('# Info Positions with MI > 0.9')
            ax2.set_title('N=128: Polarization Progress')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'plot_mi_trajectory_N128.pdf'))
            plt.close()

            # Plot 4: Weakest info MI
            print("Plot: weakest info MI...")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([it / 1000 for it in iters_list], weakest, 'ms-', linewidth=2, markersize=8)
            if 'N64_comparison' in a2_results:
                ax.axhline(y=a2_results['N64_comparison']['weakest_info_mi'], color='g',
                            linestyle='--', label='N=64 final')
                ax.legend()
            ax.set_xlabel('Iteration (K)')
            ax.set_ylabel('MI of Weakest Info Position')
            ax.set_title('N=128: Bottleneck Position MI vs Training')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'plot_mi_weakest_info.pdf'))
            plt.close()

    # ISI MC scatter
    if a4_results:
        for N_str, data in a4_results.items():
            N = int(N_str)
            print(f"Plot: ISI vs GMAC Pe N={N}...")
            pe_isi = np.array(data['pe_rate1'])
            gmac_pe = np.array(data['gmac_pe_u'])

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(gmac_pe, pe_isi, alpha=0.5, s=20)
            lims = [0, max(gmac_pe.max(), pe_isi.max()) * 1.1]
            ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
            ax.set_xlabel('GMAC Proxy Pe')
            ax.set_ylabel('ISI-MAC NPD Rate-1 Pe')
            ax.set_title(f'ISI vs GMAC Error Rate (N={N})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'plot_isi_vs_gmac_pe_N{N}.pdf'))
            plt.close()

    print("All plots saved.")


# ─── Summary ─────────────────────────────────────────────────────────────────

def write_summary(a1, a2, a4):
    summary_dir = os.path.join(BASE, 'project_summary')
    os.makedirs(summary_dir, exist_ok=True)

    lines = []
    lines.append("# Frozen Set Analysis: Why the ISI-MAC NPD Wall Exists at Large N")
    lines.append(f"\nDate: 2026-04-16 | SNR=6dB | ISI h=0.3 | sigma2={SIGMA2:.4f}")
    lines.append("")

    # Analysis 1
    lines.append("## Analysis 1: Frozen Set Comparison (NPD-MI vs GMAC Proxy)")
    lines.append("")
    lines.append("Genie-aided MI: at each position, use TRUE prior bits and measure logit quality.")
    lines.append("NPD-MI info set: top-ku positions by MI. GMAC info set: actual design positions.")
    lines.append("")
    lines.append("| N | ku | Overlap | % | Spearman | Info MI mean | Info MI min | Info Pe max | Low MI info |")
    lines.append("|---|---|---------|---|----------|-------------|-------------|-------------|-------------|")
    for N in sorted([int(k) for k in a1.keys()]):
        r = a1[N]
        sp = f"{r['spearman_corr']:.3f}" if r['spearman_corr'] is not None else "N/A"
        lines.append(f"| {N} | {r['ku']} | {r['overlap']}/{r['ku']} | {r['overlap_pct']:.0f}% | {sp} | {r['info_mi_mean']:.4f} | {r['info_mi_min']:.4f} | {r['info_pe_max']:.4f} | {r['n_low_mi_info']} |")
    lines.append("")

    # Overlap trend
    Ns = sorted([int(k) for k in a1.keys()])
    overlaps = [a1[N]['overlap_pct'] for N in Ns]
    if len(Ns) >= 3:
        trend = "DECREASES" if overlaps[-1] < overlaps[0] else "STABLE/INCREASES"
        lines.append(f"**Overlap trend**: {trend} with N ({overlaps[0]:.0f}% at N={Ns[0]} -> {overlaps[-1]:.0f}% at N={Ns[-1]})")
    lines.append("")

    # Analysis 2
    if a2:
        lines.append("## Analysis 2: MI Trajectory During Training (N=128)")
        lines.append("")
        lines.append("| Iteration | Avg Info MI | Weakest MI | Weakest Pos | # High (>0.9) | # Low (<0.1) |")
        lines.append("|-----------|------------|------------|-------------|---------------|-------------|")
        iters_list = sorted([int(k) for k in a2.keys() if k.isdigit()])
        for it in iters_list:
            r = a2[str(it)]
            lines.append(f"| {it:,} | {r['avg_info_mi']:.4f} | {r['weakest_info_mi']:.4f} | {r['weakest_info_pos']} | {r['n_high']} | {r['n_low']} |")

        if 'N64_comparison' in a2:
            r64 = a2['N64_comparison']
            lines.append(f"| N=64 final | {r64['avg_info_mi']:.4f} | {r64['weakest_info_mi']:.4f} | - | {r64['n_high']} | - |")
        lines.append("")

        # Plateau analysis
        if len(iters_list) >= 3:
            last3 = [a2[str(it)]['avg_info_mi'] for it in iters_list[-3:]]
            delta = last3[-1] - last3[0]
            if abs(delta) < 0.01:
                lines.append(f"**MI plateau**: YES, avg info MI stable at ~{last3[-1]:.4f} over last 3 checkpoints (delta={delta:.4f})")
            else:
                lines.append(f"**MI plateau**: NO, still improving (delta={delta:.4f} over last 3 checkpoints)")
        lines.append("")

    # Analysis 3
    lines.append("## Analysis 3: Frozen Set Quality Test (Existing Data)")
    lines.append("")
    lines.append("| Config | BLER | Notes |")
    lines.append("|--------|------|-------|")
    lines.append("| N=64 GMAC proxy | 0.028 | Well-trained |")
    lines.append("| N=64 NPD-MI design | 0.054 | Under-trained (3-phase) |")
    lines.append("| N=128 GMAC proxy | 0.073 | d=16 h=100 |")
    lines.append("| N=128 NPD-MI design | 0.033 | Under-trained (3-phase) |")
    lines.append("")
    lines.append("**Note**: Comparison confounded by training duration.")
    lines.append("")

    # Analysis 4
    if a4:
        lines.append("## Analysis 4: ISI-MAC MC Design (Rate-1 NPD Decode Errors)")
        lines.append("")
        lines.append("| N | ISI-GMAC Overlap | % | Spearman |")
        lines.append("|---|-----------------|---|----------|")
        for N_str in sorted(a4.keys(), key=int):
            r = a4[N_str]
            sp = f"{r['spearman_corr']:.3f}" if r['spearman_corr'] is not None else "N/A"
            lines.append(f"| {N_str} | {r['overlap']}/{r['ku']} | {r['overlap_pct']:.0f}% | {sp} |")
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Does the GMAC proxy frozen set get worse at larger N?**")
    if len(Ns) >= 2:
        if overlaps[-1] < overlaps[0] - 5:
            lines.append(f"   YES. Overlap drops from {overlaps[0]:.0f}% at N={Ns[0]} to {overlaps[-1]:.0f}% at N={Ns[-1]}.")
        else:
            lines.append(f"   NOT CLEARLY. Overlap is {overlaps[0]:.0f}%-{overlaps[-1]:.0f}% across tested N.")
    lines.append("")

    lines.append("2. **Does the NPD fail at positions GMAC thinks are good but ISI finds hard?**")
    for N in [64, 128]:
        if N in a1:
            low = a1[N]['low_mi_info_positions']
            if low:
                lines.append(f"   N={N}: {len(low)} info positions have MI<0.5:")
                for p, m, e in low[:5]:
                    lines.append(f"     pos {p}: MI={m:.4f}, decode_pe={e:.4f}")
    lines.append("")

    lines.append("3. **Does MI plateau during training at N=128?**")
    if a2:
        iters_list = sorted([int(k) for k in a2.keys() if k.isdigit()])
        if iters_list:
            first = a2[str(iters_list[0])]['avg_info_mi']
            last = a2[str(iters_list[-1])]['avg_info_mi']
            lines.append(f"   Info MI goes from {first:.4f} at {iters_list[0]:,} to {last:.4f} at {iters_list[-1]:,} iterations.")
    lines.append("")

    lines.append("4. **Would an ISI-specific MC design give different positions?**")
    if a4:
        for N_str in sorted(a4.keys(), key=int):
            r = a4[N_str]
            lines.append(f"   N={N_str}: {r['overlap']}/{r['ku']} overlap ({r['overlap_pct']:.0f}%). ISI-only: {r['isi_only_0idx'][:5]}")
    lines.append("")

    with open(os.path.join(summary_dir, 'FROZEN_SET_ANALYSIS.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f"Summary written to {summary_dir}/FROZEN_SET_ANALYSIS.md")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()
    print("Starting Frozen Set Analysis...")
    print(f"SNR={SNR_DB}dB, ISI h={ISI_H}, sigma2={SIGMA2:.4f}")

    a1 = analysis1()
    print(f"\nAnalysis 1 done. Elapsed: {(time.time() - t0) / 60:.1f} min")

    a2 = analysis2()
    print(f"\nAnalysis 2 done. Elapsed: {(time.time() - t0) / 60:.1f} min")

    a4 = analysis4()
    print(f"\nAnalysis 4 done. Elapsed: {(time.time() - t0) / 60:.1f} min")

    make_plots(a1, a2, a4)
    write_summary(a1, a2, a4)

    print(f"\nTotal elapsed: {(time.time() - t0) / 60:.1f} min")
    print("DONE.")
