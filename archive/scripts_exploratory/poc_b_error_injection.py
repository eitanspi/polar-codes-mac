#!/usr/bin/env python3
"""
POC-B: Error-injection cascade test.
Flip a single U-info bit at positions 17, 20, 30 (teacher forcing with one flipped bit)
and measure downstream MI drop vs baseline teacher forcing.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch

torch.set_num_threads(2)

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 32
SIGMA2 = 10 ** (-6.0 / 10)
CKPT = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd',
                    'results', 'ncg_r1_32', 'iter300000.pt')
N_SAMPLES = 2000
BATCH = 256
SEED = 42
# path = [0]*16 + [1]*32 + [0]*16.
# U positions 1-16 decoded at steps 0-15; V positions 1-32 at steps 16-47; U 17-32 at steps 48-63.
# "Flip pos 20 (U)" ambiguous; user specifies "U-info, first N/2 = 16 steps". But they also
# say "flip pos 30 (U-info, strong polarization)" — pos 30 is U pos 30 (decoded at step 61).
# Let's interpret the positions as U POSITIONS (1-32), since they say "U-info":
#   pos 17 = U position 17 (step 48, 1st U after V block)
#   pos 20 = U position 20 (step 51)
#   pos 30 = U position 30 (step 61)
FLIPS_U = [17, 20, 30]


def binary_entropy(p):
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def mi_per_position(all_logits, b):
    """Return per-U-position MI and per-V-position MI (length N each)."""
    h_u = np.zeros(N); cnt_u = np.zeros(N)
    h_v = np.zeros(N); cnt_v = np.zeros(N)
    idx_u = idx_v = 0
    for step in range(2 * N):
        gamma = b[step]
        lg = all_logits[step]
        if gamma == 0:
            lp0 = torch.logsumexp(lg[:, :2], 1)
            lp1 = torch.logsumexp(lg[:, 2:], 1)
        else:
            lp0 = torch.logsumexp(lg[:, [0, 2]], 1)
            lp1 = torch.logsumexp(lg[:, [1, 3]], 1)
        p0 = torch.sigmoid(lp0 - lp1).cpu().numpy()
        h = binary_entropy(p0).sum()
        Bsize = lg.shape[0]
        if gamma == 0:
            idx_u += 1
            h_u[idx_u - 1] += h; cnt_u[idx_u - 1] += Bsize
        else:
            idx_v += 1
            h_v[idx_v - 1] += h; cnt_v[idx_v - 1] += Bsize
    mi_u = 1.0 - h_u / np.maximum(cnt_u, 1)
    mi_v = 1.0 - h_v / np.maximum(cnt_v, 1)
    return mi_u, mi_v, cnt_u, cnt_v


def measure_with_optional_flip(model, ch, b, n_samples, batch, device, seed, flip_u=None):
    """If flip_u is a 1-indexed U position, flip that bit in u_true before feeding model."""
    rng = np.random.default_rng(seed)
    h_u = np.zeros(N); cnt_u = np.zeros(N)
    h_v = np.zeros(N); cnt_v = np.zeros(N)
    done = 0
    while done < n_samples:
        B = min(batch, n_samples - done)
        u = rng.integers(0, 2, (B, N)).astype(int)
        v = rng.integers(0, 2, (B, N)).astype(int)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float().to(device)
        ut = torch.from_numpy(u).float().to(device)
        vt = torch.from_numpy(v).float().to(device)

        if flip_u is not None:
            # flip 1-indexed U position (u_true's (flip_u - 1) column)
            ut = ut.clone()
            ut[:, flip_u - 1] = 1.0 - ut[:, flip_u - 1]

        with torch.no_grad():
            L, _, _, _, _ = model(z, b, {}, {}, u_true=ut, v_true=vt)

        # Accumulate
        idx_u = idx_v = 0
        for step in range(2 * N):
            gamma = b[step]
            lg = L[step]
            if gamma == 0:
                lp0 = torch.logsumexp(lg[:, :2], 1)
                lp1 = torch.logsumexp(lg[:, 2:], 1)
            else:
                lp0 = torch.logsumexp(lg[:, [0, 2]], 1)
                lp1 = torch.logsumexp(lg[:, [1, 3]], 1)
            p0 = torch.sigmoid(lp0 - lp1).cpu().numpy()
            h = binary_entropy(p0).sum()
            if gamma == 0:
                idx_u += 1
                h_u[idx_u - 1] += h; cnt_u[idx_u - 1] += B
            else:
                idx_v += 1
                h_v[idx_v - 1] += h; cnt_v[idx_v - 1] += B
        done += B

    mi_u = 1.0 - h_u / np.maximum(cnt_u, 1)
    mi_v = 1.0 - h_v / np.maximum(cnt_v, 1)
    return mi_u, mi_v


def main():
    device = torch.device('cpu')
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()
    ch = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)

    # Step index for each U position (1-indexed)
    u_pos_to_step = []
    idx_u = 0
    for step, g in enumerate(b):
        if g == 0:
            idx_u += 1
            u_pos_to_step.append((idx_u, step))
    u_pos_step_map = {p: s for p, s in u_pos_to_step}

    print(f'POC-B: N=32, samples={N_SAMPLES}, seed={SEED}')
    print(f'U position -> step index: {u_pos_step_map}')
    print()

    # Baseline
    t0 = time.time()
    mi_u_base, mi_v_base = measure_with_optional_flip(
        model, ch, b, N_SAMPLES, BATCH, device, SEED, flip_u=None)
    print(f'Baseline (teacher forced, no flip): '
          f'avg MI_U={mi_u_base.mean():.4f}  avg MI_V={mi_v_base.mean():.4f}  '
          f'({time.time()-t0:.1f}s)')

    results = {}
    for flip_u in FLIPS_U:
        t0 = time.time()
        mi_u_f, mi_v_f = measure_with_optional_flip(
            model, ch, b, N_SAMPLES, BATCH, device, SEED, flip_u=flip_u)
        results[flip_u] = (mi_u_f, mi_v_f)
        print(f'Flip U-pos {flip_u} (step {u_pos_step_map[flip_u]}): '
              f'avg MI_U={mi_u_f.mean():.4f}  avg MI_V={mi_v_f.mean():.4f}  '
              f'({time.time()-t0:.1f}s)')

    # Analysis: MI drop downstream of flip
    print()
    print('=== Downstream MI impact ===')
    print('For each flipped U-pos, report drop at later decoded positions (all that come after step s_flip).')
    print()
    print(f'{"flip_U":<8} {"step":<5} {"mean_drop_after_U":<20} {"mean_drop_after_V":<20} '
          f'{"local (next 1 U)":<18} {"local (next 1 V after)":<22}')

    for flip_u in FLIPS_U:
        mi_u_f, mi_v_f = results[flip_u]
        s_flip = u_pos_step_map[flip_u]

        # Compute per-position drop
        drop_u = mi_u_base - mi_u_f   # +ve = MI dropped after flip
        drop_v = mi_v_base - mi_v_f

        # Which U positions are decoded after step s_flip?
        later_u_pos = [p for p, s in u_pos_step_map.items() if s > s_flip]
        # Which V positions are decoded after step s_flip? (V 1..32 decoded steps 16..47)
        later_v_pos = [vp for vp in range(1, N + 1) if (16 + (vp - 1)) > s_flip]

        drops_u_later = [drop_u[p - 1] for p in later_u_pos] if later_u_pos else [0.0]
        drops_v_later = [drop_v[p - 1] for p in later_v_pos] if later_v_pos else [0.0]

        # Find the immediately next U / V position by step
        next_u = min(later_u_pos) if later_u_pos else None
        next_v = min(later_v_pos) if later_v_pos else None
        local_u = f'pos {next_u}: {drop_u[next_u-1]:+.4f}' if next_u else 'none'
        local_v = f'pos {next_v}: {drop_v[next_v-1]:+.4f}' if next_v else 'none'

        print(f'{flip_u:<8} {s_flip:<5} '
              f'mean={np.mean(drops_u_later):+.4f} n={len(later_u_pos):<3} '
              f'mean={np.mean(drops_v_later):+.4f} n={len(later_v_pos):<3} '
              f'{local_u:<18} {local_v:<22}')

    print()
    print('Per-position drops (MI_baseline - MI_flipped), U-side:')
    hdr = 'pos  base    ' + '  '.join(f'flip{p:02d}' for p in FLIPS_U) + '   ' + \
          '  '.join(f'drp{p:02d}' for p in FLIPS_U)
    print(hdr)
    for p in range(1, N + 1):
        row = f'{p:3d}  {mi_u_base[p-1]:.4f}  '
        for fu in FLIPS_U:
            row += f'{results[fu][0][p-1]:.4f}  '
        row += '  '
        for fu in FLIPS_U:
            d = mi_u_base[p-1] - results[fu][0][p-1]
            row += f'{d:+.4f}  '
        print(row)

    print()
    print('Per-position drops (MI_baseline - MI_flipped), V-side:')
    print(hdr)
    for p in range(1, N + 1):
        row = f'{p:3d}  {mi_v_base[p-1]:.4f}  '
        for fu in FLIPS_U:
            row += f'{results[fu][1][p-1]:.4f}  '
        row += '  '
        for fu in FLIPS_U:
            d = mi_v_base[p-1] - results[fu][1][p-1]
            row += f'{d:+.4f}  '
        print(row)


if __name__ == '__main__':
    main()
