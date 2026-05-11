#!/usr/bin/env python3
"""Try chaining p3 stage 1 with curriculum stage 2 (cross-checkpoint chained eval)."""
import os, sys, math, json, time
import numpy as np
import torch
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
torch.set_num_threads(2)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.channels.mac_channel import build_channel as build_c_channel

SNR_DB = 6.0
N = 64
n_log = 6


def wilson_ci(errs, total, z=1.96):
    if total == 0: return (0.0, 1.0)
    p = errs / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_model(path):
    ck = torch.load(path, weights_only=False, map_location='cpu')
    m = NPDSingleUser(d=ck.get('d', 16), hidden=ck.get('hidden', 64),
                     n_layers=ck.get('n_layers', 2), z_dim=ck.get('z_dim', 1))
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m, ck


def eval_chain(m1, m2, Au, Av, n_cw=2000, seed=20260416 + 600 + N):
    sigma2 = 10 ** (-SNR_DB / 10)
    channel = build_c_channel('gmac', sigma2=sigma2)
    br = bit_reversal_perm(n_log)
    fu_set = {p - 1 for p in range(1, N + 1) if p not in Au}
    fv_set = {p - 1 for p in range(1, N + 1) if p not in Av}
    errs = 0
    errs_u = 0
    errs_v = 0
    total = 0
    batch = 16
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u_msg = np.zeros((actual, N), dtype=np.int8)
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Au: u_msg[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av: v_msg[:, p - 1] = rng.integers(0, 2, actual)
            x_phys = polar_encode_batch(u_msg.astype(int))
            y_phys = polar_encode_batch(v_msg.astype(int))
            z = channel.sample_z(x_phys.astype(int), y_phys.astype(int))
            features1 = channel.stage1_features(z)
            features1_npd = (features1[..., br] if features1.ndim == 2
                             else features1[:, br, :])
            ft1 = torch.from_numpy(features1_npd).float()
            if ft1.dim() == 2: ft1 = ft1.unsqueeze(-1)
            emb1 = m1.encode_channel(ft1)
            u_hat = m1.decode(emb1, fu_set)
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)
            features2 = channel.stage2_features(z, x_hat.astype(int))
            features2_npd = (features2[..., br] if features2.ndim == 2
                             else features2[:, br, :])
            ft2 = torch.from_numpy(features2_npd).float()
            if ft2.dim() == 2: ft2 = ft2.unsqueeze(-1)
            emb2 = m2.encode_channel(ft2)
            v_hat = m2.decode(emb2, fv_set)
            for i in range(actual):
                u_wrong = any(int(u_hat[i, p - 1].item()) != int(u_msg[i, p - 1]) for p in Au)
                v_wrong = any(int(v_hat[i, p - 1].item()) != int(v_msg[i, p - 1]) for p in Av)
                if u_wrong: errs_u += 1
                if v_wrong: errs_v += 1
                if u_wrong or v_wrong: errs += 1
            total += actual
    return errs, errs_u, errs_v, total


def main():
    p3 = os.path.join(_ROOT, 'class_c_npd/results/npd_design_p3_N64_best.pt')
    s2 = os.path.join(_ROOT, 'class_c_npd/results/curriculum_gmac_c_s2_N64_best.pt')
    m_p3, ck_p3 = load_model(p3)
    m_s2, ck_s2 = load_model(s2)
    Au = sorted(ck_p3['Au'])  # NPD-picked design
    Av = sorted(ck_s2['Av'])  # curriculum stage 2 Av
    print(f'Au (p3): {Au}')
    print(f'Av (s2): {Av}')
    # Note: p3's Av may differ from s2's Av — mismatch!
    # For a valid chained eval, we need the encoder Av to match what stage 2
    # was trained to decode.
    if sorted(ck_p3['Av']) != Av:
        print(f'WARNING: p3 Av={sorted(ck_p3["Av"])} != s2 Av={Av}')

    # Run chained eval with p3_s1 + curriculum_s2, using Av from s2
    t0 = time.time()
    errs, eu, ev, tot = eval_chain(m_p3, m_s2, Au, Av, n_cw=2000)
    t = time.time() - t0
    bler = errs / tot; ci = wilson_ci(errs, tot)
    print(f'\nChain (p3_s1 + curriculum_s2): BLER={bler:.4f} [{ci[0]:.4f},{ci[1]:.4f}] '
          f'u_err={eu} v_err={ev}/{tot} t={t:.1f}s')

    # Save
    out = {
        'description': 'Chained eval with p3 stage1 + curriculum stage2',
        's1_Au': Au, 's2_Av': Av,
        's1_checkpoint': 'npd_design_p3_N64_best.pt',
        's2_checkpoint': 'curriculum_gmac_c_s2_N64_best.pt',
        'bler': bler, 'errs_u': eu, 'errs_v': ev, 'n_cw': tot,
        'ci_lo': ci[0], 'ci_hi': ci[1],
        'note': 'Cross-ckpt chained: Stage 1 uses npd-chosen Au, Stage 2 uses '
                'curriculum Av. Because p3 was trained for a different design, '
                'this is not a clean production path.',
    }
    with open(os.path.join(_ROOT, 'results/snr_sweep/task1b_chain_swap.json'), 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
