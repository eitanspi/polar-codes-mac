#!/usr/bin/env python3
"""Same MI diagnostic as n256_mi_diagnostic.py but for N=128 (where CG matches SC)."""
import os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from polar.encoder import polar_encode_batch, build_message_batch
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 128; n_log = 7; ku = kv = 62
SNR_DB = 6.0; SIGMA2 = 10 ** (-SNR_DB / 10)
SEED = 42; N_CW = 5000; BATCH = 64

def load_nn(ckpt_path, d=16, hidden=64):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model = GmacNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=2)
    model_sd = model.state_dict()
    for k, v in sd.items():
        nk = k
        if nk.startswith('tree.'): nk = nk[5:]
        elif nk.startswith('z_enc.'): nk = 'z_encoder.' + nk[6:]
        if 'embedding_z' in nk: continue
        if nk in model_sd and model_sd[nk].shape == v.shape:
            model_sd[nk] = v
    model.load_state_dict(model_sd)
    model.eval()
    return model

def H_binary(p):
    if p <= 0 or p >= 1: return 0.0
    return -(p * math.log2(p) + (1-p) * math.log2(1-p))

def main():
    os.chdir(ROOT)
    Au, Av, frozen_u, frozen_v, pe_u_raw, pe_v_raw, path_i = design_from_file(
        f'designs/gmac_B_n{n_log}_snr6dB.npz', n_log, ku=ku, kv=kv)
    b = make_path(N, path_i)
    channel = GaussianMAC(sigma2=SIGMA2)
    Au_set, Av_set = set(Au), set(Av)

    step_meta = []
    i_u, i_v = 0, 0
    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0: i_u += 1; i_t = i_u; fdict = frozen_u
        else: i_v += 1; i_t = i_v; fdict = frozen_v
        if i_t in fdict: continue
        if i_t in Au_set and i_t in Av_set: pos_type = 'both_info'
        elif i_t in Au_set: pos_type = 'u_only'
        elif i_t in Av_set: pos_type = 'v_only'
        else: pos_type = 'unknown'
        step_meta.append({
            'step': step, 'pos': i_t, 'user': gamma, 'pos_type': pos_type,
            'pe_u': float(pe_u_raw[i_t-1]), 'pe_v': float(pe_v_raw[i_t-1]),
        })

    n_info_steps = len(step_meta)
    print(f"N={N}: {n_info_steps} non-frozen steps")

    rng = np.random.default_rng(SEED)
    U_info = rng.integers(0, 2, size=(N_CW, ku), dtype=np.int32)
    V_info = rng.integers(0, 2, size=(N_CW, kv), dtype=np.int32)
    U_msg = build_message_batch(N, U_info, Au)
    V_msg = build_message_batch(N, V_info, Av)
    X = polar_encode_batch(U_msg)
    Y = polar_encode_batch(V_msg)
    np.random.seed(SEED + 7919)
    Z = channel.sample_batch(X, Y).astype(np.float32)

    model = load_nn('saved_models/ncg_gmac_mlp_N128.pt')
    print(f"Loaded N=128 model")

    ce_user_sum = np.zeros(n_info_steps, dtype=np.float64)
    correct_count = np.zeros(n_info_steps, dtype=np.int64)
    count = np.zeros(n_info_steps, dtype=np.int64)

    t0 = time.time()
    for start in range(0, N_CW, BATCH):
        end = min(start + BATCH, N_CW)
        bs = end - start
        z_t = torch.from_numpy(Z[start:end]).float()
        u_t = torch.from_numpy(U_msg[start:end].astype(np.float32))
        v_t = torch.from_numpy(V_msg[start:end].astype(np.float32))
        with torch.no_grad():
            all_logits, all_targets, _, _, _ = model(
                z_t, b, frozen_u, frozen_v, u_true=u_t, v_true=v_t)
        for idx in range(n_info_steps):
            logits = all_logits[idx]
            log_probs = F.log_softmax(logits, dim=-1)
            gamma = step_meta[idx]['user']
            if gamma == 0:
                lp0 = torch.logsumexp(log_probs[:, :2], dim=1)
                lp1 = torch.logsumexp(log_probs[:, 2:], dim=1)
                true_val = u_t[:, step_meta[idx]['pos']-1].long()
            else:
                lp0 = torch.logsumexp(log_probs[:, [0,2]], dim=1)
                lp1 = torch.logsumexp(log_probs[:, [1,3]], dim=1)
                true_val = v_t[:, step_meta[idx]['pos']-1].long()
            ce = -torch.where(true_val == 0, lp0, lp1)
            ce_user_sum[idx] += ce.sum().item()
            pred = (lp1 > lp0).long()
            correct_count[idx] += (pred == true_val).sum().item()
            count[idx] += bs
    print(f"Done in {time.time()-t0:.1f}s")

    mean_ce = ce_user_sum / count
    accuracy = correct_count / count
    mi_user = np.clip(1.0 - mean_ce / math.log(2), 0, 1)
    genie_mi = np.array([1.0 - H_binary(s['pe_u'] if s['user']==0 else s['pe_v'])
                         for s in step_meta])
    mi_gap = genie_mi - mi_user

    print(f"\nN={N} MI diagnostic:")
    print(f"  MI gap: mean={mi_gap.mean():.4f}, max={mi_gap.max():.4f}")
    print(f"  Accuracy: mean={accuracy.mean():.4f}, min={accuracy.min():.4f}")
    print(f"  User MI: mean={mi_user.mean():.4f}, min={mi_user.min():.4f}")

    # Top 10 worst
    sorted_idx = np.argsort(mi_gap)[::-1]
    print(f"\nTop 10 worst MI gaps:")
    print(f"{'step':<6}{'pos':<6}{'user':<6}{'genie':>8}{'model':>8}{'gap':>8}{'acc':>8}")
    for i in sorted_idx[:10]:
        s = step_meta[i]
        user = 'U' if s['user']==0 else 'V'
        print(f"{s['step']:<6}{s['pos']:<6}{user:<6}"
              f"{genie_mi[i]:>8.4f}{mi_user[i]:>8.4f}"
              f"{mi_gap[i]:>8.4f}{accuracy[i]:>8.4f}")

if __name__ == '__main__':
    main()
