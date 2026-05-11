#!/usr/bin/env python3
"""
Independent verification of NPD failure mode at N=256.
Tasks 1A-1D: Teacher-forced vs free-running, first-error histogram,
oracle-first-k, overfit-10 codewords.
"""

import sys, os, math, time, json
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file
from polar.design import make_path

# ─── Config ─────────────────────────────────────────────────────────────────
N = 256; n = 8
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
D = 16; HIDDEN = 64; N_LAYERS = 2
DESIGN_FILE = 'designs/gmac_B_n8_snr6dB.npz'
DEVICE = 'cpu'

# ─── Load design ────────────────────────────────────────────────────────────
Au, Av, frozen_u, frozen_v, pe_u, pe_v, path_i = design_from_file(DESIGN_FILE, n, ku=123, kv=123)
b = make_path(N, path_i)
br_np = bit_reversal_perm(n)
channel = GaussianMAC(sigma2=SIGMA2)

print(f"Design loaded: N={N}, path_i={path_i}, ku={len(Au)}, kv={len(Av)}")
print(f"  frozen_u positions: {len(frozen_u)}, frozen_v: {len(frozen_v)}")

# ─── NPD Model ──────────────────────────────────────────────────────────────
def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)

class NPDDecoder(nn.Module):
    def __init__(self, d=D, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, hidden), nn.ELU(), nn.Linear(hidden, d))
        self.checknode = _make_mlp(2*d, hidden, d, n_layers)
        self.bitnode_mlp = _make_mlp(2*d, hidden, d, n_layers)
        self.emb2logits = _make_mlp(d, hidden, 4, n_layers)

    def bitnode(self, e_odd, e_even, uv_left):
        u_left = uv_left // 2; v_left = uv_left % 2
        u_sign = (1.0 - 2.0 * u_left.float()).unsqueeze(-1)
        v_sign = (1.0 - 2.0 * v_left.float()).unsqueeze(-1)
        h = self.d // 2
        e_signed = torch.cat([e_odd[:,:,:h]*u_sign, e_odd[:,:,h:]*v_sign], dim=-1)
        inp = torch.cat([e_signed, e_even], dim=-1)
        return self.bitnode_mlp(inp) + e_signed + e_even

    def fast_ce_leaf_only(self, emb, joint_cw):
        B, N_, d = emb.shape; n_ = int(math.log2(N_))
        E_chunks = [emb]; J_chunks = [joint_cw]
        for depth in range(n_):
            E_odds, E_evens, J_odds, J_evens = [], [], [], []
            for e, j in zip(E_chunks, J_chunks):
                M = e.shape[1]
                E_odds.append(e.reshape(B, M//2, 2, d)[:,:,0,:])
                E_evens.append(e.reshape(B, M//2, 2, d)[:,:,1,:])
                J_odds.append(j.reshape(B, M//2, 2)[:,:,0])
                J_evens.append(j.reshape(B, M//2, 2)[:,:,1])
            E_odd=torch.cat(E_odds,1); E_even=torch.cat(E_evens,1)
            J_odd=torch.cat(J_odds,1); J_even=torch.cat(J_evens,1)
            u_o=J_odd//2; v_o=J_odd%2; u_e=J_even//2; v_e=J_even%2
            J_left=(u_o^u_e)*2+(v_o^v_e); J_right=J_even
            e_left=self.checknode(torch.cat([E_odd,E_even],-1))
            e_right=self.bitnode(E_odd, E_even, J_left)
            nc=2**depth; cs=(N_//2)//nc
            el=torch.split(e_left,cs,1); er=torch.split(e_right,cs,1)
            jl=torch.split(J_left,cs,1); jr=torch.split(J_right,cs,1)
            E_chunks=[]; J_chunks=[]
            for a,bb,c,dd in zip(el,er,jl,jr):
                E_chunks+=[a,bb]; J_chunks+=[c,dd]
        e_all=torch.cat(E_chunks,1); j_all=torch.cat(J_chunks,1)
        logits=self.emb2logits(e_all)
        return F.cross_entropy(logits.reshape(-1,4), j_all.reshape(-1))

    def sc_decode_detailed(self, emb, fu_nat, fv_nat, br_np, true_joint_br=None, oracle_k=0):
        B=emb.shape[0]; N_=emb.shape[1]
        br_t=torch.from_numpy(br_np).long()
        u_hat=torch.zeros(B,N_,dtype=torch.long)
        v_hat=torch.zeros(B,N_,dtype=torch.long)
        leaf_idx=[0]; info_leaf_idx=[0]
        per_leaf_correct=[]; leaf_logits_list=[]

        def _decode(eb):
            bs=eb.shape[1]
            if bs==1:
                logits=self.emb2logits(eb[:,0,:])
                idx=leaf_idx[0]; leaf_idx[0]+=1
                nat_idx=int(br_t[idx])
                uf_=nat_idx in fu_nat; vf_=nat_idx in fv_nat

                if uf_ and vf_:
                    dec=torch.zeros(B,dtype=torch.long)
                elif uf_:
                    dec=(logits[:,1]>logits[:,0]).long()
                elif vf_:
                    dec=(logits[:,2]>logits[:,0]).long()*2
                else:
                    dec=logits.argmax(dim=-1)

                is_info = not (uf_ and vf_)
                if is_info and true_joint_br is not None:
                    true_dec = true_joint_br[:, idx]
                    correct = (dec == true_dec).all().item()
                    per_leaf_correct.append(correct)
                    leaf_logits_list.append(logits.detach())

                    if info_leaf_idx[0] < oracle_k:
                        dec = true_dec
                    info_leaf_idx[0] += 1

                u_hat[:,nat_idx]=dec//2; v_hat[:,nat_idx]=dec%2
                return dec.unsqueeze(1)

            half=bs//2
            e_odd=eb[:,0::2,:]; e_even=eb[:,1::2,:]
            e_left=self.checknode(torch.cat([e_odd,e_even],-1))
            cw_left=_decode(e_left)
            e_right=self.bitnode(e_odd,e_even,cw_left)
            cw_right=_decode(e_right)
            u_l=cw_left//2;v_l=cw_left%2;u_r=cw_right//2;v_r=cw_right%2
            cw_odd=(u_l^u_r)*2+(v_l^v_r);cw_even=cw_right
            result=torch.zeros(B,bs,dtype=torch.long)
            result[:,0::2]=cw_odd;result[:,1::2]=cw_even
            return result

        with torch.no_grad(): _decode(emb)
        return u_hat, v_hat, per_leaf_correct, leaf_logits_list


# ─── Data generation ────────────────────────────────────────────────────────
def generate_batch(batch_size, seed=None):
    rng = np.random.default_rng(seed)
    U = np.zeros((batch_size, N), dtype=np.int32)
    V = np.zeros((batch_size, N), dtype=np.int32)
    for pos in Au:
        U[:, pos-1] = rng.integers(0, 2, size=batch_size)
    for pos in Av:
        V[:, pos-1] = rng.integers(0, 2, size=batch_size)
    X = polar_encode_batch(U)
    Y = polar_encode_batch(V)
    Z = channel.sample_batch(X, Y)
    return U, V, X, Y, Z

def prepare_emb_and_truth(model, Z, U, V):
    z_t = torch.from_numpy(Z).float().unsqueeze(0) if Z.ndim == 1 else torch.from_numpy(Z).float()
    emb = model.z_encoder(z_t.unsqueeze(-1))
    # Build true joint in bit-reversed order
    B = z_t.shape[0]
    joint_nat = torch.from_numpy(U).long() * 2 + torch.from_numpy(V).long()
    true_joint_br = joint_nat[:, br_np]
    return emb, true_joint_br

# Build frozen sets in 0-indexed natural order for NPD
fu_nat = set(pos - 1 for pos in frozen_u.keys())
fv_nat = set(pos - 1 for pos in frozen_v.keys())

# ─── Load best model ────────────────────────────────────────────────────────
model = NPDDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
checkpoint_candidates = [
    'saved_models/decisive_B_best.pt',
    'saved_models/decisive_A_best.pt',
    'saved_models/fastce_n256_best.pt',
    'saved_models/decisive_C_best.pt',
]
loaded = False
for ckpt in checkpoint_candidates:
    try:
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt}")
        loaded = True
        break
    except Exception as e:
        print(f"  Failed to load {ckpt}: {e}")

if not loaded:
    print("No checkpoint loaded. Training fresh model with fast_ce...")
    model = NPDDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    model.train()
    for it in range(30000):
        U, V, X, Y, Z = generate_batch(16, seed=it)
        z_t = torch.from_numpy(Z).float()
        emb = model.z_encoder(z_t.unsqueeze(-1))
        joint_cw_nat = torch.from_numpy(U).long()*2 + torch.from_numpy(V).long()
        joint_cw_br = joint_cw_nat[:, br_np]
        loss = model.fast_ce_leaf_only(emb, joint_cw_br)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it+1) % 5000 == 0:
            print(f"  iter {it+1}/30000, loss={loss.item():.4f}")
    torch.save(model.state_dict(), 'saved_models/verify_fastce_n256.pt')
    print("Saved fresh model to saved_models/verify_fastce_n256.pt")

model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params}")

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1A: Teacher-forced vs free-running leaf accuracy
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 1A: Teacher-forced (oracle) vs Free-running leaf accuracy")
print("="*70)

N_CW = 200
tf_correct_total = 0; tf_total = 0
fr_correct_total = 0; fr_total = 0
fr_bler_count = 0

for i in range(N_CW):
    U, V, X, Y, Z = generate_batch(1, seed=1000+i)
    emb, true_joint_br = prepare_emb_and_truth(model, Z, U, V)

    # Teacher-forced (all oracle)
    _, _, tf_correct, _ = model.sc_decode_detailed(emb, fu_nat, fv_nat, br_np, true_joint_br, oracle_k=999999)
    tf_correct_total += sum(tf_correct)
    tf_total += len(tf_correct)

    # Free-running
    u_hat, v_hat, fr_correct, _ = model.sc_decode_detailed(emb, fu_nat, fv_nat, br_np, true_joint_br, oracle_k=0)
    fr_correct_total += sum(fr_correct)
    fr_total += len(fr_correct)

    # Check BLER
    u_np = u_hat.numpy(); v_np = v_hat.numpy()
    u_err = (u_np != U).any(); v_err = (v_np != V).any()
    if u_err or v_err:
        fr_bler_count += 1

tf_acc = tf_correct_total / tf_total if tf_total > 0 else 0
fr_acc = fr_correct_total / fr_total if fr_total > 0 else 0
fr_bler = fr_bler_count / N_CW

print(f"  Teacher-forced leaf accuracy: {tf_acc:.6f} ({tf_correct_total}/{tf_total})")
print(f"  Free-running  leaf accuracy: {fr_acc:.6f} ({fr_correct_total}/{fr_total})")
print(f"  GAP (TF - FR):              {tf_acc - fr_acc:.6f}")
print(f"  Free-running BLER:          {fr_bler:.4f} ({fr_bler_count}/{N_CW})")

results = {
    'task1a': {
        'tf_leaf_acc': tf_acc,
        'fr_leaf_acc': fr_acc,
        'gap': tf_acc - fr_acc,
        'fr_bler': fr_bler,
        'n_codewords': N_CW,
    }
}

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1B: First-error-position histogram
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 1B: First-error-position histogram (free-running)")
print("="*70)

first_error_positions = []
for i in range(N_CW):
    U, V, X, Y, Z = generate_batch(1, seed=1000+i)
    emb, true_joint_br = prepare_emb_and_truth(model, Z, U, V)
    _, _, fr_correct, _ = model.sc_decode_detailed(emb, fu_nat, fv_nat, br_np, true_joint_br, oracle_k=0)

    first_err = -1
    for j, c in enumerate(fr_correct):
        if not c:
            first_err = j
            break
    first_error_positions.append(first_err)

# Histogram
buckets = {'[0-4]': 0, '[5-9]': 0, '[10-19]': 0, '[20-49]': 0, '[50-99]': 0, '[100+]': 0, 'no_error': 0}
for pos in first_error_positions:
    if pos == -1:
        buckets['no_error'] += 1
    elif pos <= 4:
        buckets['[0-4]'] += 1
    elif pos <= 9:
        buckets['[5-9]'] += 1
    elif pos <= 19:
        buckets['[10-19]'] += 1
    elif pos <= 49:
        buckets['[20-49]'] += 1
    elif pos <= 99:
        buckets['[50-99]'] += 1
    else:
        buckets['[100+]'] += 1

print("  First-error position histogram:")
for k, v in buckets.items():
    print(f"    {k:>10s}: {v:3d} ({v/N_CW*100:.1f}%)")

results['task1b'] = buckets

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1C: Oracle-first-k experiment
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 1C: Oracle-first-k BLER curve")
print("="*70)

k_values = [0, 5, 10, 20, 50, 100, 150, 200]
oracle_results = {}

for k in k_values:
    bler_count = 0
    for i in range(N_CW):
        U, V, X, Y, Z = generate_batch(1, seed=1000+i)
        emb, true_joint_br = prepare_emb_and_truth(model, Z, U, V)
        u_hat, v_hat, _, _ = model.sc_decode_detailed(emb, fu_nat, fv_nat, br_np, true_joint_br, oracle_k=k)
        u_np = u_hat.numpy(); v_np = v_hat.numpy()
        if (u_np != U).any() or (v_np != V).any():
            bler_count += 1
    bler = bler_count / N_CW
    oracle_results[k] = bler
    print(f"  oracle_k={k:4d}  BLER={bler:.4f}  ({bler_count}/{N_CW})")

results['task1c'] = oracle_results

# ═════════════════════════════════════════════════════════════════════════════
# TASK 1D: Overfit 10 codewords
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TASK 1D: Overfit 10 fixed codewords with sequential teacher forcing")
print("="*70)

def sequential_forward_tf(model, emb, br_np, fu_nat, fv_nat, u_true, v_true):
    B=emb.shape[0]; N_=emb.shape[1]
    br_t=torch.from_numpy(br_np).long()
    leaf_idx=[0]; all_logits=[]; all_targets=[]
    def _decode(eb):
        bs=eb.shape[1]
        if bs==1:
            logits=model.emb2logits(eb[:,0,:])
            idx=leaf_idx[0]; leaf_idx[0]+=1
            nat_idx=int(br_t[idx])
            uf_=nat_idx in fu_nat; vf_=nat_idx in fv_nat
            if not (uf_ and vf_):
                all_logits.append(logits)
                all_targets.append(u_true[:,nat_idx]*2+v_true[:,nat_idx])
            dec=u_true[:,nat_idx]*2+v_true[:,nat_idx]
            return dec.unsqueeze(1)
        half=bs//2
        e_odd=eb[:,0::2,:]; e_even=eb[:,1::2,:]
        e_left=model.checknode(torch.cat([e_odd,e_even],-1))
        cw_left=_decode(e_left)
        e_right=model.bitnode(e_odd,e_even,cw_left)
        cw_right=_decode(e_right)
        u_l=cw_left//2;v_l=cw_left%2;u_r=cw_right//2;v_r=cw_right%2
        cw_odd=(u_l^u_r)*2+(v_l^v_r);cw_even=cw_right
        result=torch.zeros(B,bs,dtype=torch.long)
        result[:,0::2]=cw_odd;result[:,1::2]=cw_even
        return result
    _decode(emb)
    if all_logits:
        return F.cross_entropy(torch.stack(all_logits,1).reshape(-1,4),
                               torch.stack(all_targets,1).reshape(-1))
    return torch.tensor(0.0)

# Generate 10 fixed codewords
rng_fix = np.random.default_rng(42)
U_fix = np.zeros((10, N), dtype=np.int32)
V_fix = np.zeros((10, N), dtype=np.int32)
for pos in Au:
    U_fix[:, pos-1] = rng_fix.integers(0, 2, size=10)
for pos in Av:
    V_fix[:, pos-1] = rng_fix.integers(0, 2, size=10)
X_fix = polar_encode_batch(U_fix)
Y_fix = polar_encode_batch(V_fix)
Z_fix = channel.sample_batch(X_fix, Y_fix)

z_fix_t = torch.from_numpy(Z_fix).float()
u_fix_t = torch.from_numpy(U_fix).long()
v_fix_t = torch.from_numpy(V_fix).long()

# Fresh model for overfit test
overfit_model = NPDDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
opt = torch.optim.Adam(overfit_model.parameters(), lr=1e-3)

overfit_log = []
for it in range(5000):
    overfit_model.train()
    emb = overfit_model.z_encoder(z_fix_t.unsqueeze(-1))
    loss = sequential_forward_tf(overfit_model, emb, br_np, fu_nat, fv_nat, u_fix_t, v_fix_t)
    opt.zero_grad(); loss.backward(); opt.step()

    if (it+1) % 500 == 0 or it == 0:
        # Evaluate BLER on these 10 codewords
        overfit_model.eval()
        bler_count = 0
        with torch.no_grad():
            for ci in range(10):
                emb_i = overfit_model.z_encoder(z_fix_t[ci:ci+1].unsqueeze(-1))
                true_joint_br_i = (u_fix_t[ci:ci+1] * 2 + v_fix_t[ci:ci+1])[:, br_np]
                u_h, v_h, _, _ = overfit_model.sc_decode_detailed(emb_i, fu_nat, fv_nat, br_np, true_joint_br_i, oracle_k=0)
                if (u_h.numpy() != U_fix[ci:ci+1]).any() or (v_h.numpy() != V_fix[ci:ci+1]).any():
                    bler_count += 1
        bler = bler_count / 10
        overfit_log.append((it+1, loss.item(), bler))
        print(f"  iter {it+1:5d}  loss={loss.item():.4f}  BLER={bler:.2f} ({bler_count}/10)")

results['task1d'] = {
    'overfit_log': [(it, float(l), float(b)) for it, l, b in overfit_log],
    'final_bler': overfit_log[-1][2] if overfit_log else None,
}

# Save results
with open('results/task1_npd_verification.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nResults saved to results/task1_npd_verification.json")
print("\n" + "="*70)
print("TASK 1 COMPLETE")
print("="*70)
