#!/usr/bin/env python3
"""
TASK 2: Demonstrate sign-flip cascade at N=8.
Train tiny model, then show how one wrong decision corrupts all subsequent embeddings.
"""

import sys, os, math, json
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC

# ─── Config ─────────────────────────────────────────────────────────────────
N = 8; n = 3
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
D = 16; HIDDEN = 64; N_LAYERS = 2
channel = GaussianMAC(sigma2=SIGMA2)
br_np = bit_reversal_perm(n)

# For N=8, use simple frozen sets: all info (ku=kv=8 for simplicity during training,
# but for demonstration use partial frozen)
# Actually let's use a simple setup: no frozen bits for training, then demo with specific codeword
fu_nat = set()  # no frozen
fv_nat = set()

# ─── NPD Model (same architecture) ─────────────────────────────────────────
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

    def sc_decode_trace(self, emb, fu_nat, fv_nat, br_np, forced_decisions=None):
        """
        Sequential decode with full trace of leaf embeddings.
        forced_decisions: dict {leaf_idx: forced_value} to override decisions.
        Returns: decisions, leaf_embeddings (pre-logits), leaf_logits
        """
        B=emb.shape[0]; N_=emb.shape[1]
        br_t=torch.from_numpy(br_np).long()
        u_hat=torch.zeros(B,N_,dtype=torch.long)
        v_hat=torch.zeros(B,N_,dtype=torch.long)
        leaf_idx=[0]
        decisions=[]; leaf_embs=[]; leaf_logits=[]

        def _decode(eb):
            bs=eb.shape[1]
            if bs==1:
                leaf_emb = eb[:,0,:].clone()
                logits=self.emb2logits(leaf_emb)
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

                # Override if forced
                if forced_decisions is not None and idx in forced_decisions:
                    dec = torch.full((B,), forced_decisions[idx], dtype=torch.long)

                leaf_embs.append(leaf_emb.detach())
                leaf_logits.append(logits.detach())
                decisions.append(dec.item())

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
        return decisions, leaf_embs, leaf_logits, u_hat, v_hat


# ─── Train model at N=8 ────────────────────────────────────────────────────
print("Training NPD model at N=8 for 5000 iterations...")
model = NPDDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
rng = np.random.default_rng(123)

model.train()
for it in range(5000):
    U = rng.integers(0, 2, size=(32, N)).astype(np.int32)
    V = rng.integers(0, 2, size=(32, N)).astype(np.int32)
    X = polar_encode_batch(U)
    Y = polar_encode_batch(V)
    Z = channel.sample_batch(X, Y)
    z_t = torch.from_numpy(Z).float()
    emb = model.z_encoder(z_t.unsqueeze(-1))
    joint = torch.from_numpy(U).long()*2 + torch.from_numpy(V).long()
    joint_br = joint[:, br_np]
    loss = model.fast_ce_leaf_only(emb, joint_br)
    opt.zero_grad(); loss.backward(); opt.step()
    if (it+1) % 1000 == 0:
        print(f"  iter {it+1}/5000 loss={loss.item():.4f}")

model.eval()
print("Training complete.")

# ─── Generate one specific codeword ─────────────────────────────────────────
print("\n" + "="*70)
print("TASK 2: Sign-flip cascade demonstration at N=8")
print("="*70)

# Find a codeword where correct decode works
rng2 = np.random.default_rng(999)
for trial in range(100):
    U = rng2.integers(0, 2, size=(1, N)).astype(np.int32)
    V = rng2.integers(0, 2, size=(1, N)).astype(np.int32)
    X = polar_encode_batch(U)
    Y = polar_encode_batch(V)
    Z = channel.sample_batch(X, Y)
    z_t = torch.from_numpy(Z).float()
    emb = model.z_encoder(z_t.unsqueeze(-1))

    # True joint in BR order
    joint_nat = U[0]*2 + V[0]
    true_joint_br = joint_nat[br_np]

    # Run correct decode (forced with true values)
    forced_true = {i: int(true_joint_br[i]) for i in range(N)}
    decs_correct, embs_correct, logits_correct, u_c, v_c = model.sc_decode_trace(
        emb, fu_nat, fv_nat, br_np, forced_decisions=forced_true)

    # Check if free-run would get everything right
    decs_free, embs_free, logits_free, u_f, v_f = model.sc_decode_trace(
        emb, fu_nat, fv_nat, br_np, forced_decisions=None)

    n_correct_free = sum(1 for d, t in zip(decs_free, true_joint_br) if d == t)
    if n_correct_free >= 6:  # at least 6/8 correct = decent model
        print(f"Found good codeword at trial {trial}")
        break

print(f"\nTrue joint (BR order): {true_joint_br.tolist()}")
print(f"Free-run decisions:   {decs_free}")
print(f"Correct decisions:    {decs_correct}")
print(f"Free-run correct: {n_correct_free}/{N}")

# ─── Now flip decision at position 0 ────────────────────────────────────────
print("\n--- Corrupting decision at position 0 ---")
true_val_0 = int(true_joint_br[0])
flipped_val_0 = (true_val_0 + 1) % 4  # flip to a different value

forced_flip = {0: flipped_val_0}  # only force position 0, rest free-run
decs_flipped, embs_flipped, logits_flipped, u_fl, v_fl = model.sc_decode_trace(
    emb, fu_nat, fv_nat, br_np, forced_decisions=forced_flip)

print(f"  Position 0: true={true_val_0}, flipped to={flipped_val_0}")
print(f"  Correct decisions: {decs_correct}")
print(f"  Flipped decisions: {decs_flipped}")

# Compare embeddings
print("\n--- Embedding corruption analysis ---")
print(f"  {'Pos':>3s} | {'Correct dec':>11s} | {'Flipped dec':>11s} | {'L2 distance':>11s} | {'Relative':>8s} | {'Correct?':>8s}")
print("  " + "-"*70)

corruption_magnitudes = []
correct_after_flip = []
for i in range(N):
    emb_c = embs_correct[i][0]  # shape (d,)
    emb_f = embs_flipped[i][0]
    l2 = torch.norm(emb_c - emb_f).item()
    norm_c = torch.norm(emb_c).item()
    rel = l2 / (norm_c + 1e-8)
    corruption_magnitudes.append(l2)
    is_correct = decs_flipped[i] == int(true_joint_br[i])
    correct_after_flip.append(is_correct)
    print(f"  {i:3d} | {decs_correct[i]:11d} | {decs_flipped[i]:11d} | {l2:11.4f} | {rel:8.4f} | {'YES' if is_correct else 'NO':>8s}")

print(f"\n  Summary:")
print(f"    Position 0 embedding L2 change: {corruption_magnitudes[0]:.4f}")
print(f"    Avg subsequent (pos 1-7) L2:    {np.mean(corruption_magnitudes[1:]):.4f}")
print(f"    Max subsequent L2:              {np.max(corruption_magnitudes[1:]):.4f}")
print(f"    Positions correct after flip:   {sum(correct_after_flip)}/{N}")
print(f"    Cascade: single flip at pos 0 corrupted {N - sum(correct_after_flip) - (0 if correct_after_flip[0] else 1)} additional positions")

# Also show actual embedding values for positions 0-2
print("\n--- Actual embedding values (first 8 dims) ---")
for i in range(min(3, N)):
    emb_c = embs_correct[i][0, :8].numpy()
    emb_f = embs_flipped[i][0, :8].numpy()
    print(f"  Pos {i} correct:  {np.array2string(emb_c, precision=3, floatmode='fixed')}")
    print(f"  Pos {i} flipped:  {np.array2string(emb_f, precision=3, floatmode='fixed')}")
    print(f"  Pos {i} diff:     {np.array2string(emb_f - emb_c, precision=3, floatmode='fixed')}")
    print()

# Show logits comparison
print("--- Logits comparison ---")
for i in range(N):
    lc = logits_correct[i][0].numpy()
    lf = logits_flipped[i][0].numpy()
    true_v = int(true_joint_br[i])
    print(f"  Pos {i}: true={true_v} correct_logits={np.array2string(lc, precision=2)} "
          f"flipped_logits={np.array2string(lf, precision=2)} "
          f"argmax_c={lc.argmax()} argmax_f={lf.argmax()}")

results = {
    'corruption_magnitudes': corruption_magnitudes,
    'correct_after_flip': correct_after_flip,
    'true_joint_br': true_joint_br.tolist(),
    'decs_correct': decs_correct,
    'decs_flipped': decs_flipped,
}

with open('results/task2_signflip_n8.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "="*70)
print("TASK 2 COMPLETE")
print("="*70)
