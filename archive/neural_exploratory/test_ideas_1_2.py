#!/usr/bin/env python3
"""
test_ideas_1_2.py — Test two ideas for improving neural polar MAC decoder.

Idea 1: Hybrid fast_ce pretrain -> sequential fine-tune
Idea 2: PSS on corrected decoder
"""
import sys, os, math, time
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file

D=16; HIDDEN=64; N_LAYERS=2; N=32; n=5; BATCH=128
SNR_DB=6.0; SIGMA2=10**(-SNR_DB/10)
SC_BLER=0.046

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)

class Decoder(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, d))
        self.checknode = _make_mlp(2*d, HIDDEN, d, N_LAYERS)
        self.bitnode_mlp = _make_mlp(2*d, HIDDEN, d, N_LAYERS)
        self.emb2logits = _make_mlp(d, HIDDEN, 4, N_LAYERS)

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
            for a,b,c,dd in zip(el,er,jl,jr):
                E_chunks+=[a,b]; J_chunks+=[c,dd]
        e_all=torch.cat(E_chunks,1); j_all=torch.cat(J_chunks,1)
        logits=self.emb2logits(e_all)
        return F.cross_entropy(logits.reshape(-1,4), j_all.reshape(-1))

    def fast_ce_with_preds(self, emb, joint_cw):
        """fast_ce that also returns per-depth predictions for PSS."""
        B, N_, d = emb.shape; n_ = int(math.log2(N_))
        E_chunks = [emb]; J_chunks = [joint_cw]
        depth_preds = []  # predictions at each depth
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
            J_left_true=(u_o^u_e)*2+(v_o^v_e); J_right=J_even
            e_left=self.checknode(torch.cat([E_odd,E_even],-1))
            # For PSS: compute predictions at this depth's left children
            with torch.no_grad():
                left_logits = self.emb2logits(e_left)
                left_preds = left_logits.argmax(dim=-1)
                depth_preds.append(left_preds)
            e_right=self.bitnode(E_odd, E_even, J_left_true)
            nc=2**depth; cs=(N_//2)//nc
            el=torch.split(e_left,cs,1); er=torch.split(e_right,cs,1)
            jl=torch.split(J_left_true,cs,1); jr=torch.split(J_right,cs,1)
            E_chunks=[]; J_chunks=[]
            for a,b,c,dd in zip(el,er,jl,jr):
                E_chunks+=[a,b]; J_chunks+=[c,dd]
        e_all=torch.cat(E_chunks,1); j_all=torch.cat(J_chunks,1)
        logits=self.emb2logits(e_all)
        loss = F.cross_entropy(logits.reshape(-1,4), j_all.reshape(-1))
        return loss, depth_preds

    def fast_ce_pss(self, emb, joint_cw, pred_left_list, p_sample=0.5):
        """fast_ce with PSS: mix true and predicted left decisions at BitNode."""
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
            J_left_true=(u_o^u_e)*2+(v_o^v_e); J_right=J_even

            e_left=self.checknode(torch.cat([E_odd,E_even],-1))

            # PSS: mix true and predicted left decisions for bitnode
            if pred_left_list is not None:
                pred_l = pred_left_list[depth]
                # Derive pred_left from predictions using XOR logic
                # pred_left_list[depth] contains predictions for left children embeddings
                # We need to convert those to "J_left" format
                # The predictions are the joint (u,v) decisions at left children
                # For bitnode, J_left is used directly
                mask = (torch.rand(B, J_left_true.shape[1]) < p_sample).long()
                J_left_mixed = J_left_true * (1 - mask) + pred_l * mask
            else:
                J_left_mixed = J_left_true

            e_right=self.bitnode(E_odd, E_even, J_left_mixed)
            nc=2**depth; cs=(N_//2)//nc
            el=torch.split(e_left,cs,1); er=torch.split(e_right,cs,1)
            jl=torch.split(J_left_true,cs,1); jr=torch.split(J_right,cs,1)
            E_chunks=[]; J_chunks=[]
            for a,b,c,dd in zip(el,er,jl,jr):
                E_chunks+=[a,b]; J_chunks+=[c,dd]
        e_all=torch.cat(E_chunks,1); j_all=torch.cat(J_chunks,1)
        logits=self.emb2logits(e_all)
        return F.cross_entropy(logits.reshape(-1,4), j_all.reshape(-1))

    def sequential_forward(self, emb, br_np, fu_nat, fv_nat, u_true_nat=None, v_true_nat=None):
        """Sequential tree walk with teacher forcing for training.
        Returns (loss, u_hat, v_hat)."""
        B=emb.shape[0]; N_=emb.shape[1]; n_=int(math.log2(N_))
        br_t = torch.from_numpy(br_np).long()
        u_hat=torch.zeros(B,N_,dtype=torch.long)
        v_hat=torch.zeros(B,N_,dtype=torch.long)
        leaf_idx=[0]
        all_logits=[]; all_targets=[]
        teacher = u_true_nat is not None

        def _decode(eb):
            bs=eb.shape[1]
            if bs==1:
                logits=self.emb2logits(eb[:,0,:])
                idx=leaf_idx[0]; leaf_idx[0]+=1
                nat_idx = int(br_t[idx])
                uf = nat_idx in fu_nat; vf = nat_idx in fv_nat

                if not (uf and vf):
                    all_logits.append(logits)
                    if teacher:
                        target = u_true_nat[:, nat_idx]*2 + v_true_nat[:, nat_idx]
                        all_targets.append(target)

                if teacher:
                    dec_u = u_true_nat[:, nat_idx]; dec_v = v_true_nat[:, nat_idx]
                    dec = dec_u * 2 + dec_v
                else:
                    if uf and vf: dec=torch.zeros(B,dtype=torch.long)
                    elif uf: dec=(logits[:,1]>logits[:,0]).long()
                    elif vf: dec=(logits[:,2]>logits[:,0]).long()*2
                    else: dec=logits.argmax(dim=-1)
                    dec_u = dec // 2; dec_v = dec % 2

                u_hat[:, nat_idx] = dec_u; v_hat[:, nat_idx] = dec_v
                return dec.unsqueeze(1)

            half=bs//2
            e_odd=eb[:,0::2,:]; e_even=eb[:,1::2,:]
            e_left=self.checknode(torch.cat([e_odd,e_even],-1))
            cw_left=_decode(e_left)
            e_right=self.bitnode(e_odd, e_even, cw_left)
            cw_right=_decode(e_right)
            u_l=cw_left//2; v_l=cw_left%2
            u_r=cw_right//2; v_r=cw_right%2
            cw_odd = (u_l^u_r)*2 + (v_l^v_r)
            cw_even = cw_right
            result=torch.zeros(B,bs,dtype=torch.long)
            result[:,0::2]=cw_odd; result[:,1::2]=cw_even
            return result

        _decode(emb)
        if teacher and all_logits:
            loss = F.cross_entropy(torch.stack(all_logits,1).reshape(-1,4),
                                   torch.stack(all_targets,1).reshape(-1))
        else:
            loss = torch.tensor(0.0)
        return loss, u_hat, v_hat

    def sc_decode_correct(self, emb, fu_nat, fv_nat, br_np):
        """Correct sequential decode (no teacher forcing)."""
        _, u_hat, v_hat = self.sequential_forward(emb, br_np, fu_nat, fv_nat)
        return u_hat, v_hat


# ── Setup ──
channel=GaussianMAC(sigma2=SIGMA2)
mc_path=f'/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs/gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz'
Au,Av,fu,fv,_,_,_=design_from_file(mc_path,n,15,15)
br_np = bit_reversal_perm(n)
br=torch.from_numpy(br_np).long()
fu_nat={p-1 for p in range(1,N+1) if p not in Au}
fv_nat={p-1 for p in range(1,N+1) if p not in Av}

def evaluate(model, n_cw=2000):
    model.eval(); errs=0; total=0
    rng=np.random.default_rng(999)
    with torch.no_grad():
        while total<n_cw:
            actual=min(32,n_cw-total)
            uf=np.zeros((actual,N),dtype=int); vf=np.zeros((actual,N),dtype=int)
            for p in Au: uf[:,p-1]=rng.integers(0,2,actual)
            for p in Av: vf[:,p-1]=rng.integers(0,2,actual)
            xf=polar_encode_batch(uf); yf=polar_encode_batch(vf)
            zf=torch.from_numpy(channel.sample_batch(xf,yf)).float()
            emb=model.z_encoder(zf.unsqueeze(-1))[:,br]
            u_dec,v_dec=model.sc_decode_correct(emb, fu_nat, fv_nat, br_np)
            for i in range(actual):
                ue=any(u_dec[i,p-1].item()!=uf[i,p-1] for p in Au)
                ve=any(v_dec[i,p-1].item()!=vf[i,p-1] for p in Av)
                if ue or ve: errs+=1
            total+=actual
    model.train()
    return errs/total

def gen_batch(rng):
    uf=np.zeros((BATCH,N),dtype=int); vf=np.zeros((BATCH,N),dtype=int)
    for p in Au: uf[:,p-1]=rng.integers(0,2,BATCH)
    for p in Av: vf[:,p-1]=rng.integers(0,2,BATCH)
    xf=polar_encode_batch(uf); yf=polar_encode_batch(vf)
    zf=torch.from_numpy(channel.sample_batch(xf,yf)).float()
    return uf, vf, xf, yf, zf


# ══════════════════════════════════════════════════════════════════════════════
#  IDEA 1: Hybrid fast_ce pretrain -> sequential fine-tune
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("IDEA 1: Hybrid fast_ce -> sequential fine-tune")
print("=" * 60)

torch.manual_seed(42)
np.random.seed(42)
model1 = Decoder(d=D)
opt1 = torch.optim.Adam(model1.parameters(), lr=3e-4)
rng1 = np.random.default_rng(42)

# Phase 1: fast_ce 15K iters
print("\nPhase 1: fast_ce pretraining (15K iters)...")
t0 = time.time()
for it in range(15000):
    uf, vf, xf, yf, zf = gen_batch(rng1)
    emb = model1.z_encoder(zf.unsqueeze(-1))[:, br]
    joint_cw = torch.from_numpy(xf * 2 + yf).long()
    loss = model1.fast_ce_leaf_only(emb, joint_cw)
    opt1.zero_grad(); loss.backward(); opt1.step()
    if (it+1) % 5000 == 0:
        print(f"  iter {it+1}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

bler1_phase1 = evaluate(model1)
print(f"\nPhase 1 result: BLER={bler1_phase1:.4f} ({bler1_phase1/SC_BLER:.1f}x SC)")

# Phase 2: sequential fine-tune 10K iters
print("\nPhase 2: sequential fine-tune (10K iters, lr=1e-4)...")
opt1_ft = torch.optim.Adam(model1.parameters(), lr=1e-4)
t0 = time.time()
for it in range(10000):
    uf, vf, xf, yf, zf = gen_batch(rng1)
    emb = model1.z_encoder(zf.unsqueeze(-1))[:, br]
    u_true = torch.from_numpy(uf).long()
    v_true = torch.from_numpy(vf).long()
    loss, _, _ = model1.sequential_forward(emb, br_np, fu_nat, fv_nat,
                                            u_true_nat=u_true, v_true_nat=v_true)
    opt1_ft.zero_grad(); loss.backward(); opt1_ft.step()
    if (it+1) % 2000 == 0:
        print(f"  iter {it+1}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

bler1_phase2 = evaluate(model1)
print(f"\nPhase 2 result: BLER={bler1_phase2:.4f} ({bler1_phase2/SC_BLER:.1f}x SC)")


# ══════════════════════════════════════════════════════════════════════════════
#  IDEA 2: PSS on corrected decoder
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("IDEA 2: PSS on corrected decoder")
print("=" * 60)

torch.manual_seed(42)
np.random.seed(42)
model2 = Decoder(d=D)
opt2 = torch.optim.Adam(model2.parameters(), lr=3e-4)
rng2 = np.random.default_rng(42)

# Phase 1: fast_ce 15K iters (same as Idea 1)
print("\nPhase 1: fast_ce pretraining (15K iters)...")
t0 = time.time()
for it in range(15000):
    uf, vf, xf, yf, zf = gen_batch(rng2)
    emb = model2.z_encoder(zf.unsqueeze(-1))[:, br]
    joint_cw = torch.from_numpy(xf * 2 + yf).long()
    loss = model2.fast_ce_leaf_only(emb, joint_cw)
    opt2.zero_grad(); loss.backward(); opt2.step()
    if (it+1) % 5000 == 0:
        print(f"  iter {it+1}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

bler2_phase1 = evaluate(model2)
print(f"\nPhase 1 result: BLER={bler2_phase1:.4f} ({bler2_phase1/SC_BLER:.1f}x SC)")

# Phase 2: PSS fine-tuning 15K iters
print("\nPhase 2: PSS fine-tuning (15K iters, lr=1e-4, p=0.5)...")
opt2_ft = torch.optim.Adam(model2.parameters(), lr=1e-4)
t0 = time.time()
for it in range(15000):
    uf, vf, xf, yf, zf = gen_batch(rng2)
    emb = model2.z_encoder(zf.unsqueeze(-1))[:, br]
    joint_cw = torch.from_numpy(xf * 2 + yf).long()

    # Pass 1 (no grad): get predictions
    model2.eval()
    with torch.no_grad():
        _, depth_preds = model2.fast_ce_with_preds(emb, joint_cw)
    model2.train()

    # Pass 2 (with grad): PSS training with p=0.5 mixing
    loss = model2.fast_ce_pss(emb, joint_cw, depth_preds, p_sample=0.5)
    opt2_ft.zero_grad(); loss.backward(); opt2_ft.step()
    if (it+1) % 5000 == 0:
        print(f"  iter {it+1}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

bler2_phase2 = evaluate(model2)
print(f"\nPhase 2 result: BLER={bler2_phase2:.4f} ({bler2_phase2/SC_BLER:.1f}x SC)")


# ══════════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n=== IDEA 1: Hybrid fast_ce -> sequential fine-tune ===")
print(f"Phase 1 (fast_ce 15K): BLER={bler1_phase1:.4f} ({bler1_phase1/SC_BLER:.1f}x SC)")
print(f"Phase 2 (sequential 10K): BLER={bler1_phase2:.4f} ({bler1_phase2/SC_BLER:.1f}x SC)")
print(f"\n=== IDEA 2: PSS on corrected decoder ===")
print(f"Phase 1 (fast_ce 15K): BLER={bler2_phase1:.4f} ({bler2_phase1/SC_BLER:.1f}x SC)")
print(f"Phase 2 (PSS p=0.5 15K): BLER={bler2_phase2:.4f} ({bler2_phase2/SC_BLER:.1f}x SC)")
