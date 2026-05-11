#!/usr/bin/env python3
"""Test Ideas 3 & 4: More training/capacity at N=32, and scaling to N=64."""

import sys, os, math, time
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file

SNR_DB=6.0; SIGMA2=10**(-SNR_DB/10)
DESIGNS_DIR='designs'

def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)

class Decoder(nn.Module):
    def __init__(self, d, hidden=64, n_layers=2):
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
            for a,b,c,dd in zip(el,er,jl,jr):
                E_chunks+=[a,b]; J_chunks+=[c,dd]
        e_all=torch.cat(E_chunks,1); j_all=torch.cat(J_chunks,1)
        logits=self.emb2logits(e_all)
        return F.cross_entropy(logits.reshape(-1,4), j_all.reshape(-1))

    def sc_decode_correct(self, emb, fu_nat, fv_nat, br_np):
        B=emb.shape[0]; N_=emb.shape[1]
        br_t = torch.from_numpy(br_np).long()
        u_hat=torch.zeros(B,N_,dtype=torch.long); v_hat=torch.zeros(B,N_,dtype=torch.long)
        leaf_idx=[0]
        def _decode(eb):
            bs=eb.shape[1]
            if bs==1:
                logits=self.emb2logits(eb[:,0,:])
                idx=leaf_idx[0]; leaf_idx[0]+=1
                nat_idx = int(br_t[idx])
                uf = nat_idx in fu_nat; vf = nat_idx in fv_nat
                if uf and vf: dec=torch.zeros(B,dtype=torch.long)
                elif uf: dec=(logits[:,1]>logits[:,0]).long()
                elif vf: dec=(logits[:,2]>logits[:,0]).long()*2
                else: dec=logits.argmax(dim=-1)
                u_hat[:, nat_idx] = dec // 2; v_hat[:, nat_idx] = dec % 2
                return dec.unsqueeze(1)
            half=bs//2
            e_odd=eb[:,0::2,:]; e_even=eb[:,1::2,:]
            e_left=self.checknode(torch.cat([e_odd,e_even],-1))
            cw_left=_decode(e_left)
            e_right=self.bitnode(e_odd, e_even, cw_left)
            cw_right=_decode(e_right)
            u_l=cw_left//2; v_l=cw_left%2; u_r=cw_right//2; v_r=cw_right%2
            cw_odd=(u_l^u_r)*2+(v_l^v_r); cw_even=cw_right
            result=torch.zeros(B,bs,dtype=torch.long)
            result[:,0::2]=cw_odd; result[:,1::2]=cw_even
            return result
        with torch.no_grad(): _decode(emb)
        return u_hat, v_hat

def setup(N_):
    n_=int(math.log2(N_))
    channel=GaussianMAC(sigma2=SIGMA2)
    sc_ref={32:{'ku':15,'kv':15,'sc':0.046}, 64:{'ku':31,'kv':31,'sc':0.025}, 128:{'ku':62,'kv':62,'sc':0.016}}
    ref=sc_ref[N_]
    mc_path=f'designs/gmac_B_n{n_}_snr{SNR_DB:.0f}dB.npz'
    Au,Av,fu,fv,_,_,_=design_from_file(mc_path,n_,ref['ku'],ref['kv'])
    br_np=bit_reversal_perm(n_)
    br=torch.from_numpy(br_np).long()
    fu_nat={p-1 for p in range(1,N_+1) if p not in Au}
    fv_nat={p-1 for p in range(1,N_+1) if p not in Av}
    return channel, Au, Av, br_np, br, fu_nat, fv_nat, ref['sc']

def evaluate(model, channel, N_, Au, Av, br_np, br, fu_nat, fv_nat, n_cw=2000):
    model.eval(); errs=0; total=0
    rng=np.random.default_rng(999)
    with torch.no_grad():
        while total<n_cw:
            actual=min(32,n_cw-total)
            uf=np.zeros((actual,N_),dtype=int); vf=np.zeros((actual,N_),dtype=int)
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
            if total % 100 == 0:
                print(f'    eval: {total}/{n_cw} cw, errs={errs}', flush=True)
    model.train()
    return errs/total

def train_fastce(model, channel, Au, Av, br, N_, iters, lr, batch=128):
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    rng=np.random.default_rng()
    model.train()
    for it in range(1, iters+1):
        uf=np.zeros((batch,N_),dtype=int); vf=np.zeros((batch,N_),dtype=int)
        for p in Au: uf[:,p-1]=rng.integers(0,2,batch)
        for p in Av: vf[:,p-1]=rng.integers(0,2,batch)
        xf=polar_encode_batch(uf); yf=polar_encode_batch(vf)
        zf=torch.from_numpy(channel.sample_batch(xf,yf)).float()
        emb=model.z_encoder(zf.unsqueeze(-1))[:,br]
        joint_cw=torch.from_numpy(xf*2+yf).long()[:,br]
        loss=model.fast_ce_leaf_only(emb, joint_cw)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it%5000==0:
            print(f'    [{it}/{iters}] loss={loss.item():.4f}', flush=True)


def run_experiment(label, N_train, N_eval, d, iters, batch, lr):
    print(f'\n--- {label} ---')
    print(f'  Train N={N_train}, Eval N={N_eval}, d={d}, iters={iters}, batch={batch}')

    # Setup for training
    ch_train, Au_train, Av_train, br_np_train, br_train, fu_nat_train, fv_nat_train, sc_ref_train = setup(N_train)

    # Build model and train
    model = Decoder(d=d)
    t0 = time.time()
    train_fastce(model, ch_train, Au_train, Av_train, br_train, N_train, iters, lr, batch)
    t_train = time.time() - t0
    print(f'  Training time: {t_train:.1f}s')

    # Evaluate
    ch_eval, Au_eval, Av_eval, br_np_eval, br_eval, fu_nat_eval, fv_nat_eval, sc_ref_eval = setup(N_eval)
    bler = evaluate(model, ch_eval, N_eval, Au_eval, Av_eval, br_np_eval, br_eval, fu_nat_eval, fv_nat_eval, n_cw=500)
    ratio = bler / sc_ref_eval if sc_ref_eval > 0 else float('inf')
    print(f'  BLER={bler:.4f} ({ratio:.1f}x SC, SC={sc_ref_eval})')
    return bler, sc_ref_eval, ratio, model


if __name__ == '__main__':
    os.chdir('/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')
    torch.manual_seed(42)
    np.random.seed(42)

    results = {}

    # === IDEA 3: More training / d=32 at N=32 ===
    print('='*50)
    print('=== IDEA 3: More training / d=32 ===')
    print('='*50)

    # A) d=16, 50K iters
    bler_a, sc_a, ratio_a, _ = run_experiment(
        'Idea 3A: d=16, 50K iters', N_train=32, N_eval=32, d=16, iters=50000, batch=128, lr=3e-4)
    results['3A'] = (bler_a, sc_a, ratio_a)

    # B) d=32, 15K iters
    bler_b, sc_b, ratio_b, _ = run_experiment(
        'Idea 3B: d=32, 15K iters', N_train=32, N_eval=32, d=32, iters=15000, batch=128, lr=3e-4)
    results['3B'] = (bler_b, sc_b, ratio_b)

    # C) d=32, 50K iters
    bler_c, sc_c, ratio_c, model_3c = run_experiment(
        'Idea 3C: d=32, 50K iters', N_train=32, N_eval=32, d=32, iters=50000, batch=128, lr=3e-4)
    results['3C'] = (bler_c, sc_c, ratio_c)

    # === IDEA 4: Scale to N=64 ===
    print('\n' + '='*50)
    print('=== IDEA 4: Scale to N=64 ===')
    print('='*50)

    # Train N=64, eval N=64
    bler_d, sc_d, ratio_d, _ = run_experiment(
        'Idea 4A: Train N=64, eval N=64', N_train=64, N_eval=64, d=16, iters=30000, batch=64, lr=3e-4)
    results['4A'] = (bler_d, sc_d, ratio_d)

    # Train N=32, eval N=64 (generalization) — use model_3c (d=32, best N=32 model)
    # But we need d=16 for fair comparison, so train a fresh d=16 model at N=32
    print('\n--- Idea 4B: Train N=32 d=16 50K, eval N=64 (generalization) ---')
    # Reuse 3A model concept but need the actual model — retrain quickly or just use a d=16 model
    # Actually let's train a dedicated one
    bler_gen, sc_gen, ratio_gen, _ = run_experiment(
        'Idea 4B: Train N=32, eval N=64 (generalization)', N_train=32, N_eval=64, d=16, iters=30000, batch=128, lr=3e-4)
    results['4B'] = (bler_gen, sc_gen, ratio_gen)

    # === Summary ===
    print('\n' + '='*50)
    print('SUMMARY')
    print('='*50)
    print(f'\n=== IDEA 3: More training / d=32 ===')
    print(f'A) d=16, 50K iters: BLER={results["3A"][0]:.4f} ({results["3A"][2]:.1f}x SC)')
    print(f'B) d=32, 15K iters: BLER={results["3B"][0]:.4f} ({results["3B"][2]:.1f}x SC)')
    print(f'C) d=32, 50K iters: BLER={results["3C"][0]:.4f} ({results["3C"][2]:.1f}x SC)')
    print(f'\n=== IDEA 4: Scale to N=64 ===')
    print(f'Train N=64, eval N=64: BLER={results["4A"][0]:.4f} ({results["4A"][2]:.1f}x SC)')
    print(f'Train N=32, eval N=64: BLER={results["4B"][0]:.4f} (generalization)')
