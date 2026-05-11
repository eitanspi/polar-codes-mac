#!/usr/bin/env python3
"""
decisive_n256.py — 30-hour decisive comparison at N=256.

Three experiments, same architecture, same evaluation:
  A: fast_ce (parallel, leaf-only)
  B: Sequential baseline (curriculum N=32→64→128→256)
  C: Hybrid (fast_ce pretrain at N=256 → sequential fine-tune)

Usage:
  python -u neural/decisive_n256.py A   # run experiment A only
  python -u neural/decisive_n256.py B   # run experiment B only
  python -u neural/decisive_n256.py C   # run experiment C only
"""
import sys, os, math, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design_mc import design_from_file

# ─── Config ──────────────────────────────────────────────────────────────────
D = 16; HIDDEN = 64; N_LAYERS = 2
SNR_DB = 6.0; SIGMA2 = 10 ** (-SNR_DB / 10)
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SC_REF = {32: 0.046, 64: 0.025, 128: 0.016, 256: 0.005}
KU_KV = {32: 15, 64: 31, 128: 62, 256: 123}

THRESHOLDS = [0.99, 0.95, 0.90, 0.80, 0.50, 0.20, 0.10, 0.05]

# ─── Model ───────────────────────────────────────────────────────────────────
def _make_mlp(in_dim, hidden, out_dim, n_layers=2):
    layers = [nn.Linear(in_dim, hidden), nn.ELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)

class Decoder(nn.Module):
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
            for a,b,c,dd in zip(el,er,jl,jr):
                E_chunks+=[a,b]; J_chunks+=[c,dd]
        e_all=torch.cat(E_chunks,1); j_all=torch.cat(J_chunks,1)
        logits=self.emb2logits(e_all)
        return F.cross_entropy(logits.reshape(-1,4), j_all.reshape(-1))

    def sequential_forward(self, emb, br_np, fu_nat, fv_nat, u_true, v_true):
        """Sequential tree walk with teacher forcing. Returns loss."""
        B=emb.shape[0]; N_=emb.shape[1]
        br_t = torch.from_numpy(br_np).long()
        leaf_idx=[0]; all_logits=[]; all_targets=[]
        def _decode(eb):
            bs=eb.shape[1]
            if bs==1:
                logits=self.emb2logits(eb[:,0,:])
                idx=leaf_idx[0]; leaf_idx[0]+=1
                nat_idx = int(br_t[idx])
                uf_=nat_idx in fu_nat; vf_=nat_idx in fv_nat
                if not (uf_ and vf_):
                    all_logits.append(logits)
                    all_targets.append(u_true[:,nat_idx]*2+v_true[:,nat_idx])
                dec = u_true[:,nat_idx]*2+v_true[:,nat_idx]
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
        _decode(emb)
        if all_logits:
            return F.cross_entropy(torch.stack(all_logits,1).reshape(-1,4),
                                   torch.stack(all_targets,1).reshape(-1))
        return torch.tensor(0.0)

    def sc_decode(self, emb, fu_nat, fv_nat, br_np):
        """Correct sequential decode (inference)."""
        B=emb.shape[0]; N_=emb.shape[1]
        br_t=torch.from_numpy(br_np).long()
        u_hat=torch.zeros(B,N_,dtype=torch.long);v_hat=torch.zeros(B,N_,dtype=torch.long)
        leaf_idx=[0]
        def _decode(eb):
            bs=eb.shape[1]
            if bs==1:
                logits=self.emb2logits(eb[:,0,:])
                idx=leaf_idx[0];leaf_idx[0]+=1
                nat_idx=int(br_t[idx])
                uf_=nat_idx in fu_nat;vf_=nat_idx in fv_nat
                if uf_ and vf_: dec=torch.zeros(B,dtype=torch.long)
                elif uf_: dec=(logits[:,1]>logits[:,0]).long()
                elif vf_: dec=(logits[:,2]>logits[:,0]).long()*2
                else: dec=logits.argmax(dim=-1)
                u_hat[:,nat_idx]=dec//2;v_hat[:,nat_idx]=dec%2
                return dec.unsqueeze(1)
            half=bs//2
            e_odd=eb[:,0::2,:];e_even=eb[:,1::2,:]
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
        return u_hat, v_hat

# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_design(N):
    n=int(math.log2(N)); ku=KU_KV[N]
    mc_path=os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz')
    Au,Av,fu,fv,_,_,_=design_from_file(mc_path,n,ku,ku)
    br_np=bit_reversal_perm(n); br=torch.from_numpy(br_np).long()
    fu_nat={p-1 for p in range(1,N+1) if p not in Au}
    fv_nat={p-1 for p in range(1,N+1) if p not in Av}
    return Au,Av,br_np,br,fu_nat,fv_nat

def evaluate(model, channel, N, Au, Av, br_np, br, fu_nat, fv_nat, n_cw=500):
    model.eval(); errs=0; total=0
    rng=np.random.default_rng(999)
    with torch.no_grad():
        while total<n_cw:
            actual=min(8, n_cw-total)
            uf=np.zeros((actual,N),dtype=int);vf=np.zeros((actual,N),dtype=int)
            for p in Au: uf[:,p-1]=rng.integers(0,2,actual)
            for p in Av: vf[:,p-1]=rng.integers(0,2,actual)
            xf=polar_encode_batch(uf);yf=polar_encode_batch(vf)
            zf=torch.from_numpy(channel.sample_batch(xf,yf)).float()
            emb=model.z_encoder(zf.unsqueeze(-1))[:,br]
            u_dec,v_dec=model.sc_decode(emb,fu_nat,fv_nat,br_np)
            for i in range(actual):
                ue=any(u_dec[i,p-1].item()!=uf[i,p-1] for p in Au)
                ve=any(v_dec[i,p-1].item()!=vf[i,p-1] for p in Av)
                if ue or ve: errs+=1
            total+=actual
    model.train()
    return errs/total

class Logger:
    def __init__(self, tag):
        self.tag = tag
        self.log = []
        self.best_bler = 1.0
        self.best_iter = 0
        self.crossed = {}
        self.t0 = time.time()
    def record(self, it, loss, bler):
        elapsed = (time.time()-self.t0)/60
        if bler < self.best_bler:
            self.best_bler = bler; self.best_iter = it
        for t in THRESHOLDS:
            if t not in self.crossed and bler < t:
                self.crossed[t] = (it, elapsed)
                print(f'  *** THRESHOLD: BLER < {t} at iter {it} ({elapsed:.0f}min) ***', flush=True)
        sc = SC_REF.get(256, 0.005)
        ratio = bler / max(sc, 1e-8)
        print(f'[{self.tag}][{it:>7}] loss={loss:.4f} BLER={bler:.4f} '
              f'(best={self.best_bler:.4f}, {ratio:.1f}x SC) {elapsed:.0f}min', flush=True)
        self.log.append({'iter':it,'loss':loss,'bler':bler,'best':self.best_bler,'min':elapsed})
    def save(self):
        path = os.path.join(RESULTS_DIR, f'decisive_{self.tag}.json')
        with open(path,'w') as f:
            json.dump({'tag':self.tag,'best_bler':self.best_bler,'best_iter':self.best_iter,
                        'thresholds':self.crossed,'log':self.log}, f, indent=2, default=str)
        print(f'Results saved to {path}', flush=True)

# ─── Experiment A: fast_ce ────────────────────────────────────────────────────
def run_A():
    N=256; BATCH=16; LR=5e-4
    MAX_HOURS=10; MAX_ITERS=500000
    channel=GaussianMAC(sigma2=SIGMA2)
    Au,Av,br_np,br,fu_nat,fv_nat=load_design(N)
    model=Decoder(d=D); opt=torch.optim.Adam(model.parameters(),lr=LR)
    rng=np.random.default_rng()
    log=Logger('A_fastce')
    print(f'\n{"="*60}', flush=True)
    print(f'Experiment A: fast_ce at N={N}, d={D}, batch={BATCH}, lr={LR}', flush=True)
    print(f'Max {MAX_HOURS}h or {MAX_ITERS} iters', flush=True)
    print(f'{"="*60}', flush=True)
    t0=time.time(); model.train()
    for it in range(1, MAX_ITERS+1):
        if (time.time()-t0)/3600 > MAX_HOURS:
            print(f'Time limit reached at iter {it}', flush=True); break
        uf=np.zeros((BATCH,N),dtype=int);vf=np.zeros((BATCH,N),dtype=int)
        for p in Au: uf[:,p-1]=rng.integers(0,2,BATCH)
        for p in Av: vf[:,p-1]=rng.integers(0,2,BATCH)
        xf=polar_encode_batch(uf);yf=polar_encode_batch(vf)
        zf=torch.from_numpy(channel.sample_batch(xf,yf)).float()
        emb=model.z_encoder(zf.unsqueeze(-1))[:,br]
        joint_cw=torch.from_numpy(xf*2+yf).long()[:,br]
        loss=model.fast_ce_leaf_only(emb,joint_cw)
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if it%2000==0:
            bler=evaluate(model,channel,N,Au,Av,br_np,br,fu_nat,fv_nat,500)
            log.record(it,loss.item(),bler)
            if bler < log.best_bler + 0.001:
                torch.save(model.state_dict(),os.path.join(SAVE_DIR,f'decisive_A_best.pt'))
    log.save()
    return log

# ─── Experiment B: Sequential curriculum ──────────────────────────────────────
def run_B():
    channel=GaussianMAC(sigma2=SIGMA2)
    stages = [
        {'N':32,  'iters':15000, 'batch':32, 'lr':3e-4, 'eval_every':5000},
        {'N':64,  'iters':20000, 'batch':16, 'lr':2e-4, 'eval_every':5000},
        {'N':128, 'iters':30000, 'batch':8,  'lr':1e-4, 'eval_every':5000},
        {'N':256, 'iters':200000,'batch':4,  'lr':5e-5, 'eval_every':5000},
    ]
    MAX_HOURS=10
    model=Decoder(d=D)
    log=Logger('B_sequential')
    print(f'\n{"="*60}', flush=True)
    print(f'Experiment B: Sequential curriculum → N=256', flush=True)
    print(f'Stages: {" → ".join(str(s["N"]) for s in stages)}', flush=True)
    print(f'Max {MAX_HOURS}h total', flush=True)
    print(f'{"="*60}', flush=True)
    t0_global=time.time(); total_it=0
    for si, stage in enumerate(stages):
        N=stage['N']; BATCH=stage['batch']; LR=stage['lr']
        n=int(math.log2(N))
        Au,Av,br_np,br,fu_nat,fv_nat=load_design(N)
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        rng=np.random.default_rng()
        print(f'\n--- Stage {si+1}: N={N}, batch={BATCH}, lr={LR}, {stage["iters"]} iters ---', flush=True)
        model.train()
        for it in range(1, stage['iters']+1):
            if (time.time()-t0_global)/3600 > MAX_HOURS:
                print(f'Global time limit at stage {si+1} iter {it}', flush=True); break
            uf=np.zeros((BATCH,N),dtype=int);vf=np.zeros((BATCH,N),dtype=int)
            for p in Au: uf[:,p-1]=rng.integers(0,2,BATCH)
            for p in Av: vf[:,p-1]=rng.integers(0,2,BATCH)
            xf=polar_encode_batch(uf);yf=polar_encode_batch(vf)
            zf=torch.from_numpy(channel.sample_batch(xf,yf)).float()
            emb=model.z_encoder(zf.unsqueeze(-1))[:,br]
            u_t=torch.from_numpy(uf).long();v_t=torch.from_numpy(vf).long()
            loss=model.sequential_forward(emb,br_np,fu_nat,fv_nat,u_t,v_t)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            total_it+=1
            if it%stage['eval_every']==0:
                # Always eval at target N=256
                Au256,Av256,br256_np,br256,fu256,fv256=load_design(256)
                bler=evaluate(model,channel,256,Au256,Av256,br256_np,br256,fu256,fv256,300)
                log.record(total_it,loss.item(),bler)
                if bler < log.best_bler + 0.001:
                    torch.save(model.state_dict(),os.path.join(SAVE_DIR,f'decisive_B_best.pt'))
                # Also eval at current N
                if N != 256:
                    bler_cur=evaluate(model,channel,N,Au,Av,br_np,br,fu_nat,fv_nat,300)
                    sc_cur=SC_REF.get(N,0.05)
                    print(f'  (N={N} BLER={bler_cur:.4f}, {bler_cur/sc_cur:.1f}x SC)', flush=True)
    log.save()
    return log

# ─── Experiment C: Hybrid ─────────────────────────────────────────────────────
def run_C():
    N=256; MAX_HOURS=10
    channel=GaussianMAC(sigma2=SIGMA2)
    Au,Av,br_np,br,fu_nat,fv_nat=load_design(N)
    model=Decoder(d=D)
    log=Logger('C_hybrid')
    print(f'\n{"="*60}', flush=True)
    print(f'Experiment C: Hybrid (fast_ce pretrain → sequential fine-tune) at N={N}', flush=True)
    print(f'Phase 1: fast_ce 30K iters | Phase 2: sequential remaining time', flush=True)
    print(f'Max {MAX_HOURS}h total', flush=True)
    print(f'{"="*60}', flush=True)
    t0=time.time(); rng=np.random.default_rng()

    # Phase 1: fast_ce pretrain
    P1_ITERS=30000; P1_BATCH=16; P1_LR=5e-4
    opt=torch.optim.Adam(model.parameters(),lr=P1_LR)
    print(f'\n--- Phase 1: fast_ce, batch={P1_BATCH}, lr={P1_LR} ---', flush=True)
    model.train()
    for it in range(1, P1_ITERS+1):
        if (time.time()-t0)/3600 > MAX_HOURS*0.4:
            print(f'Phase 1 time limit at iter {it}', flush=True); break
        uf=np.zeros((P1_BATCH,N),dtype=int);vf=np.zeros((P1_BATCH,N),dtype=int)
        for p in Au: uf[:,p-1]=rng.integers(0,2,P1_BATCH)
        for p in Av: vf[:,p-1]=rng.integers(0,2,P1_BATCH)
        xf=polar_encode_batch(uf);yf=polar_encode_batch(vf)
        zf=torch.from_numpy(channel.sample_batch(xf,yf)).float()
        emb=model.z_encoder(zf.unsqueeze(-1))[:,br]
        joint_cw=torch.from_numpy(xf*2+yf).long()[:,br]
        loss=model.fast_ce_leaf_only(emb,joint_cw)
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if it%5000==0:
            bler=evaluate(model,channel,N,Au,Av,br_np,br,fu_nat,fv_nat,300)
            log.record(it,loss.item(),bler)
    torch.save(model.state_dict(),os.path.join(SAVE_DIR,'decisive_C_phase1.pt'))
    print(f'Phase 1 done. Best BLER: {log.best_bler:.4f}', flush=True)

    # Phase 2: sequential fine-tune
    P2_BATCH=4; P2_LR=5e-5
    opt=torch.optim.Adam(model.parameters(),lr=P2_LR)
    print(f'\n--- Phase 2: sequential fine-tune, batch={P2_BATCH}, lr={P2_LR} ---', flush=True)
    total_it=P1_ITERS
    model.train()
    for it in range(1, 500001):
        if (time.time()-t0)/3600 > MAX_HOURS:
            print(f'Phase 2 time limit at iter {it}', flush=True); break
        uf=np.zeros((P2_BATCH,N),dtype=int);vf=np.zeros((P2_BATCH,N),dtype=int)
        for p in Au: uf[:,p-1]=rng.integers(0,2,P2_BATCH)
        for p in Av: vf[:,p-1]=rng.integers(0,2,P2_BATCH)
        xf=polar_encode_batch(uf);yf=polar_encode_batch(vf)
        zf=torch.from_numpy(channel.sample_batch(xf,yf)).float()
        emb=model.z_encoder(zf.unsqueeze(-1))[:,br]
        u_t=torch.from_numpy(uf).long();v_t=torch.from_numpy(vf).long()
        loss=model.sequential_forward(emb,br_np,fu_nat,fv_nat,u_t,v_t)
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        total_it+=1
        if it%2000==0:
            bler=evaluate(model,channel,N,Au,Av,br_np,br,fu_nat,fv_nat,300)
            log.record(total_it,loss.item(),bler)
            if bler < log.best_bler + 0.001:
                torch.save(model.state_dict(),os.path.join(SAVE_DIR,'decisive_C_best.pt'))
    log.save()
    return log

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['A','B','C'])
    args = parser.parse_args()

    print(f'Decisive N=256 comparison — Experiment {args.experiment}', flush=True)
    print(f'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    print(f'SC reference at N=256: BLER=0.005', flush=True)

    if args.experiment == 'A': run_A()
    elif args.experiment == 'B': run_B()
    elif args.experiment == 'C': run_C()

    print(f'\nFinished: {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
