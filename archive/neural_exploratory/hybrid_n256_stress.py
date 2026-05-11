#!/usr/bin/env python3
"""
30-hour Hybrid stress test at N=256.
Phase 1: Base run (LR=1e-4, batch=4, 100K iters)
Phase 2: If stuck, try higher LR (5e-4) and smaller batch (2)
Phase 3: Best config runs as long as possible
"""
import sys, os, math, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N=256; n=8; ku=123; kv=123
SIGMA2=10**(-6/10); SC_BLER=0.005; SCRATCH_BEST=0.015
SAVE_DIR=os.path.join(os.path.dirname(__file__),'..','saved_models')
DESIGNS_DIR=os.path.join(os.path.dirname(__file__),'..','designs')
LOG=os.path.join(os.path.dirname(__file__),'hybrid_n256_stress.log')
TOTAL_HOURS=30

def _make_mlp(ind,h,outd,nl=2):
    l=[nn.Linear(ind,h),nn.ELU()]
    for _ in range(nl-1):l+=[nn.Linear(h,h),nn.ELU()]
    l.append(nn.Linear(h,outd));return nn.Sequential(*l)

import torch.nn as nn

class FastCE(nn.Module):
    def __init__(s,d=16,h=64):
        super().__init__();s.d=d
        s.z_encoder=nn.Sequential(nn.Linear(1,32),nn.ELU(),nn.Linear(32,d))
        s.ck=_make_mlp(2*d,h,d);s.bt=_make_mlp(2*d,h,d);s.emb2logits=_make_mlp(d,h,4)
    def checknode(s,eo,ee):return s.ck(torch.cat([eo,ee],-1))+eo+ee
    def bitnode(s,eo,ee,du,dv):
        B,M,d=eo.shape;g=d//4
        sg=torch.ones(B,M,4,device=eo.device)
        sg[:,:,1]=1-2*dv.float();sg[:,:,2]=1-2*du.float();sg[:,:,3]=1-2*((du^dv).float())
        ep=(eo.reshape(B,M,4,g)*sg.unsqueeze(-1)).reshape(B,M,d)
        return s.bt(torch.cat([ep,ee],-1))+ep+ee
    def fast_ce(s,emb,xcw,ycw):
        B,N_,d=emb.shape;n_=int(math.log2(N_));losses=[]
        jcw=xcw*2+ycw
        losses.append(F.cross_entropy(s.emb2logits(emb).reshape(-1,4),jcw.reshape(-1).long()))
        Vj=[jcw];E=[emb]
        for depth in range(n_):
            jo_l,je_l,eo_l,ee_l=[],[],[],[]
            for vj,e in zip(Vj,E):
                jo_l.append(vj[:,0::2]);je_l.append(vj[:,1::2])
                eo_l.append(e[:,0::2,:]);ee_l.append(e[:,1::2,:])
            jo=torch.cat(jo_l,1);je=torch.cat(je_l,1)
            eo=torch.cat(eo_l,1);ee=torch.cat(ee_l,1)
            uo=jo//2;vo=jo%2;ue=je//2;ve=je%2
            jl=(uo^ue)*2+(vo^ve);jr=je
            nc=2**depth;cs=(N_//2)//nc
            jl_c=torch.split(jl,cs,1);jr_c=torch.split(jr,cs,1)
            Vjn=[i for p in zip(jl_c,jr_c) for i in p]
            Vjleft=torch.cat(Vjn[0::2],1)
            el=s.checknode(eo,ee);er=s.bitnode(eo,ee,Vjleft//2,Vjleft%2)
            losses.append(F.cross_entropy(s.emb2logits(el).reshape(-1,4),jl.reshape(-1).long()))
            losses.append(F.cross_entropy(s.emb2logits(er).reshape(-1,4),jr.reshape(-1).long()))
            el_c=torch.split(el,cs,1);er_c=torch.split(er,cs,1)
            En=[i for p in zip(el_c,er_c) for i in p];Vj=Vjn;E=En
        return torch.stack(losses).mean()

def log(msg):
    with open(LOG,'a') as f:f.write(msg+'\n')
    print(msg,flush=True)

def pretrain_fastce():
    """Fast_CE pretrain — shared across all variants."""
    fce=FastCE(d=16,h=64)
    opt=torch.optim.AdamW(fce.parameters(),lr=3e-4,weight_decay=1e-5)
    rng=np.random.default_rng(42)
    br=bit_reversal_perm(n)
    channel=GaussianMAC(sigma2=SIGMA2)
    Au,Av,fu,fv,_,_,_=design_from_file(
        os.path.join(DESIGNS_DIR,f'gmac_B_n{n}_snr6dB.npz'),n,ku,kv)
    t0=time.time()
    for it in range(1,10001):
        lr=3e-4*(0.01+0.99*0.5*(1+math.cos(math.pi*it/10000)))
        for pg in opt.param_groups:pg['lr']=lr
        uf=np.zeros((16,N),dtype=int);vf=np.zeros((16,N),dtype=int)
        for p in Au:uf[:,p-1]=rng.integers(0,2,16)
        for p in Av:vf[:,p-1]=rng.integers(0,2,16)
        xf=polar_encode_batch(uf);yf=polar_encode_batch(vf)
        zf=channel.sample_batch(xf,yf)
        emb=fce.z_encoder(torch.from_numpy(zf[:,br]).float().unsqueeze(-1))
        loss=fce.fast_ce(emb,torch.from_numpy(xf[:,br]).long(),torch.from_numpy(yf[:,br]).long())
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(fce.parameters(),1.0);opt.step()
        if it%5000==0:
            log(f"  fast_ce [{it}/10000] loss={loss.item():.4f} {(time.time()-t0)/60:.0f}min")
    log(f"  Fast_CE done: {(time.time()-t0)/60:.0f}min")
    return fce

def make_hybrid_model(fce):
    """Create fresh model with fast_ce z_encoder."""
    model=GmacNeuralCompGraphDecoder(d=16,hidden=64,n_layers=2,z_hidden=32)
    with torch.no_grad():
        model.z_encoder[0].weight.copy_(fce.z_encoder[0].weight)
        model.z_encoder[0].bias.copy_(fce.z_encoder[0].bias)
        model.z_encoder[2].weight.copy_(fce.z_encoder[2].weight)
        model.z_encoder[2].bias.copy_(fce.z_encoder[2].bias)
    return model

def eval_model(model,b,fu_seq,fv_seq,Au,Av,channel,n_cw=200):
    model.eval();errs=0;trng=np.random.default_rng(999)
    for _ in range(n_cw):
        u1=np.zeros((1,N),dtype=int);v1=np.zeros((1,N),dtype=int)
        for p in Au:u1[:,p-1]=trng.integers(0,2,1)
        for p in Av:v1[:,p-1]=trng.integers(0,2,1)
        x1=polar_encode_batch(u1);y1=polar_encode_batch(v1)
        z1=torch.from_numpy(channel.sample_batch(x1,y1)).float()
        with torch.no_grad():_,_,uh,vh,_=model(z1,b,fu_seq,fv_seq)
        ue=any(int(uh[p].item())!=u1[0,p-1] for p in Au)
        ve=any(int(vh[p].item())!=v1[0,p-1] for p in Av)
        if ue or ve:errs+=1
    model.train();return errs/n_cw

def run_variant(model,label,lr,batch,max_iters,time_budget_hrs,
                b,fu_seq,fv_seq,Au,Av,channel):
    """Run a single training variant."""
    log(f"\n  [{label}] LR={lr}, batch={batch}, max_iters={max_iters}")
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    rng=np.random.default_rng(42)
    t0=time.time()
    best=1.0
    milestones={0.99:None,0.95:None,0.90:None,0.80:None,0.50:None,0.30:None,0.10:None}
    
    model.train()
    for it in range(1,max_iters+1):
        if (time.time()-t0)/3600 > time_budget_hrs:
            log(f"  [{label}] Time budget {time_budget_hrs}h reached at iter {it}")
            break
        progress=it/max_iters
        lr_now=lr*(0.01+0.99*0.5*(1+math.cos(math.pi*progress)))
        for pg in opt.param_groups:pg['lr']=lr_now
        
        uf=np.zeros((batch,N),dtype=int);vf=np.zeros((batch,N),dtype=int)
        for p in Au:uf[:,p-1]=rng.integers(0,2,batch)
        for p in Av:vf[:,p-1]=rng.integers(0,2,batch)
        xf=polar_encode_batch(uf);yf=polar_encode_batch(vf)
        zf=torch.from_numpy(channel.sample_batch(xf,yf)).float()
        al,at,_,_,_=model(zf,b,fu_seq,fv_seq,
                          u_true=torch.from_numpy(uf).long(),
                          v_true=torch.from_numpy(vf).long())
        if len(al)>0:
            loss=F.cross_entropy(torch.cat(al),torch.cat(at))
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        
        if it%2000==0:
            bler=eval_model(model,b,fu_seq,fv_seq,Au,Av,channel)
            imp=''
            if bler<best:
                best=bler
                torch.save(model.state_dict(),
                          os.path.join(SAVE_DIR,f'hybrid_stress_{label}_best.pt'))
                imp=' *BEST*'
            # Check milestones
            for m in sorted(milestones.keys(),reverse=True):
                if milestones[m] is None and bler<m:
                    milestones[m]=it
                    log(f"  [{label}] *** MILESTONE: BLER < {m} at iter {it} ({(time.time()-t0)/60:.0f}min) ***")
            elapsed=(time.time()-t0)/60
            log(f"  [{label}] [{it:>6}] loss={loss.item():.4f} BLER={bler:.4f} "
                f"best={best:.4f} {elapsed:.0f}min lr={lr_now:.1e}{imp}")
    
    elapsed_total=(time.time()-t0)/60
    log(f"  [{label}] DONE: best={best:.4f} time={elapsed_total:.0f}min")
    log(f"  [{label}] Milestones: {dict((k,v) for k,v in milestones.items() if v is not None)}")
    return best, milestones

def main():
    open(LOG,'w').close()
    t_global=time.time()
    
    channel=GaussianMAC(sigma2=SIGMA2)
    b=make_path(N,N//2)
    Au,Av,fu,fv,_,_,_=design_from_file(
        os.path.join(DESIGNS_DIR,f'gmac_B_n{n}_snr6dB.npz'),n,ku,kv)
    fu_seq={i:0 for i in range(1,N+1) if i not in Au}
    fv_seq={i:0 for i in range(1,N+1) if i not in Av}
    
    log("="*60)
    log("  30-HOUR HYBRID STRESS TEST AT N=256")
    log(f"  SC={SC_BLER}, scratch baseline={SCRATCH_BEST}")
    log("="*60)
    
    # ── Fast_CE pretrain (shared) ────────────────────────────
    log("\n=== FAST_CE PRETRAIN ===")
    fce=pretrain_fastce()
    
    # ── PHASE 1: Base run (LR=1e-4, batch=4) ────────────────
    log("\n=== PHASE 1: Base run (LR=1e-4, batch=4, 100K iters, 8h budget) ===")
    m1=make_hybrid_model(fce)
    b1,ms1=run_variant(m1,"base",lr=1e-4,batch=4,max_iters=100000,
                       time_budget_hrs=8,b=b,fu_seq=fu_seq,fv_seq=fv_seq,
                       Au=Au,Av=Av,channel=channel)
    
    # Decide if Phase 2 needed
    stuck = b1 >= 0.99
    
    # ── PHASE 2: LR sweep (if stuck) ────────────────────────
    if stuck:
        log("\n=== PHASE 2: Base stuck at BLER≈1.0, trying variants ===")
        
        log("\n--- Variant A: Higher LR (5e-4, batch=4) ---")
        m2a=make_hybrid_model(fce)
        b2a,ms2a=run_variant(m2a,"highLR",lr=5e-4,batch=4,max_iters=50000,
                            time_budget_hrs=4,b=b,fu_seq=fu_seq,fv_seq=fv_seq,
                            Au=Au,Av=Av,channel=channel)
        
        log("\n--- Variant B: Smaller batch (LR=1e-4, batch=2) ---")
        m2b=make_hybrid_model(fce)
        b2b,ms2b=run_variant(m2b,"smallBatch",lr=1e-4,batch=2,max_iters=50000,
                            time_budget_hrs=4,b=b,fu_seq=fu_seq,fv_seq=fv_seq,
                            Au=Au,Av=Av,channel=channel)
        
        log("\n--- Variant C: High LR + small batch (5e-4, batch=2) ---")
        m2c=make_hybrid_model(fce)
        b2c,ms2c=run_variant(m2c,"highLR_smallB",lr=5e-4,batch=2,max_iters=50000,
                            time_budget_hrs=4,b=b,fu_seq=fu_seq,fv_seq=fv_seq,
                            Au=Au,Av=Av,channel=channel)
        
        # Pick best
        variants={"base":b1,"highLR":b2a,"smallBatch":b2b,"highLR_smallB":b2c}
        best_var=min(variants,key=variants.get)
        log(f"\n  Best variant: {best_var} (BLER={variants[best_var]:.4f})")
    else:
        log("\n=== PHASE 2: Skipped (base run showed learning) ===")
        best_var="base"
    
    # ── PHASE 3: Long run with best config ───────────────────
    remaining_hrs = TOTAL_HOURS - (time.time()-t_global)/3600
    if remaining_hrs > 1:
        log(f"\n=== PHASE 3: Long run ({remaining_hrs:.0f}h remaining) ===")
        
        if best_var=="base":
            lr3=1e-4; batch3=4
        elif best_var=="highLR":
            lr3=5e-4; batch3=4
        elif best_var=="smallBatch":
            lr3=1e-4; batch3=2
        else:
            lr3=5e-4; batch3=2
        
        m3=make_hybrid_model(fce)
        # Load best checkpoint if exists
        ckpt=os.path.join(SAVE_DIR,f'hybrid_stress_{best_var}_best.pt')
        if os.path.exists(ckpt):
            m3.load_state_dict(torch.load(ckpt,weights_only=False))
            log(f"  Loaded best checkpoint from {best_var}")
        
        b3,ms3=run_variant(m3,"long",lr=lr3,batch=batch3,max_iters=500000,
                          time_budget_hrs=remaining_hrs,b=b,fu_seq=fu_seq,
                          fv_seq=fv_seq,Au=Au,Av=Av,channel=channel)
    
    # ── FINAL REPORT ─────────────────────────────────────────
    total_time=(time.time()-t_global)/3600
    log(f"\n{'='*60}")
    log(f"  FINAL REPORT — {total_time:.1f} hours total")
    log(f"{'='*60}")
    log(f"  SC baseline:      {SC_BLER}")
    log(f"  Scratch baseline:  {SCRATCH_BEST}")
    log(f"  Hybrid best:       {min(b1, b3 if 'b3' in dir() else 1.0):.4f}")
    
    final_best = min(b1, b3 if 'b3' in dir() else 1.0)
    if final_best < 0.10:
        verdict = "STRONG convergence — Hybrid is VIABLE at N=256"
    elif final_best < 0.50:
        verdict = "Moderate convergence — Hybrid shows PROMISE at N=256"
    elif final_best < 0.90:
        verdict = "Weak convergence — Hybrid has LIMITED value at N=256"
    elif final_best < 0.99:
        verdict = "Minimal learning — Hybrid barely works at N=256"
    else:
        verdict = "NO convergence — Hybrid FAILS at N=256"
    
    log(f"\n  VERDICT: {verdict}")
    log(f"{'='*60}")

if __name__=='__main__':
    main()
