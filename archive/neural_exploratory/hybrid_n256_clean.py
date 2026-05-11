#!/usr/bin/env python3
"""
Clean Hybrid test at N=256: fast_ce z_encoder + fresh tree ops.
NO curriculum, NO checkpoint loading. Pure test of whether
fast_ce z_encoder initialization helps sequential training at N=256.
"""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 256; n = 8; ku = 123; kv = 123
SIGMA2 = 10**(-6/10)
SC_BLER = 0.005
SCRATCH_BEST = 0.015

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG = os.path.join(os.path.dirname(__file__), 'hybrid_n256_clean.log')

def _make_mlp(ind, h, outd, nl=2):
    l = [nn.Linear(ind, h), nn.ELU()]
    for _ in range(nl-1): l += [nn.Linear(h, h), nn.ELU()]
    l.append(nn.Linear(h, outd))
    return nn.Sequential(*l)

def log(msg):
    with open(LOG, 'a') as f: f.write(msg + '\n')
    print(msg, flush=True)

class FastCE(nn.Module):
    def __init__(s, d=16, h=64):
        super().__init__(); s.d = d
        s.z_encoder = nn.Sequential(nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d))
        s.ck = _make_mlp(2*d, h, d); s.bt = _make_mlp(2*d, h, d)
        s.emb2logits = _make_mlp(d, h, 4)
    def checknode(s, eo, ee): return s.ck(torch.cat([eo, ee], -1)) + eo + ee
    def bitnode(s, eo, ee, du, dv):
        B,M,d = eo.shape; g = d//4
        sg = torch.ones(B,M,4,device=eo.device)
        sg[:,:,1] = 1-2*dv.float(); sg[:,:,2] = 1-2*du.float()
        sg[:,:,3] = 1-2*((du^dv).float())
        ep = (eo.reshape(B,M,4,g)*sg.unsqueeze(-1)).reshape(B,M,d)
        return s.bt(torch.cat([ep,ee],-1)) + ep + ee
    def fast_ce(s, emb, xcw, ycw):
        B,N_,d = emb.shape; n_ = int(math.log2(N_)); losses = []
        jcw = xcw*2+ycw
        losses.append(F.cross_entropy(s.emb2logits(emb).reshape(-1,4), jcw.reshape(-1).long()))
        Vj = [jcw]; E = [emb]
        for depth in range(n_):
            jo_l,je_l,eo_l,ee_l = [],[],[],[]
            for vj,e in zip(Vj,E):
                jo_l.append(vj[:,0::2]); je_l.append(vj[:,1::2])
                eo_l.append(e[:,0::2,:]); ee_l.append(e[:,1::2,:])
            jo=torch.cat(jo_l,1); je=torch.cat(je_l,1)
            eo=torch.cat(eo_l,1); ee=torch.cat(ee_l,1)
            uo=jo//2; vo=jo%2; ue=je//2; ve=je%2
            jl=(uo^ue)*2+(vo^ve); jr=je
            nc=2**depth; cs=(N_//2)//nc
            jl_c=torch.split(jl,cs,1); jr_c=torch.split(jr,cs,1)
            Vjn=[i for p in zip(jl_c,jr_c) for i in p]
            Vjleft=torch.cat(Vjn[0::2],1)
            el=s.checknode(eo,ee); er=s.bitnode(eo,ee,Vjleft//2,Vjleft%2)
            losses.append(F.cross_entropy(s.emb2logits(el).reshape(-1,4),jl.reshape(-1).long()))
            losses.append(F.cross_entropy(s.emb2logits(er).reshape(-1,4),jr.reshape(-1).long()))
            el_c=torch.split(el,cs,1); er_c=torch.split(er,cs,1)
            En=[i for p in zip(el_c,er_c) for i in p]; Vj=Vjn; E=En
        return torch.stack(losses).mean()

def main():
    open(LOG, 'w').close()
    channel = GaussianMAC(sigma2=SIGMA2)
    br = bit_reversal_perm(n)
    b = make_path(N, N//2)
    Au, Av, fu, fv, _, _, _ = design_from_file(
        os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr6dB.npz'), n, ku, kv)
    fu_seq = {i: 0 for i in range(1, N+1) if i not in Au}
    fv_seq = {i: 0 for i in range(1, N+1) if i not in Av}

    log("=" * 60)
    log("  CLEAN HYBRID N=256")
    log("  fast_ce z_encoder + FRESH tree ops (no checkpoint)")
    log(f"  N={N}, ku={ku}, kv={kv}, SNR=6dB")
    log(f"  SC={SC_BLER}, scratch baseline={SCRATCH_BEST}")
    log("=" * 60)

    # Phase 1: fast_ce pretrain
    log("\n--- Phase 1: Fast_CE pretrain (10K iters) ---")
    fce = FastCE(d=16, h=64)
    opt = torch.optim.AdamW(fce.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    t0 = time.time()
    for it in range(1, 10001):
        lr = 3e-4*(0.01+0.99*0.5*(1+math.cos(math.pi*it/10000)))
        for pg in opt.param_groups: pg['lr'] = lr
        uf = np.zeros((16,N),dtype=int); vf = np.zeros((16,N),dtype=int)
        for p in Au: uf[:,p-1] = rng.integers(0,2,16)
        for p in Av: vf[:,p-1] = rng.integers(0,2,16)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        emb = fce.z_encoder(torch.from_numpy(zf[:,br]).float().unsqueeze(-1))
        loss = fce.fast_ce(emb, torch.from_numpy(xf[:,br]).long(), torch.from_numpy(yf[:,br]).long())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(fce.parameters(), 1.0); opt.step()
        if it % 5000 == 0:
            log(f"  [{it}/10000] loss={loss.item():.4f} {(time.time()-t0)/60:.1f}min")
    log(f"  Phase 1 done: {(time.time()-t0)/60:.1f}min")

    # Phase 2: Fresh tree walk model + z_encoder transfer ONLY
    log("\n--- Phase 2: Fresh model + z_encoder transfer ---")
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    with torch.no_grad():
        model.z_encoder[0].weight.copy_(fce.z_encoder[0].weight)
        model.z_encoder[0].bias.copy_(fce.z_encoder[0].bias)
        model.z_encoder[2].weight.copy_(fce.z_encoder[2].weight)
        model.z_encoder[2].bias.copy_(fce.z_encoder[2].bias)
    log("  Fresh model created, z_encoder transferred")
    log(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 3: Sequential fine-tune
    log("\n--- Phase 3: Sequential training ---")
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    rng2 = np.random.default_rng(42)
    t1 = time.time()
    best = 1.0
    TOTAL = 50000; BATCH = 4

    model.train()
    for it in range(1, TOTAL+1):
        progress = it / TOTAL
        lr = 1e-4*(0.01+0.99*0.5*(1+math.cos(math.pi*progress)))
        for pg in opt2.param_groups: pg['lr'] = lr
        uf = np.zeros((BATCH,N),dtype=int); vf = np.zeros((BATCH,N),dtype=int)
        for p in Au: uf[:,p-1] = rng2.integers(0,2,BATCH)
        for p in Av: vf[:,p-1] = rng2.integers(0,2,BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf,yf)).float()
        al,at,_,_,_ = model(zf, b, fu_seq, fv_seq,
                             u_true=torch.from_numpy(uf).long(),
                             v_true=torch.from_numpy(vf).long())
        if len(al) > 0:
            loss = F.cross_entropy(torch.cat(al), torch.cat(at))
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt2.step()

        if it % 5000 == 0:
            model.eval()
            errs = 0; total = 200; trng = np.random.default_rng(999)
            for _ in range(total):
                u1 = np.zeros((1,N),dtype=int); v1 = np.zeros((1,N),dtype=int)
                for p in Au: u1[:,p-1] = trng.integers(0,2,1)
                for p in Av: v1[:,p-1] = trng.integers(0,2,1)
                x1 = polar_encode_batch(u1); y1 = polar_encode_batch(v1)
                z1 = torch.from_numpy(channel.sample_batch(x1,y1)).float()
                with torch.no_grad():
                    _,_,uh,vh,_ = model(z1, b, fu_seq, fv_seq)
                ue = any(int(uh[p].item()) != u1[0,p-1] for p in Au)
                ve = any(int(vh[p].item()) != v1[0,p-1] for p in Av)
                if ue or ve: errs += 1
            bler = errs/total
            imp = ''
            if bler < best:
                best = bler
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'hybrid_n256_clean_best.pt'))
                imp = ' *BEST*'
            model.train()
            elapsed = time.time()-t1
            log(f"  [{it:>6}/{TOTAL}] loss={loss.item():.4f} BLER={bler:.4f} "
                f"(best={best:.4f}, SC={SC_BLER}, scratch={SCRATCH_BEST}) "
                f"{elapsed/60:.0f}min lr={lr:.1e}{imp}")

    log(f"\n{'='*60}")
    log(f"  RESULT: best={best:.4f}")
    log(f"  SC={SC_BLER}, scratch={SCRATCH_BEST}")
    log(f"  Verdict: {'HELPS' if best < SCRATCH_BEST else 'NO HELP'}")
    log(f"  Time: {(time.time()-t0)/60:.0f}min")
    log(f"{'='*60}")

if __name__ == '__main__':
    main()
