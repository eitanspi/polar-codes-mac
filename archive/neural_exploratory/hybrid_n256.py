#!/usr/bin/env python3
"""
Hybrid fast_ce pretrain → sequential fine-tune at N=256.
The key test: does fast_ce z_encoder initialization help at the N where scratch struggles?

Scratch baseline (d=16): BLER=0.015 (3x SC) after 100K iters
SC baseline: BLER=0.005

Plan:
1. fast_ce pretrain at N=256 (10K iters, ~5 min)
2. Transfer z_encoder to tree walk decoder  
3. Sequential fine-tune with curriculum init from N=128 checkpoint if available,
   otherwise from scratch with z_encoder transfer
4. Log BLER every 5K iters, save best checkpoint
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
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
SC_BLER = 0.005
SCRATCH_BEST = 0.015  # d=16 baseline from previous training

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'hybrid_n256.log')

def _make_mlp(ind, h, outd, nl=2):
    l = [nn.Linear(ind, h), nn.ELU()]
    for _ in range(nl-1): l += [nn.Linear(h, h), nn.ELU()]
    l.append(nn.Linear(h, outd))
    return nn.Sequential(*l)

def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')
    print(msg, flush=True)

# ── Fast_CE model ────────────────────────────────────────────
class FastCE(nn.Module):
    def __init__(self, d=16, h=64):
        super().__init__()
        self.d = d
        self.z_encoder = nn.Sequential(nn.Linear(1, 32), nn.ELU(), nn.Linear(32, d))
        self.ck = _make_mlp(2*d, h, d)
        self.bt = _make_mlp(2*d, h, d)
        self.emb2logits = _make_mlp(d, h, 4)

    def checknode(self, eo, ee):
        return self.ck(torch.cat([eo, ee], -1)) + eo + ee

    def bitnode(self, eo, ee, du, dv):
        B, M, d = eo.shape; g = d // 4
        sg = torch.ones(B, M, 4, device=eo.device)
        sg[:,:,1] = 1 - 2*dv.float()
        sg[:,:,2] = 1 - 2*du.float()
        sg[:,:,3] = 1 - 2*((du ^ dv).float())
        ep = (eo.reshape(B, M, 4, g) * sg.unsqueeze(-1)).reshape(B, M, d)
        return self.bt(torch.cat([ep, ee], -1)) + ep + ee

    def fast_ce(self, emb, xcw, ycw):
        B, N, d = emb.shape; n_ = int(math.log2(N)); losses = []
        jcw = xcw * 2 + ycw
        losses.append(F.cross_entropy(self.emb2logits(emb).reshape(-1, 4), jcw.reshape(-1).long()))
        Vj = [jcw]; E = [emb]
        for depth in range(n_):
            jo_l, je_l, eo_l, ee_l = [], [], [], []
            for vj, e in zip(Vj, E):
                jo_l.append(vj[:,0::2]); je_l.append(vj[:,1::2])
                eo_l.append(e[:,0::2,:]); ee_l.append(e[:,1::2,:])
            jo = torch.cat(jo_l, 1); je = torch.cat(je_l, 1)
            eo = torch.cat(eo_l, 1); ee = torch.cat(ee_l, 1)
            uo = jo//2; vo = jo%2; ue = je//2; ve = je%2
            jl = (uo ^ ue)*2 + (vo ^ ve); jr = je
            nc = 2**depth; cs = (N//2) // nc
            jl_c = torch.split(jl, cs, 1); jr_c = torch.split(jr, cs, 1)
            Vjn = [i for p in zip(jl_c, jr_c) for i in p]
            Vjleft = torch.cat(Vjn[0::2], 1)
            el = self.checknode(eo, ee)
            er = self.bitnode(eo, ee, Vjleft//2, Vjleft%2)
            losses.append(F.cross_entropy(self.emb2logits(el).reshape(-1,4), jl.reshape(-1).long()))
            losses.append(F.cross_entropy(self.emb2logits(er).reshape(-1,4), jr.reshape(-1).long()))
            el_c = torch.split(el, cs, 1); er_c = torch.split(er, cs, 1)
            En = [i for p in zip(el_c, er_c) for i in p]
            Vj = Vjn; E = En
        return torch.stack(losses).mean()


def main():
    # Clear log
    open(LOG_FILE, 'w').close()

    channel = GaussianMAC(sigma2=SIGMA2)
    br = bit_reversal_perm(n)
    b = make_path(N, N//2)

    mc_path = os.path.join(DESIGNS_DIR, f'gmac_B_n{n}_snr{int(SNR_DB)}dB.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
    fu_seq = {i: 0 for i in range(1, N+1) if i not in Au}
    fv_seq = {i: 0 for i in range(1, N+1) if i not in Av}

    log("=" * 60)
    log("  HYBRID N=256: fast_ce pretrain → sequential fine-tune")
    log(f"  N={N}, ku={ku}, kv={kv}, SNR={SNR_DB}dB")
    log(f"  SC BLER={SC_BLER}, Scratch baseline={SCRATCH_BEST}")
    log("=" * 60)

    # ── Phase 1: Fast_CE pretrain ────────────────────────────
    log("\n--- Phase 1: Fast_CE pretrain (10K iters) ---")
    fce = FastCE(d=16, h=64)
    opt_fce = torch.optim.AdamW(fce.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = np.random.default_rng(42)
    t0 = time.time()

    for it in range(1, 10001):
        lr = 3e-4 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * it / 10000)))
        for pg in opt_fce.param_groups: pg['lr'] = lr
        uf = np.zeros((16, N), dtype=int); vf = np.zeros((16, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, 16)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, 16)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        emb = fce.z_encoder(torch.from_numpy(zf[:, br]).float().unsqueeze(-1))
        loss = fce.fast_ce(emb, torch.from_numpy(xf[:, br]).long(), torch.from_numpy(yf[:, br]).long())
        opt_fce.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(fce.parameters(), 1.0); opt_fce.step()
        if it % 5000 == 0:
            log(f"  fast_ce [{it}/10000] loss={loss.item():.4f} {(time.time()-t0)/60:.1f}min")

    fce_time = time.time() - t0
    log(f"  Phase 1 done: {fce_time/60:.1f}min")

    # ── Phase 2: Transfer z_encoder ──────────────────────────
    log("\n--- Phase 2: Transfer z_encoder to tree walk decoder ---")
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)

    # Try to load N=128 checkpoint for curriculum init
    ckpt_128 = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N128.pt')
    if os.path.exists(ckpt_128):
        try:
            sd = torch.load(ckpt_128, map_location='cpu', weights_only=False)
            model.load_state_dict(sd, strict=False)
            log(f"  Loaded N=128 checkpoint: {ckpt_128}")
        except Exception as e:
            log(f"  Failed to load N=128 checkpoint: {e}")

    # Overwrite z_encoder with fast_ce pretrained weights
    with torch.no_grad():
        model.z_encoder[0].weight.copy_(fce.z_encoder[0].weight)
        model.z_encoder[0].bias.copy_(fce.z_encoder[0].bias)
        model.z_encoder[2].weight.copy_(fce.z_encoder[2].weight)
        model.z_encoder[2].bias.copy_(fce.z_encoder[2].bias)
    log("  z_encoder weights transferred from fast_ce")

    # ── Phase 3: Sequential fine-tune ────────────────────────
    log("\n--- Phase 3: Sequential fine-tune ---")
    log(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    rng2 = np.random.default_rng(42)
    t1 = time.time()
    best_bler = 1.0
    TOTAL_ITERS = 50000
    BATCH = 4

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        progress = it / TOTAL_ITERS
        lr = 5e-5 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
        for pg in opt.param_groups: pg['lr'] = lr

        uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng2.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng2.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

        al, at, _, _, _ = model(
            zf, b, fu_seq, fv_seq,
            u_true=torch.from_numpy(uf).long(),
            v_true=torch.from_numpy(vf).long())

        if len(al) > 0:
            loss = F.cross_entropy(torch.cat(al), torch.cat(at))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if it % 5000 == 0:
            model.eval()
            errs = 0; total = 200
            trng = np.random.default_rng(999)
            for _ in range(total):
                u1 = np.zeros((1, N), dtype=int); v1 = np.zeros((1, N), dtype=int)
                for p in Au: u1[:, p-1] = trng.integers(0, 2, 1)
                for p in Av: v1[:, p-1] = trng.integers(0, 2, 1)
                x1 = polar_encode_batch(u1); y1 = polar_encode_batch(v1)
                z1 = torch.from_numpy(channel.sample_batch(x1, y1)).float()
                with torch.no_grad():
                    _, _, uh, vh, _ = model(z1, b, fu_seq, fv_seq)
                ue = any(int(uh[p].item()) != u1[0, p-1] for p in Au)
                ve = any(int(vh[p].item()) != v1[0, p-1] for p in Av)
                if ue or ve: errs += 1
            bler = errs / total
            improved = ''
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'hybrid_n256_best.pt'))
                improved = ' *BEST*'
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'hybrid_n256_latest.pt'))
            model.train()
            elapsed = time.time() - t1
            total_elapsed = time.time() - t0
            log(f"  [{it:>6}/{TOTAL_ITERS}] loss={loss.item():.4f} BLER={bler:.4f} "
                f"(best={best_bler:.4f}, SC={SC_BLER}, scratch={SCRATCH_BEST}) "
                f"{elapsed/60:.0f}min total={total_elapsed/60:.0f}min lr={lr:.1e}{improved}")

    log(f"\n{'='*60}")
    log(f"  DONE: best BLER={best_bler:.4f}")
    log(f"  SC={SC_BLER}, Scratch baseline={SCRATCH_BEST}")
    log(f"  Hybrid vs scratch: {'BETTER' if best_bler < SCRATCH_BEST * 0.9 else 'SIMILAR' if best_bler < SCRATCH_BEST * 1.1 else 'WORSE'}")
    log(f"  Total time: {(time.time()-t0)/60:.0f}min")
    log(f"{'='*60}")


if __name__ == '__main__':
    main()
