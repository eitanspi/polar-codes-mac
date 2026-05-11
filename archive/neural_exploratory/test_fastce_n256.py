#!/usr/bin/env python3
"""
Test whether fast_ce parallel training scales to N=256 for neural MAC polar decoder.

Key question: The sequential CG decoder fails to train at N=256 due to O(N log N)
gradient depth. fast_ce has O(log N) gradient depth. Does it produce a working decoder?

SC reference at N=256: BLER=0.005
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

# ─── Hyperparameters ───
D = 16; HIDDEN = 64; N_LAYERS = 2
SNR_DB = 6.0; SIGMA2 = 10**(-SNR_DB/10)
N = 256; n = 8
ku = 123; kv = 123
BATCH = 32
LR = 3e-4
TOTAL_ITERS = 100000
EVAL_EVERY = 5000
EVAL_CW = 500
SC_BLER_256 = 0.005

SAVE_DIR = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/saved_models'
SAVE_PATH = os.path.join(SAVE_DIR, 'fastce_n256_best.pt')
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Model ───
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
            E_odd = torch.cat(E_odds, 1); E_even = torch.cat(E_evens, 1)
            J_odd = torch.cat(J_odds, 1); J_even = torch.cat(J_evens, 1)
            u_o = J_odd//2; v_o = J_odd%2; u_e = J_even//2; v_e = J_even%2
            J_left = (u_o^u_e)*2 + (v_o^v_e); J_right = J_even
            e_left = self.checknode(torch.cat([E_odd, E_even], -1))
            e_right = self.bitnode(E_odd, E_even, J_left)
            nc = 2**depth; cs = (N_//2)//nc
            el = torch.split(e_left, cs, 1); er = torch.split(e_right, cs, 1)
            jl = torch.split(J_left, cs, 1); jr = torch.split(J_right, cs, 1)
            E_chunks = []; J_chunks = []
            for a, b, c, dd in zip(el, er, jl, jr):
                E_chunks += [a, b]; J_chunks += [c, dd]
        e_all = torch.cat(E_chunks, 1); j_all = torch.cat(J_chunks, 1)
        logits = self.emb2logits(e_all)
        return F.cross_entropy(logits.reshape(-1, 4), j_all.reshape(-1))

    def sc_decode_correct(self, emb, fu_nat, fv_nat, br_np):
        B = emb.shape[0]; N_ = emb.shape[1]
        br_t = torch.from_numpy(br_np).long()
        u_hat = torch.zeros(B, N_, dtype=torch.long)
        v_hat = torch.zeros(B, N_, dtype=torch.long)
        leaf_idx = [0]
        def _decode(eb):
            bs = eb.shape[1]
            if bs == 1:
                logits = self.emb2logits(eb[:,0,:])
                idx = leaf_idx[0]; leaf_idx[0] += 1
                nat_idx = int(br_t[idx])
                uf = nat_idx in fu_nat; vf = nat_idx in fv_nat
                if uf and vf: dec = torch.zeros(B, dtype=torch.long)
                elif uf: dec = (logits[:,1] > logits[:,0]).long()
                elif vf: dec = (logits[:,2] > logits[:,0]).long() * 2
                else: dec = logits.argmax(dim=-1)
                u_hat[:, nat_idx] = dec // 2; v_hat[:, nat_idx] = dec % 2
                return dec.unsqueeze(1)
            half = bs // 2
            e_odd = eb[:,0::2,:]; e_even = eb[:,1::2,:]
            e_left = self.checknode(torch.cat([e_odd, e_even], -1))
            cw_left = _decode(e_left)
            e_right = self.bitnode(e_odd, e_even, cw_left)
            cw_right = _decode(e_right)
            u_l = cw_left//2; v_l = cw_left%2; u_r = cw_right//2; v_r = cw_right%2
            cw_odd = (u_l^u_r)*2 + (v_l^v_r); cw_even = cw_right
            result = torch.zeros(B, bs, dtype=torch.long)
            result[:,0::2] = cw_odd; result[:,1::2] = cw_even
            return result
        with torch.no_grad(): _decode(emb)
        return u_hat, v_hat


# ─── Setup ───
print(f"N=256 fast_ce scaling test", flush=True)
print(f"d={D}, hidden={HIDDEN}, batch={BATCH}, lr={LR}", flush=True)
print(f"SC reference BLER={SC_BLER_256}", flush=True)
print(flush=True)

channel = GaussianMAC(sigma2=SIGMA2)
mc_path = f'/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs/gmac_B_n{n}_snr{SNR_DB:.0f}dB.npz'
if not os.path.exists(mc_path):
    print(f"ERROR: Design file not found: {mc_path}")
    print("Using analytical fallback...")
    from polar.design import design_gmac
    Au, Av, fu, fv = design_gmac(n, ku, kv, SNR_DB)
else:
    Au, Av, fu, fv, _, _, _ = design_from_file(mc_path, n, ku, kv)
    print(f"Loaded MC design from {mc_path}", flush=True)

br_np = bit_reversal_perm(n)
br = torch.from_numpy(br_np).long()
fu_nat = {p-1 for p in range(1, N+1) if p not in Au}
fv_nat = {p-1 for p in range(1, N+1) if p not in Av}
print(f"|Au|={len(Au)}, |Av|={len(Av)}, |fu_nat|={len(fu_nat)}, |fv_nat|={len(fv_nat)}", flush=True)

sys.setrecursionlimit(2000)  # N=256 needs deep recursion for SC decode

model = Decoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS)
nparams = sum(p.numel() for p in model.parameters())
print(f"Model params: {nparams:,}", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def generate_batch(batch_size):
    """Generate a batch of MAC codewords and channel outputs."""
    u_msg = np.zeros((batch_size, N), dtype=np.int32)
    v_msg = np.zeros((batch_size, N), dtype=np.int32)
    for pos in Au:
        u_msg[:, pos-1] = np.random.randint(0, 2, batch_size)
    for pos in Av:
        v_msg[:, pos-1] = np.random.randint(0, 2, batch_size)
    x = polar_encode_batch(u_msg)
    y = polar_encode_batch(v_msg)
    # Channel: z = (1-2x) + (1-2y) + w
    sx = 1 - 2*x.astype(np.float64)
    sy = 1 - 2*y.astype(np.float64)
    w = np.random.randn(batch_size, N) * np.sqrt(SIGMA2)
    z = sx + sy + w
    # Joint codeword labels: u*2 + v (after bit-reversal)
    joint_cw = x * 2 + y  # in encoded domain
    joint_cw_br = joint_cw[:, br_np]  # bit-reverse for tree ordering
    z_br = z[:, br_np]
    return (torch.tensor(z_br, dtype=torch.float32),
            torch.tensor(joint_cw_br, dtype=torch.long),
            torch.tensor(u_msg, dtype=torch.long),
            torch.tensor(v_msg, dtype=torch.long),
            torch.tensor(z, dtype=torch.float32))


def evaluate(model, n_cw=EVAL_CW):
    """Evaluate BLER using sequential SC decode."""
    model.eval()
    n_block_err = 0
    eval_batch = 50  # process in small batches
    n_done = 0
    with torch.no_grad():
        while n_done < n_cw:
            bs = min(eval_batch, n_cw - n_done)
            z_br, joint_cw_br, u_true, v_true, z_raw = generate_batch(bs)
            emb = model.z_encoder(z_br.unsqueeze(-1))
            u_hat, v_hat = model.sc_decode_correct(emb, fu_nat, fv_nat, br_np)
            # Block error: either user wrong
            u_err = (u_hat != u_true).any(dim=1)
            v_err = (v_hat != v_true).any(dim=1)
            n_block_err += (u_err | v_err).sum().item()
            n_done += bs
    model.train()
    return n_block_err / n_done


# ─── Training ───
best_bler = 1.0
best_iter = 0
t0 = time.time()
loss_acc = 0.0
loss_count = 0

print(f"\nStarting training for {TOTAL_ITERS} iterations...", flush=True)
print()

for it in range(1, TOTAL_ITERS + 1):
    z_br, joint_cw_br, _, _, _ = generate_batch(BATCH)
    emb = model.z_encoder(z_br.unsqueeze(-1))
    loss = model.fast_ce_leaf_only(emb, joint_cw_br)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    loss_acc += loss.item()
    loss_count += 1

    if it % 1000 == 0 and it % EVAL_EVERY != 0:
        elapsed = (time.time() - t0) / 60
        print(f"  iter {it//1000}K loss={loss.item():.4f} {elapsed:.1f}min", flush=True)

    if it % EVAL_EVERY == 0:
        avg_loss = loss_acc / loss_count
        loss_acc = 0.0; loss_count = 0
        elapsed = (time.time() - t0) / 60

        bler = evaluate(model)
        ratio = bler / SC_BLER_256 if SC_BLER_256 > 0 else float('inf')

        if bler < best_bler:
            best_bler = bler
            best_iter = it
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"[{it//1000}K] loss={avg_loss:.4f} BLER={bler:.4f} ({ratio:.1f}x SC) {elapsed:.1f}min", flush=True)

        # Adaptive: if stuck at BLER=1.0 after 20K, adjust
        if it == 20000 and best_bler >= 0.99:
            print("  -> BLER stuck near 1.0, reducing batch to 16, LR to 5e-4")
            BATCH = 16
            for pg in optimizer.param_groups:
                pg['lr'] = 5e-4

elapsed_total = (time.time() - t0) / 60
print()
print("=" * 50)
print("=== CONCLUSION ===")
print(f"Best BLER: {best_bler:.4f} ({best_bler/SC_BLER_256:.1f}x SC) at iter {best_iter//1000}K")
print(f"Training time: {elapsed_total:.0f} min")
if best_bler < 0.5:
    print("Does fast_ce scale to N=256? YES")
elif best_bler < 0.9:
    print("Does fast_ce scale to N=256? PARTIALLY (trains but high BLER)")
else:
    print("Does fast_ce scale to N=256? NO (failed to learn)")
print("=" * 50)
