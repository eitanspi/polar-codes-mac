"""
Focused N=256 bottleneck experiments — streamlined version.
Each experiment answers one specific question.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, torch.nn.functional as F, numpy as np
from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder
import math

LOG_FILE = os.path.join(os.path.dirname(__file__), 'bottleneck_analysis.log')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

N = 256; n = 8; ku = 123; kv = 123
sigma2 = 10**(-6/10)
channel = GaussianMAC(sigma2=sigma2)
b = make_path(N, N//2)
Au, Av, _, _, _, _, _ = design_from_file('designs/gmac_B_n8_snr6dB.npz', n, ku, kv)
fu_seq = {i:0 for i in range(1, N+1) if i not in Au}
fv_seq = {i:0 for i in range(1, N+1) if i not in Av}

def gen(B):
    u = torch.zeros(B, N); v = torch.zeros(B, N)
    for a in Au: u[:, a-1] = torch.randint(0, 2, (B,)).float()
    for a in Av: v[:, a-1] = torch.randint(0, 2, (B,)).float()
    cu = polar_encode_batch(u.numpy().astype(int))
    cv = polar_encode_batch(v.numpy().astype(int))
    x = 1 - 2*torch.from_numpy(cu).float()
    y = 1 - 2*torch.from_numpy(cv).float()
    z = x + y + torch.randn(B, N) * np.sqrt(sigma2)
    return u, v, z

def load_ckpt(model, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    mapped = {}
    for k, v_ in sd.items():
        if k.startswith('tree.embedding_z'): continue
        mapped[k[5:] if k.startswith('tree.') else k] = v_
    model_sd = model.state_dict()
    loaded = {k: v_ for k, v_ in mapped.items() if k in model_sd and model_sd[k].shape == v_.shape}
    model_sd.update(loaded)
    model.load_state_dict(model_sd)
    return list(loaded.keys())

def bler(model, u, v, z):
    model.eval()
    with torch.no_grad():
        _, _, uh, vh, _ = model(z, b, fu_seq, fv_seq)
    ud = torch.zeros_like(u); vd = torch.zeros_like(v)
    for i, bit in uh.items(): ud[:, i-1] = bit
    for i, bit in vh.items(): vd[:, i-1] = bit
    return ((ud != u) | (vd != v)).any(dim=1).float().mean().item()

def train_one(model, opt, u, v, z):
    model.train()
    al, at, _, _, _ = model(z, b, fu_seq, fv_seq, u_true=u, v_true=v)
    if not al: return 0.0
    loss = F.cross_entropy(torch.cat(al), torch.cat(at))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


# ================================================================
# EXP A: Overfit 1 fixed codeword (pure memorization capacity test)
# Q: Can the architecture represent a correct decode at N=256?
# ================================================================
def exp_a_overfit_one():
    log("=== EXP A: Overfit 1 fixed codeword (fixed noise) ===")
    u1, v1, z1 = gen(1)
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for it in range(2000):
        loss = train_one(model, opt, u1, v1, z1)
        if (it+1) % 500 == 0:
            bl = bler(model, u1, v1, z1)
            log(f"  A iter {it+1}: loss={loss:.6f}, BLER={bl:.4f}")
    return {'final_loss': loss, 'final_bler': bler(model, u1, v1, z1)}


# ================================================================
# EXP B: Teacher forcing gap (from scratch)
# Q: Does loss improve but BLER stay 1.0?
# ================================================================
def exp_b_tf_gap():
    log("=== EXP B: Teacher forcing gap (from scratch, 1500 iters) ===")
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B = 32
    for it in range(1500):
        u, v, z = gen(B)
        loss = train_one(model, opt, u, v, z)
        if (it+1) % 300 == 0:
            ue, ve, ze = gen(64)
            bl = bler(model, ue, ve, ze)
            # Also compute TF loss
            model.eval()
            with torch.no_grad():
                al, at, _, _, _ = model(ze, b, fu_seq, fv_seq, u_true=ue, v_true=ve)
                tf_loss = F.cross_entropy(torch.cat(al), torch.cat(at)).item() if al else 0
            log(f"  B iter {it+1}: train_loss={loss:.4f}, TF_eval_loss={tf_loss:.4f}, BLER={bl:.4f}")
    torch.save(model.state_dict(), 'saved_models/scratch_n256_1500.pt')
    return model


# ================================================================
# EXP C: Component isolation — freeze z_enc, train tree from scratch
# Q: Is the bottleneck z_encoder or tree ops?
# ================================================================
def exp_c_freeze_zenc():
    log("=== EXP C: Frozen z_encoder (N=128), fresh tree ops, 1500 iters ===")
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    load_ckpt(model, 'saved_models/ncg_gmac_mlp_N128.pt')
    # Freeze z_encoder
    for name, p in model.named_parameters():
        if 'z_encoder' in name:
            p.requires_grad = False
    # Re-init tree ops
    for name, p in model.named_parameters():
        if 'z_encoder' not in name and 'no_info' not in name:
            if p.dim() >= 2: torch.nn.init.xavier_uniform_(p)
            else: torch.nn.init.zeros_(p)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    B = 32
    for it in range(1500):
        u, v, z = gen(B)
        loss = train_one(model, opt, u, v, z)
        if (it+1) % 300 == 0:
            ue, ve, ze = gen(64)
            bl = bler(model, ue, ve, ze)
            log(f"  C iter {it+1}: loss={loss:.4f}, BLER={bl:.4f}")


# ================================================================
# EXP D: Component isolation — freeze tree, train z_enc from scratch
# ================================================================
def exp_d_freeze_tree():
    log("=== EXP D: Frozen tree ops (N=128), fresh z_encoder, 1500 iters ===")
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    load_ckpt(model, 'saved_models/ncg_gmac_mlp_N128.pt')
    # Freeze tree ops
    for name, p in model.named_parameters():
        if 'z_encoder' not in name:
            p.requires_grad = False
    # Re-init z_encoder
    for name, p in model.named_parameters():
        if 'z_encoder' in name:
            if p.dim() >= 2: torch.nn.init.xavier_uniform_(p)
            else: torch.nn.init.zeros_(p)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    B = 32
    for it in range(1500):
        u, v, z = gen(B)
        loss = train_one(model, opt, u, v, z)
        if (it+1) % 300 == 0:
            ue, ve, ze = gen(64)
            bl = bler(model, ue, ve, ze)
            log(f"  D iter {it+1}: loss={loss:.4f}, BLER={bl:.4f}")


# ================================================================
# EXP E: Bottleneck localization — per-leaf accuracy (curriculum model)
# ================================================================
def exp_e_bottleneck(ckpt_path, label):
    log(f"=== EXP E: Per-leaf accuracy for {label} ===")
    from polar.encoder import bit_reversal_perm

    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    if ckpt_path:
        load_ckpt(model, ckpt_path)
    model.eval()

    B = 128
    u, v, z = gen(B)
    d = model.d
    br = torch.from_numpy(bit_reversal_perm(int(math.log2(N)))).long()
    root = model.z_encoder(z.unsqueeze(-1))[:, br]

    edge_data = [None] * (2 * N)
    edge_data[1] = root
    no_info = model.no_info_emb.unsqueeze(0).unsqueeze(0)
    for beta in range(2, 2 * N):
        level = beta.bit_length() - 1
        size = N >> level
        edge_data[beta] = no_info.expand(B, size, d).clone()

    dec_head = 1
    u_hat, v_hat = {}, {}
    i_u, i_v = 0, 0
    accs, confs = [], []

    with torch.no_grad():
        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = fu_seq
            else:
                i_v += 1; i_t = i_v; fdict = fv_seq

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1
            dec_head = model._step_to(dec_head, target_vtx, edge_data)
            temp = edge_data[leaf_edge][:, 0].clone()
            if leaf_edge & 1 == 0:
                model._neural_calc_left(target_vtx, edge_data)
            else:
                model._neural_calc_right(target_vtx, edge_data)
            top_down = edge_data[leaf_edge][:, 0]
            combined = top_down + temp
            logits = model.emb2logits(combined)

            if i_t in fdict:
                bit = torch.full((B,), fdict[i_t], dtype=torch.float32)
            else:
                if gamma == 0:
                    p0 = torch.logsumexp(logits[:, :2], dim=1)
                    p1 = torch.logsumexp(logits[:, 2:], dim=1)
                else:
                    p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                    p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                bit = (p1 > p0).float()
                true_bit = u[:, i_t-1] if gamma == 0 else v[:, i_t-1]
                acc = (bit == true_bit).float().mean().item()
                conf = torch.abs(p1 - p0).mean().item()
                accs.append(acc)
                confs.append(conf)

            if gamma == 0: u_hat[i_t] = bit
            else: v_hat[i_t] = bit
            new_emb = model._make_leaf_emb(u_hat.get(i_t), v_hat.get(i_t), B, 'cpu')
            edge_data[leaf_edge] = new_emb.unsqueeze(1)

    n_info = len(accs)
    q = n_info // 4
    for i, name in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        s, e = i*q, (i+1)*q if i < 3 else n_info
        log(f"  {label} {name}: acc={np.mean(accs[s:e]):.3f}, conf={np.mean(confs[s:e]):.3f}")

    # Find first sustained error (3 consecutive below 0.8)
    first_bad = None
    for i in range(len(accs) - 2):
        if accs[i] < 0.8 and accs[i+1] < 0.8 and accs[i+2] < 0.8:
            first_bad = i; break
    log(f"  {label} first_sustained_error at info_bit #{first_bad} / {n_info}")

    bl = bler(model, u, v, z)
    log(f"  {label} BLER={bl:.4f}")
    return accs, confs


# ================================================================
# EXP F: Oracle injection
# ================================================================
def exp_f_oracle(ckpt_path, label):
    log(f"=== EXP F: Oracle injection for {label} ===")
    from polar.encoder import bit_reversal_perm

    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    if ckpt_path:
        load_ckpt(model, ckpt_path)
    model.eval()

    B = 128
    u, v, z = gen(B)
    d = model.d

    oracle_ks = [0, 64, 128, 192, 256, 384, 448, 500]
    results = {}

    for K in oracle_ks:
        br = torch.from_numpy(bit_reversal_perm(int(math.log2(N)))).long()
        root = model.z_encoder(z.unsqueeze(-1))[:, br]
        edge_data = [None] * (2 * N)
        edge_data[1] = root
        no_info = model.no_info_emb.unsqueeze(0).unsqueeze(0)
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_data[beta] = no_info.expand(B, size, d).clone()

        dec_head = 1; u_hat = {}; v_hat = {}; i_u = 0; i_v = 0

        with torch.no_grad():
            for step in range(2 * N):
                gamma = b[step]
                if gamma == 0:
                    i_u += 1; i_t = i_u; fdict = fu_seq
                else:
                    i_v += 1; i_t = i_v; fdict = fv_seq
                leaf_edge = i_t + N - 1
                target_vtx = leaf_edge >> 1
                dec_head = model._step_to(dec_head, target_vtx, edge_data)
                temp = edge_data[leaf_edge][:, 0].clone()
                if leaf_edge & 1 == 0:
                    model._neural_calc_left(target_vtx, edge_data)
                else:
                    model._neural_calc_right(target_vtx, edge_data)
                top_down = edge_data[leaf_edge][:, 0]
                combined = top_down + temp
                logits = model.emb2logits(combined)

                if i_t in fdict:
                    bit = torch.full((B,), fdict[i_t], dtype=torch.float32)
                elif step < K:
                    bit = u[:, i_t-1] if gamma == 0 else v[:, i_t-1]
                else:
                    if gamma == 0:
                        p0 = torch.logsumexp(logits[:, :2], dim=1)
                        p1 = torch.logsumexp(logits[:, 2:], dim=1)
                    else:
                        p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                        p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                    bit = (p1 > p0).float()

                if gamma == 0: u_hat[i_t] = bit
                else: v_hat[i_t] = bit
                new_emb = model._make_leaf_emb(u_hat.get(i_t), v_hat.get(i_t), B, 'cpu')
                edge_data[leaf_edge] = new_emb.unsqueeze(1)

        ud = torch.zeros_like(u); vd = torch.zeros_like(v)
        for i, bit in u_hat.items(): ud[:, i-1] = bit
        for i, bit in v_hat.items(): vd[:, i-1] = bit
        bl = ((ud != u) | (vd != v)).any(dim=1).float().mean().item()
        results[K] = bl
        log(f"  Oracle K={K}: BLER={bl:.4f}")

    return results


# ================================================================
# EXP G: Full model from checkpoint (baseline BLER)
# ================================================================
def exp_g_baseline():
    log("=== EXP G: Baseline BLER from checkpoints ===")
    for name, path in [
        ('campaign_n256_sched_best', 'saved_models/campaign_n256_sched_best.pt'),
        ('ncg_gmac_mlp_N128', 'saved_models/ncg_gmac_mlp_N128.pt'),
        ('ncg_gmac_mlp_N256', 'saved_models/ncg_gmac_mlp_N256.pt'),
    ]:
        model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
        loaded = load_ckpt(model, path)
        ue, ve, ze = gen(128)
        bl = bler(model, ue, ve, ze)
        log(f"  {name}: BLER={bl:.4f} (loaded {len(loaded)} params)")


# ================================================================
if __name__ == '__main__':
    # Clear log
    with open(LOG_FILE, 'w') as f:
        f.write('')

    log("=" * 60)
    log("N=256 FOCUSED BOTTLENECK INVESTIGATION")
    log("=" * 60)

    t_total = time.time()

    # Quick baselines
    exp_g_baseline()

    # Overfit test (answers: can the architecture represent N=256?)
    t0 = time.time()
    ra = exp_a_overfit_one()
    log(f"  EXP A took {time.time()-t0:.0f}s")

    # Teacher forcing gap
    t0 = time.time()
    exp_b_tf_gap()
    log(f"  EXP B took {time.time()-t0:.0f}s")

    # Component isolation
    t0 = time.time()
    exp_c_freeze_zenc()
    log(f"  EXP C took {time.time()-t0:.0f}s")

    t0 = time.time()
    exp_d_freeze_tree()
    log(f"  EXP D took {time.time()-t0:.0f}s")

    # Bottleneck localization (curriculum model)
    t0 = time.time()
    exp_e_bottleneck('saved_models/campaign_n256_sched_best.pt', 'curriculum')
    log(f"  EXP E (curriculum) took {time.time()-t0:.0f}s")

    # Bottleneck localization (scratch model after 1500 iters from exp B)
    t0 = time.time()
    exp_e_bottleneck('saved_models/scratch_n256_1500.pt', 'scratch_1500')
    log(f"  EXP E (scratch) took {time.time()-t0:.0f}s")

    # Oracle injection
    t0 = time.time()
    exp_f_oracle('saved_models/campaign_n256_sched_best.pt', 'curriculum')
    log(f"  EXP F (curriculum) took {time.time()-t0:.0f}s")

    t0 = time.time()
    exp_f_oracle('saved_models/scratch_n256_1500.pt', 'scratch_1500')
    log(f"  EXP F (scratch) took {time.time()-t0:.0f}s")

    t0 = time.time()
    exp_f_oracle('saved_models/ncg_gmac_mlp_N128.pt', 'n128_direct')
    log(f"  EXP F (N128 direct) took {time.time()-t0:.0f}s")

    log(f"\nTOTAL TIME: {time.time()-t_total:.0f}s")
    log("DONE")
