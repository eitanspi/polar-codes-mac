"""
N=256 Bottleneck Investigation — Component Isolation, Overfit Tests, Bottleneck Localization.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, torch.nn.functional as F, numpy as np
from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

DEVICE = 'cpu'
LOG_FILE = os.path.join(os.path.dirname(__file__), 'bottleneck_analysis.log')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def setup(N=256):
    n = int(np.log2(N))
    sigma2 = 10**(-6/10)
    channel = GaussianMAC(sigma2=sigma2)
    b = make_path(N, N//2)
    Au, Av, fu_raw, fv_raw, _, _, _ = design_from_file(f'designs/gmac_B_n8_snr6dB.npz', n, 123, 123)
    fu_seq = {i:0 for i in range(1, N+1) if i not in Au}
    fv_seq = {i:0 for i in range(1, N+1) if i not in Av}
    return n, sigma2, channel, b, Au, Av, fu_seq, fv_seq

def generate_batch(B, N, Au, Av, channel):
    u = torch.zeros(B, N)
    v = torch.zeros(B, N)
    for a in Au:
        u[:, a-1] = torch.randint(0, 2, (B,)).float()
    for a in Av:
        v[:, a-1] = torch.randint(0, 2, (B,)).float()
    cu = polar_encode_batch(u.numpy().astype(int))
    cv = polar_encode_batch(v.numpy().astype(int))
    cu_t = torch.from_numpy(cu).float()
    cv_t = torch.from_numpy(cv).float()
    x = (1 - 2*cu_t)
    y = (1 - 2*cv_t)
    noise = torch.randn(B, N) * np.sqrt(channel.sigma2)
    z = x + y + noise
    return u, v, z

def load_checkpoint_weights(model, ckpt_path):
    """Load checkpoint with tree. prefix mapping."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

    # Map tree.X -> X and z_encoder.X -> z_encoder.X
    mapped = {}
    for k, v in sd.items():
        if k.startswith('tree.embedding_z'):
            continue  # skip discrete embedding
        if k.startswith('tree.'):
            mapped[k[5:]] = v  # strip 'tree.' prefix
        else:
            mapped[k] = v

    # Only load matching keys
    model_sd = model.state_dict()
    loaded = {k: v for k, v in mapped.items() if k in model_sd and model_sd[k].shape == v.shape}
    model_sd.update(loaded)
    model.load_state_dict(model_sd)
    return list(loaded.keys())

def compute_bler(model, z, b, fu_seq, fv_seq, u, v, N):
    model.eval()
    with torch.no_grad():
        _, _, u_hat, v_hat, _ = model(z, b, fu_seq, fv_seq)
    u_dec = torch.zeros_like(u)
    v_dec = torch.zeros_like(v)
    for i, bit in u_hat.items():
        u_dec[:, i-1] = bit
    for i, bit in v_hat.items():
        v_dec[:, i-1] = bit
    block_err = ((u_dec != u) | (v_dec != v)).any(dim=1).float().mean().item()
    return block_err

def train_step(model, optimizer, z, b, fu_seq, fv_seq, u, v):
    model.train()
    al, at, u_hat, v_hat, _ = model(z, b, fu_seq, fv_seq, u_true=u, v_true=v)
    if len(al) == 0:
        return 0.0
    loss = F.cross_entropy(torch.cat(al), torch.cat(at))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


# ============================================================
# EXPERIMENT 1: Component Isolation
# ============================================================
def experiment_1_component_isolation():
    log("=" * 60)
    log("EXPERIMENT 1: Component Isolation")
    log("=" * 60)

    N = 256
    n, sigma2, channel, b, Au, Av, fu_seq, fv_seq = setup(N)
    B = 32
    ITERS = 3000

    results = {}

    # 1A: Freeze z_encoder, train tree ops from scratch
    log("--- 1A: Frozen z_encoder (from N=128), fresh tree ops ---")
    model_a = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    loaded_a = load_checkpoint_weights(model_a, 'saved_models/ncg_gmac_mlp_N128.pt')
    log(f"  Loaded {len(loaded_a)} weight tensors from N=128 checkpoint")

    # Freeze z_encoder
    for name, p in model_a.named_parameters():
        if 'z_encoder' in name:
            p.requires_grad = False

    # Re-init tree ops
    for name, p in model_a.named_parameters():
        if 'z_encoder' not in name and 'no_info' not in name:
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    optimizer_a = torch.optim.Adam(filter(lambda p: p.requires_grad, model_a.parameters()), lr=1e-3)

    losses_a = []
    for it in range(ITERS):
        u, v, z = generate_batch(B, N, Au, Av, channel)
        loss = train_step(model_a, optimizer_a, z, b, fu_seq, fv_seq, u, v)
        losses_a.append(loss)
        if (it+1) % 500 == 0:
            u_eval, v_eval, z_eval = generate_batch(64, N, Au, Av, channel)
            bler = compute_bler(model_a, z_eval, b, fu_seq, fv_seq, u_eval, v_eval, N)
            log(f"  1A iter {it+1}: loss={np.mean(losses_a[-100:]):.4f}, BLER={bler:.4f}")

    u_eval, v_eval, z_eval = generate_batch(128, N, Au, Av, channel)
    bler_a = compute_bler(model_a, z_eval, b, fu_seq, fv_seq, u_eval, v_eval, N)
    results['1A_frozen_zenc_fresh_tree'] = {'final_loss': np.mean(losses_a[-100:]), 'final_bler': bler_a}
    log(f"  1A FINAL: loss={results['1A_frozen_zenc_fresh_tree']['final_loss']:.4f}, BLER={bler_a:.4f}")

    # 1B: Freeze tree ops, train z_encoder from scratch
    log("--- 1B: Frozen tree ops (from N=128), fresh z_encoder ---")
    model_b = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    loaded_b = load_checkpoint_weights(model_b, 'saved_models/ncg_gmac_mlp_N128.pt')

    # Freeze tree ops (everything except z_encoder)
    for name, p in model_b.named_parameters():
        if 'z_encoder' not in name:
            p.requires_grad = False

    # Re-init z_encoder
    for name, p in model_b.named_parameters():
        if 'z_encoder' in name:
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    optimizer_b = torch.optim.Adam(filter(lambda p: p.requires_grad, model_b.parameters()), lr=1e-3)

    losses_b = []
    for it in range(ITERS):
        u, v, z = generate_batch(B, N, Au, Av, channel)
        loss = train_step(model_b, optimizer_b, z, b, fu_seq, fv_seq, u, v)
        losses_b.append(loss)
        if (it+1) % 500 == 0:
            u_eval, v_eval, z_eval = generate_batch(64, N, Au, Av, channel)
            bler = compute_bler(model_b, z_eval, b, fu_seq, fv_seq, u_eval, v_eval, N)
            log(f"  1B iter {it+1}: loss={np.mean(losses_b[-100:]):.4f}, BLER={bler:.4f}")

    u_eval, v_eval, z_eval = generate_batch(128, N, Au, Av, channel)
    bler_b = compute_bler(model_b, z_eval, b, fu_seq, fv_seq, u_eval, v_eval, N)
    results['1B_frozen_tree_fresh_zenc'] = {'final_loss': np.mean(losses_b[-100:]), 'final_bler': bler_b}
    log(f"  1B FINAL: loss={results['1B_frozen_tree_fresh_zenc']['final_loss']:.4f}, BLER={bler_b:.4f}")

    # 1C: Teacher forcing vs free-running gap
    log("--- 1C: Teacher forcing vs free-running (from scratch) ---")
    model_c = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=1e-3)

    losses_c = []
    for it in range(ITERS):
        u, v, z = generate_batch(B, N, Au, Av, channel)
        loss = train_step(model_c, optimizer_c, z, b, fu_seq, fv_seq, u, v)
        losses_c.append(loss)
        if (it+1) % 500 == 0:
            u_eval, v_eval, z_eval = generate_batch(64, N, Au, Av, channel)
            bler = compute_bler(model_c, z_eval, b, fu_seq, fv_seq, u_eval, v_eval, N)
            # Also compute teacher-forced loss on eval
            model_c.eval()
            with torch.no_grad():
                al_tf, at_tf, _, _, _ = model_c(z_eval, b, fu_seq, fv_seq, u_true=u_eval, v_true=v_eval)
                if len(al_tf) > 0:
                    tf_loss = F.cross_entropy(torch.cat(al_tf), torch.cat(at_tf)).item()
                else:
                    tf_loss = 0.0
            log(f"  1C iter {it+1}: train_loss={np.mean(losses_c[-100:]):.4f}, TF_eval_loss={tf_loss:.4f}, BLER={bler:.4f}")

    results['1C_from_scratch'] = {'final_loss': np.mean(losses_c[-100:]), 'final_bler': bler}

    return results


# ============================================================
# EXPERIMENT 2: Overfit Tests
# ============================================================
def experiment_2_overfit():
    log("=" * 60)
    log("EXPERIMENT 2: Overfit Tests")
    log("=" * 60)

    N = 256
    n, sigma2, channel, b, Au, Av, fu_seq, fv_seq = setup(N)
    results = {}

    # 2A: Overfit 10 fixed codewords
    log("--- 2A: Overfit 10 fixed codewords at N=256 ---")
    u_fixed, v_fixed, z_fixed = generate_batch(10, N, Au, Av, channel)

    model_2a = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    optimizer_2a = torch.optim.Adam(model_2a.parameters(), lr=1e-3)

    for it in range(5000):
        # Re-generate noise each time (same info bits, different noise)
        cu = polar_encode_batch(u_fixed.numpy().astype(int))
        cv = polar_encode_batch(v_fixed.numpy().astype(int))
        x = 1 - 2*torch.from_numpy(cu).float()
        y = 1 - 2*torch.from_numpy(cv).float()
        z_noisy = x + y + torch.randn(10, N) * np.sqrt(sigma2)

        loss = train_step(model_2a, optimizer_2a, z_noisy, b, fu_seq, fv_seq, u_fixed, v_fixed)
        if (it+1) % 1000 == 0:
            bler = compute_bler(model_2a, z_noisy, b, fu_seq, fv_seq, u_fixed, v_fixed, N)
            log(f"  2A iter {it+1}: loss={loss:.4f}, BLER(train)={bler:.4f}")

    bler_2a = compute_bler(model_2a, z_noisy, b, fu_seq, fv_seq, u_fixed, v_fixed, N)
    results['2A_overfit_10'] = {'final_loss': loss, 'final_bler': bler_2a}
    log(f"  2A FINAL: loss={loss:.4f}, BLER={bler_2a:.4f}")

    # 2B: Overfit 1 FIXED codeword with FIXED noise (pure memorization)
    log("--- 2B: Overfit 1 fixed codeword, fixed noise (memorization) ---")
    u1, v1, z1 = generate_batch(1, N, Au, Av, channel)

    model_2b = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    optimizer_2b = torch.optim.Adam(model_2b.parameters(), lr=1e-3)

    for it in range(5000):
        loss = train_step(model_2b, optimizer_2b, z1, b, fu_seq, fv_seq, u1, v1)
        if (it+1) % 1000 == 0:
            bler = compute_bler(model_2b, z1, b, fu_seq, fv_seq, u1, v1, N)
            log(f"  2B iter {it+1}: loss={loss:.6f}, BLER={bler:.4f}")

    bler_2b = compute_bler(model_2b, z1, b, fu_seq, fv_seq, u1, v1, N)
    results['2B_overfit_1_fixed'] = {'final_loss': loss, 'final_bler': bler_2b}
    log(f"  2B FINAL: loss={loss:.6f}, BLER={bler_2b:.4f}")

    return results


# ============================================================
# EXPERIMENT 3: Bottleneck Localization
# ============================================================
def experiment_3_bottleneck(ckpt_path, label):
    log(f"--- 3: Bottleneck analysis for {label} ---")

    N = 256
    n, sigma2, channel, b, Au, Av, fu_seq, fv_seq = setup(N)

    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    if ckpt_path:
        loaded = load_checkpoint_weights(model, ckpt_path)
        log(f"  Loaded {len(loaded)} weights from {ckpt_path}")

    model.eval()
    B = 64
    u, v, z = generate_batch(B, N, Au, Av, channel)

    # Instrument the decode: record per-leaf accuracy and confidence
    import math
    from polar.encoder import bit_reversal_perm

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

    leaf_stats = []  # (step, gamma, i_t, accuracy, mean_confidence, first_error_frac)

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
                is_frozen = True
            else:
                # Decode
                if gamma == 0:
                    p0 = torch.logsumexp(logits[:, :2], dim=1)
                    p1 = torch.logsumexp(logits[:, 2:], dim=1)
                else:
                    p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                    p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                bit = (p1 > p0).float()

                # Compute accuracy
                true_bit = u[:, i_t-1] if gamma == 0 else v[:, i_t-1]
                acc = (bit == true_bit).float().mean().item()
                confidence = (torch.abs(p1 - p0)).mean().item()

                leaf_stats.append({
                    'step': step, 'gamma': gamma, 'i_t': i_t,
                    'accuracy': acc, 'confidence': confidence, 'is_frozen': False
                })
                is_frozen = False

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            new_emb = model._make_leaf_emb(u_hat.get(i_t), v_hat.get(i_t), B, 'cpu')
            edge_data[leaf_edge] = new_emb.unsqueeze(1)

    # Analyze where errors start
    if leaf_stats:
        accs = [s['accuracy'] for s in leaf_stats]
        confs = [s['confidence'] for s in leaf_stats]

        # Find first leaf where accuracy drops below 0.9
        first_bad = None
        for i, s in enumerate(leaf_stats):
            if s['accuracy'] < 0.9:
                first_bad = i
                break

        n_info = len(leaf_stats)
        early_acc = np.mean(accs[:n_info//4]) if n_info > 4 else 0
        mid_acc = np.mean(accs[n_info//4:n_info//2]) if n_info > 4 else 0
        late_acc = np.mean(accs[n_info//2:3*n_info//4]) if n_info > 4 else 0
        final_acc = np.mean(accs[3*n_info//4:]) if n_info > 4 else 0

        log(f"  {label}: {n_info} info bits, first_bad_leaf={first_bad}")
        log(f"  Accuracy by quarter: early={early_acc:.3f}, mid={mid_acc:.3f}, late={late_acc:.3f}, final={final_acc:.3f}")
        log(f"  Confidence by quarter: early={np.mean(confs[:n_info//4]):.3f}, mid={np.mean(confs[n_info//4:n_info//2]):.3f}, late={np.mean(confs[n_info//2:3*n_info//4]):.3f}, final={np.mean(confs[3*n_info//4:]):.3f}")

        return {'first_bad': first_bad, 'early_acc': early_acc, 'mid_acc': mid_acc,
                'late_acc': late_acc, 'final_acc': final_acc, 'all_accs': accs, 'all_confs': confs}
    return {}


# ============================================================
# EXPERIMENT 4: Oracle Injection
# ============================================================
def experiment_4_oracle(ckpt_path, label):
    log(f"--- 4: Oracle injection for {label} ---")

    N = 256
    n, sigma2, channel, b, Au, Av, fu_seq, fv_seq = setup(N)

    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    if ckpt_path:
        loaded = load_checkpoint_weights(model, ckpt_path)

    model.eval()
    B = 128
    u, v, z = generate_batch(B, N, Au, Av, channel)

    oracle_ks = [0, 64, 128, 192, 256, 384, 448]
    results = {}

    for K in oracle_ks:
        import math
        from polar.encoder import bit_reversal_perm

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
                    # Oracle: use true bits
                    bit = u[:, i_t-1] if gamma == 0 else v[:, i_t-1]
                else:
                    if gamma == 0:
                        p0 = torch.logsumexp(logits[:, :2], dim=1)
                        p1 = torch.logsumexp(logits[:, 2:], dim=1)
                    else:
                        p0 = torch.logsumexp(logits[:, [0, 2]], dim=1)
                        p1 = torch.logsumexp(logits[:, [1, 3]], dim=1)
                    bit = (p1 > p0).float()

                if gamma == 0:
                    u_hat[i_t] = bit
                else:
                    v_hat[i_t] = bit

                new_emb = model._make_leaf_emb(u_hat.get(i_t), v_hat.get(i_t), B, 'cpu')
                edge_data[leaf_edge] = new_emb.unsqueeze(1)

        # Compute BLER
        u_dec = torch.zeros_like(u)
        v_dec = torch.zeros_like(v)
        for i, bit in u_hat.items():
            u_dec[:, i-1] = bit
        for i, bit in v_hat.items():
            v_dec[:, i-1] = bit
        bler = ((u_dec != u) | (v_dec != v)).any(dim=1).float().mean().item()

        results[K] = bler
        log(f"  Oracle K={K}: BLER={bler:.4f}")

    return results


# ============================================================
# EXPERIMENT 5: From-scratch model (short training) for comparison
# ============================================================
def experiment_5_scratch_model():
    log("=" * 60)
    log("EXPERIMENT 5: Train from scratch for 3K iters, then analyze")
    log("=" * 60)

    N = 256
    n, sigma2, channel, b, Au, Av, fu_seq, fv_seq = setup(N)
    B = 32

    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for it in range(3000):
        u, v, z = generate_batch(B, N, Au, Av, channel)
        loss = train_step(model, optimizer, z, b, fu_seq, fv_seq, u, v)
        if (it+1) % 1000 == 0:
            u_eval, v_eval, z_eval = generate_batch(64, N, Au, Av, channel)
            bler = compute_bler(model, z_eval, b, fu_seq, fv_seq, u_eval, v_eval, N)
            log(f"  Scratch iter {it+1}: loss={loss:.4f}, BLER={bler:.4f}")

    # Save for bottleneck analysis
    torch.save(model.state_dict(), 'saved_models/scratch_n256_3k.pt')
    return model


# ============================================================
# EXPERIMENT 6: N=32 vs N=256 from scratch comparison
# ============================================================
def experiment_6_n32_sanity():
    log("=" * 60)
    log("EXPERIMENT 6: N=32 from scratch (sanity check)")
    log("=" * 60)

    from polar.design import design_gmac
    N = 32
    n = 5
    sigma2 = 10**(-6/10)
    channel = GaussianMAC(sigma2=sigma2)
    b = make_path(N, N//2)
    Au, Av, fu_raw, fv_raw, _, _ = design_gmac(n, 13, 13, sigma2=sigma2)
    fu_seq = {i:0 for i in range(1, N+1) if i not in Au}
    fv_seq = {i:0 for i in range(1, N+1) if i not in Av}

    B = 32
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for it in range(5000):
        u = torch.zeros(B, N)
        v = torch.zeros(B, N)
        for a in Au:
            u[:, a-1] = torch.randint(0, 2, (B,)).float()
        for a in Av:
            v[:, a-1] = torch.randint(0, 2, (B,)).float()
        cu = polar_encode_batch(u.numpy().astype(int))
        cv = polar_encode_batch(v.numpy().astype(int))
        x = 1 - 2*torch.from_numpy(cu).float()
        y = 1 - 2*torch.from_numpy(cv).float()
        z = x + y + torch.randn(B, N) * np.sqrt(sigma2)

        loss = train_step(model, optimizer, z, b, fu_seq, fv_seq, u, v)
        if (it+1) % 1000 == 0:
            u_e = torch.zeros(64, N)
            v_e = torch.zeros(64, N)
            for a in Au:
                u_e[:, a-1] = torch.randint(0, 2, (64,)).float()
            for a in Av:
                v_e[:, a-1] = torch.randint(0, 2, (64,)).float()
            cu_e = polar_encode_batch(u_e.numpy().astype(int))
            cv_e = polar_encode_batch(v_e.numpy().astype(int))
            x_e = 1 - 2*torch.from_numpy(cu_e).float()
            y_e = 1 - 2*torch.from_numpy(cv_e).float()
            z_e = x_e + y_e + torch.randn(64, N) * np.sqrt(sigma2)
            bler = compute_bler(model, z_e, b, fu_seq, fv_seq, u_e, v_e, N)
            log(f"  N=32 iter {it+1}: loss={loss:.4f}, BLER={bler:.4f}")

    return model


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    log("=" * 60)
    log("N=256 BOTTLENECK INVESTIGATION — Starting")
    log("=" * 60)

    all_results = {}

    # Experiment 6: N=32 sanity (quick)
    t0 = time.time()
    experiment_6_n32_sanity()
    log(f"  N=32 experiment took {time.time()-t0:.0f}s")

    # Experiment 2: Overfit tests
    t0 = time.time()
    r2 = experiment_2_overfit()
    all_results['exp2'] = r2
    log(f"  Overfit experiments took {time.time()-t0:.0f}s")

    # Experiment 1: Component isolation
    t0 = time.time()
    r1 = experiment_1_component_isolation()
    all_results['exp1'] = r1
    log(f"  Component isolation took {time.time()-t0:.0f}s")

    # Experiment 5: Train scratch model for bottleneck comparison
    t0 = time.time()
    scratch_model = experiment_5_scratch_model()
    log(f"  Scratch training took {time.time()-t0:.0f}s")

    # Experiment 3: Bottleneck localization
    log("=" * 60)
    log("EXPERIMENT 3: Bottleneck Localization")
    log("=" * 60)

    r3_curriculum = experiment_3_bottleneck('saved_models/campaign_n256_sched_best.pt', 'curriculum_N256')
    all_results['exp3_curriculum'] = r3_curriculum

    r3_scratch = experiment_3_bottleneck(None, 'scratch_3K')
    # Load the just-trained scratch model
    scratch_model2 = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32)
    scratch_model2.load_state_dict(torch.load('saved_models/scratch_n256_3k.pt', weights_only=False))
    r3_scratch2 = experiment_3_bottleneck('saved_models/scratch_n256_3k.pt', 'scratch_3K_trained')
    all_results['exp3_scratch'] = r3_scratch2

    # Experiment 4: Oracle injection
    log("=" * 60)
    log("EXPERIMENT 4: Oracle Injection")
    log("=" * 60)

    r4_curriculum = experiment_4_oracle('saved_models/campaign_n256_sched_best.pt', 'curriculum_N256')
    all_results['exp4_curriculum'] = r4_curriculum

    r4_scratch = experiment_4_oracle('saved_models/scratch_n256_3k.pt', 'scratch_3K')
    all_results['exp4_scratch'] = r4_scratch

    # Also test with N=128 checkpoint directly (no N=256 training)
    r4_n128 = experiment_4_oracle('saved_models/ncg_gmac_mlp_N128.pt', 'n128_direct')
    all_results['exp4_n128'] = r4_n128

    log("=" * 60)
    log("ALL EXPERIMENTS COMPLETE")
    log("=" * 60)

    # Save raw results
    with open(os.path.join(os.path.dirname(__file__), 'bottleneck_results.json'), 'w') as f:
        # Convert numpy to python types
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(all_results), f, indent=2)

    log("Results saved to neural/bottleneck_results.json")
