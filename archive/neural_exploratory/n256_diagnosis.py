#!/usr/bin/env python3
"""
N=256 Failure Diagnosis — All 5 Tasks
======================================
"""

import sys, os, time, math, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

# ── Logging setup ────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(__file__), 'n256_diagnosis.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

DEVICE = 'cpu'
torch.manual_seed(42)
np.random.seed(42)


# ── SimpleMLP_Gmac: the wrapper class used to train the checkpoint ───────────

class SimpleMLP_Gmac(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = 16
        self.z_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.ELU(), nn.Linear(32, 16))
        self.tree = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)

    def forward_decode(self, z, b, fu, fv, u_true=None, v_true=None):
        """Decode using z_encoder + tree.forward with root_emb."""
        n = int(math.log2(z.shape[1]))
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(z.device)
        root = self.z_encoder(z.unsqueeze(-1))[:, br]
        return self.tree(z=None, b=b, frozen_u=fu, frozen_v=fv,
                         root_emb=root, u_true=u_true, v_true=v_true)


# ── Helpers ──────────────────────────────────────────────────────────────────

def setup_code(N, n, ku, kv, snr_db=6):
    sigma2 = 10**(-snr_db/10)
    channel = GaussianMAC(sigma2=sigma2)
    b = make_path(N, N//2)
    Au, Av, fu, fv, _, _, _ = design_from_file(
        f'designs/gmac_B_n{n}_snr{snr_db}dB.npz', n, ku, kv)
    fu_seq = {i: 0 for i in range(1, N+1) if i not in Au}
    fv_seq = {i: 0 for i in range(1, N+1) if i not in Av}
    return channel, b, Au, Av, fu_seq, fv_seq


def generate_batch(channel, Au, Av, N, batch):
    rng = np.random.default_rng()
    uf = np.zeros((batch, N), dtype=np.int32)
    vf = np.zeros((batch, N), dtype=np.int32)
    for p in Au:
        uf[:, p-1] = rng.integers(0, 2, batch)
    for p in Av:
        vf[:, p-1] = rng.integers(0, 2, batch)
    xu = polar_encode_batch(uf)
    xv = polar_encode_batch(vf)
    z_np = channel.sample_batch(xu, xv)
    z = torch.from_numpy(z_np).float().to(DEVICE)
    return uf, vf, z


def compute_bler(u_hat, v_hat, uf, vf, Au, Av, batch):
    errs = 0
    for s in range(batch):
        ok = True
        for p in Au:
            if p in u_hat and int(u_hat[p][s].item()) != uf[s, p-1]:
                ok = False; break
        if ok:
            for p in Av:
                if p in v_hat and int(v_hat[p][s].item()) != vf[s, p-1]:
                    ok = False; break
        if not ok:
            errs += 1
    return errs / batch


def load_campaign_model(path='saved_models/campaign_n256_sched_best.pt'):
    model = SimpleMLP_Gmac()
    sd = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1: N=32 Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

def task1_sanity_check():
    log.info("=" * 70)
    log.info("TASK 1: N=32 Sanity Check (10K iters, batch=32)")
    log.info("=" * 70)

    N, n, ku, kv = 32, 5, 15, 15
    channel, b, Au, Av, fu_seq, fv_seq = setup_code(N, n, ku, kv)

    # Use the GmacNeuralCompGraphDecoder (self-contained) for N=32 training
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, z_hidden=32).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    log.info(f"  Parameters: {model.count_parameters()}")
    log.info(f"  Au ({len(Au)} info): {sorted(Au)}")
    log.info(f"  Av ({len(Av)} info): {sorted(Av)}")

    batch = 32
    results = []

    for it in range(1, 10001):
        model.train()
        uf, vf, z = generate_batch(channel, Au, Av, N, batch)

        al, at, u_hat, v_hat, _ = model(
            z, b, fu_seq, fv_seq,
            u_true=torch.from_numpy(uf).long().to(DEVICE),
            v_true=torch.from_numpy(vf).long().to(DEVICE))

        loss = F.cross_entropy(torch.cat(al), torch.cat(at))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 2000 == 0 or it == 1:
            model.eval()
            with torch.no_grad():
                uf_e, vf_e, z_e = generate_batch(channel, Au, Av, N, 200)
                _, _, u_hat_e, v_hat_e, _ = model(z_e, b, fu_seq, fv_seq)
                bler = compute_bler(u_hat_e, v_hat_e, uf_e, vf_e, Au, Av, 200)

                al2, at2, _, _, _ = model(
                    z_e, b, fu_seq, fv_seq,
                    u_true=torch.from_numpy(uf_e).long().to(DEVICE),
                    v_true=torch.from_numpy(vf_e).long().to(DEVICE))
                tf_loss = F.cross_entropy(torch.cat(al2), torch.cat(at2)).item()

            results.append((it, loss.item(), tf_loss, bler))
            log.info(f"  iter={it:5d}  train_loss={loss.item():.4f}  "
                     f"eval_TF_loss={tf_loss:.4f}  BLER={bler:.4f}")

    log.info("\nTask 1 Summary:")
    log.info(f"  {'Iter':>6s}  {'TrainLoss':>10s}  {'EvalLoss':>10s}  {'BLER':>8s}")
    for it, tl, el, bl in results:
        log.info(f"  {it:6d}  {tl:10.4f}  {el:10.4f}  {bl:8.4f}")

    final_bler = results[-1][3]
    if final_bler > 0.9:
        log.info("\n  *** N=32 FAILED TO LEARN -- BUG DETECTED ***")
        return False
    else:
        log.info(f"\n  N=32 learned successfully (BLER={final_bler:.4f})")
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2: Teacher Forcing vs Free Running (N=256 trained model)
# ══════════════════════════════════════════════════════════════════════════════

def task2_tf_vs_free():
    log.info("\n" + "=" * 70)
    log.info("TASK 2: Teacher Forcing vs Free Running (N=256)")
    log.info("=" * 70)

    N, n, ku, kv = 256, 8, 123, 123
    channel, b, Au, Av, fu_seq, fv_seq = setup_code(N, n, ku, kv)

    model = load_campaign_model()
    log.info(f"  Loaded campaign_n256_sched_best.pt")

    batch = 100
    uf, vf, z = generate_batch(channel, Au, Av, N, batch)

    # --- Teacher-forced ---
    with torch.no_grad():
        al_tf, at_tf, _, _, _ = model.forward_decode(
            z, b, fu_seq, fv_seq,
            u_true=torch.from_numpy(uf).long().to(DEVICE),
            v_true=torch.from_numpy(vf).long().to(DEVICE))

    n_leaves = len(al_tf)
    tf_correct_per_leaf = []
    tf_total_correct = 0
    for i in range(n_leaves):
        preds = al_tf[i].argmax(dim=1)
        targets = at_tf[i]
        correct = (preds == targets).float().mean().item()
        tf_correct_per_leaf.append(correct)
        tf_total_correct += (preds == targets).sum().item()

    tf_acc = tf_total_correct / (n_leaves * batch)
    tf_loss = F.cross_entropy(torch.cat(al_tf), torch.cat(at_tf)).item()

    log.info(f"\n  Teacher-Forced ({n_leaves} info leaves):")
    log.info(f"    Loss = {tf_loss:.4f}")
    log.info(f"    Per-leaf accuracy (mean) = {tf_acc:.6f}")
    log.info(f"    Per-leaf accuracy (min)  = {min(tf_correct_per_leaf):.4f}")
    log.info(f"    Per-leaf accuracy (max)  = {max(tf_correct_per_leaf):.4f}")

    # Accuracy by bucket
    bucket_size = max(1, n_leaves // 8)
    for bi in range(8):
        start = bi * bucket_size
        end = min((bi + 1) * bucket_size, n_leaves)
        if start >= n_leaves:
            break
        avg = np.mean(tf_correct_per_leaf[start:end])
        log.info(f"    Leaves [{start:3d}-{end:3d}): avg_acc = {avg:.4f}")

    # --- Free-running ---
    with torch.no_grad():
        _, _, u_hat_fr, v_hat_fr, _ = model.forward_decode(z, b, fu_seq, fv_seq)

    bler_fr = compute_bler(u_hat_fr, v_hat_fr, uf, vf, Au, Av, batch)

    # Track first error position
    decode_order = []
    i_u, i_v = 0, 0
    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u
        else:
            i_v += 1; i_t = i_v
        fdict = fu_seq if gamma == 0 else fv_seq
        if i_t not in fdict:
            decode_order.append((gamma, i_t, step))

    first_error_positions = []
    for s in range(batch):
        found = False
        for idx, (gamma, i_t, step_idx) in enumerate(decode_order):
            hat_dict = u_hat_fr if gamma == 0 else v_hat_fr
            true_arr = uf if gamma == 0 else vf
            if int(hat_dict[i_t][s].item()) != true_arr[s, i_t - 1]:
                first_error_positions.append(idx)  # leaf index (not step index)
                found = True
                break
        if not found:
            first_error_positions.append(-1)

    log.info(f"\n  Free-Running:")
    log.info(f"    BLER = {bler_fr:.4f}")

    err_positions = [p for p in first_error_positions if p >= 0]
    if err_positions:
        log.info(f"    Errors in {len(err_positions)}/{batch} codewords")
        bucket_size_fr = max(1, n_leaves // 8)
        for bi in range(8):
            lo = bi * bucket_size_fr
            hi = (bi + 1) * bucket_size_fr
            cnt = sum(1 for p in err_positions if lo <= p < hi)
            log.info(f"    First error in leaf [{lo:3d}-{hi:3d}): {cnt} codewords")
    else:
        log.info(f"    No errors in {batch} codewords")

    return tf_acc, tf_correct_per_leaf, bler_fr, first_error_positions


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 3: Oracle Injection Test
# ══════════════════════════════════════════════════════════════════════════════

def _oracle_forward(model, z, b, fu_seq, fv_seq, uf, vf, Au, Av, K, N, batch):
    """Forward with oracle for first K info decisions, then free-running."""
    B = z.shape[0]
    d = model.d
    device = z.device

    n = int(math.log2(N))
    br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
    root = model.z_encoder(z.unsqueeze(-1))[:, br]

    tree = model.tree

    edge_data = [None] * (2 * N)
    edge_data[1] = root
    no_info = tree.no_info_emb.unsqueeze(0).unsqueeze(0)
    for beta in range(2, 2 * N):
        level = beta.bit_length() - 1
        size = N >> level
        edge_data[beta] = no_info.expand(B, size, d).clone()

    dec_head = 1
    u_hat, v_hat = {}, {}
    i_u, i_v = 0, 0
    info_count = 0

    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u; fdict = fu_seq
        else:
            i_v += 1; i_t = i_v; fdict = fv_seq

        leaf_edge = i_t + N - 1
        target_vtx = leaf_edge >> 1
        dec_head = tree._step_to(dec_head, target_vtx, edge_data, None)

        temp = edge_data[leaf_edge][:, 0].clone()
        if leaf_edge & 1 == 0:
            tree._neural_calc_left(target_vtx, edge_data)
        else:
            tree._neural_calc_right(target_vtx, edge_data)
        top_down = edge_data[leaf_edge][:, 0]

        combined = top_down + temp
        logits = tree.emb2logits(combined)

        if i_t in fdict:
            bit = torch.full((B,), fdict[i_t], dtype=torch.float32, device=device)
        else:
            info_count += 1
            if info_count <= K:
                # Oracle
                if gamma == 0:
                    bit = torch.from_numpy(uf[:, i_t - 1].copy()).float().to(device)
                else:
                    bit = torch.from_numpy(vf[:, i_t - 1].copy()).float().to(device)
            else:
                # Free-running
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

        new_emb = tree._make_leaf_emb(u_hat.get(i_t), v_hat.get(i_t), B, device)
        edge_data[leaf_edge] = new_emb.unsqueeze(1)

    return compute_bler(u_hat, v_hat, uf, vf, Au, Av, batch)


def task3_oracle_injection():
    log.info("\n" + "=" * 70)
    log.info("TASK 3: Oracle Injection Test (N=256)")
    log.info("=" * 70)

    N, n, ku, kv = 256, 8, 123, 123
    channel, b, Au, Av, fu_seq, fv_seq = setup_code(N, n, ku, kv)

    model = load_campaign_model()
    batch = 100
    uf, vf, z = generate_batch(channel, Au, Av, N, batch)

    total_info = len(Au) + len(Av)
    log.info(f"  Total info leaves: {total_info}")

    K_values = [0, 50, 100, 150, 200, total_info]
    results = []

    for K in K_values:
        with torch.no_grad():
            bler = _oracle_forward(model, z, b, fu_seq, fv_seq,
                                   uf, vf, Au, Av, K, N, batch)
        results.append((K, bler))
        log.info(f"  Oracle K={K:4d}/{total_info}: BLER = {bler:.4f}")

    log.info("\n  Oracle Injection Summary:")
    log.info(f"  {'K':>6s}  {'BLER':>8s}")
    for K, bler in results:
        log.info(f"  {K:6d}  {bler:8.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 4: Random vs Trained Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

def task4_scratch_vs_curriculum():
    log.info("\n" + "=" * 70)
    log.info("TASK 4: Random vs Trained Model Comparison (N=256)")
    log.info("=" * 70)

    N, n, ku, kv = 256, 8, 123, 123
    channel, b, Au, Av, fu_seq, fv_seq = setup_code(N, n, ku, kv)

    batch = 100
    uf, vf, z = generate_batch(channel, Au, Av, N, batch)
    u_true_t = torch.from_numpy(uf).long().to(DEVICE)
    v_true_t = torch.from_numpy(vf).long().to(DEVICE)

    for label, load_fn in [
        ("TRAINED (curriculum)", lambda: load_campaign_model()),
        ("RANDOM (fresh)", lambda: SimpleMLP_Gmac().eval()),
    ]:
        log.info(f"\n  --- {label} ---")
        model = load_fn()

        with torch.no_grad():
            # Teacher-forced
            al_tf, at_tf, _, _, _ = model.forward_decode(
                z, b, fu_seq, fv_seq,
                u_true=u_true_t, v_true=v_true_t)
            tf_loss = F.cross_entropy(torch.cat(al_tf), torch.cat(at_tf)).item()

            tf_correct = 0
            tf_total = 0
            for i in range(len(al_tf)):
                preds = al_tf[i].argmax(dim=1)
                targets = at_tf[i]
                tf_correct += (preds == targets).sum().item()
                tf_total += batch
            tf_acc = tf_correct / tf_total

            # Free-running
            _, _, u_hat_fr, v_hat_fr, _ = model.forward_decode(z, b, fu_seq, fv_seq)
            bler_fr = compute_bler(u_hat_fr, v_hat_fr, uf, vf, Au, Av, batch)

        log.info(f"    TF Loss:     {tf_loss:.4f}")
        log.info(f"    TF Accuracy: {tf_acc:.6f}")
        log.info(f"    FR BLER:     {bler_fr:.4f}")

        # Per-leaf accuracy breakdown for TF
        n_leaves = len(al_tf)
        accs = []
        for i in range(n_leaves):
            preds = al_tf[i].argmax(dim=1)
            targets = at_tf[i]
            accs.append((preds == targets).float().mean().item())

        # Show quartile accuracy
        accs_np = np.array(accs)
        log.info(f"    TF per-leaf acc: min={accs_np.min():.4f} "
                 f"Q1={np.percentile(accs_np,25):.4f} "
                 f"median={np.median(accs_np):.4f} "
                 f"Q3={np.percentile(accs_np,75):.4f} "
                 f"max={accs_np.max():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    log.info("N=256 Failure Diagnosis -- Starting")
    log.info(f"Device: {DEVICE}")

    t0 = time.time()

    # Task 1
    t1_ok = task1_sanity_check()
    if not t1_ok:
        log.info("\n*** STOPPING: N=32 failed. Debug required. ***")

    # Task 2
    t2 = task2_tf_vs_free()

    # Task 3
    t3 = task3_oracle_injection()

    # Task 4
    task4_scratch_vs_curriculum()

    elapsed = time.time() - t0
    log.info(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info("\n  DONE")
