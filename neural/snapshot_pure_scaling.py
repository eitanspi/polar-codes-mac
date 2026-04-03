"""
snapshot_pure_scaling.py
========================
Curriculum training of the snapshot-pure neural MAC polar decoder.

Scales from N=8 to N=256 using curriculum:
  For each N: collect snapshots -> train -> evaluate BLER -> compare vs SC.

Uses the same prob-vector architecture from poc_snapshot_pure.py.
"""

import sys, os, time, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from polar.channels import GaussianMAC
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.decoder import build_log_W_leaf, _norm_prod_single, decode_single
from polar.design import design_gmac, make_path

torch.manual_seed(42)
np.random.seed(42)

D_EMB = 32
D_HIDDEN = 128
EPS = 1e-10

LOG_FILE = os.path.join(os.path.dirname(__file__), 'snapshot_pure_scaling.log')

# =============================================================================
#  Logging
# =============================================================================

log_fh = None

def log(msg=""):
    print(msg, flush=True)
    if log_fh:
        log_fh.write(msg + "\n")
        log_fh.flush()


# =============================================================================
#  Model (identical to poc_snapshot_pure.py)
# =============================================================================

class SnapshotModel(nn.Module):
    def __init__(self, d=D_EMB, h=D_HIDDEN):
        super().__init__()
        self.d = d
        self.logits2emb = nn.Sequential(
            nn.Linear(4, h), nn.GELU(), nn.Linear(h, d))
        self.emb2logits = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Linear(h, 4))
        self.calc_left_net = nn.Sequential(
            nn.Linear(3 * d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, d))
        self.calc_right_net = nn.Sequential(
            nn.Linear(3 * d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, d))
        self.calc_parent_net = nn.Sequential(
            nn.Linear(2 * d, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, d))

    def _to_emb(self, prob):
        log_p = torch.log(prob.clamp(min=EPS))
        return self.logits2emb(log_p)

    def _to_prob(self, emb):
        logits = self.emb2logits(emb)
        return F.softmax(logits, dim=-1)

    def calc_left(self, parent_top, parent_bot, right):
        e1, e2, e3 = self._to_emb(parent_top), self._to_emb(parent_bot), self._to_emb(right)
        return self._to_prob(self.calc_left_net(torch.cat([e1, e2, e3], dim=-1)))

    def calc_right(self, parent_top, parent_bot, left):
        e1, e2, e3 = self._to_emb(parent_top), self._to_emb(parent_bot), self._to_emb(left)
        return self._to_prob(self.calc_right_net(torch.cat([e1, e2, e3], dim=-1)))

    def calc_parent(self, left, right):
        e1, e2 = self._to_emb(left), self._to_emb(right)
        return self._to_prob(self.calc_parent_net(torch.cat([e1, e2], dim=-1)))


# =============================================================================
#  Design helper: get frozen sets for Class B at given N, SNR
# =============================================================================

def get_design(n, sigma2, err_threshold=0.01):
    """
    Get frozen sets for Class B (path_i = N//2) at given N and sigma2.
    Uses MC design if available, otherwise GA design.

    Returns: frozen_u, frozen_v, ku, kv, path_i
    """
    N = 1 << n
    path_i = N // 2

    # Try MC design first
    mc_file = os.path.join(os.path.dirname(__file__), '..',
                           'designs', f'gmac_B_n{n}_snr6dB.npz')
    if os.path.exists(mc_file):
        d = np.load(mc_file)
        u_err = d['u_error_rates']
        v_err = d['v_error_rates']

        # Reliable channels: error rate < threshold
        u_reliable = np.where(u_err < err_threshold)[0]  # 0-indexed
        v_reliable = np.where(v_err < err_threshold)[0]
        ku = len(u_reliable)
        kv = len(v_reliable)

        # Use GA design with these ku, kv to get frozen sets in the right format
        Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)
        log(f"  Design: MC-guided GA, ku={ku}, kv={kv}, rate=({ku/N:.3f}, {kv/N:.3f})")
        return frozen_u, frozen_v, ku, kv, path_i
    else:
        # Fallback: use ~50% rate for U and ~70% for V (typical Class B)
        ku = max(1, int(N * 0.5))
        kv = max(1, int(N * 0.7))
        Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)
        log(f"  Design: GA fallback, ku={ku}, kv={kv}, rate=({ku/N:.3f}, {kv/N:.3f})")
        return frozen_u, frozen_v, ku, kv, path_i


# =============================================================================
#  Snapshot generation (from poc_snapshot_pure.py)
# =============================================================================

def prob_from_log(log_tensor):
    flat = log_tensor.ravel()
    p = np.exp(flat - np.max(flat))
    s = p.sum()
    return p / s if s > 0 else np.ones(4) / 4.0


def generate_snapshots(n, N, num_cw, sigma2, frozen_u, frozen_v, path_i):
    """Run analytical SC decoder and record snapshots."""
    channel = GaussianMAC(sigma2=sigma2)
    b = make_path(N, path_i)

    left_snaps = []
    right_snaps = []
    parent_snaps = []

    from polar.decoder import (_CompGraph, _norm_prod_single, _LOG_HALF, _LOG_QUARTER,
                                _circ_conv_batch, _norm_prod_batch)
    _NEG_INF = -np.inf

    class InstrumentedCompGraph(_CompGraph):
        def calc_left(self, beta):
            parent = self.edge_data[beta]
            right = self.edge_data[2 * beta + 1]
            l = right.shape[0]
            for i in range(l):
                temp_i = _norm_prod_single(parent[l + i], right[i])
                out_i = _circ_conv_batch(parent[i:i+1], temp_i[np.newaxis])[0]
                left_snaps.append((
                    prob_from_log(parent[i]),
                    prob_from_log(parent[l + i]),
                    prob_from_log(right[i]),
                    prob_from_log(out_i),
                ))
            super().calc_left(beta)

        def calc_right(self, beta):
            parent = self.edge_data[beta]
            left = self.edge_data[2 * beta]
            l = left.shape[0]
            for i in range(l):
                temp_i = _circ_conv_batch(left[i:i+1], parent[i:i+1])[0]
                out_i = _norm_prod_single(parent[l + i], temp_i)
                right_snaps.append((
                    prob_from_log(parent[i]),
                    prob_from_log(parent[l + i]),
                    prob_from_log(left[i]),
                    prob_from_log(out_i),
                ))
            super().calc_right(beta)

        def calc_parent(self, beta):
            left = self.edge_data[2 * beta]
            right = self.edge_data[2 * beta + 1]
            l = left.shape[0]
            parent_top = _circ_conv_batch(left, right)
            for i in range(l):
                parent_snaps.append((
                    prob_from_log(left[i]),
                    prob_from_log(right[i]),
                    prob_from_log(parent_top[i]),
                ))
            super().calc_parent(beta)

    for trial in range(num_cw):
        u_msg = np.zeros(N, dtype=np.int32)
        v_msg = np.zeros(N, dtype=np.int32)
        for i in range(1, N + 1):
            if i not in frozen_u:
                u_msg[i - 1] = np.random.randint(0, 2)
            if i not in frozen_v:
                v_msg[i - 1] = np.random.randint(0, 2)

        x_enc = polar_encode_batch(u_msg.reshape(1, -1))[0]
        y_enc = polar_encode_batch(v_msg.reshape(1, -1))[0]
        z = channel.sample_batch(x_enc, y_enc)
        log_W = build_log_W_leaf(z, channel)

        graph = InstrumentedCompGraph(n, log_W)
        u_hat = {}
        v_hat = {}
        i_u = 0
        i_v = 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; frozen_dict = frozen_u
            else:
                i_v += 1; i_t = i_v; frozen_dict = frozen_v

            leaf_edge = i_t + N - 1
            target_vertex = leaf_edge >> 1
            graph.step_to(target_vertex)

            temp = graph.edge_data[leaf_edge][0].copy()
            if leaf_edge & 1 == 0:
                graph.calc_left(target_vertex)
            else:
                graph.calc_right(target_vertex)

            top_down = graph.edge_data[leaf_edge][0]
            combined = _norm_prod_single(top_down, temp)

            if i_t in frozen_dict:
                bit = frozen_dict[i_t]
            else:
                if gamma == 0:
                    p0 = np.logaddexp(combined[0, 0], combined[0, 1])
                    p1 = np.logaddexp(combined[1, 0], combined[1, 1])
                    bit = 1 if p1 > p0 else 0
                else:
                    p0 = np.logaddexp(combined[0, 0], combined[1, 0])
                    p1 = np.logaddexp(combined[0, 1], combined[1, 1])
                    bit = 1 if p1 > p0 else 0

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            from polar.decoder import _LOG_HALF, _LOG_QUARTER
            new_leaf = np.full((2, 2), _NEG_INF, dtype=np.float64)
            u_val = u_hat.get(i_t)
            v_val = v_hat.get(i_t)
            if u_val is not None and v_val is not None:
                new_leaf[u_val, v_val] = 0.0
            elif u_val is not None:
                new_leaf[u_val, 0] = _LOG_HALF
                new_leaf[u_val, 1] = _LOG_HALF
            elif v_val is not None:
                new_leaf[0, v_val] = _LOG_HALF
                new_leaf[1, v_val] = _LOG_HALF
            else:
                new_leaf[:, :] = _LOG_QUARTER
            graph.edge_data[leaf_edge][0] = new_leaf

    return left_snaps, right_snaps, parent_snaps


# =============================================================================
#  Training
# =============================================================================

def train_snapshot(model, left_snaps, right_snaps, parent_snaps,
                   n_iters=10000, batch_size=256, lr=1e-3, label=""):
    """Train all NNs jointly on snapshot data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def to_tensors(snaps, n_in):
        arrays = [np.array([s[i] for s in snaps]) for i in range(n_in + 1)]
        return [torch.tensor(a, dtype=torch.float32) for a in arrays]

    left_t = to_tensors(left_snaps, 3)
    right_t = to_tensors(right_snaps, 3)
    parent_t = to_tensors(parent_snaps, 2)

    n_left = left_t[0].shape[0]
    n_right = right_t[0].shape[0]
    n_parent = parent_t[0].shape[0]

    log(f"  {label}Training: {n_left} L + {n_right} R + {n_parent} P = "
        f"{n_left+n_right+n_parent} total snapshots")
    log(f"  {label}Iters: {n_iters}, Batch: {batch_size}, LR: {lr}")

    t0 = time.time()
    final_loss = 0.0

    for it in range(n_iters):
        bs = max(1, batch_size // 3)

        idx_l = torch.randint(0, n_left, (bs,))
        idx_r = torch.randint(0, n_right, (bs,))
        idx_p = torch.randint(0, n_parent, (bs,))

        l_pred = model.calc_left(left_t[0][idx_l], left_t[1][idx_l], left_t[2][idx_l])
        l_target = left_t[3][idx_l]
        loss_l = -torch.sum(l_target * torch.log(l_pred.clamp(min=EPS)), dim=-1).mean()

        r_pred = model.calc_right(right_t[0][idx_r], right_t[1][idx_r], right_t[2][idx_r])
        r_target = right_t[3][idx_r]
        loss_r = -torch.sum(r_target * torch.log(r_pred.clamp(min=EPS)), dim=-1).mean()

        p_pred = model.calc_parent(parent_t[0][idx_p], parent_t[1][idx_p])
        p_target = parent_t[2][idx_p]
        loss_p = -torch.sum(p_target * torch.log(p_pred.clamp(min=EPS)), dim=-1).mean()

        loss = loss_l + loss_r + loss_p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss.item()

        if it % 2000 == 0 or it == n_iters - 1:
            elapsed = time.time() - t0
            log(f"    iter {it:5d}  loss={loss.item():.4f}  "
                f"(L={loss_l.item():.4f} R={loss_r.item():.4f} P={loss_p.item():.4f})  "
                f"[{elapsed:.1f}s]")

    train_time = time.time() - t0
    log(f"  {label}Training done in {train_time:.1f}s, final loss={final_loss:.4f}")
    return final_loss, train_time


# =============================================================================
#  Neural SC decode (from poc_snapshot_pure.py)
# =============================================================================

def neural_sc_decode(model, log_W, n, N, b, frozen_u, frozen_v):
    br = bit_reversal_perm(n)
    log_W_br = log_W[br].copy()
    for t in range(N):
        total = np.logaddexp(
            np.logaddexp(log_W_br[t, 0, 0], log_W_br[t, 0, 1]),
            np.logaddexp(log_W_br[t, 1, 0], log_W_br[t, 1, 1]))
        if np.isfinite(total):
            log_W_br[t] -= total

    root_probs = np.exp(log_W_br.reshape(N, 4))
    root_probs = root_probs / (root_probs.sum(axis=1, keepdims=True) + EPS)

    model.eval()
    with torch.no_grad():
        edge_data = [None] * (2 * N)
        edge_data[1] = root_probs
        uniform = np.ones(4) / 4.0
        for beta in range(2, 2 * N):
            level = beta.bit_length() - 1
            size = N >> level
            edge_data[beta] = np.tile(uniform, (size, 1))

        def do_calc_left(beta):
            parent = edge_data[beta]
            right = edge_data[2 * beta + 1]
            l = right.shape[0]
            pt = torch.tensor(parent[:l], dtype=torch.float32)
            pb = torch.tensor(parent[l:], dtype=torch.float32)
            rt = torch.tensor(right, dtype=torch.float32)
            edge_data[2 * beta] = model.calc_left(pt, pb, rt).numpy()

        def do_calc_right(beta):
            parent = edge_data[beta]
            left = edge_data[2 * beta]
            l = left.shape[0]
            pt = torch.tensor(parent[:l], dtype=torch.float32)
            pb = torch.tensor(parent[l:], dtype=torch.float32)
            lt = torch.tensor(left, dtype=torch.float32)
            edge_data[2 * beta + 1] = model.calc_right(pt, pb, lt).numpy()

        def do_calc_parent(beta):
            left = edge_data[2 * beta]
            right = edge_data[2 * beta + 1]
            lt = torch.tensor(left, dtype=torch.float32)
            rt = torch.tensor(right, dtype=torch.float32)
            parent_top = model.calc_parent(lt, rt).numpy()
            edge_data[beta] = np.concatenate([parent_top, right], axis=0)

        dec_head = 1

        def get_path(current, target):
            if current == target:
                return []
            path_up, path_down = [], []
            c, t = current, target
            while c != t:
                if c > t:
                    c = c >> 1
                    path_up.append(c)
                else:
                    path_down.append(t)
                    t = t >> 1
            path_down.reverse()
            return path_up + path_down

        def step_one(beta):
            nonlocal dec_head
            if beta == dec_head >> 1:
                do_calc_parent(dec_head)
                dec_head = beta
            elif beta >> 1 == dec_head:
                if beta & 1 == 0:
                    do_calc_left(dec_head)
                else:
                    do_calc_right(dec_head)
                dec_head = beta

        def step_to(target):
            nonlocal dec_head
            if dec_head == target:
                return
            for beta in get_path(dec_head, target):
                step_one(beta)

        u_hat, v_hat = {}, {}
        i_u, i_v = 0, 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; frozen_dict = frozen_u
            else:
                i_v += 1; i_t = i_v; frozen_dict = frozen_v

            leaf_edge = i_t + N - 1
            target_vertex = leaf_edge >> 1
            step_to(target_vertex)

            temp = edge_data[leaf_edge][0].copy()
            if leaf_edge & 1 == 0:
                do_calc_left(target_vertex)
            else:
                do_calc_right(target_vertex)

            top_down = edge_data[leaf_edge][0]
            combined = top_down * temp
            s = combined.sum()
            if s > EPS:
                combined = combined / s
            else:
                combined = np.ones(4) / 4.0

            if i_t in frozen_dict:
                bit = frozen_dict[i_t]
            else:
                if gamma == 0:
                    p0 = combined[0] + combined[1]
                    p1 = combined[2] + combined[3]
                    bit = 1 if p1 > p0 else 0
                else:
                    p0 = combined[0] + combined[2]
                    p1 = combined[1] + combined[3]
                    bit = 1 if p1 > p0 else 0

            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            u_val = u_hat.get(i_t)
            v_val = v_hat.get(i_t)
            new_prob = np.ones(4) * 0.25
            if u_val is not None and v_val is not None:
                new_prob = np.zeros(4)
                new_prob[u_val * 2 + v_val] = 1.0
            elif u_val is not None:
                new_prob = np.zeros(4)
                new_prob[u_val * 2 + 0] = 0.5
                new_prob[u_val * 2 + 1] = 0.5
            elif v_val is not None:
                new_prob = np.zeros(4)
                new_prob[0 * 2 + v_val] = 0.5
                new_prob[1 * 2 + v_val] = 0.5
            edge_data[leaf_edge] = new_prob.reshape(1, 4)

    u_dec = [u_hat.get(k, 0) for k in range(1, N + 1)]
    v_dec = [v_hat.get(k, 0) for k in range(1, N + 1)]
    return u_dec, v_dec


# =============================================================================
#  Evaluation
# =============================================================================

def eval_bler_neural(model, n, N, num_cw, sigma2, frozen_u, frozen_v, path_i):
    channel = GaussianMAC(sigma2=sigma2)
    b = make_path(N, path_i)
    info_u = set(range(1, N + 1)) - set(frozen_u.keys())
    info_v = set(range(1, N + 1)) - set(frozen_v.keys())
    block_errors = 0

    for trial in range(num_cw):
        u_msg = np.zeros(N, dtype=np.int32)
        v_msg = np.zeros(N, dtype=np.int32)
        for i in range(1, N + 1):
            if i not in frozen_u:
                u_msg[i - 1] = np.random.randint(0, 2)
            if i not in frozen_v:
                v_msg[i - 1] = np.random.randint(0, 2)

        x_enc = polar_encode_batch(u_msg.reshape(1, -1))[0]
        y_enc = polar_encode_batch(v_msg.reshape(1, -1))[0]
        z = channel.sample_batch(x_enc, y_enc)
        log_W = build_log_W_leaf(z, channel)

        u_dec, v_dec = neural_sc_decode(model, log_W, n, N, b, frozen_u, frozen_v)

        u_err = sum(1 for i in info_u if u_dec[i - 1] != u_msg[i - 1])
        v_err = sum(1 for i in info_v if v_dec[i - 1] != v_msg[i - 1])
        if u_err > 0 or v_err > 0:
            block_errors += 1

    return block_errors / num_cw


def eval_bler_sc(n, N, num_cw, sigma2, frozen_u, frozen_v, path_i):
    channel = GaussianMAC(sigma2=sigma2)
    b = make_path(N, path_i)
    info_u = set(range(1, N + 1)) - set(frozen_u.keys())
    info_v = set(range(1, N + 1)) - set(frozen_v.keys())
    block_errors = 0

    for trial in range(num_cw):
        u_msg = np.zeros(N, dtype=np.int32)
        v_msg = np.zeros(N, dtype=np.int32)
        for i in range(1, N + 1):
            if i not in frozen_u:
                u_msg[i - 1] = np.random.randint(0, 2)
            if i not in frozen_v:
                v_msg[i - 1] = np.random.randint(0, 2)

        x_enc = polar_encode_batch(u_msg.reshape(1, -1))[0]
        y_enc = polar_encode_batch(v_msg.reshape(1, -1))[0]
        z = channel.sample_batch(x_enc, y_enc)

        u_dec, v_dec = decode_single(N, z.tolist(), b, frozen_u, frozen_v, channel)

        u_err = sum(1 for i in info_u if u_dec[i - 1] != u_msg[i - 1])
        v_err = sum(1 for i in info_v if v_dec[i - 1] != v_msg[i - 1])
        if u_err > 0 or v_err > 0:
            block_errors += 1

    return block_errors / num_cw


# =============================================================================
#  Main: Curriculum training
# =============================================================================

def main():
    global log_fh
    log_fh = open(LOG_FILE, 'w')

    overall_t0 = time.time()
    TIME_BUDGET = 2 * 3600  # 2 hours

    log("=" * 70)
    log("Snapshot-Pure Neural MAC Decoder: Curriculum Scaling")
    log(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    SNR_dB = 6.0
    sigma2 = 10 ** (-SNR_dB / 10)
    log(f"\nConfig: SNR={SNR_dB}dB, sigma2={sigma2:.6f}, Class B")

    # =========================================================================
    #  Step 0: Train at N=8 (reproduce POC)
    # =========================================================================
    log("\n" + "=" * 70)
    log("STEP 0: N=8 — Initial training (reproduce POC)")
    log("=" * 70)

    n8, N8 = 3, 8
    # Use POC's original ku/kv for N=8
    Au, Av, frozen_u_8, frozen_v_8, _, _ = design_gmac(n8, 3, 4, sigma2)
    path_i_8 = N8 // 2
    ku_8 = N8 - len(frozen_u_8)
    kv_8 = N8 - len(frozen_v_8)
    log(f"  N={N8}, n={n8}, ku={ku_8}, kv={kv_8}, path_i={path_i_8}")

    # Generate snapshots at N=8
    np.random.seed(42)
    t0 = time.time()
    left_s, right_s, parent_s = generate_snapshots(
        n8, N8, 2000, sigma2, frozen_u_8, frozen_v_8, path_i_8)
    log(f"  Snapshot generation: {len(left_s)} L, {len(right_s)} R, "
        f"{len(parent_s)} P in {time.time()-t0:.1f}s")

    # Create and train model
    model = SnapshotModel(d=D_EMB, h=D_HIDDEN)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: {total_params} params, d={D_EMB}, hidden={D_HIDDEN}")

    train_snapshot(model, left_s, right_s, parent_s,
                   n_iters=15000, batch_size=384, lr=5e-4, label="[N=8] ")

    # Evaluate at N=8
    np.random.seed(123)
    t0 = time.time()
    neural_bler_8 = eval_bler_neural(model, n8, N8, 1000, sigma2,
                                      frozen_u_8, frozen_v_8, path_i_8)
    neural_time_8 = time.time() - t0

    np.random.seed(123)
    t0 = time.time()
    sc_bler_8 = eval_bler_sc(n8, N8, 1000, sigma2,
                              frozen_u_8, frozen_v_8, path_i_8)
    sc_time_8 = time.time() - t0

    ratio_8 = neural_bler_8 / max(sc_bler_8, 1e-10)
    log(f"\n  N=8 RESULT: Neural={neural_bler_8:.4f}, SC={sc_bler_8:.4f}, "
        f"ratio={ratio_8:.3f}")
    log(f"  Time: Neural={neural_time_8:.1f}s, SC={sc_time_8:.1f}s")

    # =========================================================================
    #  Curriculum: N = 16, 32, 64, 128, 256
    # =========================================================================

    results = [(8, neural_bler_8, sc_bler_8, ratio_8, neural_time_8)]

    curriculum = [
        # (N, num_snapshot_cw, num_eval_cw, train_iters)
        (16,  200, 1000, 12000),
        (32,  200, 1000, 12000),
        (64,  200, 500,  12000),
        (128, 100, 500,  15000),
        (256, 100, 500,  15000),
    ]

    for N_cur, num_snap_cw, num_eval_cw, train_iters in curriculum:
        elapsed = time.time() - overall_t0
        remaining = TIME_BUDGET - elapsed
        if remaining < 300:  # Less than 5 min left
            log(f"\n  TIME BUDGET: {elapsed/60:.1f}min elapsed, "
                f"{remaining/60:.1f}min remaining. Stopping curriculum.")
            break

        n_cur = int(np.log2(N_cur))
        path_i = N_cur // 2

        log(f"\n{'='*70}")
        log(f"CURRICULUM: N={N_cur} (n={n_cur})")
        log(f"  Time elapsed: {elapsed/60:.1f}min, remaining: {remaining/60:.1f}min")
        log(f"{'='*70}")

        # Get design
        frozen_u, frozen_v, ku, kv, path_i = get_design(n_cur, sigma2)

        # Generate snapshots
        np.random.seed(42 + N_cur)
        t0 = time.time()
        left_s, right_s, parent_s = generate_snapshots(
            n_cur, N_cur, num_snap_cw, sigma2, frozen_u, frozen_v, path_i)
        snap_time = time.time() - t0
        log(f"  Snapshots: {len(left_s)} L, {len(right_s)} R, "
            f"{len(parent_s)} P in {snap_time:.1f}s")

        if len(left_s) == 0 or len(right_s) == 0 or len(parent_s) == 0:
            log(f"  WARNING: No snapshots generated. Skipping N={N_cur}.")
            continue

        # Train (continue from previous model)
        train_snapshot(model, left_s, right_s, parent_s,
                       n_iters=train_iters, batch_size=384, lr=3e-4,
                       label=f"[N={N_cur}] ")

        # Evaluate neural
        np.random.seed(789 + N_cur)
        t0 = time.time()
        neural_bler = eval_bler_neural(model, n_cur, N_cur, num_eval_cw, sigma2,
                                        frozen_u, frozen_v, path_i)
        neural_time = time.time() - t0

        # Evaluate SC
        np.random.seed(789 + N_cur)
        t0 = time.time()
        sc_bler = eval_bler_sc(n_cur, N_cur, num_eval_cw, sigma2,
                                frozen_u, frozen_v, path_i)
        sc_time = time.time() - t0

        ratio = neural_bler / max(sc_bler, 1e-10)
        results.append((N_cur, neural_bler, sc_bler, ratio, neural_time))

        log(f"\n  N={N_cur} RESULT: Neural={neural_bler:.4f}, SC={sc_bler:.4f}, "
            f"ratio={ratio:.3f}")
        log(f"  Time: Neural={neural_time:.1f}s, SC={sc_time:.1f}s")

        # Save checkpoint
        ckpt_path = os.path.join(os.path.dirname(__file__), 'saved_models',
                                  f'snapshot_pure_N{N_cur}.pt')
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        log(f"  Saved checkpoint: {ckpt_path}")

    # =========================================================================
    #  Summary
    # =========================================================================
    total_time = time.time() - overall_t0
    log(f"\n{'='*70}")
    log("FINAL SUMMARY")
    log(f"{'='*70}")
    log(f"  Total time: {total_time/60:.1f} min")
    log(f"  Model: {total_params} params, d={D_EMB}, hidden={D_HIDDEN}")
    log(f"  SNR={SNR_dB}dB, Class B")
    log(f"\n  {'N':>5s}  {'Neural':>8s}  {'SC':>8s}  {'Ratio':>7s}  {'NeuralTime':>10s}")
    log(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*10}")
    for N_val, n_bler, s_bler, rat, n_time in results:
        log(f"  {N_val:5d}  {n_bler:8.4f}  {s_bler:8.4f}  {rat:7.3f}  {n_time:10.1f}s")
    log(f"{'='*70}")
    log(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log_fh.close()


if __name__ == "__main__":
    main()
