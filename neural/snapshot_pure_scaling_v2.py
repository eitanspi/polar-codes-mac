"""
snapshot_pure_scaling_v2.py
===========================
Corrected curriculum scaling of snapshot-pure neural MAC polar decoder.

Key fix: Uses EXACT ku/kv from verified SC reference points, and
design_from_file (MC designs) for N>=32 frozen sets.

SC_REF (verified BLER at SNR=6dB, Class B):
  N=8:   ku=3,  kv=4
  N=16:  ku=8,  kv=8
  N=32:  ku=15, kv=15,  sc_bler~0.046
  N=64:  ku=31, kv=31,  sc_bler~0.025
  N=128: ku=62, kv=62,  sc_bler~0.016
  N=256: ku=123,kv=123, sc_bler~0.005
"""

import sys, os, time, datetime, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from polar.channels import GaussianMAC
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.decoder import build_log_W_leaf, _norm_prod_single, decode_single
from polar.design import design_gmac, make_path
from polar.design_mc import design_from_file

torch.manual_seed(42)
np.random.seed(42)

D_EMB = 32
D_HIDDEN = 128
EPS = 1e-10

LOG_FILE = os.path.join(os.path.dirname(__file__), 'snapshot_pure_scaling_v2.log')
DESIGN_DIR = os.path.join(os.path.dirname(__file__), '..', 'designs')

# Verified SC reference points
SC_REF = {
    8:   {'ku': 3,   'kv': 4},
    16:  {'ku': 8,   'kv': 8},
    32:  {'ku': 15,  'kv': 15,  'sc_bler': 0.046},
    64:  {'ku': 31,  'kv': 31,  'sc_bler': 0.025},
    128: {'ku': 62,  'kv': 62,  'sc_bler': 0.016},
    256: {'ku': 123, 'kv': 123, 'sc_bler': 0.005},
}

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
#  Design helper: CORRECT frozen sets
# =============================================================================

def get_design_correct(N, sigma2):
    """
    Get frozen sets using the EXACT ku/kv from SC_REF.
    For N>=32: use MC design from designs/gmac_B_n{n}_snr6dB.npz
    For N=8,16: use design_gmac (GA design)
    """
    n = int(math.log2(N))
    ref = SC_REF[N]
    ku, kv = ref['ku'], ref['kv']
    path_i = N // 2

    if N >= 32:
        mc_path = os.path.join(DESIGN_DIR, f'gmac_B_n{n}_snr6dB.npz')
        if not os.path.exists(mc_path):
            raise FileNotFoundError(f"MC design file not found: {mc_path}")
        Au, Av, frozen_u, frozen_v, _, _, _ = design_from_file(mc_path, n, ku, kv)
        log(f"  Design: MC (design_from_file), N={N}, ku={ku}, kv={kv}, "
            f"rate=({ku/N:.3f}, {kv/N:.3f})")
    else:
        Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)
        log(f"  Design: GA (design_gmac), N={N}, ku={ku}, kv={kv}, "
            f"rate=({ku/N:.3f}, {kv/N:.3f})")

    # Verify counts
    actual_ku = N - len(frozen_u)
    actual_kv = N - len(frozen_v)
    assert actual_ku == ku, f"ku mismatch: expected {ku}, got {actual_ku}"
    assert actual_kv == kv, f"kv mismatch: expected {kv}, got {actual_kv}"

    return frozen_u, frozen_v, ku, kv, path_i


# =============================================================================
#  Snapshot generation
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

        if it % 3000 == 0 or it == n_iters - 1:
            elapsed = time.time() - t0
            log(f"    iter {it:5d}  loss={loss.item():.4f}  "
                f"(L={loss_l.item():.4f} R={loss_r.item():.4f} P={loss_p.item():.4f})  "
                f"[{elapsed:.1f}s]")

    train_time = time.time() - t0
    log(f"  {label}Training done in {train_time:.1f}s, final loss={final_loss:.4f}")
    return final_loss, train_time


# =============================================================================
#  Neural SC decode
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
#  Main: Curriculum training with CORRECT rates
# =============================================================================

def main():
    global log_fh
    log_fh = open(LOG_FILE, 'w')

    overall_t0 = time.time()
    TIME_BUDGET = 2 * 3600  # 2 hours

    log("=" * 70)
    log("Snapshot-Pure Neural MAC Decoder: Curriculum Scaling v2")
    log("  (CORRECTED rates and frozen sets)")
    log(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    SNR_dB = 6.0
    sigma2 = 10 ** (-SNR_dB / 10)
    log(f"\nConfig: SNR={SNR_dB}dB, sigma2={sigma2:.6f}, Class B")
    log(f"\nSC_REF rate points:")
    for N_val in sorted(SC_REF.keys()):
        ref = SC_REF[N_val]
        sc_str = f", sc_bler~{ref['sc_bler']}" if 'sc_bler' in ref else ""
        log(f"  N={N_val:>4d}: ku={ref['ku']:>3d}, kv={ref['kv']:>3d}, "
            f"rate=({ref['ku']/N_val:.3f}, {ref['kv']/N_val:.3f}){sc_str}")

    # =========================================================================
    #  Step 0: Train at N=8
    # =========================================================================
    log(f"\n{'='*70}")
    log("STEP 0: N=8 -- Initial training")
    log(f"{'='*70}")

    n8, N8 = 3, 8
    frozen_u_8, frozen_v_8, ku_8, kv_8, path_i_8 = get_design_correct(N8, sigma2)
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
    sc_bler_8 = eval_bler_sc(n8, N8, 500, sigma2,
                              frozen_u_8, frozen_v_8, path_i_8)
    sc_time_8 = time.time() - t0

    np.random.seed(123)
    t0 = time.time()
    neural_bler_8 = eval_bler_neural(model, n8, N8, 500, sigma2,
                                      frozen_u_8, frozen_v_8, path_i_8)
    neural_time_8 = time.time() - t0

    ratio_8 = neural_bler_8 / max(sc_bler_8, 1e-10)
    log(f"\n  N=8 RESULT: SC={sc_bler_8:.4f}, Neural={neural_bler_8:.4f}, "
        f"ratio={ratio_8:.3f}")
    log(f"  Time: SC={sc_time_8:.1f}s, Neural={neural_time_8:.1f}s")

    # Save N=8 checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), 'saved_models',
                              'snapshot_pure_v2_N8.pt')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    # =========================================================================
    #  Curriculum: N = 16, 32, 64, 128, 256
    # =========================================================================

    results = [(8, ku_8, kv_8, neural_bler_8, sc_bler_8, ratio_8)]

    curriculum = [
        # (N, num_snapshot_cw, num_eval_cw, train_iters)
        (16,  200, 500,  15000),
        (32,  200, 500,  15000),
        (64,  200, 500,  15000),
        (128, 200, 500,  15000),
        (256, 100, 500,  15000),
    ]

    for N_cur, num_snap_cw, num_eval_cw, train_iters in curriculum:
        elapsed = time.time() - overall_t0
        remaining = TIME_BUDGET - elapsed
        if remaining < 300:
            log(f"\n  TIME BUDGET: {elapsed/60:.1f}min elapsed, "
                f"{remaining/60:.1f}min remaining. Stopping.")
            break

        n_cur = int(math.log2(N_cur))

        log(f"\n{'='*70}")
        log(f"CURRICULUM: N={N_cur} (n={n_cur})")
        log(f"  Time elapsed: {elapsed/60:.1f}min, remaining: {remaining/60:.1f}min")
        log(f"{'='*70}")

        # Get CORRECT design
        frozen_u, frozen_v, ku, kv, path_i = get_design_correct(N_cur, sigma2)

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

        # Evaluate SC FIRST (to validate design)
        np.random.seed(789 + N_cur)
        t0 = time.time()
        sc_bler = eval_bler_sc(n_cur, N_cur, num_eval_cw, sigma2,
                                frozen_u, frozen_v, path_i)
        sc_time = time.time() - t0

        # Check SC BLER against reference
        ref_sc = SC_REF[N_cur].get('sc_bler')
        if ref_sc is not None:
            log(f"  SC BLER check: got {sc_bler:.4f}, expected ~{ref_sc:.3f}")

        # Evaluate neural
        np.random.seed(789 + N_cur)
        t0 = time.time()
        neural_bler = eval_bler_neural(model, n_cur, N_cur, num_eval_cw, sigma2,
                                        frozen_u, frozen_v, path_i)
        neural_time = time.time() - t0

        ratio = neural_bler / max(sc_bler, 1e-10)
        results.append((N_cur, ku, kv, neural_bler, sc_bler, ratio))

        log(f"\n  N={N_cur} RESULT: SC={sc_bler:.4f}, Neural={neural_bler:.4f}, "
            f"ratio={ratio:.3f}")
        log(f"  ku={ku}, kv={kv}, rate=({ku/N_cur:.3f}, {kv/N_cur:.3f})")
        log(f"  Time: SC={sc_time:.1f}s, Neural={neural_time:.1f}s")

        # Save checkpoint
        ckpt_path = os.path.join(os.path.dirname(__file__), 'saved_models',
                                  f'snapshot_pure_v2_N{N_cur}.pt')
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
    log(f"  SNR={SNR_dB}dB, Class B, sigma2={sigma2:.6f}")
    log(f"\n  {'N':>5s}  {'ku':>4s}  {'kv':>4s}  {'Rate':>10s}  {'Neural':>8s}  "
        f"{'SC':>8s}  {'Ratio':>7s}")
    log(f"  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*7}")
    for N_val, ku, kv, n_bler, s_bler, rat in results:
        log(f"  {N_val:5d}  {ku:4d}  {kv:4d}  ({ku/N_val:.3f},{kv/N_val:.3f})  "
            f"{n_bler:8.4f}  {s_bler:8.4f}  {rat:7.3f}")
    log(f"{'='*70}")
    log(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log_fh.close()


if __name__ == "__main__":
    main()
