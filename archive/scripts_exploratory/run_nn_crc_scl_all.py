#!/usr/bin/env python3
"""
run_nn_crc_scl_all.py — NN-CRC-SCL L=4 evaluation for ALL neural checkpoints.

Channels: GMAC Class B, BEMAC Class B, ABNMAC Class B
For each: NN-SCL L=4, NN-CA-SCL L=4 (with CRC-8)

Results saved to results/crc_scl_sweep/nn_crc_scl_{channel}_{class}.json
"""

import os, sys, json, time, math
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.design import make_path
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder
from neural.neural_scl import SimpleMLP_Gmac, NeuralSCLDecoder
from scripts.eval_crc_aided_nn_scl import CRCAidedNeuralSCLDecoder as GmacCRCSCLDecoder

# ─── CRC-8 ────────────────────────────────────────────────────────────────

CRC_POLY = 0x107
CRC_BITS = 8

def compute_crc8(message_bits):
    msg = int(''.join(str(int(b)) for b in message_bits), 2) << CRC_BITS
    for i in range(len(message_bits)):
        if msg & (1 << (len(message_bits) + CRC_BITS - 1 - i)):
            msg ^= (CRC_POLY << (len(message_bits) - 1 - i))
    crc = msg & ((1 << CRC_BITS) - 1)
    return [(crc >> (CRC_BITS - 1 - i)) & 1 for i in range(CRC_BITS)]


# ─── BEMAC / ABNMAC wrapper ──────────────────────────────────────────────

class DiscreteModelWrapper(torch.nn.Module):
    """Wraps PureNeuralCompGraphDecoder (discrete channel) to look like SimpleMLP_Gmac."""
    def __init__(self, tree_model):
        super().__init__()
        self.tree = tree_model
        self.d = tree_model.d
        # z_encoder is the embedding_z from the tree
        self.z_encoder = tree_model.embedding_z  # nn.Embedding

    def forward(self, z, b, fu, fv, u_true=None, v_true=None):
        n = z.shape[1].bit_length() - 1
        br = torch.from_numpy(bit_reversal_perm(n)).long().to(z.device)
        root = self.z_encoder(z.long())[:, br]
        return self.tree(z=None, b=b, frozen_u=fu, frozen_v=fv,
                         u_true=u_true, v_true=v_true, root_emb=root)


class DiscreteNeuralSCLDecoder(NeuralSCLDecoder):
    """SCL decoder for discrete-channel models (BEMAC, ABNMAC)."""

    @torch.no_grad()
    def decode(self, z_single, b, frozen_u, frozen_v):
        self.model.eval()
        device = z_single.device
        N = z_single.shape[0]
        n = N.bit_length() - 1
        assert (1 << n) == N
        d = self.tree.d
        L = self.L

        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        # For discrete: z is integer, use embedding
        root = self.model.z_encoder(z_single.long().unsqueeze(0))[:, br]  # (1, N, d)

        no_info = self.tree.no_info_emb.unsqueeze(0).unsqueeze(0)

        def _make_edge_data():
            ed = [None] * (2 * N)
            ed[1] = root.clone()
            for beta in range(2, 2 * N):
                level = beta.bit_length() - 1
                size = N >> level
                ed[beta] = no_info.expand(1, size, d).clone()
            return ed

        def _clone_edge_data(ed):
            return [e.clone() if e is not None else None for e in ed]

        paths = [{
            'ed': _make_edge_data(),
            'dh': 1,
            'uh': {},
            'vh': {},
            'pm': 0.0,
        }]

        i_u, i_v = 0, 0

        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = frozen_v

            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1
            is_frozen = i_t in fdict

            path_logits = []
            for p in paths:
                ed = p['ed']
                dh = p['dh']
                dh = self._step_to_path(dh, target_vtx, ed)
                p['dh'] = dh
                temp = ed[leaf_edge][:, 0].clone()
                if leaf_edge & 1 == 0:
                    self.tree._neural_calc_left(target_vtx, ed)
                else:
                    self.tree._neural_calc_right(target_vtx, ed)
                top_down = ed[leaf_edge][:, 0]
                if self.tree.use_combine_nn:
                    combined = self.tree.combine_nn(
                        torch.cat([top_down, temp], dim=-1))
                else:
                    combined = top_down + temp
                logits = self.tree.emb2logits(combined)
                log_probs = F.log_softmax(logits, dim=-1)
                path_logits.append(log_probs[0])

            if is_frozen:
                frozen_bit = fdict[i_t]
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]
                    if gamma == 0:
                        p['uh'][i_t] = torch.tensor([float(frozen_bit)], device=device)
                        u_val = frozen_bit
                        if i_t in frozen_v:
                            v_val = frozen_v[i_t]
                            p['pm'] += lp[u_val * 2 + v_val].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[u_val*2:u_val*2+2], dim=0).item()
                    else:
                        p['vh'][i_t] = torch.tensor([float(frozen_bit)], device=device)
                        v_val = frozen_bit
                        if i_t in frozen_u:
                            u_val = frozen_u[i_t]
                            p['pm'] += lp[u_val * 2 + v_val].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[[v_val, v_val + 2]], dim=0).item()
                    new_emb = self.tree._make_leaf_emb(
                        p['uh'].get(i_t), p['vh'].get(i_t), 1, device)
                    p['ed'][leaf_edge] = new_emb.unsqueeze(1)
            else:
                candidates = []
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]
                    if gamma == 0:
                        options = [
                            (0.0, torch.logsumexp(lp[:2], dim=0).item()),
                            (1.0, torch.logsumexp(lp[2:], dim=0).item()),
                        ]
                    else:
                        options = [
                            (0.0, torch.logsumexp(lp[[0, 2]], dim=0).item()),
                            (1.0, torch.logsumexp(lp[[1, 3]], dim=0).item()),
                        ]
                    for bit_val, log_p in options:
                        candidates.append((p['pm'] + log_p, pidx, bit_val))
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:L]
                new_paths = []
                for new_pm, pidx, bit_val in candidates:
                    old_p = paths[pidx]
                    new_p = {
                        'ed': _clone_edge_data(old_p['ed']),
                        'dh': old_p['dh'],
                        'uh': dict(old_p['uh']),
                        'vh': dict(old_p['vh']),
                        'pm': new_pm,
                    }
                    bit_tensor = torch.tensor([bit_val], device=device)
                    if gamma == 0:
                        new_p['uh'][i_t] = bit_tensor
                    else:
                        new_p['vh'][i_t] = bit_tensor
                    new_emb = self.tree._make_leaf_emb(
                        new_p['uh'].get(i_t), new_p['vh'].get(i_t), 1, device)
                    new_p['ed'][leaf_edge] = new_emb.unsqueeze(1)
                    new_paths.append(new_p)
                paths = new_paths

        best = max(paths, key=lambda p: p['pm'])
        u_hat = {k: int(v[0].item()) for k, v in best['uh'].items()}
        v_hat = {k: int(v[0].item()) for k, v in best['vh'].items()}
        return u_hat, v_hat

    def decode_all_candidates(self, z_single, b, frozen_u, frozen_v):
        """Return all L candidate paths sorted by metric (for CRC selection)."""
        self.model.eval()
        device = z_single.device
        N = z_single.shape[0]
        n = N.bit_length() - 1
        d = self.tree.d
        L = self.L

        br = torch.from_numpy(bit_reversal_perm(n)).long().to(device)
        root = self.model.z_encoder(z_single.long().unsqueeze(0))[:, br]

        no_info = self.tree.no_info_emb.unsqueeze(0).unsqueeze(0)

        def _make_edge_data():
            ed = [None] * (2 * N)
            ed[1] = root.clone()
            for beta in range(2, 2 * N):
                level = beta.bit_length() - 1
                size = N >> level
                ed[beta] = no_info.expand(1, size, d).clone()
            return ed

        def _clone_edge_data(ed):
            return [e.clone() if e is not None else None for e in ed]

        paths = [{'ed': _make_edge_data(), 'dh': 1, 'uh': {}, 'vh': {}, 'pm': 0.0}]

        i_u, i_v = 0, 0
        for step in range(2 * N):
            gamma = b[step]
            if gamma == 0:
                i_u += 1; i_t = i_u; fdict = frozen_u
            else:
                i_v += 1; i_t = i_v; fdict = frozen_v
            leaf_edge = i_t + N - 1
            target_vtx = leaf_edge >> 1
            is_frozen = i_t in fdict

            path_logits = []
            for p in paths:
                ed = p['ed']
                dh = self._step_to_path(p['dh'], target_vtx, ed)
                p['dh'] = dh
                temp = ed[leaf_edge][:, 0].clone()
                if leaf_edge & 1 == 0:
                    self.tree._neural_calc_left(target_vtx, ed)
                else:
                    self.tree._neural_calc_right(target_vtx, ed)
                top_down = ed[leaf_edge][:, 0]
                if self.tree.use_combine_nn:
                    combined = self.tree.combine_nn(torch.cat([top_down, temp], dim=-1))
                else:
                    combined = top_down + temp
                logits = self.tree.emb2logits(combined)
                log_probs = F.log_softmax(logits, dim=-1)
                path_logits.append(log_probs[0])

            if is_frozen:
                frozen_bit = fdict[i_t]
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]
                    if gamma == 0:
                        p['uh'][i_t] = frozen_bit
                        u_val = frozen_bit
                        if i_t in frozen_v:
                            p['pm'] += lp[u_val * 2 + frozen_v[i_t]].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[u_val*2:u_val*2+2], dim=0).item()
                    else:
                        p['vh'][i_t] = frozen_bit
                        v_val = frozen_bit
                        if i_t in frozen_u:
                            p['pm'] += lp[frozen_u[i_t] * 2 + v_val].item()
                        else:
                            p['pm'] += torch.logsumexp(lp[[v_val, v_val + 2]], dim=0).item()
                    u_t = torch.tensor([float(p['uh'][i_t])]) if i_t in p['uh'] else None
                    v_t = torch.tensor([float(p['vh'][i_t])]) if i_t in p['vh'] else None
                    new_emb = self.tree._make_leaf_emb(u_t, v_t, 1, device)
                    p['ed'][leaf_edge] = new_emb.unsqueeze(1)
            else:
                candidates = []
                for pidx, p in enumerate(paths):
                    lp = path_logits[pidx]
                    if gamma == 0:
                        opts = [(0, torch.logsumexp(lp[:2], dim=0).item()),
                                (1, torch.logsumexp(lp[2:], dim=0).item())]
                    else:
                        opts = [(0, torch.logsumexp(lp[[0,2]], dim=0).item()),
                                (1, torch.logsumexp(lp[[1,3]], dim=0).item())]
                    for bv, lp_val in opts:
                        candidates.append((p['pm'] + lp_val, pidx, bv))
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:L]
                new_paths = []
                for new_pm, pidx, bit_val in candidates:
                    old_p = paths[pidx]
                    new_p = {
                        'ed': _clone_edge_data(old_p['ed']),
                        'dh': old_p['dh'],
                        'uh': dict(old_p['uh']),
                        'vh': dict(old_p['vh']),
                        'pm': new_pm,
                    }
                    if gamma == 0:
                        new_p['uh'][i_t] = bit_val
                    else:
                        new_p['vh'][i_t] = bit_val
                    u_t = torch.tensor([float(new_p['uh'][i_t])]) if i_t in new_p['uh'] else None
                    v_t = torch.tensor([float(new_p['vh'][i_t])]) if i_t in new_p['vh'] else None
                    new_emb = self.tree._make_leaf_emb(u_t, v_t, 1, device)
                    new_p['ed'][leaf_edge] = new_emb.unsqueeze(1)
                    new_paths.append(new_p)
                paths = new_paths

        paths.sort(key=lambda p: p['pm'], reverse=True)
        return paths


# ─── CRC-Aided SCL (wraps either Gmac or Discrete SCL) ──────────────────

class CRCAidedSCL:
    """CRC-aided SCL that works with any decoder that has decode_all_candidates."""
    def __init__(self, decoder, Au, crc_positions):
        self.decoder = decoder
        self.Au = Au
        self.crc_positions = crc_positions
        self.msg_positions = [p for p in Au if p not in crc_positions]

    def decode_crc(self, z_single, b, fu, fv):
        if hasattr(self.decoder, 'decode_all_candidates'):
            candidates = self.decoder.decode_all_candidates(z_single, b, fu, fv)
        elif hasattr(self.decoder, '_decode_all_candidates'):
            candidates = self.decoder._decode_all_candidates(z_single, b, fu, fv)
        else:
            raise RuntimeError("Decoder has no candidate extraction method")

        for cand in candidates:
            uh = cand['uh']
            msg_bits = [uh.get(p, 0) for p in self.msg_positions]
            crc_decoded = [uh.get(p, 0) for p in self.crc_positions]
            # Convert tensor values if needed
            msg_bits = [int(b.item()) if hasattr(b, 'item') else int(b) for b in msg_bits]
            crc_decoded = [int(b.item()) if hasattr(b, 'item') else int(b) for b in crc_decoded]
            crc_expected = compute_crc8(msg_bits)
            if crc_decoded == crc_expected:
                uh_int = {k: (int(v.item()) if hasattr(v, 'item') else int(v)) for k, v in uh.items()}
                vh_int = {k: (int(v.item()) if hasattr(v, 'item') else int(v)) for k, v in cand['vh'].items()}
                return uh_int, vh_int

        # No CRC pass -- return best
        best = candidates[0]
        uh_int = {k: (int(v.item()) if hasattr(v, 'item') else int(v)) for k, v in best['uh'].items()}
        vh_int = {k: (int(v.item()) if hasattr(v, 'item') else int(v)) for k, v in best['vh'].items()}
        return uh_int, vh_int


# ─── Design loading ───────────────────────────────────────────────────────

def load_design(channel_type, path_class, N, ku, kv):
    """Load frozen sets from design files."""
    n = int(math.log2(N))
    if channel_type == 'gmac':
        dp = os.path.join(BASE, 'designs', f'gmac_B_n{n}_snr6dB.npz')
    elif channel_type == 'bemac':
        dp = os.path.join(BASE, 'designs', f'bemac_B_n{n}.npz')
    elif channel_type == 'abnmac':
        dp = os.path.join(BASE, 'designs', f'abnmac_B_n{n}.npz')
    else:
        raise ValueError(f"Unknown channel: {channel_type}")

    if not os.path.exists(dp):
        print(f"  Design file not found: {dp}")
        return None
    d = np.load(dp)
    su = np.argsort(d['u_error_rates'])
    sv = np.argsort(d['v_error_rates'])
    Au = sorted([int(i+1) for i in su[:ku]])
    Av = sorted([int(i+1) for i in sv[:kv]])
    all_pos = set(range(1, N + 1))
    fu = {p: 0 for p in sorted(all_pos - set(Au))}
    fv = {p: 0 for p in sorted(all_pos - set(Av))}
    return Au, Av, fu, fv


# ─── Model loading ────────────────────────────────────────────────────────

def load_gmac_model(ckpt_path, d=16, hidden=64, n_layers=2, z_hidden=32):
    """Load GMAC SimpleMLP_Gmac model."""
    model = SimpleMLP_Gmac(d=d, hidden=hidden, n_layers=n_layers, z_hidden=z_hidden)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    fixed = {k.replace('z_enc.', 'z_encoder.'): v for k, v in sd.items()}
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model


def load_discrete_model(ckpt_path, vocab_size=3, d=16, hidden=64, n_layers=2):
    """Load BEMAC/ABNMAC PureNeuralCompGraphDecoder model."""
    tree = PureNeuralCompGraphDecoder(d=d, hidden=hidden, n_layers=n_layers,
                                       vocab_size=vocab_size)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    tree.load_state_dict(sd, strict=False)
    tree.eval()
    wrapper = DiscreteModelWrapper(tree)
    wrapper.eval()
    return wrapper


# ─── Channel factories ───────────────────────────────────────────────────

def make_channel(channel_type):
    if channel_type == 'gmac':
        sigma2 = 10 ** (-6.0 / 10)
        return GaussianMAC(sigma2=sigma2)
    elif channel_type == 'bemac':
        from polar.channels import BEMAC
        return BEMAC()
    elif channel_type == 'abnmac':
        from polar.channels import ABNMAC
        return ABNMAC()
    else:
        raise ValueError(f"Unknown channel: {channel_type}")


# ─── Evaluation ───────────────────────────────────────────────────────────

def eval_nn_scl(model, decoder_cls, channel, N, b, Au, Av, fu, fv,
                n_cw, L=4, use_crc=False, crc_positions=None, channel_type='gmac'):
    """
    Evaluate NN-SCL or NN-CA-SCL.
    Returns (bler, n_errors, n_cw, time_s).
    """
    if channel_type == 'gmac':
        scl_decoder = NeuralSCLDecoder(model, L=L)
    else:
        scl_decoder = DiscreteNeuralSCLDecoder(model, L=L)

    if use_crc and crc_positions:
        if channel_type == 'gmac':
            crc_aided = GmacCRCSCLDecoder(model, L=L)
        else:
            crc_aided = CRCAidedSCL(scl_decoder, Au, crc_positions)

    errs = 0
    rng = np.random.default_rng(42)
    t0 = time.time()

    with torch.no_grad():
        for i in range(n_cw):
            uf = np.zeros(N, dtype=int)
            vf = np.zeros(N, dtype=int)
            for p in Au: uf[p-1] = rng.integers(0, 2)
            for p in Av: vf[p-1] = rng.integers(0, 2)

            # Set CRC bits if using CRC
            if use_crc and crc_positions:
                msg_positions = [p for p in Au if p not in crc_positions]
                msg_bits = [uf[p-1] for p in msg_positions]
                crc_vals = compute_crc8(msg_bits)
                for cp, cv in zip(crc_positions, crc_vals):
                    uf[cp-1] = cv

            xf = polar_encode_batch(uf.reshape(1, N))
            yf = polar_encode_batch(vf.reshape(1, N))

            if channel_type == 'gmac':
                zf = channel.sample_batch(xf, yf).astype(np.float32)
                z_t = torch.from_numpy(zf[0]).float()
            elif channel_type == 'bemac':
                zf = channel.sample_batch(xf, yf)
                z_t = torch.from_numpy(zf[0]).long()
            elif channel_type == 'abnmac':
                zf = channel.sample_batch(xf, yf)
                # ABNMAC: object array of tuples -> encode as 2*zx + zy
                z_enc = np.empty(N, dtype=np.int64)
                for j in range(N):
                    zx, zy = zf[0][j]
                    z_enc[j] = 2 * int(zx) + int(zy)
                z_t = torch.from_numpy(z_enc).long()

            if use_crc and crc_positions:
                if channel_type == 'gmac':
                    uh, vh = crc_aided.decode_crc(z_t, b, fu, fv, Au, crc_positions)
                else:
                    uh, vh = crc_aided.decode_crc(z_t, b, fu, fv)
            else:
                uh, vh = scl_decoder.decode(z_t, b, fu, fv)

            # Check errors
            # uh/vh may have int or tensor values
            def get_val(d, p):
                v = d.get(p, 0)
                if hasattr(v, 'item'):
                    return int(v.item())
                return int(v)

            e = any(get_val(uh, p) != uf[p-1] for p in Au) or \
                any(get_val(vh, p) != vf[p-1] for p in Av)
            if e:
                errs += 1

            if (i+1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    [{i+1}/{n_cw}] errs={errs} BLER={errs/(i+1):.4f} "
                      f"[{elapsed:.0f}s, {elapsed/(i+1):.1f}s/cw]", flush=True)

    t_total = time.time() - t0
    bler = errs / n_cw
    return bler, errs, n_cw, t_total


# ─── Job definitions ──────────────────────────────────────────────────────

GMAC_CLASSB_JOBS = [
    {'N': 32,  'ku': 15, 'kv': 15, 'ckpt': 'ncg_gmac_mlp_N32.pt',  'n_cw': 1500},
    {'N': 64,  'ku': 31, 'kv': 31, 'ckpt': 'ncg_gmac_mlp_N64.pt',  'n_cw': 1000},
    {'N': 128, 'ku': 62, 'kv': 62, 'ckpt': 'ncg_gmac_mlp_N128.pt', 'n_cw': 500},
    {'N': 256, 'ku': 123,'kv': 123,'ckpt': 'ncg_gmac_mlp_N256.pt', 'n_cw': 300},
]

BEMAC_CLASSB_JOBS = [
    {'N': 32,  'ku': 16, 'kv': 22, 'ckpt': 'ncg_pure_neural_N32.pt',  'n_cw': 1500, 'vocab': 3},
    {'N': 64,  'ku': 32, 'kv': 44, 'ckpt': 'ncg_pure_neural_N64.pt',  'n_cw': 1000, 'vocab': 3},
    {'N': 128, 'ku': 64, 'kv': 89, 'ckpt': 'ncg_pure_neural_N128.pt', 'n_cw': 500,  'vocab': 3},
]

ABNMAC_CLASSB_JOBS = [
    {'N': 32,  'ku': 10, 'kv': 10, 'ckpt': 'ncg_abnmac_classB_N32_best.pt',  'n_cw': 1500, 'vocab': 4},
    {'N': 64,  'ku': 22, 'kv': 22, 'ckpt': 'ncg_abnmac_classB_N64_best.pt',  'n_cw': 1000, 'vocab': 4},
]


def run_channel_jobs(channel_type, path_class, jobs, out_name):
    """Run all jobs for a channel/class combo."""
    out_path = os.path.join(BASE, 'results', 'crc_scl_sweep', out_name)

    # Check existing
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        print(f"\n  Existing results in {out_name}: N={list(existing.keys())}")

    channel = make_channel(channel_type)
    results = dict(existing)

    for job in jobs:
        N = job['N']
        ku, kv = job['ku'], job['kv']
        n_cw = job['n_cw']

        key = str(N)
        if key in results and 'nn_crc_scl_l4_bler' in results[key]:
            print(f"\n  {channel_type} N={N}: Already done (BLER={results[key]['nn_crc_scl_l4_bler']}), skipping")
            continue

        ckpt = os.path.join(BASE, 'saved_models', job['ckpt'])
        if not os.path.exists(ckpt):
            print(f"\n  {channel_type} N={N}: Checkpoint not found: {ckpt}")
            continue

        # Load design
        design = load_design(channel_type, path_class, N, ku, kv)
        if design is None:
            continue
        Au, Av, fu, fv = design
        b = make_path(N, N // 2) if path_class == 'B' else make_path(N, N)

        # Load model
        try:
            if channel_type == 'gmac':
                model = load_gmac_model(ckpt)
            else:
                model = load_discrete_model(ckpt, vocab_size=job.get('vocab', 3))
        except Exception as e:
            print(f"\n  {channel_type} N={N}: Failed to load model: {e}")
            continue

        # CRC positions: last CRC_BITS of Au
        crc_positions = Au[-CRC_BITS:] if ku > CRC_BITS else []

        print(f"\n{'='*70}")
        print(f"  {channel_type.upper()} Class {path_class} N={N} ku={ku} kv={kv}")
        print(f"  Checkpoint: {job['ckpt']}")
        print(f"  CRC positions: {crc_positions[:3]}... ({len(crc_positions)} total)")
        print(f"{'='*70}")

        entry = {'N': N, 'ku': ku, 'kv': kv, 'channel': channel_type, 'class': path_class}

        # NN-SCL L=4 (no CRC)
        print(f"\n  NN-SCL L=4 ({n_cw} CW)...", flush=True)
        bler_scl, errs_scl, _, t_scl = eval_nn_scl(
            model, None, channel, N, b, Au, Av, fu, fv,
            n_cw, L=4, use_crc=False, channel_type=channel_type)
        print(f"  -> NN-SCL L=4: BLER={bler_scl:.4f} ({errs_scl}/{n_cw}) [{t_scl:.0f}s]")
        entry['nn_scl_l4_bler'] = bler_scl
        entry['nn_scl_l4_errs'] = errs_scl
        entry['nn_scl_l4_cw'] = n_cw
        entry['nn_scl_l4_time'] = round(t_scl, 1)

        # NN-CA-SCL L=4 (with CRC)
        if len(crc_positions) == CRC_BITS:
            print(f"\n  NN-CA-SCL L=4 ({n_cw} CW)...", flush=True)
            bler_crc, errs_crc, _, t_crc = eval_nn_scl(
                model, None, channel, N, b, Au, Av, fu, fv,
                n_cw, L=4, use_crc=True, crc_positions=crc_positions,
                channel_type=channel_type)
            print(f"  -> NN-CA-SCL L=4: BLER={bler_crc:.4f} ({errs_crc}/{n_cw}) [{t_crc:.0f}s]")
            entry['nn_crc_scl_l4_bler'] = bler_crc
            entry['nn_crc_scl_l4_errs'] = errs_crc
            entry['nn_crc_scl_l4_cw'] = n_cw
            entry['nn_crc_scl_l4_time'] = round(t_crc, 1)
        else:
            print(f"  Skipping CRC (ku={ku} <= CRC_BITS={CRC_BITS})")
            entry['nn_crc_scl_l4_bler'] = None
            entry['nn_crc_scl_l4_note'] = f'ku={ku} too small for CRC-8'

        results[key] = entry

        # Save after each N
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {out_path}")

    return results


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  NN-CRC-SCL L=4 Evaluation — All Channels")
    print("=" * 70)

    all_results = {}

    # GMAC Class B
    print("\n\n" + "#" * 70)
    print("  GMAC Class B")
    print("#" * 70)
    r = run_channel_jobs('gmac', 'B', GMAC_CLASSB_JOBS,
                         'nn_crc_scl_gmac_B.json')
    all_results['gmac_B'] = r

    # BEMAC Class B
    print("\n\n" + "#" * 70)
    print("  BEMAC Class B")
    print("#" * 70)
    r = run_channel_jobs('bemac', 'B', BEMAC_CLASSB_JOBS,
                         'nn_crc_scl_bemac_B.json')
    all_results['bemac_B'] = r

    # ABNMAC Class B
    print("\n\n" + "#" * 70)
    print("  ABNMAC Class B")
    print("#" * 70)
    r = run_channel_jobs('abnmac', 'B', ABNMAC_CLASSB_JOBS,
                         'nn_crc_scl_abnmac_B.json')
    all_results['abnmac_B'] = r

    # Summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for ch_key, res in all_results.items():
        print(f"\n  {ch_key}:")
        for n_key, entry in sorted(res.items()):
            if isinstance(entry, dict) and 'nn_scl_l4_bler' in entry:
                scl = entry.get('nn_scl_l4_bler', '?')
                crc = entry.get('nn_crc_scl_l4_bler', '?')
                print(f"    N={entry['N']}: SCL={scl:.4f}, CRC-SCL={crc if crc is None else f'{crc:.4f}'}")


if __name__ == '__main__':
    main()
