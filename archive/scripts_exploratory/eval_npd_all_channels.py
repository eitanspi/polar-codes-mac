#!/usr/bin/env python3
"""
eval_npd_all_channels.py
========================
Evaluate all existing NPD model checkpoints across ISI-MAC, Ising MAC,
and MA-AGN MAC channels with 5K CW for reliable paper-style results.

Handles two model architectures:
  1. MemoryStageNPD (from neural.npd_memory_mac) - ISI, Ising, MAAGN BiGRU/window models
  2. NPDSingleUser (from class_c_npd.models) - some d16_h100 standalone models

Saves consolidated results to results/paper_style/npd_all_channels_5kcw.json.
"""
import sys, os, time, json, math
import numpy as np

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import torch
torch.set_num_threads(4)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)

from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.channels_memory import ISIMAC
from polar.channels_memory_new import IsingMAC, MAAGNMAC

DESIGNS_DIR = os.path.join(_ROOT, 'designs')
RESULTS_DIR = os.path.join(_ROOT, 'results', 'paper_style')
os.makedirs(RESULTS_DIR, exist_ok=True)

RATES = {
    16:  {'ku': 4,   'kv': 7},
    32:  {'ku': 7,   'kv': 15},
    64:  {'ku': 15,  'kv': 29},
    128: {'ku': 30,  'kv': 58},
    256: {'ku': 59,  'kv': 117},
    512: {'ku': 119, 'kv': 233},
    1024: {'ku': 238, 'kv': 467},
}


def wilson_ci(n_err, n_total, z=1.96):
    if n_total == 0:
        return (0.0, 1.0)
    p_hat = n_err / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def load_design(N):
    n = int(math.log2(N))
    rates = RATES[N]
    ku, kv = rates['ku'], rates['kv']
    design_path = os.path.join(DESIGNS_DIR, f'gmac_C_n{n}_snr6dB.npz')
    Au_list, Av_list, fu_dict, fv_dict, _, _, _ = design_from_file(design_path, n, ku, kv)
    Au = sorted(Au_list)
    Av = sorted(Av_list)
    frozen_u_0idx = {p - 1 for p in range(1, N+1) if p not in Au}
    frozen_v_0idx = {p - 1 for p in range(1, N+1) if p not in Av}
    return Au, Av, frozen_u_0idx, frozen_v_0idx, ku, kv


def infer_model_type(ckpt_path):
    """Infer model type from state_dict keys."""
    sd = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    state_dict = sd.get('state_dict', sd)
    keys = list(state_dict.keys())
    if any(k.startswith('tree.') for k in keys):
        # MemoryStageNPD
        if any('gru' in k for k in keys):
            return 'memory_bigru', sd
        else:
            return 'memory_window', sd
    elif any(k.startswith('z_encoder.') for k in keys):
        return 'npd_single_user', sd
    else:
        return 'unknown', sd


def load_memory_stage_model(ckpt_data, stage='s1'):
    """Load a MemoryStageNPD model from checkpoint data."""
    from neural.npd_memory_mac import MemoryStageNPD
    sd = ckpt_data.get('state_dict', ckpt_data)

    # Infer d from tree.emb2llr last layer input dim
    d = None
    hidden = None
    n_layers = 2  # default
    encoder_type = 'bigru' if any('gru' in k for k in sd.keys()) else 'window'
    extra_dim = 0

    # Infer d from tree.checknode.0.weight: (hidden, 2*d)
    cn0 = sd['tree.checknode.0.weight']
    hidden = cn0.shape[0]
    d = cn0.shape[1] // 2

    # Infer n_layers from checknode MLP
    cn_layers = [k for k in sd if k.startswith('tree.checknode.') and k.endswith('.weight')]
    n_layers = len(cn_layers) - 1  # minus output layer

    # Infer extra_dim from z_encoder
    if encoder_type == 'bigru':
        gru_ih = sd['z_encoder.gru.weight_ih_l0']
        in_size = gru_ih.shape[1]
        extra_dim = in_size - 1
        gru_hh = sd['z_encoder.gru.weight_hh_l0']
        gru_hidden = gru_hh.shape[1]
        # GRU layers
        gru_layer_keys = [k for k in sd if 'weight_ih_l' in k and 'reverse' not in k]
        gru_layers = len(gru_layer_keys)
    elif encoder_type == 'window':
        mlp0 = sd['z_encoder.mlp.0.weight']
        in_dim = mlp0.shape[1]
        # in_dim = (2W+1)*(1+extra_dim)
        # Hard to infer W and extra_dim separately, assume W=1 for now
        extra_dim = 0  # Will adjust if needed
        gru_layers = 1

    model = MemoryStageNPD(
        d=d, hidden=hidden, n_layers=n_layers,
        encoder_type=encoder_type,
        window_size=1 if encoder_type == 'window' else 1,
        extra_dim=extra_dim,
        gru_layers=gru_layers if encoder_type == 'bigru' else 1,
    )
    model.load_state_dict(sd)
    model.eval()
    return model, d, hidden


def chain_eval_memory_npd(s1_path, s2_path, channel_sample_fn, N,
                           n_cw=5000, batch_size=8, seed=999):
    """
    Chained NPD evaluation for memory MAC channels.

    channel_sample_fn(x, y) -> z  (both (B, N) int arrays -> (B, N) float)

    Stage 1: feed raw z through BiGRU/window encoder, decode U
    Stage 2: feed (z, u_hat_bpsk) through BiGRU/window encoder, decode V
    """
    n = int(math.log2(N))
    Au, Av, frozen_u, frozen_v, ku, kv = load_design(N)
    br = bit_reversal_perm(n)

    # Load models
    _, s1_data = infer_model_type(s1_path)
    _, s2_data = infer_model_type(s2_path)
    model_s1, d1, h1 = load_memory_stage_model(s1_data, stage='s1')
    model_s2, d2, h2 = load_memory_stage_model(s2_data, stage='s2')

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    errs_u = errs_v = errs_total = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)

            # Random messages
            u_msg = np.zeros((actual, N), dtype=np.int8)
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Au:
                u_msg[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                v_msg[:, p - 1] = rng.integers(0, 2, actual)

            # Encode
            x_phys = polar_encode_batch(u_msg.astype(int))
            y_phys = polar_encode_batch(v_msg.astype(int))

            # Transmit
            z = channel_sample_fn(x_phys, y_phys)
            z = np.asarray(z, dtype=np.float32)
            if z.ndim == 1:
                z = z.reshape(1, -1)

            # Stage 1: encode z, decode U
            z_t = torch.from_numpy(z).float()
            # Encode channel features in NATURAL order, then bit-reverse for NPD tree
            emb1 = model_s1.encode_channel(z_t)  # (B, N, d) in natural order
            emb1_br = emb1[:, br, :]  # bit-reverse for NPD tree
            u_hat = model_s1.tree.decode(emb1_br, frozen_u)

            # Reconstruct X_hat
            u_hat_np = u_hat.numpy().astype(int)
            x_hat = polar_encode_batch(u_hat_np)

            # Stage 2: encode (z, u_hat_bpsk), decode V
            u_hat_bpsk = (1 - 2 * x_hat.astype(np.float32))
            side2 = torch.from_numpy(u_hat_bpsk).float().unsqueeze(-1)
            emb2 = model_s2.encode_channel(z_t, side=side2)
            emb2_br = emb2[:, br, :]
            v_hat = model_s2.tree.decode(emb2_br, frozen_v)

            # Count errors
            for i in range(actual):
                u_wrong = any(u_hat[i, p - 1].item() != u_msg[i, p - 1] for p in Au)
                v_wrong = any(v_hat[i, p - 1].item() != v_msg[i, p - 1] for p in Av)
                if u_wrong: errs_u += 1
                if v_wrong: errs_v += 1
                if u_wrong or v_wrong: errs_total += 1

            total += actual
            if total % 500 == 0:
                elapsed = time.time() - t0
                print(f'    [{total}/{n_cw}] errs={errs_total} ({elapsed:.0f}s)', flush=True)

    elapsed = time.time() - t0
    ci = wilson_ci(errs_total, n_cw)
    return {
        'N': N, 'ku': ku, 'kv': kv, 'n_cw': n_cw,
        'errs_u': errs_u, 'errs_v': errs_v, 'errs_total': errs_total,
        'bler_u': errs_u / n_cw, 'bler_v': errs_v / n_cw,
        'bler_total': errs_total / n_cw,
        'wilson_95_ci': list(ci),
        'time_s': elapsed,
        's1_ckpt': os.path.basename(s1_path),
        's2_ckpt': os.path.basename(s2_path),
        'd': d1, 'hidden': h1,
    }


# ============================================================================
#  ISI-MAC NPD models
# ============================================================================

def eval_isi_mac_npd():
    print('=' * 60)
    print('ISI-MAC NPD Evaluations (5K CW)')
    print('=' * 60)

    base = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_memory_mac')
    ch = ISIMAC(sigma2=10**(-0.6), h=0.3)

    def sample_fn(x, y):
        return ch.sample_batch(x, y)

    # Define model pairs
    models = []

    # d16_h100 standalone (N=64, 128)
    for N in [64, 128]:
        s1 = os.path.join(base, f'd16_h100_standalone_s1_N{N}_best.pt')
        s2 = os.path.join(base, f'd16_h100_standalone_s2_N{N}_best.pt')
        if os.path.exists(s1) and os.path.exists(s2):
            models.append(('isi_d16_h100', N, s1, s2))

    # BiGRU models (N=16, 32, 64, 128)
    for N in [16, 32, 64, 128]:
        s1 = os.path.join(base, f'isi_mac_bigru_L1_s1_N{N}_best.pt')
        s2 = os.path.join(base, f'isi_mac_bigru_L1_s2_N{N}_best.pt')
        if os.path.exists(s1) and os.path.exists(s2):
            models.append(('isi_bigru_L1', N, s1, s2))

    # d64 models (N=128, 256)
    s1_d64_128 = os.path.join(base, 'isi_mac_bigru_L1_cont_d64_s1_N128_best.pt')
    s2_d64_128 = os.path.join(base, 'd64_s2_N128_best.pt')
    if os.path.exists(s1_d64_128) and os.path.exists(s2_d64_128):
        models.append(('isi_d64_cont', 128, s1_d64_128, s2_d64_128))

    s2_d64_256 = os.path.join(base, 'd64_s2_N256_best.pt')
    if os.path.exists(s1_d64_128) and os.path.exists(s2_d64_256):
        models.append(('isi_d64_cont', 256, s1_d64_128, s2_d64_256))

    # GPU curriculum models
    for N in [16, 32, 64, 128]:
        s1 = os.path.join(base, f'gpu_curriculum_s1_N{N}_best.pt')
        s2 = os.path.join(base, f'gpu_curriculum_s2_N{N}_best.pt')
        if os.path.exists(s1) and os.path.exists(s2):
            models.append(('isi_gpu_curr', N, s1, s2))

    results = {}
    for model_name, N, s1, s2 in models:
        key = f'{model_name}_N{N}'
        print(f'\n  Evaluating {key}...')
        print(f'    S1: {os.path.basename(s1)}')
        print(f'    S2: {os.path.basename(s2)}')

        try:
            r = chain_eval_memory_npd(s1, s2, sample_fn, N, n_cw=5000, seed=999)
            r['model_name'] = model_name
            results[key] = r
            print(f'    bler={r["bler_total"]:.4f} CI=[{r["wilson_95_ci"][0]:.4f}, {r["wilson_95_ci"][1]:.4f}]')
        except Exception as e:
            import traceback
            print(f'    ERROR: {e}')
            traceback.print_exc()
            results[key] = {'error': str(e), 'N': N, 'model_name': model_name}

    return results


# ============================================================================
#  Ising MAC NPD models
# ============================================================================

def eval_ising_mac_npd():
    print('\n' + '=' * 60)
    print('Ising MAC NPD Evaluations (5K CW)')
    print('=' * 60)

    base = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_ising_mac')
    ch = IsingMAC(sigma2=0.251, p_flip=0.1)

    def sample_fn(x, y):
        return ch.sample_batch(x, y).astype(np.float32)

    models = []
    for N in [16, 32, 64]:
        s1 = os.path.join(base, f'ising_d16_h100_s1_N{N}_best.pt')
        s2 = os.path.join(base, f'ising_d16_h100_s2_N{N}_best.pt')
        if os.path.exists(s1) and os.path.exists(s2):
            models.append(('ising_d16_h100', N, s1, s2))

    results = {}
    for model_name, N, s1, s2 in models:
        key = f'{model_name}_N{N}'
        print(f'\n  Evaluating {key}...')

        try:
            r = chain_eval_memory_npd(s1, s2, sample_fn, N, n_cw=5000, seed=2999)
            r['model_name'] = model_name
            results[key] = r
            print(f'    bler={r["bler_total"]:.4f} CI=[{r["wilson_95_ci"][0]:.4f}, {r["wilson_95_ci"][1]:.4f}]')
        except Exception as e:
            import traceback
            print(f'    ERROR: {e}')
            traceback.print_exc()
            results[key] = {'error': str(e), 'N': N, 'model_name': model_name}

    return results


# ============================================================================
#  MA-AGN NPD models
# ============================================================================

def eval_maagn_mac_npd():
    print('\n' + '=' * 60)
    print('MA-AGN MAC NPD Evaluations (5K CW)')
    print('=' * 60)

    base = os.path.join(_ROOT, 'class_c_npd', 'results', 'npd_maagn_mac')
    ch = MAAGNMAC(sigma2=10**(-0.6), alpha=0.3)

    def sample_fn(x, y):
        return ch.sample_batch(x, y).astype(np.float32)

    models = []
    for N in [16, 32, 64]:
        s1 = os.path.join(base, f'maagn_bigru_L1_s1_N{N}_best.pt')
        s2 = os.path.join(base, f'maagn_bigru_L1_s2_N{N}_best.pt')
        if os.path.exists(s1) and os.path.exists(s2):
            models.append(('maagn_bigru_L1', N, s1, s2))

    for N in [64, 128]:
        s1 = os.path.join(base, f'maagn_d16_h100_s1_N{N}_best.pt')
        s2 = os.path.join(base, f'maagn_d16_h100_s2_N{N}_best.pt')
        if os.path.exists(s1) and os.path.exists(s2):
            models.append(('maagn_d16_h100', N, s1, s2))

    results = {}
    for model_name, N, s1, s2 in models:
        key = f'{model_name}_N{N}'
        print(f'\n  Evaluating {key}...')

        try:
            r = chain_eval_memory_npd(s1, s2, sample_fn, N, n_cw=5000, seed=3999)
            r['model_name'] = model_name
            results[key] = r
            print(f'    bler={r["bler_total"]:.4f} CI=[{r["wilson_95_ci"][0]:.4f}, {r["wilson_95_ci"][1]:.4f}]')
        except Exception as e:
            import traceback
            print(f'    ERROR: {e}')
            traceback.print_exc()
            results[key] = {'error': str(e), 'N': N, 'model_name': model_name}

    return results


# ============================================================================

if __name__ == '__main__':
    print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    all_results = {}

    isi_results = eval_isi_mac_npd()
    all_results['isi_mac'] = isi_results

    # Save incrementally
    out_path = os.path.join(RESULTS_DIR, 'npd_all_channels_5kcw.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    ising_results = eval_ising_mac_npd()
    all_results['ising_mac'] = ising_results
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    maagn_results = eval_maagn_mac_npd()
    all_results['maagn_mac'] = maagn_results
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\nSaved: {out_path}')
    print(f'Done: {time.strftime("%Y-%m-%d %H:%M:%S")}')
