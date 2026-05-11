#!/usr/bin/env python3
"""
Compute proper per-position MI for MAC polar codes using soft SC posteriors.

MI_i = 1 - E[H(U_i | Z^N, U_1^{i-1})]   (for user U)
MI_i = 1 - E[H(V_i | Z^N, V_1^{i-1})]   (for user V)

where H(p) = -p*log2(p) - (1-p)*log2(1-p) is the binary entropy
of the posterior probability from the SC decoder at position i.

This gives MI in [0, 1] and conserves: avg(MI_U) = I(X;Z)/1, avg(MI_V) = I(Y;Z|X)/1.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC, BEMAC, ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from polar.decoder_interleaved import (
    _CompGraph, _norm_prod_single, _NEG_INF, _LOG_HALF, _LOG_QUARTER
)
from polar.efficient_decoder import build_log_W_leaf


def binary_entropy(p):
    """H(p) in bits. Handles p=0, p=1."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def measure_mi_soft(N, channel, path_b, n_samples=2000, seed=789):
    """
    Run genie-aided SC decode, extract posterior probabilities at each step,
    compute per-position MI for U and V.

    Returns: mi_u (N,), mi_v (N,) — proper MI in [0, 1].
    """
    n = N.bit_length() - 1
    assert (1 << n) == N
    rng = np.random.default_rng(seed)

    # Accumulate conditional entropy per position
    h_u = np.zeros(N)  # sum of H(U_i | obs)
    h_v = np.zeros(N)
    count_u = np.zeros(N)
    count_v = np.zeros(N)

    for _ in range(n_samples):
        # Generate random codewords
        u = rng.integers(0, 2, N)
        v = rng.integers(0, 2, N)
        x = polar_encode_batch(u.reshape(1, -1))[0]
        y = polar_encode_batch(v.reshape(1, -1))[0]
        z = channel.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]

        # Build channel posteriors
        log_W = build_log_W_leaf(z, channel)
        graph = _CompGraph(n, log_W)

        u_hat = {}
        v_hat = {}
        i_u = 0
        i_v = 0

        for step in range(2 * N):
            gamma = path_b[step]
            if gamma == 0:
                i_u += 1
                i_t = i_u
            else:
                i_v += 1
                i_t = i_v

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

            # Extract posterior probabilities
            if gamma == 0:
                # P(u=0) and P(u=1), marginalizing over v
                log_p0 = np.logaddexp(combined[0, 0], combined[0, 1])
                log_p1 = np.logaddexp(combined[1, 0], combined[1, 1])
            else:
                # P(v=0) and P(v=1), marginalizing over u
                log_p0 = np.logaddexp(combined[0, 0], combined[1, 0])
                log_p1 = np.logaddexp(combined[0, 1], combined[1, 1])

            # Normalize to get proper probabilities
            log_total = np.logaddexp(log_p0, log_p1)
            p0 = np.exp(log_p0 - log_total)
            p1 = np.exp(log_p1 - log_total)

            # Conditional entropy at this position
            h = binary_entropy(p0)

            if gamma == 0:
                h_u[i_t - 1] += h
                count_u[i_t - 1] += 1
                # Genie: use true bit
                bit = int(u[i_t - 1])
            else:
                h_v[i_t - 1] += h
                count_v[i_t - 1] += 1
                bit = int(v[i_t - 1])

            # Record decision (genie-aided)
            if gamma == 0:
                u_hat[i_t] = bit
            else:
                v_hat[i_t] = bit

            # Set leaf to partially deterministic
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

        assert i_u == N and i_v == N

    # Average conditional entropy
    avg_h_u = np.where(count_u > 0, h_u / count_u, 1.0)
    avg_h_v = np.where(count_v > 0, h_v / count_v, 1.0)

    # MI = 1 - H(U|obs)
    mi_u = 1.0 - avg_h_u
    mi_v = 1.0 - avg_h_v

    return mi_u, mi_v


def get_channel(channel_name):
    if channel_name == 'gmac':
        return GaussianMAC(sigma2=10 ** (-6.0 / 10))
    elif channel_name == 'bemac':
        return BEMAC()
    elif channel_name == 'abnmac':
        return ABNMAC()
    else:
        raise ValueError(f"Unknown channel: {channel_name}")


if __name__ == '__main__':
    import json, time

    results = {}

    for channel_name in ['gmac', 'bemac', 'abnmac']:
        channel = get_channel(channel_name)
        results[channel_name] = {}

        for cls in ['B', 'C']:
            results[channel_name][cls] = {}

            for N in [16, 32, 64]:
                n = int(np.log2(N))
                path_i = N // 2 if cls == 'B' else (N if cls == 'C' else 0)
                b = make_path(N, path_i)

                n_samples = 3000 if N <= 32 else 1000
                print(f'{channel_name} {cls} N={N} ({n_samples} samples)...',
                      end=' ', flush=True)
                t0 = time.time()
                mi_u, mi_v = measure_mi_soft(N, channel, b, n_samples=n_samples)
                elapsed = time.time() - t0

                mi_sum = mi_u + mi_v
                print(f'avg MI_U={mi_u.mean():.4f} MI_V={mi_v.mean():.4f} '
                      f'sum={mi_sum.mean():.4f} ({elapsed:.1f}s)', flush=True)

                results[channel_name][cls][str(N)] = {
                    'mi_u': mi_u.tolist(),
                    'mi_v': mi_v.tolist(),
                    'mi_sum': mi_sum.tolist(),
                    'avg_mi_u': float(mi_u.mean()),
                    'avg_mi_v': float(mi_v.mean()),
                    'avg_sum': float(mi_sum.mean()),
                    'n_samples': n_samples,
                }

    out_path = os.path.join(os.path.dirname(__file__), 'soft_mi_data.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')
