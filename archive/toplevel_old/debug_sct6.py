"""Test: what if we remove the initial state constraint from leaf tensors?"""
import sys
import numpy as np
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

from polar.channels_memory import ISIMAC
from polar.channels import GaussianMAC
from polar.encoder import polar_encode
from polar.design import design_gmac, make_path
from polar.decoder_trellis_sct import _CompGraphTrellis, _norm_prod_trellis_single, _logsumexp_axis

_NEG_INF = -np.inf
_LOG_HALF = np.log(0.5)

N = 4; n = 2; sigma2 = 0.5; S = 4
ch_ml = GaussianMAC(sigma2=sigma2)
ch_mem = ISIMAC(sigma2=sigma2, h=0.0)
b = make_path(N, N)
ku, kv = 1, 2
Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)

# Custom decode that removes initial state constraint
def decode_no_initial_state(N, z, b, frozen_u, frozen_v, channel):
    n = N.bit_length() - 1
    S = channel.num_states
    log_W = channel.build_leaf_tensors(z)

    # Remove initial state constraint: allow all s_in at position 0
    log_W[0, :, :, 1:, :] = log_W[0, :, :, 0:1, :]  # copy s=0 data to all s

    graph = _CompGraphTrellis(n, log_W, S)

    u_hat = {}; v_hat = {}; i_u = 0; i_v = 0

    for step in range(2 * N):
        gamma = b[step]
        if gamma == 0:
            i_u += 1; i_t = i_u; fd = frozen_u
        else:
            i_v += 1; i_t = i_v; fd = frozen_v

        leaf_edge = i_t + N - 1
        target_vertex = leaf_edge >> 1
        graph.step_to(target_vertex)

        temp = graph.edge_data[leaf_edge][0].copy()
        if leaf_edge & 1 == 0:
            graph.calc_left(target_vertex)
        else:
            graph.calc_right(target_vertex)

        top_down = graph.edge_data[leaf_edge][0]
        combined = _norm_prod_trellis_single(top_down, temp)
        bit_post = _logsumexp_axis(combined.reshape(2, 2, -1), axis=2)

        if i_t in fd:
            bit = fd[i_t]
        else:
            if gamma == 0:
                p0 = np.logaddexp(bit_post[0,0], bit_post[0,1])
                p1 = np.logaddexp(bit_post[1,0], bit_post[1,1])
            else:
                p0 = np.logaddexp(bit_post[0,0], bit_post[1,0])
                p1 = np.logaddexp(bit_post[0,1], bit_post[1,1])
            bit = 1 if p1 > p0 else 0

        if gamma == 0: u_hat[i_t] = bit
        else: v_hat[i_t] = bit

        # Use uniform-state leaves (no state constraint in prior)
        new_leaf = np.full((2, 2, S, S), _NEG_INF)
        u_val = u_hat.get(i_t); v_val = v_hat.get(i_t)
        log_us = -np.log(float(S*S))
        if u_val is not None and v_val is not None:
            new_leaf[u_val, v_val, :, :] = log_us
        elif u_val is not None:
            new_leaf[u_val, :, :, :] = -np.log(2.0*S*S)
        elif v_val is not None:
            new_leaf[:, v_val, :, :] = -np.log(2.0*S*S)
        else:
            new_leaf[:,:,:,:] = -np.log(4.0*S*S)
        graph.edge_data[leaf_edge][0] = new_leaf

    u_dec = [u_hat.get(k,0) for k in range(1, N+1)]
    v_dec = [v_hat.get(k,0) for k in range(1, N+1)]
    return u_dec, v_dec

from polar.decoder_interleaved import decode_single as ml_decode

np.random.seed(42)
n_trials = 200
matches = 0; sct_correct = 0; ml_correct = 0

for trial in range(n_trials):
    u_msg = np.zeros(N, dtype=int)
    v_msg = np.zeros(N, dtype=int)
    for pos in Au: u_msg[pos-1] = np.random.randint(0, 2)
    for pos in Av: v_msg[pos-1] = np.random.randint(0, 2)
    X = np.array(polar_encode(u_msg.tolist()))
    Y = np.array(polar_encode(v_msg.tolist()))
    Z = ch_ml.sample_batch(X, Y)
    u_ml, v_ml = ml_decode(N, list(Z), b, frozen_u, frozen_v, ch_ml)
    u_sct, v_sct = decode_no_initial_state(N, list(Z), b, frozen_u, frozen_v, ch_mem)
    if u_ml == u_sct and v_ml == v_sct: matches += 1
    if u_ml == u_msg.tolist() and v_ml == v_msg.tolist(): ml_correct += 1
    if u_sct == u_msg.tolist() and v_sct == v_msg.tolist(): sct_correct += 1

print(f'Match: {matches}/{n_trials}')
print(f'ML correct: {ml_correct}/{n_trials}')
print(f'SCT (no init state) correct: {sct_correct}/{n_trials}')
