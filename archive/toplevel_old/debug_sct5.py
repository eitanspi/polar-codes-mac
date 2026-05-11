"""Debug SCT vs ML at N=4 to find where the tree operations diverge."""
import sys
import numpy as np
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

from polar.channels_memory import ISIMAC
from polar.channels import GaussianMAC
from polar.encoder import polar_encode, bit_reversal_perm
from polar.design import design_gmac, make_path
from polar.decoder_trellis_sct import (
    _CompGraphTrellis, _circ_conv_trellis, _norm_prod_trellis,
    _norm_prod_trellis_single, _logsumexp_axis,
)
from polar.decoder_interleaved import (
    _CompGraph, _circ_conv_batch, _norm_prod_batch, _norm_prod_single,
    _LOG_HALF, _LOG_QUARTER,
)
from polar.efficient_decoder import build_log_W_leaf

_NEG_INF = -np.inf

N = 4; n = 2; sigma2 = 0.5; S = 4
ch_ml = GaussianMAC(sigma2=sigma2)
ch_mem = ISIMAC(sigma2=sigma2, h=0.0)

b = make_path(N, N)  # [0,0,0,0,1,1,1,1]
ku, kv = 1, 2
Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)
print(f"Au={Au}, Av={Av}")
print(f"frozen_u={frozen_u}, frozen_v={frozen_v}")
print(f"Path b={b}")

np.random.seed(42)
# First trial
u_msg = np.zeros(N, dtype=int)
v_msg = np.zeros(N, dtype=int)
for pos in Au: u_msg[pos-1] = np.random.randint(0, 2)
for pos in Av: v_msg[pos-1] = np.random.randint(0, 2)

X = np.array(polar_encode(u_msg.tolist()))
Y = np.array(polar_encode(v_msg.tolist()))
Z = ch_ml.sample_batch(X, Y)

print(f"u_msg={u_msg.tolist()}, v_msg={v_msg.tolist()}")
print(f"X={X.tolist()}, Y={Y.tolist()}")
print(f"Z={[f'{z:.3f}' for z in Z]}")

log_W_ml = build_log_W_leaf(list(Z), ch_ml)
log_W_sct = ch_mem.build_leaf_tensors(list(Z))

marg = lambda t: _logsumexp_axis(t.reshape(2, 2, -1), axis=2) if t.ndim == 4 else \
                 _logsumexp_axis(t.reshape(t.shape[0], 2, 2, -1), axis=3)

# Build both graphs
graph_ml = _CompGraph(n, log_W_ml)
graph_sct = _CompGraphTrellis(n, log_W_sct, S)

print(f"\nBit-reversal: {bit_reversal_perm(n)}")
print(f"\nRoot edge comparison:")
for t in range(N):
    ml_r = graph_ml.edge_data[1][t]
    sct_r = graph_sct.edge_data[1][t]
    sct_m = marg(sct_r)
    diff = np.max(np.abs(np.exp(ml_r) - np.exp(sct_m)))
    print(f"  t={t}: ML={np.exp(ml_r).round(5)}, SCT_marg={np.exp(sct_m).round(5)}, diff={diff:.2e}")

# Now step through the decode, comparing at each step
u_hat_ml = {}; v_hat_ml = {}
u_hat_sct = {}; v_hat_sct = {}
i_u = 0; i_v = 0

for step in range(2 * N):
    gamma = b[step]
    if gamma == 0:
        i_u += 1; i_t = i_u; fd = frozen_u; label = f"U{i_t}"
    else:
        i_v += 1; i_t = i_v; fd = frozen_v; label = f"V{i_t}"

    leaf_edge = i_t + N - 1
    target_vertex = leaf_edge >> 1

    graph_ml.step_to(target_vertex)
    graph_sct.step_to(target_vertex)

    # Save temps
    temp_ml = graph_ml.edge_data[leaf_edge][0].copy()
    temp_sct = graph_sct.edge_data[leaf_edge][0].copy()

    if leaf_edge & 1 == 0:
        graph_ml.calc_left(target_vertex)
        graph_sct.calc_left(target_vertex)
    else:
        graph_ml.calc_right(target_vertex)
        graph_sct.calc_right(target_vertex)

    td_ml = graph_ml.edge_data[leaf_edge][0]
    td_sct = graph_sct.edge_data[leaf_edge][0]

    combined_ml = _norm_prod_single(td_ml, temp_ml)
    combined_sct = _norm_prod_trellis_single(td_sct, temp_sct)

    # ML decision
    if i_t in fd:
        bit_ml = fd[i_t]
    else:
        if gamma == 0:
            p0 = np.logaddexp(combined_ml[0,0], combined_ml[0,1])
            p1 = np.logaddexp(combined_ml[1,0], combined_ml[1,1])
        else:
            p0 = np.logaddexp(combined_ml[0,0], combined_ml[1,0])
            p1 = np.logaddexp(combined_ml[0,1], combined_ml[1,1])
        bit_ml = 1 if p1 > p0 else 0

    # SCT decision
    bit_post_sct = marg(combined_sct)
    if i_t in fd:
        bit_sct = fd[i_t]
    else:
        if gamma == 0:
            p0s = np.logaddexp(bit_post_sct[0,0], bit_post_sct[0,1])
            p1s = np.logaddexp(bit_post_sct[1,0], bit_post_sct[1,1])
        else:
            p0s = np.logaddexp(bit_post_sct[0,0], bit_post_sct[1,0])
            p1s = np.logaddexp(bit_post_sct[0,1], bit_post_sct[1,1])
        bit_sct = 1 if p1s > p0s else 0

    match = "OK" if bit_ml == bit_sct else "MISMATCH"
    print(f"\nStep {step} ({label}, frozen={i_t in fd}): ML={bit_ml} SCT={bit_sct} {match}")
    print(f"  ML  combined: {np.exp(combined_ml).round(6)}")
    print(f"  SCT combined (marg): {np.exp(bit_post_sct).round(6)}")

    if gamma == 0:
        u_hat_ml[i_t] = bit_ml; u_hat_sct[i_t] = bit_sct
    else:
        v_hat_ml[i_t] = bit_ml; v_hat_sct[i_t] = bit_sct

    # Set ML leaf
    new_leaf_ml = np.full((2, 2), _NEG_INF)
    uv_ml = u_hat_ml.get(i_t); vv_ml = v_hat_ml.get(i_t)
    if uv_ml is not None and vv_ml is not None:
        new_leaf_ml[uv_ml, vv_ml] = 0.0
    elif uv_ml is not None:
        new_leaf_ml[uv_ml, 0] = _LOG_HALF; new_leaf_ml[uv_ml, 1] = _LOG_HALF
    elif vv_ml is not None:
        new_leaf_ml[0, vv_ml] = _LOG_HALF; new_leaf_ml[1, vv_ml] = _LOG_HALF
    else:
        new_leaf_ml[:, :] = _LOG_QUARTER
    graph_ml.edge_data[leaf_edge][0] = new_leaf_ml

    # Set SCT leaf
    new_leaf_sct = np.full((2, 2, S, S), _NEG_INF)
    uv_sct = u_hat_sct.get(i_t); vv_sct = v_hat_sct.get(i_t)
    if uv_sct is not None and vv_sct is not None:
        s_next = ch_mem._encode_state(uv_sct, vv_sct)
        for s in range(S):
            new_leaf_sct[uv_sct, vv_sct, s, s_next] = -np.log(float(S))
    elif uv_sct is not None:
        for v in range(2):
            s_next = ch_mem._encode_state(uv_sct, v)
            for s in range(S):
                new_leaf_sct[uv_sct, v, s, s_next] = -np.log(2.0 * S)
    elif vv_sct is not None:
        for u in range(2):
            s_next = ch_mem._encode_state(u, vv_sct)
            for s in range(S):
                new_leaf_sct[u, vv_sct, s, s_next] = -np.log(2.0 * S)
    else:
        for u in range(2):
            for v in range(2):
                s_next = ch_mem._encode_state(u, v)
                for s in range(S):
                    new_leaf_sct[u, v, s, s_next] = -np.log(4.0 * S)
    graph_sct.edge_data[leaf_edge][0] = new_leaf_sct

print(f"\nFinal: ML u={[u_hat_ml.get(k,0) for k in range(1,N+1)]}")
print(f"Final: ML v={[v_hat_ml.get(k,0) for k in range(1,N+1)]}")
print(f"Final: SCT u={[u_hat_sct.get(k,0) for k in range(1,N+1)]}")
print(f"Final: SCT v={[v_hat_sct.get(k,0) for k in range(1,N+1)]}")
print(f"Sent:  u={u_msg.tolist()}, v={v_msg.tolist()}")
