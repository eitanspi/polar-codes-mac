"""Debug SCT vs FB at N=4, h=0.5."""
import sys
import numpy as np
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode, bit_reversal_perm
from polar.design import design_gmac, make_path
from polar.decoder_trellis_sct import (
    _CompGraphTrellis, _circ_conv_trellis, _norm_prod_trellis,
    _norm_prod_trellis_single, _logsumexp_axis,
)
from polar.decoder_trellis import decode_single as fb_decode

_NEG_INF = -np.inf

N = 4; n = 2; snr_db = 20; h = 0.5; S = 4
channel = ISIMAC.from_snr_db(snr_db, h=h)
b = make_path(N, N)  # [0,0,0,0,1,1,1,1]
ku, kv = 1, 2
_, _, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2=0.1)

np.random.seed(42)
u_msg = np.zeros(N, dtype=int)
v_msg = np.zeros(N, dtype=int)
Au = sorted(set(range(1,N+1)) - set(frozen_u.keys()))
Av = sorted(set(range(1,N+1)) - set(frozen_v.keys()))
for pos in Au: u_msg[pos-1] = np.random.randint(0, 2)
for pos in Av: v_msg[pos-1] = np.random.randint(0, 2)

X = np.array(polar_encode(u_msg.tolist()))
Y = np.array(polar_encode(v_msg.tolist()))
Z = channel.sample_batch(X, Y)

print(f"u_msg={u_msg.tolist()}, v_msg={v_msg.tolist()}")
print(f"X={X.tolist()}, Y={Y.tolist()}")
print(f"Z=[{', '.join(f'{z:.3f}' for z in Z)}]")
print(f"Au={Au}, Av={Av}")
print(f"frozen_u={frozen_u}, frozen_v={frozen_v}")
print(f"Path b={b}")
print(f"Bit-reversal: {bit_reversal_perm(n)}")

# FB decode for comparison
u_fb, v_fb = fb_decode(N, list(Z), b, frozen_u, frozen_v, channel)
print(f"\nFB result: u={u_fb}, v={v_fb}")

# SCT step-by-step
log_W = channel.build_leaf_tensors(list(Z))
graph = _CompGraphTrellis(n, log_W, S)

marg = lambda t: _logsumexp_axis(t.reshape(2, 2, -1), axis=2)

print(f"\nRoot edge (bit-reversed) marginals:")
br = bit_reversal_perm(n)
for t in range(N):
    m = marg(graph.edge_data[1][t])
    print(f"  root[{t}] (pos {br[t]}): {np.exp(m).round(5)}")

# For N=4, tree: vertex 1 has children edges 2, 3.
# Edge 2 is the left child (2 entries). Edge 3 is the right child (2 entries).
# Edges 4,5 are children of vertex 2 (1 entry each = leaves for positions 1,2).
# Edges 6,7 are children of vertex 3 (1 entry each = leaves for positions 3,4).

# Path [0,0,0,0,1,1,1,1] means U1, U2, U3, U4, V1, V2, V3, V4.
# Positions: U1->pos1, U2->pos2, U3->pos3, U4->pos4, V1->pos1, V2->pos2, V3->pos3, V4->pos4

# Step 0: U1 at pos 1. leaf_edge = 1+4-1 = 4. target_vertex = 4>>1 = 2.
# Need to navigate from vertex 1 to vertex 2 (calc_left at vertex 1).

u_hat = {}; v_hat = {}; i_u = 0; i_v = 0

for step in range(2 * N):
    gamma = b[step]
    if gamma == 0:
        i_u += 1; i_t = i_u; fd = frozen_u
    else:
        i_v += 1; i_t = i_v; fd = frozen_v

    leaf_edge = i_t + N - 1
    target_vertex = leaf_edge >> 1

    label = f"{'U' if gamma==0 else 'V'}{i_t}"

    graph.step_to(target_vertex)
    temp = graph.edge_data[leaf_edge][0].copy()

    if leaf_edge & 1 == 0:
        graph.calc_left(target_vertex)
    else:
        graph.calc_right(target_vertex)

    top_down = graph.edge_data[leaf_edge][0]
    combined = _norm_prod_trellis_single(top_down, temp)
    bit_post = marg(combined)

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

    # Check against FB
    if gamma == 0:
        fb_bit = u_fb[i_t - 1]
    else:
        fb_bit = v_fb[i_t - 1]

    match = "OK" if bit == fb_bit else "MISMATCH"
    print(f"\nStep {step} ({label}, frozen={i_t in fd}): SCT={bit} FB={fb_bit} {match}")
    print(f"  bit_post = {np.exp(bit_post).round(6)}")

    # Set decided leaf
    new_leaf = np.full((2, 2, S, S), _NEG_INF)
    u_val = u_hat.get(i_t); v_val = v_hat.get(i_t)
    if u_val is not None and v_val is not None:
        s_next = channel._encode_state(u_val, v_val)
        for s in range(S):
            new_leaf[u_val, v_val, s, s_next] = -np.log(float(S))
    elif u_val is not None:
        for v in range(2):
            s_next = channel._encode_state(u_val, v)
            for s in range(S):
                new_leaf[u_val, v, s, s_next] = -np.log(2.0 * S)
    elif v_val is not None:
        for u in range(2):
            s_next = channel._encode_state(u, v_val)
            for s in range(S):
                new_leaf[u, v_val, s, s_next] = -np.log(2.0 * S)
    else:
        for u in range(2):
            for v in range(2):
                s_next = channel._encode_state(u, v)
                for s in range(S):
                    new_leaf[u, v, s, s_next] = -np.log(4.0 * S)
    graph.edge_data[leaf_edge][0] = new_leaf

print(f"\nSCT: u={[u_hat.get(k,0) for k in range(1,N+1)]}, v={[v_hat.get(k,0) for k in range(1,N+1)]}")
print(f"FB:  u={u_fb}, v={v_fb}")
print(f"Sent: u={u_msg.tolist()}, v={v_msg.tolist()}")
