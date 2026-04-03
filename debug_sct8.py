"""Trace calcParent at vertex 2 for the N=4 failing case."""
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

_NEG_INF = -np.inf

N = 4; n = 2; snr_db = 20; h = 0.5; S = 4
channel = ISIMAC.from_snr_db(snr_db, h=h)
b = make_path(N, N)
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

log_W = channel.build_leaf_tensors(list(Z))
graph = _CompGraphTrellis(n, log_W, S)

marg = lambda t: _logsumexp_axis(t.reshape(2, 2, -1), axis=2)

# Step 0: U1 at pos 1. Navigate to vertex 2 (calc_left at vertex 1).
graph.step_to(2)  # calc_left at vertex 1

# Step 0: U1 (frozen=0)
temp0 = graph.edge_data[4][0].copy()
graph.calc_left(2)
td0 = graph.edge_data[4][0]
combined0 = _norm_prod_trellis_single(td0, temp0)
# Decision: U1=0 (frozen)
# Set leaf
new_leaf0 = np.full((2,2,S,S), _NEG_INF)
for v in range(2):
    s_next = channel._encode_state(0, v)
    for s in range(S):
        new_leaf0[0, v, s, s_next] = -np.log(2.0 * S)
graph.edge_data[4][0] = new_leaf0

# Step 1: U2 at pos 2. leaf_edge=5, vertex=2. Already there.
temp1 = graph.edge_data[5][0].copy()
graph.calc_right(2)
td1 = graph.edge_data[5][0]
combined1 = _norm_prod_trellis_single(td1, temp1)
# Decision: U2=0 (frozen)
new_leaf1 = np.full((2,2,S,S), _NEG_INF)
for v in range(2):
    s_next = channel._encode_state(0, v)
    for s in range(S):
        new_leaf1[0, v, s, s_next] = -np.log(2.0 * S)
graph.edge_data[5][0] = new_leaf1

print("After U1=0, U2=0 decided:")
print(f"  edge[4] (pos 1 leaf) marg: {np.exp(marg(graph.edge_data[4][0])).round(5)}")
print(f"  edge[5] (pos 2 leaf) marg: {np.exp(marg(graph.edge_data[5][0])).round(5)}")

# Step 2: U3 at pos 3. leaf_edge=6, vertex=3.
# Navigation from vertex 2 to vertex 3:
# calcParent at vertex 2 -> vertex 1
# calc_right at vertex 1 -> vertex 3

# Before calcParent, check edge[4] and edge[5]
print(f"\nBefore calcParent at vertex 2:")
print(f"  edge[4] (left of vertex 2): 1 entry")
e4 = graph.edge_data[4][0]
for x in range(2):
    for y in range(2):
        for s in range(S):
            for sp in range(S):
                if np.isfinite(e4[x,y,s,sp]):
                    print(f"    [{x},{y},{s},{sp}] = {e4[x,y,s,sp]:.4f}")

print(f"  edge[5] (right of vertex 2): 1 entry")
e5 = graph.edge_data[5][0]
for x in range(2):
    for y in range(2):
        for s in range(S):
            for sp in range(S):
                if np.isfinite(e5[x,y,s,sp]):
                    print(f"    [{x},{y},{s},{sp}] = {e5[x,y,s,sp]:.4f}")

# calcParent at vertex 2
graph.calc_parent(2)  # This will write to edge[2]

print(f"\nAfter calcParent at vertex 2:")
print(f"  edge[2]: 2 entries")
e2 = graph.edge_data[2]
for l in range(2):
    print(f"  Entry {l}:")
    for x in range(2):
        for y in range(2):
            for s in range(S):
                for sp in range(S):
                    if np.isfinite(e2[l,x,y,s,sp]):
                        print(f"    [{x},{y},{s},{sp}] = {e2[l,x,y,s,sp]:.4f}")
    print(f"    marg: {np.exp(marg(e2[l])).round(5)}")

# Now calc_right at vertex 1
print(f"\nRoot (edge[1]) entries:")
e1 = graph.edge_data[1]
for l in range(4):
    br = bit_reversal_perm(n)
    print(f"  Entry {l} (pos {br[l]}) marg: {np.exp(marg(e1[l])).round(5)}")
    for x in range(2):
        for y in range(2):
            for s in range(S):
                for sp in range(S):
                    if np.isfinite(e1[l,x,y,s,sp]):
                        print(f"    [{x},{y},{s},{sp}] = {e1[l,x,y,s,sp]:.4f}")

# manual calc_right at vertex 1:
# right = norm_prod(parent[l:], circ_conv(left, parent[:l]))
# left = edge[2] (2 entries), parent = edge[1] (4 entries)
# l = 2
left = graph.edge_data[2]  # (2, 2, 2, S, S)
parent = graph.edge_data[1]  # (4, 2, 2, S, S)

cc = _circ_conv_trellis(left, parent[:2], S)  # (2, 2, 2, S, S)
print(f"\ncirc_conv(left, parent[:2]):")
for l_idx in range(2):
    print(f"  Entry {l_idx} marg: {np.exp(marg(cc[l_idx])).round(5)}")

right = _norm_prod_trellis(parent[2:], cc)  # (2, 2, 2, S, S)
print(f"\nnorm_prod(parent[2:], cc):")
for l_idx in range(2):
    print(f"  Entry {l_idx} marg: {np.exp(marg(right[l_idx])).round(5)}")
