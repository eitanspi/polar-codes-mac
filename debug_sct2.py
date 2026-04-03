"""Detailed step-by-step trace of SCT vs ML for the failing trial."""
import sys
import numpy as np
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

from polar.channels_memory import ISIMAC
from polar.channels import GaussianMAC
from polar.encoder import polar_encode, bit_reversal_perm
from polar.design import make_path
from polar.decoder_trellis_sct import (
    _CompGraphTrellis, _norm_prod_trellis_single, _logsumexp_axis,
    _circ_conv_trellis, _norm_prod_trellis,
)
from polar.decoder_interleaved import (
    _CompGraph, _norm_prod_single, _circ_conv_batch, _norm_prod_batch,
    _LOG_HALF, _LOG_QUARTER,
)
from polar.efficient_decoder import build_log_W_leaf

_NEG_INF = -np.inf

N = 2
n = 1
sigma2 = 0.1

ch_memoryless = GaussianMAC(sigma2=sigma2)
ch_memory = ISIMAC(sigma2=sigma2, h=0.0)
S = ch_memory.num_states

b = make_path(N, N)  # [0, 0, 1, 1]
frozen_u = {1: 0}
frozen_v = {1: 0}

# Use the failing case from trial 43
u_msg = np.array([0, 1])
v_msg = np.array([0, 0])
X = np.array(polar_encode(u_msg.tolist()))
Y = np.array(polar_encode(v_msg.tolist()))

np.random.seed(123)
for _ in range(44):  # Skip to trial 43
    np.random.randint(0, 2)
    np.random.randint(0, 2)

# Reconstruct Z
Z_ml = np.array([0.6196202715115012, -0.6403756977145533])

log_W_ml = build_log_W_leaf(list(Z_ml), ch_memoryless)
log_W_sct = ch_memory.build_leaf_tensors(list(Z_ml))

# Create both graphs
graph_ml = _CompGraph(n, log_W_ml)
graph_sct = _CompGraphTrellis(n, log_W_sct, S)

# For N=2, n=1:
# Edges: 1 (root, size=2), 2 (left leaf), 3 (right leaf)
# Vertices: 1 (root vertex, has edges 2 and 3)
# Path b = [0, 0, 1, 1] -> steps: U1, U2, V1, V2

def marg(tensor):
    """Marginalize (2, 2, S, S) -> (2, 2)"""
    return _logsumexp_axis(tensor.reshape(2, 2, -1), axis=2)

print("=" * 60)
print("Step 0: U1 at position i_t=1")
print("=" * 60)

# leaf_edge = 1 + 2 - 1 = 2 (left child of vertex 1)
# target_vertex = 2 >> 1 = 1
# Already at vertex 1, no navigation needed

# Save leaf's getAsParent
temp_ml = graph_ml.edge_data[2][0].copy()  # (2, 2)
temp_sct = graph_sct.edge_data[2][0].copy()  # (2, 2, S, S)

print(f"ML temp (leaf 2, entry 0): {np.exp(temp_ml)}")
print(f"SCT temp (leaf 2, entry 0) marg: {np.exp(marg(temp_sct))}")

# calc_left at vertex 1 (leaf_edge=2 is even -> left child)
graph_ml.calc_left(1)
graph_sct.calc_left(1)

top_down_ml = graph_ml.edge_data[2][0]
top_down_sct = graph_sct.edge_data[2][0]

print(f"\nML top_down: {np.exp(top_down_ml)}")
print(f"SCT top_down marg: {np.exp(marg(top_down_sct))}")

# Combined
combined_ml = _norm_prod_single(top_down_ml, temp_ml)
combined_sct = _norm_prod_trellis_single(top_down_sct, temp_sct)
combined_sct_marg = marg(combined_sct)

print(f"\nML combined: {np.exp(combined_ml)}")
print(f"SCT combined marg: {np.exp(combined_sct_marg)}")

# Decision for U1: frozen -> bit = 0
print(f"U1: frozen -> bit = 0")

# Set leaf
# ML
new_leaf_ml = np.full((2, 2), _NEG_INF)
new_leaf_ml[0, 0] = _LOG_HALF
new_leaf_ml[0, 1] = _LOG_HALF
graph_ml.edge_data[2][0] = new_leaf_ml

# SCT
new_leaf_sct = np.full((2, 2, S, S), _NEG_INF)
for v in range(2):
    s_next = ch_memory._encode_state(0, v)
    for s in range(S):
        new_leaf_sct[0, v, s, s_next] = -np.log(2.0 * S)
graph_sct.edge_data[2][0] = new_leaf_sct

print(f"\nML new leaf: {np.exp(new_leaf_ml)}")
print(f"SCT new leaf marg: {np.exp(marg(new_leaf_sct))}")

print("\n" + "=" * 60)
print("Step 1: U2 at position i_t=2")
print("=" * 60)

# leaf_edge = 2 + 2 - 1 = 3 (right child of vertex 1)
# target_vertex = 3 >> 1 = 1
# Already at vertex 1

temp_ml2 = graph_ml.edge_data[3][0].copy()
temp_sct2 = graph_sct.edge_data[3][0].copy()

print(f"ML temp (leaf 3, entry 0): {np.exp(temp_ml2)}")
print(f"SCT temp (leaf 3, entry 0) marg: {np.exp(marg(temp_sct2))}")

# calc_right at vertex 1
graph_ml.calc_right(1)
graph_sct.calc_right(1)

top_down_ml2 = graph_ml.edge_data[3][0]
top_down_sct2 = graph_sct.edge_data[3][0]

print(f"\nML top_down: {np.exp(top_down_ml2)}")
print(f"SCT top_down marg: {np.exp(marg(top_down_sct2))}")

combined_ml2 = _norm_prod_single(top_down_ml2, temp_ml2)
combined_sct2 = _norm_prod_trellis_single(top_down_sct2, temp_sct2)

print(f"\nML combined: {np.exp(combined_ml2)}")
print(f"SCT combined marg: {np.exp(marg(combined_sct2))}")

# Decision for U2: info bit
p0_ml = np.logaddexp(combined_ml2[0, 0], combined_ml2[0, 1])
p1_ml = np.logaddexp(combined_ml2[1, 0], combined_ml2[1, 1])
bit_ml = 1 if p1_ml > p0_ml else 0

bit_post_sct2 = marg(combined_sct2)
p0_sct = np.logaddexp(bit_post_sct2[0, 0], bit_post_sct2[0, 1])
p1_sct = np.logaddexp(bit_post_sct2[1, 0], bit_post_sct2[1, 1])
bit_sct = 1 if p1_sct > p0_sct else 0

print(f"\nU2 decision: ML -> p0={np.exp(p0_ml):.6f}, p1={np.exp(p1_ml):.6f}, bit={bit_ml}")
print(f"U2 decision: SCT -> p0={np.exp(p0_sct):.6f}, p1={np.exp(p1_sct):.6f}, bit={bit_sct}")

print("\n--- Key: checking calcRight internals ---")
# ML calcRight: right = norm_prod(parent[l:], circ_conv(left, parent[:l]))
parent_ml = graph_ml.edge_data[1]  # after calc_left it may be modified...
# Actually we need to reconstruct. Let me just show the key intermediate values.

# Let me re-create fresh graphs and trace calcRight carefully
print("\n--- Re-trace with fresh graphs ---")
graph_ml_fresh = _CompGraph(n, log_W_ml)
graph_sct_fresh = _CompGraphTrellis(n, log_W_sct, S)

# After step 0 (U1 decided as 0), set leaf 2:
graph_ml_fresh.edge_data[2][0] = new_leaf_ml.copy()
graph_sct_fresh.edge_data[2][0] = new_leaf_sct.copy()

# Now calcRight at vertex 1:
# ML: right = norm_prod(parent[1:], circ_conv(left, parent[:1]))
parent_ml_f = graph_ml_fresh.edge_data[1]  # (2, 2, 2)
left_ml_f = graph_ml_fresh.edge_data[2]    # (1, 2, 2) after setting leaf
print(f"ML parent[0]: {np.exp(parent_ml_f[0])}")
print(f"ML parent[1]: {np.exp(parent_ml_f[1])}")
print(f"ML left[0]: {np.exp(left_ml_f[0])}")

cc_ml = _circ_conv_batch(left_ml_f, parent_ml_f[:1])
print(f"ML circ_conv(left, parent[:1]): {np.exp(cc_ml[0])}")

right_ml_f = _norm_prod_batch(parent_ml_f[1:], cc_ml)
print(f"ML norm_prod(parent[1:], cc): {np.exp(right_ml_f[0])}")

parent_sct_f = graph_sct_fresh.edge_data[1]  # (2, 2, 2, S, S)
left_sct_f = graph_sct_fresh.edge_data[2]    # (1, 2, 2, S, S) after setting leaf

print(f"\nSCT parent[0] marg: {np.exp(marg(parent_sct_f[0]))}")
print(f"SCT parent[1] marg: {np.exp(marg(parent_sct_f[1]))}")
print(f"SCT left[0] marg: {np.exp(marg(left_sct_f[0]))}")

cc_sct = _circ_conv_trellis(left_sct_f, parent_sct_f[:1], S)
print(f"SCT circ_conv(left, parent[:1]) marg: {np.exp(marg(cc_sct[0]))}")

right_sct_f = _norm_prod_trellis(parent_sct_f[1:], cc_sct)
print(f"SCT norm_prod(parent[1:], cc) marg: {np.exp(marg(right_sct_f[0]))}")
