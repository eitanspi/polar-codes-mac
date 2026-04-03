"""Debug: trace calcRight internals for the mismatch case."""
import sys
import numpy as np
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

from polar.channels_memory import ISIMAC
from polar.channels import GaussianMAC
from polar.encoder import polar_encode, bit_reversal_perm
from polar.design import make_path
from polar.decoder_trellis_sct import (
    _CompGraphTrellis, _circ_conv_trellis, _norm_prod_trellis, _logsumexp_axis,
)
from polar.decoder_interleaved import (
    _CompGraph, _circ_conv_batch, _norm_prod_batch,
    _LOG_HALF, _LOG_QUARTER,
)
from polar.efficient_decoder import build_log_W_leaf

_NEG_INF = -np.inf
N = 2; n = 1; sigma2 = 0.1; S = 4
ch_ml = GaussianMAC(sigma2=sigma2)
ch_mem = ISIMAC(sigma2=sigma2, h=0.0)

Z = np.array([0.6196202715115012, -0.6403756977145533])

log_W_ml = build_log_W_leaf(list(Z), ch_ml)
log_W_sct = ch_mem.build_leaf_tensors(list(Z))

# Build graphs
graph_ml = _CompGraph(n, log_W_ml)
graph_sct = _CompGraphTrellis(n, log_W_sct, S)

# Step 0: U1=0 (frozen), set leaves
# ML leaf
leaf_ml = np.full((2,2), _NEG_INF)
leaf_ml[0, 0] = _LOG_HALF; leaf_ml[0, 1] = _LOG_HALF
graph_ml.edge_data[2][0] = leaf_ml

# SCT leaf (uniform states)
leaf_sct = np.full((2,2,S,S), _NEG_INF)
leaf_sct[0, :, :, :] = -np.log(2.0 * S * S)
graph_sct.edge_data[2][0] = leaf_sct

# Now manually compute calcRight at vertex 1:
# right = norm_prod(parent[1:], circ_conv(left, parent[:1]))

# ML:
parent_ml = graph_ml.edge_data[1]  # (2, 2, 2)
left_ml = graph_ml.edge_data[2]    # (1, 2, 2)

cc_ml = _circ_conv_batch(left_ml, parent_ml[:1])  # (1, 2, 2)
raw_ml = parent_ml[1:] + cc_ml  # before normalization

print("=== ML ===")
print(f"parent[0] = {parent_ml[0]}")
print(f"parent[1] = {parent_ml[1]}")
print(f"left[0]   = {left_ml[0]}")
print(f"cc[0]     = {cc_ml[0]}")
print(f"raw = parent[1] + cc = {raw_ml[0]}")

# Normalize
total_ml = np.logaddexp(np.logaddexp(raw_ml[0,0,0], raw_ml[0,0,1]),
                        np.logaddexp(raw_ml[0,1,0], raw_ml[0,1,1]))
print(f"raw normalized = {raw_ml[0] - total_ml}")

# SCT:
parent_sct = graph_sct.edge_data[1]  # (2, 2, 2, S, S)
left_sct = graph_sct.edge_data[2]    # (1, 2, 2, S, S)

cc_sct = _circ_conv_trellis(left_sct, parent_sct[:1], S)  # (1, 2, 2, S, S)
raw_sct = parent_sct[1:] + cc_sct  # before normalization

print("\n=== SCT ===")
marg = lambda t: _logsumexp_axis(t.reshape(2,2,-1), axis=2)

print(f"parent[0] marg = {marg(parent_sct[0])}")
print(f"parent[1] marg = {marg(parent_sct[1])}")
print(f"left[0] marg   = {marg(left_sct[0])}")
print(f"cc[0] marg     = {marg(cc_sct[0])}")
print(f"raw marg       = {marg(raw_sct[0])}")

# Now let's look at specific state entries in raw_sct
print(f"\nraw_sct[0] non-inf entries:")
for x in range(2):
    for y in range(2):
        for s in range(S):
            for sp in range(S):
                v = raw_sct[0, x, y, s, sp]
                if np.isfinite(v):
                    print(f"  [{x},{y},{s},{sp}] = {v:.4f}")

# Compare the specific entries that survive
print(f"\nFor comparison:")
print(f"  raw_ml[0,0] = {raw_ml[0,0,0]:.6f}, raw_ml[0,1] = {raw_ml[0,0,1]:.6f}")
print(f"  raw_ml[1,0] = {raw_ml[0,1,0]:.6f}, raw_ml[1,1] = {raw_ml[0,1,1]:.6f}")

# Check: do the surviving SCT entries match ML?
print(f"\n  raw_sct[0,0,0,0] = {raw_sct[0,0,0,0,0]:.6f}")  # x=0,y=0,s=0,s'=0
print(f"  raw_sct[0,1,0,1] = {raw_sct[0,0,1,0,1]:.6f}")  # x=0,y=1,s=0,s'=1
print(f"  raw_sct[1,0,0,2] = {raw_sct[0,1,0,0,2]:.6f}")  # x=1,y=0,s=0,s'=2
print(f"  raw_sct[1,1,0,3] = {raw_sct[0,1,1,0,3]:.6f}")  # x=1,y=1,s=0,s'=3
