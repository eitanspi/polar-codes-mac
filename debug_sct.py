"""Debug SCT vs memoryless for N=2, h=0."""
import sys
import numpy as np
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

from polar.channels_memory import ISIMAC
from polar.channels import GaussianMAC
from polar.encoder import polar_encode
from polar.design import make_path
from polar.decoder_trellis_sct import (
    decode_single as sct_decode_single,
    _CompGraphTrellis, _norm_prod_trellis_single, _logsumexp_axis,
)
from polar.decoder_interleaved import (
    decode_single as memoryless_decode_single,
    _CompGraph, _norm_prod_single,
)
from polar.efficient_decoder import build_log_W_leaf

N = 2
n = 1
sigma2 = 0.1

ch_memoryless = GaussianMAC(sigma2=sigma2)
ch_memory = ISIMAC(sigma2=sigma2, h=0.0)

b = make_path(N, N)  # [0, 0, 1, 1]
frozen_u = {1: 0}
frozen_v = {1: 0}

np.random.seed(123)

# Find a mismatch
for trial in range(50):
    u_info = np.random.randint(0, 2)
    v_info = np.random.randint(0, 2)
    u_msg = np.array([0, u_info])
    v_msg = np.array([0, v_info])
    X = np.array(polar_encode(u_msg.tolist()))
    Y = np.array(polar_encode(v_msg.tolist()))
    Z_ml = ch_memoryless.sample_batch(X, Y)

    u_ml, v_ml = memoryless_decode_single(N, list(Z_ml), b, frozen_u, frozen_v, ch_memoryless)
    u_sct, v_sct = sct_decode_single(N, list(Z_ml), b, frozen_u, frozen_v, ch_memory)

    if u_ml != u_sct or v_ml != v_sct:
        print(f"Trial {trial}: MISMATCH")
        print(f"  sent u={u_msg.tolist()}, v={v_msg.tolist()}")
        print(f"  X={X.tolist()}, Y={Y.tolist()}")
        print(f"  Z={Z_ml.tolist()}")
        print(f"  ML: u={u_ml}, v={v_ml}")
        print(f"  SCT: u={u_sct}, v={v_sct}")

        # Compare leaf tensors
        log_W_ml = build_log_W_leaf(list(Z_ml), ch_memoryless)
        log_W_sct = ch_memory.build_leaf_tensors(list(Z_ml))
        print(f"\n  Memoryless leaf tensors (N, 2, 2):")
        for t in range(N):
            print(f"    t={t}: {np.exp(log_W_ml[t])}")
        print(f"\n  SCT leaf tensors (N, 2, 2, S, S):")
        for t in range(N):
            # Marginalize over states
            marg = _logsumexp_axis(log_W_sct[t].reshape(2, 2, -1), axis=2)
            print(f"    t={t} marginalized: {np.exp(marg)}")
            print(f"    t={t} full shape: finite entries = {np.sum(np.isfinite(log_W_sct[t]))}")

        # Now step through SCT decoder manually
        print(f"\n  --- Step-by-step SCT decode ---")
        S = ch_memory.num_states
        from polar.encoder import bit_reversal_perm
        br = bit_reversal_perm(n)
        print(f"  bit_reversal: {br}")
        print(f"  Path b: {b}")

        # Build graph and trace
        graph_sct = _CompGraphTrellis(n, log_W_sct, S)
        graph_ml = _CompGraph(n, log_W_ml)

        print(f"\n  Root edge (SCT):")
        root_sct = graph_sct.edge_data[1]
        for t in range(N):
            marg = _logsumexp_axis(root_sct[t].reshape(2, 2, -1), axis=2)
            print(f"    t={t}: marg={np.exp(marg)}")

        print(f"  Root edge (ML):")
        root_ml = graph_ml.edge_data[1]
        for t in range(N):
            print(f"    t={t}: {np.exp(root_ml[t])}")

        break
