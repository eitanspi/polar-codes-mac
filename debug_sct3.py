"""Test SCT correctness for h=0 at high SNR."""
import sys
import numpy as np
sys.path.insert(0, '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2')

from polar.channels_memory import ISIMAC
from polar.channels import GaussianMAC
from polar.encoder import polar_encode
from polar.design import design_gmac, make_path
from polar.decoder_trellis_sct import decode_single as sct_decode_single
from polar.decoder_interleaved import decode_single as memoryless_decode_single

N = 8
n = 3
sigma2 = 0.001  # high SNR

ch_memoryless = GaussianMAC(sigma2=sigma2)
ch_memory = ISIMAC(sigma2=sigma2, h=0.0)

b = make_path(N, N)
ku, kv = 2, 4
Au, Av, frozen_u, frozen_v, _, _ = design_gmac(n, ku, kv, sigma2)

np.random.seed(42)
n_trials = 100
sct_correct = 0
ml_correct = 0
both_agree = 0

for trial in range(n_trials):
    u_msg = np.zeros(N, dtype=int)
    v_msg = np.zeros(N, dtype=int)
    for pos in Au:
        u_msg[pos - 1] = np.random.randint(0, 2)
    for pos in Av:
        v_msg[pos - 1] = np.random.randint(0, 2)

    X = np.array(polar_encode(u_msg.tolist()))
    Y = np.array(polar_encode(v_msg.tolist()))
    Z = ch_memoryless.sample_batch(X, Y)

    u_ml, v_ml = memoryless_decode_single(N, list(Z), b, frozen_u, frozen_v, ch_memoryless)
    u_sct, v_sct = sct_decode_single(N, list(Z), b, frozen_u, frozen_v, ch_memory)

    ml_ok = (u_ml == u_msg.tolist()) and (v_ml == v_msg.tolist())
    sct_ok = (u_sct == u_msg.tolist()) and (v_sct == v_msg.tolist())

    if ml_ok:
        ml_correct += 1
    if sct_ok:
        sct_correct += 1
    if u_ml == u_sct and v_ml == v_sct:
        both_agree += 1

    if not sct_ok and trial < 5:
        print(f"Trial {trial}: SCT wrong")
        print(f"  sent u={u_msg.tolist()}, v={v_msg.tolist()}")
        print(f"  SCT: u={u_sct}, v={v_sct}")
        print(f"  ML:  u={u_ml}, v={v_ml}")

print(f"\nML correct:  {ml_correct}/{n_trials}")
print(f"SCT correct: {sct_correct}/{n_trials}")
print(f"Both agree:  {both_agree}/{n_trials}")
