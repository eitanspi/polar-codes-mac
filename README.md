# Polar Codes for the Binary-Input MAC

Implementation of successive cancellation (SC) and SC list (SCL)
decoding of polar codes for the two-user binary-input multiple
access channel, reproducing results from Önay (ISIT 2013).

## Setup

```bash
pip install -r requirements.txt
pytest tests/
```

## Quick start

```python
import numpy as np
from polar import (
    BEMAC, polar_encode, build_message,
    design_bemac, make_path,
    decode_single, decode_single_list,
)

# Design a polar code for BE-MAC
N, n = 64, 6
ku, kv = 16, 64
Au, Av, frozen_u, frozen_v, _, _ = design_bemac(n, ku, kv)
b = make_path(N, path_i=N)

# Encode
rng = np.random.default_rng(42)
u = build_message(N, rng.integers(0, 2, ku), Au)
v = build_message(N, rng.integers(0, 2, kv), Av)
x = polar_encode(u.tolist())
y = polar_encode(v.tolist())

# Channel
channel = BEMAC()
z = channel.sample_batch(np.array(x), np.array(y)).tolist()

# Decode (SC)
u_dec, v_dec = decode_single(N, z, b, frozen_u, frozen_v, channel)

# Decode (SCL, list size L=4)
u_dec, v_dec = decode_single_list(N, z, b, frozen_u, frozen_v, channel, L=4)
```

## Channels

| Channel  | Model                      | Capacity region             |
|----------|----------------------------|-----------------------------|
| BE-MAC   | Z = X + Y ∈ {0, 1, 2}     | Rx ≤ 0.5, Ry ≤ 0.5, Rx+Ry ≤ 1.5 |
| ABN-MAC  | Z = (X⊕Ex, Y⊕Ey), (Ex,Ey)~p | Rx ≤ 0.4, Ry ≤ 0.4, Rx+Ry ≤ 1.2 |

## File structure

```
polar/
  __init__.py       — public API exports
  encoder.py        — batched polar encoder (numpy + optional TF)
  channels.py       — BEMAC, ABNMAC channel classes
  design.py         — analytical Bhattacharyya code design
  design_mc.py      — Monte Carlo genie-aided code design
  decoder.py        — O(N log N) SC MAC decoder (public)
  decoder_scl.py    — O(L·N log N) SCL MAC decoder (public)
  _decoder_base.py  — O(N²) recursive SC decoder (internal fallback)
  _decoder_numba.py — numba-accelerated O(N²) decoder (internal)
  _decoder_scl_base.py — O(L·N²) SCL decoder (internal)
scripts/
  sim_bemac_class_c.py — BE-MAC BLER simulation
  plot_results.py      — plot simulation results
tests/
  test_correctness.py  — pytest suite
```

## Decoders

The primary decoders are `decoder.py` (SC) and `decoder_scl.py` (SCL),
both using an O(N log N) LLR-based factor-graph traversal. For paths
other than 0^N 1^N and 1^N 0^N, `decoder.py` automatically falls back
to the O(N²) recursive decoder in `_decoder_base.py`.

Files prefixed with `_` are internal implementation details and should
not be imported directly.

## References

Önay, G. (2013). *Successive Cancellation Decoding of Polar Codes for the
Two-User Binary-Input MAC*. IEEE ISIT 2013.

Arikan, E. (2009). *Channel Polarization: A Method for Constructing
Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels*.
IEEE Trans. Inf. Theory.
