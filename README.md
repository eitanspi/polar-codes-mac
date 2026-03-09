# Polar Codes for the Two-User Binary Erasure MAC

Polar code construction and successive cancellation (SC / SCL) decoding
for the two-user binary erasure multiple access channel (BE-MAC),
reproducing results from **Onay (ISIT 2013)**. Supports arbitrary
monotone-chain decoding paths via the computational graph approach of
Ren, Nasser & Bhatt (2025). All simulations use block length N = 1024.

## Results

### Rate region (N = 1024, SCL L = 32)

![Rate region](figures/fig1_rate_region.pdf)

Operating points with zero block errors (BLER < 10^{-3}) for all three
code classes, compared against the BE-MAC capacity region.

### BLER vs sum-rate — Class B (N = 1024, MC design)

![BLER vs sum-rate](figures/fig2_bler_class_B.pdf)

SC (L = 1) vs SCL (L = 32) comparison along the Class B rate direction.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

```python
import numpy as np
from polar import (
    BEMAC, polar_encode, build_message,
    design_bemac, make_path,
    decode_single, decode_single_list,
)

# Parameters
N, n = 1024, 10
ku, kv = 256, 512
Au, Av, frozen_u, frozen_v, _, _ = design_bemac(n, ku, kv)
b = make_path(N, path_i=N)  # extreme path: all U then all V

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

# Decode (SCL, L=32)
u_dec, v_dec = decode_single_list(N, z, b, frozen_u, frozen_v, channel, L=32)
```

For Monte Carlo genie-aided designs (included in `designs/`):

```python
from polar.design_mc import design_from_file

Au, Av, frozen_u, frozen_v, _, _, _ = design_from_file(
    "designs/bemac_B_n10.npz", n=10, ku=320, kv=448
)
```

## Code classes

Three code classes from Onay (ISIT 2013), each targeting a different
rate direction on the BE-MAC dominant face (R_u + R_v = 1.5):

| Class | (R_u, R_v) direction | path_i (N=1024) | Description |
|-------|---------------------|-----------------|-------------|
| A     | (0.75, 0.75)        | 384             | Symmetric rates |
| B     | (0.625, 0.875)      | 512             | Moderate asymmetry |
| C     | (0.5, 1.0)          | 1024 (extreme)  | Maximum asymmetry |

The `path_i` parameter controls the monotone-chain decoding order:
the first `path_i` steps decode U bits, then N steps decode V bits,
then the remaining `N - path_i` steps decode U bits.

## Reproducing results

Pre-computed MC designs are included in `designs/`. To regenerate them:

```bash
python scripts/run_design.py --class A --N 1024 --hours 1
python scripts/run_design.py --class B --N 1024 --hours 1
```

Run BLER simulations (results saved to `results/`):

```bash
# Class C — analytical design (extreme path, fast)
python scripts/simulate.py --class C --L 1 --N 1024 --hours 0.5
python scripts/simulate.py --class C --L 32 --N 1024 --hours 1

# Class A — MC design
python scripts/simulate.py --class A --L 1 --N 1024 --hours 0.5 --design mc
python scripts/simulate.py --class A --L 32 --N 1024 --hours 2 --design mc

# Class B — MC design
python scripts/simulate.py --class B --L 1 --N 1024 --hours 0.5 --design mc
python scripts/simulate.py --class B --L 32 --N 1024 --hours 2 --design mc
```

Generate figures:

```bash
python scripts/plot_results.py --rate-region -o figures/fig1_rate_region.pdf \
    results/sim_bemac_A_mc_L32.json results/sim_bemac_B_mc_L32.json \
    results/sim_bemac_C_L32.json

python scripts/plot_results.py -o figures/fig2_bler_class_B.pdf \
    results/sim_bemac_B_mc_L1.json results/sim_bemac_B_mc_L32.json
```

## File structure

```
polar/
  __init__.py          Public API
  encoder.py           Polar encoder (bit-reversal + GF(2) butterfly)
  channels.py          BE-MAC and ABN-MAC channel models
  design.py            Analytical Bhattacharyya code design
  design_mc.py         Monte Carlo genie-aided code design
  decoder.py           O(N log N) SC decoder (LLR / tensor-based)
  decoder_scl.py       O(LN log N) SCL decoder
  _decoder_numba.py    Numba-accelerated internals
scripts/
  simulate.py          BLER simulation for all code classes
  run_design.py        MC genie design file generation
  plot_results.py      Publication-quality PDF figures
tests/
  test_correctness.py  Encoder, channel, design, and decoder tests
designs/               Pre-computed MC design files (.npz)
results/               Simulation results (.json)
figures/               Generated figures (.pdf)
```

## BE-MAC capacity region

The binary erasure MAC has Z = X + Y over {0, 1, 2}:
- R_u <= 1,  R_v <= 1,  R_u + R_v <= 1.5
- The dominant face connects (1, 0.5) and (0.5, 1)

## References

- Onay, G. (2013). *Successive Cancellation Decoding of Polar Codes for
  the Two-User Binary-Input MAC*. IEEE ISIT 2013.
- Arikan, E. (2009). *Channel Polarization: A Method for Constructing
  Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels*.
  IEEE Trans. Inf. Theory, 55(7).
- Ren, Y., Nasser, R., & Bhatt, A. (2025). *The Computational Graph of
  Polar Codes*. arXiv:2509.03128.
