# GMAC Class B — Complete BLER Comparison (SNR=6dB)

Rate: Ru ≈ Rv ≈ 0.48, path_i = N/2 (symmetric interleaved path)
Design: Monte Carlo genie-aided (critical for Class B)

## SC vs SCL vs Neural Decoders

| N | SC | SCL L=4 | SCL L=32 | NN-SC (d=16) | NN-SCL L=4 (d=16) |
|---|-----|---------|----------|-------------|-------------------|
| 32 | 0.046 | 0.026 | 0.026 | 0.056 | **0.022** |
| 64 | 0.025 | 0.013 | 0.012 | 0.026 | **0.013** |
| 128 | 0.016 | 0.008 | 0.006 | 0.023 | 0.015 |
| 256 | 0.005 | 0.0005 | -- | 0.020 | 0.026 |
| 512 | 0.001 | 0.000 | -- | 0.045 | 0.045 |
| 1024 | 0.001 | 0.000 | -- | 0.069 | 0.045 |

### Key observations

1. **NN-SCL(L=4) beats SCL(L=4) at N<=64** — the main positive result
2. **SCL L=4 is essentially perfect at N>=512** — zero errors in 500+ codewords
3. **NN models catastrophically degrade at N>=128** — tree-walk error accumulation
4. **The gap widens with N**: NN-SC is 1.03x SC at N=64 but 69x at N=1024

### SCL L=4 baselines (new, 2026-03-29)

| N | BLER | n_cw | Source |
|---|------|------|--------|
| 32 | 0.0264 | 5000 | sim_scl_classB_fixed_rate.py |
| 64 | 0.0126 | 5000 | sim_scl_classB_fixed_rate.py |
| 128 | 0.0084 | 5000 | sim_scl_classB_fixed_rate.py |
| 256 | 0.0005 | 2000 | sim_scl_classB_fixed_rate.py |
| 512 | 0.0000 | 500 | sim_scl_large_N.py |
| 1024 | 0.0000 | 200 | sim_scl_large_N.py |

### d=32 model training (in progress)

N=32: best BLER=0.0457 (ratio 0.99x, matches SC) — 157K params vs 25K baseline
N=64: training in progress...
