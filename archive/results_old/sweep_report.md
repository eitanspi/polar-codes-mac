# Class C NPD: GMAC Class C Sweep Report

Generated: 2026-04-11 05:31:38

## Setup

- **Channel**: Gaussian MAC, SNR = 6 dB
- **Path**: Class C (`path_i = N`), decode all U first then V
- **Rate**: 50% of per-user capacity (R_u ≈ 0.232, R_v ≈ 0.456)
- **Architecture**: single-user NPD per stage, d=16 hidden=64 (~21K params)
- **Training**: fast_ce parallel, curriculum across N with warm-starting
- **Evaluation**: 5000 codewords per N, Wilson 95% CIs

## SC Reference

| N | ku | kv | SC BLER | NPD target (1.5×) |
|---|----|----|---------|---|
| 16 | 4 | 7 | 0.1626 | 0.2439 |
| 32 | 7 | 15 | 0.0684 | 0.1026 |
| 64 | 15 | 29 | 0.0266 | 0.0399 |
| 128 | 30 | 58 | 0.0054 | 0.0081 |
| 256 | 59 | 117 | 0.0020 | 0.0030 |
| 512 | 119 | 233 | 0.0008 | 0.0012 |
| 1024 | 238 | 467 | 0.0002 | 0.0003 |

## Curriculum Sweep Results

| N | Stage 1 BLER | Stage 2 BLER | Chained BLER | 95% CI | Ratio | Pass? |
|---|---|---|---|---|---|---|
| 16 | 0.1420 | 0.0060 | 0.1362 | [0.1270, 0.1460] | 0.84x | ✓ PASS |
| 32 | 0.1160 | 0.0000 | 0.1290 | [0.1200, 0.1386] | 1.89x | ✗ FAIL |
| 64 | 0.1600 | 0.0000 | 0.1758 | [0.1655, 0.1866] | 6.61x | ✗ FAIL |

### Summary

- **PASSED at**: 16
- **FAILED at**: 32, 64

## Architecture diagram

```
       GMAC channel: z = (1-2X) + (1-2Y) + W
              │
              ▼
  ┌──────────────────────┐
  │  Stage 1 (NPD)       │     U on marginal channel
  │  decode U from raw z │     mixture LLR via z_encoder
  └──────────┬───────────┘
             │ û
             ▼
   x̂ = polar_encode(û)
             │
             ▼
    z' = z - (1 - 2·x̂)     subtract known U contribution
             │
             ▼
  ┌──────────────────────┐
  │  Stage 2 (NPD)       │     V on clean BPSK+AWGN
  │  decode V from z'    │     standard single-user NPD
  └──────────┬───────────┘
             │ v̂
             ▼
         output
```

## Reproduction

```bash
# Run smoke test (5 min)
python -u class_c_npd/smoke_test.py

# Run curriculum sweep
python -u class_c_npd/training/curriculum_sweep.py --N_list 16,32,64,128

# Generate this report
python -u class_c_npd/eval/generate_report.py
```
