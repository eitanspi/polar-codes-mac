# Paper-Style BLER Tables

Generated from `results/paper_style/` baseline and NPD evaluations.
All entries include Wilson 95% confidence intervals.
**Updated 2026-04-26** with GPU-trained curriculum models.

## Table A: ISI-MAC (h=0.3, SNR=6 dB)

| N | ku/kv | Chained Trellis SC | Memoryless SC | NPD (best) | Source | NPD/Trellis |
|---|-------|-------------------|---------------|------------|--------|-------------|
| 16 | 4/7 | 0.1689 [0.1617, 0.1764] | 0.1866 [0.1791, 0.1944] | **0.1376** [0.1283, 0.1474] | GPU curriculum | **0.81** |
| 32 | 7/15 | 0.0822 [0.0770, 0.0877] | 0.1129 [0.1068, 0.1193] | **0.0566** [0.0505, 0.0634] | GPU curriculum | **0.69** |
| 64 | 15/29 | 0.0407 [0.0370, 0.0448] | 0.0790 [0.0739, 0.0845] | **0.0278** [0.0236, 0.0327] | GPU curriculum | **0.68** |
| 128 | 30/58 | 0.0223 [0.0196, 0.0254] | 0.1009 [0.0951, 0.1070] | 0.0740 [0.0671, 0.0816] | CPU standalone | 3.32 |
| 256 | 59/117 | 0.0061 [0.0048, 0.0078] | 0.2256 [0.2175, 0.2339] | 0.0120 [0.0093, 0.0154] | CPU d=64 | 1.97 |
| 512 | 119/233 | 0.0041 [0.0030, 0.0056] | 0.5018 [0.4920, 0.5116] | --- | --- | --- |
| 1024 | 238/467 | 0.0074 [0.0059, 0.0093] | 0.8704 [0.8637, 0.8768] | --- | --- | --- |

**Headline result:** NPD beats trellis SC by 19-32% at N=16,32,64 with GPU-trained curriculum models.

## Table B: Ising MAC (p_flip=0.1, sigma2=0.251)

| N | ku/kv | Chained Trellis SC | Memoryless SC | NPD d=16 h=100 | NPD/Memoryless |
|---|-------|-------------------|---------------|----------------|----------------|
| 16 | 4/7 | 0.5704 [0.5566, 0.5841] | 0.6160 [0.6024, 0.6294] | 0.5916 [0.5779, 0.6051] | 0.96 |
| 32 | 7/15 | 0.6872 [0.6742, 0.6999] | 0.7840 [0.7724, 0.7952] | 0.7658 [0.7539, 0.7773] | 0.98 |
| 64 | 15/29 | 0.8976 [0.8889, 0.9057] | 0.9410 [0.9341, 0.9472] | --- | --- |
| 128 | 30/58 | --- | --- | (training: BLER~0.99) | --- |

Note: Ising MAC BLER >50% at all N -- channel is extremely lossy at this rate.

## Table C: MA-AGN MAC (alpha=0.3, SNR=6 dB)

| N | ku/kv | Memoryless SC | NPD (best) | Source | NPD/Memoryless |
|---|-------|---------------|------------|--------|----------------|
| 16 | 4/7 | 0.1654 [0.1554, 0.1760] | **0.1438** [0.1343, 0.1538] | BiGRU d=16 h=64 | **0.87** |
| 32 | 7/15 | 0.0696 [0.0629, 0.0770] | 0.1134 [0.1049, 0.1225] | BiGRU d=32 h=128 | 1.63 |
| 64 | 15/29 | 0.0292 [0.0249, 0.0342] | 0.0292 [0.0249, 0.0342] | d=16 h=100 | 1.00 |
| 128 | 30/58 | 0.0052 [0.0036, 0.0076] | 0.1014 [0.0933, 0.1101] | d=16 h=100 | 19.50 |
| 256 | 59/117 | --- | (training: ~0.033 at 40K) | d=16 h=100 CPU | --- |

---
Notes:
- Wilson 95% CI format: BLER [CI_low, CI_high]
- NPD/baseline < 1.0 means NPD is better (bold)
- Trellis SC uses chained 2-stage decoder with forward-backward on 2-state trellis
- Memoryless SC ignores channel memory, uses GMAC decoder
- All baselines use 10K CW; NPD uses 5K CW
- GPU curriculum = d=16 h=100 BiGRU trained on GPU with N=16->32->64->128 warm-starting
