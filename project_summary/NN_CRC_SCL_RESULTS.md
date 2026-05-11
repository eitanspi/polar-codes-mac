# NN-CRC-SCL L=4 Results

Neural CRC-aided SCL decoding: using trained NCG neural decoder with list size L=4, CRC-8 for candidate selection.

## GMAC Class B (SNR=6 dB)

| N | ku=kv | SC | NCG (greedy) | Anal. CRC-SCL L=4 | NN-SCL L=4 | NN-CRC-SCL L=4 | NN-CRC/SC |
|---|---|---|---|---|---|---|---|
| 32 | 15 | 0.0450 | 0.0503 | **0.0013** | 0.032 | **0.004** | **0.09x** |
| 64 | 31 | 0.0276 | 0.0282 | **0.0005** | 0.018 | **0.004** | **0.14x** |
| 128 | 62 | 0.0187 | 0.023 | **0.0001** | 0.006 | 0.006 | 0.32x |
| 256 | 123 | 0.006 | 0.023 | **0.0000** | 0.020 | 0.013 | 2.2x (worse) |

**Key findings:**
- NN-CRC-SCL provides 7-11x improvement over SC at N=32-64
- At N=128, NN-CRC-SCL matches SCL but CRC provides no additional gain (3 unrecoverable errors)
- At N=256, neural model degrades -- NN-CRC-SCL is worse than SC
- Analytical CRC-SCL massively outperforms NN-CRC-SCL at all N

Data: `results/crc_scl_sweep/nn_crc_scl_gmac_B.json`

## BEMAC Class B

| N | ku | kv | SC | NCG | Anal. CRC-SCL | NN-SCL L=4 | NN-CRC-SCL L=4 |
|---|---|---|---|---|---|---|---|
| 32 | 16 | 22 | 0.0097 | 0.0076 | **0.0000** | 0.006 | **0.0000** |
| 64 | 32 | 44 | 0.0032 | 0.0032 | **0.0000** | **0.0000** | **0.0000** |
| 128 | 64 | 89 | 0.0016 | 0.0017 | **0.0000** | **0.0000** | **0.0000** |

**Key findings:**
- NN-SCL already achieves zero errors at N>=64 (no CRC needed)
- NN-CRC-SCL achieves zero errors at all tested N (N=32-128)
- BEMAC is "easy" enough that neural SCL eliminates all errors

Data: `results/crc_scl_sweep/nn_crc_scl_bemac_B.json`

## ABNMAC Class B

| N | ku=kv | SC | NCG | Anal. CRC-SCL | NN-SCL L=4 | NN-CRC-SCL L=4 | NN-CRC/SC |
|---|---|---|---|---|---|---|---|
| 32 | 10 | 0.0213 | 0.0182 | **0.0022** | 0.029 | **0.012** | **0.56x** |
| 64 | 22 | 0.0438 | 0.0416 | **0.0057** | 0.039 | **0.009** | **0.21x** |

**Key findings:**
- NN-CRC-SCL provides 2-5x improvement over SC
- NN-SCL L=4 is slightly worse than greedy NCG (list decoding doesn't help much without CRC)
- CRC provides significant additional gain: 2-4x over NN-SCL alone
- Analytical CRC-SCL still outperforms NN-CRC-SCL by 2-5x

Data: `results/crc_scl_sweep/nn_crc_scl_abnmac_B.json`

## GMAC Class C

Not evaluated -- GMAC Class C neural models use a different architecture (2-class output per step instead of 4-class joint), which is incompatible with the current NeuralSCLDecoder.

## Summary

| Channel | Best NN-CRC-SCL Regime | vs SC | vs Analytical CRC-SCL |
|---|---|---|---|
| GMAC B | N=32-64 | 7-11x better | 2-3x worse |
| BEMAC B | N=32-128 | Zero errors | Same (both zero) |
| ABNMAC B | N=32-64 | 2-5x better | 2-5x worse |

The neural CRC-SCL decoder provides meaningful gains over SC but cannot match the analytical CRC-SCL decoder. The gap widens at larger N where the neural model quality degrades. For BEMAC, the channel is simple enough that neural SCL achieves perfect decoding.
