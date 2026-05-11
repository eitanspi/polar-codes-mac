# Chained NPD at N=128 — ISI-MAC Class C

Session date: 2026-04-17.
Channel: `ISIMAC(h=0.3)` at SNR = 6 dB (sigma^2 = 10^(-6/10)).
Design: GMAC_C proxy via `designs/gmac_C_n7_snr6dB.npz`, ku=30, kv=58.
Path: Class C corner-rate `make_path(128, 128)`.

## Summary of runs

| Model | d | hidden | GRU layers | Params | S1 iters | S1 best BLER | Chained BLER | Wilson 95% CI | vs chained SC (0.018) |
|-------|---|--------|------------|--------|----------|-------------|--------------|---------------|----------------------|
| d=16 (warm from N=64) | 16 | 64 | 1 | 40,978 | 80K | 0.160 | 0.2225 | [0.205, 0.241] | 12.4x |
| d=32 (from scratch) | 32 | 128 | 1 | 161,314 | 100K | 0.155 | 0.1560 | [0.141, 0.173] | 8.7x |
| d=64 (from scratch) | 64 | 128 | 1 | 228,674 | 100K | 0.070 | 0.0980 | [0.086, 0.112] | 5.4x |
| d=64 + continuation | 64 | 128 | 1 | 228,674 | 100K + 100K | 0.057 | **0.0890** | [0.077, 0.102] | 4.9x |

Baselines at N=128 (from `chained_trellis_sc_isi_mac.json`, 1000 codewords):
- Chained trellis SC: 0.018
- Joint trellis SC: 0.008
- Memoryless SC: 0.095

**Best result: d=64 BiGRU + 200K total iters achieves chained BLER = 0.089, within 4.9x of chained SC.**

## Training curves

### d=16 (warm-start from N=64, 80K iters, batch 8, lr 5e-4 cosine)

Stage 1 (116.5 min):
```
iter   2000  loss=0.183  BLER=0.435
iter  10000  loss=0.173  BLER=0.360
iter  24000  loss=0.163  BLER=0.290
iter  42000  loss=0.154  BLER=0.220
iter  56000  loss=0.154  BLER=0.195
iter  62000  loss=0.153  BLER=0.160  (best)
iter  80000  loss=0.154  BLER=0.185
```
Stage 2 (49.2 min): BLER(V|true X) = 0.0 throughout.
Chained (2000 cw): BLER = 0.2225.

### d=32 (from scratch, 100K iters, batch 8, lr 5e-4 cosine)

Stage 1 (123.0 min):
```
iter   2000  loss=0.299  BLER=0.975
iter  16000  loss=0.184  BLER=0.450
iter  30000  loss=0.157  BLER=0.225
iter  46000  loss=0.151  BLER=0.190
iter  84000  loss=0.143  BLER=0.160
iter  96000  loss=0.139  BLER=0.155  (best)
iter 100000  loss=0.139  BLER=0.155
```
Stage 2 (50.9 min): BLER(V|true X) = 0.0.
Chained (2000 cw): BLER = 0.1560.

### d=64 (from scratch, 100K iters, batch 8, lr 5e-4 cosine)

Stage 1 (155.7 min):
```
iter   2000  loss=0.289  BLER=0.980
iter  12000  loss=0.165  BLER=0.280
iter  24000  loss=0.143  BLER=0.200
iter  36000  loss=0.136  BLER=0.130
iter  50000  loss=0.123  BLER=0.080  (new best)
iter  70000  loss=0.116  BLER=0.075
iter  86000  loss=0.115  BLER=0.070  (best)
iter 100000  loss=0.113  BLER=0.100
```
Stage 2 (59.3 min): BLER(V|true X) = 0.0 (after iter 4K).
Chained (2000 cw): BLER = 0.098.

### d=64 continuation (flat lr=1e-4, 100K more iters)

Starts from d=64 best. Pre-continuation S1 BLER = 0.073.
```
iter   6000  loss=0.114  BLER=0.070
iter  26000  loss=0.108  BLER=0.067
iter  46000  loss=0.105  BLER=0.057  (best)
iter  74000  loss=0.102  BLER=0.083
iter 100000  loss=0.101  BLER=0.077
```
Chained (2000 cw): BLER = 0.089.

## Analysis

### What worked

1. **Larger embedding dimension is critical at N=128.** d=64 (228K params) achieved 0.089 chained BLER, while d=16 (41K params) plateaued at 0.22 and d=32 (161K params) at 0.16. The tree operations (checknode, bitnode) need sufficient representational capacity to propagate ISI state information through the polar tree. At N=128 with n=7 tree depths, the effective channel at each node is more complex than at N=64.

2. **Continuation training with flat lr helps modestly.** Going from 100K cosine-decayed to 200K total (100K more at flat lr=1e-4) improved chained BLER from 0.098 to 0.089 (~9% relative improvement).

3. **Stage 2 is trivially easy at all scales.** BLER(V|true X) = 0.0 within the first few thousand iterations for every model variant. The V decoder with perfect U side info faces an essentially noiseless effective channel. The entire bottleneck is Stage 1 U decoding.

4. **Warm-start from N=64 helps convergence speed but not final quality.** The d=16 warm-started model converged faster initially but reached the same plateau as the architecture allows.

### What didn't work / known gaps

1. **The 4.9x gap to chained SC remains large.** The target was 2-4x; we achieved 4.9x at best. The gap is entirely in Stage 1 (U-only BLER). Trellis SC achieves 0.018 U BLER by exactly computing the ISI state trellis marginals, while the neural decoder must learn these from data through a BiGRU + tree structure.

2. **GMAC_C proxy design.** Using the Gaussian MAC Class C design as a proxy for the ISI-MAC frozen set is suboptimal. The actual ISI-MAC mutual information profile differs from GMAC. A proper ISI-MAC MC design could improve results significantly.

3. **The d=16 BiGRU that worked well at N=64 (1.5x gap) does not scale to N=128.** At N=64 the 1-layer BiGRU with d//2=8 hidden per direction captured enough ISI state for 6 tree depths. At N=128 (7 depths), the same architecture's information bottleneck becomes severe. The BiGRU hidden state of size 8 per direction cannot represent the growing state space.

4. **Loss floor.** All models eventually plateau in loss (d=16 at 0.15, d=32 at 0.14, d=64 at 0.10). The loss floor correlates with BLER floor but both continue slowly declining with more training.

## Checkpoint paths

All in `class_c_npd/results/npd_memory_mac/`.

Best checkpoints:
- d=16: `isi_mac_bigru_L1_s1_N128_best_d16.pt`, `isi_mac_bigru_L1_s2_N128_best_d16.pt`
- d=32: `isi_mac_bigru_L1_s1_N128_best_d32.pt`, `isi_mac_bigru_L1_s2_N128_best_d32.pt`
- d=64 (initial): `isi_mac_bigru_L1_s1_N128_best_d64.pt`, `isi_mac_bigru_L1_s2_N128_best_d64.pt`
- d=64 (continued): `isi_mac_bigru_L1_cont_d64_s1_N128_best.pt` (stage 2 same as initial)

Result JSONs:
- `isi_mac_bigru_N128_d16_results.json`
- `isi_mac_bigru_N128_d32_results.json`
- `isi_mac_bigru_N128_d64_results.json`
- `isi_mac_bigru_L1_cont_d64_N128_results.json`

Periodic checkpoints every 5K-10K iters also saved (see ls output).

## Compute cost

| Run | Stage 1 | Stage 2 | Total |
|-----|---------|---------|-------|
| d=16, 80K iters | 117 min | 49 min | 166 min |
| d=32, 100K iters | 123 min | 51 min | 174 min |
| d=64, 100K iters | 156 min | 59 min | 215 min |
| d=64, +100K cont | 159 min | (reused) | 159 min |

Total compute: ~12 hours across all runs (including the aborted d=16 continuation).

## Suggested next steps

1. **Larger BiGRU** (d=64, 2 GRU layers) or transformer-based channel encoder. The 1-layer BiGRU at d=64 has 32 hidden per direction; a 2-layer variant would allow richer state representation.

2. **Proper ISI-MAC MC design.** Generate ISI-MAC-specific frozen sets using Monte Carlo density evolution. This removes the GMAC proxy approximation.

3. **Error-injection training for Stage 2.** During Stage 2 training, occasionally use the decoded U_hat (from Stage 1) instead of true U. This bridges the teacher-forcing gap, since at inference Stage 2 receives noisy U_hat.

4. **Larger batch size** (requires more memory or gradient accumulation). Batch=8 at N=128 means high variance per gradient step.

5. **SCL-style list decoding.** The neural tree could maintain L candidate paths rather than greedy SC, potentially closing the gap.
