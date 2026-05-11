# Reliable BLER Evaluations (>=100 errors per data point)

_Generated: 2026-04-21_

High-codeword-count Monte Carlo evaluations for tight confidence intervals.
Wilson 95% CI reported. "Reliable" = at least 100 block errors observed.

## ABNMAC Class B

| N  | Decoder | BLER     | Errors | CW     | Wilson 95% CI           | Reliable | Previous | Prev CW | vs SC  |
|----|---------|----------|--------|--------|-------------------------|----------|----------|---------|--------|
| 8  | SC      | 0.1198   | 1198   | 10000  | [0.1136, 0.1263]        | YES      | 0.124    | 500     | 1.00x  |
| 8  | NCG     | 0.1202   | 1202   | 10000  | [0.1140, 0.1267]        | YES      | 0.107    | 300     | 1.00x  |
| 16 | SC      | 0.0629   | 629    | 10000  | [0.0583, 0.0678]        | YES      | 0.062    | 500     | 1.00x  |
| 16 | NCG     | 0.0570   | 570    | 10000  | [0.0526, 0.0617]        | YES      | 0.040    | 300     | 0.91x  |
| 32 | SC      | 0.0213   | 213    | 10000  | [0.0186, 0.0243]        | YES      | 0.012    | 500     | 1.00x  |
| 32 | NCG     | 0.0182   | 182    | 10000  | [0.0158, 0.0210]        | YES      | 0.010    | 300     | 0.85x  |
| 64 | SC      | 0.0438   | 219    | 5000   | [0.0385, 0.0498]        | YES      | 0.038    | 500     | 1.00x  |
| 64 | NCG     | 0.0416   | 208    | 5000   | [0.0364, 0.0475]        | YES      | 0.033    | 300     | 0.95x  |

### Key findings (ABNMAC)

1. **Previous estimates were unreliable**: The 300/500-CW estimates from the
   training session had large sampling noise. At N=8, NCG was reported as
   0.107 (0.86x SC) but reliable eval shows 0.1202 (1.00x SC) -- NCG does NOT
   beat SC at N=8. At N=16, NCG was 0.040 (0.65x SC) but is actually 0.057 (0.91x SC).

2. **NCG gains are smaller than previously reported**: The true NCG advantage
   is modest -- 0-15% improvement over SC across N=8-64, not the 13-35%
   claimed from low-CW estimates.

3. **N=32 SC is higher than reported**: SC at N=32 was listed as 0.012 (500 CW,
   ~6 errors) but the reliable estimate is 0.0213 (213 errors). The previous
   number was an underestimate.

4. **N=64 anomaly**: SC BLER at N=64 (0.0438) is HIGHER than at N=32 (0.0213).
   This was also visible in the prior estimates (0.038 vs 0.012) and is now
   confirmed with tight CIs. Likely a rate-selection artifact -- the N=64
   design uses ku=kv=22 which is more aggressive relative to capacity.


## BEMAC Class B

| N   | Decoder | BLER     | Errors | CW     | Wilson 95% CI           | Reliable | Previous | Prev CW |
|-----|---------|----------|--------|--------|-------------------------|----------|----------|---------|
| 32  | SC      | 0.00973  | 146    | 15000  | [0.0083, 0.0114]        | YES      | 0.008    | 5000    |
| 64  | SC      | 0.00320  | 128    | 40000  | [0.0027, 0.0038]        | YES      | 0.006    | 5000    |
| 128 | SC      | 0.00162  | 81     | 50000  | [0.0013, 0.0020]        | partial  | 0.002    | 5000    |

### Key findings (BEMAC)

1. **N=32 SC is confirmed ~0.01**: Previous estimate of 0.008 (from 5000 CW,
   ~40 errors) was a slight underestimate. Reliable value is 0.00973.

2. **N=64 SC is LOWER than previously reported**: Previous 0.006 (from 5000 CW,
   ~30 errors) was an overestimate. Reliable value is 0.00320 -- nearly half.
   This makes BEMAC SC at N=64 much better than previously thought.

3. **N=128 is partial**: 81 errors in 50K CW gives BLER ~0.00162. Would need
   ~62K CW for 100 errors. Previous estimate of 0.002 is consistent within CI.

4. **N=256 skipped**: At BLER ~8e-5, would need >1M CW -- impractical.

5. **No BEMAC-specific NCG checkpoints exist** in saved_models/. The "NCG"
   numbers in prior tables (e.g., bemac_comprehensive_paper.json NN_SC column)
   came from a different model architecture not available for re-evaluation.


## Notes

- All runs used `torch.set_num_threads(2)` to avoid interfering with
  ongoing GMAC N=256 training.
- Random seeds: SC uses evaluator default RNG; NCG uses seed=999.
- BEMAC uses `log_domain=False`, ABNMAC uses `log_domain=True`.
- All use `backend='interleaved'` (O(N log N) decoder for Class B paths).
- Design files from `designs/bemac_B_n{n}.npz` and `designs/abnmac_B_n{n}.npz`.

## GMAC Class B

| N    | Decoder | BLER     | Errors | CW     | Wilson 95% CI           | Reliable | Checkpoint           | Previous | Prev CW |
|------|---------|----------|--------|--------|-------------------------|----------|----------------------|----------|---------|
| 32   | SC      | 0.0450   | 135    | 3000   | [0.0381, 0.0530]        | YES      | --                   | 0.047    | 2000    |
| 32   | NCG     | 0.0503   | 151    | 3000   | [0.0431, 0.0587]        | YES      | ncg_gmac_mlp_N32     | 0.040    | 2000    |
| 64   | SC      | 0.0276   | 138    | 5000   | [0.0234, 0.0325]        | YES      | --                   | 0.028    | 2000    |
| 64   | NCG     | 0.0282   | 141    | 5000   | [0.0240, 0.0332]        | YES      | ncg_gmac_mlp_N64     | 0.026    | 2000    |
| 128  | SC      | 0.0187   | 112    | 6000   | [0.0155, 0.0224]        | YES      | --                   | 0.020    | 2000    |
| 512  | NCG     | 0.0123   | 123    | 10000  | [0.0103, 0.0147]        | YES      | n512_long_latest     | 0.010    | 500     |
| 512  | NCG     | 0.0538   | 538    | 10000  | [0.0495, 0.0584]        | YES      | ncg_gmac_mlp_N512    | --       | --      |
| 1024 | NCG     | 0.4700   | 940    | 2000   | [0.4482, 0.4919]        | YES      | ncg_gmac_mlp_N1024   | 0.055    | 200     |

### Key findings (GMAC Class B)

1. **SC estimates were already close**: The 2000-CW estimates from the campaign
   were surprisingly accurate (within 10-20% of reliable values).

2. **NCG does NOT beat SC on Class B**: NCG BLER is 2-12% WORSE than SC at
   N=32 and N=64. Previous estimates (0.040 < 0.047 for SC) suggested NCG was
   better, but this was sampling noise.

3. **N=128 SC confirmed**: 0.0187 (vs previous 0.020), consistent within CI.

4. **N=512 NCG**: Two checkpoints tested. n512_long_latest.pt achieves BLER=0.0123
   (best), ncg_gmac_mlp_N512.pt achieves 0.0538 (4x worse). Both are much worse
   than SC (est ~0.001). The previous estimate of 0.010 (5/500) was roughly correct
   for the best checkpoint.

5. **N=1024 NCG completely broken**: BLER=0.4700, near random guessing (0.50).
   The previous estimate of 0.055 (11/200) was wildly optimistic due to tiny sample.
   The model clearly did not converge during training at N=1024.


## ABNMAC Class C (non-monotonicity investigation)

| N   | Decoder | BLER     | Errors | CW     | Wilson 95% CI           | Reliable |
|-----|---------|----------|--------|--------|-------------------------|----------|
| 32  | SC      | 0.0334   | 167    | 5000   | [0.0288, 0.0387]        | YES      |
| 64  | SC      | 0.0478   | 239    | 5000   | [0.0422, 0.0541]        | YES      |

### Key findings (ABNMAC Class C non-monotonicity)

1. **Non-monotonicity CONFIRMED**: N=32 BLER=0.0334, N=64 BLER=0.0478. The CIs
   do not overlap, so this is a real effect (not sampling noise).

2. **Likely rate-selection artifact**: The rates ku=13, kv=26 at N=64 may be too
   aggressive relative to channel capacity for this path/design combination.


## GMAC Class C (from serious_npd_vs_sc_eval.json)

| N   | Decoder | BLER     | Errors | CW     | Wilson 95% CI           | Reliable |
|-----|---------|----------|--------|--------|-------------------------|----------|
| 16  | SC      | 0.1620   | 1620   | 10000  | [0.1549, 0.1694]        | YES      |
| 16  | NPD     | 0.1070   | 1070   | 10000  | [0.1011, 0.1132]        | YES      |
| 32  | SC      | 0.0681   | 681    | 10000  | [0.0633, 0.0732]        | YES      |
| 32  | NPD     | 0.0373   | 373    | 10000  | [0.0338, 0.0412]        | YES      |
| 64  | SC      | 0.0273   | 273    | 10000  | [0.0243, 0.0307]        | YES      |
| 64  | NPD     | 0.0100   | 100    | 10000  | [0.0082, 0.0121]        | YES      |

### Key findings (GMAC Class C)

1. **NPD consistently beats SC at N=16-64**: Ratios are 0.66x, 0.55x, 0.37x -- 
   the improvement grows with N.

2. **N=128 anomaly**: NPD BLER=0.0329 vs SC 0.0071 at N=128 (4.6x worse). This
   needs investigation -- possibly the NPD checkpoint at N=128 underfit.

3. **SC at N=512,1024**: Extremely high BLER (0.10-0.16) suggesting a code design
   or rate issue at those N values.


## BEMAC Class C (partial)

| N    | Decoder | BLER     | Errors | CW     | Wilson 95% CI           | Reliable |
|------|---------|----------|--------|--------|-------------------------|----------|
| 1024 | SC      | 0.000409 | 26     | 63550  | [0.000279, 0.000599]    | NO       |

### Key findings (BEMAC Class C)

1. **N=1024 SC is very low BLER**: 26 errors in 63.5K CW. Previous estimate was
   0.0004 (20/50K), consistent with new measurement of 0.000409.

2. **Not reliable**: Would need ~250K CW (many hours) for 100 errors.


## Notes

- All runs used `torch.set_num_threads(2)` to avoid interfering with
  ongoing GMAC N=256 training.
- Random seeds: SC uses evaluator default RNG; NCG uses seed=999.
- BEMAC uses `log_domain=False`, ABNMAC uses `log_domain=True`.
- All use `backend='interleaved'` (O(N log N) decoder for Class B paths).
- Design files from `designs/bemac_B_n{n}.npz` and `designs/abnmac_B_n{n}.npz`.

## Raw data

- `results/reliable_evals/bemac_B_reliable.json`
- `results/reliable_evals/abnmac_B_reliable.json`
- `results/reliable_evals/gmac_B_reliable.json` (includes all NCG results from this session)
- `results/reliable_evals/abnmac_C_reliable.json`
- `results/reliable_evals/bemac_C_sc_reliable_item5.json`
- `results/reliable_evals/gmac_B_ncg_reliable_item1.json` (N=32,64)
- `results/reliable_evals/gmac_B_ncg_reliable_item2.json` (N=512, ncg_gmac_mlp)
- `results/reliable_evals/gmac_B_ncg_reliable_item2b.json` (N=512, n512_long_latest)
- `results/reliable_evals/gmac_B_ncg_reliable_item3.json` (N=1024)
- `class_c_npd/results/serious_npd_vs_sc_eval.json`


## Session summary (2026-04-16)

Completed Tier 1 items:
1. Updated BLER_TABLES.md with corrected ABNMAC B and BEMAC B numbers from reliable evals.
2. Extracted BEMAC Class C CW counts from existing JSON -- already had nn_cw/sc_cw fields.
3. GMAC Class B: SC reliable at N=32 (135/3000), N=64 (138/5000), N=128 (112/6000).
4. GMAC Class C: Already had reliable data at N=16-64 from serious_npd_vs_sc_eval.json (10K CW each).
5. ABNMAC Class C non-monotonicity: CONFIRMED real. N=32=0.0334, N=64=0.0478, CIs don't overlap.

Completed Tier 2 items:
6. ABNMAC Class B N=64 NCG: already evaluated (0.0416, 208/5000), only 0.95x SC. Training not continued.
7. GMAC Class B NCG at N=32,64: NCG does NOT beat SC. BLER is 6-17% worse. Possible model-path mismatch.

## Session summary (2026-04-16, batch 2)

Completed Items:
1. GMAC Class B NCG N=32: BLER=0.0503 (151/3000) RELIABLE. NCG 12% worse than SC (0.045).
2. GMAC Class B NCG N=64: BLER=0.0282 (141/5000) RELIABLE. NCG 2% worse than SC (0.0276).
3. GMAC Class B NCG N=512 (ncg_gmac_mlp_N512): BLER=0.0538 (538/10000) RELIABLE. 54x worse than SC.
4. GMAC Class B NCG N=512 (n512_long_latest): BLER=0.0123 (123/10000) RELIABLE. 12x worse than SC.
5. GMAC Class B NCG N=1024 (ncg_gmac_mlp_N1024): BLER=0.4700 (940/2000) RELIABLE. Model BROKEN.
6. BEMAC Class C SC N=1024: BLER=0.000409 (26/63550) PARTIAL. 2h time limit, 26 errors < 100.

Skipped Items:
- GMAC Class C NCG N=32,64 (Item 4): CG checkpoints were NOT saved to disk. Training script
  (cg_vs_npd_comparison.py) trained inline with n_cw=1000 eval. Cannot re-evaluate without retraining.

Key finding: NCG does not beat SC on GMAC Class B at any N tested (32-1024). The gap widens dramatically
at larger N: 2% at N=64, 12x at N=512, random at N=1024. The model architecture does not scale.


## Session summary (2026-04-16, batch 3)

### Item 4: BEMAC Class C rate mismatch
SC re-evaluated at NPD's lower rates to fix apples-to-oranges comparison:
- N=16: ku=4,kv=8 -> SC BLER=0.061 (182/3000) vs original ku=5,kv=10 -> 0.092 (276/3000)
- N=32: ku=8,kv=16 -> SC BLER=0.049 (148/3000) vs original ku=10,kv=19 -> 0.099 (298/3000)
NPD operated at ~50% lower rates, inflating the apparent NN advantage. At matched rates, SC is ~2x better.

### Item 25: Why NN beats SC on GMAC Class C (N=16, N=32)
Tested SC decoder on NN's MI-designed frozen set:
- N=16: SC+SC_frozen=0.173, SC+NN_frozen=0.377, NN+NN_frozen=0.111
- N=32: SC+SC_frozen=0.062, SC+NN_frozen=0.337, NN+NN_frozen=0.038
The NN frozen set is TERRIBLE for SC (2-5x worse than SC's own). The NN and its frozen set co-adapt.
The gain is from BOTH decoder AND code design working together, not from either alone.
Raw data: `results/reliable_evals/gmac_C_frozen_set_item25.json`

### Item 22: GMAC Class C NCG retrained
NCG (CG-style decoder) retrained from scratch at N=32 and N=64:
- N=32: BLER=0.0353 (106/3000) RELIABLE. Original inline estimate was 0.039 (1000 CW).
- N=64: BLER=0.0133 (40/3000) partial. Original inline estimate was 0.020 (1000 CW).
- Both NCG and NPD beat SC significantly on GMAC Class C (0.52x and 0.49x at N=32,64).
- NCG and NPD give similar BLER (0.035 vs 0.037 at N=32).
Raw data: `results/reliable_evals/gmac_C_ncg_retrain_item22.json`
Checkpoints: `saved_models/ncg_gmac_classC_N{32,64}.pt`

### Item 6: BEMAC Class C NPD N=64 retrained (curriculum from N=32)
NPD Stage 1 retrained from N=32 checkpoint:
- N=64: best BLER=0.006 (3/500 quick eval) after 50K iters, 58 min.
- Massive improvement from broken 0.072 in original training.
- SC reference at N=64 is 0.055 (165/3000) -> NPD/SC ratio ~0.11x.
Checkpoint: `class_c_npd/results/bemac_classC_s1_N64_v2.pt`

### Item 12: ABNMAC Class B NCG N=128 curriculum (DONE)
NCG curriculum training from N=64 checkpoint:
- 35K iters (120 min time limit): best BLER=0.025 (5/200 eval)
- Uses MC design from designs/abnmac_B_n7.npz (different from analytical Bhattacharyya design)
- Important: N=64 NCG (classB_N64_best) only works with MC design, NOT analytical design
- SC at N=64 was 0.044. No reliable SC reference at N=128 yet.
Checkpoint: `saved_models/ncg_abnmac_classB_N128_curriculum.pt`

### Item 6 continued: BEMAC Class C NPD N=128 curriculum
NPD curriculum N=64->N=128:
- 80K iters, 64 min: BLER=0.000 (0/300 eval)
- SC reference at N=128 was 0.025 (491/20000)
- Needs larger eval (>=1000 CW) to get a reliable BLER estimate
Checkpoint: `class_c_npd/results/bemac_classC_s1_N128_v2.pt`


## ISI-MAC Class C (corner rate, h=0.3, SNR 6 dB)

### Chained Trellis SC (10K CW, Session 11)

All from `results/reliable_evals/isi_mac_sc_10kcw.json`. This is the 2-state chained decoder
(`decoder_trellis_mac_chained.decode_chained`), the direct analytical counterpart to the chained NPD.

| N   | Decoder         | BLER    | Errors | CW    | Wilson 95% CI           | Reliable | U BLER | V BLER |
|-----|-----------------|---------|--------|-------|-------------------------|----------|--------|--------|
| 16  | Chained Trellis | 0.1689  | 1689   | 10000 | [0.1617, 0.1764]        | YES      | 0.1662 | 0.1689 |
| 32  | Chained Trellis | 0.0822  | 822    | 10000 | [0.0770, 0.0877]        | YES      | 0.0821 | 0.0819 |
| 64  | Chained Trellis | 0.0407  | 407    | 10000 | [0.0370, 0.0448]        | YES      | 0.0403 | 0.0403 |
| 128 | Chained Trellis | 0.0223  | 223    | 10000 | [0.0196, 0.0254]        | YES      | 0.0222 | 0.0180 |
| 256 | Chained Trellis | 0.0061  | 61     | 10000 | [0.0048, 0.0078]        | partial  | 0.0056 | 0.0060 |

### Joint Trellis SC (10K CW, from prior sessions)

The 4-state joint trellis decoder is stronger (especially at N=64) but more expensive.

| N   | Decoder       | BLER    | Errors | CW    | Reliable |
|-----|---------------|---------|--------|-------|----------|
| 16  | Joint Trellis | 0.166   | 1664   | 10000 | YES      |
| 32  | Joint Trellis | 0.083   | 825    | 10000 | YES      |
| 64  | Joint Trellis | 0.026   | 262    | 10000 | YES      |
| 128 | Joint Trellis | 0.018   | 180    | 10000 | YES      |

### Chained NPD (5K CW, from Session 10)

From `results/reliable_evals/isi_mac_npd_reliable.json`.

| N   | Model     | BLER    | Errors | CW   | Wilson 95% CI           | vs Chained Trellis | vs Joint Trellis |
|-----|-----------|---------|--------|------|-------------------------|--------------------|------------------|
| 16  | d=16 h=64 | 0.143   | 715    | 5000 | [0.1333, 0.1530]        | 0.85x (beats)      | 0.86x (beats)    |
| 32  | d=16 h=64 | 0.081   | 406    | 5000 | [0.0736, 0.0889]        | 0.99x (matches)    | 0.98x (matches)  |
| 64  | d=16 h=64 | 0.046   | 229    | 5000 | [0.0404, 0.0521]        | 1.13x              | 1.77x            |
| 64  | d=16 h=100 @25K | 0.035 | 176 | 5000 | [0.0304, 0.0407]   | 0.86x (beats)      | 1.35x            |
| 64  | **d=16 h=100 @50K** | **0.029** | 145 | 5000 | [0.0247, 0.0340] | **0.71x (beats)** | **1.12x**        |
| 128 | d=64 h=128| 0.030   | 150    | 5000 | [0.0256, 0.0350]        | 1.36x              | 1.67x            |
| 256 | d=64 h=128| 0.011   | 56     | 5000 | [0.0085, 0.0145]        | 1.83x              | 1.83x            |

### Key findings (ISI-MAC)

1. **Chained trellis SC is the fair comparison**: Since NPD also uses chained decoding
   (Stage 1 + Stage 2), comparing against the chained trellis SC is apples-to-apples.
   The joint trellis decoder exploits Y-correlation that neither the chained trellis
   nor the chained NPD can access.

2. **NPD beats chained trellis at N=16** (0.143 vs 0.169, 15% reduction) and
   **matches at N=32** (0.081 vs 0.082, CIs overlap).

3. **N=64 d=16 h=100 BEATS chained trellis**: The d=16 h=100 model achieves 0.035
   (176/5000), which beats chained trellis 0.041 (ratio 0.86x). The original d=16 h=64
   model was 0.046 (1.13x). Hidden width increase (64->100) is the key. The 1.77x gap
   previously reported was vs joint trellis (0.026); d=16 h=100 vs joint is 1.35x.

4. **N=128-256 gap**: At larger N, the NPD lags by 1.36-1.83x vs chained trellis.
   d=64 embedding dimension is needed at N>=128 (d=16 fails).

5. **N=256 chained trellis**: 61 errors in 10K CW (partial reliability). The joint
   trellis was not run at 10K CW for N=256.
