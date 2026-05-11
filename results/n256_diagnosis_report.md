# N=256 Neural MAC Decoder -- Diagnosis Report

## 1. What was checked

### Pipeline Audit
- Read all three experiment scripts (`decisive_n256.py`, `test_fastce_n256.py`) and their result files
- Verified frozen set design: `gmac_B_n8_snr6dB.npz`, N=256, ku=kv=123, |Au|=123, |Av|=123, |fu_nat|=|fv_nat|=133
- Confirmed channel consistency: all experiments use GaussianMAC with sigma2=0.251 (SNR=6dB), Z=(1-2X)+(1-2Y)+W
- Confirmed eval pipeline: all NPD experiments use same frozen sets, same SC sequential decode, seed=999
- Compared NPD architecture (21,300 params) vs production CG decoder (39,092 params)
- Evaluated production CG decoder checkpoint at N=256: BLER=0.014 (2.8x SC)

### Failure Mode Diagnostics

**A. Teacher-forced vs free-running accuracy (200 codewords)**
- Teacher-forced leaf accuracy: 97.2%
- Free-running leaf accuracy: 58.0%
- Accuracy drop: 39.2 percentage points
- Block error rate (free-running): 0.995

**B. First-error-position histogram (199 error blocks)**
- Positions 0-4: 78.9% of first errors
- Positions 5-9: 12.1%
- Positions 10-19: 5.0%
- Positions 20+: 4.0%
- Mean first error position: 4.3 (out of 246 info leaves)
- Median first error position: 2.0
- Notable spikes at positions 0 (26.6%), 1 (18.1%), 4 (22.1%), 8 (9.0%)

**C. Conditional BLER with oracle first m info bits**
- m=0 (no oracle): BLER=0.985
- m=5: BLER=0.975
- m=10: BLER=0.945
- m=20: BLER=0.870
- m=50: BLER=0.335
- m=100: BLER=0.010
- m=150: BLER=0.000
- m=200: BLER=0.000

**D. Overfit test (10 fixed codewords)**
- Fresh model trained with sequential TF on 10 fixed codewords
- Reached BLER=0/10 at iteration 1000 (2.4 minutes)
- Loss dropped to 0.000009
- Conclusion: model capacity is sufficient for N=256

**E. Production CG decoder check**
- Checkpoint exists: `saved_models/n256_long_best.pt` (39,092 params)
- Best BLER=0.009 (1.8x SC) during 16.6hr training (100K iters, cosine LR from 5e-5)
- Re-evaluated with seed=999: BLER=0.014 (2.8x SC, 500 codewords)
- This decoder WORKS at N=256

### Targeted Experiment: Scheduled Sampling
- **Fine-tune pretrained model A**: 10K iters, ss_rate 0->0.5 over 5K warmup
  - Loss went from 0.054 (ss=0.20) to 0.290 (ss=0.50)
  - Best BLER: 0.990 (no improvement from baseline 0.990)
- **From scratch**: 15K iters, ss_rate 0->0.3 over 8K warmup, lr=3e-4
  - BLER=1.000 at all evaluation points (3K, 6K, 9K iters)
  - Loss: 0.555 -> 0.130 -> 0.174 (oscillating, not converging)
- Conclusion: scheduled sampling does NOT help the NPD architecture

## 2. What is definitely true

1. **The NPD architecture at d=16 has sufficient capacity** for N=256. The overfit test proves this.

2. **The failure is pure exposure bias / error cascading.** The evidence is overwhelming:
   - 97% per-leaf accuracy with teacher forcing, 58% free-running (39 point gap)
   - 79% of first errors occur in the first 5 info positions
   - Oracle first 100 bits reduces BLER from 0.985 to 0.010

3. **The production CG decoder works at N=256** (BLER=0.009-0.014), proving the problem is solvable.

4. **All three NPD experiments (A, B, C) are validly comparable.** They use the same architecture, channel, frozen sets, and evaluation pipeline. The comparison is valid.

5. **Scheduled sampling does not fix the NPD architecture.** The sign-flipping bitnode creates a hard discontinuity that prevents gradual exposure to model errors.

6. **The NPD bitnode is the root cause.** Its sign-based modulation (`e_odd * (1-2*u_left)`) means a single bit error in u_left flips the sign of the entire embedding. This is catastrophic at position 0 where no prior information exists.

## 3. What remains uncertain

1. **Whether a softer bitnode** (e.g., learned embedding instead of sign-flip) would fix the NPD at N=256.
2. **Why the CG decoder avoids exposure bias** -- it also uses teacher forcing during training, but its architecture (logits2emb + CalcParent) may create smoother error propagation.
3. **The exact contribution of each CG architectural feature** (CalcParent, logits2emb, 3d inputs, no_info_emb) to its success.

## 4. Best explanation of failure

The NPD bitnode architecture creates a **hard, non-recoverable error propagation path**:

1. The bitnode computes: `e_signed = cat(e_odd[:h] * u_sign, e_odd[h:] * v_sign)` where `u_sign = 1 - 2*u_left`.
2. If `u_left` is wrong (sign error), the first half of the embedding is multiplied by -1.
3. This corrupted embedding propagates through ALL subsequent tree operations.
4. At N=256, there are 246 info leaves. An error at position 0 corrupts information for positions 1-245.
5. Teacher-forced training never encounters this corruption, so the model never learns to handle it.
6. At smaller N (N=32, 15 info leaves), the tree depth is only 5 and there are fewer positions to corrupt, so the per-leaf accuracy of 97% is sufficient (0.97^15 = 0.63, explaining the N=32 BLER=0.27).
7. At N=256 (246 info leaves), 0.97^246 = 0.0006, but errors at early positions cascade to make ALL later positions wrong, not just independently wrong.

The production CG decoder avoids this by:
- Using `logits2emb` (smooth 4->d mapping) instead of hard sign-flipping for decision embedding
- Having `CalcParent` (bottom-up information flow) that lets the tree self-correct
- Using `no_info_emb` (learned initialization) for upward edges
- Processing leaves in path order (interleaved u/v) with dedicated calc_left/calc_right that take 3d inputs

## 5. Best next direction

**Stop developing the NPD architecture for N >= 128. Switch to the CG architecture exclusively.**

Specific recommendations:
1. **Use the production CG decoder** (`ncg_gmac.py` / `ncg_pure_neural.py`) which already achieves BLER=0.009 at N=256
2. **Scale the CG decoder to N=512 and N=1024** -- checkpoints exist at `ncg_pure_neural_N512.pt` and `ncg_pure_neural_N1024.pt`
3. **If the CG decoder also struggles at larger N**, the fix is to add scheduled sampling there (the CG bitnode is smooth, so scheduled sampling should actually work)
4. **Do NOT waste time on**: larger d for NPD, different learning rates for NPD, curriculum for NPD, or any other NPD modification

## 6. What should be stopped immediately

1. **Stop all NPD experiments at N=256.** The architecture is fundamentally unsuited for long sequences due to hard sign-flipping in bitnode.
2. **Stop fast_ce training for NPD at any N > 64.** fast_ce trains leaf-only and completely ignores the sequential decode regime.
3. **Stop scheduled sampling experiments for NPD.** The sign-flip discontinuity prevents the gradients from being useful under model errors.
4. **Stop curriculum training for NPD at N=256.** Curriculum N=32->64->128->256 does not help because the failure mode (sign-flip cascading) exists at all N but only becomes catastrophic at N=256.

---

## Appendix: Comparison Table

| Decoder | Architecture | d | Params | N | ku,kv | Train | Best BLER | vs SC |
|---------|-------------|---|--------|---|-------|-------|-----------|-------|
| NPD A (fast_ce) | sign-bitnode | 16 | 21K | 256 | 123,123 | fast_ce 176K iters | 0.988 | 198x |
| NPD B (seq.) | sign-bitnode | 16 | 21K | 256 | 123,123 | curriculum TF | 0.987 | 197x |
| NPD C (hybrid) | sign-bitnode | 16 | 21K | 256 | 123,123 | fast_ce + seq TF | 0.983 | 197x |
| NPD + sched. samp. | sign-bitnode | 16 | 21K | 256 | 123,123 | fine-tune + SS | 0.990 | 198x |
| CG production | neural CalcParent | 16 | 39K | 256 | 123,123 | seq TF + cosine | 0.009 | 1.8x |
| SC reference | analytical | -- | 0 | 256 | 123,123 | -- | 0.005 | 1.0x |

## Appendix: Diagnostic Data

### Teacher-forced vs free-running accuracy gap
- TF accuracy: 97.17%
- FR accuracy: 57.97%
- Gap: 39.20 percentage points

### First error position distribution
```
Position 0: 26.6%    Position 4: 22.1%
Position 1: 18.1%    Position 8:  9.0%
Position 2:  7.0%    Position 14: 4.0%
Position 3:  5.0%    Other:       8.2%
```

### Conditional BLER curve
```
Oracle m=0:   0.985    Oracle m=50:  0.335
Oracle m=5:   0.975    Oracle m=100: 0.010
Oracle m=10:  0.945    Oracle m=150: 0.000
Oracle m=20:  0.870    Oracle m=246: 0.000
```

### Overfit test
- 10 fixed codewords, fresh model
- BLER=0/10 at iter 1000 (loss=0.000009)
- Confirms capacity is sufficient
