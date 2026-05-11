# Class B CG Decoder — NPD-Guided Code Design at N=256

## Summary

The code-design-mismatch hypothesis (which explained the Class C NPD gap at N=32) does **NOT** explain the Class B CG decoder gap at N=256. The N=256 gap is caused by **inference-time error cascade**, not by poor position selection.

## 1. Per-position MI diagnostic (teacher forcing)

Under teacher forcing with the genie frozen set (ku=kv=123):

| metric | N=128 | N=256 |
|--------|-------|-------|
| mean per-user accuracy | 0.9998 | 0.9999 |
| min per-user accuracy  | 0.9966 | 0.9976 |
| max MI gap (genie - model) | 0.0023 bits | 0.0082 bits |

**All 246 info positions have MI within 0.01 bits of the genie.** No bottleneck position exists. The model extracts all available information under teacher forcing.

Compare to Class C at N=32: position 29 had MI = 0.71 vs genie MI = 0.93 (a 0.22-bit gap that dominated BLER). The N=256 max gap is 40x smaller.

## 2. Inference-time error analysis

Free-running (no teacher forcing) per-step error rates:

| quintile | NN err rate | SC err rate | NN/SC |
|----------|-------------|-------------|-------|
| Q1 (early) | 0.0009 | 0.0002 | 3.8x |
| Q3 (mid) | 0.0027 | 0.0002 | 11x |
| Q5 (late) | 0.0040 | 0.0002 | **18x** |

Correlation of error rate with step position: **NN = +0.70, SC = -0.05**.

NN errors grow linearly through the tree walk. SC errors are flat. The gap is entirely error cascade: wrong decisions at early steps corrupt the embeddings for later steps.

## 3. Code redesign experiment

Swapped the 5 worst NN-inference positions per user (all with genie Pe=0) for 5 currently-frozen positions with lowest Pe (Pe ≤ 0.0016).

| decoder | frozen set | BLER |
|---------|-----------|------|
| SC | genie | 0.0040 |
| SC | swapped-5 | 0.0100 |
| NN | genie | 0.0208 |
| NN | swapped-5 | **0.0500** (5K iters, warm-started) |

The swap made NN **2.4x worse** (0.050 vs 0.021). The positions being removed (genie Pe=0) are genuinely easy — removing them hurts both decoders. The positions being added (Pe > 0) are genuinely harder — they don't help the NN.

### Why code redesign worked for Class C but not Class B

| | Class C N=32 | Class B N=256 |
|---|---|---|
| Gap source | 1 position with MI=0.71 (capacity bottleneck) | No position bottleneck; error cascade |
| TF accuracy | 6/7 positions perfect, 1 weak | All 246 positions perfect |
| Fix | Remove the weak position | Cannot fix by removing positions |
| Root cause | Tree ops can't process 2 leading CheckNodes | Tree ops can't handle wrong inputs |

## 4. Why the all-info training failed

Attempted to train CG decoder with ku=kv=256 (all positions info) to rank positions by neural MI. This fails fundamentally because:

- GMAC sum-rate capacity = **1.5 bits/use** (even at infinite SNR)
- With ku=kv=256: sum rate = 2.0 > 1.5
- The model cannot learn at any SNR because the problem is above capacity

This approach worked for single-user NPD (capacity > 1 bit/use at high SNR) but not for 2-user MAC.

## 5. Conclusion

The N=256 Class B gap (5x SC) has a **different root cause** from the Class C gap:

- **Class C:** architectural bottleneck at specific positions → fix with code design
- **Class B at N=256:** inference error cascade over 246 sequential steps → cannot fix with code design

The CG decoder's tree operations produce perfect predictions when given correct inputs (99.99% TF accuracy) but have never been trained on erroneous inputs. When an early decision is wrong (~0.1% per step), the wrong embedding propagates and corrupts all downstream decisions. The cascade grows linearly with the number of steps (correlation 0.70), producing a 5x BLER gap at 246 steps that doesn't exist at 124 steps (N=128).

## Scripts

```bash
# MI diagnostic (30 seconds)
python scripts/n256_mi_diagnostic.py
python scripts/n128_mi_diagnostic.py

# Inference error cascade analysis (1 minute)
python scripts/n256_inference_errors.py

# Code swap experiment (2 hours)
python scripts/n256_swap_design.py
```

## Data files

- `results/n256_mi_diagnostic.json` — per-step teacher-forced MI
- `results/n256_inference_errors.json` — per-step free-running error rates
- `results/n256_swap_design.json` — swap experiment results
