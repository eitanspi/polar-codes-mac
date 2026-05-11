# N=256 Independent Verification Report

Date: 2026-04-09
Conditions: N=256, Class B (path_i=128), GMAC SNR=6dB, ku=kv=123

## Claim 1: NPD failure is exposure bias

Evidence: **REFUTED**

The exposure bias hypothesis predicts that teacher-forced (TF) leaf accuracy should be substantially higher than free-running (FR) leaf accuracy, with the gap growing over the decoding sequence. The data shows the opposite:

- Teacher-forced leaf accuracy: 0.4209 (3830/9100)
- Free-running leaf accuracy:  0.4214 (3835/9100)
- GAP (TF - FR): **-0.0005** (essentially zero, slightly favoring FR)

A TF-FR gap of ~0 means the model is not suffering from error accumulation at all. Instead, the model is fundamentally unable to decode individual leaves even with perfect prior context. The 42% per-leaf accuracy on 4-class decisions means the model is barely above random (25%) on average.

Further evidence from oracle-first-k:
- oracle_k=0 through oracle_k=150: BLER = 1.0000 (100%)
- oracle_k=200 (almost all leaves oracle): BLER = 0.0000

This means there is NO regime where partial oracle helps. The model either needs ALL decisions to be correct (oracle), or it fails on EVERY codeword. This is not exposure bias; this is fundamental model incapacity.

## Claim 2: Hard sign-flip bitnode is the mechanism

Evidence: **PARTIALLY CONFIRMED** but it is not the primary cause

N=8 sign-flip cascade experiment:
- Flipping one decision at position 0 (value 1 -> 2) causes:
  - Position 1: L2 embedding change = 6.75 (relative 0.64), decision changed (2->3, wrong)
  - Position 5: L2 change = 0.26 but decision changed (0->2, wrong)
  - Average L2 change across positions 1-7: 2.55
  - 2 additional positions corrupted by cascade from single flip

The sign-flip mechanism IS real: flipping one bit causes large (L2~6.75) embedding perturbations downstream, which propagate through the tree. However, this is NOT the primary failure mode at N=256 because:

1. The TF accuracy is already only 42% -- the model cannot decode even WITH correct context
2. The first-error histogram shows 100% of codewords have their first error in positions [0-4]
3. The model fails at the very first information leaves, before any cascade can occur

The sign-flip is a secondary amplifier, not the root cause. The root cause is that the NPD architecture (d=16, 21K params) has insufficient representational capacity for N=256.

## Claim 3: CG decoder achieves BLER ~0.009 at N=256

Evidence: **CONFIRMED** (within margin)

Independently verified CG decoder results (500 codewords each, seed=999-1498):
- n256_long_best:        BLER = 0.0100 (5/500)
- campaign_n256_sched:   BLER = 0.0280 (14/500)

The best CG model achieves BLER = 0.010, consistent with the claimed ~0.009. The CG architecture (39K params, d=16) uses:
- CalcLeft/CalcRight with 3d input (parent, sibling, self)
- Gated NeuralCalcParent with residual
- logits2emb for smooth re-embedding (not hard sign-flip)
- Edge-data storage with no_info_emb initialization

Key architectural difference: the CG decoder uses `logits2emb` which maps soft 4-class log-probabilities to d-dimensional embeddings. This is SMOOTH and differentiable. The NPD bitnode uses hard sign-flips (1-2*u, 1-2*v) which are discontinuous and create the cascade.

## Claim 4: NPD should be abandoned for N>=128

Evidence: **CONFIRMED** with nuance

All four NPD checkpoints (fast_ce, sequential, hybrid, and variant) achieve BLER=1.0000 at N=256. The overfit experiment shows the architecture CAN memorize 10 fixed codewords (BLER=0 after 2000 iters), proving the architecture is expressive enough in principle, but fails completely on unseen data.

The fundamental problem is NOT exposure bias but rather:
1. The NPD bitnode architecture creates a non-smooth mapping through hard sign-flips
2. With d=16 and 21K params, the model lacks capacity for N=256 (the CG model uses 39K params and a richer tree structure)
3. The fast_ce training (parallel) provides leaf embeddings that are unrelated to the sequential decode order
4. Even sequential TF training cannot fix this because the per-leaf accuracy is only 42%

The CG architecture avoids all these issues and achieves 100x better BLER.

## Overfit experiment

A fresh NPD model trained with sequential teacher-forcing on 10 fixed codewords:
- iter 1:    loss=1.348, BLER=1.00
- iter 1000: loss=0.056, BLER=0.80
- iter 2000: loss=0.000, BLER=0.00
- iter 5000: loss=0.000, BLER=0.00

This proves the architecture is not fundamentally broken -- it CAN memorize. But 10 codewords vs the full code space (2^246 possible message pairs) is a factor of ~10^73.

## Final verdict

The NPD failure at N=256 is NOT primarily exposure bias. It is a combination of:
1. **Insufficient model capacity** for the sequential decode task at N=256 (21K params vs 39K for CG)
2. **Architectural flaw**: hard sign-flip bitnode creates discontinuous gradients and brittle embedding propagation
3. **Train-test mismatch**: fast_ce training computes all leaves in parallel, but inference is sequential
4. **All NPD training strategies fail equally**: fast_ce, sequential TF, hybrid, and curriculum all yield BLER=1.0

The CG decoder solves this by:
- Using smooth logits2emb re-embedding instead of hard sign-flips
- Having richer tree operations (CalcLeft, CalcRight, GatedCalcParent with residual)
- Using edge_data with proper tree traversal
- Having ~2x more parameters (39K vs 21K)
