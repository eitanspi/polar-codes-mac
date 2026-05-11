# NPD-Guided Design: Implications for Previously Stuck Problems

## Background

At N=32 GMAC Class C, the NPD-guided frozen set design breakthrough showed that
letting the neural decoder choose its own info positions (based on per-position MI
measured via fast_ce) beats SC by 2-3x, with the advantage growing with N:

| N  | SC (genie) | NPD (NPD design) | Ratio |
|----|-----------|-------------------|-------|
| 16 | 0.170     | 0.113             | 0.66x |
| 32 | 0.074     | 0.034             | 0.45x |
| 64 | 0.022     | 0.007             | 0.32x |

The mechanism: certain positions (e.g., pos 29 at N=32, tree path CCBBB) had
NPD MI=0.71 vs genie MI=0.93 -- a 0.22 gap. Dropping these and picking others
gave a 4x BLER improvement.

## 1. CG Decoder at N=256 Class B

### Per-position MI analysis

Ran teacher-forced evaluation on the trained `n256_long_best.pt` checkpoint
(39,092 params, SimpleMLP_Gmac wrapping PureNeuralCompGraphDecoder) with
2,000 codewords at SNR=6dB, ku=kv=123, path_i=128 (Class B).

### Results: NO significant bottlenecks found

| Metric                   | Value    |
|--------------------------|----------|
| CG MI mean               | 0.9997   |
| CG MI min                | 0.9887   |
| Genie MI mean            | 0.9997   |
| Genie MI min             | 0.9945   |
| Max gap (CG - genie)     | -0.0113  |
| Positions with gap < -0.1| 0        |
| Positions with gap < -0.2| 0        |
| Positions with CG MI<0.95| 0        |

The worst CG position is U-177 with MI=0.9887 (gap = -0.011 vs genie MI=1.0).
Compare this to the N=32 Class C breakthrough where position 29 had a gap of 0.22.

### Key insight: The 3.3x BLER gap is NOT caused by frozen set mismatch

At N=256 Class B, the CG decoder achieves near-perfect MI on every info position
under teacher forcing. The 3.3x gap to SC (BLER 0.017 vs 0.005) must come from
**error propagation during sequential decoding** (not from position-dependent
architectural limitations). When one position makes an error, it corrupts the
leaf embeddings for all subsequent positions.

This is fundamentally different from the N=32 Class C case where certain tree
paths were architecturally hard for the NPD to compute, creating persistent
MI deficits even under teacher forcing.

### Why CG-guided design will NOT help here

- Teacher-forced MI is uniformly high (>0.98) for all positions
- The borderline frozen positions have genie MI in 0.96-0.99 range
- Even swapping the worst info position (MI=0.9887) for the best frozen
  position (genie MI=0.9950) would change MI by only 0.006
- Expected BLER improvement: negligible (<5%)

### What would help instead

The real bottleneck is **autoregressive error accumulation**:
1. SCL (list decoding) to maintain multiple hypotheses
2. CRC-aided decoding to detect/recover from error cascades
3. Better error propagation resistance in the embedding architecture

## 2. 4-class NPD at N=32 Class B

### Applicable? Partially, but different mechanism

The 4-class joint (u,v) NPD (poc_joint_fastce.py) uses fast_ce training which
is parallel (not sequential). The per-position MI could be measured from the
4-class logits at each leaf position. However:

- fast_ce trains with ground-truth codewords at every level, so there is no
  autoregressive error propagation during training
- The 5.8x gap (BLER 0.27 vs SC 0.047) at N=32 is likely a combination of:
  (a) Architecture limitations in the joint bitnode/checknode operations
  (b) Train/test mismatch: fast_ce trains with known decisions, but SC decode
      uses hard decisions that may be wrong
  (c) Possible frozen set mismatch (worth testing, lower confidence)

### Expected improvement from NPD-guided design: modest (10-30%)

The 4-class architecture's bottleneck is more likely the train/test gap than
frozen set selection. But measuring per-position MI from the joint fast_ce
output is cheap and could identify if specific positions are disproportionately
hard for the 4-class model. Priority: medium.

## 3. Revisiting "Failed" Approaches

### NPD fast_ce for MAC: BLER=0.27 at N=32 (5.8x SC)

- Was judged against genie-designed frozen set: YES
- NPD-guided design might help somewhat, but the gap is too large (5.8x)
  to be explained by frozen set mismatch alone
- Primary bottleneck: train/test mismatch (fast_ce parallel vs SC sequential)
- Priority for re-testing with NPD design: LOW

### NPD at N=256: BLER~0.99

- This was the CG decoder failing catastrophically at N=256
- Our analysis shows teacher-forced MI is high, so the issue is error
  accumulation in the long N=256 sequence, not frozen set design
- NPD-guided design would NOT help
- Priority: NONE

### What the breakthrough DOES apply to

The NPD-guided design breakthrough is most applicable when:
1. **Architecture-dependent MI gaps exist**: certain tree paths are hard for
   the specific neural architecture to compute
2. **The gap is persistent under teacher forcing**: not caused by error
   propagation
3. **Alternative positions exist**: there are frozen positions with better
   genie MI that the NPD can handle well

This profile matches:
- Small-to-medium N (8-64) where error propagation is manageable
- Class C (path 0^N 1^N) where the interleaving creates challenging paths
- The single-user NPD (not the joint 4-class model)

## Priority Ranking for Re-testing

1. **NPD-guided design at N=128 Class C** - Extrapolating the N=16/32/64 trend,
   this should give NPD beating SC by ~4x. HIGH priority.
2. **4-class joint NPD MI measurement at N=32** - Quick diagnostic to see if
   specific positions are hard for the joint model. MEDIUM priority.
3. **CG decoder at smaller N with Class C** - Test whether CG + NPD design
   beats SC at N=64-128 for Class C. MEDIUM priority.
4. **CG decoder at N=256 with SCL/CRC** - Address the actual bottleneck
   (error propagation). HIGH priority but different direction.
5. **NPD fast_ce N=256 re-test** - NOT recommended, wrong bottleneck.
