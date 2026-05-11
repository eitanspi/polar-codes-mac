# Breakthrough Experiments Summary

## Problem Statement
Neural SC decoder for 2-user MAC polar codes works at N<=128 with sequential
training (O(N log N) gradient depth) but fails at N>=256. The fast_ce approach
(O(log N) training) gives BLER=0.38 at N=32 vs SC=0.046 -- a 7-8x gap.

## Critical Bug Found
**Frozen set mapping error in NPD evaluation**: Previous fast_ce experiments
used standard-order frozen sets `{p-1 for p in fu}` with NPD-order (bit-reversed)
SC decode. The correct mapping is `{int(br[p-1]) for p in fu}`. This caused
7 u-info and 3 v-info positions to be incorrectly frozen, producing BLER=1.0
in many experiments and BLER=0.34 in the original POC (poc_joint_fastce.py)
instead of the true value.

With the correct mapping, joint 4-class fast_ce gives **BLER=0.38** at N=32
(SC=0.046), confirming the 8x gap is real and not a bug artifact.

## Experiments Conducted

### 1. Joint 4-class fast_ce (baseline)
- **Result**: BLER=0.381 at N=32 (SC=0.046, 8.3x gap)
- Loss converges to ~0.29 but SC decode BLER plateaus at ~0.35-0.40
- Root cause confirmed: 4-class BitNode conditioning creates 3 distinct error
  patterns during inference that differ from teacher-forced training

### 2. Scheduled Sampling
- **Result**: BLER=0.40 at N=32 (SC=0.046, 8.7x gap)
- Ramped sampling probability from 0 to 0.5 over 30K iterations
- No improvement over standard fast_ce -- the model-generated errors during
  fast_ce don't match the sequential decode error patterns

### 3. Binary Decomposition (u then v|u)
- **Result**: BLER=0.95 at N=32
- u decoder fast_ce loss=0.32 (high -- MAC interference from unknown v)
- v|u decoder loss=0.03 (good with true u conditioning)
- Fails because: u decoder cannot decode from z=(1-2x)+(1-2y)+w without
  knowing y. The MAC channel is fundamentally 4-class, not decomposable into
  independent binary problems.

### 4. Gradient Detaching (sequential training with truncated gradients)
- K=4: BLER=1.0 (no learning after 10K iters)
- K=8: BLER=1.0 (loss=0.36, declining but too slow)
- K=16: BLER=0.992 (loss=0.24, near breakthrough point)
- K=32: BLER=1.0 (loss=0.83, pre-transition)
- Full gradients: BLER=0.112 (10K iters)
- **Conclusion**: Detaching delays the training phase transition proportionally.
  K=16 and K=32 need more iterations but should eventually converge.
  However, at 30K iters with cosine schedule, K=16 still showed BLER=1.0,
  suggesting the truncated gradients fundamentally limit learning of the
  full sequential dependency chain.

### 5. Curriculum Training (MOST PROMISING)
Sequential training with full gradients, transferring across N sizes:

| Stage | N   | Iters | Best BLER | SC BLER | Ratio |
|-------|-----|-------|-----------|---------|-------|
| 1     | 32  | 20K   | 0.062     | 0.046   | 1.35x |
| 2     | 64  | 30K   | 0.050     | 0.025   | 2.0x  |
| 3     | 128 | 10K*  | 0.124     | 0.016   | 7.75x |

*N=128 still training (converging: 0.152@5K -> 0.124@10K)

Key observations:
- Model transfers well between N sizes (initial BLER after transfer: 0.90 at
  N=64, 0.69 at N=128 -- much better than random ~1.0)
- Each stage converges significantly in 20-30K iterations
- Full gradients work at N=32 and N=64 (gradient depth manageable)
- N=128 is slower but making progress
- N=256 planned with K=64 detaching

## Key Findings

1. **The fast_ce approach is fundamentally limited for 4-class MAC**: The
   train-test gap comes from the 4-class BitNode conditioning. In binary
   (single-user), a wrong bit just flips a sign (bounded perturbation). In
   4-class, a wrong joint (u,v) creates one of 3 distinct error patterns
   never seen during teacher-forced training.

2. **Gradient detaching alone doesn't solve scalability**: Truncating
   gradients prevents learning of the full sequential dependency chain.
   The model needs to see the entire decode sequence to learn error
   propagation and correction.

3. **Curriculum transfer is the viable path**: Training at small N with
   full gradients (where it works) and transferring to larger N gives
   better initialization than random. The tree operations are N-independent,
   so learned weights transfer naturally.

4. **The real bottleneck at N>=256 is training time, not gradient depth**:
   Full-gradient training at N=128 works but is very slow (~1 hour per 5K
   iters). At N=256, it would be 4x slower. The solution may be:
   - Larger batch with gradient accumulation
   - Mixed precision training
   - More efficient tree walk implementation (C++ backend)

## Recommendations for N=256+

1. **Use curriculum**: Train N=32->64->128->256 with full gradients at each
   stage (no detaching needed if memory allows)
2. **Increase model capacity**: d=32, hidden=128 (157K params vs 39K)
   converges 2-3x faster per iteration
3. **Use the existing C++ backend**: The project has a `csrc/` directory
   that may contain optimized implementations
4. **Memory optimization**: Gradient checkpointing could reduce memory at
   N=256 without truncating gradients

## Files Created
- `neural/breakthrough_experiments.py` — First round experiments (hybrid, detach)
- `neural/breakthrough_v2.py` — Consistent NPD architecture experiments
- `neural/breakthrough_fix.py` — Frozen set bug fix + correct evaluation
- `neural/breakthrough_detach.py` — Gradient detaching comparison
- `neural/breakthrough_long.py` — Extended detaching runs
- `neural/breakthrough_curriculum.py` — Curriculum training (best approach)
- `neural/breakthrough_agent.log` — Full experiment log
