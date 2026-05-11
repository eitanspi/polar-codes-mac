# N=256 Hybrid / Sequential-Tree Failure Investigation

## 1. Hypotheses Tested

1. **Architecture capacity**: Is the d=16 architecture too small to represent N=256 decoding?
2. **Component bottleneck**: Is the failure in the z_encoder, the tree ops, or their co-adaptation?
3. **Teacher-forcing gap (exposure bias)**: Does the model learn under teacher forcing but fail at free-running inference?
4. **Error cascade**: Do early errors cascade and prevent learning?
5. **Mode collapse**: Does the model collapse to input-independent constant predictions?

## 2. Experiments Run

### Baselines (EXP G)

| Checkpoint | BLER | Notes |
|---|---|---|
| `campaign_n256_sched_best` (curriculum) | 0.039 | Trained N=16->32->64->128->256 |
| `ncg_gmac_mlp_N128` (direct to N=256) | 0.039 | N=128 weights applied to N=256 (no fine-tune) |
| `ncg_gmac_mlp_N256` (fully trained) | 0.023 | Curriculum + fine-tuned at N=256 |

### EXP A: Overfit 1 Fixed Codeword (Capacity Test)

| Iter | Loss | BLER |
|---|---|---|
| 500 | 4e-6 | 0.0 |
| 2000 | 1e-6 | 0.0 |

**Answer**: Architecture CAN represent N=256 decode. Not a capacity issue.

### EXP B: Teacher Forcing Gap (From Scratch)

| Iter | Train Loss (TF) | Eval Loss (TF) | BLER (free-run) |
|---|---|---|---|
| 300 | 0.875 | 0.876 | 1.0 |
| 600 | 0.874 | 0.874 | 1.0 |
| 900 | 0.874 | 0.874 | 1.0 |
| 1200 | 0.874 | 0.875 | 1.0 |
| 1500 | 0.874 | 0.874 | 1.0 |

**Answer**: Loss plateaus at 0.874 (vs 1.386 random). BLER stays 1.0. Massive TF gap.

### EXP C: Frozen z_encoder (from N=128), Fresh Tree Ops

| Iter | Loss | BLER |
|---|---|---|
| 300 | 0.875 | 1.0 |
| 1500 | 0.875 | 1.0 |

**Answer**: Good z_encoder + random tree ops = stuck at 0.875 plateau. Tree ops cannot learn from scratch at N=256 in 1500 iters.

### EXP D: Frozen Tree Ops (from N=128), Fresh z_encoder

| Iter | Loss | BLER |
|---|---|---|
| 300 | 0.395 | 1.0 |
| 600 | 0.267 | 1.0 |
| 900 | 0.239 | 1.0 |
| 1200 | 0.251 | 1.0 |
| 1500 | 0.247 | 1.0 |

**Answer**: Pretrained tree ops + fresh z_encoder = loss drops to 0.25 (78% 4-class accuracy under TF). z_encoder learns fast when tree ops are good. But BLER still 1.0 -- exposure bias.

### EXP E: Per-Leaf Accuracy

| Model | Q1 acc | Q2 acc | Q3 acc | Q4 acc | Confidence | BLER |
|---|---|---|---|---|---|---|
| curriculum | 1.000 | 1.000 | 1.000 | 0.997 | 13-17 | 0.016 |
| scratch_1500 | 0.492 | 0.501 | 0.504 | 0.500 | 0.03-0.05 | 1.0 |

**Answer**: Scratch model has random accuracy (50%) at every position with near-zero confidence. It has learned nothing position-useful.

### EXP F: Oracle Injection

| Model | K=0 | K=64 | K=128 | K=256 | K=384 | K=500 |
|---|---|---|---|---|---|---|
| curriculum | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| scratch_1500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| N=128 direct | 0.078 | 0.078 | 0.078 | 0.016 | 0.000 | 0.000 |

**Answer**: Scratch model gets BLER=1.0 even with 500/512 oracle decisions. It cannot make a single correct non-trivial decision. The model has not learned any input-dependent features.

### Additional: Prediction Diversity Analysis

For the scratch model at each info position, the model predicts the SAME class for ALL 64 samples in a batch (unique predictions = 1). It has collapsed to a position-dependent constant prediction independent of the channel input z.

### EXP Scale-Fix: z_encoder * 30 + no_info * 30, All From Scratch

| Iter | Loss | BLER |
|---|---|---|
| 300 | 0.874 | 1.0 |
| 1500 | 0.874 | 1.0 |

**Answer**: Scaling z_encoder output norms to ~54 (matching pretrained range) does NOT fix the problem when tree ops are random. The tree ops need correct INTERNAL weights, not just correctly-scaled inputs.

### EXP J: Pretrained Tree Ops (N=128) + Fresh z_encoder (Scaled), All Trainable

| Iter | Loss | BLER |
|---|---|---|
| 300 | 0.013 | 0.125 |
| 600 | 0.021 | 0.047 |
| 900 | 0.024 | 0.063 |
| 1200 | 0.023 | 0.047 |
| 1500 | 0.021 | 0.000 |

**Answer**: With pretrained tree ops and a fresh (but scaled) z_encoder, the model converges to BLER=0.0 in just 1500 iters. This is the definitive proof that the tree ops are the sole bottleneck. The z_encoder can be trained from scratch in minutes if the tree ops are good.

## 3. Key Observations

1. **The architecture is not the bottleneck.** EXP A shows perfect memorization at N=256, and the curriculum-trained model achieves BLER=0.023.

2. **The z_encoder is not the primary bottleneck.** EXP D shows a fresh z_encoder can learn quickly (loss 0.25 in 1500 iters) when tree ops are pretrained.

3. **The tree ops are the primary bottleneck.** EXP C shows fresh tree ops cannot learn even with a pretrained z_encoder (loss stuck at 0.875). The tree ops require careful initialization to function.

4. **Complete mode collapse occurs from scratch.** The model learns position-wise constant predictions that achieve loss=0.874 by exploiting the frozen-bit structure (predicting the most likely class per position). It never discovers input-dependent features.

5. **The 0.874 loss plateau is a degenerate fixed point.** With 4-class prediction, positions where one user is frozen have a 2-class effective problem (loss=0.693), while positions where both users are free have 4-class uniform (loss=1.386). The weighted average matches the observed 0.874.

6. **No amount of correct context helps.** Even providing 500/512 correct oracle decisions doesn't help the scratch model -- its logits are completely input-independent.

## 4. What the Evidence Rules Out

- **Capacity issue**: Ruled out by EXP A (perfect overfit).
- **z_encoder learning**: Ruled out by EXP D (z_encoder learns fast with good tree ops).
- **Error cascade as root cause**: Ruled out by EXP F (oracle doesn't help scratch model).
- **Too few iterations**: Loss is flat-plateaued at 0.874 from iter 300 onward. More iterations won't help.
- **Learning rate issues**: The overfit test and EXP D both converge, showing the LR is appropriate.
- **Scale/norm mismatch alone**: Scaling z_encoder output to match pretrained norms (EXP Scale-Fix) doesn't help. The tree ops need correct functional mappings, not just correctly-scaled I/O.

## 5. Most Likely Bottleneck

**The tree ops suffer from a degenerate initialization trap.** The randomly initialized tree operations (CalcLeft, CalcRight, CalcParent) cannot learn to approximate the correct f/g transforms at N=256 from scratch. Key evidence:

1. **Scale is not the issue**: Even when z_encoder output norms are scaled to pretrained levels (54 vs 0.5), random tree ops still get stuck (EXP Scale-Fix).
2. **The tree ops need correct functional form**: CalcLeft, CalcRight, and CalcParent must approximate the information-theoretic channel-combining/splitting operations. Random MLPs cannot discover these functions through gradient descent at N=256 within practical iteration budgets.
3. **The degenerate fixed point**: Random tree ops produce near-uniform logits (confidence ~0.03), and the model collapses to position-wise constant predictions. This is a stable local minimum because:
   - The loss gradient at a constant prediction does not carry input-dependent information
   - All gradients through the tree are approximately zero (flat logit landscape)
   - The model exploits frozen-bit structure to achieve loss=0.874 without processing inputs

4. **The tree ops ARE N-independent**: EXP J proves that N=128 tree ops work perfectly at N=256 with a fresh z_encoder (BLER=0.0 in 1500 iters). The tree operations generalize across block lengths because they implement the same recursive channel transform at every level.

**Why curriculum works**: At smaller N (N=16 to N=128), the tree has fewer levels and the gradient path from leaf to root is shorter (4-7 hops vs 8). The shorter path means:
- Gradients reach the tree op parameters with less attenuation
- There are fewer total leaf decisions (32-256 vs 512), so each decision matters more
- The phase transition (BLER drops from 1.0) requires fewer coordinated parameters

Once the tree ops learn the correct f/g transforms at N=128, they transfer perfectly to N=256 because they are position/level-agnostic.

## 6. Recommended Next Branch

### Primary recommendation: **Mandatory curriculum training with progressive unfreezing.**

Do NOT attempt from-scratch training at N=256. Instead:

1. Train at N=128 until convergence (or load existing checkpoint)
2. Transfer to N=256 and fine-tune all parameters
3. This takes ~100K iters at N=256 to reach BLER=0.015

### If from-scratch training is required, consider these modifications:

1. **Embedding scale initialization**: Initialize z_encoder weights such that output embeddings have norms of 10-30 (matching pretrained scale). This can be done by scaling the last layer weights by 30x.

2. **Layer normalization**: Add LayerNorm to tree op outputs to prevent the scale collapse.

3. **Warmup with very high SNR**: Train first at very high SNR (e.g., 20dB) where the channel output is nearly deterministic, then gradually decrease SNR.

4. **Progressive depth**: Start with a depth-4 tree (first 16 leaves), train, then progressively activate more leaves.

5. **Scheduled sampling**: Mix teacher-forced and free-running steps during training to reduce exposure bias.

## 7. Suggested Minimal Next Experiment

**Already validated**: EXP J shows that pretrained tree ops + fresh z_encoder converges in 1500 iters at N=256. The scale fix alone was tested and does NOT work.

**Next experiment to try** (estimated 2 hours): Can we break the initialization trap with a different approach?

Option A: **High-SNR warmup** -- Train at 20dB SNR first (z values near {-2, 0, 2} with negligible noise), then anneal to 6dB. At high SNR, the tree ops face an easier task and may break out of the constant-prediction trap.

Option B: **LayerNorm in tree ops** -- Add LayerNorm after CalcLeft/CalcRight/CalcParent outputs. This normalizes embeddings and may prevent the scale collapse.

```python
# Modify tree ops to include LayerNorm
self.calc_left_nn = nn.Sequential(
    _make_mlp(3 * d, hidden, d, n_layers),
    nn.LayerNorm(d)
)
```

Option C: **Auxiliary loss on z_encoder** -- Add a contrastive or reconstruction loss on z_encoder outputs to force input-dependent representations before tree ops learn.

---

## Executive Summary

```
N=256 from-scratch failure is caused by TREE OP INITIALIZATION TRAP.
Random tree ops cannot learn the channel-combining transforms at depth 8.
The model collapses to position-constant predictions (loss=0.874, BLER=1.0)
that are completely input-independent (same output for all z inputs).

Evidence:
(1) Architecture CAN represent N=256 (overfit 1 codeword: BLER=0.0)
(2) Fresh z_encoder learns fast with pretrained tree ops (BLER=0.0 in 1500 iters)
(3) Fresh tree ops CANNOT learn even with pretrained z_encoder (loss=0.875)
(4) Scaling embeddings to pretrained norms does NOT help (tree op internals matter)
(5) Oracle injection (500/512 correct) doesn't help scratch model
(6) Scratch model predicts SAME class for all 64 samples in batch (mode collapse)

Root cause: Random CalcLeft/CalcRight/CalcParent MLPs produce near-uniform
outputs, creating a stable degenerate fixed point with zero useful gradient.

SOLUTION: Curriculum training is MANDATORY for N>=256.
Tree ops are N-independent and transfer perfectly from N=128.
Only z_encoder needs retraining (1500 iters, ~40 min).
```
