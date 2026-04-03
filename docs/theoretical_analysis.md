# Theoretical Analysis: Why the Neural Decoder Fails at N >= 256

## 1. The Observed Phenomenon

The neural SC decoder for the Gaussian MAC achieves BLER performance matching
the analytical SC decoder at block lengths N <= 128, but exhibits a persistent
~0.015 BLER ceiling at N >= 256 while the analytical SC BLER continues to
decrease (0.005 at N=256, 0.001 at N=512).

Key observations:
- The ceiling is ~0.015 BLER regardless of N (N=128: 0.017, N=256: 0.015, N=512: 0.012)
- SC BLER drops rapidly: 0.016 -> 0.005 -> 0.001 for N=128,256,512
- Errors are uniformly distributed across info positions (no specific failing positions)
- The per-codeword BER is ~0.27% (0.67 errors per codeword at N=256)
- Signal range does NOT collapse during the tree walk (CalcParent re-injects information)
- Larger model (d=32, 157K params) does not help without proportionally more training

## 2. Error Accumulation Analysis

### 2.1 Per-Operation Error Model

Let each MLP call introduce a small approximation error ε on the d-dimensional
embedding. If the MLP's output has an L2 error of ε relative to the true
analytical tensor operation, and the d-dimensional embedding carries the
equivalent of 4 log-probability values (for the 2×2 joint distribution), then
the information error per operation is approximately:

    δI ≈ ε · (∂I/∂embedding)

where I is the mutual information between the embedding and the true probability
tensor.

### 2.2 Error Accumulation Through Sequential Operations

For a block length N with Class B (interleaved) path:
- Total MLP calls per codeword: ~6N (CalcLeft + CalcRight + CalcParent + leaf ops)
- At N=256: ~1517 MLP calls
- At N=128: ~750 MLP calls

If each MLP call introduces an independent error ε, the accumulated error
after K calls grows as:

    E_total ≈ √K · ε    (independent errors)
    or
    E_total ≈ K · ε     (correlated/worst-case errors)

### 2.3 Connecting Error to BLER

Each leaf decision depends on the combined top-down and bottom-up embeddings.
An error in the embedding leads to an incorrect 4-class decision when the error
exceeds the margin between the correct class and the next-best class.

At N=256:
- Per-position BER = 0.0027 (0.67 errors / 246 info positions)
- This means the embedding error exceeds the decision margin 0.27% of the time
- The SC per-position BER at N=256 is ~0.0001 (0.04 errors / 246 info positions)
- So the NN's per-position error rate needs to decrease by 27x to match SC

### 2.4 Why More Training Doesn't Help

The gradient depth is O(N log N):
- At N=128: gradients flow through ~750 sequential operations
- At N=256: gradients flow through ~1500 sequential operations

The gradient signal for correcting a specific MLP's error at step t must
backpropagate through all subsequent operations (steps t+1 through K).
With K=1500, the gradient of the final loss with respect to an early operation
is subject to:

1. **Vanishing/exploding gradients**: Despite using ELU activations and gradient
   clipping (clip=1.0), the effective gradient magnitude at early operations
   is much smaller than at late operations.

2. **Gradient interference**: All MLP calls share the same weights
   (weight-sharing). The gradient update for one call may conflict with the
   optimal update for another call at a different tree position.

## 3. Embedding Dimension Analysis

### 3.1 Information Content of 2×2 Tensor

The analytical decoder works with 2×2 log-probability tensors:
```
T = [[log P(u=0,v=0), log P(u=0,v=1)],
     [log P(u=1,v=0), log P(u=1,v=1)]]
```

This has 3 degrees of freedom (4 values minus 1 normalization constraint).
A d=16 embedding has 16 dimensions — more than sufficient capacity for 3 DOF.

However, the embedding must represent not just the current tensor but also
carry information needed for:
- Reconstruction of the parent tensor (CalcParent needs to invert the
  CalcLeft/CalcRight operation)
- Conditioning on the full decoding history
- Supporting both CalcLeft and CalcRight operations from the same edge

### 3.2 Mutual Information Between Embedding and Tensor

We can estimate I(embedding; tensor) by:
1. Running the analytical decoder to get true 2×2 tensors at every edge
2. Running the neural decoder to get d-dimensional embeddings at every edge
3. Training a probe (MLP) to predict the tensor from the embedding
4. Measuring the prediction quality (MSE or KL divergence)

From the report (Section 9.5, snapshot training):
- Standalone probe achieves per-operation MSE of ~0.02
- This corresponds to ~0.14 nats of information loss per operation
- Over 1500 operations, this accumulates to significant distortion

### 3.3 Minimum Embedding Dimension

The minimum d to losslessly represent a 2×2 probability tensor is d=3
(3 degrees of freedom). However, the CalcLeft operation requires the
parent embedding (2d elements = parent_first_half + parent_second_half)
plus the sibling embedding (d elements), totalling 3d input dimensions.

For the MLP to implement the circular convolution without information loss,
d must be large enough that the MLP can represent the nonlinear operation:

    CalcLeft(parent, sibling) = circ_conv(parent_tensor, sibling_tensor)

The circular convolution involves products of exponentials (in log domain)
and logsumexp operations. A 2-layer MLP with hidden dimension h can represent
functions with complexity O(h²) Lipschitz constraints.

**Key insight**: The circular convolution is a specific structured operation
on 4-element tensors. With d=16, the MLP has 3×16 = 48 input dimensions
mapped to 16 output dimensions through a hidden layer of 64. This has
2×(48×64 + 64×64 + 64×16) = 2×(3072 + 4096 + 1024) = 16,384 parameters
per CalcLeft MLP call. This is ample capacity for a 4→4 tensor function.

The issue is not capacity per operation but error accumulation over many
sequential operations with shared weights.

## 4. Comparison with NPD (Single-User Neural Decoder)

### 4.1 Why NPD Scales to N=1024

The NPD (Neural Polar Decoder) by Aharoni et al. works at N=1024 because:

1. **Binary output**: NPD outputs 1 bit per leaf (not 4-class), using a
   sign-flip BitNode structure that naturally decomposes the g-node operation.
   The skip connection `output = MLP + parent_first * sign` IS the analytical
   formula — the MLP starts from a perfect decoder.

2. **Parallel training**: fast_ce processes all N positions at each tree level
   simultaneously, giving O(log N) gradient depth (only 10 sequential steps
   at N=1024) vs. our O(N log N) sequential depth (~10,000 steps).

3. **Smaller model**: NPD uses d=8, ~11K params vs. our d=16, ~39K params.
   The simpler model converges faster per training iteration.

### 4.2 Why the NPD Approach Cannot Directly Apply to MAC

The MAC 4-class joint structure doesn't decompose through binary sign flips:
- For single-user: g-node(parent, sign) = parent * sign + even is element-wise
- For MAC: CalcRight(parent, left) involves circular convolution of 2×2 tensors

We tried:
- 2-group sign encoding (d/2 for u, d/2 for v): loss plateau at 0.30
- WHT 4-group sign encoding: loss plateau at 0.29
- One-hot decision + residual: loss plateau at 0.30

All variants converge to the same ~0.30 loss plateau. The MAC's 2×2 circular
convolution doesn't factor into element-wise operations.

## 5. Theoretical Bounds

### 5.1 Finite-Width Approximation Error

By the universal approximation theorem, a 2-layer MLP can approximate any
continuous function to arbitrary precision given sufficient width. However:

- The approximation error ε decreases as O(1/√h) where h is hidden width
- With h=64, ε ≈ O(0.13)
- Over K=1500 calls, accumulated error ≈ K·ε ≈ 195 (worst case)
- Even with √K accumulation: √1500 · 0.13 ≈ 5.0

This suggests the per-operation accuracy with d=16, h=64 is fundamentally
limited to ~0.1-0.3% error per operation, which compounds to ~1-2 wrong
bit decisions per codeword at N=256.

### 5.2 Required Per-Operation Accuracy to Match SC at N=256

To achieve SC-level BLER at N=256 (BLER=0.005):
- Need ~0.25 errors per codeword (0.005 × 50% of codewords have ≥1 error)
- With 246 info positions: need BER ≈ 0.001 (0.25/246)
- Current NN BER: 0.0027

So per-position error needs to decrease by 2.7x. Given that each position's
accuracy depends on ~6-8 sequential MLP operations, each MLP needs to improve
its accuracy by roughly 2.7^(1/7) ≈ 1.15x, i.e., 15% improvement.

This is a training/optimization gap, not a fundamental capacity issue.

### 5.3 Potential Solutions

1. **Much longer training**: The d=32 model was never given enough training
   (30K iters vs needed 200K+). With sufficient compute, larger models may
   close the gap.

2. **Per-level MLPs**: Using separate MLPs per tree level eliminates the
   weight-sharing conflict. Validated to help (starts at BLER=0.265 vs 1.0
   for shared), but needs ~5x more training due to 189K params.

3. **Parallel training for MAC**: Finding a decomposition of the 4-class
   circular convolution that enables fast_ce-style parallel training would
   reduce gradient depth from O(N log N) to O(log N).

4. **Hybrid approach**: Use analytical operations where possible (e.g.,
   CalcLeft/CalcRight are channel-independent after sufficient depth) and
   only learn the z_encoder + CalcParent.

## 6. Summary

The N>=256 gap is caused by:
1. **Error accumulation**: ~0.3% error per MLP operation × 1500 operations = ~1 wrong bit
2. **Gradient depth**: O(N log N) = 1500 sequential steps makes fine-tuning difficult
3. **Weight sharing**: The same MLP must handle all tree levels, limiting specialization

This is NOT a fundamental limitation of neural decoders but a training/optimization
challenge. The key evidence: the BEMAC neural decoder (same architecture, same
weight sharing) achieves BLER matching or beating SC at ALL N including 1024.
The difference is that BEMAC has a discrete 3-symbol output alphabet (much easier
to learn), while GMAC has a continuous Gaussian mixture output requiring the
z_encoder MLP to maintain fine probability resolution.
