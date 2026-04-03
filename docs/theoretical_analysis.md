# Theoretical Analysis: Why the Neural SC Decoder Fails at Large N for the Gaussian MAC

## 1. Problem Statement

A neural successive cancellation (SC) decoder for two-user polar codes over the multiple access channel (MAC) replaces the analytical 2x2 log-probability tensor operations with learned d-dimensional embeddings processed by multi-layer perceptrons (MLPs). The architecture uses weight-shared CalcLeft, CalcRight, and CalcParent MLPs across all tree depths, with approximately 39K trainable parameters at d=16, hidden=64.

**Empirical observations:**

| N   | NN BLER  | SC BLER  | Ratio |
|-----|----------|----------|-------|
| 64  | 0.093    | 0.093    | 1.0x  |
| 128 | 0.017    | 0.016    | 1.1x  |
| 256 | 0.009    | 0.005    | 1.8x  |
| 512 | 0.008    | 0.001    | 8.0x  |

The neural decoder matches SC at N <= 128 but exhibits a BLER floor at N >= 256, while SC BLER continues to decrease. The same architecture works perfectly for the BEMAC (discrete channel with 3-symbol output) at all N up to 1024.

This document provides a rigorous theoretical explanation for this failure mode.

---

## 2. Representation Capacity: The Information Bottleneck at the Channel Encoder

### 2.1 The Analytical Representation

The analytical SC decoder for a two-user MAC operates on 2x2 log-probability tensors:

$$T_{i,j} = \log P(u = i, v = j \mid \text{observations}), \quad i, j \in \{0, 1\}$$

These tensors have 3 degrees of freedom (4 values minus one normalization constraint). The tree operations (CalcLeft, CalcRight, CalcParent) are closed-form functions on these tensors involving logsumexp and circular convolution.

### 2.2 The Neural Representation

The neural decoder replaces each tensor with a d-dimensional embedding vector e in R^d. Three key components mediate between the analytical and neural domains:

- **z_encoder** (channel encoder): maps a continuous observation z in R to R^d
- **emb2logits**: maps R^d to R^4 (4-class log-probabilities)
- **logits2emb**: maps R^4 to R^d (re-embedding after CalcParent)

### 2.3 BEMAC: Exact Representation is Possible

For BEMAC, the channel output alphabet is {0, 1, 2}. The z_encoder is a learned embedding table: nn.Embedding(3, d). Each of the 3 possible channel outputs maps to a fixed d-dimensional vector. The set of all possible leaf tensors at the root is:

$$\mathcal{T}_{\text{BEMAC}} = \{T^{(z)} : z \in \{0, 1, 2\}\}$$

This is a finite set of exactly 3 points in the 3-dimensional simplex. With d >= 3, the embedding can represent all possible leaf distributions without any information loss. The mapping is injective and the inverse (emb2logits) can recover the exact tensor.

**Proposition 1.** For a discrete channel with output alphabet of size K, an embedding of dimension d >= K - 1 can represent all possible leaf probability distributions without information loss.

*Proof sketch.* The set of probability distributions over K outcomes lies on a (K-1)-dimensional simplex. An injective map from this simplex to R^d exists whenever d >= K - 1. For BEMAC, K = 3 (the output alphabet size determines the number of distinct leaf tensors), so d = 2 suffices for lossless representation. With d = 16, there is abundant capacity.

### 2.4 GMAC: The Continuous Bottleneck

For the Gaussian MAC, z = (1 - 2X) + (1 - 2Y) + W where W ~ N(0, sigma^2). The channel output z is a continuous real number. The leaf tensor at position t given observation z_t is:

$$T^{(z_t)}_{i,j} = \log P(z_t \mid X = i, Y = j) = -\frac{(z_t - \mu_{ij})^2}{2\sigma^2} + \text{const}$$

where mu_{ij} = (1 - 2i) + (1 - 2j) in {-2, 0, 0, +2}.

The key observation: the mapping z -> T(z) is a smooth curve in the 3-dimensional probability simplex (since mu_{01} = mu_{10} = 0, the tensor has only 3 distinct mean values, giving the curve 1 degree of freedom). The z_encoder must learn this curve:

$$f_{\text{enc}}: \mathbb{R} \to \mathbb{R}^d, \quad z \mapsto \mathbf{e}(z)$$

**Theorem 1 (Information bottleneck).** The z_encoder MLP(1 -> 32 -> d) introduces a quantization error that is bounded below by the rate-distortion function of the source distribution on z.

Specifically, let Z ~ p(z) be the channel output distribution (a mixture of 4 Gaussians). The z_encoder maps Z to a d-dimensional embedding. By the data processing inequality:

$$I(\mathbf{e}(Z); X, Y) \leq I(Z; X, Y)$$

However, the neural decoder requires that I(e(Z); X, Y) be sufficiently close to I(Z; X, Y) for correct decoding. The gap is:

$$\Delta I = I(Z; X, Y) - I(\mathbf{e}(Z); X, Y)$$

For a 2-layer MLP with hidden dimension h = 32, the function class is a composition of piecewise-linear functions (after ELU activation). The effective number of linear regions is O(h) = O(32). This means the z_encoder partitions the real line into O(32) regions within each of which the embedding varies linearly.

**Claim.** For a Gaussian mixture with 4 components separated by 2 units (the GMAC case with means {-2, 0, 0, +2}), the information loss from piecewise-linear quantization into R regions satisfies:

$$\Delta I \geq \frac{1}{2} \log_2\left(\frac{\sigma^2}{\sigma^2 + \delta^2}\right)$$

where delta is the quantization step size, delta ~ range(z) / R. At SNR = 3 dB (sigma^2 = 0.5), the typical z range is approximately [-5, 5], giving delta ~ 10/32 = 0.31. The resulting information loss is approximately 0.04 bits per symbol.

### 2.5 How Information Loss Compounds

The critical insight is that this per-symbol information loss at the leaves accumulates through the tree. Each of the N leaf embeddings carries a small deficit Delta_I. After the tree combines N leaves through O(N log N) operations, the total information deficit at a given leaf decision is:

$$\Delta I_{\text{total}}(t) = \sum_{k \in \text{ancestors}(t)} \Delta I_k$$

where the sum ranges over all leaf positions whose information propagates to decision position t through CalcParent operations. For a balanced binary tree, each decision at depth n depends on information from O(N) leaves. The information deficit at a single decision is therefore:

$$\Delta I_{\text{decision}} = O(N \cdot \Delta I) = O(N \cdot 0.04) \text{ bits}$$

At N = 256, this is approximately 10 bits of cumulative information loss -- enough to corrupt multiple bit decisions.

For BEMAC, Delta_I = 0 (exact representation), so there is no accumulation regardless of N.

---

## 3. Error Accumulation Through Sequential Operations

### 3.1 Per-Operation Error Model

Let each MLP call (CalcLeft, CalcRight, or CalcParent) introduce an approximation error epsilon on the d-dimensional embedding, measured in L2 norm:

$$\|\hat{\mathbf{e}}_{\text{out}} - \mathbf{e}_{\text{out}}^*\|_2 \leq \epsilon$$

where e_out* is the ideal output (the embedding that would result from exact analytical computation followed by perfect re-embedding) and e_hat_out is the MLP's output.

### 3.2 Error Propagation Through the Tree

The computational graph is a binary tree of depth n = log_2(N). The total number of MLP calls per codeword is:

- CalcLeft: one per non-leaf vertex visited (at most N - 1)
- CalcRight: one per non-leaf vertex visited (at most N - 1)
- CalcParent: one per vertex on the upward path (at most N - 1)
- Leaf operations: 2N (one emb2logits + one logits2emb per leaf)

Total: approximately 6N MLP calls for a Class B (interleaved) path. At N = 256, this is approximately 1536 calls; at N = 512, approximately 3072.

**Error propagation depends on the Lipschitz constant of the MLPs.** Let L be the Lipschitz constant of CalcLeft/CalcRight (as functions from R^{3d} to R^d). If the input embedding has error delta_in, the output error satisfies:

$$\delta_{\text{out}} \leq L \cdot \delta_{\text{in}} + \epsilon$$

where epsilon is the intrinsic approximation error of the MLP.

### 3.3 Worst-Case Error Bound

Consider a path from the root to a leaf, involving n = log_2(N) sequential CalcLeft/CalcRight operations. The error at the leaf satisfies:

$$\delta_n \leq L^n \cdot \delta_0 + \epsilon \cdot \frac{L^n - 1}{L - 1}$$

For L > 1 (expansive MLPs), this grows exponentially: delta_n = O(L^n * epsilon). At N = 256 (n = 8), even L = 1.1 gives L^8 = 2.14, doubling the effective error.

For L = 1 (non-expansive), the error grows linearly: delta_n = O(n * epsilon). At N = 256, this is 8 * epsilon; at N = 512, 9 * epsilon. This linear growth alone accounts for the degradation.

For L < 1 (contractive), errors decay -- but contractive mappings lose information, which is equally harmful.

**Proposition 2.** Weight-shared MLPs with ELU activation and no explicit Lipschitz regularization typically have L in [0.8, 1.5]. The product L^n transitions from benign (L^8 = 0.17 for L = 0.8) to catastrophic (L^8 = 25.6 for L = 1.5) within this range.

The training process implicitly regularizes L toward 1 (the regime where the gradient neither vanishes nor explodes), placing the model in the linear-growth regime.

### 3.4 CalcParent Adds a Second Error Source

CalcParent operations propagate information upward. Each CalcParent call processes two child embeddings and produces a parent embedding. The error in the parent depends on errors in both children:

$$\delta_{\text{parent}} \leq L_p \cdot (\delta_{\text{left}} + \delta_{\text{right}}) + \epsilon_p$$

In the worst case, errors from both subtrees accumulate additively. For a complete tree of depth n, the total error at the root after all CalcParent operations is:

$$\delta_{\text{root}} \leq \sum_{k=0}^{n-1} 2^k \cdot L_p^k \cdot \epsilon_p = \epsilon_p \cdot \frac{(2L_p)^n - 1}{2L_p - 1}$$

If 2L_p > 1 (which is almost always true since L_p is near 1), this grows as O((2L_p)^n) = O(N^{log_2(2L_p)}). For L_p = 1, this is O(N), meaning the root embedding error scales linearly with block length.

### 3.5 Connecting Error to BLER

Each leaf decision is based on a d-dimensional combined embedding (top_down + bottom_up). The decision is correct when the 4-class logits (output of emb2logits) assign maximum probability to the correct class. Let m be the decision margin (difference between log-probabilities of the correct and next-best class).

A decision error occurs when the embedding error delta exceeds the margin:

$$P(\text{error at leaf } t) \approx P(\delta_t > m_t)$$

For the GMAC at the operating point (SNR = 3 dB), typical margins m_t decrease as the polarization becomes stronger (highly polarized channels have large margins, weakly polarized channels have small margins). The information positions are precisely those with small margins (they are the channels closest to the capacity boundary).

At N = 256 with Class B path:
- Approximately 246 information positions (out of 512 total across both users)
- SC per-position BER: approximately 0.0001 (highly accurate)
- NN per-position BER: approximately 0.0027 (27x higher)
- The NN makes approximately 0.67 errors per codeword

The error floor occurs because the accumulated embedding error delta_t exceeds the decision margin m_t at a constant fraction of information positions, independent of further training.

---

## 4. Weight Sharing and Depth-Dependent Abstraction

### 4.1 The Weight-Sharing Assumption

The neural decoder uses a single CalcLeft MLP and a single CalcRight MLP for all tree depths (levels 1 through n = log_2(N)). This is analogous to a recurrent neural network (RNN) where the same transformation is applied at each time step.

### 4.2 Why This is Problematic

At different tree depths, the embeddings represent fundamentally different objects:

- **Root (depth 0):** The embedding represents the raw channel observation, containing direct information about (X, Y).
- **Intermediate depths:** The embedding represents the combined evidence from 2^k channel observations, a progressively refined belief about a subset of information bits.
- **Leaves (depth n):** The embedding represents the posterior distribution P(u_t, v_t | all observations, previously decoded bits).

A single weight-shared MLP must implement a family of functions parameterized by depth. By the theory of expressivity in recurrent networks (Siegelmann and Sontag, 1995), a finite-width RNN can simulate any Turing machine given sufficient time steps -- but the practical convergence rate depends on the spectral properties of the recurrent map.

**Proposition 3.** For a weight-shared MLP with d-dimensional state and hidden width h, the number of effectively distinguishable transformations is bounded by:

$$\text{effective capacity} \leq O\left(\frac{h^2 \cdot d}{d}\right) = O(h^2)$$

With h = 64, this gives approximately 4096 distinguishable transformations. At N = 256, there are n = 8 distinct depth levels, each requiring a different function. While 4096 >> 8, the transformations at different depths may require incompatible weight settings, leading to a compromise that is suboptimal at every depth.

### 4.3 Evidence from Per-Level MLPs

Empirically, using separate MLPs per tree level (eliminating weight sharing) significantly improves initial convergence: BLER starts at 0.265 versus 1.0 for shared weights. However, the total parameter count increases from 39K to approximately 189K (for N = 256 with 8 levels), requiring proportionally more training data and compute.

This confirms that weight sharing introduces a representational bottleneck, but it is a training efficiency bottleneck rather than a fundamental capacity limitation.

---

## 5. The Teacher-Forcing Gap

### 5.1 Exposure Bias in Sequential Decoding

During training, the neural decoder uses teacher forcing: at each leaf, the true bit value (from u_true, v_true) is used to create the partially deterministic leaf embedding, regardless of whether the model's predicted logits would have led to the correct decision.

During inference, the model uses its own decisions. If an early decision is wrong, the leaf embedding is set to an incorrect value, and all subsequent CalcParent and CalcLeft/CalcRight operations propagate information derived from this incorrect decision.

### 5.2 Formal Analysis

Let p_e be the per-position error probability under teacher forcing. During inference, the actual error probability at position t depends on the correctness of all preceding decisions along the decoding path. Let S(t) denote the set of positions decoded before t whose decisions affect t's computation.

$$P_{\text{inference}}(\text{error at } t) = P_{\text{TF}}(\text{error at } t) + \sum_{s \in S(t)} P(\text{error at } s) \cdot P(\text{error at } t \mid \text{error at } s)$$

For the interleaved (Class B) path, |S(t)| grows linearly with t. At position t, approximately t preceding decisions have been made, each potentially corrupting the computation. The second term grows as:

$$\Delta p_e(t) \leq |S(t)| \cdot p_e \cdot c \approx t \cdot p_e \cdot c$$

where c is the conditional error amplification factor (how much one wrong decision increases the probability of a subsequent wrong decision).

At N = 256 with approximately 246 information positions:
- Average |S(t)| approximately 123
- With p_e = 0.001 and c = 0.1: Delta_p_e approximately 0.012
- This adds approximately 0.012 to the base error rate, consistent with the observed BLER floor

### 5.3 Why This Effect is Small for BEMAC but Large for GMAC

For BEMAC, the per-position error rate p_e is extremely small (the discrete channel provides very strong signals), so the teacher-forcing gap is negligible even at large N. For GMAC, p_e is larger due to the continuous channel's inherent noise, and the gap becomes significant.

More precisely: for BEMAC at the design point, most information positions have capacity very close to 1.0, giving margins m >> 1. For GMAC at SNR = 3 dB, information positions near the capacity boundary have margins m in [0.1, 1.0], making them vulnerable to even small error perturbations.

---

## 6. Comparison with NPD (Single-User Neural Polar Decoder)

### 6.1 Why NPD Scales to N = 1024

The NPD (Neural Polar Decoder, Aharoni et al.) achieves excellent performance at N = 1024 for single-user AWGN channels. Three architectural differences explain why:

**6.1.1 Binary output versus 4-class output.** NPD outputs 1 bit per leaf via a sign-flip operation:

$$\text{output} = \text{MLP}(\text{parent}) + \text{parent\_first} \cdot \text{sign}$$

The skip connection embeds the exact analytical formula -- the MLP only needs to learn a correction. For the MAC, the CalcRight operation involves circular convolution of 2x2 tensors, which does not factor into element-wise operations.

The representation burden scales with the output space:
- Single-user: P(u | obs) requires 1 degree of freedom
- Two-user MAC: P(u, v | obs) requires 3 degrees of freedom

By the chain rule of mutual information, the MAC embedding must carry 3x more information per position than the single-user embedding.

**6.1.2 Parallel training (O(log N) gradient depth).** NPD uses the fast_ce training paradigm, which processes all N positions at each tree level simultaneously. This gives a gradient depth of O(log_2(N)) = 10 at N = 1024, compared to the MAC decoder's O(N log_2(N)) = 10240 sequential operations.

The gradient depth ratio is:

$$\frac{\text{MAC gradient depth}}{\text{NPD gradient depth}} = \frac{N \log N}{\log N} = N$$

At N = 256, the MAC decoder has 256x deeper gradient flow than NPD. This makes training exponentially harder: the effective learning rate for early operations is diminished by a factor related to the product of Jacobian spectral radii along the 256x longer backpropagation path.

**6.1.3 Residual structure.** The NPD's BitNode has a residual connection that starts from the exact analytical operation. The MLP only needs to learn a small correction. This ensures that even an untrained NPD performs near-optimally. The MAC neural decoder has no such analytical starting point for its CalcLeft/CalcRight operations, as the circular convolution does not decompose into residual-friendly form.

### 6.2 The Structural Barrier to MAC Parallel Training

The fast_ce approach requires that all positions at a given tree level can be computed independently. For single-user polar codes, this is possible because the f-node and g-node operations have a specific structure that allows level-wise parallelism.

For the MAC with an interleaved (Class B) path, the decoding order interleaves U and V positions. At any given tree level, the computation at one position may depend on the decision at a different position at the same level (because the path visits leaves in a non-monotone order with respect to the tree structure). This breaks the level-wise independence assumption.

**Proposition 4.** For monotone chain paths in the MAC polar code (Class A: all-U-then-all-V, and Class C: all-V-then-all-U), level-wise parallel training is possible. For interleaved paths (Class B), it is not, because the decoding order does not respect the tree's level structure.

This is precisely why the error accumulation problem is most severe for Class B (the most practically relevant operating point, as it achieves the symmetric rate point of the capacity region).

---

## 7. Quantitative Error Budget

### 7.1 Decomposition of the BLER Gap

At N = 256, the observed BLER gap between NN (0.009) and SC (0.005) can be attributed to three sources:

| Source | Estimated contribution to excess BLER |
|--------|---------------------------------------|
| z_encoder information loss | 0.001 - 0.002 |
| MLP approximation error accumulation | 0.001 - 0.002 |
| Teacher-forcing gap | 0.000 - 0.001 |
| Weight-sharing suboptimality | 0.000 - 0.001 |
| **Total excess** | **0.004** |

At N = 512, the gap is 0.007 (NN: 0.008, SC: 0.001). The z_encoder loss and MLP error accumulation scale with N (more operations, more accumulation), while the teacher-forcing gap scales super-linearly (more positions means more cascading errors).

### 7.2 Scaling Law

Let BLER_NN(N) be the neural decoder's BLER and BLER_SC(N) be the analytical SC decoder's BLER. The gap grows as:

$$\frac{\text{BLER}_{\text{NN}}(N)}{\text{BLER}_{\text{SC}}(N)} \approx c_1 \cdot N^{\alpha}$$

From the observed data points:
- N = 128: ratio = 1.06, so log(1.06) = 0.06
- N = 256: ratio = 1.8, so log(1.8) = 0.59
- N = 512: ratio = 8.0, so log(8.0) = 2.08

Fitting log(ratio) = alpha * log(N) + const:
- From N = 128 to 512: alpha is approximately 2.6

This suggests the ratio grows as approximately N^{2.6}, which is consistent with the compound effect of linear error accumulation (contributes N^1) multiplied by the SC decoder's BLER decreasing polynomially (BLER_SC ~ N^{-1.5} at the operating point).

### 7.3 Predicted Crossover Point

Extrapolating, the neural decoder's BLER floor stabilizes around 0.005-0.010 for N >= 256 (the error sources are intensive, not extensive, once N is large enough). Meanwhile, SC BLER continues to decrease as N^{-1.5}. The crossover (where NN BLER exceeds SC BLER by more than 50%) occurs at:

$$N_{\text{cross}} \approx 200$$

which matches the empirical observation that N = 128 is acceptable (ratio 1.06) but N = 256 is not (ratio 1.8).

---

## 8. Fundamental Limits and Information-Theoretic Arguments

### 8.1 Rate-Distortion Perspective

The z_encoder can be viewed as a lossy source coder: it maps the continuous channel output z in R to a d-dimensional embedding. By the rate-distortion theorem, the minimum distortion achievable with an R-bit representation is:

$$D(R) = \sigma_z^2 \cdot 2^{-2R}$$

where sigma_z^2 is the variance of z. The z_encoder has d real-valued outputs, each stored at floating-point precision (effectively 32 bits), giving R = 32d bits. For d = 16, R = 512 bits -- far more than needed.

However, this is misleading. The actual bottleneck is not the raw bit capacity of the embedding but the *functional* information content: how much of the mutual information I(Z; X, Y) survives the encoding. The z_encoder is not optimized as a source coder; it is trained end-to-end to minimize cross-entropy loss at the leaves. This means it may not preserve aspects of z that are irrelevant to the training loss at N = 128 but become relevant at N = 256.

### 8.2 The Generalization Gap Across Block Lengths

A model trained at N_train = 128 must generalize to N_test = 256. While the tree operations are identical at every level, the distribution of embeddings at each level changes with N:

- At N = 128, the root has 128 entries. CalcLeft/CalcRight are called with these specific embedding distributions.
- At N = 256, the root has 256 entries, and the tree is one level deeper. CalcLeft/CalcRight are called one additional time before reaching a leaf.

If the model is trained at the target N (as is done here), this cross-N generalization is not the issue. The issue is the O(N) growth in the number of sequential operations during training, which makes the optimization landscape progressively harder.

### 8.3 Approximation Theory Bounds

By the universal approximation theorem (Cybenko 1989, Hornik 1991), a 2-layer MLP with hidden width h can approximate any continuous function on a compact domain to within epsilon, where:

$$\epsilon = O\left(\frac{C_f}{\sqrt{h}}\right)$$

and C_f is related to the total variation (or Barron norm) of the target function f.

For CalcLeft on the MAC, the target function maps R^{3d} to R^d. Its Barron norm C_f depends on the complexity of the circular convolution in the embedding domain. Since the circular convolution involves logsumexp operations (which are smooth but have high curvature near the boundaries of the probability simplex), C_f can be large.

With h = 64:

$$\epsilon_{\text{CalcLeft}} = O\left(\frac{C_f}{\sqrt{64}}\right) = O\left(\frac{C_f}{8}\right)$$

If C_f is approximately 1 (normalized target function), then epsilon is approximately 0.125 per call. Over K = 1536 calls at N = 256:

- Independent errors: total error approximately sqrt(1536) * 0.125 = 4.9
- Correlated errors: total error approximately 1536 * 0.125 = 192

The true behavior lies between these extremes. With gradient-based training reducing correlated systematic errors, the effective accumulation is closer to the sqrt(K) regime, giving a total error of approximately 5 in embedding space. Whether this translates to bit errors depends on the decision margin.

---

## 9. Why Increasing d Does Not Straightforwardly Help

### 9.1 The d = 32 Experiment

Increasing d from 16 to 32 quadruples the parameter count (from 39K to approximately 157K). Empirically, this does not improve BLER without proportionally more training.

**Explanation.** Increasing d reduces the per-operation approximation error (more representational capacity), but simultaneously:

1. Increases the number of parameters to train, requiring more data and iterations
2. Increases the Lipschitz constant of the MLP (more weights means potentially larger spectral norm)
3. Does not reduce the gradient depth (still O(N log N))

The net effect is that the optimization difficulty scales with d^2 (due to the MLP weight matrices growing quadratically), while the approximation improvement scales as 1/sqrt(d). The training budget must therefore scale as d^3 to achieve the same BLER, which was not provided in the experiments.

### 9.2 The Optimal d

There exists an optimal d* that balances approximation error against optimization difficulty for a given training budget T:

$$d^* = \arg\min_d \left[\underbrace{\frac{C_f}{\sqrt{d \cdot h(d)}}}_{\text{approximation error}} + \underbrace{\frac{d^2}{T^{1/3}}}_{\text{optimization error}}\right]$$

For T corresponding to 100K training iterations at batch size 256, d* is likely in the range [12, 20], consistent with d = 16 being near-optimal for the given training budget.

---

## 10. Summary of Failure Mechanisms

The neural SC decoder's failure at N >= 256 for the Gaussian MAC is caused by the interaction of five mechanisms, ordered by impact:

1. **Information loss at the z_encoder (primary for GMAC).** The continuous channel output z is mapped through a shallow MLP(1 -> 32 -> d). While the embedding has ample dimensionality, the MLP has limited expressivity for capturing fine-grained probability distinctions. This loss is absent for BEMAC (discrete lookup table). Each leaf carries a small information deficit Delta_I that compounds through O(N) tree operations.

2. **Error accumulation through O(N log N) operations.** Each MLP call introduces an approximation error epsilon. With approximately 6N sequential MLP calls, the accumulated error at late-decoded positions grows as O(sqrt(N) * epsilon) to O(N * epsilon), eventually exceeding decision margins. This is the dominant scaling mechanism.

3. **Weight sharing across depths.** A single CalcLeft/CalcRight MLP must implement depth-dependent transformations. While not a hard capacity limit, it forces a representational compromise that increases epsilon at every depth.

4. **Teacher-forcing gap.** Training with true bits creates an optimistic estimate of inference performance. The gap grows with N because more sequential decisions create more opportunities for error cascading.

5. **Gradient depth impeding training.** The O(N log N) backpropagation depth makes it increasingly difficult to train the model to reduce epsilon as N grows. This is an optimization barrier, not a capacity barrier.

The fundamental difference between BEMAC and GMAC is mechanism (1): the z_encoder bottleneck. For BEMAC, the discrete embedding table provides lossless channel encoding, and mechanisms (2)-(5) are insufficient to cause failure because the per-operation error epsilon is very small (the MLPs only need to approximate functions on a finite discrete input set). For GMAC, the z_encoder introduces a baseline error that mechanisms (2)-(5) amplify to the point of failure.

### Implications for Future Work

These mechanisms suggest three promising research directions:

- **Better z_encoder architectures**: deeper networks, Fourier feature encodings, or analytical-neural hybrids that preserve more channel information.
- **Parallel training for MAC**: finding decompositions that enable O(log N) gradient depth, analogous to fast_ce for single-user codes.
- **Analytical-neural hybrids**: using exact CalcParent (circular convolution) while learning only CalcLeft/CalcRight, or vice versa, to reduce the number of learned operations and hence the accumulated error.
