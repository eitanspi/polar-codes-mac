# Neural SC Decoding of Polar Codes for the Two-User Gaussian MAC
## 5-Page Summary Report (March 2026)

---

## 1. Problem Statement

We develop a **neural network-based successive cancellation (SC) decoder** for polar codes on the two-user Gaussian Multiple Access Channel (GMAC). Two users transmit independently encoded polar codewords X = F^n(U) and Y = F^n(V) through Z = (1-2X) + (1-2Y) + W, where W ~ N(0, sigma^2). The receiver must decode both U and V.

The analytical SC decoder achieves excellent BLER by walking the polar code factor graph tree and computing exact transition probabilities. Our goal: replace ALL analytical operations with learned neural networks, enabling deployment on channels where analytical probabilities are unavailable.

**Channels implemented**: BEMAC (Z=X+Y, discrete), ABNMAC (binary noise), GaussianMAC (BPSK + AWGN).

**Decoding paths**: Three classes determined by the decoding order of U and V bits:
- Class A/C (extreme paths): decode one user entirely before the other — simpler, no CalcParent needed
- Class B (interleaved): alternating U and V decisions — hardest, requires CalcParent operations

---

## 2. Neural Decoder Architecture

The decoder replaces each analytical tree operation with a learned MLP:

| Operation | Analytical | Neural | Input -> Output |
|-----------|-----------|--------|-----------------|
| Channel embedding | W(z\|x,y) | z_encoder MLP | R -> R^d |
| CalcLeft (f-node) | Circ. conv. | MLP | R^{3d} -> R^d |
| CalcRight (g-node) | Cond. product | MLP | R^{3d} -> R^d |
| CalcParent | Inv. circ. conv. | Gated residual MLP | R^{2d} -> R^d |
| Decision | Prob. extraction | MLP | R^d -> R^4 |

**Key design**: All operations are **weight-shared** across tree positions and depths, making the model N-independent (~25K-39K parameters). A model trained at N=128 can decode at any N.

**CalcParent** uses a gated residual structure: `output = gate * MLP(input) + (1-gate) * (left+right)/2`, providing stable gradient flow through deep trees.

**Training**: Sequential teacher-forced SC tree walk with cross-entropy loss on the 4-class joint (u,v) decisions. Curriculum learning (N=16 -> 32 -> 64 -> 128 -> ...) is essential — from-scratch training fails at N >= 32.

---

## 3. Key Results

### BEMAC (solved)
The neural decoder **matches or beats** the analytical SC decoder at ALL N up to 1024:
- N=64: NN-SC achieves 0.50x SC BLER (2x better)
- Neural SCL(L=4) beats analytical SCL(L=4) by up to 8x at N=64

### GMAC (partially solved)

| N | SC BLER | NN-SC BLER | Ratio | Status |
|---|---------|-----------|-------|--------|
| 32 | 0.046 | 0.046 | **1.0x** | Solved |
| 64 | 0.025 | 0.026 | **1.03x** | Solved |
| 128 | 0.016 | 0.019 | **1.17x** | Close |
| 256 | 0.005 | 0.019 | 3.7x | Gap |
| 512 | 0.001 | 0.045 | 40x | Large gap |

**N <= 128 is essentially solved.** The gap at N >= 256 is the core open problem.

### SCL Baselines (analytical, for comparison)
SCL(L=4) at GMAC 6dB: BLER = 0.026 (N=32), 0.013 (N=64), 0.008 (N=128), **0.0005 (N=256)**, 0.000 (N>=512).

---

## 4. What Was Tried

### Approaches that helped:
- **Stable cosine LR** (no warm restarts): Improved N=128 from 1.69x to 1.17x SC
- **Larger model (d=32, 157K params)**: Matches SC at N=32 (vs d=16's 1.22x), but under-trained at N >= 64
- **Per-level CalcLeft/CalcRight**: Separate MLPs per tree depth — solves curriculum transfer failure (d=32 model started at BLER=1.0 at N=128, per-level started at 0.265)
- **Neural SCL**: List decoding on NN output beats analytical SCL(L=4) at N <= 64

### Approaches that failed:
- **NPD-style fast_ce (parallel training)**: Works for single-user binary, but 4-class MAC joint prediction plateaus at near-random loss. Two-decoder binary approach impossible for Class B (R_u > marginal capacity).
- **d=32 at N >= 64**: Needs disproportionately more training, curriculum transfer fails entirely at N=128
- **Knowledge distillation variants, LLR front-end, FiLM conditioning**: Faster convergence but same BLER ceiling
- **BEMAC-to-GMAC weight transfer**: Works at N=32, fails at N >= 64

### Root cause of N >= 256 failure:
Weight-shared MLPs accumulate small per-node approximation errors through log2(N) tree levels. At N=256 (8 levels), the accumulated error dominates. The sequential training's O(N log N) gradient depth exacerbates this by making it hard to learn fine-grained corrections at deep levels.

---

## 5. Open Problems and Directions

1. **Adapting NPD's fast_ce to MAC**: The single-user NPD uses binary sign-flips that match the analytical g-node structure. The MAC's 2x2 circular convolution doesn't factor this way. A correct parallel training formulation for the joint MAC decoder is the highest-impact unsolved problem.

2. **Residual connections in CalcLeft/CalcRight**: NPD's BitNode uses `output = MLP(...) + analytical_approximation`. Our CalcLeft/CalcRight have no residual structure. Adding one could stabilize training through deep trees.

3. **Per-level with long training**: Per-level ops solve curriculum transfer. Combined with stable LR and sufficient budget (days), this could close the N=256 gap.

4. **Architecture search**: Transformer-based tree operations, attention across levels, or other architectures that don't accumulate errors multiplicatively.

5. **GA design is wrong for Class B**: Must use MC design. This was a critical bug that went unnoticed for a long time.

**Code and data**: https://github.com/eitanspi/polar-codes-mac
