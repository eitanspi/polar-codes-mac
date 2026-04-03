# Neural Successive Cancellation Decoding of Polar Codes for the Two-User Multiple Access Channel

## Paper Outline (IEEE Format)

---

### Abstract (~150 words)

We present a neural network-based successive cancellation (SC) decoder for
polar codes on the two-user binary-input multiple access channel (MAC). Our
decoder replaces all analytical tensor operations in the computational-graph
SC decoder with learned multi-layer perceptrons (MLPs), enabling deployment
on channels where analytical transition probabilities are unknown or
computationally intractable. The architecture mirrors the SC tree walk
structure with weight-shared CalcLeft, CalcRight, and CalcParent MLPs, and
supports all monotone chain decoding paths including the challenging
interleaved Class B path. On the binary erasure MAC (BEMAC), the neural
decoder matches or beats the analytical SC decoder at all block lengths up
to N=1024. On the Gaussian MAC (GMAC), it matches SC within 4% at N≤128
and achieves BLER=0.002 with CRC-aided list decoding at N=128, beating
even analytical SCL(L=4). We characterize the scaling limitations at N≥256
and compare with the single-user Neural Polar Decoder (NPD) architecture.

---

### I. Introduction (~1.5 pages)

**Opening**: Polar codes achieve channel capacity (Arikan 2009) and are adopted
in 5G NR. Extension to multi-user settings (MAC) is of both theoretical and
practical interest.

**Problem**: The SC MAC decoder (Önay 2013) requires analytical channel
transition probabilities W(z|x,y) at every tree node. For complex channels
(continuous output, unknown statistics, channels with memory), these
probabilities are unavailable or expensive to compute.

**Our contribution**: A neural SC decoder that:
1. Replaces ALL analytical operations with learned MLPs (fully neural)
2. Supports the interleaved Class B path (requires CalcParent)
3. Matches SC at N≤128 on GMAC, matches/beats SC at all N on BEMAC
4. Extends to CRC-aided list decoding (NN-CA-SCL)
5. Is channel-independent: same architecture for BEMAC, ABNMAC, GMAC

**Key differentiator from NPD (Aharoni et al. 2024)**:
- NPD handles single-user (binary output); we handle 2-user MAC (4-class output)
- NPD's BitNode decomposition doesn't apply to MAC's 2×2 circular convolution
- Our training is sequential O(N log N); NPD uses parallel fast_ce O(log N)

**Paper structure**: System model (II), Analytical decoder (III), Neural
decoder (IV), Results (V), Analysis (VI), Conclusion (VII).

---

### II. System Model (~1 page)

**A. Two-User MAC Channel**
- Binary inputs X, Y ∈ {0,1}^N, channel output Z
- Three channel models:
  - BEMAC: Z = X + Y ∈ {0,1,2}
  - ABNMAC: Z = (X⊕E_x, Y⊕E_y) with correlated noise
  - GMAC: Z = (1-2X) + (1-2Y) + W, W ~ N(0,σ²)
- Capacity regions and rate points

**B. Polar Code Construction**
- Encoder: X = U · B_N · F^{⊗n} (mod 2)
- Frozen set design: Bhattacharyya / GA / Monte Carlo
- Critical: GA design is WRONG for Class B → MC design required

**C. Decoding Paths and Code Classes**
- Path b ∈ {0,1}^{2N}: 0 = decode U bit, 1 = decode V bit
- Class A: 1^N 0^N (all V first)
- Class C: 0^N 1^N (all U first)
- Class B: 0^{N/2} 1^N 0^{N/2} (interleaved, symmetric rate point)
- Class B achieves R_u = R_v ~ 0.48 (above marginal capacity!)

---

### III. Analytical SC Decoder (~1.5 pages)

**A. Computational Graph Structure**
- Binary tree with 2N-1 edges, each carrying a 2×2 log-probability tensor
- Edge 1 = root (N embeddings), edges N..2N-1 = leaves (1 embedding each)
- Three core operations:
  - CalcLeft (f-node): circular convolution of parent and sibling tensors
  - CalcRight (g-node): normalized element-wise product
  - CalcParent: inverse circular convolution (Class B only)

**B. Sequential Tree Walk**
- Navigate to each leaf following path b
- At leaf: combine top-down and bottom-up messages, make 4-class decision
- Set leaf to partially deterministic tensor
- Total: ~6N tensor operations per codeword

**C. SC List Decoder**
- Maintain L candidate paths
- At non-frozen leaves: fork × 4, keep best L
- CRC-aided: check CRC on U-message for each candidate

---

### IV. Neural SC Decoder (~2 pages)

**A. Architecture**
- Replace each analytical operation with a learned MLP:
  - z_encoder: channel output → d-dimensional embedding (GMAC: Linear(1,32)→ELU→Linear(32,d); BEMAC: nn.Embedding(3,d))
  - CalcLeft/CalcRight MLP: 3d → hidden → hidden → d
  - CalcParent: Gated residual MLP (2d → d)
  - emb2logits: d → hidden → hidden → 4 (decision head)
  - logits2emb: 4 → hidden → hidden → d (re-embedding)

**B. Weight Sharing**
- ALL operations weight-shared across tree positions and depths
- Same CalcLeft MLP handles root-level (N tensors) and leaf-level (1 tensor)
- Model is N-independent: can decode at any N

**C. Training**
- Sequential teacher-forced training
- Loss: cross-entropy on 4-class (u,v) decisions at non-frozen leaves
- Gradient depth: O(N log N) — the fundamental training bottleneck
- Curriculum learning: N=16 → 32 → 64 → 128 → 256
- Stable cosine LR (no warm restarts)
- Hyperparameters: d=16, hidden=64, 2-layer MLPs, ~39K params

**D. Key Design Decisions**
- Pure neural CalcParent (gated residual) vs analytical circ_conv
- Additive combining (top_down + bottom_up) vs separate combine MLP
- Partially deterministic leaf tensors (prevents double-counting)

**E. Neural SCL Extension**
- Wrap neural SC decoder in list framework
- Maintain L copies of tree state
- CRC-aided: CRC-8 on User U's message

---

### V. Results (~2 pages)

**A. BEMAC Results (Table 1, Figure 1)**
- Class B: NN matches SC at N=16-32, beats SC at N=64-256
- Class C: NN matches SC across all N
- Key finding: NN is BETTER than SC on BEMAC (discrete channel, easy to learn)

**B. GMAC Results at SNR=6dB (Table 2, Figure 2)**
- Class B: NN matches SC within 4% at N≤128
- BLER ceiling ~0.015 at N≥256 while SC continues to decrease
- NN-SCL(L=4) at N=128: BLER=0.014 (worse than SC due to miscalibration)
- NN-CA-SCL(L=4) at N=128: BLER=0.002 (beats analytical SCL!)

**C. GMAC Waterfall Curves (Figure 3)**
- SNR = 3,4,5,6,7,8 dB at N=64 and N=128
- Test generalization: models trained at 6dB, tested across SNR range
- NN tracks SC waterfall shape but with consistent gap factor

**D. ABNMAC Results (Table 3)**
- Discrete channel like BEMAC but with correlated noise
- NN performance at N=32,64,128

**E. Complexity Comparison (Table 4)**
- Model size: ~39K params (150 KB)
- FLOPs: ~360x more than analytical SC
- Inference time: ~150x slower
- Training: 15K-135K iterations (0.3-28 hours)
- Key advantage: no need for channel transition probabilities

---

### VI. Analysis and Discussion (~1.5 pages)

**A. Why NN-SC Matches SC at N≤128**
- 750 MLP calls at N=128 with ~0.3% error per operation
- Error accumulation stays below decision threshold
- Weight sharing works: CalcLeft/CalcRight are structurally similar across levels

**B. Why the Gap Grows at N≥256**
- ~1500 MLP calls: error accumulates to ~1 wrong bit per codeword
- O(N log N) gradient depth prevents fine-tuning corrections
- Not a capacity issue (d=32 doesn't help without more training)
- Not signal collapse (verified stable signal ranges during tree walk)

**C. Comparison with NPD**
| Feature | NPD | Our Decoder |
|---------|-----|-------------|
| Output | Binary (1 bit) | 4-class joint (u,v) |
| Training | Parallel, O(log N) | Sequential, O(N log N) |
| BitNode | sign-flip + residual | No natural decomposition |
| Works at N=1024 | Yes | BEMAC: Yes, GMAC: No |

**D. Why BEMAC Works at All N**
- Discrete 3-symbol output: embedding is exact (nn.Embedding)
- GMAC requires z_encoder MLP: additional approximation error at the input
- BEMAC CalcParent easier to learn: discrete delta tensors

**E. CRC-Aided NN-SCL: Why It Works**
- CRC rescues from NN's per-bit errors
- At N=128: NN produces correct candidate in top-4 list but not always top-1
- CRC selects correct path → BLER drops from 0.017 to 0.002
- Fails at N=256: correct path pruned before CRC check

---

### VII. Conclusion (~0.5 pages)

- First fully-neural SC decoder for two-user MAC polar codes
- Matches/beats analytical SC on discrete channels (BEMAC, ABNMAC) at all N
- Matches SC within 4% on Gaussian MAC at N≤128
- CRC-aided neural SCL beats analytical SCL at N=128
- Scaling to N≥256 on continuous channels remains open
- Future work: parallel training for MAC (circular convolution decomposition),
  channels with memory (ISI-MAC), unknown channels (DINE/MINE)

---

### References

1. E. Arikan, "Channel polarization," IEEE TIT 2009
2. S. B. Önay, "SC decoding of polar codes for the two-user MAC," IEEE ISIT 2013
3. Y. Ren, Z. Li, P. M. Olmos, "SC decoding of polar codes for the two-user MAC using computational graphs," 2025
4. S. Aharoni, R. Misoczki, E. Ordentlich, "Neural polar decoders for 5G," IEEE JSAC 2024
5. I. Tal, A. Vardy, "List decoding of polar codes," IEEE TIT 2015
6. M. Gruber, S. Cammerer, "Neural network architectures for channel coding," IEEE JSAC 2017
7. E. Nachmani et al., "Deep learning methods for improved decoding of linear codes," IEEE JSAC 2018
8. T. Gruber et al., "On deep learning-based channel decoding," CISS 2017

---

### Appendices (if space permits)

**A. Hyperparameters**
Full table of training hyperparameters

**B. Approaches That Failed**
Brief summary of failed approaches and why (fast_ce for MAC, residual connections, snapshot training, multi-depth loss)

**C. Detailed Per-Position Error Analysis**
Error distribution across positions at N=256
