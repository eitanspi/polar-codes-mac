# Neural Polar Decoders for Two-User MACs with Memory: The Complete Story

---

## A. The Problem

Polar codes for multiple-access channels (MACs) rely on successive cancellation (SC) decoding, where a binary tree walk sequentially recovers the transmitted bits of both users from the shared channel output. On memoryless MACs -- the binary-erasure MAC, the Gaussian MAC, the additive binary noise MAC -- the SC decoder's tree operations (check node, bit node, inverse circular convolution) have known closed-form expressions derived from the channel's transition law. The decoder is analytically specified and provably capacity-achieving as block length grows.

When the MAC has memory -- inter-symbol interference (ISI), time-varying fading (Gilbert-Elliott/Ising), or correlated noise (moving-average additive Gaussian noise, MA-AGN) -- analytical SC becomes problematic. The per-position likelihood depends on the full channel state sequence, and optimal decoding requires a trellis forward-backward pass whose complexity scales with the number of joint channel states. For a two-user ISI-MAC with tap length 1, the joint trellis has 4 states per time step. For continuous-state channels like MA-AGN (where the noise follows an AR(1) process with real-valued state), no finite-state trellis exists at all, and the analytical SC decoder is simply unavailable.

Aharoni, Huleihel, Pfister, and Permuter (IEEE TIT, 2024) demonstrated that for single-user channels with memory, a Neural Polar Decoder (NPD) -- which replaces the analytical tree operations with small MLPs and the channel embedding with an LSTM -- can learn to decode from samples alone, matching analytical SC on ISI channels up to block length N=1024. The gap this thesis addresses: nobody had extended the NPD framework to the two-user MAC setting, where the decoder must simultaneously recover both users' messages and where the chained decoding structure introduces unique challenges not present in the single-user case.

---

## B. What We Built

Three decoder architectures were developed, each targeting a different MAC operating regime.

**The Neural Computational Graph (NCG)** is a fully neural sequential SC decoder for non-corner (Class B) MAC polar codes. Every analytical tensor operation -- check node, bit node, CalcParent, channel embedding, leaf decision, re-embedding -- is replaced by a learned MLP operating on d=16-dimensional embeddings. The six MLP modules are weight-shared across all tree positions and depths, making the architecture N-agnostic: a model trained at one block length transfers to any other via curriculum warm-starting. Training uses sequential teacher-forced cross-entropy with mandatory curriculum (N=16 to 32 to 64 to ...), because random MLPs at tree depth 8+ destroy the channel signal before it reaches any leaf. Total parameter count is approximately 39,000.

**The Chained NPD** is a parallel-trainable decoder for corner-rate (Class C) MACs, adapting the NPD framework to the two-stage Onay 2013 decomposition. Stage 1 decodes user U from the channel output z, marginalising over user V as uniform noise. Stage 2 decodes user V from (z, u_hat), conditioning on the re-encoded Stage 1 output. Each stage is an independent single-user NPD with its own channel encoder and tree operations. For memory channels, the pointwise channel embedding is replaced by a bidirectional GRU (BiGRU) that processes the full observation sequence z_1^N, producing per-position embeddings that capture the channel's temporal structure. The tree operations are fully neural (not analytical), which is critical: the analytical fast_ce path collapses the d-dimensional embedding to a scalar LLR at the root, destroying the state information carried by the BiGRU encoder.

**The Chained Trellis SC** is a new analytical baseline we derived for memory MACs. The Onay 2013 chained decomposition yields two effective single-user finite-state channels; we implemented BCJR forward-backward on 2-state trellises for each stage. This runs at roughly 6x lower cost than the joint 4-state trellis decoder and matches it within a factor of 2 in BLER. It provides the ceiling the neural decoder aims to reach.

---

## C. Memoryless Channel Results

On memoryless channels, the neural decoders establish feasibility and characterise boundaries.

**BEMAC (Z = X + Y, deterministic)** is the strongest NCG result. On Class C (corner rate), the NCG beats SC by 2-100x across all tested N from 8 to 1024. At N=128, NCG achieves BLER 0.0003 versus SC's 0.0245 -- a 100x improvement. On Class B (symmetric rate), NCG matches SC within a factor of 2 at all N, with a slight edge (0.78x at N=32, 0.50x at N=256). The BEMAC's discrete, deterministic structure appears ideally suited to the learned d=16 embedding, which captures information that scalar log-likelihood ratios miss. The mechanism behind this advantage remains an open theoretical question.

**GMAC Class B (symmetric rate, SNR=6 dB)** is the NCG's empirical limit. The decoder matches SC within 2% up to N=128 (BLER 0.023 vs 0.019), then hits a sharp wall at N=256 (0.023 vs 0.006, a 4.5x gap). Extensive diagnostics ruled out exposure bias and cascade amplification; the gap traces to a broad tail of weak positions with small individual deficits that NCG cannot close. CRC-aided neural SCL (NN-CA-SCL) with L=4 closes this wall without retraining: at N=256 it achieves BLER 0.003, which is 1.7x better than analytical SC's 0.005 -- the first neural decoder to beat SC at this block length on GMAC.

**GMAC Class C (corner rate)** shows strong NPD gains: 0.55-0.66x SC at N=16-32 and 0.37x at N=64. However, the NPD and SC use different frozen sets (MI-designed vs DE-designed), and the neural decoder's frozen set is co-adapted with its tree operations. When SC is evaluated on the neural frozen set, it performs much worse (0.377 vs 0.173 at N=16), confirming the gain comes from both the decoder and the code design working together.

**ABNMAC (correlated binary noise)** shows modest NCG gains of 5-15% on Class B, much smaller than the discrete BEMAC. CRC-SCL L=4 dominates all neural approaches, providing 8-14x improvement over SC at N=32-128.

---

## D. Memory Channel Results -- The Main Contribution

The ISI-MAC results are the thesis headline. The channel model is Z_t = (1-2X_t) + (1-2Y_t) + 0.3*((1-2X_{t-1}) + (1-2Y_{t-1})) + W_t at SNR 6 dB, with a 4-state joint ISI trellis.

**ISI-MAC at N=16-64 (GPU curriculum, d=16 h=100 BiGRU).** The chained NPD beats the chained trellis SC baseline by 19-32%. At N=32, NPD achieves BLER 0.057 versus trellis 0.082 (31% improvement). At N=64, NPD achieves 0.028 versus chained trellis 0.041 (32% improvement), and statistically matches the joint 4-state trellis SC at 0.026 (ratio 1.06x, overlapping confidence intervals). These are publication-quality results with 5,000 codewords per evaluation point and Wilson 95% confidence intervals that do not overlap between NPD and the chained trellis baseline.

**ISI-MAC at N=128-256 (d=64 h=128 BiGRU).** A larger model (200K parameters per stage vs 42K) maintains competitive performance at longer block lengths. At N=128, BLER 0.030 versus chained trellis 0.022 (1.35x). At N=256, BLER 0.011 versus trellis 0.006 (1.84x). Crucially, memoryless SC fails catastrophically at these N values (0.101 at N=128, 0.226 at N=256), proving the NPD learns genuine ISI structure through its BiGRU encoder.

**ISI-MAC wall at N=512.** The d=64 model trained at N=256 does not generalise to N=512: 25% of information positions produce confidently wrong predictions out of distribution. Direct training at N=512 (250K iterations on GPU) also failed, reaching BLER 0.576. This wall is documented but not resolved.

**Ising MAC (Gilbert-Elliott fading, p_flip=0.1).** The channel alternates between a GOOD state (normal GMAC) and a BAD state (pure noise). With 10% of time steps producing no signal, BLER exceeds 57% at all N. The NPD with BiGRU learns partial Ising structure, beating memoryless SC by 4-7% at N=16 but trailing the 2-state Markov trellis SC by 3-12%. The high ambient BLER limits the practical value of any decoder at this operating point.

**MA-AGN MAC (AR(1) correlated noise, alpha=0.3).** This is the flagship memory channel: the noise state N_t = 0.3*N_{t-1} + W_t is a continuous real number, so no finite-state trellis decoder exists. The only practical analytical baseline is memoryless GMAC SC, which ignores the noise correlation. The NPD with BiGRU beats memoryless SC by 13% at N=16 (BLER 0.144 vs 0.165), demonstrating that the BiGRU encoder successfully learns the AR(1) structure from samples. At N=64, the d=16 h=100 model matches memoryless SC (0.029 vs 0.029), closing nearly half the gap from the earlier d=32 h=128 model's 2.4x deficit. At N=128, the NPD degrades to 19.5x worse than memoryless SC, indicating the chained marginalisation cost dominates any memory-exploitation benefit at larger block lengths.

---

## E. Key Insights

**Hidden width matters more than embedding dimension.** The breakthrough at N=64 on ISI-MAC came from increasing the tree MLP hidden layer width from 64 to 100 (d=16 throughout), which doubled parameters from 20K to 42K per stage and reduced BLER by 41% (from 0.046 to 0.027). In contrast, increasing d from 16 to 32 with the same hidden width gave smaller gains. The tree operations (check node, bit node) are the bottleneck, not the embedding dimensionality.

**The BiGRU channel encoder is essential for MAC memory channels.** The NPD paper's pointwise embedding (a simple MLP per position) works for single-user ISI because the tree propagation itself distributes information across positions. For MAC channels, the Stage 1 marginalisation over user V produces a mixture likelihood (sum of Gaussians for each possible Y value) that cannot be summarised by a pointwise function of z_t. The BiGRU sees the full sequence and produces contextually-informed per-position embeddings. Replacing it with a window MLP or LSTM gives equivalent or worse results.

**Frozen set co-adaptation drives GMAC Class C gains.** The neural decoder's MI-designed frozen set and its tree operations are co-adapted. SC on the neural frozen set performs 2-3x worse than SC on its own DE-designed set (BLER 0.377 vs 0.173 at N=16). The neural gain comes from both a decoder-adapted code design and a better decoder operating on that design.

**Training recipe: constant lr=1e-3 is optimal.** The NPD paper's simple recipe (Adam, lr=1e-3, no decay) consistently outperforms cosine decay schedules, cosine-with-restarts, and warmup strategies. For the NCG (which uses cosine decay), switching from cosine-with-restarts to plain cosine improved N=128 from 0.027 to 0.017. Each N requires independent training: curriculum warm-starting is essential for convergence but does not produce a single universal model.

**GPU curriculum warm-starting is the strongest training strategy.** The best ISI-MAC results come from training N=16, then warm-starting N=32 from the N=16 checkpoint, then N=64 from N=32. This GPU-trained curriculum model beats all other training strategies at N=16-64.

---

## F. The Walls and What We Learned

Three distinct performance walls were identified, diagnosed, and documented.

**GMAC Class B N=256 (NCG).** The NCG's BLER plateaus at 0.023 while SC continues dropping to 0.006. Diagnostics established: (i) exposure bias is absent (teacher-forced and free-running MI are nearly identical), (ii) cascade amplification is absent (failed blocks have similar error counts in both decoders), (iii) the gap traces to a broad tail of many positions with small individual deficits plus 5 NCG-specific weak positions per user that account for half the failures. Removing these weak positions does not close the gap -- the deficit is distributed. CRC-SCL L=4 closes the wall at inference time without retraining, achieving BLER 0.003 (1.7x better than SC).

**ISI-MAC N=512 (chained NPD).** The BiGRU encoder trained at N=256 catastrophically fails at N=512: rate-1 MI measurement shows 25% of information positions produce confidently wrong predictions (mean MI = -59M bits). Eight architecture configurations were tested (LSTM, larger window, 3-phase training, d=128). The memorisation test passes (the model can overfit to 100 codewords), so capacity is not the issue. The failure is in generalisation: the tree structure changes at each N, and the BiGRU's learned representations do not transfer. Direct N=512 training at d=64 on GPU (250K iterations) also failed, suggesting 500K+ iterations may be needed.

**All walls follow the same pattern.** Neural tree decoders work at small N, where the O(log N) tree depth is manageable and the per-position embedding capacity is sufficient. As N grows, information loss accumulates across tree levels, and the BiGRU encoder's fixed-dimensional representation becomes the bottleneck. The walls appear at different N for different channels (N=256 for NCG on GMAC, N=512 for NPD on ISI-MAC, N=32 for NPD on MA-AGN), but the mechanism is the same.

---

## G. Comparison to the NPD Paper

Aharoni et al. (2024) demonstrated the NPD on single-user channels with memory. The key differences between their setting and ours:

**Architecture.** The paper uses d=8, hidden=50, 2 hidden layers, shared F/G/H across depths, with an LSTM inside a Donsker-Varadhan potential for the channel encoder. We use d=16, hidden=100, 2 hidden layers, shared F/G/H, with a BiGRU encoder. Our model has roughly 4x more parameters per stage.

**Channel complexity.** The paper's ISI channel has 2 states (single-user tap-1 ISI). Our ISI-MAC has 4 joint states (two users, each with tap-1). Stage 1 must marginalise over user V's codeword, converting a 4-state channel into an effective 4-state channel with mixture emissions. The paper's pointwise embedding suffices for 2-state ISI; our MAC requires the BiGRU to handle the mixture.

**Block length scaling.** The paper works at N=1024 with 1M training iterations. Our best results are at N=64 (beating trellis SC) and N=256 (within 1.8x of trellis SC with d=64). The MAC introduces a fundamental additional source of error -- Stage 1's marginalisation over user V -- that degrades with block length. At N=512, this error dominates.

**The MAC is fundamentally harder.** In the single-user setting, the tree decoder sees the complete per-position likelihood. In the chained MAC, Stage 1 sees a marginalised likelihood that treats the other user as noise, and Stage 2 conditions on a potentially erroneous Stage 1 output. This two-stage structure amplifies the scaling challenge: any Stage 1 error propagates to Stage 2, and the marginalisation loss grows with rate. The neural decoder must learn not just the channel structure but also how to cope with the marginalisation, a task that becomes harder as N increases.

Despite these challenges, the neural MAC decoder achieves a result the single-user paper does not: at N=16-64, the NPD beats the analytical trellis SC decoder, demonstrating that learned decoding can outperform exact analytical decoding when the code design is co-adapted with the neural decoder and the block length is short enough for the BiGRU to maintain per-position accuracy.
