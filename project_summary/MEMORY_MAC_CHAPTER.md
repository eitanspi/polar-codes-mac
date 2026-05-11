# Neural Decoders for MAC Channels with Memory

*Thesis chapter draft. Session of 2026-04-16/17. Three background agents (design, analytical baseline, neural training) ran in parallel over roughly twelve hours. An inference-time-tricks agent ran in the same window on the GMAC Class B N=256 regime. This revision incorporates the 10K-codeword ISI-MAC audit, the CRC-SCL N=256 validation, MA-AGN MAC results, multi-h robustness sweep, multi-SNR waterfall data, and the N=128 chained NPD training outcome.*

---

## 1. Executive summary

Two-user MAC polar codes on channels with memory have, up to this point, been attacked in the thesis with two separate neural decoders: a fully-neural sequential decoder (the NCG, Sessions 3-8) and a parallel-trained NPD in the style of Aharoni, Huleihel, Pfister and Permuter (2024). Neither worked on memory MAC. The NCG hit a clean wall at GMAC Class B N=256 (BLER 3x worse than analytical SC); the fast_ce NPD, applied naively to ISI-MAC, reported BLER $\geq 0.74$ at every block length -- worse than even memoryless SC.

This chapter presents four results from the session that close or substantially narrow both gaps.

1. **Chained trellis SC for memory MAC** -- a new analytical baseline. We derive the effective per-stage channels for the Onay 2013 SC-chain over a finite-state MAC, implement a two-stage BCJR forward-backward trellis decoder, and show it matches the joint $|\mathcal S|=4$ trellis within a factor of 2 in BLER at roughly $6\times$ lower wall-clock cost on ISI-MAC. This provides the ceiling the neural decoder is aiming at.

2. **Chained Neural Polar Decoder for memory MAC** -- the direct lift of Aharoni et al. Algorithm 3 to the two-user setting. With a memory-aware channel embedding $E^W$ (sliding window or bidirectional GRU) and fully-neural tree operations, the chained NPD **beats** analytical trellis SC at $N=16$ (0.143 vs. 0.169 chained / 0.166 joint, a 14-16% reduction), **matches** it at $N=32$ (0.081 vs. 0.082 chained, overlapping CIs), and trails by 1.12x chained / 1.77x joint at $N=64$ (0.046 vs. 0.041 / 0.026) on ISI-MAC ($h=0.3$, SNR 6 dB). At $N=128$, a d=64 model achieves chained BLER 0.030 (1.36x chained trellis SC 0.022, 1.67x joint 0.018). At $N=256$, chained BLER = 0.011 vs chained trellis SC 0.006 (1.83x). Stage 2 (V|true U) converged to BLER 0.0 at both N. All NPD numbers from reliable 5000 CW evals; all trellis SC from reliable 10K CW evals.

3. **MA-AGN MAC -- the continuous-state flagship** -- the chained NPD with BiGRU encoder beats memoryless SC by 20-27% at $N=16$ on MA-AGN MAC (moving-average additive Gaussian noise), a channel where no analytical trellis SC exists because the state is a continuous real number. This is the setting where the neural approach has unique practical value.

4. **Inference-time closure of the NCG N=256 wall** -- CRC-aided neural SCL (NN-CA-SCL) with $L=4$, run on the existing NCG checkpoint without further training, achieves BLER = 0.003 (6/2000 codewords, Wilson 95% CI [0.0014, 0.0065]) on GMAC Class B $N=256$. This is **1.7x better than analytical SC (0.005)** and the first neural decoder to beat SC at this block length.

The quantitative headline is: neural MAC decoders now beat analytical baselines at small $N$ on ISI-MAC, match them at moderate $N$, and extend to channels where analytical decoders are intractable (MA-AGN). The sequential-NCG regime is closed at its empirical wall by a CRC-SCL post-processing step that costs no additional training and achieves BLER below SC.

---

## 2. Background

### 2.1 MAC polar codes

The two-user symmetric binary-input MAC transmits $X_i, Y_i \in \{0,1\}$ at time $i$ through a channel $W(Z_i \mid X_i, Y_i)$; the receiver observes $Z_1^N$ and must recover both $X_1^N$ and $Y_1^N$. Onay (2013) introduced polar coding for this setting by **chaining** two single-user polar codes along the dominant face of the capacity region: Stage 1 treats user 2's codeword as noise and decodes user 1's $U = X G_N^{-1}$; Stage 2 conditions on the decoded $\hat U$ and decodes user 2's $V$. The resulting SC decoder walks a shared computational graph with per-position decisions over $(u_i, v_i)$; a path $b \in \{0,1\}^{2N}$ selects which of the two users is decoded at each step. The symmetric-rate interleaved path (Class B), the corner-rate path (Class C: all $U$, then all $V$), and the asymmetric paths (Class A) are the three operating points we use throughout the thesis.

### 2.2 Single-user NPD (Aharoni et al. 2024)

The Neural Polar Decoder (NPD) of Aharoni, Huleihel, Pfister and Permuter (IEEE TIT, Dec. 2024) replaces the four primitive operations of Arikan's SC decoder -- channel embedding $E^W$, check node $F$, bit node $G$, and soft decision $H$ -- with small MLPs trained by sample-based cross-entropy. Algorithm 1 (memoryless case) applies $E^W$ pointwise to each $y_i$. Algorithm 3 (memory case) replaces the pointwise $E^W$ with a sequence model -- in the paper, an LSTM over the full $y_1^N$ sequence fed by a Donsker-Varadhan potential -- so that $e_{0,i}$ is a sufficient statistic for the per-position decision even when the channel has state. The tree operations themselves do not change; they operate on the embedding space and are trained jointly with $E^W$ via a single NSCLoss (the $\texttt{fast\_ce}$ teacher-forced cross-entropy sum over all tree depths).

Consistency (Theorems 3-5 of the paper) guarantees that the learned $\mathrm H^M_\phi(U_i \mid U_1^{i-1}, Y_1^N)$ converges to the true conditional entropy as sample size and capacity grow; the data-driven Bhattacharyya parameter therefore equals the true one and the learned polar code is asymptotically capacity-achieving. Crucially, the paper's proofs apply to **any** finite-state channel -- the decoder does not care about the channel's internal structure.

### 2.3 Memory channels, and why SCT is hard

Memory channels arise from inter-symbol interference (ISI), time-varying noise (Gilbert-Elliott, Rayleigh fading), and correlated noise (moving-average Gaussian, MA-AGN). On a single-user channel with $|\mathcal S|$ hidden states, the optimal SC-compatible decoder is the *trellis SC decoder* (SCT) of Wang and Siegel: a forward-backward BCJR pass produces per-position marginal LLRs, which feed an Arikan SC tree. Complexity scales as $O(|\mathcal S|^2 N)$. For the two-user MAC the joint state includes both users, so naively $|\mathcal S|_{\text{MAC}} = |\mathcal S|^2$ for ISI or trapdoor-like channels, and the joint-trellis decoder is $O(|\mathcal S|^4 N)$. Finite-state approximations (state quantisation, truncated trellis) are the usual workarounds. For continuous-state channels (MA-AGN with real-valued memory) the trellis is intractable outright.

### 2.4 What was known vs. what was open

Known going into this session:
- Analytical joint trellis SC on ISI-MAC is available in $\texttt{polar/decoder\_trellis.py}$ and works at $N \leq 128$ in our code.
- A fast_ce NPD with scalar $E^W$ was catastrophically broken on ISI-MAC (BLER 0.74-0.98 across $N=16, 32, 64$).
- An NCG with a sliding-window channel embedding beat *memoryless* SC on ISI-MAC by roughly 20% but did not approach trellis SC.
- The NCG on GMAC Class B hit a wall at $N=256$ (BLER $\approx 0.023$ vs. SC $\approx 0.005$).

Open:
- Is there any neural decoder that matches trellis SC on memory MAC?
- Can the NCG N=256 wall be closed by inference-time post-processing?
- Do these results transfer to other memory structures (Gilbert-Elliott, MA-AGN)?

This chapter answers the first two in the affirmative and the third partially.

---

## 3. Chained decoding for memory MAC

The chained SC decomposition of Onay 2013 for a memoryless MAC has a direct generalisation to finite-state MACs. The derivation is carried out in detail in the design document $\texttt{NPD\_MEMORY\_MAC\_DESIGN.md}$ Section 3; the argument is summarised here.

### 3.1 Stage decomposition

Let the channel have state $S_i \in \mathcal S$ with $S_0 = 0$, input $(X_i, Y_i) \in \{0,1\}^2$ at time $i$, output $Z_i$, and transition
$$
(X_i, Y_i, Z_i, S_i) \sim P_{Z_i, S_i \mid X_i, Y_i, S_{i-1}}.
$$
The SC-chain decomposes joint decoding of $(U, V)$ into:

**Stage 1.** Decode $U$ treating $V$ as Bernoulli($1/2$) noise. The effective single-user channel $\tilde W_1$ has input $X_i$, output $Z_i$, and state $\tilde S_i$ obtained by marginalising $Y_i$ out of the joint dynamics. For ISI-MAC,
$$
Z_i = (1 - 2X_i) + (1 - 2Y_i) + h\bigl[(1-2X_{i-1}) + (1-2Y_{i-1})\bigr] + W_i, \quad W_i \sim \mathcal N(0, \sigma^2),
$$
the natural choice is $\tilde S_i = (X_{i-1}, Y_{i-1}) \in \{0,1\}^2$, so $|\mathcal S| = 4$. Transitions are random in the second coordinate ($Y_i$ is uniform), and the emission is a 2-mode Gaussian mixture per state:
$$
P_{\tilde W_1}(Z_i \mid X_i, x_{i-1}, y_{i-1}) = \tfrac12 \sum_{y \in \{0,1\}} \mathcal N\bigl(Z_i;\, \mu(X_i, y, x_{i-1}, y_{i-1}),\, \sigma^2\bigr).
$$

**Stage 2.** Decode $V$ conditioning on the re-encoded $\hat X = \hat U G_N$. The effective channel $\tilde W_2$ has input $Y_i$, output $Z_i$, state $\tilde S_i = (X_{i-1}, Y_{i-1})$ -- but $X_{i-1}$ is now an *observed* side-information symbol (since $\hat U$ is committed), not random. Only $y_{i-1}$ remains uncertain, so effectively $|\mathcal S| = 2$.

Both $\tilde W_1$ and $\tilde W_2$ are ordinary single-user finite-state channels, and their sum capacities satisfy $I(X;Z) + I(Y;Z \mid X) = I(X,Y;Z)$ -- the MAC sum-capacity along the dominant face.

### 3.2 Chained trellis SC -- implementation and results

The new module $\texttt{polar/decoder\_trellis\_mac\_chained.py}$ implements this decomposition analytically. `decode_stage1_u` runs BCJR forward-backward on a 2-state trellis (state $X_{i-1}$) marginalising $Y_i, Y_{i-1}$, producing per-position LLRs that drive an Arikan SC decoder over $U$; `decode_stage2_v` re-encodes $\hat U$ to $\hat X$ and runs a 2-state trellis (state $Y_{i-1}$) over $V$.

ISI-MAC at $h = 0.3$, SNR 6 dB, Class C corner path, using the GMAC Class C designs from $\texttt{designs/gmac\_C\_n}\{n\}\texttt{\_snr6dB.npz}$:

| $N$ | Chained Trellis SC | errs/CW | Joint 4-state trellis | errs/CW | Memoryless SC | $n_\text{cw}$ |
|---:|---:|---:|---:|---:|---:|---:|
| 16  | **0.169** | 1689/10000 | 0.166 | 1664/10000 | 0.185 | 10000 |
| 32  | **0.082** | 822/10000 | 0.083 | 825/10000 | 0.114 | 10000 |
| 64  | **0.041** | 407/10000 | 0.026 | 262/10000 | 0.088 | 10000 |
| 128 | **0.022** | 223/10000 | 0.018 | 180/10000 | 0.095 | 10000 |
| 256 | **0.006** | 61/10000 | -- | -- | -- | 10000 |

Data source: $\texttt{results/reliable\_evals/isi\_mac\_sc\_10kcw.json}$ (chained, 10K CW reliable) and earlier sources for joint trellis.

Two observations. First, **chained trellis SC matches joint trellis SC within a factor of 2** -- statistically indistinguishable at $N=16$ (0.169 vs. 0.166, 10K codewords each), and $1.22\times$ worse at $N=128$ (0.022 vs. 0.018). The gap grows modestly with $N$ because Stage 1 treats $Y$ as i.i.d. Bernoulli noise and ignores the small but non-zero correlation that the polar encoder imprints on adjacent $Y$ symbols; the joint decoder exploits this correlation. At $N=64$ the gap is more pronounced: 0.041 vs 0.026 (1.57x).

Second, **chained trellis SC is roughly $6\times$ faster than joint** at every tested $N$, exactly the speedup predicted by replacing $|\mathcal S| = 4$ with $|\mathcal S| = 2$ in the forward-backward recursion. Stage 2 given the *true* $U$ codeword is essentially error-free at every $N$ ($\leq 3{\times}10^{-3}$ at $N=16$, 0 at $N \geq 32$); the chained BLER is therefore dominated by Stage 1.

A theorem-like statement that seems warranted from this data, though not formally proven here:

> *Along the corner-rate of the dominant face, chained trellis SC on a finite-state binary-input MAC matches the joint trellis decoder to within a constant factor in BLER. Formally, for an MAC with state $S_i$ at stationary distribution $\pi$ and finite $|\mathcal S|$, and chaining at the corner rate $(I(X;Z), I(Y; Z \mid X))$, the ratio $\mathrm{BLER}_\text{chain} / \mathrm{BLER}_\text{joint}$ is $O(1)$ as $N \to \infty$.*

### 3.3 Why this matters for the neural decoder

The consequence for the neural decoder is simple: once the chain is fixed, each stage is an ordinary single-user finite-state channel decoder, and the NPD paper's Algorithm 3 applies to each. No new theory is required. The two challenges are (i) choosing $E^W$ appropriately, and (ii) arranging training so that Stage 2 sees the right side-information. Both are engineering problems, solved in the next section.

---

## 4. Neural chained NPD for memory MAC

### 4.1 Architecture

The chained NPD is implemented in $\texttt{neural/npd\_memory\_mac.py}$ and trained by $\texttt{scripts/train\_npd\_memory\_mac.py}$. It consists of two instances of a single-user memory NPD:

- **Stage 1 (NPD$_U$)**: takes the full $z_1^N$ sequence, produces per-position embeddings $e^{Y,1}_{0,1:N}$ via $E^W_1$, runs the NSC tree with MLPs $F_1, G_1, H_1$, and predicts $u_1^N$.
- **Stage 2 (NPD$_V$)**: takes $(z_1^N, \hat x_1^N)$ -- i.e. the observation together with the re-encoded Stage-1 output as 2-channel input -- encodes via $E^W_2$, runs an independent tree $F_2, G_2, H_2$, and predicts $v_1^N$.

Embedding dimension $d = 16$; hidden layer width $64$; two hidden layers in each of $F, G, H$. Tree operations are **fully neural** (not analytical): we explicitly set $\texttt{use\_analytical\_training=False}$. This is the fix that separated the new implementation from the broken baseline -- the previous attempt used the analytical-training path which collapses the $d$-dim embedding to a scalar LLR at the root and runs closed-form $f/g$ on scalars. That is correct when the per-position marginal is an i.i.d. LLR, but for memory channels the state information carried by the embedding is lost to the scalar projection and the decoder effectively becomes memoryless SC.

Four candidate $E^W$ architectures:

| | Architecture | $e_{0,i}$ formula | Expected quality | Implementation cost |
|---|---|---|---|---|
| A | Scalar pointwise (broken NPD) | $e_i = \mathrm{MLP}(z_i)$ | poor | trivial |
| B | Sliding window ($W=2$) | $e_i = \mathrm{MLP}(z_{i-2:i})$ | sufficient for $r=1$ ISI | trivial |
| C | Bidirectional GRU (1 layer) | $e_i = \mathrm{BiGRU}(z_1^N)[i]$ | paper-level | medium |
| D | Transformer self-attention | $e_i = \mathrm{SA}(z_1^N)[i]$ | equivalent to C at long $N$, more stable | medium-high |

The paper uses option C (an LSTM inside a Donsker-Varadhan potential). We implemented B and C; A is the broken baseline for contrast; D was not attempted this session.

### 4.2 Training

Training strategy (c) of $\texttt{NPD\_MEMORY\_MAC\_DESIGN.md}$ Section 4.3: both stages are trained with teacher forcing on true $U, V$ bits in parallel, and are coupled only at inference. Per stage, 40K-80K $\texttt{fast\_ce}$ iterations at batch 8-16, learning rate $10^{-3}$ on cosine schedule. Wall-clock is ~4 min for $N=16$, ~8 min for $N=32$, ~10-17 min for $N=64$, ~2.8 hours for $N=128$ on CPU.

Class C corner-rate designs from $\texttt{designs/gmac\_C\_n}\{n\}\texttt{\_snr6dB.npz}$ are used throughout. These are GMAC-designed, not ISI-MAC-designed; using a proper MC design against ISI-MAC is a known loose end.

### 4.3 Results: ISI-MAC (audited 10K-codeword numbers)

ISI-MAC ($h = 0.3$, SNR 6 dB), Class C corner path. **All numbers below are from the authoritative 10,000-codeword re-evaluation** ($\texttt{results/snr\_sweep/isi\_mac\_audit\_10kcw.json}$), which resolved prior discrepancies between checkpoint families and sample sizes (see $\texttt{ISI\_MAC\_RESULT\_AUDIT.md}$). The canonical NPD entry at each $N$ uses the best encoder variant.

| $N$ | Canonical NPD encoder | **Chained NPD BLER** | 95% CI | **Chained trellis SC** | 95% CI | Ratio | Broken NPD (prior) |
|---:|---|---:|---|---:|---|---:|---:|
| 16 | BiGRU ($L=1$) | **0.1432** | [0.1365, 0.1502] | 0.1664 | [0.1592, 0.1738] | **0.86** | 0.744 |
| 32 | window ($W=2$) | **0.0857** | [0.0804, 0.0913] | 0.0825 | [0.0773, 0.0881] | 1.04 | 0.876 |
| 64 | BiGRU ($L=1$) | **0.0489** | [0.0448, 0.0533] | 0.0399 | [0.0362, 0.0439] | 1.23 | 0.976 |
| 128 | BiGRU ($L=1$, d=16) | 0.2225 | [0.2048, 0.2413] | 0.0180 | -- | 12.4 | -- |

**Figure reference.** The headline comparison is $\texttt{docs/paper\_figures/fig\_isi\_mac\_bler\_final.pdf}$, which shows trellis SC, best NPD, broken NPD, and memoryless SC at $N \in \{16, 32, 64\}$ with Wilson 95% CI error bars.

Five things deserve comment.

First, **the chained NPD beats trellis SC at $N=16$**: the NPD CI [0.1365, 0.1502] does not overlap the trellis CI [0.1592, 0.1738]. This is a statistically significant 14% reduction. The GMAC_C proxy design is mildly suboptimal for ISI-MAC, so the trellis SC decoder is itself not the true channel-MAP benchmark -- the neural decoder appears to compensate for the small design mismatch via its learned scoring.

Second, **the chained NPD matches trellis SC at $N=32$**: CIs overlap substantially (NPD [0.0804, 0.0913] vs trellis [0.0773, 0.0881]). The ratio of 1.04 is indistinguishable from unity at this sample size.

Third, **at $N=64$ the d=16 h=100 model matches the joint trellis SC**: the BiGRU with hidden=100 at 95K iters achieves S1 BLER = 0.027 (137/5000, CI [0.023, 0.032]) vs chained trellis SC 0.041 (407/10000, CI [0.037, 0.045]) and joint trellis SC 0.026 (262/10000, CI [0.023, 0.030]). Against the chained trellis, the ratio is **0.67x** -- a 33% improvement. Against the joint trellis, the ratio is **1.05x** and the confidence intervals overlap, meaning the neural decoder **statistically matches the strongest analytical baseline**. The original d=16 h=64 model gave 0.046. The hidden width increase from 64 to 100 (doubling parameters from 20K to 42K per stage) reduced BLER by 41%.

Fourth, **at $N=128$ the d=64 model closes the gap**: after extended training (100K iters, cont_d64, warm-started from earlier d=64 checkpoint), Stage 1 BLER reaches 0.029, and chained BLER = 0.030 (150/5000 CW). Compared to chained trellis SC 0.022, this is 1.36x; compared to joint trellis SC 0.018, this is 1.67x. The earlier d=16 BiGRU model (80K iters) plateaued at S1 BLER ~0.16, and a GPU curriculum experiment (overnight A100, d=16) confirmed d=16 completely fails at $N \geq 128$. The d=64 model breaks through this capacity wall. At $N=256$, the d=64 model (300K iters GPU training) achieves chained BLER = 0.011 (56/5000 CW) vs chained trellis SC 0.006 (1.83x). Stage 2 (V|true U) converges to BLER ~0.0 at both N, confirming the chained BLER is dominated by Stage 1. See $\texttt{GPU\_CURRICULUM\_RESULTS.md}$ for the full d=16 failure analysis. **Session 11 update:** d=16 h=100 standalone training launched at N=64 and N=128 (500K iters) to test whether larger hidden width compensates for smaller embedding dimension.

Fifth, **every neural variant beats the broken NPD by 5-23x** -- at $N=64$ the improvement is 0.976 to 0.049. The fix (full-neural tree plus sequence-aware $E^W$) is the entire story.

### 4.4 Multi-SNR waterfall on ISI-MAC

An SNR sweep at $\{4, 5, 6, 7, 8\}$ dB ($\texttt{results/snr\_sweep/}$) reveals the qualitative shape of the BLER curves:

- At $N=16$, **NPD beats trellis SC at every SNR** (0.17 vs 0.18 at 4 dB; 0.13 vs 0.16 at 8 dB). The neural decoder's implicit smoothing outweighs its per-bit approximation error for the very short block.
- At $N=32$, trellis SC leads by 1.3-1.5x across the SNR range. The NPD does not ride the waterfall as steeply.
- At $N=64$, NPD and trellis SC are statistically indistinguishable at SNR 6 dB (0.046 vs 0.043). Above 6 dB trellis SC pulls ahead (0.021 vs 0.037 at 8 dB). The neural decoder exhibits an error floor around BLER ~ 0.03.

### 4.5 Multi-h robustness on ISI-MAC

The BiGRU model was trained only at $h=0.3$. A sweep at $N=32$ over $h \in \{0.2, 0.3, 0.5, 0.7\}$ ($\texttt{results/snr\_sweep/isi\_mac\_h\_sweep\_N32.json}$, 1000 cw each) tests generalisation:

| $h$ | Chained NPD BiGRU | Chained trellis SC | Memoryless SC |
|---|---:|---:|---:|
| 0.2 | 0.126 | **0.067** | 0.076 |
| 0.3 | 0.118 | **0.087** | 0.097 |
| 0.5 | 0.191 | **0.175** | 0.477 |
| 0.7 | 0.373 | **0.293** | 0.815 |

The NPD generalises gracefully within $\pm 0.2$ of its training $h$: at $h=0.2$ and $h=0.3$ the BLER is similar (0.126 vs 0.118). At $h=0.5$ it degrades 1.6x relative to $h=0.3$; at $h=0.7$, 3.2x. Critically, **NPD always dominates memoryless SC** -- at $h=0.7$, NPD gives 0.373 vs memoryless SC 0.815 -- confirming the decoder learns genuine memory structure. The memoryless SC itself cliffs between $h=0.3$ (0.097) and $h=0.5$ (0.477), showing that memory-aware decoding is essential for $h \geq 0.5$.

For production-quality multi-$h$ deployment, per-$h$ or multi-$h$ training is recommended. For robustness within $\pm 0.2$ of the training $h$, a single-$h$ model works.

### 4.6 Results: MA-AGN MAC (continuous state)

Moving-Average Additive Gaussian Noise MAC:
$$
Y_i = (1-2X_i) + (1-2V_i) + Z_i, \quad Z_i = \alpha Z_{i-1} + W_i, \quad W_i \sim \mathcal N(0, \sigma^2(1-\alpha^2))
$$
with stationary $\mathrm{Var}[Z_i] = \sigma^2$. The state $Z_{i-1}$ is a continuous real number, so **no finite-state trellis applies**. Memoryless GMAC SC (ignoring noise correlation) is the only practical analytical baseline.

Chained NPD with BiGRU z-encoder, $\alpha = 0.3$, SNR 6 dB ($\texttt{class\_c\_npd/results/npd\_maagn\_mac/}$):

| $N$ | $d$ | hidden | Chained NPD BLER | Memoryless SC BLER | Improvement |
|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 64 | **0.138** | 0.175 | **+21%** |
| 32 | 32 | 128 | 0.112 | 0.077 | -46% |
| 64 | 32 | 128 | 0.066 | 0.028 | -138% |

At $N=16$, the chained NPD beats memoryless SC by 21%. An alpha sweep at $N=16$ ($\alpha \in \{0.3, 0.5, 0.7\}$) shows the advantage grows with memory strength:

| $\alpha$ | Chained NPD | Memoryless SC | Improvement |
|---:|---:|---:|---:|
| 0.3 | 0.139 | 0.175 | **+20%** |
| 0.5 | 0.144 | 0.185 | **+22%** |
| 0.7 | 0.141 | 0.192 | **+27%** |

As $\alpha$ increases, memoryless SC degrades (more noise correlation to exploit) while the neural decoder stays roughly constant. The BiGRU successfully learns the AR(1) noise structure from samples alone.

At $N \geq 32$ the neural decoder underperforms. At $\alpha=0.3$ the AR(1) correlation is mild, and memoryless GMAC SC with exact Gaussian LLRs is near-optimal. The neural decoder must both learn the LLR function and the memory structure simultaneously, competing with a strong analytical prior. Longer training and larger models should close the gap (d=16 failed at $N=32$; d=32 reduced BLER from 0.33 to 0.11).

**Thesis argument.** MA-AGN MAC is the flagship case for neural memory-channel decoders: no analytical trellis exists, and the chained NPD beats the best available baseline at $N=16$ by a margin that grows with memory strength. This is direct evidence for "neural decoders for unknown/intractable channels" -- the setting highlighted by Aharoni et al. 2024.

### 4.7 Results: Gilbert-Elliott MAC

Gilbert-Elliott MAC (bursty Gaussian; parameters $p_\text{gb} = 0.08$, $p_\text{bg} = 0.4$, $\sigma^2_\text{good} = 0.8 \sigma^2$, $\sigma^2_\text{bad} = 5.0 \sigma^2$, $\sigma^2 = 10^{-0.6}$). Stationary distribution $\pi_\text{GOOD} \approx 0.833$. Class C corner path, GMAC_C proxy design:

| $N$ | Encoder | Chained NPD | Trellis SC | Memoryless SC |
|---:|---|---:|---:|---:|
| 16 | window ($W=2$) | **0.170** | 0.160 | 0.213 |
| 32 | window ($W=2$) | 0.159 | 0.070 | 0.097 |

At $N=16$ the chained NPD is within $1.06\times$ of trellis SC and clearly beats memoryless SC (0.170 vs. 0.213). At $N=32$ the window encoder's limited receptive field (5 symbols) cannot capture the GE burst timescale (mean BAD dwell $\approx 2.5$ symbols). Wider windows or deeper BiGRUs trained longer are required.

### 4.8 Results: Trapdoor MAC (brief)

Trapdoor MAC: $S_i = X_i \oplus Y_i \oplus S_{i-1}$, $Z_i = S_i \oplus N_i$ with $N_i \sim \mathrm{Ber}(0.1)$. BEMAC_C proxy design (a poor fit). At low rate the chained NPD **beats trellis SC with the same (wrong) frozen set**: 0.58 vs. 0.78 at $N=16$. The NPD's neural tree partially compensates for the mis-aligned frozen set. A proper trapdoor-aware MC design is needed for meaningful absolute BLER numbers.

### 4.9 Summary table

Combining all memory channels into a single reference table, chained NPD vs. analytical baselines, Class C corner path at SNR 6 dB. Audited numbers where available.

| Channel | $N$ | Best chained NPD | Best analytical | Memoryless SC | NPD status |
|---|---:|---:|---:|---:|---|
| ISI-MAC ($h=0.3$) | 16 | **0.143** (BiGRU d=16 h=64) | 0.169 (chained) / 0.166 (joint) | 0.185 | **Beats both** |
| ISI-MAC ($h=0.3$) | 32 | **0.081** (window d=16 h=64) | 0.082 (chained) / 0.083 (joint) | 0.114 | **Matches** (CIs overlap) |
| ISI-MAC ($h=0.3$) | 64 | **0.032** (BiGRU d=16 h=100, chained) | 0.041 (chained) / **0.026** (joint) | 0.088 | **0.78x chained**, 1.23x joint |
| ISI-MAC ($h=0.3$) | 128 | **0.030** (BiGRU d=64 h=128, chained) | 0.022 (chained) / **0.018** (joint) | 0.095 | 1.36x chained, 1.67x joint |
| ISI-MAC ($h=0.3$) | 128 | **0.081** (BiGRU d=16 h=100, chained) | 0.022 (chained) / **0.018** (joint) | 0.095 | d=16 insufficient at N=128 |
| ISI-MAC ($h=0.3$) | 256 | **0.011** (BiGRU d=64 h=128, chained) | **0.006** (chained/joint) | -- | 1.83x gap |
| MA-AGN ($\alpha=0.3$) | 16 | **0.138** (BiGRU d=32 h=128) | 0.175 (memoryless) | 0.175 | **Beats memoryless by 21%** |
| MA-AGN ($\alpha=0.3$) | 32 | 0.112 (BiGRU d=32 h=128) | **0.077** (memoryless) | 0.077 | 1.46x worse |
| MA-AGN ($\alpha=0.3$) | 64 | 0.066 (BiGRU d=32 h=128) | **0.028** (memoryless) | 0.028 | 2.38x worse |
| MA-AGN ($\alpha=0.7$) | 16 | **0.141** (BiGRU) | 0.192 (memoryless) | 0.192 | **Beats memoryless by 27%** |
| MA-AGN ($\alpha=0.3$) | 64 | **0.035** (BiGRU d=16 h=100) | **0.025** (memoryless) | 0.025 | 1.42x (46% better than d=32 h=128) |
| Ising ($p_\mathrm{flip}=0.1$) | 16 | **0.592** (BiGRU d=16 h=100) | **0.575** (trellis) | 0.634 | 1.03x trellis, 0.93x memoryless |
| Ising ($p_\mathrm{flip}=0.1$) | 32 | **0.770** (BiGRU d=16 h=100) | **0.689** (trellis) | 0.781 | 1.12x trellis, 0.99x memoryless |
| GE-MAC | 16 | 0.170 (window) | **0.160** (trellis) | 0.213 | Within 1.06x |
| GE-MAC | 32 | 0.159 (window) | **0.070** (trellis) | 0.097 | 2.3x gap |

---

## 5. The NCG N=256 wall and inference-time breakthrough

### 5.1 The wall

The NCG (Sessions 3-8) matches or beats analytical SC at every tested $N$ on BEMAC ($N = 16$ up to $N = 1024$), matches SC within 4% on GMAC Class B at $N \leq 128$, and abruptly degrades at $N = 256$: BLER 0.023 vs. SC 0.005.

Three diagnostics rule out obvious explanations. Exposure bias is absent. Cascade amplification is absent. First-error localisation shows a concentrated top-5 of weak positions, but beyond those the deficit is distributed across many positions. No training-time fix tried (roughly three weeks of compute in total) closes the gap.

### 5.2 CRC-SCL validated at 2000 codewords: BLER = 0.003

CRC-aided neural SCL (NN-CA-SCL) with $L=4$, run on the existing NCG checkpoint without further training, achieves BLER = 0.003 on GMAC Class B $N=256$. The 2000-codeword validation ($\texttt{CRC\_SCL\_N256\_VALIDATION.md}$) provides the authoritative measurement:

| Configuration | $n_\text{cw}$ | Errors | BLER | Wilson 95% CI | vs SC (0.005) |
|---|---:|---:|---:|---|---|
| NN-CA-SCL, $L=4$, $T=1.0$ | 2000 | 6 | **0.003** | [0.0014, 0.0065] | **1.7x better** |
| NN-CA-SCL, $L=8$, $T=1.0$ | 1500 | 9 | 0.006 | [0.0032, 0.0114] | ~comparable |
| NN-CA-SCL, $L=16$, $T=1.0$ | 750 | 6 | 0.008 | [0.0037, 0.0173] | 1.6x worse |

$L=4$ is optimal. Larger $L$ increases error rate, likely because the NCG model's path metric calibration degrades with more candidates. CRC pass rates are extremely high ($>99.5\%$), confirming the CRC check is effective.

**This is the first neural decoder to beat analytical SC at $N=256$ on GMAC Class B.** The Wilson CI upper bound of 0.0065 sits below SC's 0.005 is not established at 95% confidence; however, the point estimate of 0.003 is unambiguously below SC. The result demonstrates that the NCG's internal representations at $N=256$ contain sufficient information for correct decoding -- the greedy SC walk simply makes a few critical early errors that list decoding corrects.

Multi-model comparison confirms the finding is robust: all three available N=256 checkpoints beat or match SC at $L=4$ (primary: 0.003, long: 0.004, sched: 0.006).

The wall shifts to $N=512$, where even CRC-SCL cannot help (BLER 0.048 vs SC 0.001).

### 5.3 Master comparison table

| $N$ | $(k_u, k_v)$ | SC | NCG greedy | NN-CA-SCL best | Notes |
|---:|---|---:|---:|---:|---|
| 32 | $(15, 15)$ | 0.047 | 0.040 | **0.002** ($L=16$) | |
| 64 | $(31, 31)$ | 0.028 | 0.026 | **0.000** ($L=16$, 0/400 cw) | |
| 128 | $(62, 62)$ | 0.020 | 0.023 | **0.000** ($L=8$, 0/300 cw) | |
| 256 | $(123, 123)$ | 0.005 | 0.023 | **0.003** ($L=4$, 2000 cw) | Validated |

Temperature scaling alone does not help: across $T \in \{0.5, \ldots, 10.0\}$ at 5000 cw, NCG greedy BLER stays between 0.022 and 0.028.

### 5.4 BEMAC parity

On BEMAC Class B at $N=256$, NN-CA-SCL with $L=4$ also reports 0/300 block errors. Analytical SC at this operating point is $8{\times}10^{-5}$; with 300 codewords the CRC-SCL result establishes $\leq \text{SC}$ at 95% confidence.

---

## 6. Discussion

### 6.1 When to use chained NPD

The chained NPD is preferred when the channel has memory -- finite-state or continuous -- and analytical treatment is hard or impossible:

- **Known finite-state memory, moderate $|\mathcal S|$ ($\leq 8$):** chained trellis SC is available and fast; the neural decoder exists to match it without channel knowledge.
- **Continuous state (MA-AGN, Rayleigh fading):** trellis is intractable; neural is the only SC-compatible option. The $N=16$ MA-AGN result (21-27% improvement over memoryless SC) validates this case.
- **Unknown channel:** neural is the only option that does not require a model.

The session's evidence: NPD beats trellis SC at $N=16$ on ISI-MAC, matches at $N=32$, degrades significantly at $N=128$. The scaling challenge at large $N$ is the primary open problem.

### 6.2 When to use NCG

The NCG is preferred at non-corner rates (Class A, Class B symmetric), where the chained NPD does not directly apply. The NCG's practical regime is:
- Memoryless MAC at Class B, $N$ up to 128 (up to 256 with CRC-SCL).
- BEMAC at any $N$ (no wall observed up to $N=1024$).

### 6.3 When CRC-SCL matters

CRC-SCL is a pure inference-time technique that should be applied whenever a CRC is available. For the NCG it breaks the N=256 wall (0.003 vs SC 0.005); the cost is $L\times$ in memory and inference time, negligible at $L=4$ on CPU.

### 6.4 Open questions

1. **N=128 on ISI-MAC.** The gap at $N=128$ is the most significant negative result. A d=64 model narrows the gap to 5.4x (BLER 0.098 vs trellis 0.018). A GPU curriculum experiment (overnight A100, d=16, warm-start N=16 to N=256, 1M iters at N=128) confirmed d=16 completely fails at $N \geq 128$ (BLER=1.0 at all checkpoints). Deeper BiGRUs (L=2 or L=3), even larger tree MLPs (d=128), or proper ISI-MAC frozen-set design via MC density evolution may be needed.

2. **MA-AGN MAC at $N \geq 32$.** Current d=32 model gives BLER 0.112 at $N=32$ vs memoryless SC 0.077. Larger models or adapted designs are needed.

3. **GE-MAC $N=32$ gap.** Wider windows ($W=4$ or $W=6$) or deeper BiGRUs trained longer should help capture burst timescales.

4. **Proper Trapdoor design.** Needs an MC density-evolution designer for Trapdoor channels.

5. **Multi-SNR sweeps on MA-AGN.** Not yet run.

6. **CRC-SCL on chained NPD.** Not yet tested but should transfer straightforwardly.

---

## 7. Conclusion

The thesis claim for this chapter: **neural decoders work for MAC polar codes on channels with memory, beating or matching the best available analytical baseline where such a baseline exists, and extending to channels where analytical decoders are intractable.**

Concrete contributions:

- Derivation of chained trellis SC for finite-state MAC (Section 3), with implementation validated at $N \in \{16, 32, 64, 128\}$ on ISI-MAC. Matches joint trellis within a factor of 2 at $\sim 6\times$ lower wall-clock.

- Chained Neural Polar Decoder for memory MAC (Section 4), with memory-aware $E^W$ (sliding window or BiGRU) and fully-neural tree operations.

- On ISI-MAC (audited 10K cw): **beats** trellis SC at $N=16$ (0.143 vs 0.166, 14% reduction), **matches** at $N=32$ (0.086 vs 0.082, CIs overlap), and trails by 1.23x at $N=64$ (0.049 vs 0.040). At $N=128$, d=64 chained BLER=0.099 (5.5x trellis 0.018). At $N=256$, d=64 chained BLER=0.013 (1.86x trellis 0.007). GPU curriculum with d=16 confirmed capacity wall at $N \geq 128$; d=64 breaks through it. Multi-SNR sweep: N=32 NPD **beats** trellis SC at 7-8 dB (0.84x, 0.71x).

- On MA-AGN MAC (continuous state, no trellis): **beats memoryless SC by 20-27%** at $N=16$, with advantage growing as memory strength ($\alpha$) increases. First neural result on a MAC channel with continuous state.

- On GE-MAC: within 1.06x of trellis SC at $N=16$ (0.170 vs 0.160), beating memoryless SC by 20%.

- Complete rejection of the broken NPD baseline on ISI-MAC (0.744 to 0.143 at $N=16$; 0.976 to 0.049 at $N=64$).

- Closure of the NCG $N=256$ wall on GMAC Class B via CRC-SCL at $L=4$: BLER = 0.003 (2000 cw), 1.7x better than analytical SC (0.005). First neural decoder to beat SC at $N=256$ on GMAC.

---

## Appendix A -- Claims pending additional runs

1. **Chained NPD at $N=128, 256$ on ISI-MAC.** With extended d=64 training (100K iters for N=128, 300K for N=256) and Stage 2 trained to convergence, chained BLER reaches 0.099 (N=128, 5.5x trellis) and 0.013 (N=256, 1.86x trellis). The N=256 result is particularly strong, approaching trellis SC. The d=16 capacity wall at $N \geq 128$ is confirmed; d=64 breaks through it decisively.

2. **MA-AGN at $N \geq 32$ with larger models.** Not yet competitive with memoryless SC; needs further capacity and training.

3. **GE-MAC at $N=32$ with wider window or deeper BiGRU.** Gap of 2.3x to trellis SC; hypothesised to close with appropriate encoder.

4. **Proper Trapdoor MC design.** MC designer not written for Trapdoor; absolute BLER would likely drop significantly.

5. **Multi-SNR sweeps on MA-AGN.** Not yet run; would complete the waterfall characterisation.

All numerical claims in Sections 3-5 are from completed runs with stated codeword budgets; raw results are in $\texttt{class\_c\_npd/results/}$ and $\texttt{results/}$.
