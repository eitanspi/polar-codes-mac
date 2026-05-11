# Extending the Neural Polar Decoder (NPD) to the Two-User MAC with Memory

Design document. Intended audience: someone picking up this work cold.

Scope: take the Aharoni / Huleihel / Pfister / Permuter NPD paper
(IEEE TIT, Dec 2024 — `papers/NPD_ziv_bashar.pdf`), whose memory-channel
algorithm is specified for a single user, and lift it to the two-user
binary-input MAC via **SC chaining** (Onay 2013 style): Stage 1 decodes $U$
treating $V$ as noise; Stage 2 decodes $V$ given the decoded $\hat U$. The
chained structure means each stage is an ordinary single-user memory-channel
NPD, which the paper already handles.

Reference code lives in `/Users/ytnspybq/PycharmProjects/NPDforCourse/` (their
TensorFlow implementation) and in `class_c_npd/`, `neural/`, and `polar/` in
this repo.

---

## 1. Paper summary (what Algorithm 1 and Algorithm 3 do)

The paper builds a data-driven polar decoder in three escalating settings:
memoryless symmetric, memoryless asymmetric (Honda–Yamamoto), and channels
*with memory*. The decoder is purely sample-based: it never uses an analytic
$W(y \mid x)$.

### Definition 2 — the "Neural SC" (NSC) decoder

The SC decoder from Arıkan has four elementary operations that are replaced
by neural networks with compact parameter space $\Phi$:

- $E^W_\phi : \mathcal Y \to \mathcal E \subset \mathbb R^p$ — **channel
  embedding**. Maps each channel output $y_i$ (or segment of them) to a
  $p$-dimensional vector.
- $F_\theta : \mathcal E \times \mathcal E \to \mathcal E$ — the
  **check-node** (formerly the exact $2\tanh^{-1}(\tanh(L_1/2)\tanh(L_2/2))$).
- $G_\theta : \mathcal E \times \mathcal E \times \mathcal X \to \mathcal E$
  — the **bit-node** (formerly $L_2 + (1-2u)L_1$).
- $H_\theta : \mathcal E \to [0,1]$ — the **soft decision** at the leaves.

Shallow MLPs are used everywhere with ELU activations; the paper uses
$d=8$ (embedding), one hidden layer of 50 units.

### Algorithm 1 — memoryless NPD

For a memoryless channel $W$, $E^W_\phi$ is applied **pointwise** to each
$y_i$. Two steps:

1. Sample $x_1^N \sim P_X^{\otimes N}$, pass through the channel to get
   $y_1^N$. Apply $E^W_\phi$ pointwise to get $e_1^N$.
2. Recursively walk the NSC tree with $F_\theta, G_\theta$; at the leaves,
   use $H_\theta$ to predict $u_1^N = x_1^N G_N$ via **parallel
   teacher-forced cross-entropy** (`fast_ce` — see their
   `polar_models.py::fast_ce`, lines 151-222).

Training minimises $\sum_i \mathrm{CE}(H_\theta(e_{0,i}), u_i)$ averaged over
all tree depths. This is Algorithm 2 (`NSCLoss`).

Consistency (Theorem 2): with enough samples and capacity, the learned
$\mathrm{H}^M_\phi(U_i \mid U_1^{i-1}, Y_1^N)$ converges to the true
$\mathrm{H}(U_i \mid U_1^{i-1}, Y_1^N)$, so the data-driven Bhattacharyya
parameter equals the true one.

### Algorithm 3 — NPD for channels with memory

What changes vs. Algorithm 1:

- The channel has state: $Y_i \sim W_{Y_i \mid X_1^i, Y_1^{i-1}}$. A purely
  pointwise $E^W$ is no longer a sufficient statistic.
- **$E^W_\phi$ becomes a full-sequence encoder** that takes $y_1^N$ and emits
  a per-position embedding $e_{0,i}$ that *is* a sufficient statistic
  (Proposition 1 in the paper).
- $E^W_\phi$ is obtained from the **DINE** network (Directed Information
  Neural Estimator, Zhang et al. 2023). Specifically: $E^W = T_{\Psi_Y}(1,
  \cdot) - T_{\Psi_Y}(0, \cdot)$ where $T_{\Psi_Y}$ is the trained
  Donsker–Varadhan potential. In the TF reference implementation
  (`NPDforCourse/models/dv_models.py::DINE`, lines 379-506), this is
  realised by a **stateful LSTM over the $y_1^N$ sequence followed by a
  small MLP**. The LSTM's hidden size equals $d=50$; a second LSTM
  estimates the $X \to Y$ branch needed for directed-information training.
- Training loop (Algorithm 3): for `Niters` outer iterations, sample a
  batch, compute $e^Y_{0,i} = E^W_\phi(y_i)$, set uniform $u_1^N$,
  compute $L = \mathrm{NSCLoss}(e^Y_{0,1}, u_1^N, 0)$, and minimise over
  $\phi, \theta$ jointly. So there is **no DINE pre-training phase** in
  Algorithm 3 itself — $E^W$ and $F/G/H$ are trained *simultaneously* by a
  single NSCLoss.
- Algorithm 4 additionally optimises a learnable input distribution
  $P_X^N$ (RNN-parametrised, in `input_models.py::BinaryRNN`) using RL
  gradients — needed when the capacity-achieving distribution is
  non-i.i.d.

So the "memory-channel" part of the paper is two things: (a) **replace the
per-sample $E^W$ with a sequence model**, and (b) keep the same tree /
NSCLoss. Point (b) is the important realisation for us — the tree
operations are unchanged.

---

## 2. Memory channels tested in the paper

| Channel | Definition | $\lvert\mathcal S\rvert$ | SCT comparison? | Paper BLER (approx.) |
|---|---|---|---|---|
| AWGN (memoryless ref) | $Y = (1-2X) + N,\ N\sim\mathcal N(0,0.5)$ | — | SC exact | matches SC, $\sim 10^{-5}$ at $n_t=10$ |
| BSC(0.1) (memoryless ref) | $Y = X \oplus N,\ N\sim\mathrm{Ber}(0.1)$ | — | SC exact | matches SC, $\sim 10^{-4}$ at $n_t=10$ |
| Ising | $Y = X$ or $Y = S$ w.p. 1/2; $S'=X$ | 2 | Trellis SC (SCT) | NSC ≈ SCT, $\sim 10^{-4}$ at $n_t=10$ |
| ISI | $Y_i = \sum_{k=0}^{r} h_k X_{i-k} + Z_i$, $Z\sim\mathcal N(0,0.5)$, $r=2$, $h_k=0.9^k$ | $2^r$ | Trellis SC | NSC ≈ SCT at $r\in\{1..4\}$; $r\ge 5$ SCT intractable |
| MA-AGN | $Y_i = (1-2X_i) + Z_i$, $Z_i = \tilde Z_i + \alpha \tilde Z_{i-1}$, $\alpha=0.9$ | $\infty$ (continuous) | **none** (SCT intractable) | NSC succeeds; only NPD result available |
| Trapdoor | $Y = X + S \bmod 2$ with deterministic trapdoor state | 2 | known analytic SCT | NSC ≈ SCT (Fig. 4 caption in paper) |

Reference SNR used throughout Section VI-B: per-user AWGN noise variance
$\sigma^2 = 0.5$. Block lengths are $N = 2^{n_t}$ with $n_t \in \{4, \dots,
10\}$, so up to $N = 1024$. Main takeaway from Fig. 4: the memory NSC
essentially **matches** SCT wherever SCT is tractable, and extends to
cases (MA-AGN, large ISI memory) where SCT isn't.

---

## 3. SCT chained approach for MAC memory channels (the actual derivation)

The two-user binary-input MAC with memory that we care about is, at time
$i$:
$$
(X_i, Y_i, S_i, Z_i) \sim P_{Z_i, S_i \mid X_i, Y_i, S_{i-1}},
\quad S_0 = 0,
$$
where $X_i$ is user 1's bit, $Y_i$ is user 2's bit (both binary), $S_i$ is
some finite-state index, and $Z_i$ is the channel output. We use
`channels_memory.py::ISIMAC` as the running example:
$$
Z_i = (1-2X_i) + (1-2Y_i) + h\bigl((1-2X_{i-1}) + (1-2Y_{i-1})\bigr) + W_i,
\quad W_i \sim \mathcal N(0, \sigma^2).
$$

Define the state as the tuple of previous bits: $S_i \triangleq
(X_{i-1}, Y_{i-1}) \in \{0,1\}^2$. Then $|\mathcal S| = 4$ and the
state transition is deterministic: $S_{i+1} = (X_i, Y_i)$. This is how
`polar/channels_memory.py::ISIMAC.build_leaf_tensors` already encodes
things (line 28-62 of that file).

### Stage 1 — decode $U$ treating $V$ as noise

In the SC chain, user 1 is decoded first assuming nothing is known about
user 2. Marginalising $Y_i$ with $Y_i \sim \mathrm{Ber}(1/2)$ (equiprobable
polar codeword bits) gives an *effective single-user memory channel*:

**Effective channel $\tilde W_1$ for Stage 1.** Input $X_i$, output $Z_i$,
state $\tilde S_i$. The key observation is that $X_{i-1}$ alone is not
enough state — we also need $Y_{i-1}$ because the ISI contribution from
user 2 persists. But $Y_{i-1}$ is *unknown* at Stage 1. Two equivalent
choices:

**Option A (large state).** Keep $\tilde S_i = (X_{i-1}, Y_{i-1})$, same
state as the full MAC. Transitions become random because $Y_i$ is random:
$$
P_{\tilde W_1}(\tilde S_{i+1} = (x, y') \mid X_i = x, \tilde S_i) = P_Y(y') = 1/2.
$$
The emission is
$$
P_{\tilde W_1}(Z_i \mid X_i = x, \tilde S_i = (x_{i-1}, y_{i-1}))
= \sum_{y \in \{0,1\}} P_Y(y)\, W\bigl(Z_i \mid x, y, (x_{i-1}, y_{i-1})\bigr).
$$
Explicitly, for ISI-MAC,
$$
P_{\tilde W_1}(Z_i \mid X_i, x_{i-1}, y_{i-1})
= \tfrac12 \sum_{y \in \{0,1\}} \mathcal N\bigl(Z_i;\,\mu(X_i,y,x_{i-1},y_{i-1}),\ \sigma^2\bigr),
$$
where $\mu(x,y,x_-,y_-) = (1-2x) + (1-2y) + h\bigl((1-2x_-)+(1-2y_-)\bigr)$.
This is a **4-state SC-trellis channel** with Gaussian emission (a 2-mode
mixture per state). Onay's SCT decoder applies as-is.

**Option B (collapsed state).** Keep only $\tilde S_i = X_{i-1}$ and treat
the $Y_{i-1}$ contribution as additional time-colored noise. This loses
optimality but shrinks $|\mathcal S|$ from 4 to 2; it matches the way
`neural/ncg_isi_mac.py` currently attacks the problem (with a sliding
window instead of explicit state).

**Mutual information bookkeeping.** The capacity of $\tilde W_1$ is
$I(X; Z)$ in the full MAC (i.e. user 1's marginal rate), which is exactly
the "Class A / Class B / Class C" rate allocation axis we already use
everywhere in this repo.

### Stage 2 — decode $V$ given decoded $\hat U$

After Stage 1 commits $\hat U_1^N$ (and therefore $\hat X_1^N$ via the
polar encoder), Stage 2 sees $Y_i$ as input, $Z_i$ as output, and can
condition on the known $X$-sequence.

**Effective channel $\tilde W_2$ for Stage 2.** Input $Y_i$, output $Z_i$,
state $\tilde S_i = (X_{i-1}, Y_{i-1})$, but $X_{i-1}$ is now an *observed
side-information symbol*, not a random variable. The emission is
$$
P_{\tilde W_2}\bigl(Z_i \mid Y_i = y, \tilde S_i = (x_{i-1}, y_{i-1});\ X_i = x_i\bigr)
= W\bigl(Z_i \mid x_i, y, (x_{i-1}, y_{i-1})\bigr),
$$
and the transition is deterministic in $y_{i-1}$:
$$
\tilde S_{i+1} = (X_i, y) \quad\text{where } y \text{ is the current input.}
$$
For ISI-MAC,
$$
P_{\tilde W_2}(Z_i \mid Y_i, x_i, x_{i-1}, y_{i-1})
= \mathcal N\bigl(Z_i;\,\mu(x_i, Y_i, x_{i-1}, y_{i-1}),\ \sigma^2\bigr).
$$
State space is still 2 (only $y_{i-1}$ is uncertain now). This is also a
standard single-user memory channel, so SCT / NPD / NSC all apply
directly.

**Rate.** The capacity of $\tilde W_2$ is $I(Y; Z \mid X)$. The sum
$I(X;Z) + I(Y;Z\mid X) = I(X,Y;Z)$ — the full MAC sum capacity, which is
the correct dominant-face rate for the chained decomposition (Onay
2013, §III).

### Why chaining is valid

Both stages are ordinary (single-user) finite-state channels with
finite-state output processes. The paper's Algorithm 3 is proven (Theorems
3, 4, 5 in the paper) to produce a consistent NPD for **any** FSC — it does
not care where the channel came from. So plugging $\tilde W_1$ and
$\tilde W_2$ into Algorithm 3 separately gives two NPDs that are, by
Onay's SC-chain argument, jointly a valid MAC decoder. No extra theory
needed.

---

## 4. Chained NPD algorithm sketch

Mirrors the paper's Algorithm 3 but instantiated twice.

### Stage 1: $\mathrm{NPD}_U$ (decode $U$ with $V$ as noise)

- **$E^W_1$**: input is the full $z_1^N$ sequence; output is
  $e^{Y,1}_{0,1:N}$ with $e^{Y,1}_{0,i} \in \mathbb R^d$. Architecture
  options ranked in §5.
- **$F_1, G_1, H_1$**: MLPs with embedding dim $d$; same shapes as the
  memoryless-MAC NPD already in `class_c_npd/models/npd_single_user.py`.
- **Training loop** (Algorithm 3):
  1. Sample $u_1^N \sim \mathrm{Unif}\{0,1\}^N$, $x_1^N = u_1^N G_N$.
  2. Sample $v_1^N \sim \mathrm{Unif}\{0,1\}^N$, $y_1^N = v_1^N G_N$.
  3. Draw $z_1^N$ from the ISI-MAC channel.
  4. $e^Y_{0,1:N} = E^W_1(z_1^N)$.
  5. Loss $L_1 = \mathrm{NSCLoss}(e^Y_{0,1:N}, u_1^N, 0)$ (the
     `fast_ce` of `npd_single_user.py::fast_ce` — already implemented,
     just with a sequence-aware $E^W$).
  6. SGD step on $\phi_1, \theta_1$.
- **Design**: after training, sweep MI per position; Stage 1 frozen set
  $\mathcal A_1^c$ has the worst positions.
- **Soft output**: $\hat u_1^N$ at inference is obtained by running the
  NSC tree recursion with hard decisions at info positions and $0$ at
  frozen positions (matches `class_c_npd/models/npd_single_user.py`).

### Stage 2: $\mathrm{NPD}_V$ (decode $V$ given $\hat U$)

- **$E^W_2$**: input is $(z_1^N, \hat u_1^N)$ — or equivalently
  $(z_1^N, \hat x_1^N)$ where $\hat x = \hat u G_N$ since the encoder is
  deterministic. The conditioning on $\hat X$ is the crucial change.
  Simplest realisation: concatenate $\hat x_i$ (or an embedding thereof)
  into each step of the Stage-2 sequence encoder.
- **$F_2, G_2, H_2$**: separate MLPs with their own parameters.
- **Training loop**: identical to Stage 1 but with the loss against
  $v_1^N$ and the $E^W_2$ input being $(z_1^N, x_1^N)$:
  - **Teacher-forcing variant (easiest)**: at training time, feed the
    *true* $x_1^N$ (so no Stage-1 error coupling); at eval time, feed
    $\hat x_1^N$ from the frozen Stage-1 decoder. This is equivalent
    to assuming a genie Stage 1 during training.
  - **Self-consistent variant (paper-faithful)**: run Stage 1 during
    each training iteration, feed $\hat x$. Slower but removes
    train/eval mismatch.

### Global training strategies — pick one

- **(a) Two-phase, frozen Stage 1**: train $\mathrm{NPD}_U$ to
  convergence, freeze it, then train $\mathrm{NPD}_V$ conditioned on
  its output. Simplest. This is how our GMAC code already does the
  chain in `class_c_npd/training/npd_design_sweep.py`.
- **(b) Joint fine-tuning**: after (a), unfreeze Stage 1 and co-train.
  Rarely necessary.
- **(c) Teacher-forced independence**: train both on true bits in
  parallel; only couple them at eval. Fastest to debug; adequate when
  Stage 1 BLER is low enough that error propagation is negligible
  (which it is at most of our operating points).

Recommendation: **(c) → (a) once (c) works**. Same recipe we already
use for class-C GMAC NPD.

---

## 5. Architecture options for $E^W$

$E^W$ in our MAC setting must produce a per-position embedding
$e_{0,i} \in \mathbb R^d$ that (i) captures whatever channel memory
there is, (ii) is invariant to permutations within a block only if the
channel is memoryless (obviously not here), (iii) is **cheap enough** to
run through `fast_ce` at block lengths up to $N=256$ with CPU batches
of 32–64. Four candidates:

| | Architecture | $e_{0,i}$ formula | Quality | Impl. cost | Verdict |
|---|---|---|---|---|---|
| **A** | Scalar pointwise (current NPD, broken) | $e_i = \mathrm{MLP}(z_i)$ | Poor — cannot represent ISI | Trivial | Reference for "worst case" |
| **B** | Sliding window (current NCG) | $e_i = \mathrm{MLP}(z_{i-k},\dots,z_i)$ | Matches true channel memory when $k \ge r$; clean CPU perf | Trivial — already in `ncg_isi_mac.py` | **Start here**, beats A easily |
| **C** | Bidirectional RNN / LSTM | $e_i = \mathrm{BiLSTM}(z_1^N)[i]$ | Highest; matches paper; unbounded context; good for unknown-memory channels | Medium; needs care with vectorised `fast_ce` | **Main deliverable** — this is what the paper uses |
| **D** | Transformer / self-attention | $e_i = \mathrm{SA}(z_1^N)[i]$ | Equally flexible; trains more stably than RNN at long $N$; O$(N^2)$ FLOPs | Higher | Optional — only if RNN is unstable |

Paper uses option C specifically: their DINE module runs an
`LSTM(units=d=50, return_sequences=True)` over the $y$-sequence
(`dv_models.py::DINE` line 412) and reads the per-step hidden state.
The output is fed through a small MLP (the "fc" layers of the DV
potential, lines 443-449) to get $e_i$.

**Stage 2 conditioning.** The simplest way to fold $\hat u_1^N$ (or
$\hat x_1^N$) into $E^W_2$ is **channel-wise concatenation at the input
of the sequence model**:
$$
E^W_2 : (z_i, \hat x_i)_{i=1}^N \longmapsto (e_i)_{i=1}^N,
$$
where $\hat x_i \in \{0,1\}$ is mapped to $\pm 1$ (or passed through a
learned embedding). For an LSTM this is a 2-dim input per step; for a
window, inputs are $(z_{i-k},\dots,z_i, \hat x_{i-k},\dots,\hat x_i)$.

Ranking for our 6-8 hour budget:

1. **B (sliding window)** — prove the chained NPD math works end-to-end
   on ISI-MAC at $N \in \{32, 64, 128\}$.
2. **C (BiLSTM)** — drop-in replacement for B's `z_encoder`; same
   tree, same loss. Should match SCT performance on ISI-MAC, r≤2.
3. A only as a regression baseline; D optional.

---

## 6. Memory channels — what exists vs. what we need

| Channel | In repo? | File | Notes |
|---|---|---|---|
| ISI-MAC (r=1, 2-tap) | yes | `polar/channels_memory.py::ISIMAC` | Gaussian emission, 4-state. Ready to use. |
| Gilbert–Elliott MAC | yes | `polar/channels_memory_new.py::GilbertElliottMAC` (name TBD) | Binary emission, 2×2 states (chain over both users). |
| Trapdoor MAC | yes | `polar/channels_memory_new.py::TrapdoorMAC` | 2 states, derived from single-user trapdoor. |
| Ising MAC | **need** | — | Two-user analogue of Ising $Y = X \text{ or } S$; spec in §6.1 below. |
| MA-AGN MAC | **need** | — | $Z_i = (1-2X_i)+(1-2Y_i) + \tilde Z_i + \alpha \tilde Z_{i-1}$; continuous state. |

### 6.1 Ising MAC spec (for when we need it)

Paper's single-user Ising (Sec. VI-B): state $S'$ = previous $X$; output
$Y = X$ or $Y = S$ w.p. 1/2 each. Two-user lift (my proposal; should be
checked against a literature search): state $S' = (X, Y)$, output
$Z = f(X, Y) \oplus N$ with $f$ chosen equal to XOR for the simplest
binary-output case — but honestly, for a first pass just implementing
continuous Ising-MAC via "$Z = X \text{ or } Y \text{ or } S_x \text{ or }
S_y$ chosen u.a.r." gives us a four-state binary channel that mirrors
the paper's test. Not blocking for Stage 1.

### 6.2 MA-AGN MAC spec

Straight extension:
$$
Z_i = (1 - 2X_i) + (1-2Y_i) + \tilde Z_i + \alpha \tilde Z_{i-1},
\quad \tilde Z_i \sim \mathcal N(0, \sigma^2).
$$
Noise state is continuous (the previous Gaussian sample), so SCT
intractable — this is the paper's highlighted case where **only the
neural decoder works**. Make this the "flagship" experiment once the
ISI-MAC result lands. Implementation: trivial; state is just the
previous noise sample, kept in the channel's internal buffer.

---

## 7. Concrete implementation plan (6-8h budget)

Priority order, with exact files to create/edit:

**Step 1 (30 min).** Verify SCT baselines exist on ISI-MAC.
- Check `polar/decoder_trellis.py` (single-user trellis) handles
  the 4-state ISI-MAC via the Option A effective channel from §3.
- If not, write a small helper `polar/sct_isi_mac.py` that instantiates
  Stage 1 and Stage 2 trellis SC decoders directly from
  `ISIMAC.build_leaf_tensors` (already produces exactly the `(N, 2, 2,
  S, S)` tensor the trellis decoder wants — see line 121-154 of
  `channels_memory.py`).
- Produce one table row `[N, SCT-BLER]` at $N\in\{32, 64, 128, 256\}$,
  $h=0.3$, SNR 6 dB. This is the target.

**Step 2 (2 h) — sliding-window NPD, Stage 1.**
- New file: `class_c_npd/models/npd_memory.py`. Subclass
  `NPDSingleUser` (existing). Override `encode_channel` to accept a
  sliding window of $z$ values, exactly like
  `neural/ncg_isi_mac.py::ISIMACNeuralDecoder.encode_z`.
  Keep `use_analytical_training=True` path intact.
- New file: `class_c_npd/training/train_npd_isimac_stage1.py`. Copy
  the structure of `npd_design_sweep.py`; swap the channel for
  `ISIMAC`, swap the decoder for `NPDMemory`. Iters: 30k at N=32,
  50k at N=64. Rate: use `design_mc.design_gmac_mc` style MC design
  against ISI-MAC (add an `ISIMAC`-aware design path if needed).

**Step 3 (2 h) — Stage 2 with $\hat X$ conditioning.**
- Same file as above; add a second model whose `encode_channel` takes
  `(z, x_hat_bpsk)` as a 2-channel sliding window.
- Training loop: two variants. (c) teacher-forced on true $x$ in
  parallel with Stage 1; then (a) freeze Stage 1 and re-train Stage 2
  on $\hat x$.
- Eval: run Stage 1 → get $\hat U$ → encode $\hat X$ → run Stage 2.
  Report `BLER_U`, `BLER_V`, `BLER_joint`.

**Step 4 (1 h) — LSTM $E^W$ upgrade.**
- New class `NPDMemoryLSTM` in `npd_memory.py`: replace the
  sliding-window MLP with `nn.LSTM(bidirectional=True)`. Everything
  else identical.
- Re-run Step 2 / Step 3 training at N=64 only; compare to window.

**Step 5 (1 h) — writeup numbers.**
- Produce `project_summary/results/npd_memory_mac_v1.json` with
  columns: `N | channel | stage1_backbone | BLER_U | BLER_V | BLER_joint
  | SCT_BLER | ratio`. Plot.

If there is time left in the 8h, Step 6: **MA-AGN MAC channel + repeat
Stage 1 + 2**. This is the experiment that the paper cannot do without a
neural decoder, so it's the strongest story.

### CPU budget check

All NPD training in this repo fits in CPU at these sizes
(`class_c_npd/results/` has existing GMAC N=256 runs at < 2 h each).
Sliding window adds essentially zero cost; LSTM roughly 2–3× per step
but still fine. PID 92903 is still consuming ~4 cores; with
`torch.set_num_threads(2)` we're fine to run one NPD trainer alongside.

---

## 8. What the paper tells us to expect

Concrete BLER predictions for our chained NPD on ISI-MAC ($h=0.3$,
$\sigma^2=0.25$ i.e. SNR 6 dB, rate = 0.5× MAC sum-capacity):

- **Chained SCT** (optimal with channel knowledge, our target): should
  behave like two concatenated single-user trellis SC decoders. From
  the paper's Fig. 4(c), ISI single-user SCT at $n_t=6$ achieves
  BLER ≈ $10^{-3}$; at $n_t=10$, $\sim 10^{-4}$. Stage 2 is *easier*
  than Stage 1 (conditioning on $X$ reduces state uncertainty), so
  the joint BLER will be dominated by Stage 1. Expected: $10^{-3}$ at
  $N=64$, $10^{-4}$ at $N=256$.
- **Chained NPD with LSTM $E^W$, correctly trained**: the paper
  reports NSC ≈ SCT on single-user ISI with $r\le 4$
  (Fig. 4(c), Fig. 7 shows both at 800 units/decoder). We should
  expect **the same match** stage-by-stage, so joint BLER within 1.5×
  of chained SCT. Concrete projection: $2\cdot 10^{-3}$ at $N=64$,
  $2\cdot 10^{-4}$ at $N=256$.
- **Chained NPD with sliding window ($k=1$)**: strictly worse than
  LSTM because it can only see 1 step of context, but for
  $r=1$ ISI it should actually be **equivalent** — the channel's
  state is fully determined by $(x_{i-1}, y_{i-1})$ which both stages
  can reconstruct from a 2-sample window. So for *this* channel the
  window should match the LSTM; for MA-AGN (infinite memory) the
  LSTM will dominate.
- **Memoryless-SC MAC baseline** (existing class-B result): BLER
  ≈ 0.47 at $N=64$ per `PIVOT_BRIEFING.md`. That's 500× worse than
  the SCT / NPD projections above — shows how much lift correctly
  handling memory gives.
- **Existing NCG-window** (`results/isi_mac_nn_results.json`): BLER
  0.47 at $N=64$ = same as memoryless SC. So the NCG-window
  implementation is **not** correctly capturing memory — almost
  certainly because its encoder doesn't see the two-user contribution
  to the state. Chained NPD should blow it away.

### Sanity target for the first successful run

A chained sliding-window NPD on ISI-MAC at $N=64$, rate 0.5, SNR 6 dB
should report $\mathrm{BLER}_\text{joint} < 0.05$ within a single
afternoon's training. Anything above $0.1$ means something is wrong in
the chain wiring (most likely: Stage 2 $E^W$ not seeing $\hat X$, or
train/eval mismatch in the $\hat X$ feed).

---

## Appendix A — symbol glossary (matches paper and repo)

| Symbol | Meaning |
|---|---|
| $N = 2^n$ | Polar block length |
| $U_1^N$ | User-1 message bits (info positions ⊂ $\mathcal A_U$) |
| $V_1^N$ | User-2 message bits (info positions ⊂ $\mathcal A_V$) |
| $X_1^N = U G_N$, $Y_1^N = V G_N$ | Polar-encoded codewords per user |
| $Z_1^N$ | Channel output |
| $S_i$ | Channel state at time $i$; for ISI-MAC, $(X_{i-1}, Y_{i-1})$ |
| $E^W_\phi$ | Channel embedding (paper's $E^W$); per-position embedding $e_i \in \mathbb R^d$ |
| $F_\theta, G_\theta, H_\theta$ | Neural tree ops: check-node, bit-node, soft-decision |
| $\mathrm{NSCLoss}$ | Paper's Algorithm 2; `fast_ce` in `NPDforCourse/models/polar_models.py:151` and `class_c_npd/models/npd_single_user.py:201` |
| DINE | Directed-Info Neural Estimator; gives paper's $E^W$ for memory channels via LSTM |
| SCT | Trellis-based SC decoder (Onay / Wang–Siegel) |

## Appendix B — exact file pointers

- Paper: `papers/NPD_ziv_bashar.pdf`. Section III pp. 4-5 (Alg 1),
  Section IV pp. 6-8 (Alg 3, 4).
- Reference `NeuralPolarDecoder`:
  `/Users/ytnspybq/PycharmProjects/NPDforCourse/models/polar_models.py`
  lines 1020-1262. Memoryless `Ey`: line 969.
- Reference `DINE` (= memory-channel $E^W$):
  `/Users/ytnspybq/PycharmProjects/NPDforCourse/models/dv_models.py`
  lines 379-506. LSTM at line 412, `channel_stats` (forward pass used
  for $E^W$) at line 497.
- Reference SCT:
  `/Users/ytnspybq/PycharmProjects/NPDforCourse/models/sc_models.py`
  `CheckNodeTrellis` line 41, `BitNodeTrellis` line 76.
- Our memoryless-MAC NPD:
  `/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/class_c_npd/models/npd_single_user.py`.
  `fast_ce` line 201; `encode_channel` line 171.
- Our NCG window (existing sliding-window memory handling):
  `/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/neural/ncg_isi_mac.py`
  lines 1-94.
- ISI-MAC channel:
  `/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/polar/channels_memory.py`.
- Trellis SC:
  `/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/polar/decoder_trellis.py`.
- Existing NPD-for-MAC training driver (copy-template for Step 2):
  `/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/class_c_npd/training/npd_design_sweep.py`.
