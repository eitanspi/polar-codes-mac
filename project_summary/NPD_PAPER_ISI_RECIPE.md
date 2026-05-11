# NPD Paper ISI Recipe Analysis

Source: Aharoni, Huleihel, Pfister, Permuter. "Data-Driven Neural Polar Decoders for Unknown Channels With and Without Memory." IEEE Trans. Inf. Theory, vol. 70, no. 12, Dec. 2024.

Reference code: `/Users/ytnspybq/PycharmProjects/NPDforCourse/`

---

## 1. Architecture

### Channel Embedding E^W (replaces analytical LLR)
- **Memoryless channels (BSC, AWGN):** Single Dense layer `Ey = Sequential([Dense(d, use_bias=True, activation=None)])` (sc_models.py line 969). Maps scalar y to d-dim embedding.
- **Memory channels (ISI, Ising, Trapdoor):** The paper uses the SAME architecture but with a SEQUENTIAL channel. The ISI channel class (channel_models.py line 176-218) computes `y_i = h1*x_i + h2*x_{i-1} + w_i` with h=[0.9, 0.5], var=0.5. The channel outputs are generated sequentially. However, the Ey embedding is still a POINTWISE Dense(d) -- it does NOT use an RNN or window.
- **Important:** The paper's ISI architecture is a per-position MLP embedding, NOT a sequential encoder. The paper mentions (Section IV-B.3, Eq. 18-19) that DINE gives sufficient statistics for channels with memory, and the RNN is used only for the INPUT DISTRIBUTION estimation (Section IV-C), NOT for the decoder embedding.

### Tree Operations (F, G, H -- shared across all depths)
- **Check node (F):** `CheckNodeNNEmb` (sc_models.py line 176-189). MLP: `concat([e1, e2], dim=-1)` -> `layers_per_op` hidden layers with `hidden_size` units, ELU activation -> linear output to `d` dims. Input dim = 2d.
- **Bit node (G):** `BitNodeNNEmb` (sc_models.py line 208-222). MLP: `concat([e1 * u_sign, e2], dim=-1)` -> same MLP structure -> output + RESIDUAL `e1*u_sign + e2`. The `u_sign = 2*u - 1`.
- **Soft decision (H):** `Embedding2LLR` (sc_models.py line 227-240). MLP: d -> hidden layers -> 1 output, clipped to [-30, 30].

### Per-depth or shared?
**SHARED.** All three neural operations (CheckNodeNNEmb, BitNodeNNEmb, Embedding2LLR) are instantiated ONCE and used at ALL depths. This is explicit in the code: `self.checknode = CheckNodeNNEmb(...)` is a single instance (polar_models.py line 973-978), and the recursive `decode`/`encode` methods call `self.checknode.call(...)` at every depth.

### Architecture parameters for ISI (from train-fscs.sh)
```
EMBEDDING_SIZE_POLAR=8    (d=8)
HIDDEN_POLAR=50           (hidden=50)
LAYERS_PER_OP=2           (2 hidden layers per MLP)
```

Paper Section VI-B states: "the channel embedding dimension is chosen to be d = 8 and all NNs have one hidden layer with 50 units."

**Correction:** The runfile says `LAYERS_PER_OP=2` but the paper says "one hidden layer." Looking at the code: `layers_per_op=2` means 2 intermediate Dense layers + 1 output layer = 3 total. But the paper says "one hidden layer with 50 units." This is a discrepancy. The code's `_layers` list creates `layers_per_op` hidden layers (each Dense with activation) plus 1 linear output. So `layers_per_op=2` = 2 hidden + 1 output = 3 layers total. The paper claims 1 hidden = 2 layers total. The runfile uses 2 hidden.

### LSTM layers
**Zero.** The NPD decoder does NOT use LSTM/GRU for the tree operations. It uses standard MLPs for F, G, H. The ISI channel's state is handled implicitly: the per-position scalar y contains information about x_{i-1} (via ISI), and the tree structure propagates this across positions. The paper does NOT use any recurrent component in the decoder -- the recurrence is only in the SC tree structure itself.

For the channel embedding estimation, the DINE method (Section IV-B.2) uses an RNN (eq. 18) but this is for ESTIMATING THE CHANNEL STATISTICS (capacity/MI), not for the actual decoder. The decoder uses a simple pointwise Dense(d).

---

## 2. Training Recipe

### Loss function: Multi-depth NSCLoss (NOT leaf-only BCE)
**Algorithm 2 (NSCLoss):** The loss accumulates binary CE at EVERY depth of the SC tree, not just the leaves.

From the paper, eq. (17):
```
L = 1/((n+1)*N) * sum_{l=0}^{n} sum_{i=1}^{N} L_ce(e_{n,i}, v_i)
```
where l ranges over all n+1 decoding depths (0 through n=log2(N)), and the loss at each depth is the binary CE of the soft-decision applied to the embedding at that depth.

Code implementation: `fast_ce()` in polar_models.py (line 152-222). It computes CE at every depth and stacks them into `loss_array`. The training loss is `tf.reduce_mean(loss_y_array)` (line 1075), which averages over ALL depths.

### Optimizer and schedule
- **Optimizer:** Adam with clipnorm=1.0 (polar_models.py line 1047)
- **Learning rate:** 1e-3 (from train-fscs.sh: `LR=0.001`)
- **No LR schedule** in the training code (no scheduler visible). Just fixed lr throughout.
- **Weight decay:** None (plain Adam, not AdamW)

### Training iterations
- **From train-fscs.sh:** `NUM_ITERS=1000000` (1M iterations)
- **Batch size:** 10 (from `BATCH=10`)
- **Train block length:** 1024 (from `TRAIN_BLOCK_LENGTH=1024`)

Paper Section VI, Fig. 6 right panel: "CE convergence as a function of training iterations" shows convergence around 2^15 = 32768 iterations for the ISI channel with n_t=10 (block length). But the training runs for 1M iterations total.

### Saving frequency
- **Checkpoint saved every 3600 seconds** (1 hour), from `saving_freq=3600`
- **Logging every 100 iterations**, from `logging_freq=100`

---

## 3. Training at N=1024

The paper trains at a FIXED block length of `TRAIN_BLOCK_LENGTH=1024` (N=1024, n=10). The same trained model is then evaluated at multiple block lengths for the DESIGN phase (polar code construction), and at the target block length for decoding.

**There is NO curriculum learning or progressive N schedule.** A single model is trained at N=1024 with batch=10 for 1M iterations.

**Nothing special for larger N:** The same architecture (d=8, hidden=50, 2 hidden layers per op) is used for all N. Since F, G, H are shared across depths and position-independent, the model trained at N=1024 works at any N.

---

## 4. ISI Channel Specifics

### ISI channel model (channel_models.py line 176-218):
```python
y_i = h1 * x_i + h2 * x_{i-1} + w_i
h = [0.9, 0.5], var = 0.5
x_bpsk = 1 - 2*x  (BPSK mapping)
```
State size |S| = 2 (binary x_{i-1}).

Paper Section VI-B: "the ISI channel... defined by the formula Y_t = sum h_i * X_{t-i-1} + Z_t... we set h_i = 0.9/(i+1) [sic] and sigma^2 = 0.5. The ISI channel has state size |S| = 2^r."

For r=2 (ISI with 2 taps as in the code), |S| = 2.

### ISI Design
Paper Section VI-B uses the MINE algorithm to estimate the channel MI, then applies Algorithm 3 (SC_design) to find the frozen set. The design phase does its own MC estimation using the trained NSC decoder's `fast_ce` for bit error probabilities.

### Achievable BLER at N=512, N=1024

From the paper's Fig. 7(a) and the BER plots:
- **Fig. 4(c) ISI Channel:** NSC decoder BER vs SNR (r_l = training block in log2). At r_l = 6 (approximately n=6, so evaluated at various N), the NSC line closely matches the SCT (analytical trellis) decoder line down to BER ~10^{-4} at r_l=10 (SNR ~10).
- **Fig. 7(a):** BERs of SCT vs NSC on ISI for varying r (memory length). At r=2, the NSC and SCT curves overlap down to ~10^{-4.5} BER.

**The paper does NOT report BLER (frame error rate) for specific N values.** It reports BER. The ISI results are in Fig. 4(c) and Fig. 7, showing BER as a function of "r_l" (which is SNR-like or block length). The code in `plot_q4_isi.py` has results from homework assignments, showing:
- N=256 (n=8): BER ~0.030, FER ~0.120
- N=1024 (n=10): BER not separately shown.

From `plot_q4_isi.py` (the homework evaluation):
```
n=4  (N=16):   BER=4.202e-02, FER=1.031e-01
n=5  (N=32):   BER=2.694e-02, FER=6.701e-02
n=6  (N=64):   BER=3.259e-02, FER=1.196e-01
n=7  (N=128):  BER=3.375e-02, FER=1.269e-01
n=8  (N=256):  BER=2.966e-02, FER=1.220e-01
```
Note: These are NPD on a SINGLE-USER ISI (not MAC). FER is 10-13% across all N, suggesting the decoder struggles with longer blocks (no improvement beyond N=32).

**Key finding:** The NPD paper's ISI results show the NSC matches the SCT (trellis) decoder for r=2. But the actual BLER numbers at N=512 and N=1024 are NOT explicitly stated -- only BER vs SNR curves.

From Fig. 7(a), at r=2:
- SCT and NSC both achieve BER ~3x10^{-5} at SNR=10
- They diverge slightly at lower SNR (NSC slightly worse)

**The paper does NOT provide N=512 or N=1024 BLER in a table.** The ISI results are purely graphical (BER vs r_l/SNR curves). The code rate used is R=0.25 throughout (configs.json line 11).

---

## 5. Comparison with Our Implementation

| Feature | NPD Paper | Our ISI-MAC NPD |
|---------|-----------|-----------------|
| Channel embedding | Pointwise Dense(d) | BiGRU or Window MLP |
| d | 8 | 16 or 64 |
| hidden | 50 | 64 or 128 |
| layers_per_op | 2 (code) / 1 (paper text) | 2 |
| F/G/H sharing | Shared across all depths | Shared across all depths |
| Loss | Multi-depth avg BCE (NSCLoss) | Multi-depth avg BCE (fast_ce) |
| Optimizer | Adam lr=1e-3 | AdamW lr=1e-3, weight_decay=1e-5 |
| Schedule | None | CosineAnnealingLR |
| Train N | Fixed 1024 | Per-N (16, 32, 64, 128, 256) |
| Iterations | 1M | 30K-300K |
| Batch | 10 | 32 |
| Channel | Single-user ISI | Two-user ISI-MAC (chained) |
| Recurrent encoder | NO (pointwise) | YES (BiGRU or sliding window) |

### Critical difference:
The NPD paper's ISI decoder uses a **pointwise** channel embedding (single Dense layer per y_i). It does NOT have any sequential/recurrent component in the embedding. The tree structure itself handles the sequential dependency. Our implementation adds a BiGRU encoder which is an architectural extension beyond what the paper does.

The paper handles ISI memory through the tree + MC design (which learns which bit positions are reliable under the memory channel), NOT through a sequential encoder. The DINE/RNN in the paper is only for channel capacity estimation and input distribution optimization (Section IV-B, IV-C), not for decoding.

---

## 6. Summary of Key Findings

1. **Architecture:** d=8, hidden=50, 2 hidden layers per MLP, shared F/G/H across all depths. Pointwise Dense(d) channel embedding.
2. **Training:** 1M iterations, batch=10, Adam lr=1e-3, no schedule, trained at N=1024. NSCLoss = multi-depth BCE average.
3. **ISI specifics:** r=2 taps, h=[0.9, 0.5], sigma^2=0.5. State size=2. No special handling for larger N.
4. **No LSTM/GRU in decoder:** Memory is handled purely by the tree structure + design.
5. **BLER not reported:** Only BER curves shown. Homework code shows FER ~10-13% for N=16-256 on single-user ISI (not MAC).
6. **Our BiGRU encoder is an extension** beyond the paper -- we explicitly model sequential dependencies that the paper handles implicitly through the tree.
