# N=256 MI Diagnostic and Inference Error Analysis

**Model:** `saved_models/ncg_gmac_mlp_N256.pt` (d=16, hidden=64, 39K params)
**Channel:** GMAC, SNR=6 dB, Class B symmetric (Ru=Rv=0.48)
**Test set:** 5000 codewords, seed=42

## 1. Teacher-Forced MI Diagnostic

Under teacher forcing, the model achieves **near-perfect per-position
accuracy at all 246 info steps.**

| metric | N=128 | N=256 |
|--------|-------|-------|
| mean per-user accuracy | 0.9998 | 0.9999 |
| min per-user accuracy  | 0.9966 | 0.9976 |
| max MI gap (genie - model) | 0.0023 bits | 0.0082 bits |
| mean MI gap | -0.0008 bits | 0.0002 bits |

Both models extract virtually all available information at every position.
The diagnostic **cannot distinguish** N=128 (matches SC) from N=256
(5x worse than SC).

### MI gap distribution at N=256

| gap range (bits) | count (of 246 steps) |
|------------------|----------------------|
| [0.00, 0.01)     | 219                  |
| [0.01, 0.05)     | 0                    |
| [0.05, 0.10)     | 0                    |
| > 0.10           | 0                    |
| < 0 (model > genie) | 27                |

219 of 246 steps have MI gap < 0.01 bits. No position is a bottleneck.

### Conclusion from teacher forcing

The N=256 gap is **NOT** a per-position capacity problem. The tree ops
correctly extract all available information when given correct inputs.
The entire gap must come from error propagation at inference time.

## 2. Inference-Time Per-Step Error Rates

Running the decoder in **free-running mode** (no teacher forcing) and
tracking per-step error rates reveals the cascade pattern:

| quintile | steps   | NN err rate | SC err rate | NN/SC ratio |
|----------|---------|-------------|-------------|-------------|
| Q1 (early) | 0-49  | 0.000935    | 0.000245    | 3.8x        |
| Q2       | 49-98   | 0.002078    | 0.000571    | 3.6x        |
| Q3       | 98-147  | 0.002743    | 0.000245    | 11.2x       |
| Q4       | 147-196 | 0.004188    | 0.000510    | 8.2x        |
| Q5 (late)| 196-246 | 0.004016    | 0.000220    | **18.2x**   |

**Correlation of error rate with step position:**

| decoder | correlation |
|---------|-------------|
| NN      | **+0.70**   |
| SC      | -0.05       |

The NN's error rate grows linearly across the tree walk -- errors at
early steps cascade into later steps. SC's error rate is flat -- the
analytical tree operations don't suffer from error propagation because
they compute exact probabilities at every step regardless of prior
decisions.

### Error heat map

The 20 highest-error NN positions are all **U-only** info positions
decoded in the second half of the tree walk (steps > 400 out of 512).
Their per-step error rates are 0.54% to 0.90%, compared to SC's
0.00-0.10%. These positions are "late" in the interleaved path and
inherit accumulated errors from all prior decisions.

| position | step | NN err | SC err | ratio |
|----------|------|--------|--------|-------|
| 177      | 432  | 0.0090 | 0.0010 | 9.0x  |
| 209      | 464  | 0.0086 | 0.0000 | --    |
| 225      | 480  | 0.0080 | 0.0010 | 8.0x  |
| 201      | 456  | 0.0068 | 0.0010 | 6.8x  |
| 241      | 496  | 0.0066 | 0.0010 | 6.6x  |

## 3. Root Cause

The N=256 gap (5x SC on BLER) is entirely **inference-time error
cascade**, not a representation or capacity limitation.

The mechanism:

1. The CG decoder is trained with teacher forcing (sequential training).
   Under teacher forcing, every tree operation receives the **correct**
   embedding from the previous step.

2. At inference, when the decoder makes a wrong decision at step k,
   the embedding fed back into the tree at step k is **wrong**. All
   subsequent tree operations receive inputs they never saw during
   training.

3. The analytical SC decoder does not have this problem because its
   tree operations (CalcLeft, CalcRight, CalcParent) compute exact
   probabilities -- they do not depend on the quality of prior
   decisions. The decision at step k is wrong only if the channel
   was too noisy, not because of cascading errors.

4. At N=128 (7 tree depths, ~124 info steps), the cascade is short
   enough that the ~0.1% per-step error rate doesn't compound
   significantly. At N=256 (8 tree depths, 246 info steps), the
   cascade is 2x longer, and per-step errors at late positions reach
   0.5-0.9% -- enough to produce a 5x BLER gap.

## 4. What this suggests

The problem is **not** that the tree operations are undertrained or
that the model is too small. The problem is that the model has never
seen erroneous inputs during training.

**Scheduled sampling** (training with a mix of teacher-forced and
self-predicted inputs) was tried and failed. This is likely because
at N=256 the error space is too large to sample effectively -- the
model sees a different random error pattern at each training step and
can't generalize.

**A more targeted approach:** Instead of random scheduled sampling,
train with **adversarial error injection** at the positions identified
above (positions 177, 209, 225, 201, 241, etc.). At training time,
for these high-error positions, randomly flip the committed bit with
probability p (e.g., p=0.01-0.05) before embedding it. This teaches
the tree ops downstream of these positions to handle wrong inputs
without requiring the model to experience the full exponential space
of error patterns.

**Alternative:** Train a small **error-correction module** that sits
between the committed embedding and the tree ops. This module takes
the committed embedding + top-down message and outputs a "corrected"
embedding that is more robust to decision errors. This is analogous
to the residual connections in the NPD's BitNode.

## 5. Reproducing

```bash
# Teacher-forced MI diagnostic at N=256 (25 seconds)
python scripts/n256_mi_diagnostic.py

# Same at N=128 (12 seconds)
python scripts/n128_mi_diagnostic.py

# Inference-time per-step error rates (1 minute)
python scripts/n256_inference_errors.py
```

## 6. Data files

- `results/n256_mi_diagnostic.json` -- per-step MI, accuracy, genie comparison
- `results/n256_inference_errors.json` -- per-step NN vs SC error rates
