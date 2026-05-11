# Session 11 Investigation Summary (2026-04-16/17)

## Overview

Session 11 focused on three areas: (1) reliable trellis SC baselines, (2) d=16 h=100 standalone training, and (3) documentation updates with corrected numbers.

---

## 1. Reliable Chained Trellis SC Baselines (10K CW)

Ran `decoder_trellis_mac_chained.decode_chained()` at all N=16-256 with 10,000 codewords for the first time. This is the 2-stage chained decoder (not the joint 4-state trellis).

| N | Chained Trellis SC | errs/10K | Joint Trellis SC | Chained/Joint |
|---|---|---|---|---|
| 16 | 0.169 | 1689 | 0.166 | 1.02x |
| 32 | 0.082 | 822 | 0.083 | 0.99x |
| 64 | 0.041 | 407 | 0.026 | 1.57x |
| 128 | 0.022 | 223 | 0.018 | 1.22x |
| 256 | 0.006 | 61 | ~0.006 | ~1.0x |

Key finding: At N=16,32 the chained and joint trellis decoders are nearly identical. At N=64 there is a meaningful gap (1.57x) -- the joint decoder benefits from exploiting Y correlation that the chained decoder marginalizes away.

This distinction matters for NPD comparison: since the NPD is also chained (Stage 1 + Stage 2), the fairer comparison is against chained trellis SC, not joint.

Saved to: `results/reliable_evals/isi_mac_sc_10kcw.json`

---

## 2. NPD vs Chained Trellis SC (corrected ratios)

With the new chained trellis baselines, the NPD-to-SC ratios improve significantly at N=64:

| N | NPD (original) | NPD (d16 h100) | Chained Trellis | NPD/Chained | Joint Trellis | NPD/Joint |
|---|---|---|---|---|---|---|
| 16 | 0.143 (d=16 h=64) | -- | 0.169 | **0.85x** | 0.166 | 0.86x |
| 32 | 0.081 (d=16 h=64) | -- | 0.082 | **0.99x** | 0.083 | 0.98x |
| 64 | 0.046 (d=16 h=64) | **0.027** (d=16 h=100 @95K) | 0.041 | **0.67x** | 0.026 | **1.05x** |
| 128 | 0.030 (d=64 h=128) | pending | 0.022 | 1.36x | 0.018 | 1.67x |
| 256 | 0.011 (d=64 h=128) | -- | 0.006 | 1.83x | ~0.006 | 1.83x |

**KEY RESULT: d=16 h=100 at N=64 achieves BLER=0.027 (137/5000), BEATING chained trellis SC (0.041) by 33% and MATCHING joint trellis SC (0.026, ratio 1.05x, CIs overlap). NPD now beats or matches the strongest analytical decoder at N=16, 32, AND 64.**

The N=64 ratio drops from 1.77x (vs joint) to 1.12x (vs chained). This is a much better story for the thesis: the NPD is within 12% of the corresponding analytical decoder at N=64.

---

## 3. d=16 h=100 Standalone Training (in progress)

### Hypothesis
Session 10 showed that hidden width matters more than embedding dimension. The h=100 BiGRU with d=16 should perform better than d=16 h=64 (the existing models) and possibly match d=64 h=128 at N=128.

### N=64 (200K iters, from scratch) -- BREAKTHROUGH
- At 25K iters: S1 BLER = 0.035 (176/5000)
- At 50K iters: S1 BLER = 0.029 (145/5000)
- At 100K iters: **S1 BLER = 0.028 (139/5000, CI [0.024, 0.033])**
- Best checkpoint (95K): **S1 BLER = 0.027 (137/5000, CI [0.023, 0.032])**
- **BEATS chained trellis SC** (0.041): ratio = **0.67x** (33% improvement!)
- **MATCHES joint trellis SC** (0.026): ratio = 1.05x, CIs overlap!
- **41% better** than d=16 h=64 (0.046)
- Training still ongoing (100K/200K) -- may improve further
- The hidden width increase (64 -> 100, 2x params from 20K->42K per stage) is transformative

### N=128 (500K iters, from scratch)
- Will start after N=64 completes
- Expected ~4h
- Target: beat d=16 h=64's plateau at ~0.16, approach d=64 h=128's 0.030

---

## 4. Documentation Updates

Updated the following with corrected numbers:
- `BLER_TABLES.md` Table 7: Added "Chained Trellis SC" column, updated N=256 trellis from 0.007 (7/1000) to 0.006 (61/10000), added chained vs joint comparison notes
- `MEMORY_MAC_CHAPTER.md`: Updated Section 1 (exec summary), Section 3.2 (chained trellis table with 10K CW numbers), Section 4.3 (N=128 corrected from 0.099 to 0.030), Section 4.9 (summary table with both chained and joint baselines)

---

## 5. Key Findings from Session 10 (carried forward)

- LSTM vs GRU: no difference in BLER
- Memorization test: passes (model generalizes to unseen codewords)
- First-error shifts to tail with N (Q1+Q2 at N=64, Q2+Q3 at N=128)
- NPD paper uses pointwise embedding (but BiGRU is better for MAC)
- SC baselines corrected (N=64 ratio changed from 0.90x to 1.77x vs joint)
- d=16 fails at N>=128; d=64 required

## 6. Open Items

- d=16 h=100 training results (pending -- running now)
- ISI-MAC trellis SC at N=512/1024 still unreliable (0/500 and 2/300)
- Direct N=512 d=64 training not started
- Figure regeneration pending (needs final training results)
