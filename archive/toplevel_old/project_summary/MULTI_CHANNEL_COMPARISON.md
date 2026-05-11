# Multi-Channel Comparison Table: Neural NPD vs Analytical Baselines

*Created 2026-04-24. Summarizes NPD performance across all memory and memoryless MAC channels.*

---

## Overview

The chained NPD (Stage 1 decodes U, Stage 2 decodes V|U-hat) has been trained and evaluated
on four distinct MAC channel types. This table consolidates all results.

---

## ISI-MAC (Finite-State, 2 states per user, 4 states joint)

**Channel:** Z_t = (1-2X_t) + (1-2Y_t) + h*((1-2X_{t-1}) + (1-2Y_{t-1})) + W_t
**Parameters:** h=0.3, SNR=6 dB (sigma^2=0.251)
**Analytical baseline:** Chained trellis SC (2-state FB per stage)

| N   | Chained NPD (best) | Chained Trellis SC | NPD/Trellis | Config |
|-----|--------------------|--------------------|-------------|--------|
| 16  | 0.143              | 0.169              | **0.85x**   | d=16 h=64 |
| 32  | 0.081              | 0.082              | **0.99x**   | d=16 h=64 |
| 64  | 0.032              | 0.041              | **0.78x**   | d=16 h=100 |
| 128 | 0.030              | 0.022              | 1.36x       | d=64 h=128 |
| 256 | 0.011              | 0.006              | 1.83x       | d=64 h=128 |

**Summary:** NPD beats or matches chained trellis SC up to N=64. At N=128-256, the gap grows to 1.4-1.8x. The hidden width increase (h=64->100) closes most of the gap at N=64.

---

## Ising MAC (Finite-State, 2 channel states, state-independent transitions)

**Channel:** Good state: Z = (1-2X)+(1-2Y)+W. Bad state: Z = W. Markov flip p=0.1.
**Parameters:** sigma^2=0.251, p_flip=0.1
**Analytical baseline:** Chained Markov trellis SC + Memoryless GMAC SC

| N   | Chained NPD | Trellis SC (chained) | Memoryless SC | NPD/Trellis | Config |
|-----|------------|---------------------|--------------|-------------|--------|
| 16  | **0.592**  | 0.575               | 0.634        | 1.03x       | d=16 h=100 |
| 32  | **0.770**  | 0.689               | 0.781        | 1.12x       | d=16 h=100 |

**Summary:** BLER is very high (>0.57) due to the BAD state (pure noise) occurring ~10% of the time. The trellis SC provides meaningful gain over memoryless (9-12%). The NPD partially learns the Ising memory (beating memoryless by 7% at N=16) but doesn't fully match the trellis SC which knows the exact channel model.

---

## MA-AGN MAC (Continuous-State, no finite-state trellis)

**Channel:** Z_t = (1-2X_t)+(1-2Y_t)+N_t, N_t = alpha*N_{t-1}+W_t (AR(1) noise)
**Parameters:** alpha=0.3, sigma^2=0.251
**Analytical baseline:** Memoryless GMAC SC (no trellis exists)

| N   | Chained NPD (best) | Memoryless SC | NPD/SC | Config |
|-----|--------------------|--------------|---------| -------|
| 16  | 0.138              | 0.175        | **0.79x** | d=32 h=128 |
| 32  | 0.112              | 0.077        | 1.46x    | d=32 h=128 |
| 64  | **0.035**          | 0.025        | **1.42x** | d=16 h=100 |
| 128 | training...        | --           | --       | d=16 h=100 |

**Summary:** At N=16, the NPD exploits the AR(1) memory and beats memoryless SC by 21%. At N>=32, the NPD underperforms, but the d=16 h=100 architecture closes nearly half the gap at N=64 (from 2.38x to 1.42x). This is the one channel where the neural approach has unique value (no trellis exists) but also the hardest to scale.

---

## GMAC (Memoryless, no state)

**Channel:** Z = (1-2X) + (1-2Y) + W, W ~ N(0, sigma^2)
**Parameters:** SNR=6 dB

### Class C (corner rate)

| N   | Chained NPD | SC    | NPD/SC | Config |
|-----|------------|-------|--------|--------|
| 16  | 0.107      | 0.162 | **0.66x** | d=16 |
| 32  | 0.037      | 0.068 | **0.55x** | d=16 |
| 64  | 0.010      | 0.027 | **0.37x** | d=16 |
| 256 | 0.0003     | 0.002 | **0.19x** | d=16 |

### Class B (symmetric rate, NCG decoder)

| N   | NCG  | SC    | NCG/SC | NN-CA-SCL L=4 |
|-----|------|-------|--------|---------------|
| 32  | 0.050 | 0.045 | 1.12x | -- |
| 64  | 0.028 | 0.028 | 1.02x | 0.004 |
| 128 | 0.023 | 0.019 | 1.23x | 0.006 |
| 256 | 0.023 | 0.006 | 4.5x (WALL) | **0.003** |

---

## Cross-Channel Summary

| Channel | Best NPD result | Where NPD excels | Where NPD struggles |
|---------|----------------|-------------------|---------------------|
| BEMAC C | BLER~0 at N>=64 | All N (discrete channel, easy) | Never |
| GMAC C  | 0.19x SC at N=256 | N=16-256 (NPD+frozen set co-adapt) | N=128 (regression) |
| GMAC B  | 1.0x SC at N=64 | N=32-64 (matches SC) | N>=256 (4.5x wall) |
| ISI-MAC | 0.78x trellis at N=64 | N<=64 (BiGRU learns ISI) | N>=128 (gap grows) |
| MA-AGN  | 0.79x SC at N=16, 1.42x at N=64 | N=16 (learns AR memory) | N>=32 (underperforms SC) |
| Ising   | 1.03x trellis at N=16 | Beats memoryless 7% | Very high BLER regime |
