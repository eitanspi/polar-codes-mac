# Neural SC Decoder for Two-User MAC Polar Codes — Executive Summary

## Problem
Two users transmit polar-coded messages through a Gaussian MAC: Z = (1-2X) + (1-2Y) + noise. We replace the analytical SC decoder's tree operations with learned neural networks (weight-shared MLPs, ~25K params, 4-class joint (u,v) output). Goal: match analytical SC BLER at all code lengths N.

## Architecture
The decoder walks the polar code factor graph, using learned MLPs for each operation: CalcLeft (f-node), CalcRight (g-node), CalcParent (gated residual), and embedding/decision heads. All ops are weight-shared across tree positions — a model trained at one N works at any N. Training uses sequential teacher-forced tree walk with curriculum learning (N=16 -> 32 -> 64 -> ...).

## Results (GMAC, SNR=6dB, Class B symmetric rate)

| N | SC BLER | NN-SC BLER | Ratio | Verdict |
|---|---------|-----------|-------|---------|
| 32 | 0.046 | 0.046 | 1.0x | **Matches SC** |
| 64 | 0.025 | 0.026 | 1.03x | **Matches SC** |
| 128 | 0.016 | 0.019 | 1.17x | **Close** |
| 256 | 0.005 | 0.019 | 3.7x | Gap |
| 512 | 0.001 | 0.045 | 40x | Fails |

On discrete BEMAC channel: NN **beats** SC at all N (including 0.5x at N=64). Neural SCL(L=4) beats analytical SCL(L=4) by up to 8x.

## What Works / What Fails

**Works**: Curriculum learning, gated residual CalcParent, stable cosine LR (improved N=128 from 1.69x to 1.17x), Neural SCL at N<=64, MC frozen set design for Class B.

**Fails**: Scaling to N>=256. Root cause: weight-shared MLPs accumulate per-node errors through 8+ tree levels. Tried: larger model (d=32, 157K params — matches SC at N=32 but fails at N>=64 due to training budget), per-level ops (solves curriculum transfer but slow), NPD-style fast_ce parallel training (4-class MAC joint structure doesn't decompose through binary sign-flips; two-decoder binary approach impossible at Class B rates where R_u > marginal capacity).

## Key Open Problem

The single-user Neural Polar Decoder (Aharoni et al.) scales to N=1024 using O(log N) gradient depth via parallel teacher-forced training. Adapting this to the MAC's joint 2x2 probability structure is the critical unsolved problem. The MAC requires joint processing of both users — neither independent per-user decoders nor naive 4-class formulations work. A correct MAC-adapted parallel training method would likely solve the N-scaling bottleneck.

**Code**: https://github.com/eitanspi/polar-codes-mac | **Models**: d=16 checkpoints for N=32-1024 | **Data**: 824+ simulation points
