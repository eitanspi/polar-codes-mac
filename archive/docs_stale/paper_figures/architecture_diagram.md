# Neural SC Decoder Architecture Diagram

## Overview

```
                    Channel Output Z = f(X,Y) + noise
                              |
                        [z_encoder]
                    BEMAC: Embedding(3, d)
                    GMAC:  MLP(1 → 32 → d)
                              |
                    z_emb ∈ R^{N×d}  (bit-reversed)
                              |
                    ┌─────────────────────┐
                    │  Computation Graph   │
                    │  (Binary Tree Walk)  │
                    │                     │
                    │  root[i] = z_emb[i] │
                    │                     │
                    │  For each leaf t:    │
                    │    Navigate to t     │
                    │    Apply CalcLeft/   │
                    │    CalcRight at each │
                    │    internal node     │
                    │    → top-down msg    │
                    │                     │
                    │    Combine with      │
                    │    CalcParent msg    │
                    │    → leaf embedding  │
                    │                     │
                    │    [emb2logits]      │
                    │    → (û,v̂) decision  │
                    │                     │
                    │    [logits2emb]      │
                    │    → update leaf     │
                    └─────────────────────┘
                              |
                        (û₁,...,ûₙ), (v̂₁,...,v̂ₙ)
```

## Neural Modules (Weight-Shared)

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  CalcLeft:  MLP(3d → 64 → 64 → d)                      │
│  ┌────────────────────────────────────┐                  │
│  │ Input: [left_emb; right_emb; msg]  │  3d = 48        │
│  │ Linear(48, 64) + ReLU              │                  │
│  │ Linear(64, 64) + ReLU              │                  │
│  │ Linear(64, 16)                     │  → d = 16       │
│  └────────────────────────────────────┘                  │
│                                                          │
│  CalcRight: MLP(3d → 64 → 64 → d)  (same architecture) │
│                                                          │
│  CalcParent: Gated Residual                              │
│  ┌────────────────────────────────────┐                  │
│  │ Input: [left_child; right_child]   │  2d = 32        │
│  │                                    │                  │
│  │ candidate = MLP(32 → 64 → 64 → d) │                  │
│  │ gate = σ(MLP(32 → 64 → d))        │                  │
│  │                                    │                  │
│  │ output = gate * candidate          │                  │
│  │        + (1-gate) * mean(children) │                  │
│  └────────────────────────────────────┘                  │
│                                                          │
│  emb2logits: MLP(d → 64 → 64 → 4)   (4 = |(u,v)|)     │
│  logits2emb: MLP(4 → 64 → 64 → d)                      │
│                                                          │
│  Total: ~39,000 parameters (d=16, hidden=64)             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Tree Walk Example (N=8, Class B path)

```
        Depth 0:     [root edge - z_emb]
                     /                \
        Depth 1:  [edge 2]          [edge 3]
                  /      \          /      \
        Depth 2: [e4]   [e5]     [e6]   [e7]
                 / \    / \      / \    / \
        Depth 3: leaves (positions 1..8)

  Path b = [0,1,0,1,0,1,0,1]  (interleaved Class B)

  Step 1: Navigate to leaf 1 (pos=8, User U)
          Apply CalcLeft at depth 0 → edge 2
          Apply CalcLeft at depth 1 → edge 4
          Apply CalcLeft at depth 2 → leaf 8
          emb2logits → decide û₁

  Step 2: Navigate to leaf 2 (pos=4, User V)
          CalcParent: combine edges 8,9 → edge 4
          CalcRight at depth 1 → edge 5
          CalcLeft at depth 2 → leaf 9... (but actually traverse)
          emb2logits → decide v̂₁

  ... (continue for all 2N = 16 leaf decisions)
```

## Training Pipeline

```
  ┌─────────┐    ┌──────────┐    ┌───────────┐
  │ Generate │    │ Forward  │    │  Compute  │
  │ (u,v)   │───>│ tree walk│───>│ CE loss   │
  │ random  │    │ teacher  │    │ at leaves │
  │ info    │    │ forcing  │    │           │
  └─────────┘    └──────────┘    └───────────┘
       |              |               |
       v              v               v
  Encode          z_encoder        Backprop
  X = Enc(u)      + tree ops       through
  Y = Enc(v)      CalcL/R/P        all ops
  Z = Channel     emb2logits       (sequential)
```

## Curriculum Learning

```
  N=16 (5K iters) → N=32 (15K) → N=64 (50K) → N=128 (30K) → N=256 (100K)
       ↓                ↓             ↓              ↓              ↓
   Transfer         Transfer     Transfer        Transfer      Transfer
   weights          weights      weights         weights       weights
   (shared ops)     (shared)     (shared)        (shared)      (shared)
```
