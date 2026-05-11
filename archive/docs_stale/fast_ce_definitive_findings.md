# Fast_CE for MAC: Definitive Findings

## The Core Result

Single-user NPD fast_ce works perfectly at all N. 4-class MAC fast_ce degrades with N. The problem is the 4-class joint (u,v) decision, not the training method.

| N | Single-User NPD | 4-Class MAC | SC Baseline |
|---|-----------------|-------------|-------------|
| 4 | 0.004 | 0.450 | 0.457 |
| 8 | **0.000** | **0.025** | 0.043 |
| 16 | **0.005** | 0.092 | 0.034 |
| 32 | **0.000** | 0.373 | 0.043 |

## Why 2-Class Works But 4-Class Doesn't

Binary (2-class): wrong bit flips a sign → symmetric, bounded perturbation. The BitNode residual `e * u_sign + e_even` handles both signs equally during training.

4-class: wrong (u,v) creates one of 3 distinct error patterns. Each pattern produces a different combination of sign flips across the 4 Z₂×Z₂ characters. The model trained with teacher forcing never sees these mixed patterns.

## Key Insight: The N=8 Sweet Spot

At N=8, the 4-class MAC decoder with fast_ce BEATS SC (BLER 0.025 vs 0.043). This works because N=8 has only 3 tree depths — the 1 shallow depth (73% accuracy) is compensated by 2 near-perfect deep depths.

At N≥16, more depths are "shallow" (stuck at ~73% accuracy), and errors compound faster than the deep depths can recover.

## Per-Depth Accuracy Pattern

| N | Depth 0 | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 |
|---|---------|---------|---------|---------|---------|---------|
| 8 | 0.73 | 0.83 | 0.95 | 1.00 | | |
| 16 | 0.73 | 0.73 | 0.84 | 0.95 | 1.00 | |
| 32 | 0.73 | 0.73 | 0.73 | 0.83 | 0.93 | 0.99 |

Depth 0 is always ~73% — this is the z_encoder's raw prediction limit.

## What We Tried (All Failed for N≥16)

1. WHT decomposition — same gap
2. Character-sign residuals — same gap
3. Residuals on all ops — same gap
4. Per-edge loss — same gap
5. Noisy teacher forcing — worse
6. Scheduled sampling — same gap
7. Larger model (d=32) — marginal improvement
8. Gradient detaching — can't learn
9. Binary decomposition (u then v|u) — U above marginal capacity

## What Works

1. **Sequential training** with full gradients: matches SC at N≤128
2. **Hybrid** (fast_ce pretrain → sequential fine-tune): better quality (0.054 vs 0.088)
3. **d=32 sequential**: beats SC at N=32,64

## The Open Question

Can the 4-class MAC be decomposed into binary sub-problems that each work with fast_ce? The two-phase approach fails because U operates above marginal capacity. A different decomposition — perhaps based on the WHT character structure — might work.
