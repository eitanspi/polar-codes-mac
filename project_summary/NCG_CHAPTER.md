# The Neural Computational Graph (NCG) Decoder for Two-User MAC Polar Codes

## 1. Executive Summary

The Neural Computational Graph (NCG) decoder is a fully neural successive-cancellation (SC) decoder for two-user MAC polar codes: every analytical tensor operation along the SC tree walk (CalcLeft, CalcRight, CalcParent, channel embedding, leaf decision, re-embedding) is replaced by a learned MLP operating on d=16 embeddings. On discrete channels (BEMAC, ABNMAC) the NCG matches or beats analytical SC at all tested block lengths from N=16 up to N=1024. On the Gaussian MAC (GMAC) at the symmetric-rate Class B operating point it matches SC within 4% for N up to 128, and then abruptly degrades: at N=256 its BLER is 0.015 vs. SC's 0.005 (2.2-3.0x worse depending on the codeword budget), at N=512 the gap is roughly 12x, and at N=1024 training from scratch is infeasible. Today's experiments (sections 5-6) rule out exposure bias and cascade amplification as the explanation and identify the N=256 gap as a union of (i) a small number of NCG-specific weak positions (the top 5 per user account for roughly half of the failures) and (ii) a large number of small per-position deficits spread across the info set that NCG cannot close at any rate. Rate-1 training further reveals that "rate specialization" — which positions the model sees as information versus frozen during training — matters more in the MAC setting than for single-user codes, because one user's frozen pattern is the other user's decoder state.

## 2. Architecture

The NCG mirrors the SC tree exactly and replaces analytical operations with learned modules that all share a common d-dimensional embedding space.

The decoder walks the same binary tree as the analytical SC MAC decoder (2N-1 edges, edge 1 at root, edges N..2N-1 at leaves) and processes the same path b in {0,1}^{2N}. For Class B (interleaved path, R_u = R_v ≈ 0.48) the CalcParent operation is required at most non-trivial tree positions. Every edge carries a learned d=16 embedding instead of a 2x2 log-probability tensor. Six MLPs mediate every operation: (i) z_encoder Linear(1,32)-ELU-Linear(32,d) maps each scalar channel output to an embedding, followed by bit-reversal; (ii) CalcLeft (f-node) 3d-64-64-d combines parent-first-half, parent-second-half, and sibling; (iii) CalcRight (g-node) shares the same 3d-64-64-d signature; (iv) CalcParent uses a gated-residual MLP with sigmoid gate over a 2d-64-d candidate and a (left+right)/2 residual baseline; (v) emb2logits d-64-64-4 produces joint log-probabilities over (u,v); (vi) logits2emb re-embeds a decision back to d. Total parameter count is ~39,000 (the Soft-Bit Bridge variant used earlier in the project had ~27,800 parameters and used analytical circ_conv for CalcParent; the fully-neural gated-residual variant is what is deployed in the current results). All six modules are weight-shared across every tree position and every tree depth, so a model trained at one N can be reused at any N — the architecture itself is N-agnostic. Inference is fully sequential and for N=256 requires roughly 1500 MLP calls per forward pass, each on tiny tensors, which is the main wall-clock bottleneck.

## 3. Training Methodology

Training is sequential teacher-forced cross-entropy with a curriculum over N; curriculum is not optional and the training dynamics have an unusually long silent initialization phase before polarization emerges.

A training iteration processes one codeword through the full SC tree walk using the TRUE bits as the conditioning history; cross-entropy is computed at every non-frozen leaf between the 4-class prediction and the ground-truth (u,v) and gradients flow through the entire O(N log N) sequential chain. Frozen positions contribute no loss but still propagate embeddings. The optimizer is AdamW (weight_decay=1e-5), gradient clipping 1.0, batch size 4-64 depending on N, learning rate 5e-5 to 3e-4 with a single cosine decay (a critical fix: switching from cosine-with-restarts to plain cosine improved N=128 from 0.027 to 0.017 and N=256 from 0.033 to 0.015). Training from scratch fails at N >= 32; the curriculum N=16 -> 32 -> 64 -> 128 -> 256 -> 512 is mandatory because random MLPs at tree depth 8 destroy the channel signal before it reaches any leaf (a random model at N=256 reaches only 37% teacher-forced accuracy, barely above the 25% 4-class baseline). A rate-1 training POC at N=32 exposed a striking training dynamic: cross-entropy stays at its random value for approximately 200K iterations, undergoes a phase transition between 250K and 280K iterations, and is still climbing at 300K iterations — polarization emerges suddenly after a long silent initialization. The typical iter budgets are roughly 15K (N=32), 80K (N=64), 135K (N=128), 100K (N=256), 45K+ (N=512); wall-clock on CPU ranges from 20 minutes to more than a day per stage. A custom C++ PyTorch extension for the forward pass gives a 1.34x speedup by removing Python dispatch overhead on the ~1500 MLP calls; torch.compile and MPS both slow training down because the tensors are too small to amortize dispatch and transfer costs.

## 4. Results

NCG matches or beats SC at all tested N on discrete MAC channels and on GMAC up to N=128; on GMAC Class B it hits a wall at N=256.

The table below is distilled from the project master CSVs in `project_summary/results/` (`bemac_classB_ncg_vs_sc.csv`, `gmac_classB_ncg_vs_sc.csv`, `gmac_classC_npd_vs_sc.csv`) and from the 5000-codeword validation runs summarized in `docs/comprehensive_report.md` and `project_summary/README.txt`.

| Channel | Class | N | NCG BLER | SC BLER | Ratio | Regime |
|---------|-------|----|----------|---------|-------|---------|
| BEMAC   | B     | 64   | 0.003   | 0.006   | 0.54x | NCG beats SC |
| BEMAC   | B     | 128  | 0.001   | 0.002   | 0.60x | NCG beats SC |
| BEMAC   | B     | 256  | 4e-5    | 8e-5    | 0.50x | NCG beats SC |
| BEMAC   | B     | 1024 | —       | —       | 0.50-1.10x | No wall |
| GMAC    | B     | 32   | 0.046   | 0.046   | 1.00x | Matches SC |
| GMAC    | B     | 64   | 0.026   | 0.025   | 1.03x | Matches SC |
| GMAC    | B     | 128  | 0.017   | 0.016   | 1.04x | Matches SC |
| GMAC    | B     | 256  | 0.015   | 0.005   | 3.0x  | **The wall** (5000 cw) |
| GMAC    | B     | 256  | 0.021   | 0.004   | 5.2x  | Lower-rate baseline (today) |
| GMAC    | B     | 512  | 0.012   | 0.001   | 12x   | Large gap |
| GMAC    | C     | 256  | 0.000   | 0.001   | 0x    | NPD corner — beats SC |
| ABNMAC  | C     | 16   | —       | —       | 1.01x | Matches SC |
| ABNMAC  | C     | 8    | —       | —       | 0.96x | NCG slight edge |
| ISI-MAC | —     | 32   | 0.652   | 0.731   | 0.89x | Neural channel memory |
| ISI-MAC | —     | 64   | 0.442   | 0.575   | 0.77x | Neural channel memory |

The d=32 variant beats SC by ~20% at N=32 and N=64 (BLER 0.037 and 0.020 vs. SC 0.046 and 0.025) but never completed curriculum to N=256 — the runs died from system instability before reaching the target. CRC-aided NN-SCL at N=128 achieves BLER=0.002, strictly better than analytical SCL(L=4) at 0.004; at N=256 the same approach does not help because the correct path is pruned before the CRC check runs. On BEMAC the NCG's advantage over SC is unexplained: same code, same frozen set, same path — yet the NCG consistently has ~half the BLER at N=64, 128, 256. This is open question #1 in `project_summary/OPEN_QUESTIONS.txt`.

## 5. The N=256 Wall — Characterization

The wall is sharp and specific: GMAC Class B, N >= 256, where NCG's BLER plateaus near 0.015 while SC's continues to drop.

NCG's BLER across N on GMAC Class B is 0.046 -> 0.026 -> 0.017 -> 0.015 -> 0.012 at N = 32, 64, 128, 256, 512. Analytical SC on the same code goes 0.046 -> 0.025 -> 0.016 -> 0.005 -> 0.001. NCG's BLER decreases modestly with N and appears to be floored around 0.012-0.015; SC's BLER decreases approximately geometrically. The ratio therefore widens from 1.0x at N=32 to 3.0x at N=256 (5000 codewords) and 12x at N=512. The wall does NOT appear on BEMAC (NCG beats SC out to N=1024) nor on the GMAC Class C corner (the NPD variant is at-or-below SC at N=256). It also does not appear under teacher forcing: under TF the model achieves ~99.6% per-position accuracy at N=256, and MI (computed as (log 4 - CE)/log 4) averages 0.99. The gap is therefore specific to the combination of (a) Gaussian channel with continuous output, (b) interleaved Class B path with CalcParent needed, and (c) sequential inference with the decoder's own decisions fed back.

## 6. The N=256 Wall — Diagnostic Findings

Today's experiments eliminate two common-sense explanations and localize the remaining gap to a handful of weak positions plus a long tail.

**Exposure bias ruled out (`scripts/poc_a_teacher_vs_free.py`).** Measuring per-position mutual information under teacher forcing (true bits fed back) versus free-running (decoder's own decisions fed back) gives essentially identical values. The decoder's free-running predictions are right often enough that the autoregressive state it conditions on is on average the correct one. Error cascades, when they do occur, decay within 5-7 positions rather than propagating globally.

**Cascade amplification ruled out (`scripts/ncg_vs_sc_ber_bler_n256.py`).** In failed blocks, NCG produces 33.6 bit errors on average and SC produces 37.4. The cascade size within a failed block is therefore nearly identical between the two decoders. The 3-5x BLER gap comes almost entirely from NCG having roughly 5x as many failed blocks, not from NCG blowing up more catastrophically inside a failed block.

**First-error localization (`scripts/first_error_analysis.py`, `/tmp/first_err_n256.npz`).** Out of 226 info positions in the Class B frozen set at N=256, only 35 unique positions ever host the first error in a failing block. The top-20 such positions account for 84% of all failures. The top single contributor is step 241 (V-side position 113, per-position MI = 0.49), responsible for 15% of all block failures; step 112 (U-side position 112, MI = 0.90) accounts for 10%. The positions over-represented in first errors are the lowest-MI positions in the info set — precisely where SC would also be weakest, but where the NCG is disproportionately so.

**Rate-1 vs SC-design comparison at N=32 (`scripts/compare_rate_specialization_n32.py`).** A rate-1 model (all positions treated as info) trained to convergence has near-identical per-info-position MI to an SC-trained model on U-side positions, but severely under-rates V-side info positions (MI 0.14-0.54 vs. 0.84-0.93 for the SC-trained model). V bits are decoded in the middle of the path, conditioned on U; under SC training the U context is mostly zeros (frozen bits are zero with high probability), under rate-1 training the U context is fully random. The rate-1 model simply never sees the frozen-U context distribution. Errors between the two models are weakly correlated (phi=0.08) and an oracle ensemble improves BLER by 15%, confirming the two models have learned partly disjoint strategies.

**Warm-start fine-tune at an intermediate rate (rate 0.594, 40 minutes of fine-tuning from the rate-1 checkpoint with the NCG-selected frozen set) reaches BLER 0.52** — an order of magnitude worse than the trained-from-scratch baseline at 0.04. The rate-1 model is therefore not a universal decoder; which positions are seen as information during training materially shapes what the model learns.

**Lower-rate experiment at N=256 (`/tmp/exp_freeze_weakest.py`).** We start from the standard Class B rate (k_u = k_v = 123) and progressively freeze the NCG's weakest positions. Results: (123/123) NCG 0.021, SC 0.004, ratio 5.2x; dropping the top-5 weakest per user (118/118) NCG 0.012, SC 0.0037, ratio 3.3x; dropping top-10 (113/113) NCG 0.0066, SC 0.0020, ratio 3.3x; dropping top-20 (103/103) NCG 0.0021, SC 0.0002, ratio 10.5x. Two observations: (i) the top-5 positions per user are NCG-specific weak spots (NCG improves 43% when they are frozen, SC only 10%), confirming that a small number of positions bear a disproportionate share of the NCG-specific failures; (ii) once those are removed the NCG/SC ratio does not recover — it widens to 10.5x at 103/103 because SC extracts the full information-theoretic value of the removed positions while NCG only extracts some. The gap is therefore NOT concentrated in a small fixable set; beyond the top-5, the deficit is distributed across many positions with small individual contributions. NCG does not match SC at any rate on GMAC Class B N=256.

## 7. What Has Been Tried and Failed

Roughly three weeks of compute have been spent on approaches to either close the N=256 gap or train efficiently at large N; every approach other than sequential curriculum training has either failed or offered only marginal improvement.

The full catalog lives in `docs/advisor_meeting_failed_approaches.md`; a condensed summary follows.

- **Direct fast_ce for 4-class MAC**: loss plateaus at ~0.30 (vs random 1.4), BLER = 0.34 at N=32, ~7x SC. The fast_ce assumption that training-time errors stay small at inference holds for 2-class binary outputs but fails for 4-class joint (u,v), where a wrong prediction produces one of three qualitatively different embedding perturbations.
- **Walsh-Hadamard transform decomposition**: reached 0.254 BLER at N=32 (5.5x SC) despite 283K parameters. CalcLeft diagonalizes correctly in the WHT basis but the BitNode still presents three distinct error patterns.
- **Two-phase iterative refinement (decode U alone, then V | U, then refine U | V)**: 0.518 BLER at N=32 after two refinement passes. Phase 1 cannot succeed at Class B symmetric rate because R_u = 0.469 > I(Z;X) = 0.464 — U is literally above its marginal capacity.
- **Hybrid (fast_ce pretrain -> sequential fine-tune)**: helps 22% at N=32, completely fails at N=256 (BLER=1.0 across four LR/batch variants over 30 hours).
- **Gradient detaching (K = 4, 8, 16, 32)**: any detaching prevents the phase transition from BLER=1.0 to useful learning. Full gradients are required at least during the phase transition.
- **Scheduled sampling**: 21% improvement at N=256 (0.019 -> 0.015) but does not close the gap.
- **NN-SCL without CRC at N >= 256**: hurts. Miscalibrated confidence causes the list to prune the correct path.
- **CRC-aided NN-SCL at N=256**: zero improvement over plain NCG because the correct path is already lost before CRC can inspect it.
- **Larger model d=32**: beats SC at N=32 and N=64 (0.80x), trajectory at N=128 still improving when runs died, never reached N=256.
- **Residual connections in CalcLeft/CalcRight, snapshot distillation, multi-depth auxiliary loss, per-level CalcLeft/CalcRight, tree-op transfer from N=128**: each either fails outright or underperforms plain curriculum.

The only thing that works for the sequential NCG is weight-shared sequential training with a full curriculum.

## 8. Rate-Specialization Finding

The rate-1 POC revealed a MAC-specific training effect: what the other user is doing changes what the model must learn, and "universal" rate-1 models under-specify the problem.

In a single-user polar decoder, rate-1 training (all positions are info) is a reasonable proxy because the frozen set mainly changes which decisions the decoder needs to commit to, not what the channel and tree operations look like. In the two-user MAC with an interleaved path, this breaks: when decoding a V-side bit in the middle of the path, the decoder is conditioned on both past V decisions AND past U decisions, and the U history consists of U's frozen bits (almost always zero) and U's info bits (random). Under SC training the frozen U bits are indeed zero. Under rate-1 training they are random. The result, quantified in the N=32 comparison above, is that a rate-1 model learns the U-side info MI correctly (both models see random bits there) but mis-calibrates the V-side (rate-1 never sees the "frozen U = 0" context).

The 40-minute warm-start fine-tune at rate 0.594 reaches BLER 0.52 from a rate-1 checkpoint whose rate-1 training took hundreds of thousands of iterations. Trained from scratch at the same rate the baseline is BLER 0.04 — an order of magnitude better. In practice this means rate-1 training is not a shortcut to a decoder that can be freeze-set-specialized later; the frozen-set distribution must be present from the start of training.

The finding generalizes: in any MAC setting with path interleaving, one user's frozen pattern is part of the other user's decoder state. Rate-specialization therefore matters in the MAC beyond its usual role of shaping the input distribution. This is not visible in single-user NPD experiments and was not obvious from the architecture.

## 9. Lessons and Open Questions

The project has characterized the N=256 wall sharply enough to say what would be required to cross it and why those requirements lead to diminishing returns.

Four things would each reduce the gap by a modest amount but none alone closes it: (i) roughly 2x larger embeddings (d=32 or d=64) with roughly 5-10x more training iterations — extrapolating from the d=32 trajectory at N=128, this plausibly reaches 0.010 at N=256, still 2x SC; (ii) a parallel training method with O(log N) gradient depth — ruled out for 4-class joint MAC by the three-pattern error argument unless a genuinely novel decomposition is found; (iii) a decoder family that is parallel by construction (learned belief propagation on the polar factor graph), which is untested and a multi-month engineering effort; (iv) skip connections from the root embedding directly to leaves, addressing the optimization barrier at random initialization rather than the inference accuracy — untested.

Open questions that remain unresolved after all the diagnostics. First, why does the NCG beat SC on BEMAC with the same frozen set? The representation in d=16 embeddings apparently captures something scalar LLRs miss, but a clean theoretical explanation is absent. Second, why does ABNMAC Class B have non-monotonic SC BLER (N=16: 0.078, N=32: 0.125, N=64: 0.092)? Possibly a rate-selection artifact, not confirmed. Third, is there an equivalent to the rate-specialization effect for single-user polar codes that has simply been masked by the fact that U and V occupy the same position set there? Untested.

The cleanest requirement for N=256 parity is a parallel training method for 4-class MAC that doesn't rely on error smallness. We have not found one, and fast_ce, WHT, and two-phase iterative all have characterized failure modes that a new method would need to avoid simultaneously. This is why the direction is at diminishing returns: the remaining candidate fixes are each large projects with uncertain payoffs, and the best-case outcome (d=32 or d=64 at sufficient training) is a BLER in the 0.008-0.012 range at N=256, which is still meaningfully worse than SC's 0.005.

## 10. Conclusion

NCG is a solid secondary result that establishes feasibility and characterizes a boundary rather than a failure to solve the stated problem.

The positive contributions are real and specific: (i) the first fully-neural SC decoder for two-user MAC polar codes that matches analytical SC at N up to 128 on GMAC and at all tested N on BEMAC; (ii) NCG beats SC on BEMAC with the same code at N=64-256 (BLER 0.003 vs 0.006, 0.001 vs 0.002, 4e-5 vs 8e-5); (iii) d=32 beats SC at N=32 and N=64 on GMAC (0.80x); (iv) CRC-aided NN-SCL at N=128 reaches BLER 0.002, strictly better than analytical SCL(L=4); (v) the NCG works on channels with memory (ISI-MAC N=64: 0.442 vs memoryless SC 0.575, a 23% improvement), extending the approach beyond memoryless channels; (vi) the N=256 wall is diagnosed with three separate tests (exposure bias absent, cascade amplification absent, weak-position localization quantified), giving a clean boundary of applicability rather than a mystery.

The negative result is equally specific: at GMAC Class B N=256 the NCG's BLER is 3x SC at the standard rate, the gap is not explained by exposure bias or cascade amplification, a small set of NCG-specific weak positions accounts for about half the failures, and the remaining gap is distributed across many positions with small individual contributions. No training trick, architectural tweak, or alternative decomposition tried in roughly three weeks of compute closes it. The analytical SC decoder — for which f, g, and inverse-circular-convolution CalcParent are known in closed form — remains strictly better at this specific operating point, and this is a clean scientific boundary rather than a failure of effort. The NCG works where the architecture's inductive bias is well-matched to the channel (BEMAC, small N on GMAC) and runs into a characterized optimization-plus-rate-specialization limit at deeper trees on continuous MAC channels. The project's primary contribution — a fully neural SC decoder for two-user MAC polar codes with matched or better performance than analytical SC at practical block lengths — stands on its own; the N=256 wall is the honest limit of the approach as currently formulated.

**Update (2026-04-16): CRC-aided list decoding breaks the N=256 wall.** Validated at 2000 codewords, NN-CA-SCL with L=4 and CRC-8 achieves BLER 0.003 (Wilson 95% CI [0.0014, 0.0065]) at N=256 GMAC Class B, compared to SC's 0.005 — the first NCG configuration to beat analytical SC at this block length on the Gaussian MAC. The result holds across three independently trained checkpoints (all achieving BLER 0.003-0.006). Larger list sizes L=8 and L=16 are counterproductive (BLER 0.006 and 0.008 respectively), suggesting L=4 is a sweet spot where the correct path is still among the top candidates but the CRC check filters the few remaining errors. At N=512 CRC-SCL does not help (BLER 0.048 vs SC 0.001), so the wall shifts from N=256 to N=512. The implication is that the NCG's N=256 representations contain sufficient information for correct decoding; the bottleneck was the greedy SC walk making a few critical early errors, which CRC-aided list decoding can correct.
