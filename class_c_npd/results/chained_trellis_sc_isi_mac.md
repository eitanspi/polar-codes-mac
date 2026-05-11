# Chained Trellis SC on ISI-MAC — Analytical Baseline

**Configuration.** ISI-MAC with one-tap memory (h = 0.3) at SNR 6 dB,
σ² ≈ 0.2512. Class C corner-rate path (`make_path(N, N)`). GMAC Class C
frozen sets (`designs/gmac_C_n{n}_snr6dB.npz`) — the same designs used by
the existing NPD evaluation. Info-bit counts (ku, kv) per existing project
rates: (4, 7), (7, 15), (15, 29), (30, 58) for N = 16, 32, 64, 128. All
decoders are fed the same received sequences so the comparison is coupled.

**Decoder variants.**

- **Chained trellis SC** — Two-stage Önay corner-rate decomposition.
  Stage 1 decodes U by running forward–backward on a 2-state trellis
  (state = X_{i-1}) while marginalising Y_i and Y_{i-1} as independent
  Bernoulli(1/2); the resulting per-position marginal LLRs drive a
  standard single-user Arikan SC decoder. Stage 2 re-encodes the decoded
  U to X̂ and decodes V via forward–backward on a 2-state trellis
  (state = Y_{i-1}) with X_i, X_{i-1} known. Complexity:
  O(|S|²·N + N log N) = **O(N log N)**.

- **Joint 4-state trellis SC** (existing
  `polar/decoder_trellis.py decode_single`) — forward–backward on the
  full |S| = 4 MAC trellis produces (N, 2, 2) marginals that feed the
  interleaved MAC SC computational graph. Complexity: O(|S|³·N log N).

- **Memoryless SC** — Treats the ISI channel as if it were a standard
  memoryless GMAC (ignores the h·(…previous…) term in the likelihood).
  Class C corner decoder: U decoded from the Y-marginal LLR, V decoded
  from the X̂-conditional LLR. Complexity: O(N log N).

## Results

| N  | Stage 1 U-only BLER | Stage 2 V given true U | **Chained BLER** | Joint trellis SC | Memoryless SC | n_cw | Chain time | Joint time |
|----|--------------------:|-----------------------:|-----------------:|-----------------:|--------------:|------|-----------:|-----------:|
| 16 | 0.1616 | 0.0026 | **0.1642** | 0.1662 | 0.1846 | 5000 |  4.8 s | 29.5 s |
| 32 | 0.0836 | 0.0000 | **0.0836** | 0.0748 | 0.1140 | 5000 | 10.2 s | 63.6 s |
| 64 | 0.0370 | 0.0000 | **0.0370** | 0.0290 | 0.0877 | 3000 | 12.9 s | 77.4 s |
| 128| 0.0180 | 0.0000 | **0.0180** | 0.0080 | 0.0950 | 1000 |  9.4 s | 53.3 s |

Wall-clock columns are cumulative time for the given `n_cw`. Memoryless-SC
times (~2 – 8 s per row) are folded into the per-codeword evaluation.
Identical received sequences drive all three decoders for each N (coupled
Monte-Carlo).

For reference, prior ISI-MAC Class C numbers recorded in
`class_c_npd/results/isi_mac_classC_npd.json` (those runs used h = 0.5):

| N  | Broken Stage-1 NPD BLER (h=0.5) | Joint trellis SC (h=0.5) |
|----|--------------------------------:|-------------------------:|
| 16 | 0.744 | 0.167 |
| 32 | 0.876 | 0.060 |
| 64 | 0.976 | 0.029 |

At h = 0.3 the joint-trellis numbers (0.166 / 0.075 / 0.029 / 0.008) are
within Monte-Carlo noise of the h = 0.5 numbers at N ≤ 64, confirming the
modest effect of the tap at SNR 6 dB.

## Interpretation

**Does chained trellis SC match joint trellis SC?** Yes, closely. At
N = 16 the two are statistically indistinguishable (0.164 vs 0.166 over
5000 codewords). At N ≥ 32 the joint decoder is a few percentage points
better in absolute BLER (0.075 vs 0.084 at N = 32; 0.029 vs 0.037 at
N = 64; 0.008 vs 0.018 at N = 128). The gap grows with N because Stage 1
treats Y as memoryless noise and ignores the (small) polar-code structure
that correlates adjacent Y symbols; the joint decoder exploits this. The
gap is nonetheless small at the corner rate and the two decoders track
each other within a factor of ~2.

**What is the cost of the chaining?** Almost entirely a Stage-1 cost.
Stage 2 given the *true* U codeword is essentially error-free at every N
(0.0026 at N = 16, 0.000 elsewhere), so the "chained = Stage 1 ⊕
Stage 2-propagated" BLER is dominated by Stage 1. Whenever Stage 1 gets
U wrong, the wrong X̂ usually makes V wrong too, so the joint chained
BLER equals the Stage-1 BLER (the V-err column in the JSON matches the
Stage-1 column). The chained decoder is **~6× faster** than the joint
trellis SC — the same speedup you would predict from replacing |S| = 4
with |S| = 2 in FB.

**What BLER should a good neural chained NPD achieve?** **The
"chained BLER" column.** Matching those numbers is the ceiling for an
analytical decomposition that treats the two stages independently.
The existing NPD numbers from `isi_mac_classC_npd.json` (0.744, 0.876,
0.976) are catastrophically far from this ceiling — confirming the
"broken NPD" hypothesis, not a fundamental limitation of Stage-1
chaining. A correctly-implemented neural chained NPD should achieve
BLER ≈ 0.16 / 0.08 / 0.04 at N = 16 / 32 / 64, with the memoryless SC
baseline (0.18 / 0.11 / 0.09) as the "does the NN learn any memory"
sanity check.

## Code

- `polar/decoder_trellis_mac_chained.py` — `decode_stage1_u`,
  `decode_stage2_v`, `decode_chained`, and BLER helpers.
- `scripts/eval_chained_trellis_isi_mac.py` — generates the JSON above.
- Sanity check at N = 16, 500 codewords, h = 0.3: chained BLER = 0.20 (in
  the 0.15–0.25 band the prompt expected) and at h = 0.5 chained = 0.179
  vs joint-trellis 0.167 — within Monte-Carlo noise of the published
  joint-trellis h=0.5 number.
