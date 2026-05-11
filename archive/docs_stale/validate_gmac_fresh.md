# Fresh GMAC Validation: Neural SC vs Analytical SC vs Analytical SCL

**Run date:** 2026-04-09
**Seed:** 42
**SNR:** 6.0 dB
**Channel:** GaussianMAC, σ² = 0.2512
**Code class:** B symmetric (Ru ≈ Rv ≈ 0.48)
**Checkpoints:** `saved_models/ncg_gmac_mlp_N{32,64,128,256}.pt` (d=16, hidden=64)

**Scripts:**
- `scripts/validate_gmac_fresh.py` — paired BLER evaluation
- `scripts/analyze_n256_failures.py` — noise-level analysis of N=256 NN-only failures
- `scripts/n256_perbit_analysis.py` — per-info-bit error count analysis

**Raw data:**
- `results/validate_gmac_fresh.json`
- `results/n256_failure_analysis.json`
- `results/n256_failure_indices.npz`
- `results/n256_perbit_counts.json`

## Why this run exists

Previous BLER numbers for the GMAC neural decoder were spread across
several result files from different training runs and different MC
evaluations, so the neural and analytical decoders never saw the same
test stream. This made it hard to tell how much of the gap came from
MC noise versus real architectural weakness.

The three scripts above generate a single fixed codeword set with a
fixed seed and evaluate every decoder on the **exact same codewords and
the exact same channel noise**. Head-to-head columns (`NN only`, `SC
only`, `SCL only`) are therefore genuine — they are not an artifact of
independent Monte Carlo draws.

## Headline results

| N   | n_cw  | NN BLER | SC BLER | SCL BLER | NN/SC   | NN/SCL    |
|-----|-------|---------|---------|----------|---------|-----------|
| 32  | 5000  | 0.0426  | 0.0390  | 0.0220   | 1.09×   | 1.94×     |
| 64  | 5000  | 0.0290  | 0.0228  | 0.0096   | 1.27×   | 3.02×     |
| 128 | 5000  | 0.0224  | 0.0182  | 0.0070   | 1.23×   | 3.20×     |
| 256 | 10000 | 0.0208  | 0.0042  | **0.0009** | **4.95×** | **23.1×** |

Paired head-to-head (same codewords, same noise):

| N   | NN errs | SC errs | SCL errs | NN only vs SC | SC only vs NN | NN only vs SCL | SCL only vs NN |
|-----|---------|---------|----------|---------------|---------------|----------------|----------------|
| 32  | 213     | 195     | 110      | 62            | 44            | 143            | 40             |
| 64  | 145     | 114     | 48       | 54            | 23            | 108            | 11             |
| 128 | 112     | 91      | 35       | 49            | 28            | 92             | 15             |
| 256 | 208     | 42      | 9        | 180           | 14            | 204            | 5              |

`NN only vs SCL` = codewords NN got wrong but SCL got right.
`SCL only vs NN` = codewords SCL got wrong but NN got right.

The neural SC decoder loses to SCL on 40–140 more codewords than it wins
at every tested N. At N=256 the asymmetry is extreme: the NN loses 204
codewords to SCL but only wins 5. The claim "neural decoder competitive
with SCL at small N" is not supported on paired codewords — even at
N=32 the NN is roughly 2× worse than SCL L=4.

## Comparison to previously reported numbers

Prior (from `results/gmac_snr6dB/nn_scl_full_comparison.json`, 2000 cw
per N, independent MC runs, no paired comparison):

| N   | prior NN | prior SC | prior NN/SC | prior SCL | fresh NN | fresh SC | fresh NN/SC | fresh SCL |
|-----|----------|----------|-------------|-----------|----------|----------|-------------|-----------|
| 32  | 0.0470   | 0.0500   | 0.94×       | 0.0230    | 0.0426   | 0.0390   | 1.09×       | 0.0220    |
| 64  | 0.0285   | 0.0235   | 1.21×       | 0.0100    | 0.0290   | 0.0228   | 1.27×       | 0.0096    |
| 128 | 0.0175   | 0.0135   | 1.30×       | 0.0040    | 0.0224   | 0.0182   | 1.23×       | 0.0070    |
| 256 | 0.0155   | 0.0070   | 2.21×       | 0.0010    | 0.0208   | 0.0042   | **4.95×**   | 0.0009    |

**Things that reproduced well:** SCL numbers at N=32, 64, 256. SC and NN
at N=64. These are within MC noise of the prior claims.

**Discrepancies flagged:**

1. **N=256 NN/SC gap is ~5× SC, not ~2× SC.** The widely quoted "2.2× SC"
   figure was measured on 2000 codewords against a separately-run SC
   baseline. On 10 000 paired codewords the SC BLER is actually 0.0042
   (better than the old 0.007) and the NN BLER is 0.0208 (worse than
   the old 0.0155). Both moved the wrong way. Any paper/thesis claim
   of "within 2× of SC at N=256" should be revised to **~5× SC**.

2. **The "NN beats SC at N=32" claim does not reproduce.** Prior was
   0.94×; fresh paired is 1.09×. It was within MC noise of parity.
   On 5000 paired codewords the NN is very slightly worse than SC, not
   better.

3. **SCL at N=128 is 0.0070, not the previously reported 0.0040.**
   This one is flipped in the NN's favour compared to the old run —
   SCL's old 0.0040 was probably a lucky MC draw on 1000 codewords.
   With 5000 paired codewords we see 35 errors, giving a tighter
   estimate. The NN/SCL ratio at N=128 is 3.2×, not 4.4× as the old
   numbers would imply.

4. **NN is ~2× worse than SCL even at N=32.** The old number suggested
   the NN was competitive (0.047 vs 0.023 = 2.04×, which we confirm).
   The paired version shows this is a real gap, not MC noise — at
   every N the neural decoder is 1.9× to 23× behind SCL.

## N=256 failure-mode analysis

Among the 10 000 paired codewords at N=256:

- **both OK:** 9778 (97.78%)
- **both err:** 28  (0.28%)
- **NN only:** 180  (1.80%)  — NN fails, SC succeeds
- **SC only:** 14   (0.14%)  — SC fails, NN succeeds

For each group we computed `||W||² / N`, the per-symbol noise energy.
The expected value for a clean GMAC at this SNR is σ² = 0.2512.

| group     | n     | mean noise/sym | median  | p10     | p90     |
|-----------|-------|----------------|---------|---------|---------|
| both_ok   | 9778  | 0.2506         | 0.2500  | 0.2233  | 0.2794  |
| both_err  | 28    | 0.2792         | 0.2846  | 0.2446  | 0.3102  |
| nn_only   | 180   | 0.2649         | 0.2644  | 0.2391  | 0.2916  |
| sc_only   | 14    | 0.2602         | 0.2715  | 0.2280  | 0.2834  |

**Interpretation.** The `both_err` group is genuinely
high-noise (+11% above σ², ~1.3σ tail). The `nn_only` group sits
**between** `both_ok` and `both_err`: +5.5% noise, only ~0.65σ above
the mean. Roughly 90% of `nn_only` failures happen on channel
realizations that are normal enough that SC has no trouble with them.

This rules out the hypothesis that the NN only fails on catastrophic
noise realizations. The NN is losing codewords that are only slightly
harder than average. In effect: the NN has **about half the noise
margin that SC has** at N=256.

## Per-bit error distribution at N=256

Total bit-error counts over the 10 000 codewords:

- **NN total bit errors:** 7577
- **SC total bit errors:** 1607
- **ratio:** 4.71× (consistent with the 4.95× BLER ratio)

Concentration of bit errors on the top-10 "worst" info positions:

| decoder | top-10 fraction of its bit errors |
|---------|------------------------------------|
| NN      | 14.9%                              |
| SC      | 15.9%                              |

**Interpretation.** The top-10 fractions are essentially the same. The
NN is **not** failing on a small number of weak positions that SC has
learnt to protect. Its errors are distributed across info positions the
same way SC's are, just roughly 5× more of them everywhere.

This rules out the hypothesis that the NN has a "weak spot" at a
particular tree depth. Combined with the noise-level analysis, the
picture is: **the N=256 neural decoder is a uniformly less margined
version of SC — same error topology, just scaled up.**

## Confidence intervals (rough)

Standard error on BLER with n codewords and m errors ≈ √m / n :

| N   | decoder | BLER   | 95% CI relative |
|-----|---------|--------|-----------------|
| 256 | NN      | 0.0208 | ±14%            |
| 256 | SC      | 0.0042 | ±31%            |
| 256 | SCL     | 0.0009 | ±67%            |

The N=256 SCL BLER is based on 9 errors, which is tight enough to say
"SCL is roughly 5× better than SC" but too loose to distinguish, say,
SCL=0.0007 from SCL=0.0012. If the paper wants a tighter SCL baseline
at N=256, push `n_cw` to 50 000 (~1 hour of CPU).

## What this means for the paper / thesis

1. **The N=256 headline number should be revised** from "~2× SC" to
   "~5× SC", and the corresponding NN-vs-SCL comparison at N=256 is
   **~23× SCL**, which is a much bigger gap than previously implied.

2. **The failure-mode analysis supports the existing diagnosis**
   (curriculum training at depth 8 reaches a worse local optimum than
   analytical SC) rather than pointing to a different root cause.
   The picture "less margin everywhere, no single bad bit or channel
   regime" is consistent with the tree ops being undertrained in a
   uniform way at this depth.

3. **At small N (32, 64, 128) the neural decoder is roughly tied with
   SC but clearly behind SCL.** The prior framing of "competitive
   with SCL at small N" does not hold under paired comparison — the
   gap is a real 2–3× not a noisy tie.

4. **The SCL L=4 analytical decoder is the natural baseline to beat**,
   not SC. At N=256 SCL reaches 0.0009, and the neural decoder would
   need to cut its BLER by ~20× to match it. That is a much more
   honest framing of where the neural approach currently sits.

## Reproducing this run

```bash
# All four N, paired NN vs SC (fast)
python scripts/validate_gmac_fresh.py

# Include SCL L=4 (adds ~15 minutes at N=256)
python scripts/validate_gmac_fresh.py --include-scl

# Single N override
python scripts/validate_gmac_fresh.py --only 256 --n-cw 10000

# N=256 noise-level failure analysis (separate script)
python scripts/analyze_n256_failures.py

# N=256 per-info-bit error analysis (separate script)
python scripts/n256_perbit_analysis.py
```

Everything is deterministic given `SEED = 42`. Both the info-bit RNG
and the channel-noise RNG are seeded. Re-running with the same seed
should reproduce these numbers exactly.
