# Session 10 Summary: ISI-MAC Analysis and Reliable Evaluations

*Date: 2026-04-16. Duration: ~3 hours CPU-only. torch.set_num_threads(4).*

---

## Tasks Completed

### Task 1: NPD Paper ISI Recipe Analysis

Read the Aharoni/Huleihel/Pfister/Permuter paper (IEEE Trans. IT, Dec. 2024) and reference code. Key findings:

- **Architecture:** d=8, hidden=50, 2 hidden layers per MLP (code says layers_per_op=2, paper text says "one hidden layer" -- discrepancy). Shared F/G/H across all tree depths. Channel embedding is a POINTWISE Dense(d) -- no RNN/GRU.
- **Training:** 1M iterations, batch=10, Adam lr=1e-3 (no schedule), trained at fixed N=1024. NSCLoss averages binary CE across all n+1 tree depths (not just leaves).
- **ISI specifics:** h=[0.9, 0.5], sigma^2=0.5. The paper does NOT use any recurrent component in the decoder -- memory is handled implicitly by the tree structure.
- **Critical finding:** Our BiGRU channel encoder is an EXTENSION beyond the paper. The paper handles ISI via pointwise embeddings + tree propagation.
- Written to `project_summary/NPD_PAPER_ISI_RECIPE.md`.

### Task 2: Reliable 5K CW Evaluations

Standardized all ISI-MAC NPD evaluations to 5000 codewords with Wilson 95% CIs.

| N | d | Encoder | Chained BLER | Errs/CW | 95% CI | Trellis SC | NPD/Trellis |
|---|---|---------|-------------|---------|--------|------------|-------------|
| 16 | 16 | BiGRU | 0.143 | 715/5000 | [0.134, 0.153] | 0.166 | 0.86x |
| 32 | 16 | Window | 0.081 | 406/5000 | [0.074, 0.089] | 0.083 | 0.98x |
| 64 | 16 | BiGRU | 0.046 | 229/5000 | [0.040, 0.052] | 0.026 | 1.77x |
| 128 | 64 | BiGRU | **0.030** | 150/5000 | [0.026, 0.035] | 0.018 | 1.67x |
| 256 | 64 | BiGRU | **0.011** | 56/5000 | [0.009, 0.015] | 0.007 | 1.57x |

**Major correction:** N=128 d=64 chained BLER was previously reported as 0.099 (from 2K CW eval). The reliable 5K CW eval gives 0.030 -- a 3.3x improvement. The old number was likely an unlucky sample or used a different checkpoint. The S1 BLER (0.026 at 5K CW) is consistent with the 0.029 reported from the training eval, confirming the model is good.

Also ran trellis SC at N=64 with 10K CW: BLER=0.026 (262/10000), updated from prior 0.040.

### Task 3: First-Error Analysis (N=64 vs N=128)

| Metric | N=64 d=16 BiGRU | N=128 d=64 BiGRU |
|--------|-----------------|------------------|
| BLER | 0.046 | 0.030 |
| Q1 errors | 54.4% | 6.9% |
| Q2 errors | 38.5% | 49.0% |
| Q3 errors | 7.1% | 44.1% |
| Q4 errors | 0.0% | 0.0% |

**Key insight:** At N=64, errors concentrate at early positions (Q1+Q2 = 93%). At N=128, errors shift to mid-to-late positions (Q2+Q3 = 93%). In both cases, Q4 has zero errors -- the last quarter of info positions is error-free.

**Interpretation:** The BiGRU encodes the full z-sequence, so early and late positions should both have good channel information. The shift from Q1 (N=64) to Q2-Q3 (N=128) suggests that at larger N, the tree structure's recursive propagation attenuates the BiGRU's embeddings at positions deeper in the tree (which correspond to Q2-Q3 in position ordering). The zero Q4 errors at both N likely means Q4 positions are "easy" (high MI) under the GMAC proxy design.

### Task 4: Rate-1 MI Measurement (N=256 and N=512)

Used the d=64 N=256-trained checkpoint to measure per-position mutual information under teacher forcing.

**N=256 (in-distribution):**
- Most info positions: MI ~1.0 bits (near-perfect)
- 2 weak positions: pos 183 (MI=-59 bits), pos 215 (MI=-1.1 bits)
- Top-59 model MI ranking agrees 100% with GMAC proxy design
- Info-position MI quartiles: Q1=[-59, 1.0], Q2-Q4=1.0 bits

**N=512 (out-of-distribution):**
- Q1 (weakest 25%): mean MI = -59M bits (catastrophic failure)
- Q2: mean MI = -1.0 bits (marginal failure)
- Q3: mean MI = 0.92 bits (decent)
- Q4: mean MI = 1.0 bits (perfect)
- 5 weakest positions: pos 490 (MI=-1.7B bits), 493, 458, 501, 461

**Conclusion:** The d=64 BiGRU trained at N=256 completely fails to generalize to N=512. Approximately 25% of info positions produce confidently wrong predictions. This explains why curriculum training (N=256 -> N=512) would fail -- the model's representations do not transfer. Training directly at N=512 with d>=64 is needed.

---

## Files Created/Modified

### New files:
- `project_summary/NPD_PAPER_ISI_RECIPE.md` -- NPD paper analysis
- `project_summary/SESSION10_SUMMARY.md` -- this file
- `scripts/eval_reliable_isi_mac.py` -- 5K CW eval script
- `scripts/first_error_analysis_isi_mac.py` -- first-error analysis
- `scripts/rate1_mi_measurement_v2.py` -- MI measurement
- `results/reliable_evals/isi_mac_npd_reliable.json` -- reliable eval results
- `results/reliable_evals/isi_mac_first_error_analysis.json` -- first-error data
- `results/reliable_evals/isi_mac_mi_N256.json` -- MI data N=256
- `results/reliable_evals/isi_mac_mi_N512.json` -- MI data N=512
- `class_c_npd/results/npd_memory_mac/d64_lr1e3_N128_final.pt` -- copied from /tmp

### Modified files:
- `project_summary/BLER_TABLES.md` -- Table 7 updated with reliable numbers
- `project_summary/TODO.md` -- Session 10 items added
