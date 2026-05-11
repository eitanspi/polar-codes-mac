# Consistency Check: Cross-Document Number Verification

**Date:** 2026-04-16
**Method:** Cross-referenced BLER_TABLES.md, PAPER_STYLE_TABLES.md, FINAL_RESULTS.md, MEMORY_MAC_CHAPTER.md, NCG_CHAPTER.md, WALL_DIAGNOSTICS.md, and authoritative JSON sources.

---

## Status: 6 inconsistencies found, all explained. No data integrity issues.

---

## 1. CONSISTENT (no issues)

### ISI-MAC Chained Trellis SC baselines
- BLER_TABLES.md Table 7, PAPER_STYLE_TABLES.md Table A, FINAL_RESULTS.md Section 1A, and `results/paper_style/isi_mac_sc_baselines.json` all agree:
  - N=16: 0.1689, N=32: 0.0822, N=64: 0.0407, N=128: 0.0223, N=256: 0.0061, N=512: 0.0041, N=1024: 0.0074

### ISI-MAC Memoryless SC baselines
- All sources agree: N=16: 0.1866, N=32: 0.1129, N=64: 0.0790, N=128: 0.1009, N=256: 0.2256

### ISI-MAC GPU curriculum NPD (best results)
- PAPER_STYLE_TABLES.md Table A and `results/paper_style/npd_paper_style_evals.json` agree:
  - N=16: 0.1376, N=32: 0.0566, N=64: 0.0278

### Ising MAC baselines
- BLER_TABLES.md Table 8, PAPER_STYLE_TABLES.md Table B, and `results/paper_style/ising_mac_baselines.json` all agree:
  - Trellis: N=16: 0.5704, N=32: 0.6872, N=64: 0.8976
  - Memoryless: N=16: 0.6160, N=32: 0.7840, N=64: 0.9410

### Ising MAC NPD
- BLER_TABLES.md Table 8 (0.592, 0.770) and JSON (0.5916, 0.7658): within rounding. OK.

### MA-AGN SC baselines
- All sources agree: N=16: 0.1654, N=32: 0.0696, N=64: 0.0292, N=128: 0.0052

### GMAC, BEMAC, ABNMAC memoryless tables
- BLER_TABLES.md Tables 1-6 and PAPER_STYLE_TABLES.md are internally consistent.
- CRC-SCL numbers from `results/crc_scl_sweep/` match BLER_TABLES.md entries.

---

## 2. INCONSISTENCIES FOUND

### Issue 1: ISI-MAC N=16 d=16 h=64 BiGRU NPD -- 0.143 vs 0.138
- **BLER_TABLES.md Table 7:** 0.143 (715/5000) from `results/reliable_evals/isi_mac_npd_reliable.json`
- **npd_all_channels_5kcw.json:** 0.1384 (692/5000) from a different eval run
- **Explanation:** Two different evaluation runs with different random seeds. Both are within each other's Wilson 95% CI [0.129, 0.148]. Not a data integrity issue.
- **Resolution:** Use 0.143 from reliable_evals (the dedicated reliable eval run) in BLER_TABLES. Use 0.1376 from GPU curriculum (the best model) in PAPER_STYLE_TABLES and MASTER_RESULTS.
- **Severity:** Low. Different models being compared (bigru L1 vs GPU curriculum).

### Issue 2: ISI-MAC N=32 best NPD -- BiGRU (0.113) vs Window (0.081)
- **BLER_TABLES.md Table 7:** 0.081 (406/5000) using **window_w2** encoder
- **npd_all_channels_5kcw.json:** 0.113 (565/5000) using **bigru L1** encoder
- **PAPER_STYLE_TABLES.md:** 0.0566 using **GPU curriculum d=16 h=100**
- **Explanation:** Three different models. Table 7 uses the original window model; the JSON has the bigru L1; GPU curriculum is the best. Not an error -- different architectures.
- **Resolution:** MASTER_RESULTS should use GPU curriculum (0.0566) as "best NPD" and note the window model in BLER_TABLES as a historical comparison. Table 7 mixes model families; PAPER_STYLE_TABLES correctly uses the latest best.
- **Severity:** Medium. Table 7 is inconsistent with PAPER_STYLE_TABLES about which model is "best" at N=32.

### Issue 3: ISI-MAC N=128 d=64 -- 0.030 vs 0.089
- **BLER_TABLES.md Table 7:** 0.030 (150/5000) from `results/reliable_evals/isi_mac_npd_reliable.json`
- **npd_all_channels_5kcw.json (isi_d64_cont_N128):** 0.0894 (447/5000)
- **Explanation:** Different checkpoints! The reliable_evals entry uses `d64_lr1e3_N128_final.pt` (the best d=64 checkpoint after extended training). The npd_all_channels entry uses `isi_mac_bigru_L1_cont_d64_s1_N128_best.pt` (an earlier checkpoint from a different training run). The 0.030 number is from the better-trained model.
- **Resolution:** The BLER_TABLES 0.030 is the correct authoritative number. The npd_all_channels JSON has stale data for this entry. PAPER_STYLE_TABLES correctly shows 0.0740 for the "CPU standalone" d=16 h=100 model (which is different from d=64).
- **Severity:** High. The npd_all_channels JSON is stale for N=128 d=64. MASTER_RESULTS uses 0.030 from the authoritative source.

### Issue 4: MA-AGN N=64 d=16 h=100 -- 0.035 vs 0.029
- **BLER_TABLES.md Table 9:** 0.035 (177/5000) from `maagn_d16h100_results.json`
- **npd_all_channels_5kcw.json:** 0.0292 (146/5000)
- **PAPER_STYLE_TABLES.md Table C:** 0.0292
- **Explanation:** Two different evaluation runs of the same model with different random seeds. The 0.035 eval has 177 errors; the 0.029 eval has 146 errors. Both are plausible given the Wilson 95% CI width at 5000 CW.
- **Resolution:** Use the average of both runs conceptually, but report the paper_style number (0.029) as the more recent eval. Flag the discrepancy. The "NPD matches memoryless SC at N=64" claim in PAPER_STYLE_TABLES (both at 0.029) may be overstated -- the BLER_TABLES eval (0.035) suggests NPD is 1.2x worse.
- **Severity:** Medium. Affects the N=64 MA-AGN narrative. The truth is likely between 0.029 and 0.035 (NPD/SC = 1.0x to 1.2x).

### Issue 5: MEMORY_MAC_CHAPTER.md Section 4.3 uses older ISI-MAC numbers
- **MEMORY_MAC_CHAPTER.md:** Reports N=64 NPD as 0.0489 (from 10K CW audit), chained trellis as 0.0399
- **Latest data:** GPU curriculum achieves 0.0278, chained trellis is 0.0407 (from 10K CW paper_style baselines)
- **Explanation:** The chapter was written before the GPU curriculum results and d=16 h=100 breakthrough.
- **Resolution:** Chapter needs update with latest numbers. The headline should be 0.028 (GPU curriculum) not 0.049 (old d=16 h=64).
- **Severity:** Medium. Chapter is outdated but not wrong for the model it describes.

### Issue 6: NCG_CHAPTER.md Section 4 GMAC Class B numbers
- **NCG_CHAPTER.md:** Reports N=64 NCG BLER as 0.026, SC as 0.025
- **BLER_TABLES.md Table 3:** Reports N=64 NCG as 0.0282, SC as 0.0276
- **Explanation:** NCG_CHAPTER uses rounded values from earlier evals. BLER_TABLES has precise reliable eval numbers.
- **Resolution:** Minor rounding difference. Both tell the same story (NCG matches SC at N=64).
- **Severity:** Low.

---

## 3. WALL_DIAGNOSTICS.md Consistency

- ISI-MAC wall table (Section 1.2) uses 0.032 for N=64 d=16 h=100, matching the reliable_evals source (0.0322).
- GMAC wall table (Section 2.2) uses 0.028 for N=64 NCG, consistent with BLER_TABLES (0.0282).
- All wall ratios are correctly computed from the stated numbers.

---

## 4. Recommendations

1. **BLER_TABLES.md Table 7** should add the GPU curriculum numbers as the primary "best NPD" entries, with historical models (window, bigru d=16 h=64) in a separate column or footnote.

2. **npd_all_channels_5kcw.json** is stale for ISI-MAC N=128 d=64. Should be updated from `results/reliable_evals/isi_mac_npd_reliable.json`.

3. **MEMORY_MAC_CHAPTER.md** should be refreshed with GPU curriculum and d=16 h=100 numbers.

4. **MA-AGN N=64** narrative should acknowledge the two-eval spread (0.029-0.035) and not claim exact parity with memoryless SC.

5. For thesis submission, use the MASTER_RESULTS.md table as the single source of truth, with authoritative sources listed per entry.
