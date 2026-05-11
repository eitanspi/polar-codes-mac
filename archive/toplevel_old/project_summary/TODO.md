# TODO List

## Status Summary (as of 2026-04-16)

**Completed:** 52 items across 12 sessions
**Remaining:** 4 future work items, 3 needed before submission

---

## COMPLETED ITEMS (Sessions 1-12)

### Memoryless Channels
- [x] CRC-SCL L=4 sweep across all channels/classes/N (GMAC/BEMAC/ABNMAC x B/C x N=8..512)
- [x] BEMAC Class B: SC reliable at N=32-128, NCG reliable at N=32-128
- [x] BEMAC Class C: rate mismatch fixed, CW counts recorded, NPD N=64/128 retrained
- [x] GMAC Class B: SC reliable at N=32-128, NCG reliable at N=32-1024, wall characterised
- [x] GMAC Class C: NPD/NCG reliable at N=16-64, SC reliable at N=16-64, frozen set co-adaptation analysed
- [x] ABNMAC Class B: SC and NCG reliable at N=8-128, non-monotonicity confirmed
- [x] ABNMAC Class C: SC reliable at N=32-128, non-monotonicity confirmed, neural decoder incompatible (tuple output)

### Memory Channels
- [x] ISI-MAC: chained trellis SC baselines at N=16-1024 (10K CW, reliable)
- [x] ISI-MAC: NPD d=16 h=64 at N=16-64 (5K CW, reliable)
- [x] ISI-MAC: NPD d=16 h=100 standalone at N=64 (BLER=0.032, beats chained trellis 0.041)
- [x] ISI-MAC: NPD d=16 h=100 standalone at N=128 (BLER=0.081)
- [x] ISI-MAC: NPD d=64 h=128 at N=128 (BLER=0.030) and N=256 (BLER=0.011)
- [x] ISI-MAC: GPU curriculum d=16 h=100 at N=16-64 (HEADLINE: beats trellis by 19-32%)
- [x] ISI-MAC: multi-SNR sweep at N=16-64
- [x] ISI-MAC: multi-h robustness sweep at N=32
- [x] ISI-MAC: first-error analysis at N=64 and N=128
- [x] ISI-MAC: rate-1 MI measurement at N=256 and N=512
- [x] Ising MAC: trellis SC baselines at N=16-64
- [x] Ising MAC: NPD d=16 h=100 at N=16-32
- [x] MA-AGN: memoryless SC baselines at N=16-128
- [x] MA-AGN: NPD d=32 h=128 at N=16-64
- [x] MA-AGN: NPD d=16 h=100 at N=64 (BLER=0.029-0.035, matches memoryless SC)

### Analysis and Documentation
- [x] NPD paper ISI recipe analysis (NPD_PAPER_ISI_RECIPE.md)
- [x] N=256 NCG wall diagnostics (NCG_CHAPTER.md Section 5-6)
- [x] ISI-MAC N=512 wall diagnostics (WALL_DIAGNOSTICS.md)
- [x] Frozen set co-adaptation analysis (GMAC_CORNER_NPD_VERIFICATION.md)
- [x] BLER_TABLES.md (Tables 1-9, all channels)
- [x] PAPER_STYLE_TABLES.md (Tables A-C, memory channels)
- [x] MEMORY_MAC_CHAPTER.md (thesis chapter draft)
- [x] NCG_CHAPTER.md (thesis chapter draft)
- [x] WALL_DIAGNOSTICS.md (comprehensive wall analysis)
- [x] MULTI_CHANNEL_COMPARISON.md (cross-channel table)
- [x] FINAL_RESULTS.md (master results with ongoing training status)
- [x] MASTER_RESULTS.md (thesis-ready single source of truth)
- [x] THESIS_STORY.md (2000-word narrative)
- [x] CONSISTENCY_CHECK.md (cross-document verification)
- [x] All 7 thesis figures generated (fig_bemac_results, fig_gmac_results, fig_abnmac_results, fig_isi_mac_final, fig_memory_channels, fig_architecture_scaling, fig_wall_analysis_thesis)

---

## NEEDED BEFORE SUBMISSION (3 items)

- [ ] **MEMORY_MAC_CHAPTER.md refresh:** Update Sections 4.3 and 4.6 with GPU curriculum numbers and d=16 h=100 results. Currently uses older d=16 h=64 numbers for ISI-MAC N=64 (0.049 instead of 0.028).

- [ ] **MA-AGN N=64 eval discrepancy:** Two eval runs give BLER 0.029 and 0.035. Run a third 5K CW eval to settle the true value. Affects whether the thesis can claim "NPD matches memoryless SC at N=64."

- [ ] **BEMAC Class C N=1024 SC baseline:** Currently 26/63550 errors (unreliable). Need ~250K CW for 100 errors. Low priority since NCG clearly dominates at this N.

---

## FUTURE WORK (4 items)

- [ ] **ISI-MAC N=512 direct training:** Train d=64 model from scratch at N=512 with 500K+ iterations on GPU. The N=256-trained model does not generalise (25% catastrophic positions). This is the most important open experiment.

- [ ] **ISI-MAC N=1024:** Would require d>=128 model based on MI analysis. Multi-day GPU training. Research-grade effort.

- [ ] **ABNMAC Class C neural decoder:** Requires z_dim=2 encoder modification (ABNMAC outputs (zx, zy) tuples). Engineering effort, not a research blocker.

- [ ] **Class B rate optimisation:** Verify optimal rate pair for path N//2 on each channel. Current symmetric rates (ku=kv) may be suboptimal.

---

## SESSION LOG

### Session 12 (2026-04-24) -- Ising MAC, MA-AGN extension, wall diagnostics
- Ising MAC channel implementation and baselines
- Ising MAC NPD d=16 h=100 at N=16,32
- MA-AGN d=16 h=100 at N=64 (46% improvement over d=32 h=128)
- WALL_DIAGNOSTICS.md, MULTI_CHANNEL_COMPARISON.md
- BLER_TABLES.md Tables 8-9

### Session 11 (2026-04-16/17) -- d=16 h=100, chained trellis baselines
- Chained trellis SC baselines (10K CW) establishing fair comparison
- d=16 h=100 BREAKTHROUGH at N=64 (matches joint trellis SC)
- d=16 h=100 at N=128 (0.081, better than d=16 h=64)
- Figure regeneration with all decoder variants

### Session 10 (2026-04-16) -- ISI-MAC analysis and reliable evals
- Reliable 5K CW evals correcting old 2K CW numbers
- First-error analysis showing error distribution shift with N
- Rate-1 MI measurement confirming N=512 out-of-distribution failure
- NPD paper recipe analysis

### Sessions 1-9 -- Foundation
- NCG decoder for memoryless MAC (BEMAC, GMAC, ABNMAC)
- Chained NPD for corner-rate MAC
- CRC-aided neural SCL breakthrough at N=256
- GMAC campaign (824 results)
- Neural SCL, inference tricks, GPU setup
