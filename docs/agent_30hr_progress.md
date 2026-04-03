# 30-Hour Agent Work Session — Progress Report

## Time Elapsed: ~2 hours (started 17:00, current ~19:00)

## Completed Tasks

### Task 1.1: BEMAC Comprehensive Results [COMPLETED]
- Consolidated existing BEMAC Class B results (SC, NN-SC at N=16-1024)
- Ran SCL(L=4) baselines for N=16-256
- Key: NN-SC beats SC at N=64 (0.003 vs 0.006) and N=128 (0.0012 vs 0.002)
- Class C results: high BLER due to model trained for Class B only

### Task 1.2: GMAC Multi-SNR Evaluation [COMPLETED]
- Evaluated NN-SC and SC at SNR=3,4,5,6,7,8 dB for N=64 and N=128
- NN matches SC at SNR=6dB and 8dB
- At lower SNR (3-5dB), NN degrades more (model trained at 6dB)
- SNR=7dB anomalous: missing MC design, GA design fails for Class B

### Task 1.3: Training Complexity Analysis [COMPLETED]
- Model: 38.5K-39K params, 150 KB
- NN ~360x more FLOPs than SC, ~150-190x slower in wall-clock
- Tree operations: ~6N MLP calls per codeword
- Training: 15K-135K iterations (0.3-28 hours)

### Task 1.4: Literature Survey [COMPLETED]
- 356-line comprehensive survey in docs/literature_survey_mac_neural.md
- Covers: single-user neural decoders, MAC decoders, NOMA, autoencoders
- Key finding: no prior work on neural SC tree-walk decoding for MAC

### Task 1.5: Theoretical Analysis [COMPLETED]
- Root cause: ~0.3% error per MLP × 1500 calls = ~1 wrong bit at N=256
- O(N log N) gradient depth prevents fine-grained correction
- BEMAC works at all N because discrete embedding is exact

### Task 3.3: Publication Plots and Tables [COMPLETED]
- 4 PDF/PNG figures: BEMAC BLER vs N, GMAC BLER vs N, waterfall, complexity
- Paper tables in markdown

### Task 3.4: Paper Draft Outline [COMPLETED]
- Full IEEE-format outline (7 sections + appendices)
- ~6 pages of structured content plan

## In-Progress Tasks

### Task 2.2: ISI-MAC Channel with Memory [IN PROGRESS — LEARNING!]
- Created ISIMACNeuralDecoder with sliding window z_encoder
- Training at N=32: BLER dropped from 1.0 → 0.676
- **Beats memoryless SC (0.731)** — neural decoder learns channel memory!
- Training continues to 20K iterations

### Task 3.1: Extended GMAC Training N=512 [IN PROGRESS]
- Training at 45K/100K iterations (15+ hours elapsed from previous session)
- Best BLER: 0.008 (8x SC target of 0.001)
- BLER trend: 0.030 → 0.016 → 0.012 → 0.010

### Task 3.2: CRC-Aided NN-SCL Results [IN PROGRESS]
- N=32: NN-SCL(L=4) BLER=0.023, NN-CA-SCL(L=4) BLER=0.009
- N=64: NN-SCL(L=4) BLER=0.017, NN-CA-SCL(L=4) running (0/200 errors so far!)
- N=128: Pending

### Task 2.3: DINE/MINE Unknown Channel [IN PROGRESS]
- Training z_encoder via MINE contrastive objective on GMAC N=32
- Process running, output buffered

### Task 2.1: ABNMAC Neural Decoder [IN PROGRESS — STUCK]
- Loss decreasing slowly (1.39 → 0.84) but BLER stuck at 1.0
- Tried transfer learning from BEMAC — testing
- The ABNMAC's correlated noise structure may require different approach

## Key Results Summary

| Channel | N | SC BLER | NN-SC BLER | NN/SC | Status |
|---------|---|---------|-----------|-------|--------|
| BEMAC B | 64 | 0.0056 | 0.003 | 0.54x | **NN wins** |
| BEMAC B | 128 | 0.002 | 0.0012 | 0.60x | **NN wins** |
| BEMAC B | 256 | 8e-5 | 4e-5 | 0.50x | **NN wins** |
| BEMAC B | 1024 | 1e-4 | 1e-4 | 1.0x | Matches |
| GMAC B | 64 | 0.025 | 0.026 | 1.04x | Matches |
| GMAC B | 128 | 0.016 | 0.017 | 1.06x | Matches |
| GMAC B | 256 | 0.005 | 0.015 | 3.0x | Gap |
| ISI-MAC | 32 | 0.731* | 0.676 | 0.92x | **NN wins** |

*Memoryless SC baseline (SC doesn't know about ISI)

## Files Created

### Scripts
- scripts/consolidate_bemac_results.py
- scripts/eval_gmac_multi_snr.py
- scripts/eval_crc_aided_nn_scl.py
- scripts/eval_bemac_comprehensive.py
- scripts/complexity_analysis.py
- scripts/create_paper_plots.py

### Neural Decoder Code
- neural/ncg_isi_mac.py — ISI-MAC decoder with temporal z_encoder
- neural/dine_mac.py — DINE/MINE unknown channel decoder
- neural/train_abnmac.py — ABNMAC training script
- neural/train_isi_mac.py — ISI-MAC training script

### Documentation
- docs/complexity_analysis.md
- docs/theoretical_analysis.md
- docs/paper_outline.md
- docs/literature_survey_mac_neural.md
- docs/paper_figures/ (4 figures + tables)

### Results
- results/complexity_analysis.json
- results/gmac_snr6dB/gmac_multi_snr_evaluation.json
- results/bemac/bemac_comprehensive_paper.json

## Next Steps (Hours 3-30)
1. Wait for CRC-aided results (N=64, 128) — expect strong results
2. Wait for ISI-MAC training convergence
3. Wait for DINE training + decoder training
4. Investigate ABNMAC failure (try analytical design instead of MC)
5. Continue N=512 GMAC training
6. Re-run plots with updated data
7. Write more of the paper draft
8. Run additional experiments as results come in
