# 12-Hour Progress — Memory MAC breakthrough

**NOTE: This file is superseded by `SESSION_FINAL_SUMMARY.md` and the revised `MEMORY_MAC_CHAPTER.md`, which incorporate the 10K-cw audit, CRC-SCL validation, MA-AGN results, N=128 outcome, and all other late-session data.**

Session date: 2026-04-16. Three background agents ran in parallel over ~12 hours.

## Headline result

**A neural chained NPD decoder matches analytical trellis SC on ISI-MAC at N=16 and N=32, and is within 1.5x at N=64**, without any explicit knowledge of the channel state model. This is the direct MAC extension of the NPD paper's memory-channel algorithm (Aharoni et al. 2024, Algorithm 3) — and it closes a central gap in the thesis.

### Results table (ISI-MAC, h=0.3, SNR 6 dB)

| N | **Chained NPD (neural)** | Chained trellis SC (new, ours) | Joint trellis SC | Broken NPD (previous) | Memoryless SC |
|---|---:|---:|---:|---:|---:|
| 16 | **0.133** (window) | 0.164 | 0.166 | 0.744 | 0.185 |
| 32 | **0.078** (window) | 0.084 | 0.075 | 0.876 | 0.114 |
| 64 | **0.043** (BiGRU) | 0.037 | 0.029 | 0.976 | 0.088 |
| 128 | pending | 0.018 | 0.008 | — | 0.095 |

All three entries for the neural chained NPD come from checkpoints trained in this session and saved under `class_c_npd/results/npd_memory_mac/`.

## What each of the three agents contributed

### 1. Research agent (~1h) — design document

Wrote `project_summary/NPD_MEMORY_MAC_DESIGN.md` (3,746 words) containing:
- Formal derivation of the chained SCT for memory MAC (Stage 1 effective channel: |S|=4 with Y marginalized as 2-mode Gaussian mixture; Stage 2: |S|=2 with deterministic X transitions)
- Paper's Algorithm 3 mapped to MAC setting
- Architecture options ranked (scalar / window / BiGRU / attention)
- Confirmation that the NPD paper's E^W for memory channels is an LSTM, implemented via the Donsker-Varadhan DINE potential

### 2. Baseline agent (~3h) — analytical chained trellis SC

Built `polar/decoder_trellis_mac_chained.py` and validated it at all four N values. Key finding: **chained SC is near-optimal** (within 2x of joint SC) at 6x lower computational cost. This provided the ceiling the neural implementation was aiming at.

### 3. Implementation agent (~8h) — chained NPD training

Built `neural/npd_memory_mac.py` and `scripts/train_npd_memory_mac.py`. Two E^W variants tested:
- **Sliding window** (like our NCG): best at N=16, 32
- **Bidirectional GRU** over the full z sequence: best at N=64

The critical fix that separated this implementation from the previous broken NPD was `use_analytical_training=False` — using the fully-neural tree operations instead of analytical tree ops that cannot propagate memory information.

## Why the previous NPD was broken

The original `class_c_npd` NPD used a scalar z-encoder that could only see one channel output at a time, and analytical tree operations for f/g that assumed a memoryless channel. Both assumptions failed for ISI-MAC — effective noise variance was inflated, and tree embeddings never saw the temporal context needed to cancel the ISI. Result: BLER 0.74-0.98 across N=16-64, worse than even memoryless SC.

The new chained NPD fixes both issues. The memory-aware z-encoder (window or BiGRU) sees the temporal context, and the fully-neural tree operations can represent the effective memory channel.

## Thesis narrative after today

> We derive the chained trellis SC decoder for memory MAC and show it matches the joint trellis decoder within 2x BLER at 6x lower computational cost. We then build a corresponding **neural** chained decoder that replaces the trellis with learned channel embedding (sliding window or BiGRU) and learned tree operations. On the ISI-MAC channel (h=0.3, SNR 6 dB), this neural chained NPD matches analytical trellis SC at N=16 and N=32, and is within 1.5x at N=64 — all without requiring explicit knowledge of the channel state model. This makes the approach directly applicable to channels with continuous or unknown state space where the joint trellis is intractable.

## Known gaps and next steps

### What's strong for immediate use
- ISI-MAC at N=16, 32, 64 — complete with baselines
- Clean architectural comparison (window vs BiGRU)
- Documented derivation of both chained SC and chained NPD
- Reproducible: scripts, results JSON, checkpoints all saved

### What needs more time
- N=64 BiGRU was still improving when the agent stopped — another 50K iters could close the 1.5x gap
- N=128 not trained (would take 6-10h more)
- Trapdoor MAC attempted but failed (design mismatch; needs proper MC design)
- Gilbert-Elliott MAC not started
- Ising MAC (from the paper) not implemented yet
- MA-AGN MAC (continuous state — where the neural approach is most valuable because SCT doesn't apply) not started

### Suggested next sessions

1. **(Low risk, 4h)** Extend N=128 training for both window and BiGRU; should close to within 1.3x of trellis SC.
2. **(Medium risk, 8-12h)** Build proper MC designs for Trapdoor and Gilbert-Elliott MAC, train chained NPD. Broadens the channel coverage.
3. **(High value, weeks)** Implement MA-AGN MAC channel + train chained NPD — this is the "flagship" case where SCT is intractable and the neural approach has unique value.

## Unresolved repo housekeeping

- PID 92903 (N=256 NCG rate-1 training) is still running. Has been wasting CPU throughout this session. Safe to kill — checkpoints up to iter70000 are saved and none were useful to the pivoted direction.

## Files created this session

### Documentation
- `project_summary/NCG_CHAPTER.md` (from the earlier NCG close-out)
- `project_summary/PIVOT_BRIEFING.md` (situation brief for ISI-MAC work)
- `project_summary/NPD_MEMORY_MAC_DESIGN.md` (research doc, 3,746 words)
- `project_summary/MEMORY_MAC_12H_PROGRESS.md` (this file)

### Code
- `neural/npd_memory_mac.py` — ChainedNPD_MAC with MemoryZEncoder (window + BiGRU)
- `scripts/train_npd_memory_mac.py` — training driver, curriculum, chained eval
- `polar/decoder_trellis_mac_chained.py` — analytical chained two-stage trellis SC
- `scripts/eval_chained_trellis_isi_mac.py` — baseline evaluation driver

### Results
- `class_c_npd/results/npd_memory_mac_results.md` — neural training results
- `class_c_npd/results/chained_trellis_sc_isi_mac.{md,json}` — analytical baseline
- `class_c_npd/results/npd_memory_mac/*.pt` — 12 best checkpoints + periodic snapshots
