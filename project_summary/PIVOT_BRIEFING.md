# Pivot briefing — ISI-MAC NPD direction

Purpose: compact situation doc written while the NCG chapter is closed. For re-entering the work after the break.

## Where we left off

NCG direction closed cleanly:
- N=256 wall is characterized (5.2x SC baseline; 3.3x at lower rate; 10.5x at very-low rate)
- Top-5 weak positions are NCG-specific, remaining gap is broadly distributed
- Full chapter at `project_summary/NCG_CHAPTER.md` (3,386 words)

## What's actually in the repo for ISI-MAC

### Two different neural attempts at ISI-MAC — different results

**NCG with sliding-window z_encoder** (`neural/ncg_isi_mac.py`, `neural/train_isi_mac.py`):
- z_encoder takes window (z[i-1], z[i]) as input, captures ISI
- Results from `results/isi_mac_nn_results.json`:
  | N  | NCG-window BLER | Memoryless SC BLER | Gain |
  |----|---|---|---|
  | 32 | 0.688 | 0.731 | −6% |
  | 64 | 0.466 | 0.575 | −19% |
- **Marginally beats memoryless SC, but nowhere near trellis SC.**
- Checkpoints: `saved_models/ncg_isi_mac_N32.pt`, `saved_models/ncg_isi_mac_N64.pt`

**NPD fast_ce without sliding window** (`neural/npd_pytorch.py`, `class_c_npd/training/npd_design_sweep.py`):
- Standard single-scalar z_encoder, no memory handling
- Results from `class_c_npd/results/isi_mac_classC_npd.json`:
  | N  | NPD BLER | Trellis SC BLER | Ratio |
  |----|---|---|---|
  | 16 | 0.744 | 0.167 | 4.5x |
  | 32 | 0.876 | 0.060 | 14.6x |
  | 64 | 0.976 | 0.029 | 33.7x |
- **Broken, gets worse with N.**
- Training log shows loss plateaus at 0.16-0.20 (never improves), BLER stuck
- Checkpoints: `class_c_npd/results/isi_mac_classC_s1_N{16,32,64}.pt`

### Critical observation
Trellis SC is the real baseline for ISI-MAC (it handles memory optimally). It has BLER 0.03 at N=64. Memoryless SC (which ignores ISI) is 0.58. The NCG-window result of 0.47 is about halfway between memoryless and trellis SC — improvement over naïve, but NOT close to trellis.

**The real target** is to match or beat trellis SC using a neural approach that generalizes to channels where trellis SC isn't tractable.

## Why NPD fails (hypothesis)

1. Single-scalar z_encoder can't represent ISI (needs previous sample)
2. fast_ce training plateaus at loss 0.18 — model can't fit ISI structure
3. Bad per-position posteriors → BLER catastrophic at inference

**Simple fix to test**: replace NPD's z_encoder with NCG's sliding-window z_encoder. Same idea, would likely work.

**Harder (more thesis-worthy) fix**: implement trellis-based NPD operations as in the NPD paper. Requires more code.

## What's been tried vs not tried

Tried:
- NPD fast_ce on ISI-MAC with scalar z → broken (above)
- NCG (sequential) with sliding window → beats memoryless SC, not trellis SC

Not tried (or not found in repo):
- NPD fast_ce with sliding-window z_encoder (cheap, likely to help)
- NPD with trellis operations (like the NPD paper for single-user memory)
- Joint NCG+NPD hybrid for memory channels
- Longer training of NCG-window with better initialization

## Key question for decision

**How close to trellis SC do we need to get for the thesis?**

Important framing from `docs/isi_mac_report.md` §6.2: the thesis value prop is "channel-agnostic decoding" — for **unknown channels** where trellis SC isn't available, the neural decoder is the best option. In that framing, **memoryless SC is the fair baseline**, because it's what you'd use in practice without channel knowledge.

Under that framing:
1. ✓ **Beat memoryless SC** (the fair, channel-agnostic baseline) — already done (NCG-window at N=64: 0.47 vs 0.58, a 19% improvement)
2. **Match trellis SC** — requires architectural work; trellis SC uses explicit channel knowledge we're trying to avoid
3. **Beat trellis SC** — essentially impossible; trellis is optimal with channel knowledge

Three possible thesis framings:
- **Conservative**: "neural decoder for unknown/complex channels, extends NPD to MAC" — uses existing NCG-window result as the main evidence
- **Medium**: "neural decoder approaches trellis SC without channel knowledge" — requires narrowing the gap, but within reach with more training/architecture
- **Ambitious**: "fast_ce-trainable NPD for memory MAC" — requires fixing the NPD-on-ISI-MAC failure (untried sliding-window NPD + longer training)

The existing ISI-MAC report §6 positions the story under the conservative framing but hasn't pushed to the medium one.

## Proposed weekend experiments (to discuss when back)

**Option A — quick win (8-12h CPU)**
1. Implement sliding-window z_encoder in NPD (steal from `neural/ncg_isi_mac.py`)
2. Retrain NPD on ISI-MAC at N=16, 32, 64
3. Expected: BLER drops from 0.74-0.98 to 0.3-0.6 (like NCG-window)
4. Result: clean parallel-trainable NPD for memory MAC. Modest, publishable.

**Option B — ambitious (multi-day)**
1. Adapt NPD paper's trellis-based tree operations to MAC
2. Retrain
3. Expected: close the gap to trellis SC
4. Result: real thesis contribution, "neural decoder matches trellis SC for memory MAC"
5. Risk: may not complete over the weekend

**Option C — diagnose first**
1. Per-position MI analysis on existing NPD ISI-MAC checkpoints (analogous to today's NCG work)
2. Understand where exactly the failure happens
3. Launch A or B based on findings

My recommendation: Option C for Friday night (cheap), then A or B on Saturday based on what C showed.

## Running jobs as of now
- PID 92903 (N=256 rate-1 training): still running at iter 70000/300000. No longer useful given pivot. **Should probably be killed.** Last checkpoint at 13:11.
- No other background jobs.

## Files mentioned

Code:
- `neural/ncg_isi_mac.py` — NCG with window z_encoder (working)
- `neural/train_isi_mac.py` — NCG ISI-MAC training driver
- `neural/npd_pytorch.py` — standard NPD
- `class_c_npd/models/npd_single_user.py` — NPD single-user (has comment about window being possible but not implemented)
- `class_c_npd/training/npd_design_sweep.py` — NPD training driver for MAC

Results:
- `results/isi_mac_nn_results.json` — NCG-window results
- `class_c_npd/results/isi_mac_classC_npd.json` — NPD fast_ce results (the broken ones)

Checkpoints:
- `saved_models/ncg_isi_mac_N{32,64}.pt` — NCG-window
- `class_c_npd/results/isi_mac_classC_s1_N{16,32,64}.pt` — NPD stage 1

Reports:
- `docs/isi_mac_report.md` — existing ISI-MAC writeup (partial)
- `project_summary/NCG_CHAPTER.md` — today's NCG chapter
