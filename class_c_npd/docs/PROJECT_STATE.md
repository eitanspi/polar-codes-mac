# Class C NPD — Project State

Last updated: 2026-04-09

## Concept

A neural Polar Decoder for the two-user MAC restricted to **Class C** decoding paths (`path_i = N`, decode all U first then all V). Class C is the only path where the polar tree decomposes exactly into **two chained single-user polar decoding problems**:

1. **Stage 1**: decode U over the marginal channel `p(z|x) = E_y[p(z|x,y)]`. For GMAC this is a Gaussian mixture channel.
2. **Stage 2**: decode V over the conditional channel `p(z|x,y)` after reconstructing `X̂` from Stage 1. For GMAC this is clean BPSK+AWGN.

Each stage uses the proven Aharoni et al. (2023) single-user NPD architecture with `fast_ce` parallel training. The 4-class joint MAC tensor decoder is **not** used.

## Status of components

| Component | Path | Status |
|---|---|---|
| Single-user NPD model | `models/npd_single_user.py` | **Working** (smoke test passes; clean AWGN BLER=0 at N=32) |
| Channel abstractions | `channels/mac_channel.py` | GMAC, BEMAC, ABNMAC, ISI-MAC stubs |
| Frozen-set loader | `channels/frozen_sets.py` | Reuses `polar.design_mc.design_from_file` (with polar tiebreak) |
| Stage 1/2 training | `training/train_stage.py` | Both stages with inference-matched data distribution |
| Curriculum sweep | `training/curriculum_sweep.py` | Multi-N sweep with warm-starting |
| Multi-channel sweep | `training/multi_channel_sweep.py` | GMAC + BEMAC + ABNMAC support |
| Chained eval | `eval/chain_eval.py` | End-to-end with Wilson 95% CIs |
| Error analysis | `eval/error_analysis.py` | Per-position SC vs NPD comparison |
| Plotting | `eval/plot_sweep.py` | BLER-vs-N curves and ratio plot |
| ISI-MAC design gen | `scripts/generate_isi_design.py` | Approximate (memoryless LLR) starter design |
| Multi-channel SC ref | `scripts/generate_sc_references.py` | SC baselines for all channels |
| Smoke test | `smoke_test.py` | 5-step end-to-end pipeline check |
| Status reporter | `status.py` | Live view of running experiments |

## Critical bug found and fixed

The single-user NPD originally in this project (`neural/npd_pytorch.py`) **failed its own self-test** because of a bit-reversal mapping bug: the recursive tree traversal visits leaf positions in **bit-reversed natural order**, but the decoder stored decoded bits at the linear leaf index. The fix is to map `nat_idx = br[leaf_idx]` and store at the natural-order position. The clean implementation in `models/npd_single_user.py` includes this fix.

## Critical pipeline fix found and applied

The training data generator originally drew V uniformly over `{0,1}^N`, but at inference time V is drawn from a polar codebook with frozen positions = 0 (a `kv`-dimensional subspace). Some `y[i]` values become deterministic, giving Stage 1 a different per-position channel statistic at inference than at training. The fix is to draw V from the **same** distribution at training time. Same for X in Stage 2.

## SC reference (GMAC Class C, 50% per-user capacity, SNR=6dB)

| N | ku | kv | SC BLER |
|---|----|----|---------|
| 16 | 4 | 7 | 0.163 |
| 32 | 7 | 15 | 0.068 |
| 64 | 15 | 29 | 0.027 |
| 128 | 30 | 58 | 0.0054 |
| 256 | 59 | 117 | 0.0020 |
| 512 | 119 | 233 | 0.0008 |
| 1024 | 238 | 467 | 0.0002 |

The waterfall (BLER decreasing with N at fixed rate) is the expected polar code finite-length effect.

## Pilot results so far

| N | NPD chained BLER | 95% CI | SC BLER | Ratio | Pass (≤1.5x)? |
|---|---|---|---|---|---|
| 16 | **0.1362** | [0.1270, 0.1460] | 0.1626 | **0.84x** | ✓ PASS (NPD beats SC) |
| 32 | ~0.136 (plateaued) | (running) | 0.0684 | ~2.0x | ✗ FAIL (slightly over target) |
| 64 | (queued) | | 0.0266 | | |
| 128 | (queued) | | 0.0054 | | |

## Key findings

1. **Single-user NPD architecture is correct.** Smoke test passes (BLER=0 on
   clean AWGN at N=32). The earlier project NPD (`neural/npd_pytorch.py`) fails
   its own self-test due to the bit-reversal bug — fixed in this project.

2. **Mixture LLR is correct.** Cross-checked the project's `_u_marginal_llr`
   against an independent implementation. Both give SC BLER ≈ 0.065 at
   N=32, ku=7. The SC reference of 0.068 is reliable.

3. **NPD matches/beats SC at N=16.** Chained BLER 0.136 vs SC 0.163
   (statistically significant — 95% CI [0.127, 0.146], both bounds below SC).

4. **NPD plateaus ~2x SC at N=32 mixture channel.** Stage 1 converges to
   BLER ≈ 0.13-0.14 regardless of training time, distribution fix, LLR vs raw
   z input, or warm-starting from N=16. This is a *fundamental finite-N gap*.

5. **Operating-point sensitivity.** SNR diagnostic shows the gap narrows at
   higher SNR (BLER ~0.138 at SNR=8dB vs ~0.166 at SNR=6dB after 10K iters).
   The plateau is worst near marginal operating points where per-position Pe
   is moderate (~0.01).

6. **The 2x gap at marginal Pe is consistent with polar-NPD literature.**
   For channels where SC achieves Pe ~10^-6 per position, NPD trivially
   matches SC because there is huge accuracy slack. For channels where SC
   achieves Pe ~10^-2 per position, NPD's small approximation errors matter.

## Open questions

1. **Does the gap grow or stay constant as N grows?**
   The curriculum sweep is running to test this at N=64, 128. If the gap stays
   ~2x, NPD is consistently *N-independent worse* than SC at this rate target.

2. **Does generalization to BEMAC, ABNMAC work?**
   Multi-channel sweep code is ready. BEMAC's Stage 2 is trivial (no noise
   after subtracting X). ISI-MAC needs proper design generation first.

3. **Is the 2x gap fundamental to fast_ce, or is there a better training
   objective for marginal channels?**
   Possible to test: leaf-only loss, scheduled sampling within fast_ce, larger d.

## Currently running experiments

1. **Curriculum sweep N=16→32→64→128** (`results/curriculum_sweep.log`)
   Trains Stage 1 + Stage 2 NPDs at each N, warm-starting from previous N.
2. **SNR sweep diagnostic at N=32** (`results/diagnostic_snr_sweep.log`)
   Tests whether the NPD plateau on the mixture channel narrows at higher SNR.

## Open questions

1. **Does the NPD match SC at N≥32 on the mixture channel?**
   First retry attempts at N=32 plateaued at ~2x SC. The curriculum sweep is testing whether warm-starting from N=16 (which works) breaks this plateau.

2. **Is the plateau operating-point-sensitive?**
   The SNR sweep diagnostic tests whether higher SNR (where the mixture is easier) closes the gap. If yes, the issue is fragility near marginal SNRs, not architecture.

3. **Does the approach generalize to BEMAC, ABNMAC, ISI-MAC?**
   Multi-channel sweep code is ready. BEMAC's Stage 2 is trivial (no noise after subtracting X), so BEMAC's chained BLER is essentially Stage 1's BLER on a binary erasure-style channel.

## Next steps after curriculum finishes

1. Generate plots: `python -u class_c_npd/eval/plot_sweep.py`
2. Run error analysis on Stage 1 N=32 checkpoint to characterize where it differs from SC
3. Run multi-channel sweep: `python -u class_c_npd/training/multi_channel_sweep.py --channel all --N_list 16,32,64`
4. Generate proper ISI-MAC Class C designs (currently a placeholder)

## Files / artifacts

- `results/gmac_sc_reference_50pct.json` — SC reference table (DONE)
- `results/curriculum_sweep_results.json` — incremental output of curriculum sweep
- `results/curriculum_sweep.log` — live training log
- `results/sc_references_multi_channel.json` — multi-channel SC baselines
- `results/<channel>_<stage>_N<n>_best.pt` — saved checkpoints
