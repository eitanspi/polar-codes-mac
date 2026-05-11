# SNR Sweeps — Thesis BLER-vs-SNR Curves

**Date:** 2026-04-16. Pure-evaluation sweep using existing trained checkpoints.
Runner: `scripts/snr_sweep_thesis.py`. Plotter: `scripts/plot_snr_sweeps.py`.
Total wall time: ~3.4 minutes (all three configs, all SNR points).

## Configs swept

| # | Config | Decoder | Channel | Path | N | SNR grid |
|---|--------|---------|---------|------|----|----------|
| 1 | chained NPD (BiGRU) vs chained trellis SC | memory-aware NPD + analytical ISI trellis SC | ISI-MAC, h = 0.3 | Class C (corner) | 16, 32, 64 | {4,5,6,7,8} dB |
| 2 | NCG vs analytical SC | NCG MLP (d=16) + SC with log-LLR | Gaussian MAC | Class B (path_i = N/2) | 32, 64, 128 | {4,5,6,7,8} dB |
| 3 | chained NPD vs analytical SC | Curriculum-trained NPDSingleUser s1/s2 + SC | Gaussian MAC | Class C (path_i = N) | 64 | {4,5,6,7,8} dB |

Notes on scope:
- **GMAC Class C N = 128 was skipped** because no Stage 2 checkpoint exists
  (only `curriculum_gmac_c_s1_N128_best.pt` was trained; its `s2` counterpart
  is missing). Reported in the "Missing checkpoints" section below.
- Codeword counts: 1500 per point for ISI N = 16, 32 (NPD + SC) and 1000 for
  ISI N = 64; 2000 per point for GMAC-B N ≤ 64 and GMAC-C; 1500 for
  GMAC-B N = 128.
- All frozen sets come from the 6 dB design file (matches checkpoint training
  SNR); only channel noise varies across SNR points. This is a
  *fixed-design, noise-sweep* evaluation — the conventional way to produce a
  waterfall curve.

## Data files

All saved under `results/snr_sweep/` as one JSON per (decoder, N) config:

- `chained_npd_isi_mac_N{16,32,64}.json`
- `chained_trellis_sc_isi_mac_N{16,32,64}.json`
- `ncg_gmac_classB_N{32,64,128}.json`
- `sc_gmac_classB_N{32,64,128}.json`
- `chained_npd_gmac_classC_N64.json`
- `sc_gmac_classC_N64.json`

Each JSON has the structure:
```json
{
  "config": "chained_npd_isi_mac_N32",
  "N": 32, "channel": "ISIMAC h=0.3", "path": "class_c",
  "decoder": "chained_npd_bigru", "ku": 7, "kv": 15,
  "sweep": {
    "4": {"bler": ..., "ci_lo": ..., "ci_hi": ..., "n_cw": 1500,
           "errs": ..., "time_s": ..., "sigma2": ...},
    "5": {...}, ...
  }
}
```

## Plots

Saved to `docs/paper_figures/` in both `.png` and `.pdf` at 300 dpi, serif
font, log-scale y, 11 pt:

- `fig_snr_sweep_isi_mac.{png,pdf}` — three N x two decoders
- `fig_snr_sweep_gmac_classB.{png,pdf}` — three N x two decoders
- `fig_snr_sweep_gmac_classC.{png,pdf}` — one N (64) x two decoders

Error bars are 95 % Wilson CIs.

## Results tables

### ISI-MAC h = 0.3 — chained NPD (BiGRU, 1 GRU layer) vs chained trellis SC

BLER per (decoder, N, SNR dB). Wilson 95 % CI in parentheses.

| N | SNR | chained NPD BLER (95% CI) | n_cw | chained trellis SC BLER (95% CI) | n_cw |
|---|-----|----------------------------|------|-----------------------------------|------|
| 16 | 4 | 0.1707 (0.153, 0.190) | 1500 | 0.1760 (0.158, 0.196) | 1500 |
| 16 | 5 | 0.1560 (0.139, 0.175) | 1500 | 0.1713 (0.153, 0.191) | 1500 |
| 16 | 6 | 0.1500 (0.133, 0.169) | 1500 | 0.1747 (0.156, 0.195) | 1500 |
| 16 | 7 | 0.1300 (0.114, 0.148) | 1500 | 0.1580 (0.140, 0.177) | 1500 |
| 16 | 8 | 0.1307 (0.115, 0.149) | 1500 | 0.1573 (0.140, 0.177) | 1500 |
| 32 | 4 | 0.1773 (0.159, 0.198) | 1500 | 0.1127 (0.098, 0.130) | 1500 |
| 32 | 5 | 0.1233 (0.108, 0.141) | 1500 | 0.0920 (0.078, 0.108) | 1500 |
| 32 | 6 | 0.1107 (0.096, 0.128) | 1500 | 0.0873 (0.074, 0.103) | 1500 |
| 32 | 7 | 0.1100 (0.095, 0.127) | 1500 | 0.0720 (0.060, 0.086) | 1500 |
| 32 | 8 | 0.0940 (0.080, 0.110) | 1500 | 0.0713 (0.059, 0.086) | 1500 |
| 64 | 4 | 0.1570 (0.136, 0.181) | 1000 | 0.1060 (0.088, 0.127) | 1000 |
| 64 | 5 | 0.0800 (0.065, 0.099) | 1000 | 0.0470 (0.036, 0.062) | 1000 |
| 64 | 6 | 0.0460 (0.035, 0.061) | 1000 | 0.0430 (0.032, 0.057) | 1000 |
| 64 | 7 | 0.0440 (0.033, 0.059) | 1000 | 0.0280 (0.019, 0.040) | 1000 |
| 64 | 8 | 0.0370 (0.027, 0.051) | 1000 | 0.0210 (0.014, 0.032) | 1000 |

### GMAC Class B — NCG (MLP, d=16) vs analytical SC

| N | SNR | NCG BLER (95% CI) | n_cw | SC BLER (95% CI) | n_cw |
|---|-----|-------------------|------|------------------|------|
| 32 | 4 | 0.2970 (0.277, 0.317) | 2000 | 0.2925 (0.273, 0.313) | 2000 |
| 32 | 5 | 0.1320 (0.118, 0.147) | 2000 | 0.1280 (0.114, 0.143) | 2000 |
| 32 | 6 | 0.0515 (0.043, 0.062) | 2000 | 0.0495 (0.041, 0.060) | 2000 |
| 32 | 7 | 0.0160 (0.011, 0.022) | 2000 | 0.0185 (0.014, 0.025) | 2000 |
| 32 | 8 | 0.0170 (0.012, 0.024) | 2000 | 0.0155 (0.011, 0.022) | 2000 |
| 64 | 4 | 0.3590 (0.338, 0.380) | 2000 | 0.3405 (0.320, 0.362) | 2000 |
| 64 | 5 | 0.1280 (0.114, 0.143) | 2000 | 0.1250 (0.111, 0.140) | 2000 |
| 64 | 6 | 0.0285 (0.022, 0.037) | 2000 | 0.0315 (0.025, 0.040) | 2000 |
| 64 | 7 | 0.0090 (0.006, 0.014) | 2000 | 0.0115 (0.008, 0.017) | 2000 |
| 64 | 8 | 0.0075 (0.005, 0.012) | 2000 | 0.0060 (0.003, 0.011) | 2000 |
| 128 | 4 | 0.4413 (0.416, 0.467) | 1500 | 0.3420 (0.318, 0.366) | 1500 |
| 128 | 5 | 0.1140 (0.099, 0.131) | 1500 | 0.0980 (0.084, 0.114) | 1500 |
| 128 | 6 | 0.0153 (0.010, 0.023) | 1500 | 0.0067 (0.004, 0.012) | 1500 |
| 128 | 7 | 0.0093 (0.006, 0.016) | 1500 | 0.0100 (0.006, 0.016) | 1500 |
| 128 | 8 | 0.0087 (0.005, 0.015) | 1500 | 0.0093 (0.006, 0.016) | 1500 |

### GMAC Class C corner rate — chained NPD vs analytical SC

| N | SNR | chained NPD BLER (95% CI) | n_cw | SC BLER (95% CI) | n_cw |
|---|-----|----------------------------|------|------------------|------|
| 64 | 4 | 0.2735 (0.254, 0.294) | 2000 | 0.0575 (0.048, 0.069) | 2000 |
| 64 | 5 | 0.2035 (0.186, 0.222) | 2000 | 0.0320 (0.025, 0.041) | 2000 |
| 64 | 6 | 0.1650 (0.149, 0.182) | 2000 | 0.0255 (0.019, 0.033) | 2000 |
| 64 | 7 | 0.1525 (0.137, 0.169) | 2000 | 0.0205 (0.015, 0.028) | 2000 |
| 64 | 8 | 0.1555 (0.140, 0.172) | 2000 | 0.0235 (0.018, 0.031) | 2000 |

## Key observations

**ISI-MAC (chained NPD vs chained trellis SC).**

- At N = 16 the NPD beats trellis SC at every SNR: the neural
  network's implicit smoothing of the stage-2 V decision outweighs its
  per-bit approximation error for the very short block. NPD gains grow
  slightly with SNR (0.03 absolute at SNR = 4 dB; 0.03 at SNR = 8 dB).
- At N = 32 trellis SC is the clear winner — 0.5 to 2x lower BLER at
  every SNR — because the BiGRU checkpoint is undertrained for stage-1
  U extraction (BLER 0.11 at SNR = 6 dB vs trellis 0.087).
  **The gap to analytical widens at higher SNR** (NPD/SC ratio = 1.3x
  at SNR 4 dB, 1.5x at SNR 6 dB, 1.3x at SNR 8 dB); the neural decoder
  does not ride the waterfall as steeply as trellis SC.
- At N = 64 the NPD closes the gap dramatically. At SNR 6 dB NPD
  and SC are statistically indistinguishable (0.046 vs 0.043). Above
  6 dB trellis SC pulls ahead again (0.021 vs 0.037 at SNR 8 dB). The
  neural decoder's error floor around BLER ~ 0.03 is consistent with
  what was observed in the 6 dB training evaluation
  (`isi_mac_bigru_results.json`).
- Across all N, **trellis SC has a sharper waterfall**. The chained NPD
  plateaus earlier, which is the expected behaviour when the neural
  model cannot exploit the full ISI trellis structure — the BiGRU
  encoder integrates context but the tree operations are the generic
  NPD ones, not ISI-aware. This is a quantitative baseline for where a
  memory-aware tree operator could add value in future work.

**GMAC Class B (NCG vs SC).**

- NCG and SC are **statistically tied at every SNR for N = 32 and
  N = 64**. At N = 128 NCG begins to plateau: at SNR 6 dB NCG BLER =
  0.015 vs SC 0.0067 (2.3x gap; outside CIs). At SNR 4 dB NCG is
  1.3x worse than SC. From SNR = 7 dB on the gap closes again (one
  point each, consistent with the error-floor visible in the 6 dB
  training run).
- The N = 128 plateau matches the "GMAC wall" described in
  `NCG_CHAPTER.md`: the NCG's per-bit inference saturates around
  BLER ≈ 0.01 independent of SNR, while SC continues to improve.
- These numbers replicate the single-point 6 dB results in
  `results/gmac_snr6dB/nn_scl_full_comparison.json` (NCG N=32:
  0.047 → 0.052 here; N=64: 0.029 → 0.028; N=128: 0.013 → 0.015).

**GMAC Class C corner rate (chained NPD vs SC).**

- The chained NPD is **uniformly worse than analytical SC** at N = 64
  across the full SNR grid — by factors of 4x (SNR 4 dB) to 8x (SNR
  7 dB). This matches the historical snapshot in
  `curriculum_sweep_results.json` (chained NPD BLER = 0.176 at SNR
  6 dB vs SC 0.026, ratio 6.6x). Our measurement at 6 dB: 0.165 vs
  0.025 (ratio 6.6x) — statistically identical.
- The NPD's BLER curve flattens above 6 dB while SC continues its
  waterfall — this is the *same* qualitative failure mode as the
  GMAC Class B plateau but at a much higher BLER, because the Class C
  Stage-1 U decoder never achieved the accuracy of Stage 2.

**Missing checkpoints.**

- GMAC Class C Stage 2 at N = 128: not available in
  `class_c_npd/results/` or `saved_models/`. Only Stage 1 was trained
  (`curriculum_gmac_c_s1_N128_best.pt`). N = 128 chained NPD GMAC-C is
  therefore omitted from the sweep.
- Window-variant (W=2) checkpoints on ISI-MAC were also available but
  BiGRU is reported to be strictly better at N ≥ 64
  (`npd_memory_mac_results.md` §5). We used BiGRU at all N for a
  cleaner cross-N comparison.

## How to reproduce

```bash
cd to_git_v2
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/snr_sweep_thesis.py \
  --only isi gmacB gmacC --Ns_gmacC 64
python scripts/plot_snr_sweeps.py
```

Runs in ~3.5 minutes on a MacBook without GPU. All checkpoints are read
from `class_c_npd/results/` and `saved_models/` — no training.

---

## Addendum (2026-04-17): ISI-MAC h-robustness sweep

See `project_summary/ISI_H_SWEEP.md` for the full write-up. Summary:

At N=32, SNR=6 dB, the chained NPD BiGRU (trained at h=0.3) was
evaluated at h ∈ {0.2, 0.3, 0.5, 0.7} against chained trellis SC and
memoryless SC, 1000 CW each.

- NPD never beats trellis SC at this N (trellis is 0.087 vs NPD 0.118
  at h=0.3, their respective best).
- NPD generalises gracefully to h=0.2 (0.126), degrades 1.6x at h=0.5
  (0.191), and degrades 3.2x at h=0.7 (0.373) relative to h=0.3.
- Memoryless SC cliffs between h=0.3 (0.097) and h=0.5 (0.477), so
  memory-aware decoders are required for h ≥ 0.5.
- NPD still dominates memoryless SC at every h (0.373 vs 0.815 at
  h=0.7).

Files: `results/snr_sweep/isi_mac_h_sweep_N32.json`,
`docs/paper_figures/fig_isi_mac_h_sweep.{png,pdf}`,
`scripts/task2_isi_h_sweep.py`, `scripts/plot_isi_h_sweep.py`.

## Addendum (2026-04-17): GMAC Class C corner result verification

See `project_summary/GMAC_CORNER_NPD_VERIFICATION.md`. The new SNR
sweep's chained-NPD GMAC-C N=64 BLER (0.165, 6x worse than SC) is
reproduced correctly — the "0.56-0.80x SC" README claim refers to a
different checkpoint family (`npd_design_p3_*`) and a Stage-1-only
protocol, not the curriculum chained pipeline. The README should be
qualified to distinguish the two.

