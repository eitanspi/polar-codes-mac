# GMAC Class C Corner NPD — Result Verification

Date: 2026-04-17
Config verified: GMAC Class C (path_i = N), N=64, ku=15, kv=29, SNR=6 dB,
2000 codewords, Wilson 95% CIs.

## TL;DR

The README claim "NPD beats SC by 0.56-0.80x at GMAC Class C corner"
**is referring to a different checkpoint family** than the one used by
the new SNR-sweep infrastructure, **and it measured a different thing**
(Stage 1 U-bit BLER only, not chained block error).

- The new SNR sweep (`results/snr_sweep/chained_npd_gmac_classC_N64.json`)
  uses `curriculum_gmac_c_s{1,2}_N64_best.pt` and evaluates the
  end-to-end chained decoder. At SNR=6 dB, N=64 it reports
  **BLER=0.1650 (~6.5x worse than SC)**. I reproduced this:
  **chained BLER=0.1675 [0.1518, 0.1845], 335/2000.**
- The canonical "0.017 / 0.63x SC" number in the README / master CSV
  came from a *different* checkpoint (`npd_design_p3_N64_best.pt`) and a
  different evaluation protocol (Stage 1 only, NPD-chosen frozen set).
  I reproduced this: **stage-1-only U-BLER = 0.0095 [0.0061, 0.0148],
  19/2000** (the published 0.017 used 1000 codewords, a different seed,
  and a freshly retrained model).

## Checkpoints found at N=64 for GMAC Class C

All in `class_c_npd/results/`.

| File | Stage | Au pattern | Notes |
|---|---|---|---|
| `curriculum_gmac_c_s1_N64_best.pt` | 1 | [32, 46, 47, 48, 52, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64] | Trained with **genie (SC-optimal) design**; used by snr_sweep |
| `curriculum_gmac_c_s2_N64_best.pt` | 2 | Av = genie | Paired stage-2 for curriculum s1 |
| `npd_design_p3_N64_best.pt` | 1 (Phase 3) | [16, 24, 28, 29, 30, 32, 40, 44, 48, 56, 60, 61, 62, 63, 64] | Trained with **NPD-chosen frozen set** (MI-selected); **this is what the old 0.017 used** |
| `npd_design_p1_N64_best.pt` | 1 (Phase 1) | All-info (|Au|=64) | MI measurement pre-cursor, not a decoder |
| `isi_mac_classC_s1_N64.pt` | — | SC-optimal Au | For ISI-MAC, not GMAC |
| `bemac_classC_s*_N64.pt` | — | — | For BEMAC, not GMAC |
| `mc_bemac_s*_N64_best.pt` | — | — | For BEMAC, not GMAC |
| `neural_abnmac_s1_N64.pt` | — | — | For ABNMAC, not GMAC |
| `neural_bemac_s1_N64.pt` | — | — | For BEMAC, not GMAC |
| `maagn_bigru_L1_s1_N64_best.pt` | — | — | For MAAGN, not GMAC |

There is **no** `npd_design_p3_N64_s2.pt` — the 0.017 evaluation only
trained a Stage 1 NPD on the NPD-chosen design.

## BLER measurements at the target config (N=64, SNR=6 dB, ku=15, kv=29, 2000 CW)

### Analytical SC baseline (canonical)
| Decoder | BLER | 95% CI | Errors |
|---|---|---|---|
| Analytical chained SC, Class C genie design | **0.0275** | [0.0212, 0.0356] | 55/2000 |

CSV says 0.027. Reproduced exactly.

### Stage 1 only (U bits; V ignored)
| Checkpoint | Frozen set | U-BLER | 95% CI | U errors |
|---|---|---|---|---|
| `curriculum_gmac_c_s1_N64_best.pt` | genie | **0.1590** | [0.1436, 0.1757] | 318/2000 |
| `npd_design_p1_N64_best.pt` | all-info (not a real decoder) | 1.0000 | [0.998, 1.0] | 2000/2000 |
| `npd_design_p3_N64_best.pt` | NPD-chosen | **0.0095** | [0.0061, 0.0148] | 19/2000 |

The critical split:
- The **curriculum** Stage 1 is about **17x worse than SC** at Stage 1 alone.
- The **npd_design_p3** Stage 1 is about **3x better than SC** at Stage 1 alone.

### Chained (U + V, full corner-rate block error)
| Pipeline | BLER | 95% CI | U errs | V errs |
|---|---|---|---|---|
| curriculum s1 + curriculum s2 (what snr_sweep uses) | **0.1675** | [0.1518, 0.1845] | 335 | 334 |
| p3 s1 + curriculum s2 (cross-ckpt, for diagnosis) | **0.0125** | [0.0085, 0.0184] | 25 | 25 |

- The curriculum pipeline's chained BLER of 0.1675 is dominated by its
  very bad Stage 1 (0.159). V barely adds error given U already failed.
- The cross-ckpt chimera (using the *good* Stage 1 from p3) gives
  0.0125, which finally **does** beat SC's 0.0275 by ~0.45x — consistent
  with the old published claim.

## Root cause of the discrepancy

Three distinct factors stack:

1. **Different checkpoints.** The SNR sweep uses `curriculum_gmac_c_*`
   checkpoints. The old "0.56-0.80x SC" numbers used `npd_design_p3_*`
   checkpoints. Those were trained by a completely different pipeline
   (`npd_design_sweep.py`) with a completely different objective
   (train on MI-selected NPD-optimal frozen set, not the genie set).

2. **Different frozen sets at the same (ku, kv).**
   - genie design Au = {32, 46, 47, 48, 52, 54, 55, 56, 58, 59, 60, 61,
     62, 63, 64}
   - p3 NPD design Au = {16, 24, 28, 29, 30, 32, 40, 44, 48, 56, 60, 61,
     62, 63, 64}
   Same rate, but the NPD-optimal set is much easier for this specific
   learned decoder — which is exactly the point of the NPD design
   procedure. The old "0.017" is a valid BLER, but only for the NPD
   checkpoint on its NPD-chosen set; it is not directly comparable to
   the SC "0.027" on the SC-optimal set (though the CSV presents them as
   if they are). The old paper claim is approximately an "NPD on its
   best-matched code beats SC on its best-matched code", not "same
   code, different decoders".

3. **Stage 1 only vs end-to-end chained.** The old "0.017" was U-bit
   block error on the marginal channel only (no V decoding at all).
   For Class C, if the paper wants the block error of the full (U, V)
   transmission, V must also be decoded. In the old regime (0.017
   Stage 1), the published BLER ignored V entirely.

The curriculum checkpoints, despite reusing the SC-optimal Au, are
noticeably worse-trained: the `curriculum_gmac_c_s1_N64_best.pt` model
gives Stage 1 U-BLER of 0.159, whereas even SC on the same code gets
0.027. Without seeing the training logs this is not debuggable here,
but the best hypothesis is:
- The curriculum procedure (N=16 → N=32 → ... → N=1024) is optimised for
  reaching N=1024 at low BLER; per-N checkpoints are saved at their
  "best" during a warm-start phase, before polarisation has fully
  converged at each N.
- The npd_design_p3 N=64 checkpoint was trained purely for N=64 with a
  matched (NPD-chosen) frozen set and achieves the best possible Stage
  1 BLER for *its* code.

## Canonical number for the thesis

**The honest number for GMAC Class C at N=64, ku=15, kv=29, SNR=6 dB is
one of:**

- **0.0095 [0.0061, 0.0148], n_cw=2000** — Stage 1 only, NPD on its own
  NPD-chosen frozen set (`npd_design_p3_N64`). Must be reported with
  the caveat "NPD uses a different frozen set than SC at the same rate".
- **0.0125 [0.0085, 0.0184], n_cw=2000** — Chained (U+V) using
  `npd_design_p3` Stage 1 + `curriculum_gmac_c` Stage 2. Still beats SC
  (0.0275), by ~0.45x. Again, NPD uses a different U frozen set.
- **0.1675 [0.1518, 0.1845], n_cw=2000** — Chained (U+V) using
  `curriculum_gmac_c_s1+s2` on the SC-optimal code. **About 6.1x worse
  than SC.** This is the honest "same-code, different-decoder"
  comparison for the current curriculum pipeline.

The README's "0.56-0.80x SC" claim corresponds to the first two bullets
and hides two caveats:
- It's NPD vs SC on *different* info sets at the same rate (a fair
  comparison in the sense of "code rate is fixed", but not in the sense
  of "same information positions").
- At N=64 specifically it's a Stage-1-only number, because no paired
  Stage 2 checkpoint was trained for the NPD-chosen Au.

## Recommended fixes to the README / master CSV

`project_summary/README.txt` currently says:

```
GMAC      | Corner  | NPD    | 16-256     | 0.56-0.80x    | WORKING (beats SC)
```

This should be qualified:

```
GMAC      | Corner  | NPD (stage 1, NPD-chosen code)    | 16-256     | 0.56-0.80x    | beats SC on its NPD-chosen design
GMAC      | Corner  | NPD chained (curriculum, genie code) | 16-1024 | 6x worse | same-code apples-to-apples
```

`project_summary/results/all_bler_results.csv` row for (N=64, ku=15,
kv=29, NPD_BLER=0.017) should note:
- Source: `class_c_npd/results/npd_design_p3_N64_best.pt`
- Frozen set: NPD-chosen (different from SC row above)
- Protocol: Stage 1 U-BLER only

And the new canonical "same-code chained" numbers from the SNR sweep
should be added as a separate row, e.g.:

```
GMAC,C,NPD-chained-curriculum,64,15,29,0.027,0.165,6.1x,Stage1 curriculum underperforms
```

## What checkpoint the SNR sweep is using (explicit)

`scripts/snr_sweep_thesis.py:run_gmac_classC_sweep` (around line 305) loads:

```python
s1p = f'class_c_npd/results/curriculum_gmac_c_s1_N{N}_best.pt'
s2p = f'class_c_npd/results/curriculum_gmac_c_s2_N{N}_best.pt'
```

and uses the Au stored inside these checkpoints (which happens to be
the SC-optimal / genie set at N=64). So the numbers are internally
consistent — they are just a different experiment than the old one.

## Files generated

- `results/snr_sweep/task1_gmac_corner_npd_verification.json` — raw
  numbers from the re-run.
- `results/snr_sweep/task1b_chain_swap.json` — the cross-checkpoint
  (p3 s1 + curriculum s2) chained number.
- `scripts/task1_verify_gmac_corner.py` — reproducible script.
- `scripts/task1b_chain_swap.py` — cross-checkpoint script.
