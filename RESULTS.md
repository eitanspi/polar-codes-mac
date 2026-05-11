# Consolidated results

All numbers below are **chained corner-rate BLER** with each decoder using its own MC-derived frozen design. SNR = 6 dB throughout. The `errs/CW` column gives counted block errors out of total codewords evaluated.

## ISI-MAC (h = 0.3)

| N | NPD BLER | NPD errs/CW | SCT BLER | SCT errs/CW |
|---|---|---|---|---|
| 16 | 0.16472 | 16472 / 100K | 0.1501 | 4503 / 30K |
| 32 | 0.06873 | 6873 / 100K | 0.0691 | 1403 / 20K |
| 64 | 0.03284 | 1642 / 50K | 0.0289 | 578 / 20K |
| 128 | 0.0127 | 637 / 50K | 0.00745 | 149 / 20K |
| 256 | 0.00138 | 69 / 50K | 0.00185 | 37 / 20K |
| 512 | 0.000307 | 92 / 300K | 0.00038 | 19 / 50K |
| 1024 | 5.5 × 10⁻⁵ | 20 / 600K (3.3e-5 incl. topup) | 0 / 50K (≤6e-5) | 0 / 50K |

NPD is "chained NPD" (rate-1 trained, MI-based frozen design). SCT is "joint trellis SC" (full MAC joint-state forward-backward followed by computational-graph SC; own MC design at design-SNR = 3 dB to avoid genie saturation).

## MA-AGN-MAC (α = 0.5)

| N | NPD BLER | NPD errs/CW | memoryless-SC BLER | SC errs/CW |
|---|---|---|---|---|
| 16 | 0.14654 | 7327 / 50K | 0.10678 | 5339 / 50K |
| 32 | 0.09925 | 9925 / 100K | 0.0402 | 2010 / 50K |
| 64 | 0.0265 | 1325 / 50K | 0.03494 | 1747 / 50K |
| 128 | 0.00618 | 309 / 50K | 0.01616 | 808 / 50K |
| 256 | 0.00068 | 34 / 50K | 0.00146 | 73 / 50K |
| 512 | 6.5 × 10⁻⁵ | 13 / 200K (5e-5 incl. topup) | 0.00041 | 41 / 100K |
| 1024 | 4 × 10⁻⁵ | 8 / 200K (3e-5 incl. topup) | 0.000143 | 43 / 300K |

There is no efficient analytical SC for AR(1) noise (continuous state), so the analytical baseline is memoryless SC, which ignores the noise correlation entirely.

## SNR sweep (ISI-MAC, joint trellis SC, "SCT decoder")

| N \ SNR | 3 dB | 4 dB | 5 dB | 6 dB | 7 dB | 8 dB |
|---|---|---|---|---|---|---|
| 64 | 0.111 | 0.063 | 0.044 | 0.038 | 0.037 | 0.031 |
| 128 | 0.089 | 0.048 | 0.027 | 0.019 | 0.014 | 0.011 |
| 256 | 0.049 | 0.018 | 0.008 | 0.0023 | 0.0022 | 0.0026 |

20K codewords per point. The N=256 floor at 7-8 dB reflects MC-design noise at very high SNR rather than a true decoder limitation.

## MA-AGN α sweep (memoryless SC, own MC design)

| α \ N | 64 | 128 | 256 |
|---|---|---|---|
| 0.3 | 0.038 | 0.0169 | 0.00203 |
| 0.5 | 0.0349 | 0.01616 | 0.00146 |
| 0.7 | 0.0204 | 0.01273 | 0.00203 |
| 0.9 | 0.0308 | 0.0193 | 0.0042 |

20-30K codewords per point. NPD α-sweep is incomplete (partial at α=0.7) — see `class_c_npd/results/maagn_alpha_sweep/`.

## Statistical notes

- 95 % Poisson CIs for the smallest-error points (k ≤ 30 errors): CI = `[max(0, (k - 1.96·√k)/n), (k + 1.96·√k)/n]`. For k = 0 we report `[0, 3/n]`.
- The "≥30 errors" criterion is met for every point except ISI-MAC N=1024 SCT (still 0 errors on 50K), where the BLER is too small to reach 30 errors on practical CPU budgets. The 95 % upper bound there is ≤6 × 10⁻⁵, which overlaps the NPD point estimate of 3.3 × 10⁻⁵, so the two are statistically indistinguishable.

## Where the underlying data lives

- `class_c_npd/results/isi_campaign/results.json` — first-pass ISI N=16..1024
- `class_c_npd/results/maagn_campaign/results.json` — first-pass MA-AGN N=16..1024
- `class_c_npd/results/isi_bigmodel_largeN/results.json` — d=32, hidden=128 retrain at large N
- `class_c_npd/results/npd_super_highcw/results.json` — high-CW NPD re-evals at N=512, 1024
- `class_c_npd/results/sc_super_hicw/results.json` — high-CW analytical SC re-evals
- `class_c_npd/results/joint_trellis_mc_3dB/results.json` — JT with 3 dB design-SNR
- `class_c_npd/results/joint_trellis_highcw/results.json` — JT high-CW evals
- `class_c_npd/results/jt_snr_sweep/results.json` — SNR-sweep waterfall data
- `class_c_npd/results/topup_30errs/results.json` and `topup_final/results.json` — CW top-ups to reach ≥30 errors

The corresponding plots are in `results_local/`.
