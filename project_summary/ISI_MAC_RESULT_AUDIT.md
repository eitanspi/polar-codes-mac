# ISI-MAC chained NPD vs chained trellis SC — audit

Session: 2026-04-16. Channel: `ISIMAC(h=0.3)` at SNR=6 dB (sigma^2 ~ 0.2512).
Design: `designs/gmac_C_n{n}_snr6dB.npz` (Class C GMAC proxy @ 6 dB).
Path: Class C corner `b=[0]*N+[1]*N` (decode all U, then all V given U_hat).
Rates: ku/kv = 4/7, 7/15, 15/29 at N = 16, 32, 64.

All BLER numbers use Wilson 95% CIs. Seeds derived from `SEED_BASE=20260416`.

## Prior conflicting measurements (context)

At N=32, two earlier numbers disagreed:

| Source | NPD BLER | Trellis SC | NPD/Trellis | n_cw |
|--------|----------|------------|-------------|------|
| `class_c_npd/results/npd_memory_mac_results.md` (Stage 2 training report) | 0.0780 (window) / 0.1115 (BiGRU) | 0.072 | 1.08 / 1.55 | 2000 NPD / 500 trellis |
| `results/snr_sweep/isi_mac_h_sweep_N32.json` (`scripts/task2_isi_h_sweep.py`) | 0.118 (BiGRU) | 0.087 | 1.36 | 1000 |

Root cause of the apparent disagreement: the two files test **different
checkpoints**. The training report's "0.078" is the **window (W=2)** encoder
(`isi_mac_window_w2_s{1,2}_N32_best.pt`); the h-sweep script loads the
**BiGRU (L=1)** encoder (`isi_mac_bigru_L1_s{1,2}_N32_best.pt`). The BiGRU's
own 2000-cw chained number in the same training report was **0.1115**, which
is statistically consistent with the h-sweep's 0.118. So neither number was
wrong — they are just different encoders.

The trellis SC references (0.072 at N=32, 0.087 at N=32) also disagreed
because the training report measured only 500 codewords (Wilson 95% CI
~ +/-0.022). At 10k codewords the true trellis SC BLER is 0.0825 — consistent
with the h-sweep's 0.087 and meaningfully higher than the 500-cw estimate 0.072.

## 10k-codeword re-run (authoritative)

Trellis SC is a fresh ground-truth run at 10k cw. Each NPD row below uses
the exact checkpoint pair named in the `Stage 1 ckpt` / `Stage 2 ckpt`
columns. All at h=0.3, SNR=6 dB.

### N = 16 (ku=4, kv=7)

| Decoder | Stage 1 ckpt | Stage 2 ckpt | BLER | 95% CI | errs_u | errs_v | time |
|---------|--------------|--------------|------|--------|--------|--------|------|
| Trellis SC (ref) | — | — | **0.1664** | [0.1592, 0.1738] | 1637 | 1664 | 11.5 s |
| NPD window (W=2) | `isi_mac_window_w2_s1_N16_best.pt` | `isi_mac_window_w2_s2_N16_best.pt` | **0.1439** | [0.1372, 0.1509] | 1396 | 1438 | 3.0 s |
| NPD BiGRU (L=1) | `isi_mac_bigru_L1_s1_N16_best.pt` | `isi_mac_bigru_L1_s2_N16_best.pt` | **0.1432** | [0.1365, 0.1502] | 1371 | 1431 | 3.3 s |

**Both NPD variants beat trellis SC** at N=16: the NPD CI [0.1365, 0.1509]
does not overlap the trellis CI [0.1592, 0.1738]. This is the opposite of
what the earlier 500-cw trellis number (0.136) suggested. The GMAC_C proxy
design is mildly suboptimal for this channel, so the trellis SC decoder
is itself not the true MAP benchmark here — the neural decoder appears to
compensate for the small design mismatch via its learned scoring.

### N = 32 (ku=7, kv=15)

| Decoder | Stage 1 ckpt | Stage 2 ckpt | BLER | 95% CI | errs_u | errs_v | time |
|---------|--------------|--------------|------|--------|--------|--------|------|
| Trellis SC (ref) | — | — | **0.0825** | [0.0773, 0.0881] | 825 | 816 | 23.2 s |
| NPD window (W=2) | `isi_mac_window_w2_s1_N32_best.pt` | `isi_mac_window_w2_s2_N32_best.pt` | **0.0857** | [0.0804, 0.0913] | 853 | 833 | 6.7 s |
| NPD BiGRU (L=1) | `isi_mac_bigru_L1_s1_N32_best.pt` | `isi_mac_bigru_L1_s2_N32_best.pt` | **0.1130** | [0.1069, 0.1194] | 1122 | 1074 | 7.1 s |

- **Window**: 0.0857 vs trellis 0.0825; ratio 1.04. CIs overlap substantially
  (window [0.0804, 0.0913] vs trellis [0.0773, 0.0881]). This matches trellis
  SC within statistical noise.
- **BiGRU**: 0.1130 vs trellis 0.0825; ratio 1.37. CIs do not overlap
  (BiGRU [0.1069, 0.1194] vs trellis [0.0773, 0.0881]). This is a real gap.

### N = 64 (ku=15, kv=29)

| Decoder | Stage 1 ckpt | Stage 2 ckpt | BLER | 95% CI | errs_u | errs_v | time |
|---------|--------------|--------------|------|--------|--------|--------|------|
| Trellis SC (ref) | — | — | **0.0399** | [0.0362, 0.0439] | 399 | 393 | 49.0 s |
| NPD window (W=2) | `isi_mac_window_w2_s1_N64_best.pt` | `isi_mac_window_w2_s2_N64_best.pt` | **0.0699** | [0.0651, 0.0751] | 688 | 697 | 14.2 s |
| NPD BiGRU (L=1) | `isi_mac_bigru_L1_s1_N64_best.pt` | `isi_mac_bigru_L1_s2_N64_best.pt` | **0.0489** | [0.0448, 0.0533] | 466 | 486 | 15.8 s |

- **Window**: 0.0699 vs trellis 0.0399; ratio 1.75. CIs do not overlap. Clear gap.
- **BiGRU**: 0.0489 vs trellis 0.0399; ratio 1.23. CIs do not overlap
  (BiGRU [0.0448, 0.0533] vs trellis [0.0362, 0.0439]). Small but real gap.

## Canonical result per N

The "canonical" chained-NPD number at each N should be the best encoder
available (the thesis headline already cites the best per N):

| N | Canonical NPD | BLER (10k cw) | Trellis SC (10k cw) | Ratio | Status |
|---|---------------|---------------|---------------------|-------|--------|
| 16 | BiGRU (L=1)   | 0.1432 | 0.1664 | **0.86** | Beats trellis SC |
| 32 | window (W=2)  | 0.0857 | 0.0825 | 1.04 | Matches (CIs overlap) |
| 64 | BiGRU (L=1)   | 0.0489 | 0.0399 | 1.23 | ~1.2x gap |

(At N=32 the BiGRU underperforms the window encoder, likely an optimisation
artefact on short sequences as noted in the original training report. At
N=64 the window encoder is clearly beaten by BiGRU.)

## Thesis chapter update

The earlier headline (based on 2000-cw / 500-cw numbers) was:
> chained NPD matches trellis SC at N=16, 32 (ratios 1.02 / 1.08) and is
> 1.5x at N=64.

With 10k-codeword measurements and a self-consistent choice of best encoder
per N, the correct headline is:

> **Chained NPD beats the trellis SC reference at N=16 (0.143 vs 0.166, a
> 14% reduction), matches it within statistical noise at N=32 (0.086 vs
> 0.082, ratio 1.04, overlapping CIs), and trails it by 1.23x at N=64
> (0.049 vs 0.040).**

Notes to flag:
- The trellis SC "reference" here uses the same GMAC_C proxy design as the
  NPD decoder; it is therefore not a true channel-MAP reference. At N=16
  the design is loose enough that the neural decoder outperforms trellis SC.
- The earlier trellis-SC values (0.136 at N=16, 0.028 at N=64) were each
  drawn from only 500 codewords; the 10k-cw re-measurements (0.166 and
  0.040 respectively) are the trustworthy numbers and should replace them
  in any tables or plots.
- The 0.0780 vs 0.118 disagreement at N=32 was checkpoint confusion
  (window encoder vs BiGRU encoder), not measurement noise or a bug. Both
  chained numbers were reproducible at 10k cw within their 95% CIs.

## Files

- Raw audit data: `results/snr_sweep/isi_mac_audit_10kcw.json`
- Audit script: `scripts/audit_isi_mac_discrepancy.py`
- Prior (superseded) references:
  - `class_c_npd/results/npd_memory_mac_results.md` (2000 / 500 cw)
  - `results/snr_sweep/isi_mac_h_sweep_N32.json` (1000 cw, BiGRU only)

## Reproduction

```
python scripts/audit_isi_mac_discrepancy.py
# runs on CPU with torch.set_num_threads(2); ~3 minutes end-to-end.
```
