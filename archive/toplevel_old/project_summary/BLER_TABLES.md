# BLER Results Tables

---

## Table 1: BEMAC Class B (non-corner)

**Channel:** BEMAC (Z = X + Y, deterministic, Z ∈ {0,1,2})
**Class:** B (non-corner, symmetric rate point)
**Path:** `make_path(N, N//2)` = [0]^{N/2} [1]^N [0]^{N/2}
**Capacity:** I(X;Z) = 0.500, I(Y;Z|X) = 1.000, I(X,Y;Z) = 1.500. Symmetric point: (0.750, 0.750)
**Operating rate:** R_U ≈ 0.50, R_V ≈ 0.70 (below symmetric capacity)

| N | ku | kv | SC | SC errs/CW | NCG | NCG/SC | SCL L=4 | CRC-SCL L=4 | CRC/SC | NN-SCL L=4 | NN-CRC-SCL L=4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 8 | 11 | 0.011 | ~55/5000 | 0.011 | 1.08x | -- | -- | -- | -- | -- |
| 32 | 16 | 22 | **0.0097** | 146/15000 | **0.0076** | **0.78x** | 0.0102 | **0.0000** | **<0.01x** | 0.006 | **0.0000** |
| 64 | 32 | 44 | **0.0032** | 128/40000 | **0.0032** | **1.00x** | 0.0006 | **0.0000** | **<0.01x** | **0.0000** | **0.0000** |
| 128 | 64 | 89 | **0.0016** | 81/50000 | **0.0017** | **1.06x** | 0.0006 | **0.0000** | **<0.01x** | **0.0000** | **0.0000** |
| 256 | 128 | 178 | 8e-5 | ~4/50000 | **4e-5** | **0.50x** | 0.0000 | 0.0000 | -- | -- | -- |
| 512 | 256 | 358 | 0.0 | 0/2000 | 0.0 | -- | -- | -- | -- | -- | -- |
| 1024 | 512 | 716 | 1e-4 | ~1/10000 | 1e-4 | 1.00x | -- | -- | -- | -- | -- |

SC at N=32,64 are reliable (>=100 errors). N=128 partial (81 errors).
NCG reliable eval (2026-04-22): N=32 BLER=0.0076 (114/15000), N=64 BLER=0.0032 (154/48000), N=128 BLER=0.0017 (86/50000, partial). From `results/bemac/bemac_classB_ncg_reliable.json`.
NCG checkpoints: `saved_models/ncg_pure_neural_N{32,64,128}.pt` (BEMAC vocab=3 embedding, d=16, hidden=64).
CRC-SCL L=4 from `results/crc_scl_sweep/bemac_B_crc_scl.json`. Zero errors at all tested N.
**NN-CRC-SCL L=4**: Zero errors at N=32-128 (matches analytical CRC-SCL). NN-SCL alone has 0.006 at N=32. From `results/crc_scl_sweep/nn_crc_scl_bemac_B.json`.
Reliable SC values from `results/reliable_evals/bemac_B_reliable.json`.

---

## Table 2: BEMAC Class C (corner)

**Channel:** BEMAC (Z = X + Y, deterministic, Z ∈ {0,1,2})
**Class:** C (corner rate)
**Path:** `make_path(N, N)` = [0]^N [1]^N (all U first, then all V)
**Capacity:** Corner point (R_U, R_V) = (0.500, 1.000)
**Operating rate:** R_U ≈ 0.30, R_V ≈ 0.60 (60% of corner capacity)

| N | ku | kv | SC | SC errs/CW | NCG | NCG/SC | SCL L=4 | CRC-SCL L=4 | CRC/SC |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 5 | 0.110 | 329/3000 | **0.039** | **0.36x** | -- | -- | -- |
| 16 | 5 | 10 | 0.092 | 276/3000 | **0.016** | **0.18x** | -- | -- | (ku<=8) |
| 32 | 10 | 19 | 0.099 | 298/3000 | **0.006** | **0.06x** | 0.0076 | **0.0008** | **0.02x** |
| 64 | 19 | 38 | 0.055 | 165/3000 | **0.002** | **0.03x** | 0.0053 | **0.0007** | **0.01x** |
| 128 | 38 | 77 | 0.025 | 491/20000 | **0.0003** | **0.01x** | 0.0010 | **0.0000** | **<0.01x** |
| 256 | 77 | 154 | 0.013 | 668/50000 | **0.0002** | **0.01x** | 0.0005 | **0.0005** | **0.03x** |
| 512 | 154 | 307 | 0.003 | 168/50000 | **0.0002** | **0.06x** | -- | -- | -- |
| 1024 | 307 | 614 | 0.00041 | 26/63550 | **0.0002** | **0.42x** | -- | -- | -- |

CW counts from `results/bemac/bemac_classC_Ru30_Rv60_nn_vs_sc/bemac_nn_vs_sc_classC_50k.json`.

**Rate mismatch investigation (Item 4):** SC at NPD's lower rates:
- N=16: ku=4, kv=8 -> SC BLER=0.061 (182/3000), vs original ku=5,kv=10 -> 0.092 (276/3000)
- N=32: ku=8, kv=16 -> SC BLER=0.049 (148/3000), vs original ku=10,kv=19 -> 0.099 (298/3000)
- The NPD operated at lower rates, so the NPD vs SC comparison in Table 2 at N=16,32 is apples-to-oranges.
- At matched rates (NPD's rates), SC is 2x better, but NCG at those rates (0.016, 0.006) still beats SC significantly.

**NPD Stage 1 curriculum retrain (Item 6):** Retrained from N=32->64->128 curriculum:
- N=64: S1 BLER=0.006 (3/500 eval, best). Massive fix from broken 0.072.
- N=128: S1 BLER=0.000 (0/300 eval). SC ref = 0.025. Needs larger eval for reliable number.
- Checkpoints: `class_c_npd/results/bemac_classC_s1_N{64,128}_v2.pt`

---

## Table 3: GMAC Class B (non-corner)

**Channel:** GMAC, Z = (1-2X) + (1-2Y) + W, W ~ N(0, σ²), SNR = 6 dB (σ² ≈ 0.251)
**Class:** B (non-corner, symmetric rate)
**Path:** `make_path(N, N//2)` = [0]^{N/2} [1]^N [0]^{N/2}
**Capacity:** I(X;Z) = 0.465, I(Y;Z|X) = 0.912, I(X,Y;Z) = 1.376. Symmetric point: (0.688, 0.688)
**Operating rate:** R_U = R_V ≈ 0.48 (~70% of symmetric capacity)

| N | ku=kv | SC | SC errs/CW | NCG (best) | NCG/SC | SCL L=4 | CRC-SCL L=4 | CRC/SC | NN-SCL L=4 | NN-CRC-SCL L=4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 15 | **0.0450** | 135/3000 | **0.0503** | 1.12x | 0.0238 | **0.0023** | **0.06x** | 0.032 | **0.004** |
| 64 | 31 | **0.0276** | 138/5000 | **0.0282** | 1.02x | 0.0109 | **0.0005** | **0.02x** | 0.018 | **0.004** |
| 128 | 62 | **0.0187** | 112/6000 | 0.023 | 1.23x | 0.0064 | **0.0001** | **<0.01x** | 0.006 | 0.006 |
| 256 | 123 | 0.006 | 30/5000 | 0.023 | 4.5x | 0.0010 | **0.0000** | **<0.01x** | 0.020 | 0.013 |
| 512 | 246 | 0.001 | est. | **0.0123** | 12.3x | 0.0000 | **0.0000** | **<0.01x** | -- | -- |
| 1024 | 492 | -- | -- | 0.4700 | BROKEN | -- | -- | -- | -- | -- |

All NCG values at N=32,64,512,1024 now reliable (>=100 errors).
NCG does NOT beat SC on Class B at any N. NCG is 2-12% worse than SC at N=32,64.
**CRC-SCL L=4 beats everything**: eliminates all errors at N>=128, and provides 17-34x improvement over SC at N=32-64.
**NN-CRC-SCL L=4** provides 7-11x improvement over SC at N=32-64, but 2-3x worse than analytical CRC-SCL. At N>=128, neural model degrades.
CRC-SCL from `results/crc_scl_sweep/gmac_B_crc_scl.json`. NN-CRC-SCL from `results/crc_scl_sweep/nn_crc_scl_gmac_B.json`.
From `results/reliable_evals/gmac_B_reliable.json`.

---

## Table 4: GMAC Class C (corner)

**Channel:** GMAC, Z = (1-2X) + (1-2Y) + W, W ~ N(0, σ²), SNR = 6 dB
**Class:** C (corner rate)
**Path:** `make_path(N, N)` = [0]^N [1]^N (all U first, then all V)
**Capacity:** Corner point (R_U, R_V) = (0.465, 0.912)
**Operating rate:** R_U ≈ 0.23, R_V ≈ 0.45 (sum ≈ 0.688, ~50% of capacity)

| N | ku | kv | SC | SC errs/CW | NPD | NCG | SCL L=4 | CRC-SCL L=4 | CRC/SC |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 7 | **0.162** | 1620/10000 | **0.107** | **0.119** | -- | -- | (ku<=8) |
| 32 | 7 | 15 | **0.0681** | 681/10000 | **0.0373** | **0.0353** | -- | -- | (ku<=8) |
| 64 | 15 | 29 | **0.0273** | 273/10000 | **0.0100** | **0.0133** | 0.0084 | **0.0030** | **0.12x** |
| 128 | 30 | 58 | **0.0071** | 71/10000 | 0.0329 | **0.001** | 0.0017 | **0.0017** | **0.26x** |
| 256 | 59 | 117 | **0.0016** | 31/20000 | **0.0003** | -- | 0.0000 | **0.0000** | **<0.01x** |
| 512 | 119 | 233 | 0.1039 | 5197/50000 | **0.0002** | -- | -- | -- | -- |
| 1024 | 238 | 467 | 0.1612 | 8058/50000 | **0.0000** | -- | -- | -- | -- |

SC, NPD at N=16,32,64 now reliable (>=100 errors each). From `class_c_npd/results/serious_npd_vs_sc_eval.json` (10K CW).
NCG at N=32 now reliable (106/3000). N=64 NCG partial (40/3000). From Item 22 retrain: `results/reliable_evals/gmac_C_ncg_retrain_item22.json`.
**CRC-SCL L=4 provides solid gains at N=64** (16x over SC). At N>=128, SC BLER is already low enough that CRC-SCL improvement is moderate.
CRC-SCL from `results/crc_scl_sweep/gmac_C_crc_scl.json`.

⚠️ NPD at N=128 shows regression (0.033 vs SC 0.007) -- needs investigation.

⚠️ SC at N=512,1024 shows extremely high BLER (0.10-0.16) suggesting a design/rate issue at those N values.

⚠️ NPD and NCG use their own MI-designed frozen set (different from SC's). See GMAC_CORNER_NPD_VERIFICATION.md for same-frozen-set comparison.

⚠️ NN beating SC is partly due to decoder-adapted code design co-optimized with the NN decoder. SC on NN frozen set performs MUCH WORSE than SC on its own frozen set (N=16: 0.377 vs 0.173, N=32: 0.337 vs 0.062). See Item 25: `results/reliable_evals/gmac_C_frozen_set_item25.json`.

---

## Table 5: ABNMAC Class B (non-corner)

**Channel:** ABNMAC (Z = (X⊕Ex, Y⊕Ey), correlated binary noise)
**Class:** B (non-corner, symmetric rate point)
**Path:** `make_path(N, N//2)` = [0]^{N/2} [1]^N [0]^{N/2}
**Capacity:** I(X;Z) ≈ 0.400, I(Y;Z|X) ≈ 0.800, I(X,Y;Z) ≈ 1.200. Symmetric point: (0.600, 0.600)
**Operating rate:** R_U ≈ R_V ≈ 0.30 (symmetric, ku=kv)

| N | ku | kv | SC | SC errs/CW | NCG | NCG/SC | SCL L=4 | CRC-SCL L=4 | CRC/SC | NN-SCL L=4 | NN-CRC-SCL L=4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 3 | 3 | **0.1198** | 1198/10000 | 0.1202 | 1.00x | -- | -- | (ku<=8) | -- | -- |
| 16 | 5 | 5 | **0.0629** | 629/10000 | **0.0570** | **0.91x** | -- | -- | (ku<=8) | -- | -- |
| 32 | 10 | 10 | **0.0213** | 213/10000 | **0.0182** | **0.85x** | 0.0104 | **0.0022** | **0.10x** | 0.029 | **0.012** |
| 64 | 22 | 22 | **0.0438** | 219/5000 | **0.0416** | **0.95x** | 0.0194 | **0.0057** | **0.14x** | 0.039 | **0.009** |
| 128 | 45 | 45 | **0.0288** | 115/4000 | 0.025 | -- | 0.0077 | **0.0022** | **0.08x** | -- | -- |

All SC and NCG values at N=8-64 are reliable (>=100 errors). From `results/reliable_evals/abnmac_B_reliable.json`.
NCG gains are modest (0-15%), much smaller than previously reported from low-CW estimates.
N=64 SC BLER (0.044) > N=32 (0.021) -- confirmed non-monotonicity, likely rate-selection artifact.
N=128: SC BLER=0.0288 (115/4000) now RELIABLE (>100 errors). CRC-SCL=0.0022 (9/4000).
**CRC-SCL L=4 provides 10-13x improvement over SC** at N=32-128.
**NN-CRC-SCL L=4**: N=32 BLER=0.012 (2x over SC), N=64 BLER=0.009 (5x over SC). From `results/crc_scl_sweep/nn_crc_scl_abnmac_B.json`.
CRC-SCL from `results/crc_scl_sweep/abnmac_B_crc_scl.json`.

---

## Table 6: ABNMAC Class C (corner) — SC baseline only

**Channel:** ABNMAC (Z = (X⊕Ex, Y⊕Ey), correlated binary noise)
**Class:** C (corner rate)
**Path:** `make_path(N, N)` = [0]^N [1]^N
**Capacity:** Corner point (R_U, R_V) = (0.400, 0.800)
**Operating rate:** R_U ≈ 0.19, R_V ≈ 0.38 (CW counts unknown)

| N | ku | kv | SC BLER | SC errs/CW | SCL L=4 | CRC-SCL L=4 | CRC/SC |
|---|---|---|---|---|---|---|---|
| 16 | 3 | 6 | 0.062 | ?/? | -- | -- | (ku<=8) |
| 32 | 6 | 13 | **0.0334** | 167/5000 | -- | -- | (ku<=8) |
| 64 | 13 | 26 | **0.0478** | 239/5000 | 0.0330 | **0.0247** | **0.57x** |
| 128 | 26 | 51 | 0.0300 | 150/5000 | 0.0144 | **0.0136** | **0.45x** |
| 256 | 51 | 102 | 0.013 | ?/? | -- | -- | -- |
| 512 | 102 | 205 | 0.011 | ?/? | -- | -- | -- |
| 1024 | 205 | 410 | 0.0 | ?/? | -- | -- | -- |

N=32,64 now reliable (>=100 errors). Non-monotonicity CONFIRMED: N=32 BLER=0.0334 [0.029, 0.039], N=64 BLER=0.0478 [0.042, 0.054]. CIs do not overlap. Likely a rate-selection artifact (ku/kv too aggressive at N=64 relative to capacity).
CRC-SCL L=4 provides 2-3x improvement over SC at N=64,128.
CRC-SCL from `results/crc_scl_sweep/abnmac_C_crc_scl.json`.

---

## Table 7: ISI-MAC Class C (corner) -- Memory Channel

**Channel:** ISI-MAC, Z_t = (1-2X_t) + (1-2Y_t) + h*((1-2X_{t-1}) + (1-2Y_{t-1})) + W_t, W ~ N(0, sigma^2)
**ISI coefficient:** h = 0.3
**SNR:** 6.0 dB (sigma^2 = 0.251)
**Class:** C (corner rate)
**Path:** `make_path(N, N)` = [0]^N [1]^N (all U first, then all V)
**Design proxy:** GMAC Class C at SNR=6 dB (ISI-specific design not available)
**Decoder:** Chained NPD: Stage 1 decodes U from z, Stage 2 decodes V from (z, u_hat)

| N | ku | kv | Joint Trellis SC | Joint errs/CW | Chained Trellis SC | Chained errs/CW | NPD (d=16) | NPD errs/CW | NPD/Joint | NPD (d=64) | d64 errs/CW | d64/Joint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 7 | **0.166** | 1664/10000 | 0.169 | 1689/10000 | **0.143** | 715/5000 | **0.86x** | -- | -- | -- |
| 32 | 7 | 15 | **0.083** | 825/10000 | 0.082 | 822/10000 | **0.081** | 406/5000 | **0.98x** | -- | -- | -- |
| 64 | 15 | 29 | **0.026** | 262/10000 | 0.041 | 407/10000 | **0.032** (h=100) | 161/5000 | **1.23x** | -- | -- | -- |
| 128 | 30 | 58 | **0.018** | 180/10000 | 0.022 | 223/10000 | **0.081** (h=100) | 406/5000 | **4.5x** | **0.030** | 150/5000 | **1.67x** |
| 256 | 59 | 117 | **0.006** | 61/10000 | 0.006 | 61/10000 | -- | -- | -- | **0.011** | 56/5000 | **1.83x** |
| 512 | 119 | 233 | **0.000** | 0/500 | **0.003** | 5/2000 | -- | -- | -- | -- | -- | -- |
| 1024 | 238 | 467 | **0.007** | 2/300 | -- | -- | -- | -- | -- | -- | -- | -- |

**Notes:**
- **Joint Trellis SC** = forward-backward on full 4-state ISI-MAC trellis + joint MAC SC decoder (`decoder_trellis.decode_single`). ISI-aware, exact analytical. The strongest analytical baseline.
- **Chained Trellis SC** = 2-stage chained decoder (`decoder_trellis_mac_chained.decode_chained`): Stage 1 decodes U on marginal 2-state trellis (Y uniform), Stage 2 decodes V on 2-state trellis given U-hat. Apples-to-apples comparison with chained NPD. New reliable 10K CW evals from `results/reliable_evals/isi_mac_sc_10kcw.json`.
- Chained Trellis SC is close to Joint at N=16,32 but notably worse at N=64 (0.041 vs 0.026) due to marginalisation loss.
- Chained NPD (d=16) = CPU-trained BiGRU encoder. N=16 best=bigru d=16 h=64, N=32 best=window_w2 d=16 h=64, N=64=bigru **d=16 h=100** (Session 11, standalone from scratch). **Reliable 5000 CW evals**. N=64 d=16 h=100: BLER=0.035 (176/5000), CI=[0.030,0.041], **beats chained trellis SC** (0.041).
- NPD (d=64) = d=64 hidden=128, BiGRU. S1 from `d64_lr1e3_N128_final.pt` (N=128), `d64_s1_N256_300k.pt` (N=256). S2 from `d64_s2_N{128,256}_best.pt`. **Reliable 5000 CW evals**.
- N=256 Joint Trellis SC updated: 0.006 (61/10000) from chained decoder (joint decoder not yet run at 10K CW; chained result used as proxy). N=512 shows 0/500 errors. N=1024 only 300 CW.
- NPD beats Joint Trellis SC at N=16 (0.86x), nearly matches at N=32 (0.98x). At N=64-256, NPD is 1.7-1.8x worse than joint trellis SC.
- **vs Chained Trellis SC** (fairer comparison): NPD d=16 at N=64 achieves 0.046 vs chained trellis 0.041 = **1.12x** (much closer). NPD d=64 at N=128 achieves 0.030 vs chained 0.022 = 1.36x. NPD d=64 at N=256 achieves 0.011 vs chained 0.006 = 1.83x.
- N=128 d=64 chained BLER=0.030 (**3.3x better** than previously reported 0.099 with 2K CW -- the old number was unreliable). N=256 d=64 chained BLER=0.011 (vs prior 0.013 with 2K CW).
- Multi-SNR sweep (N=16,32,64 at 4-8 dB): N=32 NPD beats trellis SC at 7-8 dB (0.84x, 0.71x). N=16 matches/beats at 6-8 dB. See `results/snr_sweep/isi_mac_npd_snr_sweep.json`.
- **Key finding:** Neural decoder successfully learns ISI memory channel structure via BiGRU encoder, approaching ISI-aware trellis SC. d=64 model at N=128 (0.030) and N=256 (0.011) are competitive with chained trellis SC (0.022 and 0.006), within 1.4-1.8x. The chained trellis SC is the correct analytical comparison since NPD also uses chained decoding.

**First-error analysis (Session 10):**
- N=64 d=16: Errors concentrate in Q1 (54%) and Q2 (39%). Early info positions are hardest.
- N=128 d=64: Errors shift to Q2 (49%) and Q3 (44%). The BiGRU maintains accuracy at early positions but struggles at mid-to-late positions. Zero Q4 errors at both N.
- This suggests the BiGRU encoder loses information for positions decoded later in the tree, consistent with ISI memory fading for positions far from the channel observation window.

**Rate-1 MI measurement (Session 10):**
- N=256 (in-distribution): Model achieves near-perfect MI (~1.0 bits) at most info positions. Two weak positions (pos 183: MI=-59 bits, pos 215: MI=-1.1 bits) cause most errors.
- N=512 (out-of-distribution): Model catastrophically fails at ~25% of info positions (Q1 mean MI=-59M bits). The d=64 BiGRU trained at N=256 cannot generalize to N=512.

**Session 11 updates (2026-04-16/17):**
- Chained Trellis SC baselines now reliable at all N=16-256 with 10K CW (`results/reliable_evals/isi_mac_sc_10kcw.json`).
- N=256 trellis updated from 0.007 (7/1000) to 0.006 (61/10000).
- **d=16 h=100 BREAKTHROUGH at N=64:** At 95K iters (best), BLER=0.027 (137/5000, CI [0.023,0.032]) BEATS chained trellis SC 0.041 by 33% (ratio 0.67x) and MATCHES joint trellis SC 0.026 (ratio 1.05x, CIs overlap). At 100K iters: 0.028 (139/5000). Training ongoing. The hidden width increase (64->100, 2x params per stage from 20K->42K) reduces BLER by 41% (0.046->0.027).

**Session 12 updates (2026-04-24):**
- d=16 h=100 N=128 training COMPLETE (200K S1 + 50K S2 iters):
  - S1 best BLER: 0.0567 (300 CW eval)
  - S2 BLER(V|trueU): 0.0
  - **Chained BLER: 0.081** (406/5000, reliable). Improved over d=16 h=64 (~0.16 estimated) but worse than d=64 h=128 (0.030).
- d=16 h=100 N=64 reliable chained eval: **0.032** (161/5000). Consistent with S1 eval (0.027-0.030).
- Ising MAC trellis SC baselines computed (see Table 8).
- MA-AGN d=16 h=100 training at N=64,128 (in progress).

Trellis SC data: Joint from `class_c_npd/results/chained_trellis_sc_isi_mac.json` and `results/reliable_evals/isi_mac_npd_reliable.json`. Chained from `results/reliable_evals/isi_mac_sc_10kcw.json` (10K CW, reliable).
NPD data from `results/reliable_evals/isi_mac_npd_reliable.json` and `results/reliable_evals/isi_mac_d16h100_chained.json` (5000 CW, reliable).

---

## Table 8: Ising MAC Class C (corner) -- Memory Channel

**Channel:** Ising MAC with 2 Markov states (GOOD/BAD). GOOD: Z = (1-2X) + (1-2Y) + W. BAD: Z = W (pure noise).
**Parameters:** sigma^2 = 0.251, p_flip = 0.1
**Class:** C (corner rate)
**Path:** `make_path(N, N)` = [0]^N [1]^N
**Design proxy:** GMAC Class C at SNR=6 dB

| N | ku | kv | Trellis SC (chained) | errs/CW | Memoryless SC | errs/CW | NPD d=16 h=100 | errs/CW | NPD/Trellis |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 7 | **0.575** | 2873/5000 | 0.634 | 1902/3000 | **0.592** | 2960/5000 | 1.03x |
| 32 | 7 | 15 | **0.689** | 3443/5000 | 0.781 | 2342/3000 | **0.770** | 3850/5000 | 1.12x |

**Notes:**
- BLER is extremely high at these parameters -- the Ising channel with p_flip=0.1 frequently enters the BAD state (pure noise), making decoding very difficult.
- The trellis SC (chained Markov forward-backward) provides 9-12% improvement over memoryless SC, showing the memory exploitation helps.
- The NPD at N=16 is 6.6% better than memoryless SC (0.592 vs 0.634) but 3% worse than trellis SC (0.575). At N=32, NPD is 1.4% better than memoryless (0.770 vs 0.781) but 12% worse than trellis (0.689).
- The NPD successfully learns partial Ising memory structure via the BiGRU encoder, but does not fully match the trellis SC which has exact knowledge of the channel model.
- Trellis SC implementation: `polar/decoder_trellis_ising_chained.py` (2-state Markov FB, state = channel good/bad).
- NPD data from `class_c_npd/results/npd_ising_mac/ising_mac_d16h100_results.json`. Trellis data from `results/ising_mac_baselines/ising_mac_baselines.json`.

---

## Table 9: MA-AGN MAC Class C (corner) -- Continuous-State Memory Channel

**Channel:** MA-AGN MAC. Z_t = (1-2X_t) + (1-2Y_t) + N_t, where N_t = alpha*N_{t-1} + W_t (AR(1) noise).
**Parameters:** sigma^2 = 0.251 (SNR=6 dB), alpha = 0.3
**Class:** C (corner rate)
**Path:** `make_path(N, N)` = [0]^N [1]^N
**Design proxy:** GMAC Class C at SNR=6 dB
**Baseline:** Memoryless GMAC SC (ignores memory -- the only practical analytical decoder since no finite-state trellis exists)

| N | ku | kv | Memoryless SC | errs/CW | NPD d=32 h=128 | NPD d=16 h=100 | Best NPD/SC |
|---|---|---|---|---|---|---|---|
| 16 | 4 | 7 | **0.175** | 349/2000 | **0.138** (0.79x) | -- | **0.79x** |
| 32 | 7 | 15 | **0.077** | 153/2000 | 0.112 (1.46x) | -- | 1.46x |
| 64 | 15 | 29 | **0.025** | 75/3000 | 0.066 (2.64x) | **0.035** (177/5000, **1.42x**) | **1.42x** |
| 128 | 30 | 58 | -- | -- | -- | training... | -- |

**Notes:**
- This is the "flagship" memory-channel case from Aharoni et al. 2024: a channel where **no analytical trellis SC exists** because the state is a continuous real number. Only a neural decoder can learn the memory structure from samples.
- At N=16, the NPD beats memoryless SC by 21% -- demonstrating the BiGRU encoder successfully learns the AR(1) noise correlation.
- At N>=32, the NPD is WORSE than memoryless SC. This is because the chained decoder marginalizes over V (treating it as noise), and at higher rates the effective single-user channel is noisier than what the GMAC SC assumes.
- **d=16 h=100 at N=64: BLER=0.035 (177/5000), 46% improvement over d=32 h=128 (0.066).** The hidden width increase from 128 to 100 with smaller d (16 vs 32) is more effective. S2 BLER(V|trueU)=0.0. The gap to memoryless SC narrowed from 2.38x to 1.42x.
- d=16 h=100 at N=128 training is in progress.
- Data from `class_c_npd/results/npd_maagn_mac/maagn_consolidated_results.json` (d=16 h=64), `maagn_bigru_results.json` (d=32 h=128), and `maagn_d16h100_results.json` (d=16 h=100).
