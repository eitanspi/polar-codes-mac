# Master Results Table: Neural Polar Decoders for Two-User MAC

**Compiled:** 2026-04-16 (latest data from all sessions through Session 12)
**Convention:** Wilson 95% CI in brackets. Bold = best decoder at that N. Reliable = >=100 errors.

---

## 1. BEMAC Class B (non-corner, symmetric rate)

**Channel:** Z = X + Y (deterministic, Z in {0,1,2})
**Path:** make_path(N, N//2), symmetric rate ~(0.75, 0.75)
**Operating rate:** R_U ~ 0.50, R_V ~ 0.70

| N | ku | kv | SC BLER | SC errs/CW | Wilson 95% CI | Reliable? | NCG BLER | NCG errs/CW | NCG CI | NCG/SC | SCL L=4 | CRC-SCL L=4 |
|---|---|---|---------|-----------|---------------|-----------|----------|-------------|--------|--------|---------|-------------|
| 16 | 8 | 11 | 0.0110 | ~55/5000 | -- | No | 0.0110 | -- | -- | 1.08x | -- | -- |
| 32 | 16 | 22 | **0.0097** | 146/15000 | [0.0083, 0.0115] | Yes | **0.0076** | 114/15000 | [0.0063, 0.0091] | **0.78x** | 0.0102 | **0.0000** |
| 64 | 32 | 44 | **0.0032** | 128/40000 | [0.0027, 0.0038] | Yes | **0.0032** | 154/48000 | [0.0027, 0.0037] | **1.00x** | 0.0006 | **0.0000** |
| 128 | 64 | 89 | **0.0016** | 81/50000 | [0.0013, 0.0020] | Partial (81) | **0.0017** | 86/50000 | [0.0014, 0.0021] | 1.06x | 0.0006 | **0.0000** |
| 256 | 128 | 178 | 8e-5 | ~4/50000 | -- | No | **4e-5** | ~2/50000 | -- | **0.50x** | 0.0000 | 0.0000 |
| 512 | 256 | 358 | 0.0000 | 0/2000 | -- | No | 0.0000 | 0/2000 | -- | -- | -- | -- |
| 1024 | 512 | 716 | 1e-4 | ~1/10000 | -- | No | 1e-4 | ~1/10000 | -- | 1.00x | -- | -- |

**Architecture:** NCG d=16, hidden=64, BEMAC vocab=3 embedding. Checkpoints: saved_models/ncg_pure_neural_N{32,64,128}.pt
**Sources:** results/reliable_evals/bemac_B_reliable.json, results/bemac/bemac_classB_ncg_reliable.json, results/crc_scl_sweep/bemac_B_crc_scl.json

---

## 2. BEMAC Class C (corner rate)

**Channel:** Z = X + Y (deterministic)
**Path:** make_path(N, N) = [0]^N [1]^N
**Operating rate:** R_U ~ 0.30, R_V ~ 0.60

| N | ku | kv | SC BLER | SC errs/CW | Wilson 95% CI | Reliable? | NCG BLER | NCG errs/CW | NCG/SC | SCL L=4 | CRC-SCL L=4 |
|---|---|---|---------|-----------|---------------|-----------|----------|-------------|--------|---------|-------------|
| 8 | 2 | 5 | 0.1100 | 329/3000 | [0.0995, 0.1215] | Yes | **0.0390** | 117/3000 | **0.36x** | -- | -- |
| 16 | 5 | 10 | 0.0920 | 276/3000 | [0.0823, 0.1027] | Yes | **0.0160** | 48/3000 | **0.18x** | -- | -- |
| 32 | 10 | 19 | 0.0990 | 298/3000 | [0.0889, 0.1101] | Yes | **0.0060** | 18/3000 | **0.06x** | 0.0076 | **0.0008** |
| 64 | 19 | 38 | 0.0550 | 165/3000 | [0.0475, 0.0636] | Yes | **0.0020** | 6/3000 | **0.03x** | 0.0053 | **0.0007** |
| 128 | 38 | 77 | 0.0245 | 491/20000 | [0.0225, 0.0268] | Yes | **0.0003** | ~6/20000 | **0.01x** | 0.0010 | **0.0000** |
| 256 | 77 | 154 | 0.0134 | 668/50000 | [0.0124, 0.0144] | Yes | **0.0002** | ~10/50000 | **0.01x** | 0.0005 | **0.0005** |
| 512 | 154 | 307 | 0.0034 | 168/50000 | [0.0029, 0.0039] | Yes | **0.0002** | ~10/50000 | **0.06x** | -- | -- |
| 1024 | 307 | 614 | 0.0004 | 26/63550 | [0.0003, 0.0006] | No (26) | **0.0002** | -- | **0.42x** | -- | -- |

**Architecture:** NCG d=16, hidden=64, BEMAC vocab=3 embedding.
**Note:** NCG beats SC by 2-100x across all N. This is the strongest NCG result.
**Sources:** results/bemac/bemac_classC_Ru30_Rv60_nn_vs_sc/bemac_nn_vs_sc_classC_50k.json, results/crc_scl_sweep/

---

## 3. GMAC Class B (non-corner, symmetric rate)

**Channel:** Z = (1-2X) + (1-2Y) + W, W ~ N(0, sigma^2), SNR = 6 dB
**Path:** make_path(N, N//2), symmetric rate ~(0.688, 0.688)
**Operating rate:** R_U = R_V ~ 0.48

| N | ku=kv | SC BLER | SC errs/CW | SC CI | Reliable? | NCG BLER | NCG errs/CW | NCG CI | NCG/SC | CRC-SCL L=4 | CRC/SC |
|---|-------|---------|-----------|-------|-----------|----------|-------------|--------|--------|-------------|--------|
| 32 | 15 | **0.0450** | 135/3000 | [0.0382, 0.0529] | Yes | 0.0503 | 151/3000 | [0.0431, 0.0586] | 1.12x | **0.0023** | **0.06x** |
| 64 | 31 | **0.0276** | 138/5000 | [0.0234, 0.0325] | Yes | 0.0282 | 141/5000 | [0.0240, 0.0332] | 1.02x | **0.0005** | **0.02x** |
| 128 | 62 | **0.0187** | 112/6000 | [0.0155, 0.0224] | Yes | 0.0230 | -- | -- | 1.23x | **0.0001** | **<0.01x** |
| 256 | 123 | **0.0060** | 30/5000 | [0.0042, 0.0085] | No (30) | 0.0230 | -- | -- | 4.5x (WALL) | **0.0000** | **<0.01x** |
| 512 | 246 | **0.0010** | est. | -- | No | 0.0123 | 123/10000 | [0.0103, 0.0146] | 12.3x | **0.0000** | -- |
| 1024 | 492 | -- | -- | -- | -- | 0.4700 | 940/2000 | [0.4482, 0.4919] | BROKEN | -- | -- |

**Architecture:** NCG d=16, hidden=64, GMAC z_encoder MLP.
**Key finding:** NCG does NOT beat SC on GMAC Class B. Wall at N=256. CRC-SCL L=4 is the dominant decoder.
**Sources:** results/reliable_evals/gmac_B_reliable.json, results/crc_scl_sweep/gmac_B_crc_scl.json

---

## 4. GMAC Class C (corner rate)

**Channel:** Z = (1-2X) + (1-2Y) + W, SNR = 6 dB
**Path:** make_path(N, N), corner rate (R_U ~ 0.23, R_V ~ 0.45)
**Note:** NPD and NCG use MI-designed frozen sets co-adapted with the neural decoder. SC uses its own DE-designed frozen set. See GMAC_CORNER_NPD_VERIFICATION.md.

| N | ku | kv | SC BLER | SC errs/CW | SC CI | Reliable? | NPD BLER | NPD errs/CW | NPD CI | NPD/SC | NCG BLER | NCG/SC | CRC-SCL L=4 |
|---|---|---|---------|-----------|-------|-----------|----------|-------------|--------|--------|----------|--------|-------------|
| 16 | 4 | 7 | 0.1620 | 1620/10000 | [0.1551, 0.1692] | Yes | **0.1070** | 1070/10000 | [0.1013, 0.1130] | **0.66x** | **0.1190** | **0.73x** | -- |
| 32 | 7 | 15 | 0.0681 | 681/10000 | [0.0634, 0.0731] | Yes | **0.0373** | 373/10000 | [0.0338, 0.0411] | **0.55x** | **0.0353** | **0.52x** | -- |
| 64 | 15 | 29 | 0.0273 | 273/10000 | [0.0243, 0.0307] | Yes | **0.0100** | 100/10000 | [0.0082, 0.0121] | **0.37x** | **0.0133** | **0.49x** | **0.0030** |
| 128 | 30 | 58 | **0.0071** | 71/10000 | [0.0056, 0.0089] | No (71) | 0.0329 | -- | -- | 4.6x | **0.0010** | **0.14x** | **0.0017** |
| 256 | 59 | 117 | **0.0016** | 31/20000 | [0.0011, 0.0023] | No (31) | **0.0003** | -- | -- | **0.19x** | -- | -- | **0.0000** |
| 512 | 119 | 233 | 0.1039 | 5197/50000 | [0.1013, 0.1066] | Yes | **0.0002** | -- | -- | **<0.01x** | -- | -- | -- |
| 1024 | 238 | 467 | 0.1612 | 8058/50000 | [0.1580, 0.1645] | Yes | **0.0000** | -- | -- | **<0.01x** | -- | -- | -- |

**Architecture:** NPD d=16, hidden=64, neural fast_ce (use_analytical=False).
**Warning:** NPD at N=128 shows regression. SC at N=512,1024 has high BLER (design issue). NN gain partly from frozen set co-adaptation.
**Sources:** class_c_npd/results/serious_npd_vs_sc_eval.json, results/reliable_evals/gmac_C_ncg_retrain_item22.json, results/crc_scl_sweep/gmac_C_crc_scl.json

---

## 5. ABNMAC Class B (non-corner, symmetric rate)

**Channel:** Z = (X xor Ex, Y xor Ey), correlated binary noise
**Path:** make_path(N, N//2), symmetric rate ~(0.60, 0.60)
**Operating rate:** R_U ~ R_V ~ 0.30

| N | ku=kv | SC BLER | SC errs/CW | SC CI | Reliable? | NCG BLER | NCG errs/CW | NCG CI | NCG/SC | CRC-SCL L=4 | CRC/SC |
|---|-------|---------|-----------|-------|-----------|----------|-------------|--------|--------|-------------|--------|
| 8 | 3 | **0.1198** | 1198/10000 | [0.1135, 0.1264] | Yes | 0.1202 | 1202/10000 | [0.1138, 0.1268] | 1.00x | -- | -- |
| 16 | 5 | **0.0629** | 629/10000 | [0.0583, 0.0678] | Yes | **0.0570** | 570/10000 | [0.0527, 0.0617] | **0.91x** | -- | -- |
| 32 | 10 | **0.0213** | 213/10000 | [0.0187, 0.0243] | Yes | **0.0182** | 182/10000 | [0.0158, 0.0210] | **0.85x** | **0.0022** | **0.10x** |
| 64 | 22 | **0.0438** | 219/5000 | [0.0385, 0.0497] | Yes | **0.0416** | 208/5000 | [0.0365, 0.0474] | **0.95x** | **0.0057** | **0.14x** |
| 128 | 45 | **0.0288** | 115/4000 | [0.0240, 0.0343] | Yes | 0.0250 | -- | -- | -- | **0.0022** | **0.08x** |

**Architecture:** NCG d=16, hidden=64.
**Note:** NCG gains are modest (0-15%). Non-monotonic SC BLER at N=64 > N=32 (rate-selection artifact).
**Sources:** results/reliable_evals/abnmac_B_reliable.json, results/crc_scl_sweep/abnmac_B_crc_scl.json

---

## 6. ABNMAC Class C (corner rate)

**Channel:** Z = (X xor Ex, Y xor Ey), correlated binary noise
**Path:** make_path(N, N), corner rate (R_U ~ 0.19, R_V ~ 0.38)

| N | ku | kv | SC BLER | SC errs/CW | SC CI | Reliable? | SCL L=4 | CRC-SCL L=4 | CRC/SC |
|---|---|---|---------|-----------|-------|-----------|---------|-------------|--------|
| 16 | 3 | 6 | 0.0620 | ?/? | -- | ? | -- | -- | -- |
| 32 | 6 | 13 | **0.0334** | 167/5000 | [0.0288, 0.0386] | Yes | -- | -- | -- |
| 64 | 13 | 26 | **0.0478** | 239/5000 | [0.0423, 0.0540] | Yes | 0.0330 | **0.0247** | **0.57x** |
| 128 | 26 | 51 | 0.0300 | 150/5000 | [0.0257, 0.0350] | Yes | 0.0144 | **0.0136** | **0.45x** |
| 256 | 51 | 102 | 0.0130 | ?/? | -- | ? | -- | -- | -- |
| 512 | 102 | 205 | 0.0110 | ?/? | -- | ? | -- | -- | -- |
| 1024 | 205 | 410 | 0.0000 | ?/? | -- | ? | -- | -- | -- |

**Note:** No neural decoder trained for this class (ABNMAC tuple output incompatible with current z_encoder). Non-monotonic SC BLER confirmed.
**Sources:** results/crc_scl_sweep/abnmac_C_crc_scl.json

---

## 7. ISI-MAC Class C (corner rate) -- Memory Channel

**Channel:** Z_t = (1-2X_t) + (1-2Y_t) + h*((1-2X_{t-1}) + (1-2Y_{t-1})) + W_t
**Parameters:** h = 0.3, SNR = 6.0 dB (sigma^2 = 0.251)
**Path:** make_path(N, N), corner rate
**Design proxy:** GMAC Class C at SNR=6 dB

### Analytical Baselines (10K CW, all reliable)

| N | ku | kv | Joint Trellis SC | CI | errs/CW | Chained Trellis SC | CI | errs/CW | Memoryless SC | CI | errs/CW |
|---|---|---|-----------------|----|---------|--------------------|----|---------|--------------|----|---------|
| 16 | 4 | 7 | 0.1664 | [0.1592, 0.1738] | 1664/10000 | 0.1689 | [0.1617, 0.1764] | 1689/10000 | 0.1866 | [0.1791, 0.1944] | 1866/10000 |
| 32 | 7 | 15 | 0.0825 | [0.0773, 0.0881] | 825/10000 | 0.0822 | [0.0770, 0.0877] | 822/10000 | 0.1129 | [0.1068, 0.1193] | 1129/10000 |
| 64 | 15 | 29 | 0.0262 | [0.0233, 0.0295] | 262/10000 | 0.0407 | [0.0370, 0.0448] | 407/10000 | 0.0790 | [0.0739, 0.0845] | 790/10000 |
| 128 | 30 | 58 | 0.0180 | [0.0156, 0.0208] | 180/10000 | 0.0223 | [0.0196, 0.0254] | 223/10000 | 0.1009 | [0.0951, 0.1070] | 1009/10000 |
| 256 | 59 | 117 | -- | -- | -- | 0.0061 | [0.0048, 0.0078] | 61/10000 | 0.2256 | [0.2175, 0.2339] | 2256/10000 |
| 512 | 119 | 233 | -- | -- | -- | 0.0041 | [0.0030, 0.0056] | 41/10000 | 0.5018 | [0.4920, 0.5116] | 5018/10000 |
| 1024 | 238 | 467 | -- | -- | -- | 0.0074 | [0.0059, 0.0093] | 74/10000 | 0.8704 | [0.8637, 0.8768] | 8704/10000 |

### NPD Results (5K CW, all reliable >=100 errors unless noted)

| N | Model | d | h | BLER | CI | errs/CW | Reliable? | NPD/Chained Trellis | NPD/Joint Trellis |
|---|-------|---|---|------|----|---------|-----------|--------------------|------------------|
| 16 | GPU curriculum | 16 | 100 | **0.1376** | [0.1283, 0.1474] | 688/5000 | Yes | **0.81x** | **0.83x** |
| 16 | BiGRU d=16 h=64 | 16 | 64 | 0.1384 | [0.1291, 0.1482] | 692/5000 | Yes | 0.82x | 0.83x |
| 32 | GPU curriculum | 16 | 100 | **0.0566** | [0.0505, 0.0634] | 283/5000 | Yes | **0.69x** | **0.69x** |
| 32 | BiGRU d=16 h=64 | 16 | 64 | 0.1130 | [0.1045, 0.1221] | 565/5000 | Yes | 1.37x | 1.37x |
| 64 | GPU curriculum | 16 | 100 | **0.0278** | [0.0236, 0.0327] | 139/5000 | Yes | **0.68x** | **1.06x** |
| 64 | d=16 h=100 standalone | 16 | 100 | 0.0318 | [0.0273, 0.0370] | 159/5000 | Yes | 0.78x | 1.21x |
| 64 | BiGRU d=16 h=64 | 16 | 64 | 0.0424 | [0.0369, 0.0486] | 212/5000 | Yes | 1.04x | 1.62x |
| 128 | d=64 h=128 BiGRU | 64 | 128 | **0.0300** | [0.0256, 0.0351] | 150/5000 | Yes | 1.35x | 1.67x |
| 128 | GPU curriculum d=16 | 16 | 100 | 0.0870 | [0.0795, 0.0951] | 435/5000 | Yes | 3.90x | 4.83x |
| 128 | d=16 h=100 standalone | 16 | 100 | 0.0740 | [0.0671, 0.0816] | 370/5000 | Yes | 3.32x | 4.11x |
| 256 | d=64 h=128 BiGRU | 64 | 128 | **0.0112** | [0.0086, 0.0145] | 56/5000 | No (56) | 1.84x | -- |

**HEADLINE RESULT:** GPU curriculum NPD beats chained trellis SC by 19-32% at N=16, 32, 64. At N=64, NPD (0.028) statistically matches joint trellis SC (0.026).
**Sources:** results/paper_style/npd_all_channels_5kcw.json, results/paper_style/isi_mac_sc_baselines.json, results/paper_style/isi_mac_memoryless_sc_baselines.json

---

## 8. Ising MAC Class C (corner rate) -- Memory Channel

**Channel:** Good state: Z = (1-2X)+(1-2Y)+W. Bad state: Z = W (pure noise). Markov flip p=0.1.
**Parameters:** sigma^2 = 0.251, p_flip = 0.1
**Path:** make_path(N, N), corner rate
**Design proxy:** GMAC Class C at SNR=6 dB

| N | ku | kv | Trellis SC | CI | errs/CW | Memoryless SC | CI | errs/CW | NPD d=16 h=100 | CI | errs/CW | NPD/Trellis | NPD/Memless |
|---|---|---|-----------|----|---------|--------------|----|---------|----------------|----|---------|-----------|----|
| 16 | 4 | 7 | **0.5704** | [0.5566, 0.5841] | 2852/5000 | 0.6160 | [0.6024, 0.6294] | 3080/5000 | 0.5916 | [0.5779, 0.6051] | 2960/5000 | 1.04x | **0.96x** |
| 32 | 7 | 15 | **0.6872** | [0.6742, 0.6999] | 3436/5000 | 0.7840 | [0.7724, 0.7952] | 3920/5000 | 0.7658 | [0.7539, 0.7773] | 3850/5000 | 1.11x | **0.98x** |
| 64 | 15 | 29 | **0.8976** | [0.8889, 0.9057] | 4488/5000 | 0.9410 | [0.9341, 0.9472] | 4705/5000 | -- | -- | -- | -- | -- |

**All entries reliable** (>=100 errors at these high BLERs).
**Key finding:** Ising MAC is extremely hard (BLER >57% at all N). NPD partially learns Ising memory (beats memoryless SC by 4-2%) but does not match trellis SC. Channel is too lossy at this operating point for meaningful coding gains.
**Sources:** results/paper_style/ising_mac_baselines.json, class_c_npd/results/npd_ising_mac/ising_mac_d16h100_results.json

---

## 9. MA-AGN MAC Class C (corner rate) -- Continuous-State Memory Channel

**Channel:** Z_t = (1-2X_t) + (1-2Y_t) + N_t, where N_t = alpha*N_{t-1} + W_t (AR(1) noise)
**Parameters:** alpha = 0.3, sigma^2 = 0.251 (SNR=6 dB)
**Path:** make_path(N, N), corner rate
**Baseline:** Memoryless GMAC SC (no trellis exists for continuous-state AR(1))

| N | ku | kv | Memoryless SC | CI | errs/CW | Reliable? | NPD (best) | NPD model | CI | errs/CW | Reliable? | NPD/SC |
|---|---|---|-------------|----|---------|-----------|-----------|-----------|----|---------|-----------|--------|
| 16 | 4 | 7 | 0.1654 | [0.1554, 0.1760] | 827/5000 | Yes | **0.1438** | BiGRU d=16 h=64 | [0.1343, 0.1538] | 719/5000 | Yes | **0.87x** |
| 32 | 7 | 15 | **0.0696** | [0.0629, 0.0770] | 348/5000 | Yes | 0.1134 | BiGRU d=32 h=128 | [0.1049, 0.1225] | 567/5000 | Yes | 1.63x |
| 64 | 15 | 29 | **0.0292** | [0.0249, 0.0342] | 146/5000 | Yes | 0.0292 | d=16 h=100 | [0.0249, 0.0342] | 146/5000 | Yes | 1.00x |
| 128 | 30 | 58 | **0.0052** | [0.0036, 0.0076] | 26/5000 | No (26) | 0.1014 | d=16 h=100 | [0.0933, 0.1101] | 507/5000 | Yes | 19.5x |

**KEY:** This is the channel where neural approach has unique value -- no analytical trellis decoder exists. NPD beats memoryless SC at N=16 (13% improvement) and matches at N=64. At N>=128, NPD struggles (the chained marginalisation hurts more than the memory helps).
**Sources:** results/paper_style/maagn_mac_baselines.json, results/paper_style/npd_all_channels_5kcw.json

---

## Summary: Where Neural Decoders Win

| Channel | Class | Best neural result | Best N range | Architecture |
|---------|-------|--------------------|-------------|-------------|
| BEMAC | C | NCG 0.01-0.06x SC | N=8-1024 | NCG d=16 h=64 |
| BEMAC | B | NCG 0.50-1.0x SC | N=32-256 | NCG d=16 h=64 |
| GMAC | C | NPD <0.01-0.66x SC | N=16-1024 | NPD d=16 h=64 (co-adapted frozen set) |
| ISI-MAC | C | NPD 0.68-0.81x trellis | N=16-64 | GPU curriculum d=16 h=100 BiGRU |
| MA-AGN | C | NPD 0.87x SC | N=16 | BiGRU d=16 h=64 |
| Ising | C | NPD 0.96x memoryless | N=16 | d=16 h=100 BiGRU |
| GMAC | B | NCG ~1.0x SC (wall at N=256) | N=32-128 | NCG d=16 h=64 |

## Summary: Where Walls Appear

| Channel | Class | Wall N | NPD/SC at wall | Root cause |
|---------|-------|--------|----------------|------------|
| GMAC | B | 256 | 4.5x (NCG) | Autoregressive cascade + weak-position broad tail |
| ISI-MAC | C | 512 | catastrophic | BiGRU out-of-distribution at 2x block length |
| MA-AGN | C | 32 | 1.6x | Chained marginalisation + mild memory (alpha=0.3) |
| Ising | C | 16 | 1.04x | Channel too lossy (>57% BLER) for neural advantage |
