# Agent Handoff: Neural Polar Decoders for MAC Channels with Memory

## Project Overview

Master's thesis project extending the Neural Polar Decoder (NPD) paper (Aharoni et al. 2024) from single-user to two-user Multiple Access Channel (MAC) polar codes, with focus on channels with memory (ISI-MAC, MA-AGN).

**User**: Master's student at BGU (Ben-Gurion University), advisor meeting periodically.
**GPU**: RTX 6000 Ada (48GB) on BGU cluster via RunAI workspace.
**Local**: Mac laptop (MPS, no CUDA).

## Repository Structure

- **Root**: `/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/`
- **GPU project**: `/gpfs0/bgu-haimp/users/eitansp/polar_project/` (on cluster)
- **NPD reference**: `/Users/ytnspybq/PycharmProjects/NPDforCourse/` (Aharoni et al. code, TensorFlow)

### Key Files
- `polar/encoder.py` — polar encoder with bit-reversal + XOR butterfly
- `polar/decoder.py` — O(N log N) SC MAC decoder
- `polar/decoder_trellis_mac_chained.py` — chained trellis SC for ISI-MAC (FB→LLR→SC)
- `polar/decoder_trellis_sc_proper.py` — proper trellis SC (state carried through tree, NEW)
- `polar/channels_memory.py` — ISIMAC channel
- `polar/channels_memory_new.py` — MAAGNMAC (AR(1) noise), TrapdoorMAC
- `polar/design_mc.py` — MC-based frozen set design, `design_from_file()`
- `neural/npd_memory_mac.py` — ChainedNPD_MAC, MemoryStageNPD, NPDTree (PyTorch)
- `designs/` — .npz frozen set design files (GMAC proxy, ISI-specific)
- `scripts/` — training, eval, design scripts
- `class_c_npd/results/` — all experiment results and checkpoints

### GPU Access
```bash
ssh bhn20
runai exec my-workspace -- bash -c "cd /gpfs0/bgu-haimp/users/eitansp/polar_project && <command>"
# For background jobs:
runai exec my-workspace -- bash -c "cd /gpfs0/bgu-haimp/users/eitansp/polar_project && nohup python3 -u script.py > /tmp/log.log 2>&1 & echo PID=\$!"
```
VPN required (Check Point Endpoint Security VPN to vpn.bgu.ac.il). User has recurring VPN issues — hotspot sometimes works when WiFi doesn't.

## Channel Models

### ISI-MAC (main focus)
```
Z_i = (1-2X_i) + (1-2Y_i) + h*((1-2X_{i-1}) + (1-2Y_{i-1})) + W_i
```
h=0.3, SNR=6dB, σ²=0.251. State: S=(X_{i-1}, Y_{i-1}), |S|=4 for joint, |S|=2 per stage.

### MA-AGN MAC
```
Z_i = (1-2X_i) + (1-2Y_i) + N_i,  N_i = α·N_{i-1} + W_i  (AR(1) noise)
```
α=0.9 (strong memory). **No analytical decoder exists** — continuous state.

### Chained Decoding (Corner Rate, Class C)
- Stage 1: Decode U, treat V as uniform noise
- Stage 2: Decode V given X̂ from Stage 1
- Path: `make_path(N, N)` — all U first, then all V

## Architecture: ChainedNPD_MAC

```python
class ChainedNPD_MAC:
    stage1 = MemoryStageNPD(extra_dim=0)   # U decoder
    stage2 = MemoryStageNPD(extra_dim=1)   # V decoder (gets X as side info)

class MemoryStageNPD:
    z_encoder = MemoryZEncoderBiGRU(d=16)  # BiGRU over full z sequence
    tree = NPDTree(d=16, hidden=100)        # checknode, bitnode, emb2llr MLPs
```

- **z_encoder**: BiGRU processes z sequence → per-position d-dimensional embeddings
- **tree**: MLPs for checknode (f-op), bitnode (g-op), emb2llr (leaf decision)
- **fast_ce**: Parallel training — teacher-forced BCE at every tree depth simultaneously, O(log N) gradient depth
- **decode**: Sequential SC decode through the tree (inherently serial per codeword)

### Critical Implementation Detail
The `decode()` method in `NPDTree` creates tensors internally. We fixed it to use `device=emb.device` so it works on GPU. Without this fix, decode only works on CPU (very slow for large N).

## Frozen Set Design

### Rates (GMAC Class C, SNR=6dB)
| N | ku | kv | R_U | R_V |
|---|---|---|---|---|
| 128 | 30 | 58 | 0.23 | 0.45 |
| 256 | 59 | 117 | 0.23 | 0.46 |
| 512 | 119 | 233 | 0.23 | 0.46 |

### Key Finding: GMAC Proxy is Near-Optimal for Analytical SC
We ran proper rate-1 genie-aided SC design on the actual ISI-MAC channel. Result:
- N=128: ISI design = GMAC design (30/30 overlap for U)
- N=256: 58/59 overlap
- N=512: 114/119 overlap

The GMAC proxy frozen set is essentially optimal for the analytical chained trellis SC on ISI-MAC at h=0.3. The ISI memory is too weak to shift the frozen set significantly.

### NPD MI-Based Design (3-Phase Approach)
Following the NPD project (Aharoni et al.):
1. Phase 1: Train at rate-1 (all positions info, no frozen)
2. Phase 2: Measure per-position MI via fast_ce at final tree depth
3. Phase 3: Eval (or fine-tune) with MI-selected frozen set

**Key finding**: MI design gives DIFFERENT positions than GMAC (57-68% overlap at N=512). The NPD's BiGRU learns different channel structure than the analytical SC.

## The Wall

NPD BLER vs analytical SC BLER on ISI-MAC:

| N | NPD (best) | SC | Ratio | Status |
|---|---|---|---|---|
| 16 | 0.138 | 0.169 | 0.82x | NPD wins |
| 32 | 0.057 | 0.082 | 0.69x | NPD wins |
| 64 | 0.028 | 0.041 | 0.68x | NPD wins |
| 128 | 0.029 | 0.022 | 1.32x | Wall starts |
| 256 | 0.011 | 0.006 | 1.83x | Wall grows |
| 512 | 0.108 | 0.003 | 43x | Catastrophic |

The wall is NOT caused by frozen sets (proven). It appears to be architectural.

## Current State of Experiments (as of 2026-05-06)

### What's Running on GPU

**Rate-1 N=512 full pipeline** just completed:
- S1 trained rate-1 for 1M iters (checkpoints every 50K)
- S2 trained rate-1 for 100K iters
- MI measured for both U (100K CW) and V (100K CW)
- Au_mi overlap with GMAC: 68/119 (U), 169/233 (V)
- **Chained eval with FIXED encoding (frozen=0) is running NOW** → results pending

### Key Results from Rate-1 Training

**S1-only eval (U decoder only)**:
- Rate-1 model + MI frozen + rate-1 v → **BLER = 0.0000** (0/5000)
- Rate-1 model + GMAC frozen → **BLER = 1.0000** (model only works with its own MI design)
- Rate-1 model + MI frozen + target-rate v → **BLER = 0.06** (distribution mismatch from v)

**Chained eval**: Results pending from the fixed eval script.

### MA-AGN α=0.9 Results (unique contribution — no analytical decoder)

| N | NPD | Memoryless SC | Ratio |
|---|---|---|---|
| 16 | 0.131 | 0.197 | 0.66x |
| 32 | 0.056 | 0.074 | 0.76x |
| 64 | 0.027 | 0.043 | 0.62x |
| 128 | 0.052 | 0.031 | 1.67x |
| 256 | 0.006 | 0.021 | 0.29x ← NPD wins by 3.5x! |

N=256 MA-AGN is a breakthrough — NPD massively outperforms memoryless SC.

## Bugs Found and Fixed

### 1. decode() CPU tensor bug (FIXED)
`NPDTree.decode()` created tensors on CPU via `torch.zeros(B, N, dtype=torch.long)`. Fixed to `torch.zeros(B, N, dtype=torch.long, device=dev)` where `dev = emb.device`. Without fix, GPU decode fails or falls back to CPU (very slow).

### 2. Frozen set design: rate-1 vs current-rate encoding (FIXED)
Initial iterative SC design generated random bits only at INFO positions (frozen=0). This biased the Pe measurement. Fixed to use proper rate-1 (ALL positions random) for genie-aided SC design.

### 3. Rate-1 eval: encoding must use frozen=0 (FOUND, BEING VERIFIED)
When evaluating a rate-1 trained model with a frozen set, the ENCODING must set frozen positions to 0. Generating all-random u (rate-1) but decoding with frozen set causes mismatch — decoder expects 0 at frozen positions but true u has random bits. This propagates errors through the SC tree.

### 4. MI measurement instability
MI measurement with different MC sample counts (30K vs 100K) gives slightly different Au_mi. A single position change can cause 5x BLER swing (e.g., 0.06 → 0.29). Need 100K+ CW for stable design.

### 5. v distribution mismatch
Rate-1 S1 model trained with rate-1 v (uniform y). When eval uses target-rate v (structured y), z distribution shifts. S1 BLER goes from 0.000 (rate-1 v) to 0.06 (target-rate v). The analytical SC handles this via uniform marginalization approximation. The neural model is more sensitive.

## Key Insights

### 1. NPD Project Approach (Aharoni et al.)
- Train ONE model at rate-1 for ONE specific N
- No curriculum, no multi-N
- Measure MI via fast_ce → select frozen set
- Eval same model (don't retrain from scratch)
- Uses POINTWISE encoder (Dense layer per position), NOT BiGRU
- Neural tree ops (checknode/bitnode/emb2llr) handle temporal dependencies

### 2. Our Approach Differences
- We use BiGRU z_encoder (handles memory in encoder, not tree)
- We initially trained at TARGET RATE with GMAC frozen (not rate-1)
- We retrained from scratch with MI design (should have fine-tuned)
- We're 2-user MAC (chained), they're single-user

### 3. Proper Trellis SC (state through tree)
Implemented `decoder_trellis_sc_proper.py` — carries (2,S,S) tensors through the polar tree instead of collapsing to scalar LLRs via FB. Result: nearly identical to FB→LLR→SC at h=0.3. The FB approximation loses very little for weak memory.

### 4. Frozen Set Investigation Conclusion
After extensive investigation:
- GMAC proxy is near-optimal for analytical SC on ISI at h=0.3
- NPD MI design finds different positions (57% overlap at N=512) but these are model-specific
- The wall is NOT caused by frozen sets
- Training approach (rate-1 vs target-rate) matters more than frozen set choice

## Files on GPU Cluster

### Checkpoints
```
class_c_npd/results/npd_rate1_n512_1M/
  rate1_s1_iter{50000..1000000}.pt   # S1 rate-1 checkpoints
  rate1_s2_iter{25000..100000}.pt    # S2 rate-1 checkpoints  
  full_design.json                    # Au_mi, Av_mi from 100K MC
  
class_c_npd/results/npd_3phase_isi/
  p1_rate1_s1_N128_best.pt           # N=128 Phase 1 rate-1 model
  p3_mi_s1_N128_best.pt              # N=128 Phase 3 retrained model

class_c_npd/results/npd_maagn_overnight/
  all_results.json                    # MA-AGN N=128 (0.052) and N=256 (0.006)
  maagn09_s1_N128_best.pt            # MA-AGN models
  maagn09_s1_N256_best.pt
```

### Design Files
```
designs/
  gmac_C_n{4..9}_snr6dB.npz          # GMAC proxy designs
  isi_mac_rate1_C_n{4..9}_snr6dB_h0.3.npz  # ISI SC designs (≈GMAC)
  isi_sc_design_N{128,256,512}.json   # ISI SC Au/Av
```

## BREAKTHROUGH RESULT (2026-05-06)

N=512 ISI-MAC chained eval:
- **NPD (rate-1 trained, MI design): BLER = 0.0000 (0/5000)**
- SC with GMAC frozen: BLER = 0.0043
- SC with MI frozen: BLER = 1.0000 (completely fails)

The MI-designed positions are positions ONLY the NPD can exploit. The analytical SC completely fails on them. The NPD learned a fundamentally different decomposition of the ISI-MAC channel.

**Critical caveat needing QA**: This result seems too good. See docs/QA_SCRIPTS.md for verification tasks. The result needs independent verification before claiming in thesis.

## What To Do Next

### Immediate: Verify Chained BLER
A fixed chained eval is running on GPU (`/tmp/eval_fix.log`). This uses:
- S1: 1M rate-1 model
- S2: 100K rate-1 model  
- Au_mi + Av_mi from 100K MC measurement
- Proper encoding (frozen=0)

If chained BLER < 0.003 → wall is broken.
If chained BLER ~ 0.06 → distribution mismatch issue remains.
If chained BLER ~ 1.0 → another bug.

### If Wall Not Broken: Next Steps
1. **Train with rate-1 u + target-rate v** — matches eval conditions, lets S1 learn the actual z distribution
2. **Try pointwise encoder** (like NPD project) instead of BiGRU — removes the sequential encoder as bottleneck
3. **Longer training** — 1M rate-1 at N=512 might not be enough
4. **Larger architecture** — d=64 h=128 got 0.108 with GMAC frozen; might do better with MI frozen

### For Thesis
The strongest results are:
1. NPD beats SC at N≤64 for all memory channels (ISI, MA-AGN)
2. MA-AGN N=256: NPD 0.006 vs SC 0.021 (3.5x better) — **unique contribution**
3. Frozen set analysis: GMAC proxy is near-optimal, wall is architectural
4. Rate-1 training + MI design finds fundamentally different positions at large N

## User Preferences
- Wants to understand things deeply, not just results
- Gets frustrated with wasted GPU time — plan carefully before launching
- Prefers short, direct answers
- Always verify before acting — "you sure about this?" is common
- Don't add TODOs without asking
- Don't add unnecessary features or refactoring
- Check existing code/results before creating new ones
- The user's VPN connection is unreliable — always use nohup for GPU jobs
