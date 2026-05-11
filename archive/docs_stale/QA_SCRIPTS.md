# QA Task: Verify N=512 ISI-MAC NPD Result

## Claim to Verify
Chained NPD decoder at N=512 on ISI-MAC achieves BLER=0.0000 (0/5000 CW), beating analytical SC (BLER=0.003). This would break the "wall" where NPD was previously 43x worse than SC at N=512.

## What Was Done

### Step 1: Train S1 at rate-1 (1M iters)
- All 512 U positions random, all 512 V positions random
- BiGRU z_encoder, d=16, hidden=100
- Constant lr=1e-3, batch=8
- Checkpoint: `class_c_npd/results/npd_rate1_n512_1M/rate1_s1_iter1000000.pt`

### Step 2: Measure MI for U (100K CW)
- Load S1 model, run fast_ce at final tree depth
- Per-position BCE → MI = log(2) - BCE
- Sort by MI descending → pick top 119 → Au_mi
- Result: Au_mi overlaps 68/119 with GMAC proxy

### Step 3: Train S2 at rate-1 (100K iters)
- All V positions random, true X as side info (teacher forcing)
- Same architecture, lr=1e-3, batch=8
- Checkpoint: `class_c_npd/results/npd_rate1_n512_1M/rate1_s2_iter100000.pt`

### Step 4: Measure MI for V (100K CW)
- Same process as Step 2 but for S2
- Result: Av_mi overlaps 169/233 with GMAC proxy

### Step 5: Chained eval
- Generate u (119 random at Au_mi, rest=0) and v (233 random at Av_mi, rest=0)
- Encode: x=polar_encode(u), y=polar_encode(v)
- Channel: z = ISI-MAC(x, y) with h=0.3, SNR=6dB
- S1 decodes u_hat from z using fu_mi frozen set
- S2 decodes v_hat from z + x_hat side info using fv_mi frozen set
- Check errors at Au_mi and Av_mi positions only
- Result: 0/5000 errors

## Scripts Used

### The eval script (ran on GPU):
Located at: `/gpfs0/bgu-haimp/users/eitansp/polar_project/gpu_eval_fix.py`
Also saved locally at: `/tmp/gpu_eval_fix.py`

### The S2 training + MI measurement script:
Located at: `/gpfs0/bgu-haimp/users/eitansp/polar_project/gpu_s2_rate1_full_eval.py`
Also saved locally at: `/tmp/gpu_s2_rate1_full_eval.py`

### The S1 training script (1M rate-1):
Located at: `/gpfs0/bgu-haimp/users/eitansp/polar_project/gpu_rate1_n512_1M.py`
Also saved locally at: `/tmp/gpu_rate1_n512_1M.py`

### The design file:
Located at: `/gpfs0/bgu-haimp/users/eitansp/polar_project/class_c_npd/results/npd_rate1_n512_1M/full_design.json`
Contains: Au_mi (119 positions), Av_mi (233 positions)

## What to Verify

### 1. Sanity check: Is the channel actually noisy?
Run the analytical SC decoder on the SAME test data (same seed=777, same Au_mi/Av_mi or GMAC frozen). SC should get ~0.003 BLER. If SC also gets 0, something is wrong with the test setup.

### 2. Verify encoding is correct
- u should have random bits ONLY at Au_mi positions, rest=0
- v should have random bits ONLY at Av_mi positions, rest=0
- x = polar_encode(u) and y = polar_encode(v) should produce valid codewords
- z = ISI-MAC(x, y) should add noise

### 3. Verify error checking
- Errors should be checked ONLY at info positions (Au_mi for U, Av_mi for V)
- The frozen positions in u_hat should match (both 0)
- Verify that u_hat actually differs from all-zeros (model is making real decisions)

### 4. Check for trivial solutions
- Is the model just outputting 0 everywhere? If u is mostly 0 (393/512 frozen), and the model outputs 0 for everything, it would get ~50% of the 119 info positions wrong. 0/5000 errors rules this out, but verify.

### 5. Distribution mismatch concern
- The model was trained at rate-1 (all random u, all random v)
- Eval uses target-rate (frozen=0)
- Previous tests showed this mismatch causes 6-29% BLER when Av was GMAC proxy
- With Av_mi, it's suddenly 0%. Why? Verify this isn't an artifact of the specific Av_mi positions.

### 6. MI measurement stability
- MI measurement with 30K CW gave different Au_mi than 100K CW (1-2 positions differ)
- Small changes caused 5x BLER swings before
- Run MI measurement again with different seed — does Au_mi change? Does BLER change?

## Key Files for QA

```
# GPU cluster access
ssh bhn20
runai exec my-workspace -- bash -c "<command>"

# Checkpoints
/gpfs0/bgu-haimp/users/eitansp/polar_project/class_c_npd/results/npd_rate1_n512_1M/
  rate1_s1_iter1000000.pt    # S1 model
  rate1_s2_iter100000.pt     # S2 model
  full_design.json           # Au_mi, Av_mi

# Code
/gpfs0/bgu-haimp/users/eitansp/polar_project/neural/npd_memory_mac.py  # model definition
/gpfs0/bgu-haimp/users/eitansp/polar_project/polar/encoder.py          # polar encoding
/gpfs0/bgu-haimp/users/eitansp/polar_project/polar/channels_memory.py  # ISI-MAC channel
/gpfs0/bgu-haimp/users/eitansp/polar_project/polar/design_mc.py        # design utilities

# Local code (same files)
/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/
```

## Channel Parameters
- ISI-MAC: Z_i = (1-2X_i) + (1-2Y_i) + 0.3*((1-2X_{i-1}) + (1-2Y_{i-1})) + W_i
- SNR = 6 dB → σ² = 10^(-0.6) ≈ 0.251
- N = 512, ku = 119, kv = 233
- Chained decoding: Stage 1 decodes U (V marginalized), Stage 2 decodes V (given X_hat)

## Previous Results for Context
| N | NPD (GMAC frozen, target-rate trained) | SC | 
|---|---|---|
| 64 | 0.028 | 0.041 |
| 128 | 0.029 | 0.022 |
| 512 | 0.108 | 0.003 |

The new claim: NPD with rate-1 training + MI design = 0.0000 at N=512.
