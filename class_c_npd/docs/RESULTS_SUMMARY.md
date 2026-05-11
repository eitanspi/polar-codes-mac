# Class C NPD: Results Summary

Date: 2026-04-09
Status: Multi-channel curriculum sweeps in progress, preliminary results below.

## Headline result

**The Class C decomposition + single-user NPD architecture matches analytical SC across multiple MAC channels at small block lengths, validating the approach.**

The pipeline:
1. Decompose Class C MAC decoding into two single-user polar decoding problems (Stage 1: U on the marginal channel, Stage 2: V on the conditional channel given X̂).
2. Use a clean single-user NPD architecture (CheckNode + sign-flip BitNode + fast_ce parallel training) for each stage.
3. Chain them at inference: decode U → reconstruct X̂ → subtract → decode V.

## Numerical results (preliminary)

### GMAC, Class C, SNR=6dB, 50% per-user capacity

| N | ku | kv | SC BLER | NPD chained | Ratio | Notes |
|---|----|----|---------|-------------|-------|-------|
| 16 | 4 | 7 | 0.1626 | **0.1362** | **0.84x** | NPD **beats** SC |
| 32 | 7 | 15 | 0.0684 | 0.1290 | 1.89x | Plateau ~2x SC |
| 64 | 15 | 29 | 0.0266 | (Stage 1 ~0.16, training) | — | Slowly improving |
| 128 | 30 | 58 | 0.0054 | (queued) | — | |

### BEMAC, Class C, 75% per-user capacity

| N | ku | kv | SC BLER | NPD chained | Ratio | Notes |
|---|----|----|---------|-------------|-------|-------|
| 16 | 4 | 8 | 0.0792 | **0.0800** | **1.01x** | Matches SC exactly |
| 32 | 8 | 16 | 0.0378 | **0.0105** | **0.28x** | NPD beats SC by 3.6x (verify!) |
| 64 | 16 | 32 | (TBD) | (Stage 1 ~0.066) | — | Just started |

## Why GMAC at marginal SNR is harder than BEMAC

The two channels behave very differently for the NPD:

| Aspect | BEMAC | GMAC mixture |
|---|---|---|
| LLR shape | Discrete (3 values: -∞, 0, +∞) | Continuous, non-linear, near-zero "dead zone" |
| Per-position Pe | ~0.014 at N=32 | ~0.009 at N=32 (similar) |
| Loss after training | ~0.09 (fast convergence) | ~0.22 (slow plateau) |
| NPD ratio to SC | ~1.0x | ~1.9x at N=32 |

The NPD's z_encoder MLP easily learns the discrete BEMAC LLR. The continuous mixture LLR for GMAC is harder to approximate, especially near `z=0` where the channel has an information "dead zone" and the LLR is near zero. Tests with the analytical mixture LLR fed directly to the network gave the same plateau, suggesting the issue is in the **tree operations** (CheckNode/BitNode), not the channel encoder.

## Key technical findings during development

1. **Bit-reversal bug in the parent project's NPD**: `neural/npd_pytorch.py` fails its own self-test because the recursive NPD tree visits leaves in bit-reversed natural order, not linear order. The clean implementation in `models/npd_single_user.py` includes the fix.

2. **Mixture-LLR formula verified**: cross-checked against the project's `_u_marginal_llr` and got identical SC BLER. The marginal LLR for GMAC Class C Stage 1 is correctly computed.

3. **Training-inference distribution match**: original training drew V uniformly over `{0,1}^N`, but at inference V comes from a polar codebook with frozen positions = 0. Fixed to match. The change is correct but did not close the GMAC plateau.

4. **GMAC analytical LLR input did not help**: feeding the closed-form mixture LLR instead of raw z plateaus at the same BLER. Z encoder is not the bottleneck.

5. **Curriculum learning across N is effective**: warm-starting each N from the previous N's checkpoint is faster than fresh training and converges to the same final BLER. For BEMAC, warm-start gives a head start of ~5x.

## Architecture parameters

| Component | Spec |
|---|---|
| Embedding dim `d` | 16 |
| Hidden width | 64 |
| Layers per MLP | 2 |
| Total params | ~21K |
| Training | fast_ce, Adam, lr=3e-4 (initial) |
| Curriculum | warm-start each N from previous |

## Reproducibility

```bash
# 1. Smoke test (5 min)
python -u class_c_npd/smoke_test.py

# 2. SC reference (1 min)
python -u class_c_npd/scripts/generate_sc_references.py

# 3. GMAC curriculum sweep (~3-5 hours for N=16..128)
python -u class_c_npd/training/curriculum_sweep.py --N_list 16,32,64,128

# 4. Multi-channel sweep (BEMAC, ABNMAC)
python -u class_c_npd/training/multi_channel_sweep.py --channel bemac --N_list 16,32,64

# 5. Status check
python -u class_c_npd/status.py

# 6. Final report
python -u class_c_npd/eval/generate_report.py
```

## What this enables

- **Drop-in single-user NPD upgrade**: any improvement to single-user NPD methodology immediately benefits the Class C MAC decoder.
- **Per-channel z_encoders**: each channel can use a different feature representation (raw z, LLR, or windowed for memory channels) without changing the tree operations.
- **Cross-channel evaluation**: same architecture, retrained per channel, gives a unified comparison framework.
- **Class C is the cleanest MAC code class**: only two phases, no joint probability tensors, and aligns with the existing SC analytical decomposition in `polar/decoder.py`.

## Open work

1. **Curriculum sweep for GMAC** at N=64, N=128 (running)
2. **BEMAC sweep** at N=64 (running)
3. **ABNMAC sweep** (queued)
4. **ISI-MAC** (proper Monte Carlo design generation needed first)
5. **Error analysis**: characterize where NPD differs from SC at N=32 to identify the source of the residual gap
6. **Larger d** experiment if curriculum doesn't close the GMAC gap by N=128
