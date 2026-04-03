# 30-Hour Autonomous Agent Work Plan
# Neural SC Decoder for Two-User MAC Polar Codes — Paper Preparation

## YOUR MISSION

You are working autonomously for 30 hours to advance a research project toward a publishable paper on **Neural SC Decoding of Polar Codes for the Two-User MAC**. The paper should be similar in scope to Aharoni et al.'s NPD paper (Neural Polar Decoders for 5G) but for the MAC setting.

**CRITICAL: DO NOT STOP WORKING.** You have a 30-hour budget. After completing each task, immediately move to the next one. If a task fails, document the failure and move on. If you finish all tasks early, create new useful tasks. There should ALWAYS be something running — a simulation, a training, an analysis. Never idle.

## PROJECT LOCATION

`/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/`

## WHAT EXISTS (READ THESE FIRST)

1. **Comprehensive report**: `docs/comprehensive_report.pdf` — READ THIS FIRST. It explains everything: the system, architecture, results, what worked, what failed. 30+ pages.

2. **NPD reference paper**: `/Users/ytnspybq/PycharmProjects/polar_codes_MAC/papers/NPD_ziv_bashar.pdf` — The single-user NPD paper we're modeling our work after.

3. **NPD code**: `/Users/ytnspybq/PycharmProjects/NPDforCourse/` — Working NPD implementation.

4. **Memory channel agent prompt**: `docs/agent_prompt_memory_channels.md` — Background on extending to channels with memory.

5. **Current results**: `results/gmac_snr6dB/GMAC_CLASSB_COMPARISON.md` — Latest BLER comparison table.

6. **Key source files**:
   - `polar/` — Analytical encoder, decoder, design, channels
   - `neural/ncg_pure_neural.py` — Neural decoder architecture
   - `neural/ncg_gmac.py` — GMAC variant
   - `neural/neural_scl.py` — Neural SCL decoder
   - `neural/csrc/fast_tree_walk.cpp` — C++ accelerated forward pass

## CURRENT STATE OF RESULTS

| N | NN-SC BLER | SC BLER | Ratio | Status |
|---|-----------|---------|-------|--------|
| 32 | 0.046 | 0.046 | 1.0x | Matches SC |
| 64 | 0.026 | 0.025 | 1.03x | Matches SC |
| 128 | 0.017 | 0.016 | 1.04x | Matches SC |
| 256 | 0.015 | 0.005 | 3.0x | Gap |
| 512 | 0.008 | 0.001 | 8x | Gap (still training) |

CRC-aided NN-SCL(L=4) at N=128: BLER=0.002 (beats analytical SCL!)

The NN decoder works great for BEMAC (discrete channel) at ALL N. The gap is specific to GMAC (continuous Gaussian channel) at N≥256.

## YOUR TASK LIST (in priority order)

### PHASE 1: Paper-Critical Tasks (Hours 0-10)

#### Task 1.1: BEMAC Comprehensive Results (2 hours)
The BEMAC results are our strongest story — NN matches/beats SC at all N including 1024. But the results need to be comprehensive and well-documented for the paper.

- Run BEMAC evaluations at N=32, 64, 128, 256, 512, 1024 with 5000 codewords each
- For BOTH Class B and Class C paths
- For SC, SCL(L=4), NN-SC, NN-SCL(L=4)
- Use checkpoints from `saved_models/ncg_pure_neural_N{16-1024}.pt`
- The BEMAC model uses `ncg_pure_neural.py` (PureNeuralCompGraphDecoder) with discrete embedding (nn.Embedding(3, d))
- Channel: `from polar.channels import BEMAC`
- Design: `from polar.design import design_bemac` for Class C, MC design for Class B
- Save results to `results/bemac/bemac_comprehensive_paper.json`
- Create a publication-quality plot comparing all decoders

#### Task 1.2: GMAC Multi-SNR Evaluation (2 hours)
The paper needs results across multiple SNR values, not just SNR=6dB.

- Evaluate NN-SC at SNR = 3, 4, 5, 6, 7, 8 dB
- At N=64 and N=128 (where NN matches SC)
- Compare against analytical SC at each SNR
- Use the N=64 and N=128 checkpoints (trained at 6dB — test generalization to other SNRs)
- MC designs exist for each SNR: `designs/gmac_B_n{n}_snr{snr}dB.npz`
- Create a BLER vs SNR plot (waterfall curve)

#### Task 1.3: Training Complexity Analysis (1 hour)
The paper needs to quantify the computational cost.

- Measure: FLOPs per codeword for NN-SC vs analytical SC at each N
- Measure: inference time (ms/codeword) for NN-SC vs SC vs SCL(L=4)
- Measure: training time to reach within 10% of SC at each N
- Measure: model size (parameters, memory)
- Create a complexity comparison table

#### Task 1.4: Literature Survey — Neural MAC Decoders (2 hours)
Search the web for related work on neural decoders for MAC channels. The paper needs a thorough related work section.

- Search for: "neural decoder multiple access channel", "deep learning polar codes MAC", "neural successive cancellation MAC"
- Search for: "learned decoder joint detection", "neural NOMA decoder"
- Find papers from 2020-2026 that do neural decoding on MAC channels
- For each paper found: note the channel model, architecture, results, and how it differs from our approach
- Save findings to `docs/literature_survey_mac_neural.md`

#### Task 1.5: Theoretical Analysis — Why NN Fails at Large N (3 hours)
The paper needs a theoretical explanation for the N≥256 gap. Explore:

- Read information theory literature on capacity of learned decoders
- Analyze: what is the minimum embedding dimension d needed to represent the 2×2 probability tensor without information loss?
- Compute: mutual information I(embedding; true_tensor) as a function of d
- Analyze: error accumulation rate — if each MLP introduces ε error per operation, after K operations the total error is...
- Search web for: "error accumulation neural network depth", "representation capacity neural decoder"
- Write analysis to `docs/theoretical_analysis.md`

### PHASE 2: Extension Tasks (Hours 10-20)

#### Task 2.1: ABNMAC Neural Decoder (3 hours)
We have an ABNMAC channel but never trained a neural decoder for it.

- The ABNMAC channel: Z = (X⊕Ex, Y⊕Ey) with correlated noise
- Channel: `from polar.channels import ABNMAC`
- Design: `from polar.design import design_abnmac`
- The ABNMAC has discrete output like BEMAC — the existing discrete neural decoder should work
- Train at N=32, 64, 128 using curriculum
- Evaluate NN-SC vs SC
- This gives us a THIRD channel for the paper (BEMAC + ABNMAC + GMAC)

#### Task 2.2: Channels with Memory — ISI MAC (4 hours)
Read the NPD paper's section on channels with memory. Implement a simple ISI MAC:

```
Z[i] = (1-2X[i]) + (1-2Y[i]) + 0.3*((1-2X[i-1]) + (1-2Y[i-1])) + W[i]
```

- Implement the ISI-MAC channel class (similar to GaussianMAC but with memory)
- The z_encoder needs to handle temporal dependencies — options:
  a) Sliding window: z_encoder takes (z[i], z[i-1]) as input
  b) RNN/LSTM z_encoder
  c) 1D convolution over z
- Start with option (a) — simplest
- Train at N=32 and evaluate
- This is the "channels with memory" contribution for the paper

#### Task 2.3: Unknown Channel — DINE/MINE Approach (3 hours)
The NPD paper shows that DINE can learn channel embeddings for unknown channels.

- Read the NPD paper's DINE section
- Read the NPDforCourse code for DINE implementation
- Adapt for MAC: train a z_encoder that maximizes mutual information between Z and (X,Y)
- Test: train z_encoder on GMAC WITHOUT knowing the channel formula
- Compare to the analytical z_encoder
- This would be a key contribution — "our decoder works on unknown MAC channels"

### PHASE 3: Compute-Heavy Tasks (Hours 20-30)

#### Task 3.1: Extended GMAC Training (runs in background)
Launch long training runs:

- Continue N=512 training from best checkpoint (already has BLER=0.008)
- Start N=1024 training with curriculum from N=512
- Use the C++ accelerated forward pass (neural/csrc/fast_tree_walk.cpp)
- Save checkpoints every 5K iters
- Scripts: `neural/train_n512_long.py` (adapt for N=1024)

#### Task 3.2: CRC-Aided Results for Paper (2 hours)
Run comprehensive CRC-aided NN-SCL evaluations:

- N=32, 64, 128 with L=4 and CRC-8 (3000 codewords each)
- N=128 with L=4, L=8, L=16, L=32 and CRC-8 (1000 cw each)
- Compare against analytical CA-SCL
- This is a strong result for the paper (NN-CA-SCL beats analytical SCL at N=128)

#### Task 3.3: Publication Plots and Tables (2 hours)
Create all figures needed for a paper:

- Figure 1: BLER vs N for all decoders (BEMAC, separate plot for GMAC)
- Figure 2: BLER vs SNR waterfall curves at N=64 and N=128
- Figure 3: NN decoder architecture diagram (can be text-based)
- Figure 4: Training convergence curves (loss and BLER vs iterations)
- Figure 5: CRC-aided SCL comparison
- Table 1: Complexity comparison (FLOPs, latency, parameters)
- Table 2: Complete BLER results at SNR=6dB
- Save all to `docs/paper_figures/`

#### Task 3.4: Paper Draft Outline (1 hour)
Write a detailed paper outline following IEEE format:

- Abstract
- I. Introduction
- II. System Model (channel, polar codes, MAC)
- III. Analytical SC/SCL Decoder
- IV. Neural SC Decoder (architecture, training)
- V. Results (BEMAC, GMAC, ABNMAC, CRC-aided SCL)
- VI. Discussion (N≥256 gap, comparison with NPD)
- VII. Conclusion
- Save to `docs/paper_outline.md`

## HOW TO WORK

1. **Always have something running.** While waiting for a simulation, work on analysis or literature survey.
2. **Save results frequently.** Write JSON files, create plots, commit to git periodically.
3. **Document everything.** Each task should produce a clear output file.
4. **If a task fails, document WHY and move on.** Don't spend more than 30 minutes debugging.
5. **Use existing code.** Don't rewrite the decoder — use the existing scripts and models.
6. **Use the C++ extension** for any N≥256 training: `neural/csrc/fast_tree_walk.cpp`
7. **Use correct designs**: MC design from `designs/` for Class B, `design_bemac/design_gmac` for Class C.

## RATE POINTS (CRITICAL — USE THESE EXACT VALUES)

```python
# GMAC Class B
GMAC_RATES = {
    32:  {'ku': 15,  'kv': 15},
    64:  {'ku': 31,  'kv': 31},
    128: {'ku': 62,  'kv': 62},
    256: {'ku': 123, 'kv': 123},
    512: {'ku': 246, 'kv': 246},
    1024:{'ku': 492, 'kv': 492},
}

# BEMAC Class B (Ru~0.50, Rv~0.70)
BEMAC_RATES = {
    32:  {'ku': 16, 'kv': 22},
    64:  {'ku': 32, 'kv': 45},
    128: {'ku': 64, 'kv': 90},
    256: {'ku': 128, 'kv': 179},
    512: {'ku': 256, 'kv': 358},
    1024:{'ku': 512, 'kv': 716},
}
```

## CHECKPOINTS

- BEMAC: `saved_models/ncg_pure_neural_N{16,32,64,128,256,512,1024}.pt`
- GMAC: `saved_models/ncg_gmac_mlp_N{32,64,128,256,512,1024}.pt`
- GMAC best N=256: `saved_models/campaign_n256_sched_best.pt`
- GMAC best N=512: `saved_models/n512_long_best.pt`

## GIT — COMMIT YOUR WORK

Every 2-3 hours, commit your work:
```bash
cd /Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2
git add -A
git commit -m "Agent work: [description of what was done]"
```

## REMEMBER

- **DO NOT STOP.** 30 hours of continuous work.
- After each task, start the next immediately.
- If waiting for a simulation, do analysis/writing/literature work.
- Save everything to files. Print progress.
- The goal is a PAPER — every task should contribute to publishable content.
