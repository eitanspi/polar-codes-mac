# Agent Prompt: Extending Neural MAC Decoder to Channels with Memory

## Your Role
You are helping develop a neural successive cancellation (SC) decoder for polar codes on a two-user Multiple Access Channel (MAC) with MEMORY. The project already has a working decoder for memoryless channels. Your job is to extend it to channels with memory (e.g., ISI channels, channels with state).

## Project Location
`/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/`

## Background: What Exists

### The Channel Model (Memoryless, Current)
Two users transmit polar-coded messages through a Gaussian MAC:
```
Z = (1-2X) + (1-2Y) + W,  where W ~ N(0, sigma^2)
```
X = polar_encode(U), Y = polar_encode(V). The receiver observes Z and must decode both U and V.

### Code Structure
- `polar/encoder.py` — Polar encoder (bit-reversal + XOR butterfly), O(N log N)
- `polar/channels.py` — BEMAC, ABNMAC, GaussianMAC channel models
- `polar/decoder.py` — Analytical SC decoder (auto-dispatches LLR/tensor based on path)
- `polar/decoder_scl.py` — SC List decoder
- `polar/decoder_interleaved.py` — O(N log N) SC for all monotone chain paths (Ren et al. 2025)
- `polar/design.py` — Bhattacharyya + Gaussian Approximation design, including `design_gmac()`
- `polar/design_mc.py` — Monte Carlo genie-aided design (critical for Class B paths)
- `polar/eval.py` — BER/BLER Monte Carlo evaluation

### Neural Decoder Architecture
File: `neural/ncg_pure_neural.py` (PureNeuralCompGraphDecoder)
File: `neural/ncg_gmac.py` (GmacNeuralCompGraphDecoder)

The neural decoder replaces ALL analytical tree operations with learned MLPs:
- **z_encoder**: Maps channel output z (continuous float) to d-dimensional embedding. `Linear(1, 32) -> ELU -> Linear(32, d)`.
- **CalcLeft** (f-node): `MLP(3d -> hidden -> hidden -> d)`. Takes parent + right child embeddings, produces left child.
- **CalcRight** (g-node): `MLP(3d -> hidden -> hidden -> d)`. Takes parent + left child embeddings, produces right child.
- **CalcParent**: Gated residual MLP combining left + right children into parent.
- **emb2logits**: `MLP(d -> hidden -> 4)`. Maps embedding to 4-class joint (u,v) log-probabilities.
- **logits2emb**: `MLP(4 -> hidden -> d)`. Re-embeds decisions.

All operations are **weight-shared** across tree positions and depths. Model trained at one N works at any N. Parameters: d=16, hidden=64, ~39K total params.

### Training
- Sequential tree walk following path b (determines U/V decoding order)
- Teacher forcing: true bits fed during training
- Cross-entropy loss on 4-class joint (u,v) predictions at leaf positions
- Curriculum learning: N=16 → 32 → 64 → 128 → 256
- Stable cosine LR decay (no warm restarts) is critical

### Current Results (GMAC, SNR=6dB, Class B)

| N | NN BLER | SC BLER | Ratio |
|---|---------|---------|-------|
| 32 | 0.046 | 0.046 | 1.0x |
| 64 | 0.026 | 0.025 | 1.03x |
| 128 | 0.017 | 0.016 | 1.04x |
| 256 | 0.019 | 0.005 | 3.1x |

N ≤ 128 essentially matches SC. N=256 has a 3.1x gap despite extensive training (100K iters, 16 hours).

### Key Findings from Development
1. **GA design is wrong for Class B** — must use MC genie-aided design for interleaved paths
2. **Freeze & extend** works at N=128: freeze shared CalcLeft/CalcRight, add level-specific MLPs for the new deepest level
3. **Fast-CE (NPD-style parallel training)** doesn't work for the 4-class MAC joint structure — plateaus at loss=0.30
4. **Residual connections, snapshot training, multi-depth aux loss** — all tried, none helped at N=256
5. **CalcParent output = XOR of children's message bits** (validated) — but using this as aux loss doesn't help training
6. **C++ extension** for tree walk forward pass gives 1.34x speedup (`neural/csrc/fast_tree_walk.cpp`)

### NPD Reference (Single-User Neural Polar Decoder)
Paper: `/Users/ytnspybq/PycharmProjects/polar_codes_MAC/papers/NPD_ziv_bashar.pdf`
Code: `/Users/ytnspybq/PycharmProjects/NPDforCourse/`

The NPD (Aharoni et al.) works for single-user channels including channels with memory:
- Uses `fast_ce` parallel training: O(log N) gradient depth instead of O(N log N)
- Binary output (not 4-class) with sign-flip BitNode: `output = MLP(e_odd*u_sign, e_even) + e_odd*u_sign + e_even`
- For channels with memory: uses DINE (Deep InfoNCE) to learn channel embeddings
- Handles ISI channels, channels with feedback, unknown channels

### Decoding Paths (MAC-specific)
- **Class C** (path 0^N 1^N): All U first, then all V. No CalcParent needed.
- **Class B** (path 0^{N/2} 1^N 0^{N/2}): Interleaved. CalcParent needed. Hardest case.
- **Class A** (path 1^N 0^N): All V first, then all U.

### Pre-computed Resources
- MC frozen set designs: `designs/gmac_B_n{3..11}_snr{0..10}dB.npz`
- Trained checkpoints: `saved_models/ncg_gmac_mlp_N{32..1024}.pt`
- Results: `results/gmac_snr6dB/` (extensive simulation data)

### Reports
- Full report (18 pages): `docs/report_full.pdf`
- 5-page summary: `docs/report_5page.pdf`
- 2-page executive summary: `docs/report_2page.pdf`

### Memory Files (Session History)
- `/Users/ytnspybq/.claude/projects/-Users-ytnspybq-PycharmProjects-polar-codes-MAC/memory/MEMORY.md` — Index of all memory files
- `project_session5_handoff.md` — Neural SCL breakthrough, 48hr training
- `project_session6_progress.md` — Bug fixes, freeze-extend, POC experiments

## Your Task: Channels with Memory

The current system assumes a memoryless channel: each z[i] depends only on x[i] and y[i]. For channels with memory:
- z[i] depends on x[i], y[i], AND previous transmissions (e.g., ISI: z[i] = h0*s[i] + h1*s[i-1] + noise where s = x + y)
- The channel embedding z_encoder must capture temporal dependencies
- The polar code design may need adaptation for memory channels

### What the NPD Paper Does for Memory Channels
Read the NPD paper (Section on channels with memory) and the NPDforCourse code to understand:
1. How they handle ISI channels
2. How DINE/MINE is used for channel embedding learning
3. How the decoder architecture changes (if at all) for memory channels

### Key Questions to Address
1. How should the z_encoder change to handle memory? (RNN? Transformer? Sliding window?)
2. Does the polar code design need to change for MAC with memory?
3. Can we reuse the trained CalcLeft/CalcRight/CalcParent from the memoryless model?
4. What is the simplest memory channel to start with as a POC?

### Starting Points
- Read the NPD paper's memory channel section
- Read `NPDforCourse/models/` for the implementation
- Start with a simple ISI-MAC: Z[i] = (1-2X[i]) + (1-2Y[i]) + 0.5*((1-2X[i-1]) + (1-2Y[i-1])) + W[i]
- Implement the channel model first, then adapt the z_encoder
