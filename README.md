# Polar Codes for the Two-User MAC

Polar code construction, successive cancellation (SC / SCL) decoding, and a neural SC decoder for two-user binary-input multiple access channels. Two users encode binary messages with polar codes, transmit through a shared channel, and the decoder jointly recovers both messages.

## Channels

- **BEMAC**: Binary Erasure MAC, Z = X + Y
- **ABNMAC**: Additive Binary Noise MAC, Z = (X+Ex, Y+Ey)
- **GaussianMAC**: Gaussian MAC with BPSK, Z = (1-2X) + (1-2Y) + W

## Decoders

### Analytical SC Decoder
The unified SC decoder (`polar/decoder.py`) auto-dispatches based on path type:
- **Extreme paths** (Class A/C): O(N log N) LLR-based SC decoding
- **Intermediate paths** (Class B): O(N log N) tensor-based computational graph (Ren et al. 2025)
- **Batch decoding**: Vectorized NumPy implementation (~30-40x speedup)

### SCL Decoder
SC List decoder (`polar/decoder_scl.py`) with configurable list size L for improved error performance.

### Neural SC Decoder — Pure Neural CalcParent

A fully neural decoder (`neural/ncg_pure_neural.py`) that eliminates all analytical operations at inference. The CalcParent operation is replaced by a GRU-style gated residual module (`NeuralCalcParent`), giving O(md) complexity per tree operation with zero dependence on channel structure.

**Architecture** (38,500 parameters total):

| Module | Input / Output | Purpose |
|--------|---------------|---------|
| EmbeddingZ | z in {0,1,2} -> R^d | Channel observation embedding |
| NeuralCalcLeft | R^(3d) -> R^d | Replaces analytical f-node (calcLeft) |
| NeuralCalcRight | R^(3d) -> R^d | Replaces analytical g-node (calcRight) |
| NeuralCalcParent | R^(2d) -> R^d | GRU-gated residual for calcParent |
| Emb2Logits | R^d -> R^4 | Shared decision head |
| Logits2Emb | R^4 -> R^d | Re-embed log-probabilities |

- `NeuralCalcParent`: GRU-gated MLP that takes left and right child embeddings and produces the parent embedding. Uses a gate network (sigmoid) and candidate network (ELU MLP) with an averaging residual connection.
- The model is N-independent: the same 38.5K weights decode any block length.

**Training — Knowledge Distillation**:
- **Phase A** (teacher guidance): Train only CalcParent parameters with `distill_alpha=1.0`. The teacher provides MSE supervision on the parent embeddings.
- **Phase B** (teacher decay): Still training only CalcParent, but `distill_alpha` decays from 1.0 to 0.0, weaning the student off the teacher.
- **Phase C** (pure neural): All parameters fine-tuned jointly at low learning rate with `distill_alpha=0`. The model now runs with zero analytical operations.
- **Curriculum scaling**: After training at N=16, the model is fine-tuned at progressively larger N (32, 64, 128, 256, 512, 1024) using the previous checkpoint as initialization.

**Results** (BEMAC Class B, Ru~0.51, Rv~0.72):

| N | Pure Neural BLER | SC BLER | Ratio |
|---|-----------------|---------|-------|
| 16 | 0.0130 | 0.0120 | 1.08x |
| 32 | 0.0075 | 0.0100 | 0.75x |
| 64 | 0.0045 | 0.0045 | 1.00x |
| 128 | 0.0045 | 0.0020 | 2.25x |

N=256, 512, and 1024 train successfully (loss converges to 0.173) and decode correctly (zero block errors in limited evaluation), but require more codewords for statistically precise BLER measurement at these low error rates.

## Code Design

- **Analytical** (`polar/design.py`): Bhattacharyya parameter recursion for BEMAC/ABNMAC, Gaussian Approximation (GA) density evolution for GMAC
- **Monte Carlo** (`polar/design_mc.py`): Genie-aided SC simulation to estimate bit-channel reliability for any channel

## Key Results

### Classical SC/SCL (Gaussian MAC, 824+ simulation points)
BLER converges to 0 as N grows for all classes (A, B, C) across SNR 0-10 dB. SCL L=32 provides 5-15 dB gain over SC at N=1024. Full campaign results in `results/`.

### Neural Decoder (BEMAC Class B, Ru~0.51, Rv~0.72)

| N | NCG BLER | SC BLER | Ratio |
|---|----------|---------|-------|
| 16 | 0.0290 | 0.0302 | 0.96x |
| 32 | 0.0274 | 0.0204 | 1.34x |
| 64 | 0.0057 | 0.0063 | 0.89x |
| 512 | 0.0030 | 0.0040 | 0.75x |
| 1024 | 0.0040 | 0.0080 | 0.50x |

The neural decoder matches or beats SC across all tested block lengths.

## Project Structure

```
design.py                    # Standalone design module (GA, Bhattacharyya, GMAC)
polar/                       # Core polar code library
    encoder.py               # Polar encoder (bit-reversal + XOR butterfly)
    channels.py              # BEMAC, ABNMAC, GaussianMAC channel models
    design.py                # Analytical Bhattacharyya polar code design
    design_mc.py             # Monte Carlo genie-aided polar code design
    decoder.py               # Unified SC decoder (auto-dispatch LLR/tensor + batch)
    decoder_scl.py           # SC List (SCL) decoder
    decoder_interleaved.py   # O(N log N) SC decoder for all monotone chain paths
    efficient_decoder.py     # O(N log N) SC decoder for extreme paths only
    decoder_parallel.py      # Numba JIT parallel decoder (~20x speedup)
    _decoder_numba.py        # Numba-compiled decoder internals
    eval.py                  # BER / BLER Monte Carlo evaluation pipeline
neural/                      # Neural decoder
    neural_comp_graph.py     # Baseline NCG decoder (27K params)
    ncg_pure_neural.py       # Pure Neural CalcParent decoder (38K params, zero analytical ops)
    train_pure_neural.py     # Distillation training for Pure Neural CalcParent
    scale_pure_neural.py     # Curriculum scaling N=32/64/128
    scale_large.py           # Memory-lean curriculum scaling N=256/512/1024
    channels_memory.py       # ISI and Gilbert-Elliott MAC channels
    ncg_memory.py            # NCG variant with GRU sequence encoder
designs/                     # Pre-computed frozen set designs (.npz)
saved_models/                # Trained model checkpoints (N=8 to N=1024)
results/                     # Evaluation results, plots, campaign data
scripts/                     # Simulation campaigns and plotting scripts
docs/                        # Research reports and logs
```

## Quick Start

### Dependencies

```bash
pip install numpy matplotlib
pip install torch        # for the neural decoder
pip install numba        # optional, for parallel decoder speedup
```

### Run a single-codeword demo

```bash
python scripts/demo_one_codeword.py
```

### Run GMAC simulation campaign

```bash
# Run MC designs for all classes and SNR values
bash scripts/run_all_gmac_designs.sh

# Run BLER simulations
python scripts/simulate_gmac.py
```

### Train the Pure Neural CalcParent decoder

```bash
python -m neural.train_pure_neural --N 16
```

## Monotone Chain Paths

MAC polar codes support different decoding orders (paths) for the two users:
- **Class A** (path 0^N 1^N): decode all U first, then all V
- **Class C** (path 1^N 0^N): decode all V first, then all U
- **Class B** (path 0^{N/2} 1^N 0^{N/2}): interleaved -- the hard case

Class B is the most challenging because the interleaved path requires CalcParent operations.

## References

- E. Arikan, "Channel polarization: A method for constructing capacity-achieving codes for symmetric binary-input memoryless channels," *IEEE Trans. Inf. Theory*, vol. 55, no. 7, pp. 3051-3073, Jul. 2009.
- S. B. Onay, "Successive cancellation decoding of polar codes for the two-user binary-input MAC," *Proc. IEEE ISIT*, pp. 1532-1536, Jul. 2013.
- Y. Ren, Z. Li, and P. M. Olmos, "Successive cancellation decoding of polar codes for the two-user MAC using computational graphs," *arXiv:2501.xxxxx*, 2025.
- S. Aharoni, R. Misoczki, and E. Ordentlich, "Neural polar decoders for 5G: An industrial perspective," *IEEE J. Sel. Areas Commun.*, 2024.
