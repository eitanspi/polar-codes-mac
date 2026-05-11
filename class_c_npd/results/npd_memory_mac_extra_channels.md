# Chained NPD on Additional Memory MAC Channels

Session: 2026-04-16 (3-hour extension)
Scope: extend the chained 2-stage NPD decoder (existing `neural/npd_memory_mac.py`
and `scripts/train_npd_memory_mac.py`, validated on ISI-MAC) to two additional
memory-MAC channels from `polar/channels_memory_new.py`:

1. Gilbert-Elliott MAC (GE-MAC) — Gaussian emissions with a 2-state Markov
   noise variance.
2. Trapdoor MAC — binary output, XOR-accumulator state + BSC noise.

Architecture (unchanged): `ChainedNPD_MAC` with `MemoryZEncoderWindow(W=2, d=16,
hidden=64)` + `NPDTree(d=16, hidden=64, n_layers=2)` for Stage 1 (U decode) and
Stage 2 (V decode | U side-info). Corner-rate Class C path.

Driver: `scripts/train_npd_memory_mac_extra.py` (new, mirrors
`train_npd_memory_mac.py` but accepts any channel and adds a memoryless-SC
baseline along with trellis SC).

All trainings used `torch.set_num_threads(2)` and ran on CPU alongside another
job (ncg_gmac at N=256 using ~4 cores).


## Gilbert-Elliott MAC

### Channel
- `GilbertElliottMAC(p_gb=0.08, p_bg=0.4, sigma2_good=sigma2*0.8, sigma2_bad=sigma2*5.0)`
  with `sigma2 = 10**(-0.6)` (reference SNR 6 dB).
- Two-state Markov chain (GOOD/BAD); bad state noise variance is 6.25× the good
  state's.
- Emission: `Z = (1-2X) + (1-2Y) + W`, `W ~ N(0, sigma2_state)`.
- `num_states = 2` (GOOD/BAD only — state transitions independent of inputs).

### Frozen set
- GMAC Class C design at the reference SNR:
  `designs/gmac_C_n{n}_snr6dB.npz` (this is a proxy — NOT designed on the actual
  Gilbert-Elliott channel; designed on a plain Gaussian MAC of equivalent
  average noise).

### Results

Chained NPD, 20000 iters per stage for window (15000 for BiGRU), batch 16:

| N  | ku/kv | Encoder | Chained NPD BLER | Memoryless SC BLER | Trellis SC BLER |
|----|-------|---------|-------------------|--------------------|-------------------|
| 16 | 4/7   | window W=2 | 0.1695 (U 0.144, V 0.167) | 0.2133 | 0.1600 |
| 32 | 7/15  | window W=2 | 0.1590 (U 0.148, V 0.149) | 0.0967 | 0.0700 |
| 32 | 7/15  | BiGRU L=1 @15K iters | 0.2380 (U 0.228, V 0.216) | 0.1000 | 0.0700 |

Samples: 2000 CWs for chained, 300 for memoryless/trellis SC at N=16; for
N=32 trellis/memoryless use 200 CWs.

### Training log summary
- N=16: Stage 1 final loss ≈ 0.134, best S1 BLER=0.14, 3.9 min. Stage 2 final
  loss ≈ 0.049, best S2 BLER|trueU=0.015, 4.4 min. Total 8.3 min.
- N=32: Stage 1 final loss ≈ 0.163, best S1 BLER=0.115, 10.4 min. Stage 2
  final loss ≈ 0.076, best S2 BLER|trueU=0.000, 11.8 min. Total 22.3 min.

### Verdict
Chained NPD works on GE-MAC and clearly **beats memoryless SC** at N=16 (0.17
vs 0.21). At N=32, however, memoryless SC (0.097) actually **outperforms**
NPD (0.159). This is surprising; likely causes:

1. The memoryless-SC baseline uses `channel.transition_prob(z, x, y, state=0)`
   which implicitly assumes GOOD-state emission — and GOOD is the dominant
   regime (stationary P(GOOD)≈0.833). So "memoryless" SC at N=32 benefits from
   a free Gaussian MAC decode most of the time with occasional burst errors.
2. The window-W=2 encoder only sees 5 z-values around position i — far too
   narrow to pick up the BAD-state bursts (which persist until the GOOD
   recovery transition).
3. At N=32 on the Class-C GMAC design, optimal usage is Trellis SC (0.07); the
   gap between NPD and Trellis is 2×, roughly what we saw for ISI-MAC N=32
   (0.078 vs 0.072).

We tried a BiGRU L=1 encoder (15K iters, same d=16) to see if a recurrent
state-aware encoder would help at N=32. It did **not** help — BiGRU-L=1
produced BLER 0.238, worse than window W=2 at 0.159. Possible reasons:
BiGRU with hidden=d/2=8 is too shallow, and 15K iters insufficient (the
window W=2 config used 20K iters). A deeper BiGRU (L=2 or L=3) trained longer
is worth a follow-up.

Next steps to close the gap with trellis SC: (a) longer BiGRU run with
L≥2, (b) wider window (W=4 or W=6), (c) GE-tailored design via
trellis-aware MC.


## Trapdoor MAC

### Channel
- `TrapdoorMAC(p_noise=0.1)`:
  `S[i] = X[i] XOR Y[i] XOR S[i-1]; Z[i] = S[i] XOR N[i]`, N ~ Bernoulli(0.1).
- Binary output, `num_states = 2`. Initial state 0.

### Frozen set
- BEMAC Class C design `designs/bemac_C_n{n}.npz` — crude proxy; BEMAC uses
  `Z = X+Y` (integer), not `Z = S XOR N` on an XOR-accumulator state. The
  ordering of synthetic-channel reliabilities will not match Trapdoor's true
  ordering. The prompt suggested we note this; we did not implement an MC
  design for Trapdoor as it would require a new genie-aided trellis decoder.

### Results

Chained NPD (window, W=2), 15000 iters per stage, batch 16:

| N  | ku/kv | Chained NPD BLER | Memoryless SC BLER | Trellis SC BLER |
|----|-------|-------------------|--------------------|-------------------|
| 16 | 4/7 (original) | 0.9460 | 1.0000 | 0.9867 |
| 16 | 2/3 (low rate) | 0.5780 (U 0.541, V 0.556) | 0.9333 | 0.7800 |
| 32 | 4/7 (low rate) | 0.8055 (U 0.763, V 0.804) | 1.0000 | 0.9650 |

Note: at the original prompt rates (ku=4/kv=7 at N=16 and ku=7/kv=15 at N=32),
all three decoders essentially fail (BLER ≥ 0.88). The trellis SC at 0.987 at
N=16 original rate confirms the design is fundamentally wrong for Trapdoor —
the BEMAC_C frozen positions are not the high-reliability positions for this
channel.

### Training log summary
- N=16 rate 2/3: Stage 1 final loss ≈ 0.327, best S1 BLER=0.495, 4.6 min.
  Stage 2 final loss ≈ 0.130, best S2 BLER|trueU=0.110, 4.4 min. Total 9.0 min.
- N=32 rate 4/7: Stage 1 final loss ≈ 0.320, best S1 BLER=0.705, 7.5 min.
  Stage 2 final loss ≈ 0.122, best S2 BLER|trueU=0.240, 5.0 min. Total 12.5 min.
- N=16 original rate 4/7 (for completeness): S1 stuck at BLER≈0.865
  throughout, chained BLER=0.946.

### Verdict
Chained NPD **does work directionally** on Trapdoor — it consistently beats
both memoryless SC and the trellis SC with the same (wrong) frozen set. At
N=16/rate 2/3: NPD 0.58 vs Trellis 0.78 vs Memoryless 0.93. At N=32/rate 4/7:
NPD 0.81 vs Trellis 0.97 vs Memoryless 1.00.

However, the absolute BLERs remain high because:
1. **The BEMAC_C design is genuinely suboptimal for Trapdoor.** The relative
   ordering of synthetic-channel reliabilities depends on the channel
   structure — XOR-accumulator memory is very different from binary sum.
2. With the wrong design, even the analytically optimal trellis SC fails at
   0.78–0.97. Chained NPD's advantage comes from learning a channel
   representation that partially compensates for the bad positions.

That the NPD beats a genie-like trellis SC here is specifically a consequence
of the design mismatch: the NPD's neural tree can partially "rescue" weak
positions that the trellis SC trusts.

Full-capacity Trapdoor results will require an MC design run on Trapdoor
itself (adapted genie-aided trellis SC in `polar/design_mc.py`, ~30-60 min
per N). Not executed this session due to time budget.


## Ising MAC

Not implemented. Priority was (1) Gilbert-Elliott and (2) Trapdoor; 3 hours
consumed by those two. An Ising MAC (`Y = X` when state=0, `Y = noise` when
state=1) would require a new channel class + `build_leaf_tensors` + a
sensible design proxy. Infrastructure (`ChainedNPD_MAC` + training driver) is
in place to plug in any channel that implements `sample_batch` and
`build_leaf_tensors`.


## Overall summary

The chained NPD for memory MAC channels extends cleanly to Gilbert-Elliott
with no architecture changes and, at N=16 on a GMAC_C proxy design, beats
memoryless SC (0.17 vs 0.21) and approaches the analytical trellis SC
(0.16). At N=32 it underperforms the memoryless SC (0.16 vs 0.10) with
the window W=2 encoder; a BiGRU L=1 encoder trained 15K iters did worse
(0.24). We attribute the N=32 gap to insufficient encoder capacity / training
iterations to pick up the GE state's burst timescale.

On Trapdoor MAC, all three decoders (memoryless SC, trellis SC, chained NPD)
fail at the prompt's original rates because the BEMAC_C frozen-set is not
suitable for an XOR-accumulator channel. At lower rates (ku=2/kv=3 for N=16;
ku=4/kv=7 for N=32) the chained NPD does beat both the memoryless SC and the
trellis SC, which is a notable positive — NPD's neural feature can partially
compensate for bad design choices. However, the absolute BLER remains high
(0.58 at N=16; 0.81 at N=32). A proper MC-based design on the Trapdoor
channel is required for competitive absolute numbers. An Ising MAC
implementation was out of scope for this session.

For full paper coverage, the missing pieces are: (1) a trellis-aware MC
design in `polar/design_mc.py` usable on any memory channel (estimated 1-2
hours of engineering) — this would unlock Trapdoor at proper rates and any
future memory channel; (2) retraining GE-MAC N=32 with BiGRU or W=4 to
match the trellis SC; (3) implementing a third channel (Ising or block-fading)
to demonstrate breadth.


## Files

- New training driver: `scripts/train_npd_memory_mac_extra.py`
- Checkpoints: `class_c_npd/results/npd_memory_mac/{ge_mac,trapdoor_mac}_window_w2_[suffix]_{s1,s2}_N{N}_best.pt`
- Raw JSON results:
  - `class_c_npd/results/npd_memory_mac/ge_mac_N16_results.json`
  - `class_c_npd/results/npd_memory_mac/ge_mac_N32_results.json`
  - `class_c_npd/results/npd_memory_mac/ge_mac_N32_bigru_results.json`
  - `class_c_npd/results/npd_memory_mac/trapdoor_N16_results.json` (original rate, fails)
  - `class_c_npd/results/npd_memory_mac/trapdoor_N16_lowrate_results.json`
  - `class_c_npd/results/npd_memory_mac/trapdoor_N32_lowrate_results.json`
- Training logs: `*.log` and `*_stdout.log` in the same directory.
