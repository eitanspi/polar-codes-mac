# Polar Codes for the Two-User MAC

This repository contains the code, data and results for a master's thesis on **polar codes over two-user multiple access channels (MAC) with memory**, with a focus on **neural polar decoders (NPD)** for channels where no efficient analytical decoder exists.

## Headline result

For two memory MAC channels — **ISI-MAC** (deterministic intersymbol interference) and **MA-AGN-MAC** (continuous AR(1) noise) — a chained NPD trained at rate-1 followed by mutual-information-based frozen-set design produces **block-error rates comparable to or better than the strongest tractable analytical baseline (joint-trellis SC) at every block length N ∈ {16, …, 1024}**.

| N | ISI-MAC NPD | ISI-MAC SCT decoder | MA-AGN NPD | MA-AGN SCT decoder |
|---|---|---|---|---|
| 16 | 0.165 | 0.150 | 0.147 | 0.107 |
| 32 | 0.069 | 0.069 | 0.099 | 0.040 |
| 64 | 0.033 | 0.029 | 0.027 | 0.035 |
| 128 | 0.0127 | 0.0075 | 0.0062 | 0.0162 |
| 256 | 0.00138 | 0.00185 | 0.00068 | 0.00146 |
| 512 | 0.000307 | 0.00038 | 6.5e-5 | 0.00041 |
| 1024 | 5.5e-5 | ≤6e-5 | 4e-5 | 0.0001 |

All BLERs measured at ≥30 errors per data point (except ISI N=1024 SCT, which is upper-bounded by a 0-error eval on 50K codewords; tighter measurement is impractical for this point). Chained corner-rate, SNR = 6 dB. For ISI, h = 0.3. For MA-AGN, α = 0.5.

The full data is in `class_c_npd/results/` and summary plots in `results_local/`.

## Project history

This work began as an attempt to use NPD for **memoryless** MAC channels (GMAC, BEMAC, ABNMAC). That work hit a wall at GMAC with N = 256, where the NPD could not match the analytical SC decoder. The pre-pivot state is tagged as **`v0.1-memoryless-pivot`**.

The pivot is the observation that memory MAC channels are precisely the regime where NPD has a structural advantage: there is no efficient analytical SC for AR(1) noise, and for ISI the joint-trellis SC requires expensive forward-backward on a state lattice. A neural decoder learns the effective per-position channel and bypasses both costs.

## Repository layout (the entire top level you need to know)

```
README.md                  ← you are here
RESULTS.md                 ← consolidated comparison table with statistical notes
results_local/summary.pdf  ← 6-page printable summary: math + table + plot for each channel
results_local/             ← all plots (PNG/PDF), local-cache of cluster result JSONs
polar/                     ← analytical polar primitives (encoder, channels, decoders)
neural/                    ← NPD modules (npd_memory_mac.py is the chained NPD)
designs/                   ← frozen-set design files (.npz / .json)
class_c_npd/results/<n>/   ← canonical campaign results JSONs (one subdir per campaign)
scripts/                   ← runnable scripts:
    scripts/                local utility / canonical reproduction scripts
    scripts/plotting/       summary-PDF & plot generators
    scripts/local_analysis/ local CPU analytical-baseline runs (e.g. maagn_sc_local.py)
archive/                   ← preserved-but-separated old material:
    archive/neural_exploratory/   POC / experiment scripts predating the pivot
    archive/scripts_exploratory/  Older one-off scripts (GMAC-256 era)
    archive/results_old/          Old experiment subdirs and checkpoints
    archive/docs_stale/           Older session handoffs / agent prompts
    archive/toplevel_old/         Old top-level files (debug_sct*, etc.)
```

## Read me first

1. Open `results_local/summary.pdf` for the headline (3 pages per channel: math, table, plot).
2. Look at `RESULTS.md` for the full numerical comparison.
3. The two key code modules are `polar/decoder_trellis.py` (analytical SCT) and
   `neural/npd_memory_mac.py` (chained NPD). Channel models are in `polar/channels_memory.py`
   (ISI) and `polar/channels_memory_new.py` (MA-AGN).

## Channels (in `polar/`)

- **BEMAC** (`channels.py`) — binary erasure MAC, Z = X+Y
- **ABNMAC** (`channels.py`) — additive binary noise MAC
- **GaussianMAC** (`channels.py`) — GMAC; `GaussianMAC.from_snr_db(snr_db)` for convenience
- **ISIMAC** (`channels_memory.py`) — Z_i = (1-2X_i) + (1-2Y_i) + h·((1-2X_{i-1}) + (1-2Y_{i-1})) + W_i
- **MAAGNMAC** (`channels_memory_new.py`) — Z_i = (1-2X_i) + (1-2Y_i) + N_i, N_i = α N_{i-1} + W_i

## Decoders

### Analytical
- **`polar/decoder.py`** — standard polar SC and SCL.
- **`polar/decoder_interleaved.py`** — Ren-et-al-style computational-graph SC for two-user MAC, supports arbitrary paths through the rate region.
- **`polar/decoder_trellis_mac_chained.py`** — chained corner-rate FB→LLR→SC for ISI-MAC.
- **`polar/decoder_trellis.py`** — full joint-trellis SC for two-user MAC (the "SCT decoder" referenced in plots). Uses forward-backward on the joint (X_prev, Y_prev) state lattice.

### Neural
- **`neural/npd_memory_mac.py`** — chained two-user NPD for memory MAC. BiGRU z-encoder, neural CheckNode/BitNode/Emb2LLR tree.
- **`neural/npd_pytorch.py`** — single-user reference NPD (PyTorch port of Aharoni et al.).
- **`neural/ncg_*`** — neural computational-graph variants explored during the small-N regime.

## Reproducing the headline numbers

Each `class_c_npd/results/<campaign>/results.json` records the design indices used, eval CW count, and per-stage error counts. To reproduce, the campaign scripts (now in `archive/`) point to the matching configurations; the canonical eval relies on:

- `npd_batched_reeval.py` for NPD with high-CW batched eval on GPU.
- `jt_batched_eval.py` for joint-trellis SC with batched FB + computational-graph decode.
- `topup_final.py` for "top up codewords until ≥30 errors" runs.

## Status of the result tail

For the deepest tail (N = 1024 NPD/SCT at SNR = 6 dB), we have:
- NPD: 20 errors / 600K codewords ≈ 3.3 × 10⁻⁵
- SCT decoder (joint trellis): 0 errors / 50K codewords (95 % CI upper bound 6 × 10⁻⁵)

A 500K-codeword batched JT campaign for the SCT N=1024 point is being run on the cluster (~4 h CPU); when it finishes, this README and `results_local/all_results.pdf` will be updated with the tightened number.

## Citing / handoff

Older session handoffs and the pre-pivot literature notes are in `archive/docs_stale/`. The current state of the comparison and the chained-NPD methodology is what this README is for. See also `results_local/all_results.pdf` for the consolidated plots.
