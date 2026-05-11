# Class C MAC Neural Polar Decoder

A Neural Polar Decoder (NPD) for the two-user MAC, restricted to Class C decoding paths.

## Why Class C?

The Class C path `0^N 1^N` (decode all U first, then all V) is the one corner of the MAC capacity region that **decomposes exactly into two chained single-user polar decoding problems**:

1. **Stage 1**: decode U over the *marginal* channel `p(z|x) = E_Y[p(z|x,y)]` — a standard single-user polar decoding problem over a non-Gaussian (mixture) channel.
2. **Stage 2**: decode V over the *conditional* channel `p(z|x,y)` after reconstructing `X̂` from Stage 1 — a standard single-user polar decoding problem over the clean conditional channel.

This means **any working single-user NPD implementation** plugs directly into a two-stage Class C MAC decoder via a simple chaining pipeline. No 2x2 joint probability tensors, no `CalcLeft`/`CalcRight`/`CalcParent` on joint states, no 4-class exposure bias.

The goal of this project is to demonstrate that the Aharoni et al. (2023) single-user NPD framework extends naturally to MAC channels (memoryless and with memory) when specialized to the Class C path.

## Architecture

```
                    ┌──────────────────────┐
                    │   GMAC / BEMAC / …   │  z = (1-2X) + (1-2Y) + W
                    │    channel output    │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
          ▼                                         │
┌───────────────────┐                               │
│  Stage 1 NPD      │                               │
│  (decoder for U)  │   single-user polar SC        │
│                   │   on MARGINAL channel         │
└────────┬──────────┘                               │
         │ û                                        │
         ▼                                          │
  polar_encode(û) ───► x̂                           │
                        │                           │
                        ▼                           │
                 z' = z - (1 - 2·x̂) ◄───────────────┘
                 (clean conditional channel)
                        │
                        ▼
             ┌───────────────────┐
             │  Stage 2 NPD      │
             │  (decoder for V)  │   single-user polar SC
             │                   │   on CLEAN channel
             └────────┬──────────┘
                      │ v̂
                      ▼
                    output
```

## Directory layout

- `channels/` — Channel models providing `(x, y) -> z` for training and `z -> llr_marginal / llr_conditional` for evaluation reference.
- `models/` — Single-user NPD architecture (CheckNode, BitNode, fast_ce training, sequential decode).
- `training/` — Stage 1 / Stage 2 training scripts.
- `eval/` — End-to-end chained evaluation with rigorous BLER estimates.
- `configs/` — Channel and training configurations.
- `results/` — JSON logs and checkpoints.

## Reused components from parent project

To save time, this project reuses the following from `/to_git_v2/`:

- **Polar encoder** (`polar.encoder.polar_encode_batch`, `bit_reversal_perm`)
- **Channel models** (`polar.channels.GaussianMAC`, `polar.channels.BEMAC`, `polar.channels.ABNMAC`, `polar.channels_memory.ISIMAC`)
- **Frozen set designs** (`polar.design_mc.design_from_file`, and `.npz` files in `/designs/`)
- **Analytical SC reference** (`polar.decoder.decode_batch` for ground-truth BLER)

## Target channels

Memoryless:
- GMAC: `Z = (1-2X) + (1-2Y) + W`, `W ~ N(0, σ²)`
- BEMAC: `Z = X + Y`
- ABNMAC: `Z = (X⊕E_x, Y⊕E_y)` (two binary outputs)

With memory:
- ISI-MAC: `Z[i] = (1-2X[i]) + (1-2Y[i]) + h·[(1-2X[i-1]) + (1-2Y[i-1])] + W[i]`

## Success criteria

The project succeeds if, for each channel:

1. Stage 1 NPD matches analytical SC BLER on the marginal channel at the target `ku`.
2. Stage 2 NPD matches analytical SC BLER on the clean channel at the target `kv`.
3. The chained decoder matches the analytical Class C decoder at rigorous evaluation (5000+ codewords).
4. The approach scales to `N = 1024` without architectural or training changes — inheriting the single-user NPD's proven scalability.

## Status

**Current phase:** Initial scaffold. Single-user NPD and Stage 1/Stage 2 training infrastructure being built. See `docs/progress.md` for details.
