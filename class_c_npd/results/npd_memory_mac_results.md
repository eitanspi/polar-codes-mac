# NPD Memory MAC — chained decoder results (ISI-MAC, Class C)

Session date: 2026-04-16.
Channel: `ISIMAC(h=0.3)` at SNR = 6 dB (sigma^2 = 10^(-6/10) ≈ 0.2512).
Rates: from GMAC_C proxy design — `ku/kv = 4/7, 7/15, 15/29, 30/58` for `N = 16/32/64/128`.
Design: Class C corner-rate path `b = [0]*N + [1]*N` (decode all U, then all V).

All BLER numbers are per-codeword block error on the MAC information bits
(error if any info-position U or V differs from truth). Chained: Stage 1 decodes
U, then Stage 2 decodes V given the decoded `U_hat`.

## Summary

| N | Encoder | Stage 1 BLER | Chained BLER | Broken NPD (old) | Trellis SC (analytical) | Memoryless SC | NCG window (if tracked) |
|---|---------|--------------|--------------|------------------|-------------------------|---------------|-------------------------|
| 16 | window (W=2) | 0.1133 | **0.1325** | 0.744 | 0.136 | 0.73 | (not run at N=16) |
| 16 | bigru (1 layer) | 0.1133 | 0.1465 | 0.744 | 0.136 | 0.73 | (not run) |
| 32 | window (W=2) | 0.1067 | **0.0780** | 0.876 | 0.072 | 0.58 | 0.69 |
| 32 | bigru (1 layer) | 0.1100 | 0.1115 | 0.876 | 0.072 | 0.58 | 0.69 |
| 64 | window (W=2) | 0.0800 | 0.0695 | 0.976 | 0.028 | — | 0.47 |
| 64 | bigru (1 layer) | 0.0300 | **0.0425** | 0.976 | 0.028 | — | 0.47 |
| 128 | bigru d=16 | 0.1600 | 0.2225 | — | 0.018 | 0.095 | — |
| 128 | bigru d=32 h=128 | 0.1550 | 0.1560 | — | 0.018 | 0.095 | — |
| 128 | bigru d=64 h=128 | 0.0700 | 0.0980 | — | 0.018 | 0.095 | — |
| 128 | bigru d=64 +cont | 0.0567 | **0.0890** | — | 0.018 | 0.095 | — |

*Broken NPD numbers are from `class_c_npd/results/isi_mac_classC_npd.json` (Stage 1 only — the baseline never closed on a useful Stage 2).* Memoryless SC (0.73/0.58) from NCG-window file and the prompt, provided for reference. Bold entries mark the best chained result at each N.

Takeaways:
- **Every chained NPD variant beats the broken NPD baseline by 5–23×.** At N=64 the improvement is 0.976 → 0.0425, a 23× reduction in BLER.
- **At N=16 and N=32, chained NPD is within 1.02–1.08× of the analytical trellis SC**, which is the reference that explicitly marginalises the ISI state chain. The neural decoder effectively learns the effective marginal from raw z without any explicit trellis machinery.
- **At N=64 the gap to trellis SC is 1.5×** (BiGRU) or 2.5× (window). A recurrent encoder helps more as N grows, matching the intuition that a fixed-width window discards long-range ISI information that a BiGRU can preserve.
- **Chained NPD is also well under the memoryless SC and the prior NCG-window results** across all N.
- **At N=128, the gap to trellis SC grows to 4.9x** (best d=64 model with continuation). The key finding is that embedding dimension d is critical at larger N: d=16 gives 12.4x gap, d=32 gives 8.7x, d=64 gives 4.9x. The d=64 BiGRU with 228K parameters and 200K total training iterations is the best N=128 result. Stage 2 remains trivially easy (BLER=0 given true U). The entire bottleneck is Stage 1 U decoding. See `npd_memory_mac_N128.md` for full N=128 details.

## Training curves (loss / eval BLER)

### N = 16 (window W=2, 40K iters per stage, batch 16, lr 1e-3 cosine)

Stage 1 (40K iters, 4.1 min):
```
iter  4000  loss=0.125  BLER=0.120  (best)
iter  12000 loss=0.114  BLER=0.123
iter  20000 loss=0.105  BLER=0.120
iter  28000 loss=0.103  BLER=0.113  (best)
iter  40000 loss=0.104  BLER=0.120
```
Stage 2 (teacher-forced true X; 40K iters, 4.5 min): loss 0.04 → 0.014; BLER(V|true X) reaches 0.0 at iter 16K and stays there.

Chained final (2000 codewords): BLER = 0.1325 (CI via Wilson ≈ ±0.015). Trellis SC reference on same channel (500 codewords): 0.1360.

### N = 32 (window W=2, 40K iters per stage, batch 16, lr 1e-3 cosine)

Stage 1 (8.2 min):
```
iter  4000  loss=0.193  BLER=0.193
iter  12000 loss=0.149  BLER=0.153
iter  20000 loss=0.144  BLER=0.120
iter  28000 loss=0.141  BLER=0.107  (best)
iter  40000 loss=0.139  BLER=0.107
```
Stage 2 (7.7 min): loss 0.037 → 0.029; BLER(V|true X) = 0.0.

Chained final (2000 codewords): BLER = 0.0780. Trellis SC reference: 0.0720 (500 codewords).

### N = 64 (window W=2, 40K iters, batch 16, 9.7 min)

Stage 1:
```
iter  4000  loss=0.204  BLER=0.230
iter  12000 loss=0.201  BLER=0.203
iter  20000 loss=0.183  BLER=0.150
iter  28000 loss=0.175  BLER=0.097  (best)
iter  36000 loss=0.174  BLER=0.080  (best)
iter  40000 loss=0.174  BLER=0.083
```
Stage 2 (9.7 min): loss 0.026 → 0.019; BLER(V|true X) = 0.0 throughout.

Chained: BLER = 0.0695. Trellis SC: 0.028.

### N = 64 (**BiGRU**, 1 layer, d=16, 40K iters, 17.4 min)

This is the most interesting run — BiGRU outperforms window by 39%:
```
iter  4000  loss=0.238  BLER=0.327
iter  12000 loss=0.165  BLER=0.130
iter  20000 loss=0.147  BLER=0.077
iter  24000 loss=0.140  BLER=0.040  (best)
iter  36000 loss=0.122  BLER=0.030  (best)
iter  40000 loss=0.121  BLER=0.043
```
Stage 2 (17.5 min): loss 0.024 → 0.019; BLER(V|true X) = 0.0.

Chained: BLER = 0.0425 — within 1.52× of trellis SC at 0.028.

### N = 32 (BiGRU 1 layer, 40K iters, 11.3 min)

Stage 1 best: 0.1100 at iter 40K. Chained: 0.1115. Trellis: 0.072. BiGRU slightly worse than window at this size, likely because of higher optimisation variance at smaller N where the sequence length is short enough that the window MLP already sees the full relevant context.

### N = 16 (BiGRU, 40K iters, 6.8 min)

Stage 1 best: 0.1133. Chained: 0.1465. Slightly worse than window at the smallest size.

## Checkpoint files

All in `class_c_npd/results/npd_memory_mac/`.

Window (W=2) encoder:
- `isi_mac_window_w2_s1_N{16,32,64}_best.pt`
- `isi_mac_window_w2_s2_N{16,32,64}_best.pt`
- `isi_mac_window_w2_s{1,2}_N{16,32,64}_iter{20000,40000}.pt` (periodic snapshots)

BiGRU (1 layer) encoder:
- `isi_mac_bigru_L1_s1_N{16,32,64}_best.pt`
- `isi_mac_bigru_L1_s2_N{16,32,64}_best.pt`
- matching periodic snapshots

Also JSON result files:
- `isi_mac_window_w2_results.json` — full window sweep
- `isi_mac_bigru_results.json` — BiGRU at N=32, 64
- `isi_mac_bigru_N16_results.json` — BiGRU at N=16

Log files: `*.log` alongside each result JSON.

## What worked, what didn't

**Worked** (and the key diagnostic of the old baseline):

1. **Full-neural tree** (`use_analytical_training=False`). The broken NPD used the analytical-training path from `NPDSingleUser`, which squashes the d-dim embedding into a scalar LLR at depth 0 and then runs analytical `f/g` on scalars. That is the correct algorithm when the effective channel at each position is an i.i.d. LLR, but for ISI-MAC the per-position marginal has state information that scalar LLR throws away. Our new `NPDTree.fast_ce` keeps the full d-dim embedding at every depth.

2. **Sequence-level z encoder.** Both the sliding-window MLP and the BiGRU consume the *whole* z-sequence before any bit-reversal or tree traversal. Bit-reversal is applied AFTER encoding (in natural position order) so the encoder sees true neighbours even though the tree traversal is interleaved. The old ISI-MAC config built a (B,N,3) window but passed it to an essentially per-position z_encoder — that is formally equivalent to what we do here for `window_size=1`, but combined with the scalar-LLR squash it lost.

3. **Class-C teacher forcing with true U codeword.** Stage 2 is trained with the TRUE x-codeword (what the channel actually saw) as the side signal. This removes noise in the Stage 2 target and the `BLER(V|true X)` metric goes to 0 within 20K iters at every N.

4. **BiGRU beats sliding window at N ≥ 64**, where the h=0.3 ISI tap can propagate useful information over longer distances than a fixed W=2 window (5-symbol receptive field). At N ≤ 32 the window MLP is either equal or slightly better; BiGRU seems harder to optimise when the sequence is short.

**Didn't work**:

1. **Trapdoor MAC bonus.** Ran for 30K iters at N=32 using BEMAC_C as the design proxy; loss plateaued at 0.41 and BLER stayed at 0.97. The BEMAC_C design is a very poor fit for Trapdoor (different output alphabet, different memory structure), so the frozen set isn't reliable for the actual channel. This is a design-mismatch issue, not a decoder issue — need to generate a proper Trapdoor MC design first. Aborted to free compute.

2. **Tiny designs file mismatch.** The existing `designs/isi_mac_C_n{4,5}.npz` files are all-zero placeholders from a broken generator. We fell back to the `gmac_C_n{n}_snr6dB.npz` design (same proxy the broken baseline uses), so the reported chained BLER is directly comparable to the broken NPD baseline.

## Reproduction

To rerun the window experiment at all N:
```
python scripts/train_npd_memory_mac.py --channel isi_mac \
  --N 16 32 64 --iters 40000 \
  --encoder_type window --window_size 2 \
  --batch 16 --lr 1e-3 --d 16 --hidden 64 --n_layers 2 \
  --eval_cw 300 --final_cw 2000 --trellis_cw 500
```

BiGRU sweep:
```
python scripts/train_npd_memory_mac.py --channel isi_mac \
  --N 32 64 --iters 40000 \
  --encoder_type bigru --gru_layers 1 \
  --batch 16 --lr 1e-3 --d 16 --hidden 64 --n_layers 2
```

Each run prints incremental training curves, per-stage BLER, chained BLER, and a trellis SC reference computed on the same channel realisations.

## Open issues / next steps

- **Close the N=64 gap to trellis SC** (0.043 vs 0.028, 1.5×). Candidates:
  - Longer training (100K iters; the loss was still decreasing)
  - Larger d (32) or deeper BiGRU (2 layers)
  - Stage 2 error-injection during training (occasionally use decoded `û` instead of true `u` to bridge the teacher-forcing gap)
  - Learn a proper ISI-MAC design via MC density evolution, rather than GMAC_C proxy
- **Trapdoor MAC** needs a proper Class C MC design before the decoder can be evaluated meaningfully. The decoder module supports arbitrary `MemoryStageNPD` instances.
- **Gilbert-Elliott MAC** bonus not attempted due to compute budget. Trapdoor attempt used 6 min before abort.
- **Scale to N=128, 256.** Should be straightforward — batch 8, 40K iters per stage should fit in under an hour per N. Trellis SC reference is also O(N · S^2) so it scales fine.
- **Iterative Stage 1 refinement** (Algorithm 3 of NPD paper uses several passes). Not implemented — single-pass teacher-forced training.
