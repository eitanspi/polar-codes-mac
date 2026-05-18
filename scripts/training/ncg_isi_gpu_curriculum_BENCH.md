# NCG ISI-MAC GPU-optimized training: benchmark report

Script: `scripts/training/ncg_isi_gpu_curriculum.py`

## What changed vs `scripts/local_analysis/ncg_isi_chain_local.py`

1. **Data-gen moved to device (GPU)** — `GPUDataGen` samples bits, polar-encodes
   (torch XOR butterfly + bit-reversal), and runs the ISI-MAC channel
   (BPSK + length-1 ISI + AWGN) entirely on the model's device. No more
   numpy→cuda host transfers per iter. Encoder verified bit-exact against
   `polar.encoder.polar_encode_batch` via `--self-test`.
2. **Mixed precision (bf16 autocast)** on CUDA only — bf16 keeps fp32 exponent
   range so no GradScaler needed. Auto-disabled on CPU (autocast(bf16) on
   CPU hurts perf because most kernels fall back to fp32+cast).
3. **Larger default batch on GPU** (512 vs old 32). The model's tree fwd/bwd
   is the dominant cost; per-iter wallclock grows sublinearly with batch, so
   samples/sec rises sharply (CPU at N=64: 230 samples/s @ B=32 → 620 @ B=256).
4. **Inference-mode batched BLER eval** — replaces the old Python `for cw in range(n_cw)` loop
   with batched encode+channel+decode (the SC tree decoder is already batched in
   `PureNeuralCompGraphDecoder.forward`).
5. CLI modes: `--bench`, `--verify`, `--curriculum`, `--self-test`.

## Local CPU benchmark (Mac, single-core scalar)

Setup: torch 2.11, CPU only. Batch=32 unless noted.

| Mode | N | batch | it/s | data ms | fwd ms | bwd ms |
|------|---|-------|------|---------|--------|--------|
| baseline (numpy) | 64 | 32 | **7.33** | 0.3 | 65.1 | 72.7 |
| optimized (torch) | 64 | 32 | **7.16** | 0.3 | 66.1 | 74.9 |
| baseline | 128 | 32 | **3.45** | 0.4 | 141.0 | 153.5 |
| optimized | 128 | 32 | **3.51** | 0.3 | 139.0 | 150.6 |
| optimized | 64 | 16 | 8.83 | — | — | — |
| optimized | 64 | 256 | 2.43 | 0.9 | 233 | 193 |

**Key finding (CPU):** data-gen is <1% of per-iter time → on CPU the model itself is the bottleneck, so the optimization shows no speedup locally. **This is expected**: the cluster's 2-CPU-core data-gen on numpy is what makes data-gen the bottleneck there, not algorithmic complexity. Per-iter wallclock scales sublinearly with batch (B=256 ~3× longer per-iter than B=32 but 8× more samples) — so larger batch on GPU is a free throughput win up to VRAM limit.

## Cluster GPU projection

From `project_session11_equal_rate.md`: N=128 → ~100 it/min (1.67 it/s) at 30% GPU util on RTX 6000 Ada, batch=32. Bottleneck: data-gen on 2 CPU cores feeding GPU.

If we move data-gen on-device (this script), at minimum the GPU stops idling. At 30% prior util, a fully-fed GPU yields ~3.3× iter/s on the same batch. Adding bf16 autocast (~1.3-1.6× on Ada-class GPUs for matmul-heavy nets) and batch=512 (≥2× samples/sec at same iter rate) compounds to a **~5-10× wall-time improvement**.

Conservative (3.3×, batch=32) and aggressive (10×, batch=512 + bf16) projections:

| N | Current it/min | Curriculum iters | Current wall | Conservative (3.3×) | Aggressive (10×) |
|----|---|---|---|---|---|
| 128  | 100 | 120k | 20 h  | 6.1 h | 2.0 h |
| 256  | 50  | 150k | 50 h  | 15.2 h| 5.0 h |
| 512  | 25  | 200k | 133 h | 40 h  | 13 h |
| 1024 | 12  | 250k | 347 h | 105 h | 35 h |
| **Total N=16→1024** | — | ~950k | **~22 days** | **~7 days** | **~2.3 days** |

Note: only N=128→1024 is roughly tracked above (small N is essentially free). The aggressive number aligns with the task's "~3 days" target.

The actual cluster number must be measured (run `python ncg_isi_gpu_curriculum.py --bench --N 128 --iters 1000 --batch 512` on cluster); the local CPU benchmark cannot distinguish the projection multipliers, only validate that the optimization is correct.

## Correctness verification

```
python scripts/training/ncg_isi_gpu_curriculum.py --verify --N 64 --iters 500 --eval-cw 1000
```

Result: **BLER 0.0170 (17/1000)** after 500 extra iters from existing N=64 checkpoint.

Reference (existing `ncg_chain_results.json` N=64): 0.0282 (141/5000) — wider eval, looser CI.

The new pipeline preserves and slightly improves the trained model. 95% CI overlap with reference: [0.010, 0.027] vs [0.024, 0.033] — within statistical noise of identical performance.

## Caveats

* `torch.compile` not enabled by default — the tree decoder uses Python control
  flow over `b`, `frozen_u`, `frozen_v` and dict-typed intermediate buffers,
  which `torch.compile` cannot trace cleanly. Could be revisited by lifting
  these to fixed-shape tensors and compiling sub-modules separately.
* AMP win on CPU is negative; the script auto-disables it. AMP must be tested
  empirically on cluster — bf16 can lose accuracy on the GRU encoder if used.
* The per-N curriculum iter counts are inherited from `ncg_isi_chain_local.py`
  and extrapolated for N ≥ 256. Cluster runs may want to tune these (e.g.
  fewer iters for larger N if loss plateaus earlier).
* Bench results above are wall-clock from a single short run; CI may be ±10%.
