# Validation Report -- polar_codes_MAC/to_git_v2

**Date**: 2026-03-23
**Validator**: Automated code review via Claude

---

## 1. Imports & Dependencies

### All module imports: PASS
- [x] `polar.encoder` -- OK
- [x] `polar.channels` -- OK
- [x] `polar.design` -- OK
- [x] `polar.design_mc` -- OK
- [x] `polar.decoder` -- OK
- [x] `polar.efficient_decoder` -- OK
- [x] `polar.decoder_interleaved` -- OK
- [x] `polar.decoder_parallel` -- OK (requires numba)
- [x] `polar.decoder_scl` -- OK
- [x] `polar.eval` -- OK
- [x] `polar._decoder_numba` -- OK
- [x] `neural.ncg_memory` -- OK
- [x] `neural.neural_comp_graph` -- OK
- [x] `neural.channels_memory` -- OK
- [x] `neural.train_pure_neural` -- OK
- [x] `design` (root-level) -- OK
- [x] `polar/__init__.py` re-exports -- all resolve correctly
- [x] `neural/__init__.py` -- OK (docs only, no re-exports)
- [x] No circular imports detected

### External packages required
| Package | Required by | Optional? |
|---------|------------|-----------|
| numpy | All modules | No |
| matplotlib | Plotting scripts | Yes (scripts only) |
| torch | neural/ modules | Yes (neural decoder only) |
| numba | decoder_parallel, _decoder_numba | Yes (parallel speedup only) |
| tensorflow | encoder.py (`polar_encode_batch_tf`) | Yes (guarded by try/except) |

### Cross-module import notes
- **WARNING**: Three scripts (`simulate_gmac.py`, `campaign_28h.py`, `campaign_10h.py`) add `../../` to `sys.path` to import root-level `design.py`. The first `sys.path` entry (`../`) already covers this. The `../../` entry is unnecessary and reaches outside the repo root, but is harmless since the import resolves from `../` first.

---

## 2. Core Module Correctness

### Encoder: PASS
- [x] `polar_encode()` -- single vector encoding works (returns list of int)
- [x] `polar_encode_batch()` -- batch encoding works (returns ndarray)
- [x] `build_message()` -- correctly places info bits at specified positions
- [x] Encode is involutory: `encode(encode(u)) == u` (verified in demo script)

### Channels: PASS
- [x] `BEMAC` -- Z = X + Y, deterministic, produces values in {0,1,2}
- [x] `ABNMAC` -- Z = (X xor Ex, Y xor Ey), stochastic noise sampling works
- [x] `GaussianMAC` -- Z = (1-2X) + (1-2Y) + W, constructor takes `sigma2` (not `snr_db`)
- [x] All three channels have `sample()`, `sample_batch()`, `transition_prob()`, `capacity()` methods

### Design modules: PASS
- [x] `design_bemac(n, ku, kv)` -- returns 6 values: (Au, Av, frozen_u, frozen_v, pe_u, pe_v)
- [x] `design_abnmac(n, ku, kv)` -- same interface
- [x] `make_path(N, path_i)` -- generates monotone chain path vectors
- [x] `design_mc.design_from_file()` -- loads .npz designs, selects info/frozen sets correctly
- [x] Root `design.py` -- provides `ga_gmac`, `design_gmac`, `bhattacharyya_*` functions

### Decoders

#### SC decoder (decoder.py): PASS
- [x] Auto-dispatches between LLR (extreme paths) and tensor (interleaved paths)
- [x] BEMAC N=8, path_i=3 (intermediate): 0/100 block errors on noiseless channel
- [x] GaussianMAC N=16, SNR=6dB: 1/50 block errors (expected with noise)
- [x] ABNMAC N=8: 3/50 block errors (expected with noise)
- [x] `decode_single()` and `decode_batch()` both exported

#### SCL decoder (decoder_scl.py): PASS
- [x] BEMAC N=8, L=4: 0/100 block errors on noiseless channel
- [x] `decode_single_list()` and `decode_batch_list()` both exported

#### Parallel decoder (decoder_parallel.py): PASS
- [x] `decode_parallel_single(N, log_W, b, frozen_u, frozen_v)` -- works correctly
- [x] Requires pre-computed `log_W` (from `build_log_W_leaf()`)
- [x] Numba JIT compilation successful

#### Interleaved decoder (decoder_interleaved.py): PASS
- [x] Imported and used by unified decoder.py for intermediate paths
- [x] 0/100 errors on BEMAC noiseless channel

#### Efficient decoder (efficient_decoder.py): PASS (with note)
- [x] Imports and runs without crash
- **Note**: RuntimeWarnings (`invalid value encountered in subtract`) occur on BEMAC due to log(0)=-inf arithmetic. These are handled by NaN replacement logic. Decoding correctness depends on using frozen sets designed for the matching path_i.

### Eval module: PASS
- [x] `MACEval(channel, decoder_type='sc')` -- instantiates correctly
- [x] `MACEval.run(N, b, Au, Av, fu, fv, n_codewords=50)` -- returns (bler_u, bler_v, bler_joint)
- [x] BEMAC noiseless: (0.0, 0.0, 0.0) as expected

---

## 3. Neural Module: PASS

- [x] `NeuralCompGraphDecoder(d=16, vocab_size=3)` -- instantiates
- [x] `NCGMemoryDecoder(d=16, hidden=64, ...)` -- instantiates
- [x] `channels_memory.py` -- ISI_MAC and GilbertElliott_MAC classes present
- [x] `train_pure_neural.py` -- distillation training for Pure Neural CalcParent
- [x] All 8 saved models (.pt) load successfully into NeuralCompGraphDecoder
- [x] Model files: ncg_N8_d16.pt through ncg_N1024_bler_sweep.pt

---

## 4. Scripts: PASS (with warnings)

### Syntax check: all 19 .py scripts parse correctly
### argparse scripts respond to --help: PASS (tested simulate.py, run_design.py, simulate_gmac.py)

### Warnings
- **[FIX REQUIRED]** `scripts/run_all_gmac_designs.sh` line 4: hardcoded absolute path
  `cd /Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git` -- should be relative (e.g., `cd "$(dirname "$0")/.."`)
- **[MINOR]** Three scripts add `../../` to sys.path (unnecessary but harmless)

---

## 5. Data Files: PASS

### Design files (.npz): 271 total
- BEMAC: 25 files (classes A/B/C, n=3..10)
- ABNMAC: 24 files (classes A/B/C, n=3..10)
- GMAC: 222 files (classes A/B/C, n=3..11, SNR 0..10 dB)
- All spot-checked files load correctly with expected keys: `u_error_rates`, `v_error_rates`, `path_i`, `n_trials`

### Result files: 132 JSON + 46 other (plots, logs)
- JSON files load and parse correctly
- Plot files (.png, .pdf) and log files present

### Saved models: 8 .pt files
- All load correctly via `torch.load(path, map_location='cpu')`
- Model sizes: N=8, 16, 32, 64, 128, 256, 512, 1024

---

## 6. Code Quality: PASS (with notes)

- [x] No TODO/FIXME/HACK/XXX comments found
- [x] No hardcoded absolute paths in .py files
- [x] No secrets, credentials, API keys, or tokens found
- [x] No dead code flagged

### Files that should NOT be committed
- **[FIX REQUIRED]** `.DS_Store` (root level)
- **[FIX REQUIRED]** `__pycache__/` directories (root, polar/, neural/)
- **[FIX REQUIRED]** LaTeX build artifacts in `docs/`:
  - `project_report.aux`
  - `project_report.log`
  - `project_report.out`
  - `project_report.toc`

---

## 7. README Accuracy: PASS (with minor issues)

- [x] All files/directories listed in README exist
- [x] All existing .py files are documented in README
- [x] Quick Start commands reference real scripts
- [x] Channel descriptions accurate
- [x] Decoder descriptions accurate
- [x] Neural architecture table accurate

### Minor issues
- **[MINOR]** README lists `pip install numpy matplotlib` as required but does not mention numba as optional dependency for parallel decoder. The Quick Start section doesn't mention it.
  - Actually, line 92 says `pip install numba # optional, for parallel decoder speedup` -- this IS documented. PASS.
- **[FIX SUGGESTED]** Reference on line 136: `arXiv:2501.xxxxx` is a placeholder. Should be updated with actual arXiv ID or removed.
- **[MINOR]** README says `GaussianMAC(snr_db=...)` style interface but the actual constructor takes `sigma2`. Not explicitly stated in README but could confuse users (the `sigma2` parameter is not documented in README).

---

## 8. Missing Files

### Required files not present
- **[FIX REQUIRED]** `.gitignore` -- does not exist. Must be created to exclude:
  ```
  __pycache__/
  *.pyc
  .DS_Store
  *.aux
  *.log
  *.out
  *.toc
  ```
- **[RECOMMENDED]** `requirements.txt` -- does not exist. Should list:
  ```
  numpy
  matplotlib
  torch
  numba
  ```

### Optional missing files (low priority)
- `tests/` directory -- no formal test suite. The codebase relies on scripts and inline testing. Not blocking.
- `pytest.ini` / `setup.py` / `pyproject.toml` -- no packaging configuration. Not blocking for a research codebase.

---

## Summary

### Issues to fix before committing

| # | Severity | Item | Action |
|---|----------|------|--------|
| 1 | **HIGH** | No `.gitignore` | Create `.gitignore` with standard Python exclusions |
| 2 | **HIGH** | `.DS_Store` present | Delete and add to `.gitignore` |
| 3 | **HIGH** | `__pycache__/` directories (3) | Delete all `__pycache__/` dirs |
| 4 | **MEDIUM** | LaTeX build artifacts in `docs/` | Delete `.aux`, `.log`, `.out`, `.toc` files |
| 5 | **MEDIUM** | `scripts/run_all_gmac_designs.sh` line 4 | Replace hardcoded path with `cd "$(dirname "$0")/.."` |
| 6 | **LOW** | arXiv placeholder in README | Update `arXiv:2501.xxxxx` with real ID or remove |
| 7 | **LOW** | No `requirements.txt` | Create with: numpy, matplotlib, torch, numba |

### Recommendation: **CONDITIONAL GO**

The codebase is functionally correct. All modules import cleanly, encoders/decoders produce correct results, design files load properly, neural models load and instantiate, and scripts parse and run. The code is clean with no TODOs, no secrets, and no hardcoded paths in Python files.

**Fix items 1-5 before committing.** Items 6-7 are nice-to-have.

---

## 9. Update Log

### 2026-03-29: Neural SCL Results and Documentation Update

**New files added:**
- `scripts/eval_bemac_nn_scl_large_N.py` — NN-SCL evaluation at N=256,512,1024
- `scripts/plot_bemac_comprehensive.py` — Comprehensive 4-decoder comparison plot
- `results/bemac/bemac_classB_Ru50_Rv70_nn_scl/bemac_comprehensive_comparison.{png,pdf,xlsx}`
- `results/bemac/bemac_classB_Ru50_Rv70_nn_scl/bemac_comprehensive_data.json`

**Files modified:**
- `results/bemac/bemac_classB_Ru50_Rv70_nn_scl/bemac_nn_scl_results.json` — added N=256 result (zero errors)
- `docs/gmac_nn_problem_statement.tex` — added Section 3.2 (Neural SCL results), updated Q5 as RESOLVED, updated Summary
- `docs/gmac_nn_problem_statement.pdf` — recompiled (now 6 pages)

**Key findings:**
- NN-SCL (L=4) at N=64: BLER=6.7e-4, **8.4x better than SC**, 1.5x better than SCL-32
- NN-SCL (L=4) at N=128: BLER=6.7e-4, **3x better than SC**, matches SCL-32
- NN-SCL at N=256: zero errors in 1000 cw (L=1) and 200 cw (L=4) — BLER too low for comparison
- For N>=256 on BEMAC, all decoders achieve near-zero BLER; the interesting comparison range is N=32-128
