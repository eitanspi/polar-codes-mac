# Getting Started — Polar Codes for Two-User MAC with Memory

A reproducibility and onboarding guide. Written for someone (or an agent)
who is **picking up the repository from scratch** with no prior context.

If you only have 5 minutes, read §0 ("What is this project?"), do §3
("Setup & smoke test"), and open `results_pdfs/01_isi_mac_bler_vs_n.pdf`.

---

## 0. What is this project?

### The problem
Two users transmit independent **polar codewords** over a shared
**channel with memory** (intersymbol interference, or autoregressive
noise). The receiver sees only their summed, noisy output `z` and must
recover both messages.

```
        Polar enc.            shared channel             receiver
  u  ──> [G_N] ──> x ─┐                          ┌──> û (decoder)
                      ├──> z = f(x, y) + noise ──┤
  v  ──> [G_N] ──> y ─┘                          └──> v̂
```

For memoryless channels this is the standard MAC polar coding problem
(Şaşoğlu, Önay, Ren). For memory channels (ISI, AR(1) noise) it is open;
no closed-form successive-cancellation decoder existed for the general
two-user case until this project, which:

1. **Derives a working analytical decoder (SCT)** by composing
   Wang/Honda/Yamamoto/Liu/Hou's single-user trellis SC (ITW 2015) with
   Önay's two-user monotone-chain MAC SC (ISIT 2013). Derivation in
   [`docs/sct_su_to_mac.pdf`](docs/sct_su_to_mac.pdf).
2. **Implements two neural decoders (NPD, NCG)** that learn the
   per-position LLRs from data.
3. **Compares all three** at every block length N ∈ {16, …, 1024} and
   shows they match within MC noise.

### Headline result
At ISI-MAC h=0.3, SNR=6 dB, corner-rate (`Class C`):

| N | SCT (analytical) | NPD (neural) | NCG (neural) |
|---|---|---|---|
| 16   | 0.150  | 0.165  | 0.168 |
| 64   | 0.029  | 0.033  | 0.028 |
| 128  | 0.0059 | 0.013  | 0.0075 |
| 256  | 0.0013 | 0.0014 | 0.0022 |
| 1024 | 4 ×10⁻⁵ | 3 ×10⁻⁵ | ~1 ×10⁻³ (training in progress) |

Full table & plot: `results_pdfs/01_isi_mac_bler_vs_n.pdf`.

### Two channels we care about

| | ISI-MAC | MA-AGN-MAC |
|---|---|---|
| Equation | `z_i = (1−2x_i)+(1−2y_i) + h·[(1−2x_{i−1})+(1−2y_{i−1})] + w_i` | `z_i = (1−2x_i)+(1−2y_i) + n_i,  n_i = α·n_{i−1}+w_i` |
| Memory in | Signal (ISI tap `h`) | Noise (AR(1) coefficient `α`) |
| State space | Discrete, `\|S\|=4` for `(X_{i−1}, Y_{i−1})` | Continuous (real) |
| Has analytical SCT? | **Yes** (direct) | **Yes** via whitening: `z'_i = z_i − α z_{i−1}` reduces it to ISI-MAC with `h = −α` |

---

## 1. Glossary

| Term | Meaning |
|---|---|
| **MAC** | Multiple-access channel — two transmitters share one channel |
| **N** | Block length (power of 2: 16, 32, …, 1024) |
| **U, V** | Pre-codewords for user 1 and user 2; encoded by polar transform `G_N` into transmitted X, Y |
| **Corner-rate / Class C** | Decode all of U first (treating V as noise), then all of V given the decoded X̂. Reaches one corner of the MAC rate region |
| **Equal-rate** | Decode in interleaved order so both users get equal rate (the dominant face of the capacity region) |
| **BLER** | Block-error rate (per codeword) |
| **SCT** | Successive Cancellation **T**rellis decoder. Forward-backward on the channel state lattice + Arikan SC. Our **analytical baseline**. Derivation in `docs/sct_su_to_mac.pdf` |
| **NPD** | **Neural Polar Decoder** — `neural/npd_memory_mac.py`. BiGRU z-encoder + neural tree |
| **NCG** | **Neural Computational Graph** decoder — `neural/ncg_isi_mac.py`. Ren-style comp-graph SC with neural CalcLeft/CalcRight + soft-bit bridge |
| **Joint MAC SCT** | The SCT version that uses the full 4-state (X_{t−1}, Y_{t−1}) lattice and a Ren-style MAC SC tree on top |
| **4-state chained SCT** | A specialisation for corner-rate: 4-state FB → marginalise Y per position → scalar U-LLR → Arikan SC for U; then 2-state FB for V given X̂. This is the main analytical baseline in our plots |
| **MC design** | Monte-Carlo genie-aided design of the polar frozen set |

---

## 2. Repository layout

```
to_git_v2/
├── README.md, RESULTS.md                ← project overview & headline numbers
├── GETTING_STARTED.md                   ← you are here
├── requirements.txt                     ← Python deps
├── docs/
│   ├── sct_su_to_mac.pdf                ← derivation: SU SCT → MAC SCT
│   └── sct_su_to_mac.tex
├── results_pdfs/                        ← 6 standalone result PDFs (plot+table each)
├── results_local/                       ← raw plots, summary PDFs, log files
├── polar/                               ← analytical decoders + channel models
│   ├── channels.py, channels_memory.py, channels_memory_new.py
│   ├── encoder.py
│   ├── decoder.py                       ← memoryless SC + SCL primitives
│   ├── decoder_interleaved.py           ← Ren-style MAC SC tree
│   ├── decoder_trellis.py               ← JOINT MAC SCT (decode_single)
│   ├── decoder_trellis_mac_chained.py   ← 2-state chained SCT (cruder)
│   ├── design.py, design_mc.py          ← MC genie-aided design
│   └── _decoder_numba.py                ← Numba-JIT inner loops
├── neural/
│   ├── npd_memory_mac.py                ← ChainedNPD_MAC
│   └── ncg_isi_mac.py                   ← ISIMACNeuralDecoder
├── designs/                             ← pre-computed frozen sets (gmac_C_n*.npz)
├── class_c_npd/results/                 ← canonical JSON results from cluster
├── scripts/
│   ├── smoke_test.py                    ← 30-second sanity check
│   ├── local_analysis/                  ← analytical SCT scripts + outputs
│   │   ├── chained_sct_4state.py        ← the 4-state chained SCT (key file)
│   │   ├── ncg_isi_chain_local.py       ← original CPU NCG trainer (known-good)
│   │   └── ncg_models/                  ← trained NCG ckpts (.pt files via cluster sync)
│   └── training/                        ← GPU trainers
│       ├── ncg_isi_old_recipe_gpu.py    ← OLD recipe ported to GPU (preferred)
│       └── ncg_isi_gpu_curriculum.py    ← Faster trainer; bf16 issue, see Pitfalls
└── archive/                             ← pre-pivot work (memoryless GMAC era)
```

---

## 3. Setup & smoke test

```bash
git clone https://github.com/eitanspi/polar-codes-mac.git
cd polar-codes-mac/to_git_v2
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
```

Expected output (≤ 20 seconds):
```
Encoder roundtrip    : OK  (N=16, x[:5] = [1, 1, 1, 1, 0])
Channel sample       : OK  (ISI-MAC h=0.3, |z|_max = 2.693)
Joint MAC SCT decode : OK  (u_hat[:5] = [0, 0, 0, 0, 0])
Chained SCT N=16     : BLER 0.118 (59/500) in 18.8s
All modules working.  See GETTING_STARTED.md for next steps.
```

If the smoke test passes, you have a working install. If it errors, see
§9 (Pitfalls) — most likely a missing dependency or a multiprocessing
spawn issue on certain platforms.

### Open the headline plot
```bash
python results_pdfs/build_all.py        # regenerates all 6 plots
open results_pdfs/01_isi_mac_bler_vs_n.pdf
```

---

## 4. Reading the code (10-minute tour)

If you want to understand the decoders, read these files in order:

1. **`polar/channels_memory.py`** — `ISIMAC`. Inputs `(X, Y)`, output `z`,
   state `s = (X_{i−1}, Y_{i−1})`. See `build_leaf_tensors(z)` returning
   `(N, 2, 2, S, S)` log-probabilities. ~100 lines.
2. **`polar/decoder_trellis.py`** — `decode_single(N, z, b, fu, fv, channel)`.
   The full joint MAC SCT: `_forward_backward_joint` produces `(N, 2, 2)`
   marginals; `_decode_with_marginals` runs the Ren-style MAC SC tree on
   them. ~300 lines, well commented.
3. **`scripts/local_analysis/chained_sct_4state.py`** — the same FB but
   marginalised down to scalar LLRs, then Arikan SC. This is what gets
   plotted as "SCT". `mc_design`, `eval_at`, `pick`, `decode_4state_chained`
   are the public functions.
4. **`docs/sct_su_to_mac.pdf`** — derivation: where the recursions come
   from and why they are correct (Wang/Honda 2015 + Önay 2013).
5. **`neural/npd_memory_mac.py`** — NPD architecture (BiGRU + neural tree).
   Note: training pipeline is on the cluster, not in this repo.
6. **`neural/ncg_isi_mac.py`** — NCG architecture (Ren comp-graph + neural
   ops + soft-bit bridge).

---

## 5. Reproducing the headline numbers

### Analytical SCT (any N, ~1 minute at N=128, hours at N=1024)
```python
import sys
sys.path.insert(0, "scripts/local_analysis")
from chained_sct_4state import mc_design, eval_at, pick, RATES

N = 128
ku, kv = RATES[N]                     # info-bit counts for "code rate ~0.34"
Pe_u, Pe_v, _ = mc_design(N, 50000, n_workers=7, base_seed=1)   # design
Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)                         # info sets
r = eval_at(N, 10000, Au, Av, n_workers=7, base_seed=2)         # eval BLER
print(f"BLER = {r['bler']:.5g} ({r['errs']}/{r['n_cw']})")
# Expected: ≈ 0.006 at N=128, h=0.3, SNR=6 dB
```

Use `n_workers=` number of CPU cores you have. For N=1024 use
`mc_design(... mc_trials=200_000)` and `eval_at(... n_cw=50000)` (4-6 hours
on 8 cores).

### NPD and NCG numbers
NPD numbers are already in `class_c_npd/results/isi_campaign/results.json`
(cluster campaign output). NCG numbers from `scripts/local_analysis/ncg_models/ncg_chain_results.json` plus the cluster-trained large-N checkpoints.

To **retrain** these from scratch, see §6 (NCG, can do partial locally)
and §7 (NPD, cluster only).

### Regenerating the plot
```bash
python results_pdfs/build_all.py
```
This reads the JSON outputs and rebuilds all 6 PDFs in `results_pdfs/`.

---

## 6. Training NCG

NCG corner-rate weights are universal across N: a model trained at N=128
is shape-compatible with N=16…1024. So most users will:

1. Use the pre-trained `ncg_isi_N128.pt` (~170 KB) as warm-start.
2. Fine-tune at larger N if needed.

### Local CPU training (slow; will produce N=16..128 ckpts)
```bash
python scripts/local_analysis/ncg_isi_chain_local.py
```
Runs the full N=16 → 32 → 64 → 128 curriculum. Takes ~12 hours total on
modern CPU. Saves checkpoints to `scripts/local_analysis/ncg_models/`.

### GPU training (cluster; for N ≥ 256)
Two trainers exist:
- **`scripts/training/ncg_isi_old_recipe_gpu.py`** — recommended. Batch 32,
  fp32, lr 1e-3, grad clip 1.0, Adam. Verified to preserve warm-started
  weights.
- `scripts/training/ncg_isi_gpu_curriculum.py` — faster (bf16 + batch 512)
  but **destroys warm-started weights in practice** (see §9 Pitfalls).
  Use only if you understand the failure mode.

Example (cluster):
```bash
python scripts/training/ncg_isi_old_recipe_gpu.py \
  --N 256 --iters 25000 \
  --warm-start-ckpt scripts/local_analysis/ncg_models/ncg_isi_N128.pt \
  --save-ckpt scripts/training/ncg_models_gpu/ncg_isi_N256.pt \
  --save-every 2000 --eval-cw 5000
```

---

## 7. Training NPD (cluster only)

The canonical NPD pipeline lives on the BGU cluster at
`/gpfs0/bgu-haimp/users/eitansp/polar_project/`. Key files (not in this repo):

- `isi_campaign.py` — train + design + eval, one N at a time, warm-started
- `maagn_campaign.py` — same for MA-AGN
- `npd_batched_reeval.py` — batched GPU eval of trained NPD
- `topup_final.py` — top up codewords to ≥30 errors per point

The closest local template is `scripts/isi_r2_campaign.py` (the r=2
follow-up). For full NPD reproduction you need cluster GPU access (§8).

---

## 8. Cluster access (BGU)

```bash
ssh -i ~/.ssh/id_ed25519 eitansp@bhn20.bgu.ac.il
# On the head node:
runai workload list                                  # list workloads
runai workspace exec my-workspace -- bash            # interactive shell
runai workspace exec my-workspace -- python <script>
```

Workspace `my-workspace`: NVIDIA RTX 6000 Ada, 51 GB VRAM, 2 CPU cores
(data-gen bottleneck!), 16 GB RAM. Python env at
`/gpfs0/bgu-haimp/users/eitansp/env/myenv/bin/python`. Repo at
`/gpfs0/bgu-haimp/users/eitansp/polar_project/`.

---

## 9. Pitfalls (read before debugging)

1. **MC design tiebreak.** When many channels have observed Pe=0 (common at
   N≥512 with finite trials), `np.argsort` orders them arbitrarily and may
   exclude the most-reliable channels. Always use
   `polar.design_mc._argsort_with_polar_tiebreak`, which resolves ties by
   the Arikan bit-reversal weight.

2. **NCG GPU training with bf16 destroys warm-started weights.** The
   `ncg_isi_gpu_curriculum.py` trainer uses bf16 + batch 512; this works
   for training from-scratch at N=128 but pushes the model into a bad
   local minimum when fine-tuning from a smaller-N checkpoint. Use
   `ncg_isi_old_recipe_gpu.py` (batch 32, fp32) instead.

3. **Multiprocessing on macOS / Python 3.8+.** Any script using
   `multiprocessing.Pool` must guard its main code with
   `if __name__ == "__main__":`. Without it, Pool's spawn re-imports the
   module and you get infinite recursion (we hit this twice).

4. **N=1024 SCT under-samples the reliability tail.** 50K MC design trials
   leave the synthesised-channel ranking at the long tail dominated by
   tiebreak, giving BLER ~5× pessimistic. Use ≥200K trials at N=1024.

5. **Cluster NCG warm-start path.** The trainer's default
   `--warm-start-ckpt` is `scripts/local_analysis/ncg_models/ncg_isi_N128.pt`.
   If that exact path doesn't exist on the cluster, the trainer silently
   "loads 0 tensors" and trains from random — looks fine in the log but
   produces a junk model. Always check the "warm-started X tensors"
   message.

6. **`mc_design` indexing.** Returns 0-indexed `sorted_u`, `sorted_v`
   (channels in reliability order). When converting to a frozen-bit
   dictionary, remember to add 1 (the dict uses 1-indexed positions).

---

## 10. Where to look first if something breaks

| Symptom | Most likely cause |
|---|---|
| `ImportError: cannot import name 'X' from 'polar.Y'` | Old archive shadowing — check `archive/` isn't on the path |
| `decode_single` errors at `channel.num_states` | Channel object doesn't have `.num_states` (check it's `ISIMAC`, not the memoryless variant) |
| NCG ckpt load gives all-zero BLER then random | Shape mismatch silently dropped tensors; check the "loaded N/42 tensors" line |
| Cluster job hangs at startup | Missing `if __name__ == "__main__":` guard (see Pitfall 3) |
| `np.argsort` gives unexpected info-set | Pitfall 1 — use `_argsort_with_polar_tiebreak` |
| BLER at large N is ~5× worse than expected | Pitfall 4 — design under-sampled, bump `mc_trials` |

---

## 11. Citations

The theory comes from three papers (all in `/papers/`):

- **Wang, Honda, Yamamoto, Liu, Hou** (ITW 2015) — "Construction of Polar
  Codes for Channels with Memory" — single-user SCT recursion.
- **Önay** (ISIT 2013) — "Successive Cancellation Decoding of Polar Codes
  for the Two-User Binary-Input MAC" — monotone-chain MAC SC decomposition.
- **Şaşoğlu** (ISIT 2011) — "Polarization in the Presence of Memory" —
  proves the transform polarises finite-state channels.

The Aharoni et al. NPD paper (2024) is what we **compare against**; SCT is
their analytical baseline (in their setting, single-user). Our derivation
in `docs/sct_su_to_mac.pdf` is the two-user MAC extension.

---

## 12. Getting help

Open an issue on GitHub, or contact the project owner. When reporting a
problem, please include:

- The full error message
- Output of `python scripts/smoke_test.py`
- The git commit hash (`git rev-parse HEAD`)

---

*Last reviewed: see commit `git log -1 --format=%cd GETTING_STARTED.md`.*
