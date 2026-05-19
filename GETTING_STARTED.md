# Getting Started: Polar Codes for Two-User MAC

This is a reproducibility guide for the project. Targeted at a human or
agent picking up the repo from scratch.

## TL;DR

```bash
git clone https://github.com/eitanspi/polar-codes-mac.git
cd polar-codes-mac/to_git_v2
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # if missing, install torch numpy matplotlib scipy numba
python results_pdfs/build_all.py
open results_pdfs/01_isi_mac_bler_vs_n.pdf
```

`results_pdfs/01_isi_mac_bler_vs_n.pdf` is the headline result. The rest of
this document is how to verify and extend it.

## 1. Repository layout (top level)

```
to_git_v2/
├── README.md                 ← project overview + headline table
├── RESULTS.md                ← full numerical table
├── GETTING_STARTED.md        ← this file
├── docs/
│   ├── sct_su_to_mac.pdf     ← derivation: SU SCT → MAC SCT
│   └── sct_su_to_mac.tex
├── results_pdfs/             ← clean per-result PDFs (6 of them)
├── results_local/            ← raw plots, summary PDF, log files
├── polar/                    ← analytical decoders + channels
├── neural/                   ← neural decoders (NPD, NCG)
├── designs/                  ← pre-computed frozen sets (.npz)
├── class_c_npd/results/      ← canonical JSON results from cluster campaigns
├── scripts/
│   ├── local_analysis/       ← analytical SCT scripts + JSON outputs
│   └── training/             ← NCG training scripts (GPU)
└── archive/                  ← preserved pre-pivot work (memoryless GMAC era)
```

## 2. Setup

### Python environment
Tested on Python 3.10+. Required:
```
torch        # 2.1+
numpy
scipy
matplotlib
numba        # for the JIT polar decoders in polar/_decoder_numba.py
```
GPU optional; everything outside of `neural/` works on CPU.

### Designs (pre-computed)
`designs/gmac_C_n{n}_snr6dB.npz` for n ∈ {4..10} contain the GMAC Class-C
frozen-set assignments used as defaults. They are committed to the repo
(small files). No setup needed.

### Pre-trained NCG checkpoints
`scripts/local_analysis/ncg_models/ncg_isi_N{16,32,64,128}.pt` and
`ncg_isi_equal_rate_n{32,64}.pt`. These are the trained NCG models used
in plots. The `.pt` files are excluded by `.gitignore` from the repo
(too large per the project convention); pull from the cluster:
```bash
scp -r bhn20:/gpfs0/bgu-haimp/users/eitansp/polar_project/ncg_old_recipe_ckpts/ \
       scripts/local_analysis/ncg_models/
```
or train from scratch following §6 below.

## 3. Reading the code

### Channels (`polar/`)
- `channels.py` — memoryless MACs: BEMAC, ABNMAC, GaussianMAC
- `channels_memory.py` — `ISIMAC`: `Z_t = (1-2X_t) + (1-2Y_t) + h·((1-2X_{t-1})+(1-2Y_{t-1})) + W_t`
- `channels_memory_new.py` — `MAAGNMAC` (AR(1) noise), `ISIMAC2` (r=2 ISI)

### Analytical decoders (`polar/`)
- `decoder.py` — memoryless SC + SCL primitives; `_sc_decode_from_llr` is Arikan SC from a scalar-LLR vector
- `decoder_interleaved.py` — Ren-et-al-style computational-graph SC for two-user MAC (any monotone path)
- `decoder_trellis.py` — **Joint MAC SCT**: `decode_single(N, z, b, fu, fv, channel)` runs forward-backward on the full `|S|×|S|` lattice then a Ren SC tree
- `decoder_trellis_mac_chained.py` — **2-state chained SCT** (per-position Y marginalisation; cruder approximation)
- `decoder_scl.py` — SCL list decoder

### Neural decoders (`neural/`)
- `npd_memory_mac.py` — **NPD** (`ChainedNPD_MAC`): BiGRU z-encoder + neural CheckNode/BitNode tree, chained corner-rate (stage 1 marginalises V; stage 2 conditions on X̂)
- `ncg_isi_mac.py` — **NCG** (`ISIMACNeuralDecoder`): Ren computational-graph SC with neural CalcLeft/CalcRight ops + Soft-Bit Bridge for CalcParent

### Design (`polar/design_mc.py`)
- `mc_design(n, channel, mc_trials)` — genie-aided Monte-Carlo MC design
- `_argsort_with_polar_tiebreak` — important: breaks reliability ties using polar bit-reversal weight. Hand-coded; **do not replace with `np.argsort`** (breaks at finite trials with many Pe=0 channels).

## 4. The headline result

`results_pdfs/01_isi_mac_bler_vs_n.pdf` shows BLER vs N at ISI-MAC h=0.3, SNR=6 dB for:
- **SCT** — 4-state chained SCT (the proper analytical baseline; see `docs/sct_su_to_mac.pdf` for derivation). Implemented in `scripts/local_analysis/chained_sct_4state.py`.
- **NPD** — `neural/npd_memory_mac.py`. Trained on cluster GPU.
- **NCG** — `neural/ncg_isi_mac.py`. Trained mostly on cluster GPU.

The numerical data lives in:
- SCT: `scripts/local_analysis/chained_sct_4state.json`, `chained_sct_tight_large_N.json`, `chained_sct_n1024_200k.json`, plus topup runs
- NPD: `class_c_npd/results/isi_campaign/results.json` (+ `joint_trellis_*` for headline JT-SCT)
- NCG: `scripts/local_analysis/ncg_models/ncg_chain_results.json` + cluster ckpts

## 5. Reproducing the analytical SCT

```bash
# Run the 4-state chained SCT at a particular N, do MC design + eval
python -c "
import sys; sys.path.insert(0, '.')
from scripts.local_analysis.chained_sct_4state import mc_design, eval_at, pick, RATES
N = 256
ku, kv = RATES[N]
Pe_u, Pe_v, _ = mc_design(N, 50000, 7, base_seed=1)   # 50K MC trials, 7 workers
Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
r = eval_at(N, 10000, Au, Av, 7, base_seed=2)         # 10K eval CW
print(f'BLER = {r[\"bler\"]:.5g} ({r[\"errs\"]}/{r[\"n_cw\"]})')
"
```
At N=256 you should get BLER ≈ 1.3e-3 (within Monte-Carlo noise).

## 6. Training NCG from scratch (corner-rate, ISI-MAC)

Two trainers:
- `scripts/local_analysis/ncg_isi_chain_local.py` — original CPU trainer (batch 32, fp32). Slow but known-good.
- `scripts/training/ncg_isi_old_recipe_gpu.py` — same recipe ported to GPU. Use this on cluster.

```bash
# Train N=16 first (from scratch), then chain N=32 → N=64 → N=128
# This takes ~hours on CPU. CONFIGS in the script lists iters per N.
python scripts/local_analysis/ncg_isi_chain_local.py
```
Resulting ckpts saved in `scripts/local_analysis/ncg_models/`. The local
training stops at N=128. To extend to N=256+ on cluster GPU:

```bash
# On the cluster workspace (see §8 for access)
python scripts/training/ncg_isi_old_recipe_gpu.py \
  --N 256 --iters 25000 \
  --warm-start-ckpt scripts/local_analysis/ncg_models/ncg_isi_N128.pt \
  --save-ckpt scripts/training/ncg_models_gpu/ncg_isi_N256.pt \
  --save-every 1000 --eval-cw 5000
```

## 7. NPD pipeline (cluster only)

The NPD training is heavier; canonical scripts live on the cluster at
`/gpfs0/bgu-haimp/users/eitansp/polar_project/`:

- `isi_campaign.py` — train + design + eval, one N at a time, warm-started
- `maagn_campaign.py` — same for MA-AGN
- `npd_batched_reeval.py` — batched GPU eval of trained NPD
- `topup_final.py` — top up codewords to ≥30 errors per point

The local-repo template is `scripts/isi_r2_campaign.py` (ISI r=2 follow-up); it imports the cluster scripts. For full NPD reproduction you need cluster GPU access.

## 8. Cluster access (BGU)

```bash
ssh -i ~/.ssh/id_ed25519 eitansp@bhn20.bgu.ac.il
# On bhn20:
runai workload list                                  # list workloads
runai workspace exec my-workspace -- bash            # interactive shell
runai workspace exec my-workspace -- python <script> # run a command
```

Workspace `my-workspace`: NVIDIA RTX 6000 Ada, 51 GB VRAM, 2 CPU cores,
16 GB RAM limit. Python env at `/gpfs0/bgu-haimp/users/eitansp/env/myenv/bin/python`,
repo at `/gpfs0/bgu-haimp/users/eitansp/polar_project/`.

## 9. Common tasks

### Generate all result PDFs
```bash
python results_pdfs/build_all.py
```

### Run a single SCT BLER eval at high CW
See §5 above, adjust `mc_design`'s `mc_trials` (50K minimum; 200K at N=1024 to resolve the long tail) and `eval_at`'s `n_cw` (≥30K to reach ~30 errors at typical BLER).

### Verify the encoder is bit-exact
```bash
python scripts/training/ncg_isi_gpu_curriculum.py --self-test
```

### Quick smoke test of the analytical chained SCT
```bash
python scripts/local_analysis/verify_chained_sct.py
```

## 10. Pitfalls (learned the hard way)

1. **MC design tiebreak**: do not use `np.argsort` for selecting info bits when many channels have observed Pe = 0; use `_argsort_with_polar_tiebreak`. (Polar bit-reversal weight resolves ties to keep the high-reliability channels.)
2. **GPU NCG training, bf16**: the agent-built `ncg_isi_gpu_curriculum.py` uses bf16 + batch 512; this destroyed warm-started weights in practice. Prefer the OLD recipe (`ncg_isi_old_recipe_gpu.py`): batch 32, fp32, lr 1e-3, grad clip 1.0.
3. **Cluster NCG warm-start path**: the trainer's default `--warm-start-ckpt` is `scripts/local_analysis/ncg_models/ncg_isi_N128.pt`; ensure that file exists at that exact path on the cluster, or pass `--warm-start-ckpt` explicitly. A silent "0 tensors loaded" means the file isn't there.
4. **Multiprocessing scripts**: any script that calls `multiprocessing.Pool` from a `__main__` block must be wrapped in `if __name__ == "__main__":` (Pool's spawn re-imports the module — without the guard you get infinite recursion).
5. **N=1024 SCT eval needs ≥200K design trials** to resolve the synthesised-channel reliability tail; 50K leaves the tail dominated by tiebreak and gives BLER ~5× pessimistic.

## 11. Where to look first if something breaks

- `polar/decoder_trellis.py` doesn't compile / errors at `decode_single`: check `channel.num_states` and `channel.build_leaf_tensors(z)` shape `(N, 2, 2, S, S)`.
- NCG ckpt load fails: most likely shape mismatch between the trainer's model construction and the saved state. Use the trainer's `_make_model` to instantiate, then `load_state_dict` with `strict=False`.
- Cluster job hangs at startup: check for the multiprocessing `__main__` guard issue (item 4 above).

---

That's all for reproducibility. The decoders themselves are in
`polar/decoder_trellis.py` and `neural/{npd_memory_mac.py, ncg_isi_mac.py}` —
~500 lines each. The derivation note `docs/sct_su_to_mac.pdf` is the math
companion. Everything else is plumbing.
