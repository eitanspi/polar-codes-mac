# Session Handoff — 2026-03-29

## What was done

Starting from VALIDATION_REPORT.md (7 issues), all items were addressed:

| # | Item | Status |
|---|------|--------|
| 1 | `.gitignore` | Already existed — no action needed |
| 2 | `.DS_Store` in git | Already ignored/untracked — no action needed |
| 3 | `__pycache__/` in git | Already ignored/untracked — no action needed |
| 4 | LaTeX build artifacts in `docs/` | **Deleted** (6 files: .aux, .log, .out, .toc in docs/ and results/good_weird/) |
| 5 | `scripts/run_all_gmac_designs.sh` hardcoded path | **Fixed** — replaced with `cd "$(dirname "$0")/.."` |
| 6 | arXiv placeholder in README | **Fixed** — replaced `arXiv:2501.xxxxx` with `in preparation` |
| 7 | No `requirements.txt` | **Created** — numpy, matplotlib, torch, numba |

### Additional fixes completed
- **Removed `../../` sys.path entries** from 4 scripts: `plot_gmac_final.py`, `campaign_28h.py`, `campaign_10h.py`, `overnight_gmac.py`. The `../` entry already resolves all imports.
- **README GaussianMAC docs** — added constructor clarification: `GaussianMAC(sigma2=...)` and `GaussianMAC.from_snr_db(snr_db)`.

## What was NOT done (needs attention)

### 1. Final audit
Verify all changes are correct:
- `git status` to see all modified/new files
- `python3 -m py_compile` on each modified script to confirm no syntax errors
- Check the README edit and requirements.txt look right

### 2. Legacy cross-project scripts (left alone)
These scripts have hardcoded absolute paths to sibling projects (`to_git_v3`, `nn_mac`, `to_git`). They were intentionally left untouched because they reference code outside this repo:
- `scripts/eval_bemac_nn_scl.py` — refs `../../nn_mac`
- `scripts/eval_bemac_nn_scl_large_N.py` — refs `../../nn_mac`
- `scripts/eval_bemac_classA_nn.py` — refs `to_git_v3` and `nn_mac`
- `scripts/eval_bemac_classB_scl32.py` — refs `to_git_v3`
- `scripts/eval_bemac_classC_nn_50k.py` — refs `to_git` (older version)
- `scripts/extend_classB_Ru39_10K.py` — refs `to_git_v3` and `to_git_v2`

**Decision needed**: Should these be refactored to use the in-repo `neural/` module, or are they archival scripts that won't be included in the final release?

### 3. Nothing has been committed yet
All changes are unstaged. When ready: review with `git diff`, stage, and commit.

## Files modified/created (all under to_git_v2/)
- `scripts/run_all_gmac_designs.sh` — fixed hardcoded path
- `scripts/plot_gmac_final.py` — removed ../../ sys.path
- `scripts/campaign_28h.py` — removed ../../ sys.path
- `scripts/campaign_10h.py` — removed ../../ sys.path
- `scripts/overnight_gmac.py` — removed ../../ sys.path
- `README.md` — arXiv placeholder fix + GaussianMAC constructor docs
- `requirements.txt` — **new file**
- `docs/` — deleted .aux, .log, .out, .toc artifacts
