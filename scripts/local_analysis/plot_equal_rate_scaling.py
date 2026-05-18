"""BLER vs N for equal-rate decoding (SCT vs NCG)."""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
files = {
    32:  os.path.join(DIR, "equal_rate_validate_n32.json"),
    64:  os.path.join(DIR, "equal_rate_validate_n64.json"),
    128: os.path.join(DIR, "equal_rate_validate_n128_refined_k36.json"),
}
extra = {
    (128, 30): os.path.join(DIR, "equal_rate_validate_n128_refined_k30.json"),
}

Ns, sct_bler, sct_err, sct_cw = [], [], [], []
ncg_bler, ncg_err, ncg_cw = [], [], []
for N, f in files.items():
    if not os.path.exists(f): continue
    d = json.load(open(f))
    Ns.append(N)
    # accept either nested sct dict or top-level bler (refined N=128 schema)
    if 'sct' in d and d['sct'].get('done'):
        sct_bler.append(d['sct']['bler'])
        sct_err.append(d['sct']['errs']); sct_cw.append(d['sct']['n_cw'])
    elif 'bler' in d:
        sct_bler.append(d['bler'])
        sct_err.append(d['errs']); sct_cw.append(d['n_cw'])
    else:
        sct_bler.append(None); sct_err.append(0); sct_cw.append(1)
    if 'ncg' in d and d['ncg'].get('done'):
        ncg_bler.append(d['ncg']['bler'])
        ncg_err.append(d['ncg']['errs']); ncg_cw.append(d['ncg']['n_cw'])
    else:
        ncg_bler.append(None); ncg_err.append(0); ncg_cw.append(1)

# 95% Poisson CI half-width
def ci_lo_hi(e, n):
    if e == 0 or n == 0: return (0, 0)
    p = e / n
    se = np.sqrt(p * (1 - p) / n)
    return (max(p - 1.96 * se, 1e-6), p + 1.96 * se)


fig, ax = plt.subplots(figsize=(7, 5.5))
xs = np.array(Ns)
sct_arr = np.array([b if b is not None else np.nan for b in sct_bler])
ncg_arr = np.array([b if b is not None else np.nan for b in ncg_bler])

# error bars
sct_lo = []; sct_hi = []
for e, n, b in zip(sct_err, sct_cw, sct_bler):
    if b is None: sct_lo.append(np.nan); sct_hi.append(np.nan); continue
    lo, hi = ci_lo_hi(e, n); sct_lo.append(b - lo); sct_hi.append(hi - b)
ncg_lo = []; ncg_hi = []
for e, n, b in zip(ncg_err, ncg_cw, ncg_bler):
    if b is None: ncg_lo.append(np.nan); ncg_hi.append(np.nan); continue
    lo, hi = ci_lo_hi(e, n); ncg_lo.append(b - lo); ncg_hi.append(hi - b)

ax.errorbar(xs, sct_arr, yerr=[sct_lo, sct_hi], fmt='o-',
            label='SCT @ rate 0.281', color='C0', capsize=4, lw=2, ms=8)
mask_ncg = ~np.isnan(ncg_arr)
ax.errorbar(xs[mask_ncg], ncg_arr[mask_ncg],
            yerr=[np.array(ncg_lo)[mask_ncg], np.array(ncg_hi)[mask_ncg]],
            fmt='s-', label='NCG @ rate 0.281', color='C3', capsize=4, lw=2, ms=8)

# extra rate-vs-N points
for (N, k), f in extra.items():
    if not os.path.exists(f): continue
    d = json.load(open(f))
    bler = d['bler']; errs = d['errs']; n_cw = d['n_cw']
    lo, hi = ci_lo_hi(errs, n_cw)
    ax.errorbar([N], [bler], yerr=[[bler - lo], [hi - bler]],
                fmt='D', label=f'SCT @ rate {k/N:.3f} (k={k})',
                color='C2', capsize=4, lw=2, ms=10)
    ax.annotate(f'{bler:.4f}\n({errs}/{n_cw})', xy=(N, bler),
                xytext=(N * 0.85, bler * 0.7), fontsize=7, color='C2')

# annotate exact numbers
for i, N in enumerate(Ns):
    if sct_bler[i] is not None:
        ax.annotate(f'{sct_bler[i]:.4f}\n({sct_err[i]}/{sct_cw[i]})',
                    xy=(N, sct_bler[i]), xytext=(N * 1.05, sct_bler[i] * 1.15),
                    fontsize=7, color='C0')
    if ncg_bler[i] is not None:
        ax.annotate(f'{ncg_bler[i]:.4f}\n({ncg_err[i]}/{ncg_cw[i]})',
                    xy=(N, ncg_bler[i]), xytext=(N * 1.05, ncg_bler[i] * 0.7),
                    fontsize=7, color='C3')

ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel('block length N')
ax.set_ylabel('BLER')
ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
ax.set_title('Equal-rate BLER scaling — ISI-MAC h=0.3, SNR=6 dB\n'
             '(per-user rate = 0.281, path a*(N)/N ≈ 0.594)')
ax.grid(alpha=0.3, which='both')
ax.legend()
fig.tight_layout()
out = os.path.join(DIR, "equal_rate_bler_vs_n.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
