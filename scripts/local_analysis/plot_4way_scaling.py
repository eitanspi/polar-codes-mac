"""4-way BLER scaling: corner-rate r=1/r=2, equal-rate r=1/r=2 at SNR=6 dB."""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"


def ci(e, n):
    if n == 0: return 0, 0
    p = e / n; se = np.sqrt(max(p*(1-p), 1e-12)/n)
    return max(p-1.96*se, 1e-7), p+1.96*se


def load_table(files):
    out = []
    for N, f in files.items():
        if not os.path.exists(f): continue
        d = json.load(open(f))
        if 'sct' in d and d['sct'].get('done'):
            d = d['sct']
        if 'bler' in d:
            out.append((N, d['bler'], d.get('errs', 0), d.get('n_cw', 1)))
    return sorted(out)


corner_r1 = json.load(open(os.path.join(DIR, "corner_rate_sct_isi.json")))
corner_r2 = json.load(open(os.path.join(DIR, "corner_rate_sct_isi_r2.json")))

c1 = sorted([(int(k), v['bler'], v['errs'], v['n_cw']) for k, v in corner_r1.items() if v.get('done')])
c2 = sorted([(int(k), v['bler'], v['errs'], v['n_cw']) for k, v in corner_r2.items() if v.get('done')])

eq_r1_files = {
    32:  os.path.join(DIR, "equal_rate_validate_n32.json"),
    64:  os.path.join(DIR, "equal_rate_validate_n64.json"),
    128: os.path.join(DIR, "equal_rate_validate_n128_refined_k36.json"),
    256: os.path.join(DIR, "equal_rate_validate_n256_sct.json"),
    512: os.path.join(DIR, "equal_rate_validate_n512_refined.json"),
    1024: os.path.join(DIR, "equal_rate_validate_n1024_sct.json"),
}
eq_r1 = load_table(eq_r1_files)
# r=2 equal-rate
eq_r2_files = {
    128: os.path.join(DIR, "equal_rate_isi_r2_n128.json"),
    256: os.path.join(DIR, "equal_rate_isi_r2_n256.json"),
    512: os.path.join(DIR, "equal_rate_isi_r2_n512.json"),
    1024: os.path.join(DIR, "equal_rate_isi_r2_n1024.json"),
}
eq_r2 = load_table(eq_r2_files)

# NPD r=2 numbers from cluster
npd_r2 = [(256, 0.00137, 41, 30000), (512, 9.5e-5, 19, 200000), (1024, 3e-5, 9, 300000)]

fig, ax = plt.subplots(figsize=(10, 7))

def errplot(data, fmt, color, label, lw=2, ms=8):
    Ns = [d[0] for d in data]; bs = [d[1] for d in data]
    es = [d[2] for d in data]; ns = [d[3] for d in data]
    lo, hi = zip(*[ci(e, n) for e, n in zip(es, ns)])
    yerr = [np.maximum(np.array(bs) - np.array(lo), 1e-7), np.array(hi) - np.array(bs)]
    ax.errorbar(Ns, bs, yerr=yerr, fmt=fmt, color=color, label=label,
                lw=lw, ms=ms, capsize=4)
    for N, b in zip(Ns, bs):
        ax.annotate(f'{b:.4f}', xy=(N, b), xytext=(N*1.04, b*1.18), fontsize=7, color=color)

errplot(c1, 'o-', 'C0', 'ISI r=1 corner (rate 0.688)', lw=2)
errplot(c2, 'o--', 'C0', 'ISI r=2 corner (rate 0.688)', lw=1.5, ms=7)
errplot(eq_r1, 's-', 'C3', 'ISI r=1 equal-rate (rate 0.281)', lw=2)
errplot(eq_r2, 's--', 'C3', 'ISI r=2 equal-rate (rate 0.281)', lw=1.5, ms=7)
errplot(npd_r2, '^:', 'C2', 'NPD ISI r=2 corner (cluster)', lw=2, ms=10)

ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlabel('block length N'); ax.set_ylabel('BLER')
ax.set_xticks([32, 64, 128, 256, 512, 1024])
ax.set_xticklabels(['32', '64', '128', '256', '512', '1024'])
ax.set_title('Polar MAC decoding — corner vs equal-rate, ISI r=1 vs r=2 (SNR=6 dB)\n'
             'SCT analytical (solid r=1, dashed r=2) + NPD r=2 corner cluster numbers')
ax.grid(alpha=0.3, which='both'); ax.legend(loc='upper right', fontsize=8)
ax.set_ylim(1e-5, 1.0)
fig.tight_layout()
out = os.path.join(DIR, "scaling_4way.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
