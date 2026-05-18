"""Side-by-side BLER vs N: corner-rate (clean length gain) vs equal-rate
(staircase finite-N anomaly)."""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"


def ci(e, n):
    if n == 0: return 0, 0
    p = e / n
    se = np.sqrt(max(p * (1 - p), 1e-12) / n)
    return max(p - 1.96*se, 1e-6), p + 1.96*se


# corner-rate (path a=N, rate 0.6875)
corner = json.load(open(os.path.join(DIR, "corner_rate_sct_isi.json")))
Cn = []; Cb = []; Cei = []; Cew = []
for k in sorted(corner.keys(), key=int):
    d = corner[k]
    Cn.append(d['N']); Cb.append(d['bler'])
    Cei.append(d['errs']); Cew.append(d['n_cw'])

# equal-rate (staircase a*(N), rate 0.281)
eq_files = {
    32:  os.path.join(DIR, "equal_rate_validate_n32.json"),
    64:  os.path.join(DIR, "equal_rate_validate_n64.json"),
    128: os.path.join(DIR, "equal_rate_validate_n128_refined_k36.json"),
    256: os.path.join(DIR, "equal_rate_validate_n256_sct.json"),
    512: os.path.join(DIR, "equal_rate_validate_n512_refined.json"),
    1024: os.path.join(DIR, "equal_rate_validate_n1024_sct.json"),
}
En = []; Eb = []; Eei = []; Eew = []
for N, f in eq_files.items():
    if not os.path.exists(f): continue
    d = json.load(open(f))
    if 'sct' in d:
        En.append(N); Eb.append(d['sct']['bler'])
        Eei.append(d['sct']['errs']); Eew.append(d['sct']['n_cw'])
    elif 'bler' in d:
        En.append(N); Eb.append(d['bler'])
        Eei.append(d['errs']); Eew.append(d['n_cw'])

# equal-rate at k=30, N=128 (lower rate)
extra_f = os.path.join(DIR, "equal_rate_validate_n128_refined_k30.json")
extra = json.load(open(extra_f)) if os.path.exists(extra_f) else None

fig, ax = plt.subplots(figsize=(8, 6))

# corner
lo, hi = zip(*[ci(e, n) for e, n in zip(Cei, Cew)])
ax.errorbar(Cn, Cb, yerr=[np.array(Cb)-lo, hi-np.array(Cb)],
            fmt='o-', color='C0', ms=9, lw=2, capsize=4,
            label='corner-rate path (a=N), rate 0.688')

# equal
lo, hi = zip(*[ci(e, n) for e, n in zip(Eei, Eew)])
ax.errorbar(En, Eb, yerr=[np.array(Eb)-lo, hi-np.array(Eb)],
            fmt='s-', color='C3', ms=9, lw=2, capsize=4,
            label='equal-rate staircase (a≈0.594N), rate 0.281')

if extra is not None:
    elo, ehi = ci(extra['errs'], extra['n_cw'])
    ax.errorbar([extra['N']], [extra['bler']],
                yerr=[[extra['bler']-elo], [ehi-extra['bler']]],
                fmt='D', color='C2', ms=11, lw=2, capsize=4,
                label=f'equal-rate, k={extra["k_u"]} (rate {extra["k_u"]/extra["N"]:.3f})')
    ax.annotate(f"{extra['bler']:.4f}\n({extra['errs']}/{extra['n_cw']})",
                xy=(extra['N'], extra['bler']),
                xytext=(extra['N']*0.85, extra['bler']*0.7),
                fontsize=8, color='C2')

# annotate values
for N, b, e, w in zip(Cn, Cb, Cei, Cew):
    ax.annotate(f"{b:.4f}", xy=(N, b), xytext=(N*1.05, b*1.15), fontsize=8, color='C0')
for N, b, e, w in zip(En, Eb, Eei, Eew):
    ax.annotate(f"{b:.4f}", xy=(N, b), xytext=(N*1.05, b*1.18), fontsize=8, color='C3')

ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlabel('block length N'); ax.set_ylabel('BLER')
ax.set_xticks([32, 64, 128, 256, 512, 1024]); ax.set_xticklabels(['32', '64', '128', '256', '512', '1024'])
ax.set_title('Corner-rate vs equal-rate BLER scaling — ISI-MAC h=0.3, SNR=6 dB\n'
             '(SCT analytical decoder, same channel & SNR)')
ax.grid(alpha=0.3, which='both'); ax.legend(loc='upper right', fontsize=9)
fig.tight_layout()
out = os.path.join(DIR, "corner_vs_equal_rate.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
