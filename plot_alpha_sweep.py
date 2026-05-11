"""Plot MA-AGN α sweep — NPD vs SC for each α."""
import matplotlib.pyplot as plt
import numpy as np

# (α, N, NPD_bler, NPD_errs, NPD_cw, SC_bler, SC_errs, SC_cw)
# α=0.3
a3 = [
    (0.3,  64, 0.0364,  1092, 30000, 0.038,    1140, 30000),
    (0.3, 128, 0.00683,  205, 30000, 0.01687,  506, 30000),
    (0.3, 256, 0.00083,   25, 30000, 0.002033,  61, 30000),
]
# α=0.5 from main campaign + topup
a5 = [
    (0.5,  64, 0.0265,  1325, 50000, 0.03494,  1747, 50000),
    (0.5, 128, 0.00618,  309, 50000, 0.01616,   808, 50000),
    (0.5, 256, 0.00068,   34, 50000, 0.00146,    73, 50000),
    (0.5, 512, 5e-5,      25, 500000, 0.00041,   41, 100000),
    (0.5,1024, 3e-5,      24, 800000, 0.000143,  43, 300000),
]
# α=0.7
a7 = [
    (0.7,  64, 0.0363,  1090, 30000, 0.02037,   611, 30000),
    (0.7, 128, 0.00387,  116, 30000, 0.01273,   382, 30000),
    (0.7, 256, None,     None, None, 0.002033,   61, 30000),  # NPD pending
]
# α=0.9 — SC only so far
a9 = [
    (0.9,  64, None, None, None, 0.030833, 925, 30000),
    (0.9, 128, None, None, None, 0.0193,   579, 30000),
    (0.9, 256, None, None, None, 0.0042,   126, 30000),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)

def ci(k, n):
    if k == 0 or k is None or n is None: return None
    return (max(1e-7, (k - 1.96*np.sqrt(k))/n), (k + 1.96*np.sqrt(k))/n)

for ax, data, alpha in zip(axes, [a3, a5, a7, a9], [0.3, 0.5, 0.7, 0.9]):
    Ns = [row[1] for row in data]
    npd = [row[2] for row in data]
    npd_e = [row[3] for row in data]
    npd_n = [row[4] for row in data]
    sc  = [row[5] for row in data]
    sc_e = [row[6] for row in data]
    sc_n = [row[7] for row in data]

    npd_x = [n for n, b in zip(Ns, npd) if b is not None]
    npd_y = [b for b in npd if b is not None]
    sc_x = Ns; sc_y = sc

    if npd_y:
        ax.semilogy(npd_x, npd_y, 'o-', lw=2.2, ms=11, color='#d62728', label='NPD')
    ax.semilogy(sc_x, sc_y, 's-', lw=2.2, ms=10, color='#1f77b4', label='SCT decoder')

    # annotate
    for x, y, k, n in zip(npd_x, npd_y, [e for e in npd_e if e is not None], [m for m in npd_n if m is not None]):
        ax.annotate(f"{k}/{n//1000}K", (x, y), textcoords='offset points', xytext=(8, 7), fontsize=8, color='#d62728')
    for x, y, k, n in zip(sc_x, sc_y, sc_e, sc_n):
        ax.annotate(f"{k}/{n//1000}K", (x, y), textcoords='offset points', xytext=(-35, -16), fontsize=8, color='#1f77b4')

    ax.set_xscale('log', base=2); ax.set_xticks(Ns); ax.set_xticklabels([str(x) for x in Ns])
    ax.set_xlabel('Block length N', fontsize=11)
    ax.set_title(f'MA-AGN, α={alpha}', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)

axes[0].set_ylabel('BLER', fontsize=12)
plt.suptitle('MA-AGN MAC α sweep: NPD vs SCT decoder. SNR=6 dB. Annotations: errs / CW.', fontsize=12)
plt.tight_layout()
out = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results_local/maagn_alpha_sweep.png'
plt.savefig(out, dpi=130)
print(f"Saved {out}")
