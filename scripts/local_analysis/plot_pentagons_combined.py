"""Combined plot: ISI r=1 and r=2 capacity pentagons + our design points
+ (once available) the equal-rate validation point."""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
cap1 = json.load(open(os.path.join(DIR, "capacity_region_isi.json")))
cap2 = json.load(open(os.path.join(DIR, "capacity_region_isi_r2.json")))

# corner-rate designs (k_U, k_V same for all campaigns)
DESIGNS = [(16, 4, 7), (32, 7, 15), (64, 15, 29), (128, 30, 58),
           (256, 59, 117), (512, 119, 233), (1024, 239, 467)]

# equal-rate validation point at N=32, a=19 — load if present
valid_path = os.path.join(DIR, "equal_rate_validate_n32.json")
val = json.load(open(valid_path)) if os.path.exists(valid_path) else {}


def plot_pentagon(ax, cap, color, label, alpha=0.18):
    R1 = cap['I_X_Z_Y_eq_R_U_max']; R2 = cap['I_Y_Z_X_eq_R_V_max']
    Rs = cap['I_XY_Z_eq_R_sum_max']
    pts = [(0, 0), (R1, 0), (R1, Rs - R1), (Rs - R2, R2), (0, R2), (0, 0)]
    xs, ys = zip(*pts)
    ax.fill(xs, ys, alpha=alpha, color=color, label=label)
    ax.plot(xs, ys, '-', color=color, lw=1.8)


fig, ax = plt.subplots(figsize=(7.5, 7))

# pentagons (r=2 slightly bigger → draw first, then r=1 on top)
plot_pentagon(ax, cap2, 'C1',
              f"ISI r=2 (h1=0.3, h2=0.15)  R_sum={cap2['I_XY_Z_eq_R_sum_max']:.3f}")
plot_pentagon(ax, cap1, 'C0',
              f"ISI r=1 (h=0.3)            R_sum={cap1['I_XY_Z_eq_R_sum_max']:.3f}")

# equal-rate points
for cap, c in [(cap1, 'C0'), (cap2, 'C1')]:
    Req = cap['equal_rate']
    ax.plot(Req, Req, '*', ms=18, color=c, markeredgecolor='k', markeredgewidth=0.5)

# corner-rate design points (×)
xs_d = [k[1] / k[0] for k in DESIGNS]
ys_d = [k[2] / k[0] for k in DESIGNS]
ax.plot(xs_d, ys_d, 'kx', ms=10, mew=2,
        label='our corner-rate designs (×, k_U/N, k_V/N)')
for (N, ku, kv), x, y in zip(DESIGNS, xs_d, ys_d):
    if N in (16, 1024):
        ax.annotate(f'N={N}', xy=(x, y), xytext=(x + 0.012, y + 0.005), fontsize=7)

# equal-rate validation point (★)
if val.get('sct') or val.get('ncg'):
    ku = 9; kv = 9; N = 32
    rx = ku / N; ry = kv / N
    ax.plot(rx, ry, 'gP', ms=15, label=f'equal-rate design (N=32, k=9/9)')
    txt = []
    if val.get('sct', {}).get('done'):
        txt.append(f"SCT BLER={val['sct']['bler']:.3f}")
    if val.get('ncg', {}).get('done'):
        txt.append(f"NCG BLER={val['ncg']['bler']:.3f}")
    if txt:
        ax.annotate('\n'.join(txt), xy=(rx, ry), xytext=(rx + 0.02, ry - 0.05),
                    fontsize=8, color='darkgreen')

ax.plot([0, 1], [0, 1], 'k:', lw=0.5, alpha=0.4, label='R_U = R_V')

ax.set_xlabel('$R_U$ (bits / channel use)')
ax.set_ylabel('$R_V$ (bits / channel use)')
ax.set_title('ISI-MAC capacity regions + our designs (SNR=6 dB, BPSK)')
ax.grid(alpha=0.3); ax.set_aspect('equal')
ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.02, 1.0)
ax.legend(loc='upper right', fontsize=8)
fig.tight_layout()
out = os.path.join(DIR, "pentagons_combined.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
