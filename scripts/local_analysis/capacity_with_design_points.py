"""Overlay our actual ISI corner-rate design points on the capacity pentagon."""
import sys, os, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np
import matplotlib.pyplot as plt

DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
cap = json.load(open(os.path.join(DIR, "capacity_region_isi.json")))

R1 = cap['I_X_Z_Y_eq_R_U_max']
R2 = cap['I_Y_Z_X_eq_R_V_max']
Rs = cap['I_XY_Z_eq_R_sum_max']

DESIGNS = [
    (16,    4,   7),
    (32,    7,  15),
    (64,   15,  29),
    (128,  30,  58),
    (256,  59, 117),
    (512, 119, 233),
    (1024,239, 467),
]

fig, ax = plt.subplots(figsize=(7, 6.5))

# pentagon
pts = [(0, 0), (R1, 0), (R1, Rs - R1), (Rs - R2, R2), (0, R2), (0, 0)]
xs, ys = zip(*pts)
ax.fill(xs, ys, alpha=0.18, color='C0', label='capacity region (BPSK, N→∞)')
ax.plot(xs, ys, 'C0-', lw=2)

# dominant-face corners
ax.plot(R1, Rs - R1, 'r^', ms=11, label=f'corner (V first)  ({R1:.3f}, {Rs-R1:.3f})')
ax.plot(Rs - R2, R2, 'rs', ms=11, label=f'corner (U first)  ({Rs-R2:.3f}, {R2:.3f})')

# equal rate
Req = Rs / 2
ax.plot(Req, Req, 'g*', ms=18, label=f'equal-rate  ({Req:.3f}, {Req:.3f})')

# our design points
xs_d = [k[1] / k[0] for k in DESIGNS]
ys_d = [k[2] / k[0] for k in DESIGNS]
ax.plot(xs_d, ys_d, 'kx', ms=11, mew=2,
        label='our corner-rate designs (k_U/N, k_V/N)')
for (N, ku, kv), x, y in zip(DESIGNS, xs_d, ys_d):
    ax.annotate(f'N={N}', xy=(x, y),
                xytext=(x + 0.012, y + 0.005), fontsize=7)

# arrow from origin through capacity corner (U-first direction)
ax.annotate('', xy=(Rs - R2, R2), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1))
ax.text(0.10, 0.30, 'corner direction\n(U first)', fontsize=8,
        color='gray', alpha=0.8, rotation=63)

# 45-deg line
rmax = max(R1, R2) * 1.1
ax.plot([0, rmax], [0, rmax], 'k:', lw=0.7, alpha=0.4)

ax.set_xlabel('$R_U$ (bits / channel use)')
ax.set_ylabel('$R_V$ (bits / channel use)')
ax.set_title('ISI-MAC capacity (h=0.3, SNR=6 dB) — our design points')
ax.grid(alpha=0.3); ax.set_aspect('equal')
ax.set_xlim(-0.02, R1 * 1.12); ax.set_ylim(-0.02, R2 * 1.12)
ax.legend(loc='upper right', fontsize=8)
fig.tight_layout()
out = os.path.join(DIR, "capacity_with_designs.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
