"""SCT (4-state chained SCT, analytical) vs NPD up to N=1024.

Uses the high-CW chained-SCT numbers from overnight v2/v3/v4 + the new
200K-design N=1024 result. NPD numbers from RESULTS.md (cluster campaign).
"""
import os
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# 4-state chained SCT (proper iid-Y analytical) at ISI-MAC h=0.3, SNR=6 dB
# Each entry: (BLER, n_cw)
chained_sct = {
    16:   (0.1500,   30000),
    32:   (0.0526,   30000),
    64:   (0.0285,   30000),
    128:  (0.0059,   30000),   # overnight v3 I3 (100K design)
    256:  (0.00130,  10000),
    512:  (0.000433, 30000),   # tighten run (50K design)
    1024: (4.0e-5,   50000),   # 200K-design rerun, 2/50K errs
}

# NPD numbers from RESULTS.md (cluster campaign, headline)
npd = {
    16:   (0.16472, 100000),
    32:   (0.06873, 100000),
    64:   (0.03284, 50000),
    128:  (0.01270, 50000),
    256:  (0.00138, 50000),
    512:  (0.000307, 300000),
    1024: (3.3e-5,  600000),
}


def poisson_ci(bler, n_cw, conf=1.96):
    k = bler * n_cw
    if k == 0:
        return (0.0, 3.0 / n_cw)
    lo = max(0.0, (k - conf * np.sqrt(k)) / n_cw)
    hi = (k + conf * np.sqrt(k)) / n_cw
    return lo, hi


def add_curve(ax, data, marker, color, label, ls='-'):
    Ns = sorted(data.keys())
    blers = [data[n][0] for n in Ns]
    ax.plot(Ns, blers, marker=marker, color=color, label=label,
            linestyle=ls, linewidth=2, markersize=9)


def main():
    fig, ax = plt.subplots(figsize=(10, 7))
    add_curve(ax, chained_sct, 's', '#1f77b4', 'SCT (4-state chained, analytical)')
    add_curve(ax, npd,         '^', '#2ca02c', 'NPD')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Block length N', fontsize=12)
    ax.set_ylabel('BLER', fontsize=12)
    ax.set_title('ISI-MAC corner-rate: analytical SCT vs NPD\n'
                 'h=0.3, SNR=6 dB, Class C',
                 fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(['16', '32', '64', '128', '256', '512', '1024'])
    ax.legend(loc='lower left', fontsize=11, framealpha=0.92)
    ax.set_ylim(1e-5, 0.5)

    plt.tight_layout()
    out = os.path.join(_HERE, 'sct_vs_npd_isi_mac.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
