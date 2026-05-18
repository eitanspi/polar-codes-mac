"""Plot all 4 ISI-MAC corner-rate decoders + the 4-state-chained-SCT diagnostic point.

Decoders:
  1. Chained SCT (2-state, original impl with per-position Y double-counting)
  2. Joint MAC SCT (full 4-state lattice — proper analytical baseline)
  3. NPD (BiGRU + neural tree, MI-designed info set)
  4. NCG (neural computational-graph SC, corner-rate)

Diagnostic overlay:
  5. 4-state chained SCT — verifies it matches joint MAC SCT (the bug fix).
"""
import os, sys
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# ISI-MAC h=0.3, SNR=6 dB, corner-rate (Class C)

# Chained SCT 2-state: small-N from low-CW eval; N=128, 256 from own-design high-CW.
chained_2s = {
    16:   (0.1642, 5000),
    32:   (0.0836, 5000),
    64:   (0.0370, 3000),
    128:  (0.0222, 10000),
    256:  (0.00740, 10000),
}

# Joint MAC SCT — RESULTS.md authoritative numbers (high CW).
# N=1024 had 0 errors / 50K eval CW; we use the 95% Poisson UCL (~6e-5)
# as a continuation point on the curve (open issue, but the best estimate we have).
joint_sct = {
    16:   (0.1501, 30000),
    32:   (0.0691, 20000),
    64:   (0.0289, 20000),
    128:  (0.00745, 20000),
    256:  (0.00185, 20000),
    512:  (0.00038, 50000),
    1024: (6e-5, 50000),
}

# NPD — RESULTS.md authoritative numbers (high CW, top-ups)
npd = {
    16:   (0.16472, 100000),
    32:   (0.06873, 100000),
    64:   (0.03284, 50000),
    128:  (0.01270, 50000),
    256:  (0.00138, 50000),
    512:  (0.000307, 300000),
    1024: (3.3e-5, 600000),
}

# NCG corner-rate
ncg = {
    16:   (0.1714, 5000),
    32:   (0.0716, 5000),
    64:   (0.0282, 5000),
    128:  (0.00967, 3000),
}

# 4-state chained SCT diagnostic overlay
chained_4s = {
    128:  (0.00640, 10000),
    256:  (0.00130, 10000),
}


def poisson_ci(bler, n_cw, conf=1.96):
    """95% Poisson CI for an observed BLER."""
    k = bler * n_cw
    if k == 0:
        return (0.0, 3.0 / n_cw)  # rule of three
    lo = max(0.0, (k - conf * np.sqrt(k)) / n_cw)
    hi = (k + conf * np.sqrt(k)) / n_cw
    return lo, hi


def add_curve(ax, data, marker, color, label, ls='-'):
    Ns = sorted(data.keys())
    blers = [data[n][0] for n in Ns]
    cis = [poisson_ci(*data[n]) for n in Ns]
    yerr_lo = [b - lo for b, (lo, _) in zip(blers, cis)]
    yerr_hi = [hi - b for b, (_, hi) in zip(blers, cis)]
    ax.errorbar(Ns, blers, yerr=[yerr_lo, yerr_hi],
                marker=marker, color=color, label=label,
                linestyle=ls, linewidth=2, markersize=8, capsize=3, alpha=0.95)


def main():
    fig, ax = plt.subplots(figsize=(10, 7))

    add_curve(ax, joint_sct,  's', '#1f77b4', 'Analytical')
    add_curve(ax, npd,        '^', '#2ca02c', 'NPD')
    add_curve(ax, ncg,        'd', '#9467bd', 'NCG')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Block length N', fontsize=12)
    ax.set_ylabel('BLER', fontsize=12)
    ax.set_title('ISI-MAC corner-rate decoders\n'
                 'h=0.3, SNR=6 dB, Class C  (error bars = 95% Poisson CI)',
                 fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(['16', '32', '64', '128', '256', '512', '1024'])
    ax.legend(loc='lower left', fontsize=10, framealpha=0.92)
    ax.set_ylim(1e-5, 0.5)

    plt.tight_layout()
    out = os.path.join(_HERE, 'four_decoders_isi_mac.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
