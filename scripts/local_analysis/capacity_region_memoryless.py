"""Capacity region for memoryless MAC channels at SNR=6 dB, BPSK inputs.

Computes I(X;Z|Y), I(Y;Z|X), I(X,Y;Z) by direct Monte Carlo over
per-symbol joint densities (no FB needed).

Channels:
- GaussianMAC: Z = (1-2X) + (1-2Y) + W,  W~N(0, sigma^2)
- ABNMAC: Z = (1-2X) + (1-2Y) + B,  B ~ Bernoulli{-1, +1} with p_noise
  (actually: Z = X + Y + N, N ~ Bernoulli)
"""
import sys, os, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

SIGMA2 = 10**(-0.6); SIGMA = np.sqrt(SIGMA2)
N_SYM = 200_000  # one shot per channel, lots of samples
LOG2 = np.log(2.0)

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"


def gaussian_logpdf(z, mu, sig2):
    return -0.5 * np.log(2*np.pi*sig2) - 0.5 * (z - mu) ** 2 / sig2


def gaussian_mac_mi(sigma2, N):
    """I(X,Y;Z), I(X;Z|Y) for Gaussian MAC Z = (1-2X)+(1-2Y)+W."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, N); y = rng.integers(0, 2, N)
    s_x = 1 - 2*x; s_y = 1 - 2*y
    z = s_x + s_y + rng.normal(0, np.sqrt(sigma2), N)

    # log p(z | x, y) = log N(z; s_x + s_y, sigma2) — direct
    log_p_zxy = gaussian_logpdf(z, s_x + s_y, sigma2)

    # log p(z) = log [(1/4) sum_{x',y'} N(z; s_x'+s_y', sigma2)]
    mus = np.array([-2.0, 0.0, 0.0, 2.0])  # (1-2x)+(1-2y) for (x,y) in {00,01,10,11}
    log_ps = np.stack([gaussian_logpdf(z, m, sigma2) for m in mus], axis=0)
    log_p_z = logsumexp(log_ps, axis=0) + np.log(0.25)

    # log p(z | x) = log [(1/2) sum_{y'} N(z; s_x+s_y', sigma2)]
    # for x=0, mus over y are {(1)+(-1), (1)+(1)} = {0, 2}; for x=1, {-2, 0}
    log_p_zx = np.zeros(N)
    for i in range(N):
        if x[i] == 0:
            log_p_zx[i] = logsumexp([gaussian_logpdf(z[i], 0.0, sigma2),
                                     gaussian_logpdf(z[i], 2.0, sigma2)]) + np.log(0.5)
        else:
            log_p_zx[i] = logsumexp([gaussian_logpdf(z[i], -2.0, sigma2),
                                     gaussian_logpdf(z[i], 0.0, sigma2)]) + np.log(0.5)

    # MIs in bits
    I_XY_Z = float((log_p_zxy - log_p_z).mean() / LOG2)
    I_X_Z_Y = float((log_p_zxy - log_p_zx).mean() / LOG2)
    return I_X_Z_Y, I_XY_Z


def abnmac_mi(p_noise, N):
    """I for Z = X + Y + B where X,Y,B ∈ {0,1}, B ~ Ber(p_noise) (additive XOR-ish).
    Actually use Z = (1-2X) + (1-2Y) + (1-2B): real-valued."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, N); y = rng.integers(0, 2, N)
    b = rng.binomial(1, p_noise, N)
    s_x = 1 - 2*x; s_y = 1 - 2*y; s_b = 1 - 2*b
    z = s_x + s_y + s_b   # values in {-3, -1, 1, 3}

    # discrete probabilities: each z value has prob given (x,y)
    def p_z_given_xy(z, xv, yv):
        # noise distribution: P(B=0)=1-p, P(B=1)=p
        # s_b = +1 with prob 1-p, -1 with prob p
        # z = (1-2x) + (1-2y) + s_b, so s_b = z - (1-2x) - (1-2y)
        sb = z - (1-2*xv) - (1-2*yv)
        if abs(sb - 1) < 0.5: return 1 - p_noise
        if abs(sb + 1) < 0.5: return p_noise
        return 0.0

    log_p_zxy = np.array([np.log(max(p_z_given_xy(z[i], x[i], y[i]), 1e-30)) for i in range(N)])
    p_z = np.zeros(N)
    for xv in (0,1):
        for yv in (0,1):
            for i in range(N):
                p_z[i] += 0.25 * p_z_given_xy(z[i], xv, yv)
    log_p_z = np.log(np.maximum(p_z, 1e-30))
    p_zx = np.zeros(N)
    for i in range(N):
        for yv in (0,1):
            p_zx[i] += 0.5 * p_z_given_xy(z[i], x[i], yv)
    log_p_zx = np.log(np.maximum(p_zx, 1e-30))

    I_XY_Z = float((log_p_zxy - log_p_z).mean() / LOG2)
    I_X_Z_Y = float((log_p_zxy - log_p_zx).mean() / LOG2)
    return I_X_Z_Y, I_XY_Z


def main():
    print(f"=== Capacity region — memoryless MAC channels, BPSK ===\n")
    results = {}

    # Gaussian MAC at SNR=6 dB (sigma^2=0.2512)
    I_X_Z_Y, I_XY_Z = gaussian_mac_mi(SIGMA2, N_SYM)
    print(f"Gaussian MAC (sigma^2={SIGMA2:.4f}, SNR=6dB):")
    print(f"  I(X;Z|Y) = {I_X_Z_Y:.4f}")
    print(f"  I(X,Y;Z) = {I_XY_Z:.4f}")
    print(f"  equal-rate = {I_XY_Z/2:.4f} bits/use")
    results['gmac_6dB'] = {'I_X_Z_Y': I_X_Z_Y, 'I_XY_Z': I_XY_Z,
                          'sigma2': SIGMA2, 'channel': 'GaussianMAC'}

    # ABN-MAC at moderate noise
    for pn in [0.1, 0.2, 0.3]:
        I_X_Z_Y, I_XY_Z = abnmac_mi(pn, N_SYM)
        print(f"\nABN-MAC (p_noise={pn}):")
        print(f"  I(X;Z|Y) = {I_X_Z_Y:.4f}")
        print(f"  I(X,Y;Z) = {I_XY_Z:.4f}")
        print(f"  equal-rate = {I_XY_Z/2:.4f} bits/use")
        results[f'abnmac_p{int(pn*10)}'] = {'I_X_Z_Y': I_X_Z_Y, 'I_XY_Z': I_XY_Z,
                                            'p_noise': pn, 'channel': 'ABNMAC'}

    json.dump(results, open(os.path.join(OUT_DIR, "capacity_memoryless.json"), 'w'), indent=2)

    # Combined pentagon plot
    fig, ax = plt.subplots(figsize=(8, 7))
    isi_cap = json.load(open(os.path.join(OUT_DIR, "capacity_region_isi.json")))
    colors = {'isi': 'C0', 'gmac': 'C1', 'abn0.1': 'C2', 'abn0.2': 'C3', 'abn0.3': 'C4'}

    def pentagon_pts(R1, R2, Rs):
        return [(0, 0), (R1, 0), (R1, Rs - R1), (Rs - R2, R2), (0, R2), (0, 0)]

    pent_isi = pentagon_pts(isi_cap['I_X_Z_Y_eq_R_U_max'], isi_cap['I_Y_Z_X_eq_R_V_max'],
                            isi_cap['I_XY_Z_eq_R_sum_max'])
    pent_gmac = pentagon_pts(results['gmac_6dB']['I_X_Z_Y'], results['gmac_6dB']['I_X_Z_Y'],
                              results['gmac_6dB']['I_XY_Z'])

    for nm, pent, c, lbl in [
        ('isi', pent_isi, 'C0', f"ISI-MAC h=0.3, 6dB  R_sum={isi_cap['I_XY_Z_eq_R_sum_max']:.3f}"),
        ('gmac', pent_gmac, 'C1', f"GaussianMAC, 6dB  R_sum={results['gmac_6dB']['I_XY_Z']:.3f}"),
    ]:
        xs, ys = zip(*pent)
        ax.fill(xs, ys, alpha=0.18, color=c, label=lbl)
        ax.plot(xs, ys, '-', color=c, lw=1.8)

    # ABN pentagons (smaller)
    for pn_key, c in [('abnmac_p1', 'C2'), ('abnmac_p2', 'C3'), ('abnmac_p3', 'C4')]:
        if pn_key in results:
            R1 = results[pn_key]['I_X_Z_Y']; Rs = results[pn_key]['I_XY_Z']
            pn = results[pn_key]['p_noise']
            pent = pentagon_pts(R1, R1, Rs)
            xs, ys = zip(*pent)
            ax.fill(xs, ys, alpha=0.12, color=c,
                    label=f"ABN-MAC p={pn}  R_sum={Rs:.3f}")
            ax.plot(xs, ys, '-', color=c, lw=1.5)

    ax.set_xlabel('$R_U$ (bits/use)'); ax.set_ylabel('$R_V$ (bits/use)')
    ax.set_title('Capacity regions for MAC channels — BPSK inputs')
    ax.grid(alpha=0.3); ax.set_aspect('equal')
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "capacity_multiple_channels.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
