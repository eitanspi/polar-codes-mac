"""Asymptotic capacity for MA-AGN (Moving Average Gaussian Noise) MAC.

Channel: Z_i = (1-2X_i) + (1-2Y_i) + N_i,  N_i = α*N_{i-1} + W_i
         where W_i ~ N(0, σ²(1-α²)) so Var(N_i) = σ² (stationary).

Equivalent (whitened): Z'_i = Z_i − α Z_{i-1} =
        (1-2X_i) + (1-2Y_i) − α [(1-2X_{i-1}) + (1-2Y_{i-1})] + W_i
i.e. ISI-MAC with tap h = -α, AWGN variance σ²(1-α²).

Computed via Arnold-Loeliger over the whitened ISI-MAC.
"""
import sys, os, time, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np
from scipy.special import logsumexp

SIGMA2_NOMINAL = 10**(-0.6)
ALPHAS = [0.3, 0.5, 0.7, 0.9]
N_TRIALS = 100
N_SYM = 3000
LOG2 = np.log(2.0)

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
OUT_JSON = os.path.join(OUT_DIR, "capacity_region_maagn.json")


def simulate(N, h, sigma, rng):
    x = rng.integers(0, 2, N).astype(np.int8); y = rng.integers(0, 2, N).astype(np.int8)
    s_x = 1 - 2*x.astype(np.float64); s_y = 1 - 2*y.astype(np.float64)
    s_x_p = np.concatenate(([1.0], s_x[:-1]))
    s_y_p = np.concatenate(([1.0], s_y[:-1]))
    mu = s_x + s_y + h*(s_x_p + s_y_p)
    z = mu + rng.normal(0, sigma, N)
    return x, y, z


def log_p_zxy(z, x, y, h, sigma2):
    s_x = 1 - 2*x.astype(np.float64); s_y = 1 - 2*y.astype(np.float64)
    s_x_p = np.concatenate(([1.0], s_x[:-1]))
    s_y_p = np.concatenate(([1.0], s_y[:-1]))
    mu = s_x + s_y + h*(s_x_p + s_y_p)
    return float(np.sum(-0.5*np.log(2*np.pi*sigma2) - 0.5*(z - mu)**2/sigma2))


def build_joint_mu(h):
    mu = np.zeros((4, 4))
    for s in range(4):
        xp, yp = s // 2, s % 2
        for ii in range(4):
            x, y = ii // 2, ii % 2
            mu[s, ii] = (1-2*x) + (1-2*y) + h*((1-2*xp) + (1-2*yp))
    return mu


def log_p_z_joint(z, mu_table, sigma2):
    N = len(z); log_prior = np.log(0.25)
    log_alpha = np.full(4, -np.inf); log_alpha[0] = 0.0
    log_norm = -0.5*np.log(2*np.pi*sigma2)
    for i in range(N):
        ll = log_norm - 0.5*(z[i] - mu_table)**2/sigma2
        contrib = log_alpha[:, None] + log_prior + ll
        log_alpha = logsumexp(contrib, axis=0)
    return float(logsumexp(log_alpha))


def log_p_z_given_x(z, x, h, sigma2):
    N = len(z); s_x = 1 - 2*x.astype(np.float64)
    s_x_p = np.concatenate(([1.0], s_x[:-1]))
    log_prior = np.log(0.5); log_norm = -0.5*np.log(2*np.pi*sigma2)
    log_alpha = np.full(2, -np.inf); log_alpha[0] = 0.0
    for i in range(N):
        mu = np.array([[s_x[i]+1+h*(s_x_p[i]+1), s_x[i]-1+h*(s_x_p[i]+1)],
                       [s_x[i]+1+h*(s_x_p[i]-1), s_x[i]-1+h*(s_x_p[i]-1)]])
        ll = log_norm - 0.5*(z[i] - mu)**2/sigma2
        contrib = log_alpha[:, None] + log_prior + ll
        log_alpha = logsumexp(contrib, axis=0)
    return float(logsumexp(log_alpha))


def main():
    print(f"=== MA-AGN capacity, σ²_stationary={SIGMA2_NOMINAL:.4f} ===")
    print(f"(via whitened equivalence: ISI tap h=-α, σ²_eff = σ²·(1-α²))\n")
    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    for alpha in ALPHAS:
        key = str(alpha)
        if key in results and results[key].get('done'):
            d = results[key]
            print(f"  α={alpha}  cached  I(X;Z|Y)={d['I_X_Z_Y']:.4f}  I(X,Y;Z)={d['I_XY_Z']:.4f}  eq={d['I_XY_Z']/2:.4f}")
            continue
        h = -alpha
        sigma2 = SIGMA2_NOMINAL * (1 - alpha**2)
        sigma = np.sqrt(sigma2)
        snr_eff_dB = 10*np.log10(1/sigma2)
        print(f"  α={alpha}: whitened ISI h={h}, σ²_eff={sigma2:.5f}, SNR_eff={snr_eff_dB:.2f} dB")
        rng = np.random.default_rng(700 + int(alpha*100))
        mu_joint = build_joint_mu(h)
        lp_zxy, lp_zx, lp_z = [], [], []
        t0 = time.time()
        for t in range(N_TRIALS):
            x, y, z = simulate(N_SYM, h, sigma, rng)
            lp_zxy.append(log_p_zxy(z, x, y, h, sigma2))
            lp_zx.append(log_p_z_given_x(z, x, h, sigma2))
            lp_z.append(log_p_z_joint(z, mu_joint, sigma2))
        arr_zxy = -np.array(lp_zxy) / N_SYM / LOG2
        arr_zx = -np.array(lp_zx) / N_SYM / LOG2
        arr_z = -np.array(lp_z) / N_SYM / LOG2
        h_zxy_closed = 0.5*np.log(2*np.pi*np.e*sigma2)/LOG2
        I_X_Z_Y = arr_zx.mean() - h_zxy_closed
        I_XY_Z = arr_z.mean() - h_zxy_closed
        print(f"    I(X;Z|Y) = {I_X_Z_Y:.4f}  I(X,Y;Z) = {I_XY_Z:.4f}  equal = {I_XY_Z/2:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        results[key] = dict(alpha=alpha, h=h, sigma2_eff=sigma2, SNR_eff_dB=snr_eff_dB,
                            I_X_Z_Y=I_X_Z_Y, I_XY_Z=I_XY_Z, I_X_Z=I_XY_Z - I_X_Z_Y,
                            equal_rate=I_XY_Z/2, done=True)
        json.dump(results, open(OUT_JSON, 'w'), indent=2)


if __name__ == '__main__':
    main()
