"""Asymptotic capacity region for ISI-MAC r=2  Z_i = sX_i + sY_i + h1*(sX_{i-1}+sY_{i-1}) + h2*(sX_{i-2}+sY_{i-2}) + W_i.

State space:
  Joint:  (X_{i-1}, X_{i-2}, Y_{i-1}, Y_{i-2}) -> 16 states
  Y-marg: (Y_{i-1}, Y_{i-2})                   -> 4 states
"""
import sys, os, time, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np
from scipy.special import logsumexp

SIGMA2 = 10**(-0.6); SIGMA = np.sqrt(SIGMA2)
H1, H2 = 0.3, 0.15
N_TRIALS = 100
N_SYM = 3000
LOG2 = np.log(2.0)

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
OUT_JSON = os.path.join(OUT_DIR, "capacity_region_isi_r2.json")


def simulate(N, h1, h2, sigma, rng):
    x = rng.integers(0, 2, N).astype(np.int8)
    y = rng.integers(0, 2, N).astype(np.int8)
    s_x = 1 - 2 * x.astype(np.float64); s_y = 1 - 2 * y.astype(np.float64)
    s_x_p1 = np.concatenate(([1.0], s_x[:-1]))
    s_x_p2 = np.concatenate(([1.0], s_x_p1[:-1]))
    s_y_p1 = np.concatenate(([1.0], s_y[:-1]))
    s_y_p2 = np.concatenate(([1.0], s_y_p1[:-1]))
    mu = s_x + s_y + h1 * (s_x_p1 + s_y_p1) + h2 * (s_x_p2 + s_y_p2)
    z = mu + rng.normal(0.0, sigma, N)
    return x, y, z


def log_p_zxy(z, x, y, h1, h2, sigma2):
    s_x = 1 - 2 * x.astype(np.float64); s_y = 1 - 2 * y.astype(np.float64)
    s_x_p1 = np.concatenate(([1.0], s_x[:-1]))
    s_x_p2 = np.concatenate(([1.0], s_x_p1[:-1]))
    s_y_p1 = np.concatenate(([1.0], s_y[:-1]))
    s_y_p2 = np.concatenate(([1.0], s_y_p1[:-1]))
    mu = s_x + s_y + h1 * (s_x_p1 + s_y_p1) + h2 * (s_x_p2 + s_y_p2)
    return float(np.sum(-0.5 * np.log(2 * np.pi * sigma2)
                        - 0.5 * (z - mu) ** 2 / sigma2))


# Joint trellis: state s = (xp1, xp2, yp1, yp2), each {0,1} -> 16 states
#   encode as: s = xp1*8 + xp2*4 + yp1*2 + yp2
#   on input (x, y), next state s' = (x, xp1, y, yp1) = x*8 + xp1*4 + y*2 + yp1
def _build_joint_mu(h1, h2):
    """mu[s, (x,y)] = mean for transition from state s under input (x,y)."""
    mu = np.zeros((16, 4))
    for s in range(16):
        xp1 = (s >> 3) & 1; xp2 = (s >> 2) & 1
        yp1 = (s >> 1) & 1; yp2 = s & 1
        for ii in range(4):
            x = (ii >> 1) & 1; y = ii & 1
            mu[s, ii] = ((1 - 2*x) + (1 - 2*y)
                         + h1 * ((1 - 2*xp1) + (1 - 2*yp1))
                         + h2 * ((1 - 2*xp2) + (1 - 2*yp2)))
    return mu


def _next_state_joint(s, ii):
    xp1 = (s >> 3) & 1
    yp1 = (s >> 1) & 1
    x = (ii >> 1) & 1; y = ii & 1
    return x * 8 + xp1 * 4 + y * 2 + yp1


def log_p_z_joint(z, mu_table, sigma2):
    N = len(z)
    log_prior = np.log(0.25)
    log_alpha = np.full(16, -np.inf); log_alpha[0] = 0.0
    log_norm = -0.5 * np.log(2 * np.pi * sigma2)
    next_state = np.array([[_next_state_joint(s, ii) for ii in range(4)]
                           for s in range(16)])
    for i in range(N):
        ll = log_norm - 0.5 * (z[i] - mu_table) ** 2 / sigma2     # (16, 4)
        contrib = log_alpha[:, None] + log_prior + ll              # (16, 4)
        log_alpha_new = np.full(16, -np.inf)
        for s in range(16):
            for ii in range(4):
                sp = next_state[s, ii]
                log_alpha_new[sp] = np.logaddexp(log_alpha_new[sp], contrib[s, ii])
        log_alpha = log_alpha_new
    return float(logsumexp(log_alpha))


def log_p_z_given_x(z, x, h1, h2, sigma2):
    """Y-marginal trellis: state = (yp1, yp2) -> 4 states."""
    N = len(z)
    s_x = 1 - 2 * x.astype(np.float64)
    s_x_p1 = np.concatenate(([1.0], s_x[:-1]))
    s_x_p2 = np.concatenate(([1.0], s_x_p1[:-1]))
    log_prior = np.log(0.5)
    log_alpha = np.full(4, -np.inf); log_alpha[0] = 0.0   # (yp1=0, yp2=0)
    log_norm = -0.5 * np.log(2 * np.pi * sigma2)
    # state encoding: s = yp1*2 + yp2; next state s' = y*2 + yp1
    for i in range(N):
        mu_xfixed = s_x[i] + h1 * s_x_p1[i] + h2 * s_x_p2[i]
        log_alpha_new = np.full(4, -np.inf)
        for s in range(4):
            yp1 = (s >> 1) & 1; yp2 = s & 1
            for y in (0, 1):
                mu = mu_xfixed + (1 - 2*y) + h1 * (1 - 2*yp1) + h2 * (1 - 2*yp2)
                ll = log_norm - 0.5 * (z[i] - mu) ** 2 / sigma2
                sp = y * 2 + yp1
                log_alpha_new[sp] = np.logaddexp(log_alpha_new[sp],
                                                 log_alpha[s] + log_prior + ll)
        log_alpha = log_alpha_new
    return float(logsumexp(log_alpha))


def main():
    print(f"=== ISI-MAC r=2 capacity region (N->infinity) ===")
    print(f"h1={H1}, h2={H2}, sigma2={SIGMA2:.4f}, SNR={10*np.log10(1/SIGMA2):.2f} dB")
    print(f"N_trials={N_TRIALS}, N_sym={N_SYM}\n")

    rng = np.random.default_rng(13)
    mu_joint = _build_joint_mu(H1, H2)

    lp_zxy_l, lp_zx_l, lp_z_l = [], [], []
    t0 = time.time()
    for t in range(N_TRIALS):
        x, y, z = simulate(N_SYM, H1, H2, SIGMA, rng)
        lp_zxy_l.append(log_p_zxy(z, x, y, H1, H2, SIGMA2))
        lp_zx_l.append(log_p_z_given_x(z, x, H1, H2, SIGMA2))
        lp_z_l.append(log_p_z_joint(z, mu_joint, SIGMA2))
        if (t + 1) % max(1, N_TRIALS // 10) == 0:
            print(f"  trial {t+1}/{N_TRIALS}   ({time.time()-t0:.1f}s)", flush=True)

    arr_zxy = -np.array(lp_zxy_l) / N_SYM / LOG2
    arr_zx  = -np.array(lp_zx_l)  / N_SYM / LOG2
    arr_z   = -np.array(lp_z_l)   / N_SYM / LOG2
    h_zxy_closed = 0.5 * np.log(2 * np.pi * np.e * SIGMA2) / LOG2

    I_X_Z_Y = arr_zx.mean() - h_zxy_closed
    I_XY_Z  = arr_z.mean()  - h_zxy_closed
    I_X_Z   = I_XY_Z - I_X_Z_Y

    print(f"\nh(Z|X,Y) closed = {h_zxy_closed:.4f}  (MC = {arr_zxy.mean():.4f})")
    print(f"h(Z|X)          = {arr_zx.mean():.4f}")
    print(f"h(Z)            = {arr_z.mean():.4f}")
    print(f"\n=== ISI r=2 capacity region (bits/use) ===")
    print(f"I(X;Z|Y) = {I_X_Z_Y:.4f}        (= R_U max)")
    print(f"I(Y;Z|X) = {I_X_Z_Y:.4f}        (= R_V max)   [by user symmetry]")
    print(f"I(X,Y;Z) = {I_XY_Z:.4f}        (= R_sum max)")
    print(f"I(X;Z)   = {I_X_Z:.4f}")
    print(f"\nequal-rate R = I(X,Y;Z)/2 = {I_XY_Z/2:.4f} bits/use")

    # compare to ISI r=1
    cap_r1 = json.load(open(os.path.join(OUT_DIR, "capacity_region_isi.json")))
    print(f"\nvs r=1 (h=0.3):")
    print(f"  R_sum:   r=1 {cap_r1['I_XY_Z_eq_R_sum_max']:.4f}  ->  r=2 {I_XY_Z:.4f}  "
          f"(delta {I_XY_Z - cap_r1['I_XY_Z_eq_R_sum_max']:+.4f})")
    print(f"  R_U_max: r=1 {cap_r1['I_X_Z_Y_eq_R_U_max']:.4f}  ->  r=2 {I_X_Z_Y:.4f}  "
          f"(delta {I_X_Z_Y - cap_r1['I_X_Z_Y_eq_R_U_max']:+.4f})")

    json.dump(dict(channel='ISIMAC2', h1=H1, h2=H2, sigma2=SIGMA2,
                   I_X_Z_Y_eq_R_U_max=I_X_Z_Y,
                   I_Y_Z_X_eq_R_V_max=I_X_Z_Y,
                   I_XY_Z_eq_R_sum_max=I_XY_Z,
                   I_X_Z=I_X_Z, I_Y_Z=I_X_Z,
                   equal_rate=I_XY_Z/2,
                   n_trials=N_TRIALS, n_sym=N_SYM),
              open(OUT_JSON, 'w'), indent=2)
    print(f"saved {OUT_JSON}")


if __name__ == '__main__':
    main()
