"""Asymptotic capacity region (pentagon) for ISI-MAC at N -> infinity.

Recipe (Arnold-Loeliger 2006, applied per-MAC):
  H(Z|X,Y)  = closed form  0.5 log2(2 pi e sigma^2)  [bits/use]
  H(Z|X)    = (1/N) E[-log p(Z^N | X^N)]   via 2-state FW sum-product on Y
  H(Z)      = (1/N) E[-log p(Z^N)]         via 4-state joint FW sum-product

Pentagon:
  R_U <= I(X;Z|Y) = H(Z|X) - H(Z|X,Y)            (by user symmetry)
  R_V <= I(Y;Z|X)
  R_U + R_V <= I(X,Y;Z) = H(Z) - H(Z|X,Y)

Plus marginals (other pentagon sides):
  I(X;Z) = I(X,Y;Z) - I(Y;Z|X)
"""
import sys, os, time, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

SIGMA2 = 10**(-0.6); SIGMA = np.sqrt(SIGMA2)
H_TAP = 0.3
N_TRIALS = 200
N_SYM = 4000
LOG2 = np.log(2.0)

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
OUT_PNG = os.path.join(OUT_DIR, "capacity_region_isi.png")
OUT_JSON = os.path.join(OUT_DIR, "capacity_region_isi.json")


def simulate(N, h, sigma, rng):
    x = rng.integers(0, 2, N).astype(np.int8)
    y = rng.integers(0, 2, N).astype(np.int8)
    s_x = 1 - 2 * x.astype(np.float64)
    s_y = 1 - 2 * y.astype(np.float64)
    s_x_prev = np.concatenate(([1.0], s_x[:-1]))    # X_{-1}=0 -> s=+1
    s_y_prev = np.concatenate(([1.0], s_y[:-1]))
    mu = s_x + s_y + h * (s_x_prev + s_y_prev)
    z = mu + rng.normal(0.0, sigma, N)
    return x, y, z


def log_p_zxy(z, x, y, h, sigma2):
    s_x = 1 - 2 * x.astype(np.float64); s_y = 1 - 2 * y.astype(np.float64)
    s_x_prev = np.concatenate(([1.0], s_x[:-1]))
    s_y_prev = np.concatenate(([1.0], s_y[:-1]))
    mu = s_x + s_y + h * (s_x_prev + s_y_prev)
    return float(np.sum(-0.5 * np.log(2 * np.pi * sigma2)
                        - 0.5 * (z - mu) ** 2 / sigma2))


# Pre-build transition tables
# states are indexed 0..3 as 2*xp + yp ; inputs (x,y) also as 2*x+y
def _build_joint_tables(h):
    # mu_table[sp, in] -> mean for state transition sp -> (next state = in)
    mu = np.zeros((4, 4))
    for sp in range(4):
        xp = sp // 2; yp = sp % 2
        for ii in range(4):
            x = ii // 2; y = ii % 2
            mu[sp, ii] = (1 - 2 * x) + (1 - 2 * y) + h * ((1 - 2 * xp) + (1 - 2 * yp))
    return mu


def log_p_z_joint(z, mu_table, sigma2):
    """4-state FW sum-product, prior 1/4 over each input."""
    N = len(z)
    log_prior = np.log(0.25)
    log_alpha = np.full(4, -np.inf); log_alpha[0] = 0.0   # init state (0,0)
    log_norm_const = -0.5 * np.log(2 * np.pi * sigma2)
    for i in range(N):
        ll = log_norm_const - 0.5 * (z[i] - mu_table) ** 2 / sigma2   # (4, 4)
        # contrib[sp, ii] = log_alpha[sp] + log_prior + ll[sp, ii]
        contrib = log_alpha[:, None] + log_prior + ll
        # next state = ii  (input determines next state)
        log_alpha = logsumexp(contrib, axis=0)
    return float(logsumexp(log_alpha))


def log_p_z_given_x(z, x, h, sigma2):
    """2-state FW sum-product over Y (X is known sequence)."""
    N = len(z)
    s_x = 1 - 2 * x.astype(np.float64)
    s_x_prev = np.concatenate(([1.0], s_x[:-1]))
    log_prior = np.log(0.5)
    log_alpha = np.full(2, -np.inf); log_alpha[0] = 0.0
    log_norm_const = -0.5 * np.log(2 * np.pi * sigma2)
    # mu[yp, y] = s_x[i] + (1-2y) + h*(s_x_prev[i] + (1-2yp))  — depends on i
    for i in range(N):
        # build 2x2 mu and ll
        mu = np.array([[s_x[i] + 1 + h * (s_x_prev[i] + 1),       # yp=0,y=0
                        s_x[i] - 1 + h * (s_x_prev[i] + 1)],      # yp=0,y=1
                       [s_x[i] + 1 + h * (s_x_prev[i] - 1),       # yp=1,y=0
                        s_x[i] - 1 + h * (s_x_prev[i] - 1)]])     # yp=1,y=1
        ll = log_norm_const - 0.5 * (z[i] - mu) ** 2 / sigma2
        contrib = log_alpha[:, None] + log_prior + ll
        log_alpha = logsumexp(contrib, axis=0)
    return float(logsumexp(log_alpha))


def main():
    print(f"=== ISI-MAC capacity region (N->infinity) ===")
    print(f"h={H_TAP}, sigma2={SIGMA2:.4f}, SNR(per user) = "
          f"{10*np.log10(1/SIGMA2):.2f} dB  (1/sigma2)")
    print(f"N_trials={N_TRIALS}, N_sym={N_SYM}\n")

    rng = np.random.default_rng(42)
    mu_joint = _build_joint_tables(H_TAP)

    lp_zxy_l, lp_zx_l, lp_z_l = [], [], []
    t0 = time.time()
    for t in range(N_TRIALS):
        x, y, z = simulate(N_SYM, H_TAP, SIGMA, rng)
        lp_zxy_l.append(log_p_zxy(z, x, y, H_TAP, SIGMA2))
        lp_zx_l.append(log_p_z_given_x(z, x, H_TAP, SIGMA2))
        lp_z_l.append(log_p_z_joint(z, mu_joint, SIGMA2))
        if (t + 1) % max(1, N_TRIALS // 10) == 0:
            print(f"  trial {t+1}/{N_TRIALS}   ({time.time()-t0:.1f}s)", flush=True)

    # -> bits / use
    arr_zxy = -np.array(lp_zxy_l) / N_SYM / LOG2
    arr_zx  = -np.array(lp_zx_l)  / N_SYM / LOG2
    arr_z   = -np.array(lp_z_l)   / N_SYM / LOG2
    h_zxy, h_zx, h_z = arr_zxy.mean(), arr_zx.mean(), arr_z.mean()
    se_zxy = arr_zxy.std() / np.sqrt(N_TRIALS)
    se_zx  = arr_zx.std()  / np.sqrt(N_TRIALS)
    se_z   = arr_z.std()   / np.sqrt(N_TRIALS)

    h_zxy_closed = 0.5 * np.log(2 * np.pi * np.e * SIGMA2) / LOG2

    print(f"\nh(Z|X,Y) = {h_zxy:.4f} +/- {se_zxy:.4f}  (closed form = {h_zxy_closed:.4f})")
    print(f"h(Z|X)   = {h_zx:.4f}  +/- {se_zx:.4f}")
    print(f"h(Z)     = {h_z:.4f}   +/- {se_z:.4f}")

    I_X_Z_Y = h_zx - h_zxy_closed                  # I(X;Z|Y) by user symmetry
    I_Y_Z_X = I_X_Z_Y
    I_XY_Z  = h_z  - h_zxy_closed
    I_X_Z   = I_XY_Z - I_Y_Z_X
    I_Y_Z   = I_XY_Z - I_X_Z_Y

    print(f"\n=== Capacity region (bits/use) ===")
    print(f"I(X;Z|Y) = {I_X_Z_Y:.4f}        (= R_U  max)")
    print(f"I(Y;Z|X) = {I_Y_Z_X:.4f}        (= R_V  max)")
    print(f"I(X,Y;Z) = {I_XY_Z:.4f}        (= R_U+R_V max)")
    print(f"I(X;Z)   = {I_X_Z:.4f}")
    print(f"I(Y;Z)   = {I_Y_Z:.4f}")
    print(f"\nequal-rate point R = I(X,Y;Z)/2 = {I_XY_Z/2:.4f} bits/use")

    # Compare with finite-N=32 polar MC search result
    print(f"\nfinite-N=32 polar MC (from equal_rate_search):")
    print(f"  R_U_max  = 0.5316,  R_V_max = 0.5356,  R_sum_max = 0.8175")
    print(f"finite-N gap vs N->infinity:")
    print(f"  R_U: {I_X_Z_Y - 0.5316:+.4f},  R_sum: {I_XY_Z - 0.8175:+.4f}")

    # ── pentagon plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    R1, R2, Rs = I_X_Z_Y, I_Y_Z_X, I_XY_Z
    # pentagon vertices (CCW):
    pts = [(0, 0), (R1, 0), (R1, Rs - R1), (Rs - R2, R2), (0, R2), (0, 0)]
    xs, ys = zip(*pts)
    ax.fill(xs, ys, alpha=0.25, color='C0', label='capacity region (BPSK, N→∞)')
    ax.plot(xs, ys, 'C0-', lw=2)

    # mark corners
    ax.plot(R1, 0, 'ko', ms=5)
    ax.plot(0, R2, 'ko', ms=5)
    ax.annotate(f'(I(X;Z|Y), 0) = ({R1:.3f}, 0)',
                xy=(R1, 0), xytext=(R1+0.01, 0.01), fontsize=8)
    ax.annotate(f'(0, I(Y;Z|X)) = (0, {R2:.3f})',
                xy=(0, R2), xytext=(0.01, R2+0.01), fontsize=8)

    # dominant-face corners (path family endpoints)
    cA = (R1, Rs - R1); cC = (Rs - R2, R2)
    ax.plot(*cA, 'r^', ms=10, label=f'corner a=N  ({cA[0]:.3f}, {cA[1]:.3f})')
    ax.plot(*cC, 'rs', ms=10, label=f'corner a=0  ({cC[0]:.3f}, {cC[1]:.3f})')

    # equal-rate point
    Req = Rs / 2
    ax.plot(Req, Req, 'g*', ms=16, label=f'equal rate  ({Req:.3f}, {Req:.3f})')

    # 45-deg line
    rmax = max(R1, R2) * 1.1
    ax.plot([0, rmax], [0, rmax], 'k:', lw=0.7, alpha=0.5, label='R_U = R_V')

    ax.set_xlabel('$R_U$ (bits / channel use)')
    ax.set_ylabel('$R_V$ (bits / channel use)')
    ax.set_title(f'ISI-MAC capacity region — h={H_TAP}, '
                 f'SNR={10*np.log10(1/SIGMA2):.1f} dB, BPSK')
    ax.grid(alpha=0.3); ax.set_aspect('equal')
    ax.set_xlim(-0.02, R1 * 1.15); ax.set_ylim(-0.02, R2 * 1.15)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nsaved {OUT_PNG}")

    json.dump({
        'channel': {'name': 'ISIMAC', 'h': H_TAP, 'sigma2': SIGMA2,
                    'SNR_dB': float(10*np.log10(1/SIGMA2))},
        'n_trials': N_TRIALS, 'n_sym': N_SYM,
        'h_zxy_closed': h_zxy_closed,
        'h_zx_mc': h_zx, 'h_zx_se': se_zx,
        'h_z_mc': h_z,   'h_z_se':  se_z,
        'I_X_Z_Y_eq_R_U_max': I_X_Z_Y,
        'I_Y_Z_X_eq_R_V_max': I_Y_Z_X,
        'I_XY_Z_eq_R_sum_max': I_XY_Z,
        'I_X_Z': I_X_Z, 'I_Y_Z': I_Y_Z,
        'equal_rate': Req,
    }, open(OUT_JSON, 'w'), indent=2)
    print(f"saved {OUT_JSON}")


if __name__ == '__main__':
    main()
