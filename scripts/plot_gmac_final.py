"""Generate final Gaussian MAC BLER plots with fixed MC design."""
import numpy as np, sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polar.channels import GaussianMAC
from polar.encoder import polar_encode, build_message
from polar.design import make_path
from polar.design_mc import load_design, _select_info_frozen
from polar.decoder import decode_single, decode_batch
from polar.decoder_scl import decode_single_list

sigma2 = 10**(-6/10.0)
ch = GaussianMAC(sigma2=sigma2)
I_ZX, I_ZY_X, I_ZXY = ch.capacity()

classes = {
    'C': {'pfrac': 1.0,   'Ru': I_ZX,    'Rv': I_ZY_X},
    'A': {'pfrac': 0.375, 'Ru': I_ZY_X,  'Rv': I_ZX},
    'B': {'pfrac': 0.5,   'Ru': I_ZXY/2, 'Rv': I_ZXY/2},
}
N_values = [8, 16, 32, 64, 128, 256, 512, 1024]
rho_values = [0.3, 0.5, 0.7, 0.9]
designs_dir = os.path.join(os.path.dirname(__file__), "..", "designs")
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

def run_sim(cls_name, cfg, rho, N, L=1, ncw=2000, seed=42):
    n = N.bit_length() - 1
    path_i = round(cfg['pfrac'] * N)
    su, sv, _, _, _ = load_design(f'{designs_dir}/gmac_{cls_name}_n{n}_snr6dB.npz')
    ku = max(1, min(round(rho * cfg['Ru'] * N), N-1))
    kv = max(1, min(round(rho * cfg['Rv'] * N), N-1))
    Au, fu = _select_info_frozen(N, su, ku)
    Av, fv = _select_info_frozen(N, sv, kv)
    b = make_path(N, path_i=path_i)
    rng = np.random.default_rng(seed + n*1000 + int(rho*100) + L*10000)
    
    errors = 0
    for trial in range(ncw):
        info_u = rng.integers(0, 2, ku).tolist()
        info_v = rng.integers(0, 2, kv).tolist()
        u = build_message(N, info_u, Au)
        v = build_message(N, info_v, Av)
        x = polar_encode(u.tolist())
        y = polar_encode(v.tolist())
        z = ch.sample_batch(np.array(x), np.array(y)).tolist()
        if L == 1:
            u_dec, v_dec = decode_single(N, z, b, fu, fv, ch, log_domain=True)
        else:
            u_dec, v_dec = decode_single_list(N, z, b, fu, fv, ch, log_domain=True, L=L)
        if not (all(u_dec[p-1]==bit for p,bit in zip(Au,info_u)) and
                all(v_dec[p-1]==bit for p,bit in zip(Av,info_v))):
            errors += 1
    return errors / ncw, ku, kv

# ── Collect all data ──
print("Collecting SC data...")
data = {}
for cls_name, cfg in classes.items():
    for rho in rho_values:
        for N in N_values:
            t0 = time.time()
            bler, ku, kv = run_sim(cls_name, cfg, rho, N, L=1, ncw=2000)
            elapsed = time.time() - t0
            data[(cls_name, rho, N, 1)] = bler
            print(f"  {cls_name} rho={rho} N={N:5d} L=1  ku={ku:3d} kv={kv:3d} BLER={bler:.4f} [{elapsed:.1f}s]")

print("\nCollecting SCL L=32 data...")
for cls_name, cfg in classes.items():
    for rho in [0.5, 0.7]:
        for N in N_values:
            ncw_scl = 1000 if N <= 128 else (500 if N <= 256 else (200 if N <= 512 else 100))
            t0 = time.time()
            bler, ku, kv = run_sim(cls_name, cfg, rho, N, L=32, ncw=ncw_scl)
            elapsed = time.time() - t0
            data[(cls_name, rho, N, 32)] = bler
            print(f"  {cls_name} rho={rho} N={N:5d} L=32 ku={ku:3d} kv={kv:3d} BLER={bler:.4f} [{elapsed:.1f}s]")

# Save raw data
with open(f'{results_dir}/gmac_final_data.json', 'w') as f:
    json.dump({str(k): v for k, v in data.items()}, f, indent=2)

# ── Plot 1: BLER vs N, all classes, rho=0.5 ──
fig, ax = plt.subplots(figsize=(10, 7))
styles = {'C': ('tab:blue', '-', 'o'), 'A': ('tab:orange', '--', 's'), 'B': ('tab:green', '-.', 'D')}
for cls in ['C', 'A', 'B']:
    xs = N_values
    ys = [max(data.get((cls, 0.5, N, 1), 1), 5e-4) for N in xs]
    c, ls, m = styles[cls]
    ax.semilogy(xs, ys, color=c, linestyle=ls, marker=m, linewidth=2.5, markersize=9,
                label=f'Class {cls}')
ax.axhline(0.05, color='gray', linestyle=':', alpha=0.5, label='BLER=5%')
ax.set_xlabel('Block length N', fontsize=14)
ax.set_ylabel('BLER', fontsize=14)
ax.set_title('Gaussian MAC: BLER vs N — SC Decoder, MC Design\n'
             f'SNR=6dB, $\\rho$=0.5 (50% of capacity)', fontsize=13)
ax.set_xscale('log', base=2)
ax.set_xticks(N_values)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f'$2^{{{int(np.log2(v))}}}$' if v > 0 else ''))
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
ax.set_ylim([3e-4, 1.5])
plt.tight_layout()
plt.savefig(f'{results_dir}/gmac_final_bler_vs_N_classes.png', dpi=150)
plt.close()
print(f'\nSaved: gmac_final_bler_vs_N_classes.png')

# ── Plot 2: BLER vs N, multiple rho, per class ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
colors = {0.3: 'tab:green', 0.5: 'tab:blue', 0.7: 'tab:orange', 0.9: 'tab:red'}
for ax, cls in zip(axes, ['C', 'A', 'B']):
    for rho in rho_values:
        ys = [max(data.get((cls, rho, N, 1), 1), 5e-4) for N in N_values]
        ax.semilogy(N_values, ys, color=colors[rho], marker='o', linewidth=2, markersize=7,
                    label=f'$\\rho$={rho}')
    ax.axhline(0.05, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Block length N', fontsize=12)
    ax.set_title(f'Class {cls}', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_values)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f'$2^{{{int(np.log2(v))}}}$' if v > 0 else ''))
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([3e-4, 1.5])
axes[0].set_ylabel('BLER', fontsize=13)
fig.suptitle('Gaussian MAC: BLER vs N — SC Decoder, MC Design, SNR=6dB', fontsize=14)
plt.tight_layout()
plt.savefig(f'{results_dir}/gmac_final_bler_vs_N_rho.png', dpi=150)
plt.close()
print('Saved: gmac_final_bler_vs_N_rho.png')

# ── Plot 3: SC vs SCL L=32 ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, cls in zip(axes, ['C', 'A', 'B']):
    for rho in [0.5, 0.7]:
        ys_sc = [max(data.get((cls, rho, N, 1), 1), 5e-4) for N in N_values]
        ys_scl = [max(data.get((cls, rho, N, 32), 1), 5e-4) for N in N_values]
        c = colors[rho]
        ax.semilogy(N_values, ys_sc, color=c, marker='o', linestyle='-', linewidth=2,
                    markersize=7, label=f'SC $\\rho$={rho}')
        ax.semilogy(N_values, ys_scl, color=c, marker='s', linestyle='--', linewidth=2,
                    markersize=7, label=f'SCL L=32 $\\rho$={rho}')
    ax.axhline(0.05, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Block length N', fontsize=12)
    ax.set_title(f'Class {cls}', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_values)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f'$2^{{{int(np.log2(v))}}}$' if v > 0 else ''))
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([3e-4, 1.5])
axes[0].set_ylabel('BLER', fontsize=13)
fig.suptitle('Gaussian MAC: SC vs SCL L=32 — MC Design, SNR=6dB', fontsize=14)
plt.tight_layout()
plt.savefig(f'{results_dir}/gmac_final_sc_vs_scl.png', dpi=150)
plt.close()
print('Saved: gmac_final_sc_vs_scl.png')

# ── Plot 4: BLER vs sum rate ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
palette = {8:'gray', 16:'royalblue', 32:'darkorange', 64:'forestgreen',
           128:'crimson', 256:'purple', 512:'brown', 1024:'teal'}
for ax, cls in zip(axes, ['C', 'A', 'B']):
    cfg = classes[cls]
    for N in N_values:
        n = N.bit_length() - 1
        xs, ys = [], []
        for rho in rho_values:
            ku = max(1, min(round(rho * cfg['Ru'] * N), N-1))
            kv = max(1, min(round(rho * cfg['Rv'] * N), N-1))
            sr = ku/N + kv/N
            bler = data.get((cls, rho, N, 1), None)
            if bler is not None:
                xs.append(sr)
                ys.append(max(bler, 5e-4))
        if xs:
            ax.semilogy(xs, ys, color=palette[N], marker='o', linewidth=1.5, markersize=6,
                        label=f'$N=2^{{{n}}}$')
    ax.axvline(I_ZXY, color='red', linewidth=2, alpha=0.7, label=f'$C_{{sum}}$={I_ZXY:.2f}')
    ax.set_xlabel('Sum rate (bits)', fontsize=12)
    ax.set_title(f'Class {cls}', fontsize=14)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([3e-4, 1.5])
axes[0].set_ylabel('BLER', fontsize=13)
fig.suptitle('Gaussian MAC: BLER vs Sum Rate — SC, MC Design, SNR=6dB', fontsize=14)
plt.tight_layout()
plt.savefig(f'{results_dir}/gmac_final_bler_vs_rate.png', dpi=150)
plt.close()
print('Saved: gmac_final_bler_vs_rate.png')

print('\nAll plots complete!')
