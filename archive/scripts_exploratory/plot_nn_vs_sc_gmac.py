"""Plot NN vs SC BLER for GMAC Class B at SNR=6dB — all decoder variants."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# All data: GMAC SNR=6dB, Class B, Ru≈Rv≈0.48
N_values = [32, 64, 128, 256, 512, 1024]

data = {
    'SC': {
        'bler': [0.046, 0.025, 0.016, 0.005, 0.001, 0.001],
        'marker': 's', 'color': 'black', 'ls': '-', 'lw': 2,
    },
    'SCL (L=4)': {
        'bler': [0.0264, 0.0126, 0.0084, 0.0005, None, None],
        'marker': 'D', 'color': 'gray', 'ls': '--', 'lw': 1.5,
    },
    'NN-SC (d=16)': {
        'bler': [0.056, 0.026, 0.019, 0.019, 0.045, 0.069],
        'marker': 'o', 'color': '#e74c3c', 'ls': '-', 'lw': 2,
    },
    'NN-SCL (L=4)': {
        'bler': [0.022, 0.013, 0.015, 0.026, 0.045, 0.045],
        'marker': '^', 'color': '#2ecc71', 'ls': '-', 'lw': 2,
    },
}

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

for label, d in data.items():
    ns = []
    blers = []
    for n, b in zip(N_values, d['bler']):
        if b is not None and b > 0:
            ns.append(n)
            blers.append(b)
    ax.semilogy(ns, blers, marker=d['marker'], color=d['color'],
                linestyle=d['ls'], linewidth=d['lw'], markersize=8,
                label=label, zorder=3 if 'NN' in label else 2)

# Highlight the "NN wins" region
ax.axvspan(16, 80, alpha=0.08, color='green', zorder=0)
ax.text(48, 0.002, 'NN wins', fontsize=10, ha='center', color='green', alpha=0.7)

# Highlight the "NN fails" region
ax.axvspan(200, 1200, alpha=0.08, color='red', zorder=0)
ax.text(500, 0.002, 'NN degrades', fontsize=10, ha='center', color='red', alpha=0.7)

ax.set_xlabel('Block Length N', fontsize=13)
ax.set_ylabel('Block Error Rate (BLER)', fontsize=13)
ax.set_title('GMAC Class B (SNR=6dB, Ru$\\approx$Rv$\\approx$0.48)\nNeural vs Analytical Decoders', fontsize=14)
ax.set_xscale('log', base=2)
ax.set_xticks(N_values)
ax.set_xticklabels([str(n) for n in N_values])
ax.set_ylim(1e-4, 0.15)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, which='both')

# Add ratio annotations
for i, n in enumerate(N_values):
    sc = data['SC']['bler'][i]
    nn = data['NN-SC (d=16)']['bler'][i]
    if sc and nn:
        ratio = nn / sc
        if ratio > 1.5:
            ax.annotate(f'{ratio:.0f}x', (n, nn), textcoords="offset points",
                       xytext=(12, 5), fontsize=8, color='#e74c3c', alpha=0.8)
        elif ratio <= 1.2:
            ax.annotate(f'{ratio:.1f}x', (n, nn), textcoords="offset points",
                       xytext=(12, -10), fontsize=8, color='#27ae60', alpha=0.8)

plt.tight_layout()

out_path = 'results/gmac_snr6dB/nn_vs_sc_all_decoders.pdf'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'Saved: {out_path.replace(".pdf", ".png")}')
