"""Generate 6-page summary PDF: 3 pages for ISI (math/table/plot), 3 for MA-AGN."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'DejaVu Sans'

# ── ISI-MAC numbers (latest) ─────────────────────────────────────────────────
N_isi = [16, 32, 64, 128, 256, 512, 1024]
nn_isi  = [0.16472, 0.06873, 0.03284, 0.0127, 0.00138, 0.000307, 3.3e-5]
nn_isi_e = [16472, 6873, 1642, 637, 69, 92, 20]
nn_isi_n = [100_000, 100_000, 50_000, 50_000, 50_000, 300_000, 600_000]
sct_isi  = [0.1501, 0.0691, 0.0289, 0.00745, 0.00185, 0.000433, 2.68e-5]
sct_isi_e = [4503, 1403, 578, 149, 37, 39, 9]
sct_isi_n = [30_000, 20_000, 20_000, 20_000, 20_000, 90_000, 336_000]

# ── MA-AGN numbers (memoryless SC; whitened SCT in progress) ─────────────────
N_ma = [16, 32, 64, 128, 256, 512, 1024]
nn_ma  = [0.14654, 0.09925, 0.0265, 0.00618, 0.00068, 5.17e-5, 3e-5]
nn_ma_e = [7327, 9925, 1325, 309, 34, 31, 30]
nn_ma_n = [50_000, 100_000, 50_000, 50_000, 50_000, 600_000, 1_050_000]
sc_ma  = [0.10678, 0.0402, 0.03494, 0.01616, 0.00146, 0.000409, 0.000143]
sc_ma_e = [5339, 2010, 1747, 808, 73, 41, 43]
sc_ma_n = [50_000, 50_000, 50_000, 50_000, 50_000, 100_000, 300_000]

def page_math_isi(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    y = 0.96
    ax.text(0.5, y, 'ISI-MAC — channel, decoder, capacity', ha='center', fontsize=18, weight='bold')
    y -= 0.06
    ax.text(0.06, y,
        'Two-user multiple access channel with deterministic intersymbol interference (ISI).\n'
        'Two transmitters encode polar codewords $x^N, y^N \\in \\{0,1\\}^N$ and BPSK-modulate.',
        fontsize=11, va='top', wrap=True)
    y -= 0.10
    ax.text(0.06, y, 'Channel model', fontsize=13, weight='bold')
    y -= 0.04
    ax.text(0.10, y, r'$Z_i = (1-2X_i) + (1-2Y_i) + h\,[(1-2X_{i-1}) + (1-2Y_{i-1})] + W_i$', fontsize=12)
    y -= 0.035
    ax.text(0.10, y, r'$W_i \sim \mathcal{N}(0,\sigma^2)$ i.i.d.,'
                    r'   $h=0.3$ (tap),  $\sigma^2 = 10^{-0.6}$ (SNR$=6$ dB).', fontsize=11)
    y -= 0.045
    ax.text(0.06, y, 'Channel state', fontsize=13, weight='bold')
    y -= 0.04
    ax.text(0.10, y,
        r'$S_i = (X_{i-1}, Y_{i-1}) \in \{0,1\}^2$,  $|\mathcal{S}|=4$.   '
        r'Transition is deterministic: $S_{i+1}\!=\!(X_i,Y_i)$.', fontsize=11)
    y -= 0.05
    ax.text(0.06, y, 'Analytical decoder (SCT) — joint forward–backward + computational-graph SC', fontsize=13, weight='bold')
    y -= 0.035
    ax.text(0.10, y, 'Forward and backward recursions on $\\mathcal{S}$:', fontsize=11)
    y -= 0.04
    ax.text(0.10, y,
        r'$\alpha_t(s) \;=\; \log\!\!\sum_{x,y,s_-}\!\! e^{\alpha_{t-1}(s_-)}'
        r'\,W(z_t|x,y,s_-,s)\cdot P(x)P(y)$', fontsize=11)
    y -= 0.04
    ax.text(0.10, y,
        r'$\beta_t(s) \;=\; \log\!\!\sum_{x,y,s_+}\!\! W(z_{t+1}|x,y,s,s_+)'
        r'\,e^{\beta_{t+1}(s_+)}\,P(x)P(y)$', fontsize=11)
    y -= 0.04
    ax.text(0.10, y, 'Per-position joint marginal:', fontsize=11)
    y -= 0.04
    ax.text(0.10, y,
        r'$\log P(z_1^N, X_t=x, Y_t=y) = \mathrm{LSE}_{s_-, s_+}\![\alpha_{t-1}(s_-) + \log W(z_t|x,y,s_-,s_+) + \beta_{t+1}(s_+)]$',
        fontsize=10)
    y -= 0.05
    ax.text(0.10, y,
        r'The per-position $(2{\times}2)$ marginals feed a Ren-et-al-style computational-graph SC tree '
        r'that decodes along an arbitrary path through the rate region. Class C corner-rate (all U then all V).',
        fontsize=10.5, wrap=True)
    y -= 0.07
    ax.text(0.06, y, 'Neural decoder (NPD) — chained corner-rate', fontsize=13, weight='bold')
    y -= 0.035
    ax.text(0.10, y,
        r'Stage 1 ($U$): BiGRU over $z_1^N$ produces per-position embeddings; neural CheckNode/BitNode/$\mathrm{Emb2LLR}$',
        fontsize=11)
    y -= 0.035
    ax.text(0.10, y, r'tree decodes $U$. Stage 2 ($V$): same architecture with $\hat X$ as extra per-position side info.', fontsize=11)
    y -= 0.04
    ax.text(0.10, y, r'Trained rate-1 (all positions information), then MI per synthesized channel $W_N^{(i)}$ selects $\mathcal{A}_U, \mathcal{A}_V$.', fontsize=11)
    y -= 0.06
    ax.text(0.06, y, 'Symmetric capacity and rate', fontsize=13, weight='bold')
    y -= 0.04
    ax.text(0.10, y, r'$I(W) = I(X,Y;Z)$ under uniform $X,Y$. Class C corner-rate uses', fontsize=11)
    y -= 0.04
    ax.text(0.10, y, r'$R_U = I(X;Z), \quad R_V = I(Y;Z\,|\,X)$,  total throughput $R = (R_U+R_V)/2$.', fontsize=11)
    y -= 0.04
    ax.text(0.10, y, r'In our experiments $(k_U,k_V)$ follows the GMAC proxy: $(R_U,R_V) \approx (0.23,\,0.46)$.', fontsize=11)
    y -= 0.07
    ax.text(0.06, y, 'Frozen-set design', fontsize=13, weight='bold')
    y -= 0.04
    ax.text(0.10, y,
        r'NPD: MI of each $W_N^{(i)}$ measured via teacher-forced fast-CE on 100K rate-1 codewords; pick top-$k$ MI positions.',
        fontsize=11)
    y -= 0.035
    ax.text(0.10, y,
        r'SCT: genie SC at "design SNR" $\!=\!3$ dB on the actual ISI-MAC, ranks positions by per-position BCE.',
        fontsize=11)
    pdf.savefig(fig); plt.close(fig)

def page_math_maagn(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    y = 0.96
    ax.text(0.5, y, 'MA-AGN-MAC — channel, decoder, capacity', ha='center', fontsize=18, weight='bold')
    y -= 0.06
    ax.text(0.06, y,
        'Two-user MAC with moving-average additive Gaussian noise. The noise process is AR(1) — the\n'
        'noise state is a continuous real number, so no finite-state trellis applies directly.',
        fontsize=11, va='top', wrap=True)
    y -= 0.10
    ax.text(0.06, y, 'Channel model', fontsize=13, weight='bold')
    y -= 0.04
    ax.text(0.10, y, r'$Z_i = (1-2X_i) + (1-2Y_i) + N_i,    N_i = \alpha\,N_{i-1} + W_i$', fontsize=12)
    y -= 0.035
    ax.text(0.10, y, r'$W_i \sim \mathcal{N}(0, \sigma^2(1-\alpha^2))$ i.i.d.,  '
                    r'$\mathrm{Var}(N_i) = \sigma^2$ (stationary),  $\alpha=0.5$, SNR$=6$ dB.',
        fontsize=11)
    y -= 0.05
    ax.text(0.06, y, 'Why no off-the-shelf SCT', fontsize=13, weight='bold')
    y -= 0.035
    ax.text(0.10, y,
        r'The noise carries an infinite-precision real state $N_{i-1}$ across positions. A naïve trellis on the noise',
        fontsize=11)
    y -= 0.035
    ax.text(0.10, y,
        r'state has $|\mathcal{S}|=\infty$. Aharoni et al. 2024 state "no available SC decoding rule for channels with continuous',
        fontsize=11)
    y -= 0.035
    ax.text(0.10, y, r'state space" and only show NPD curves for MA-AGN in their Fig. 4.', fontsize=11)
    y -= 0.06
    ax.text(0.06, y, 'Memoryless SC baseline (used in this report)', fontsize=13, weight='bold')
    y -= 0.04
    ax.text(0.10, y, 'Treat the AR(1) noise as i.i.d. Gaussian with variance $\\sigma^2$.', fontsize=11)
    y -= 0.04
    ax.text(0.10, y,
        r'Stage 1 LLR (V marginalized uniform):  $\mathrm{LLR}_U(z) = \log\frac{\mathcal{N}(z;2,\sigma^2)+\mathcal{N}(z;0,\sigma^2)}{\mathcal{N}(z;0,\sigma^2)+\mathcal{N}(z;-2,\sigma^2)}$',
        fontsize=10)
    y -= 0.05
    ax.text(0.10, y,
        r'Stage 2 LLR (given $\hat x_i$):  $\mathrm{LLR}_V(z;\hat x_i) = \frac{(z-m_0)^2 - (z-m_1)^2}{2\sigma^2}$, $m_v=(1-2\hat x_i)+(1-2v)$.',
        fontsize=10)
    y -= 0.06
    ax.text(0.06, y, 'Whitened-SCT baseline (stronger; running in our work, not in the paper)', fontsize=13, weight='bold')
    y -= 0.04
    ax.text(0.10, y, r'Pre-filter $z^\prime_i = z_i - \alpha\,z_{i-1}$. Then', fontsize=11)
    y -= 0.04
    ax.text(0.10, y,
        r"$z^\prime_i = [(1-2X_i) - \alpha(1-2X_{i-1})] + [(1-2Y_i) - \alpha(1-2Y_{i-1})] + W_i$",
        fontsize=11)
    y -= 0.04
    ax.text(0.10, y,
        r'with $W_i\!\sim\!\mathcal{N}(0,\sigma^2(1-\alpha^2))$ i.i.d. — exactly ISI-MAC with $h=-\alpha$ and variance $\sigma^2(1-\alpha^2)$.',
        fontsize=11)
    y -= 0.04
    ax.text(0.10, y, r'Joint trellis SC on $z^\prime$ applies with $|\mathcal{S}|=4$. This is the proper analytical baseline.', fontsize=11)
    y -= 0.07
    ax.text(0.06, y, 'Neural decoder (NPD)', fontsize=13, weight='bold')
    y -= 0.04
    ax.text(0.10, y,
        r'Same chained corner-rate NPD as ISI: BiGRU over $z_1^N$ implicitly learns noise predictability.',
        fontsize=11)
    y -= 0.035
    ax.text(0.10, y,
        r'Trained at rate-1; MI-based frozen design via fast-CE on synthesized channels $W_N^{(i)}$.',
        fontsize=11)
    y -= 0.07
    ax.text(0.06, y, 'Symmetric capacity', fontsize=13, weight='bold')
    y -= 0.04
    ax.text(0.10, y,
        r'$I(X,Y;Z)$ — computed via Monte-Carlo on rate-1 codewords. Class C uses corner-rate split.',
        fontsize=11)
    y -= 0.04
    ax.text(0.10, y, r'At $\alpha=0.5$, the effective SNR after whitening is $10\log_{10}(1/[\sigma^2(1-\alpha^2)]) \approx 7.25$ dB.', fontsize=11)
    pdf.savefig(fig); plt.close(fig)

def page_table(pdf, channel_name, header, N_vals, rows):
    """rows = [('NPD', bler_list, errs_list, cw_list), ('SCT', ...)] each a tuple of 4."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    y = 0.94
    ax.text(0.5, y, header, ha='center', fontsize=17, weight='bold')
    y -= 0.04
    ax.text(0.5, y, channel_name, ha='center', fontsize=12, style='italic', color='#555')
    y -= 0.06
    # build table
    col_labels = ['N', 'NPD BLER', 'NPD errs/CW', 'analytical BLER', 'analytical errs/CW', 'NPD/anal']
    cells = []
    npd_b, npd_e, npd_n = rows[0]
    sct_b, sct_e, sct_n = rows[1]
    for i, N in enumerate(N_vals):
        ratio = npd_b[i] / sct_b[i] if sct_b[i] > 0 else float('inf')
        cells.append([
            f'{N}',
            f'{npd_b[i]:.3e}',
            f'{npd_e[i]:,}/{npd_n[i]:,}',
            f'{sct_b[i]:.3e}',
            f'{sct_e[i]:,}/{sct_n[i]:,}',
            f'{ratio:.2f}',
        ])
    table = ax.table(cellText=cells, colLabels=col_labels, loc='upper center', cellLoc='center',
                    bbox=[0.04, 0.45, 0.92, 0.40])
    table.auto_set_font_size(False); table.set_fontsize(11)
    table.scale(1, 1.5)
    # header bold
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#d6e4ff'); table[(0, i)].set_text_props(weight='bold')
    # color ratios
    for i, N in enumerate(N_vals):
        ratio = npd_b[i] / sct_b[i] if sct_b[i] > 0 else float('inf')
        c = '#cce5cc' if ratio < 1.0 else ('#f8d7da' if ratio > 1.1 else '#ffffe0')
        table[(i+1, 5)].set_facecolor(c)
    y = 0.40
    ax.text(0.06, y, 'Notes', fontsize=12, weight='bold')
    y -= 0.03
    ax.text(0.06, y, f'• NPD: BiGRU z-encoder + neural CheckNode/BitNode tree. d=16 hidden=100 unless retrained larger at small N.', fontsize=10)
    y -= 0.025
    ax.text(0.06, y, '• Analytical: ' + ('joint-trellis SC (SCT) on $\\mathcal{S}\\!=\\!4$ MAC lattice, MC design at 3 dB.' if 'ISI' in header else 'memoryless SC ignoring noise correlation. Whitened-SCT (stronger) is running.'),
        fontsize=10)
    y -= 0.025
    ax.text(0.06, y, '• "NPD/anal" ratio < 1 ⇒ NPD better (green); > 1.1 ⇒ analytical better (red).', fontsize=10)
    y -= 0.025
    ax.text(0.06, y, '• All BLERs are chained corner-rate (Class C) at target rate. SNR = 6 dB.', fontsize=10)
    y -= 0.025
    ax.text(0.06, y, '• Most points have ≥30 block errors → tight 95% Poisson CIs.', fontsize=10)
    pdf.savefig(fig); plt.close(fig)

def page_plot(pdf, channel_name, header, N_vals, npd, npd_e, npd_n, sct, sct_e, sct_n, sct_label='SCT decoder'):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, header, ha='center', fontsize=17, weight='bold')
    fig.text(0.5, 0.91, channel_name, ha='center', fontsize=12, style='italic', color='#555')
    ax = fig.add_axes([0.10, 0.10, 0.85, 0.75])
    def ci(k, n):
        if k == 0: return (1e-7, 3.0/n)
        return (max(1e-7, (k - 1.96*np.sqrt(k))/n), (k + 1.96*np.sqrt(k))/n)
    npd_ci = [ci(k, n) for k, n in zip(npd_e, npd_n)]
    sct_ci = [ci(k, n) for k, n in zip(sct_e, sct_n)]
    ax.semilogy(N_vals, npd, 'o-', lw=2.4, ms=12, color='#d62728', label='NPD (NN)', zorder=5)
    ax.semilogy(N_vals, sct, 's-', lw=2.4, ms=11, color='#1f77b4', label=sct_label, zorder=5)
    ax.fill_between(N_vals, [c[0] for c in npd_ci], [c[1] for c in npd_ci], color='#d62728', alpha=0.15)
    ax.fill_between(N_vals, [c[0] for c in sct_ci], [c[1] for c in sct_ci], color='#1f77b4', alpha=0.15)
    for x, y, k, n in zip(N_vals, npd, npd_e, npd_n):
        ax.annotate(f"{k}/{n//1000}K" if k>0 else f"0/{n//1000}K",
                    (x, y), textcoords='offset points', xytext=(10, 8), fontsize=9, color='#d62728')
    for x, y, k, n in zip(N_vals, sct, sct_e, sct_n):
        ax.annotate(f"{k}/{n//1000}K" if k>0 else f"0/{n//1000}K",
                    (x, y), textcoords='offset points', xytext=(-40, -18), fontsize=9, color='#1f77b4')
    ax.set_xscale('log', base=2); ax.set_xticks(N_vals); ax.set_xticklabels([str(x) for x in N_vals])
    ax.set_xlabel('Block length N', fontsize=13); ax.set_ylabel('BLER', fontsize=13)
    ax.grid(True, which='both', alpha=0.3); ax.legend(loc='lower left', fontsize=12)
    ax.set_ylim(1e-6, 0.5)
    pdf.savefig(fig); plt.close(fig)

if __name__ == "__main__":
    out = '/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/results_local/summary.pdf'
    with PdfPages(out) as pdf:
        page_math_isi(pdf)
        page_table(pdf,
                   'h = 0.3, SNR = 6 dB. Chained Class C corner-rate. Each decoder uses its own MC-derived frozen design.',
                   'ISI-MAC',
                   N_isi,
                   [(nn_isi, nn_isi_e, nn_isi_n), (sct_isi, sct_isi_e, sct_isi_n)])
        page_plot(pdf, 'h = 0.3, SNR = 6 dB', 'ISI-MAC',
                  N_isi, nn_isi, nn_isi_e, nn_isi_n, sct_isi, sct_isi_e, sct_isi_n,
                  sct_label='SCT decoder')
        page_math_maagn(pdf)
        page_table(pdf,
                   'α = 0.5, SNR = 6 dB. Chained Class C corner-rate.',
                   'MA-AGN-MAC',
                   N_ma,
                   [(nn_ma, nn_ma_e, nn_ma_n), (sc_ma, sc_ma_e, sc_ma_n)])
        page_plot(pdf, 'α = 0.5, SNR = 6 dB', 'MA-AGN-MAC',
                  N_ma, nn_ma, nn_ma_e, nn_ma_n, sc_ma, sc_ma_e, sc_ma_n,
                  sct_label='memoryless SC (analytical)')
    print(f"Saved {out}")
