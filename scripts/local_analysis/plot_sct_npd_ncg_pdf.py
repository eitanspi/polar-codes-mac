"""SCT + NPD + NCG plot and PDF (one page).

NCG numbers: from cluster v4 phase 1 (N=128 ckpt truncated to all N at 30K CW)
plus the freshly-trained iter15000 N=512 ckpt eval (5K CW, will be tightened).
"""
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_HERE = os.path.dirname(os.path.abspath(__file__))

# (BLER, errs, n_cw)  ISI-MAC h=0.3, SNR=6 dB, Class C corner-rate
chained_sct = {
    16:   (0.1500,    4500,  30000),
    32:   (0.05260,   1578,  30000),
    64:   (0.02853,    856,  30000),
    128:  (0.00590,    177,  30000),
    256:  (0.00132,     66,  50000),   # topup 30errors
    512:  (0.000338,    27,  80000),   # topup 30errors
    1024: (4.0e-5,       2,  50000),   # 200K design (SCT N=1024 extra eval still running)
}
npd = {
    16:   (0.16472,  16472, 100000),
    32:   (0.06873,   6873, 100000),
    64:   (0.03284,   1642,  50000),
    128:  (0.01270,    637,  50000),
    256:  (0.00138,     69,  50000),
    512:  (0.000307,    92, 300000),
    1024: (3.3e-5,      20, 600000),
}
# NCG: cluster v4 phase 1 (N=128 ckpt truncated), plus trained N=512 iter15000
ncg = {
    16:   (0.16817,   5045, 30000),
    32:   (0.07140,   2142, 30000),
    64:   (0.02793,    838, 30000),
    128:  (0.00747,    224, 30000),
    256:  (0.00215,     43, 20000),
    512:  (0.001167,    35, 30000),   # topup 30errors (trained iter15000)
    1024: (0.001,        1,  1000),   # iter4000 trained from N=512 ckpt (very noisy)
}


def main():
    out_pdf = os.path.join(_HERE, "sct_npd_ncg_isi_mac.pdf")
    fig = plt.figure(figsize=(8.5, 11))

    ax = fig.add_axes([0.10, 0.45, 0.82, 0.46])
    Ns = sorted(chained_sct.keys())
    ax.plot(Ns, [chained_sct[n][0] for n in Ns],
            's-', color='#1f77b4', linewidth=2, markersize=9,
            label='SCT (4-state chained, analytical)')
    ax.plot(Ns, [npd[n][0] for n in Ns],
            '^-', color='#2ca02c', linewidth=2, markersize=9, label='NPD')
    ax.plot(Ns, [ncg[n][0] for n in Ns],
            'D-', color='#9467bd', linewidth=2, markersize=8, label='NCG')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('Block length N', fontsize=11)
    ax.set_ylabel('BLER', fontsize=11)
    ax.set_title('ISI-MAC corner-rate: SCT vs NPD vs NCG\n'
                 'h=0.3, SNR=6 dB, Class C',
                 fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.92)
    ax.set_ylim(1e-5, 0.5)

    # Table
    ax_tbl = fig.add_axes([0.04, 0.04, 0.94, 0.34])
    ax_tbl.axis('off')
    col_labels = ['N', 'SCT BLER', 'SCT errs/CW', 'NPD BLER', 'NPD errs/CW',
                  'NCG BLER', 'NCG errs/CW']
    rows = []
    for N in Ns:
        s_bler, s_err, s_cw = chained_sct[N]
        n_bler, n_err, n_cw = npd[N]
        g_bler, g_err, g_cw = ncg[N]
        rows.append([
            str(N),
            f"{s_bler:.5g}", f"{s_err:,} / {s_cw:,}",
            f"{n_bler:.5g}", f"{n_err:,} / {n_cw:,}",
            f"{g_bler:.5g}", f"{g_err:,} / {g_cw:,}",
        ])
    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc='center',
                      cellLoc='center', colLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_text_props(fontweight='bold')
        tbl[(0, j)].set_facecolor('#dde6f0')

    ax_tbl.set_title('Per-N numerical detail (errors / codewords)',
                     fontsize=11, pad=12)
    fig.text(0.5, 0.01,
             'NCG numbers: N≤128 from native N=128 trained ckpt (truncated to smaller N), N=256 truncated from N=128, '
             'N=512 from iter15000 trained ckpt (warm-started from N=128), N=1024 from iter4000 trained ckpt '
             '(warm-started from N=512; training continues, only 1 error in 1K CW — very noisy).',
             ha='center', fontsize=8, style='italic', wrap=True)

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == '__main__':
    main()
