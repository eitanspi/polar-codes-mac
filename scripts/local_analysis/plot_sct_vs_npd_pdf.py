"""Single PDF page: SCT vs NPD plot + numerical table."""
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
    256:  (0.00130,     13,  10000),
    512:  (0.000433,    13,  30000),
    1024: (4.0e-5,       2,  50000),
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


def main():
    out_pdf = os.path.join(_HERE, "sct_vs_npd_isi_mac.pdf")

    fig = plt.figure(figsize=(8.5, 11))  # letter portrait
    # --- plot on top ---
    ax = fig.add_axes([0.10, 0.45, 0.82, 0.46])
    Ns = sorted(chained_sct.keys())
    ax.plot(Ns, [chained_sct[n][0] for n in Ns],
            's-', color='#1f77b4', linewidth=2, markersize=9,
            label='SCT (4-state chained, analytical)')
    ax.plot(Ns, [npd[n][0] for n in Ns],
            '^-', color='#2ca02c', linewidth=2, markersize=9, label='NPD')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('Block length N', fontsize=11)
    ax.set_ylabel('BLER', fontsize=11)
    ax.set_title('ISI-MAC corner-rate: analytical SCT vs NPD\n'
                 'h=0.3, SNR=6 dB, Class C',
                 fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.92)
    ax.set_ylim(1e-5, 0.5)

    # --- table at bottom ---
    ax_tbl = fig.add_axes([0.05, 0.05, 0.92, 0.32])
    ax_tbl.axis('off')

    col_labels = ['N', 'SCT BLER', 'SCT errs/CW', 'NPD BLER', 'NPD errs/CW',
                  'SCT/NPD']
    rows = []
    for N in Ns:
        s_bler, s_err, s_cw = chained_sct[N]
        n_bler, n_err, n_cw = npd[N]
        s_str = f"{s_bler:.5g}"
        n_str = f"{n_bler:.5g}"
        ratio = s_bler / n_bler
        rows.append([
            str(N),
            s_str,
            f"{s_err:,} / {s_cw:,}",
            n_str,
            f"{n_err:,} / {n_cw:,}",
            f"{ratio:.2f}",
        ])

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc='center',
                      cellLoc='center', colLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.45)
    # Bold header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_text_props(fontweight='bold')
        tbl[(0, j)].set_facecolor('#dde6f0')
    # Color SCT/NPD ratio cell
    for i, N in enumerate(Ns, start=1):
        ratio = chained_sct[N][0] / npd[N][0]
        if ratio < 1: c = '#d8f0d8'   # SCT better
        elif ratio < 1.5: c = '#f4f4d8'
        else: c = '#f0d8d8'           # NPD better
        tbl[(i, 5)].set_facecolor(c)

    ax_tbl.set_title('Per-N numerical detail (errors / codewords)',
                     fontsize=11, pad=12)

    # Footer
    fig.text(0.5, 0.015,
             'Both decoders use Class C corner-rate path (decode U first, then V|X̂). '
             'SCT = 4-state (X_{t-1},Y_{t-1}) FB → marginalize Y per position → scalar U-LLR → Arikan SC; '
             'stage 2 conditions on the decoded X̂. SCT uses MC-genie own design (50K-200K trials).',
             ha='center', fontsize=8, style='italic', wrap=True)

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == '__main__':
    main()
