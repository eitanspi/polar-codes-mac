"""Build all results PDFs into ./results_pdfs/.

Each PDF: plot on top, minimal table below. Single page. Minimal labels:
just decoder names (NCG, NPD, SCT), no run-history annotations.
"""
import os, json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))


def save_pdf_plot_table(path, plot_fn, table_data, table_cols, title=None, table_title=None):
    fig = plt.figure(figsize=(8.5, 11))
    ax_plot = fig.add_axes([0.10, 0.45, 0.82, 0.46])
    plot_fn(ax_plot)
    if title:
        ax_plot.set_title(title, fontsize=12)

    ax_tbl = fig.add_axes([0.04, 0.04, 0.94, 0.34])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=table_data, colLabels=table_cols,
                       loc="center", cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.4)
    for j in range(len(table_cols)):
        tbl[(0, j)].set_text_props(fontweight="bold")
        tbl[(0, j)].set_facecolor("#dde6f0")
    if table_title:
        ax_tbl.set_title(table_title, fontsize=11, pad=10)

    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ============================================================================
# 1. ISI-MAC corner-rate BLER vs N (NPD, NCG, SCT)
# ============================================================================
def build_isi_bler_vs_n():
    sct = {
        16:   (0.1500,    4500,  30000),
        32:   (0.05260,   1578,  30000),
        64:   (0.02853,    856,  30000),
        128:  (0.00590,    177,  30000),
        256:  (0.00132,     66,  50000),
        512:  (0.000338,    27,  80000),
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
    ncg = {
        16:   (0.16817,   5045, 30000),
        32:   (0.07140,   2142, 30000),
        64:   (0.02793,    838, 30000),
        128:  (0.00747,    224, 30000),
        256:  (0.00215,     43, 20000),
        512:  (0.001167,    35, 30000),
        1024: (0.001,        1,  1000),
    }
    Ns = sorted(sct.keys())

    def plot(ax):
        ax.plot(Ns, [sct[n][0] for n in Ns], 's-', color='#1f77b4',
                linewidth=2, markersize=9, label='SCT')
        ax.plot(Ns, [npd[n][0] for n in Ns], '^-', color='#2ca02c',
                linewidth=2, markersize=9, label='NPD')
        ax.plot(Ns, [ncg[n][0] for n in Ns], 'D-', color='#9467bd',
                linewidth=2, markersize=8, label='NCG')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel('N', fontsize=11); ax.set_ylabel('BLER', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left', fontsize=11)
        ax.set_ylim(1e-5, 0.5)

    cols = ['N', 'SCT', 'errs/CW', 'NPD', 'errs/CW', 'NCG', 'errs/CW']
    rows = []
    for n in Ns:
        rows.append([str(n),
                     f"{sct[n][0]:.5g}", f"{sct[n][1]:,}/{sct[n][2]:,}",
                     f"{npd[n][0]:.5g}", f"{npd[n][1]:,}/{npd[n][2]:,}",
                     f"{ncg[n][0]:.5g}", f"{ncg[n][1]:,}/{ncg[n][2]:,}"])
    save_pdf_plot_table(
        os.path.join(_HERE, "01_isi_mac_bler_vs_n.pdf"),
        plot, rows, cols,
        title="ISI-MAC corner-rate BLER vs N\nh=0.3, SNR=6 dB",
        table_title="errs / codewords")


# ============================================================================
# 2. MA-AGN corner-rate BLER vs N (NPD, SCT)
# ============================================================================
def build_maagn_bler_vs_n():
    # NPD numbers from RESULTS.md (α=0.5)
    npd = {
        16:   (0.14654,   7327,  50000),
        32:   (0.09925,   9925, 100000),
        64:   (0.02650,   1325,  50000),
        128:  (0.00618,    309,  50000),
        256:  (0.00068,     34,  50000),
        512:  (6.5e-5,      31, 600000),
        1024: (3.0e-5,      30, 1050000),
    }
    # whitened-SCT (from overnight v2 E2 and overnight v4 L4); α=0.5
    sct = {
        128:  (0.00500,     50, 10000),
        256:  (0.00130,     13, 10000),
        512:  (0.00180,      9,  5000),
        1024: (0.00300,     15,  5000),
    }
    Ns_npd = sorted(npd.keys()); Ns_sct = sorted(sct.keys())

    def plot(ax):
        ax.plot(Ns_npd, [npd[n][0] for n in Ns_npd], '^-', color='#2ca02c',
                linewidth=2, markersize=9, label='NPD')
        ax.plot(Ns_sct, [sct[n][0] for n in Ns_sct], 's-', color='#1f77b4',
                linewidth=2, markersize=9, label='SCT (whitened)')
        ax.set_xscale('log', base=2); ax.set_yscale('log')
        ax.set_xticks(Ns_npd); ax.set_xticklabels([str(n) for n in Ns_npd])
        ax.set_xlabel('N', fontsize=11); ax.set_ylabel('BLER', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left', fontsize=11)
        ax.set_ylim(1e-5, 0.5)

    cols = ['N', 'SCT', 'errs/CW', 'NPD', 'errs/CW']
    rows = []
    for n in Ns_npd:
        s = sct.get(n)
        s_bler = f"{s[0]:.5g}" if s else "—"
        s_cw = f"{s[1]:,}/{s[2]:,}" if s else "—"
        rows.append([str(n), s_bler, s_cw,
                     f"{npd[n][0]:.5g}", f"{npd[n][1]:,}/{npd[n][2]:,}"])
    save_pdf_plot_table(
        os.path.join(_HERE, "02_maagn_bler_vs_n.pdf"),
        plot, rows, cols,
        title="MA-AGN corner-rate BLER vs N\nα=0.5, SNR=6 dB",
        table_title="errs / codewords")


# ============================================================================
# 3. ISI-MAC SNR sweep — SCT at N=128, 256, 512
# ============================================================================
def build_isi_snr_sweep():
    # From overnight v2 F2 (N=128), v3 H3 (N=256), v4 K4 (N=512)
    data = {}
    for fn, N_key in [("overnight2_F2_snr_sweep.json", "N=128"),
                       ("overnight3_H3_snr_sweep_n256.json", "N=256"),
                       ("overnight4_K4_snr_n512.json", "N=512")]:
        path = os.path.join(_ROOT, "scripts", "local_analysis", fn)
        d = json.load(open(path))
        data[N_key] = {}
        for snr, r in d["results"].items():
            e = r["eval"]
            data[N_key][float(snr)] = (e["bler"], e["errs"], e["n_cw"])

    def plot(ax):
        colors = {'N=128': '#1f77b4', 'N=256': '#2ca02c', 'N=512': '#d62728'}
        markers = {'N=128': 's', 'N=256': '^', 'N=512': 'D'}
        for N_key, snr_data in data.items():
            snrs = sorted(snr_data.keys())
            blers = [snr_data[s][0] for s in snrs]
            ax.plot(snrs, blers, marker=markers[N_key], color=colors[N_key],
                    linewidth=2, markersize=9, label=N_key)
        ax.set_yscale('log')
        ax.set_xlabel('SNR (dB)', fontsize=11)
        ax.set_ylabel('BLER', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)

    # Table
    snr_vals = sorted(set().union(*(d.keys() for d in data.values())))
    cols = ['SNR (dB)'] + list(data.keys())
    rows = []
    for s in snr_vals:
        row = [f"{s:g}"]
        for N_key in data.keys():
            v = data[N_key].get(s)
            if v: row.append(f"{v[0]:.5g} ({v[1]}/{v[2]})")
            else: row.append("—")
        rows.append(row)
    save_pdf_plot_table(
        os.path.join(_HERE, "03_isi_mac_snr_sweep.pdf"),
        plot, rows, cols,
        title="ISI-MAC SCT BLER vs SNR\nh=0.3, Class C corner-rate",
        table_title="BLER (errs/CW)")


# ============================================================================
# 4. ISI-MAC h sweep — SCT at N=128, SNR=6 dB
# ============================================================================
def build_isi_h_sweep():
    path = os.path.join(_ROOT, "scripts", "local_analysis", "overnight4_M4_h_sweep.json")
    d = json.load(open(path))
    data = {}
    for h, r in d["results"].items():
        e = r["eval"]
        data[float(h)] = (e["bler"], e["errs"], e["n_cw"])

    def plot(ax):
        hs = sorted(data.keys())
        blers = [data[h][0] for h in hs]
        ax.plot(hs, blers, 's-', color='#1f77b4', linewidth=2, markersize=10)
        ax.set_yscale('log')
        ax.set_xlabel('h (ISI tap)', fontsize=11)
        ax.set_ylabel('BLER', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)

    cols = ['h', 'BLER', 'errs/CW']
    rows = [[f"{h:g}", f"{data[h][0]:.5g}", f"{data[h][1]}/{data[h][2]}"]
            for h in sorted(data.keys())]
    save_pdf_plot_table(
        os.path.join(_HERE, "04_isi_mac_h_sweep.pdf"),
        plot, rows, cols,
        title="ISI-MAC SCT BLER vs h\nN=128, SNR=6 dB, corner-rate",
        table_title="errs / codewords")


# ============================================================================
# 5. MA-AGN α sweep — SCT at N=128, 256
# ============================================================================
def build_maagn_alpha_sweep():
    path = os.path.join(_ROOT, "scripts", "local_analysis", "overnight3_G3_maagn_alpha.json")
    d = json.load(open(path))
    # Also include α=0.5 from overnight v2 E2
    path_e2 = os.path.join(_ROOT, "scripts", "local_analysis", "overnight2_E2_maagn_whitened.json")
    d_e2 = json.load(open(path_e2))

    data = {}  # data[alpha][N] = (bler, errs, n_cw)
    for alpha, recs in d["results"].items():
        a = float(alpha)
        data[a] = {}
        for N, r in recs["by_N"].items():
            e = r["eval"]
            data[a][int(N)] = (e["bler"], e["errs"], e["n_cw"])
    # Add α=0.5
    data[0.5] = {}
    for N, r in d_e2["results"].items():
        e = r["eval"]
        data[0.5][int(N)] = (e["bler"], e["errs"], e["n_cw"])

    def plot(ax):
        alphas = sorted(data.keys())
        Ns = sorted(set().union(*(data[a].keys() for a in alphas)))
        markers = ['s', '^', 'D', 'o']
        for i, N in enumerate(Ns):
            xs = [a for a in alphas if N in data[a]]
            ys = [data[a][N][0] for a in xs]
            ax.plot(xs, ys, marker=markers[i % 4], linewidth=2,
                    markersize=9, label=f"N={N}")
        ax.set_yscale('log')
        ax.set_xlabel('α (AR-1 coefficient)', fontsize=11)
        ax.set_ylabel('BLER', fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)

    alphas = sorted(data.keys())
    Ns = sorted(set().union(*(data[a].keys() for a in alphas)))
    cols = ['α'] + [f"N={N}" for N in Ns]
    rows = []
    for a in alphas:
        row = [f"{a:g}"]
        for N in Ns:
            v = data[a].get(N)
            if v: row.append(f"{v[0]:.5g} ({v[1]}/{v[2]})")
            else: row.append("—")
        rows.append(row)
    save_pdf_plot_table(
        os.path.join(_HERE, "05_maagn_alpha_sweep.pdf"),
        plot, rows, cols,
        title="MA-AGN SCT BLER vs α\nSNR=6 dB, corner-rate",
        table_title="BLER (errs/CW)")


# ============================================================================
# 6. Capacity regions — pentagon for ISI-MAC, GMAC, MA-AGN, ABN-MAC
# ============================================================================
def build_capacity_pentagons():
    # Values from session 11 memory + capacity_region_*.json
    regions = {
        'ISI-MAC h=0.3': (0.922, 0.467, 1.389, '#1f77b4'),
        'GMAC':          (0.912, 0.464, 1.376, '#2ca02c'),
        'MA-AGN α=0.5':  (0.977, 0.487, 1.464, '#d62728'),
        'ABN-MAC p=0.2': (0.636, 0.358, 0.994, '#9467bd'),
    }

    def plot(ax):
        for name, (Imax, Imin, Isum, color) in regions.items():
            # Pentagon corners for symmetric MAC:
            # (0, Imax), (Isum-Imax, Imax), (Imax, Isum-Imax), (Imax, 0), (0, 0) — closing
            corners = [(0, 0), (Imax, 0), (Imax, Isum - Imax),
                       (Isum - Imax, Imax), (0, Imax), (0, 0)]
            xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
            ax.plot(xs, ys, '-', color=color, linewidth=2, label=name)
            ax.fill(xs, ys, color=color, alpha=0.12)
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, linewidth=1)
        ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
        ax.set_xlabel(r'$R_U$ (bits/use)', fontsize=11)
        ax.set_ylabel(r'$R_V$ (bits/use)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal')

    cols = ['Channel', 'I(X;Z|Y)', 'I(X;Z)', 'I(X,Y;Z)', 'Eq. rate / use']
    rows = []
    for name, (Imax, Imin, Isum, _) in regions.items():
        rows.append([name, f"{Imax:.3f}", f"{Imin:.3f}", f"{Isum:.3f}", f"{Isum/2:.3f}"])
    save_pdf_plot_table(
        os.path.join(_HERE, "06_capacity_pentagons.pdf"),
        plot, rows, cols,
        title="Capacity regions (BPSK, SNR=6 dB)",
        table_title="Mutual information (bits/use)")


# ============================================================================
def main():
    build_isi_bler_vs_n()
    build_maagn_bler_vs_n()
    build_isi_snr_sweep()
    build_isi_h_sweep()
    build_maagn_alpha_sweep()
    build_capacity_pentagons()
    print("\nAll PDFs in:", _HERE)


if __name__ == "__main__":
    main()
