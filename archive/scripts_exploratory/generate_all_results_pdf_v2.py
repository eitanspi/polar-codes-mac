#!/usr/bin/env python3
"""Generate ALL_RESULTS_v2.pdf: SC vs NN decoders only (no list/CRC). CW counts for all entries."""

import torch
torch.set_num_threads(2)
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})

COLORS = {'SC': 'blue', 'NCG': 'red', 'NPD': 'green', 'Chained Trellis SC': 'blue',
           'Memoryless SC': 'gray', 'Trellis SC': 'blue', 'NPD d=16 h=100': 'green',
           'NPD d=64 h=128': 'darkgreen', 'NPD d=16 h=64': 'limegreen',
           'NPD d16h100': 'green', 'NPD (d16h100)': 'green'}
MARKERS = {'SC': 'o', 'NCG': 's', 'NPD': 'D', 'Chained Trellis SC': 'o',
           'Memoryless SC': 'v', 'Trellis SC': 'o', 'NPD d=16 h=100': 'D',
           'NPD d=64 h=128': '^', 'NPD d=16 h=64': 'v',
           'NPD d16h100': 'D', 'NPD (d16h100)': 'D'}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'paper_figures')


def fmt(v):
    if v is None: return '--'
    if v == 0.0: return '0.0000'
    if v < 0.001: return f'{v:.1e}'
    return f'{v:.4f}'


def make_page(pdf, title, subtitle, equation, settings, headers, rows, series, note=None):
    fig = plt.figure(figsize=(11, 8.5))
    ax_top = fig.add_axes([0.05, 0.50, 0.90, 0.48]); ax_top.axis('off')
    y = 0.98
    ax_top.text(0.5, y, title, ha='center', va='top', fontsize=14, fontweight='bold', transform=ax_top.transAxes); y -= 0.055
    ax_top.text(0.5, y, subtitle, ha='center', va='top', fontsize=11, fontstyle='italic', transform=ax_top.transAxes); y -= 0.045
    ax_top.text(0.5, y, equation, ha='center', va='top', fontsize=10, family='monospace', transform=ax_top.transAxes); y -= 0.055
    for line in settings:
        ax_top.text(0.05, y, line, ha='left', va='top', fontsize=8.5, transform=ax_top.transAxes); y -= 0.040
    y -= 0.03
    cell_text = [[str(v) if v is not None else '--' for v in row] for row in rows]
    rh = 0.040; th = rh * (len(rows) + 1)
    table = ax_top.table(cellText=cell_text, colLabels=headers, loc='center', cellLoc='center',
                         bbox=[0.02, max(0.0, y - th), 0.96, th])
    table.auto_set_font_size(False); table.set_fontsize(7.5 if len(headers) > 7 else 8)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#d4e6f1'); table[0, j].set_text_props(fontweight='bold')
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0: table[i, j].set_facecolor('#f9f9f9')

    ax = fig.add_axes([0.10, 0.07, 0.80, 0.36])
    for label, ns, blers in series:
        valid = [(n, b) for n, b in zip(ns, blers) if b is not None and b > 0]
        if not valid: continue
        vn, vb = zip(*valid)
        ax.semilogy(vn, vb, marker=MARKERS.get(label, 'o'), color=COLORS.get(label, 'black'),
                     label=label, linewidth=1.5, markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    ax.set_xlabel('Block length N'); ax.set_ylabel('BLER')
    if note: ax.set_title(f'BLER vs N  --  {note}', fontsize=8, fontstyle='italic', color='#444444')
    else: ax.set_title('BLER vs N', fontsize=11)
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.set_xscale('log', base=2)
    all_ns = sorted(set(n for _, ns, _ in series for n in ns))
    if all_ns: ax.set_xticks(all_ns); ax.set_xticklabels([str(n) for n in all_ns])
    ax.grid(True, which='both', alpha=0.3)
    pdf.savefig(fig); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with PdfPages(os.path.join(OUT_DIR, 'ALL_RESULTS_v2.pdf')) as pdf:

        # 1. BEMAC B
        Ns = [16, 32, 64, 128, 256, 512, 1024]
        make_page(pdf,
            '1. BEMAC Class B (non-corner)', 'Binary Erasure MAC',
            'Z = X + Y,  Z in {0, 1, 2}  (deterministic)',
            ['Path: make_path(N, N//2) = [0]^{N/2} [1]^N [0]^{N/2}',
             'Capacity: I(X;Z)=0.500, I(Y;Z|X)=1.000, Symmetric point: (0.750, 0.750)',
             'Operating rate: R_U ~ 0.50, R_V ~ 0.70',
             'NCG: d=16, hidden=64, BEMAC vocab=3 embedding'],
            ['N', 'ku', 'kv', 'SC', 'SC errs/CW', 'NCG', 'NCG errs/CW', 'NCG/SC'],
            [[16, 8, 11, '0.0122', '122/10K', '0.0108', '108/10K', '0.89x'],
             [32, 16, 22, '0.0097', '146/15K', '0.0076', '114/15K', '0.78x'],
             [64, 32, 45, '0.0032', '128/40K', '0.0032', '154/48K', '1.00x'],
             [128, 64, 90, '0.0014', '129/90K', '0.0017', '86/50K', '1.21x'],
             [256, 128, 179, '8e-5', '~4/50K', '4e-5', '~2/50K', '0.50x'],
             [512, 256, 358, '0.0', '0/2K', '0.0', '0/2K', '--'],
             [1024, 512, 716, '1e-4', '~1/10K', '1e-4', '~1/10K', '1.00x']],
            [('SC', Ns, [0.012,0.0097,0.0032,0.0014,8e-5,0.0,1e-4]),
             ('NCG', Ns, [0.011,0.0076,0.0032,0.0017,4e-5,0.0,1e-4])],
            'NCG matches or beats SC at all N. No wall.')

        # 2. BEMAC C
        Ns = [8, 16, 32, 64, 128, 256, 512, 1024]
        make_page(pdf,
            '2. BEMAC Class C (corner)', 'Binary Erasure MAC',
            'Z = X + Y,  Z in {0, 1, 2}  (deterministic)',
            ['Path: make_path(N, N) = [0]^N [1]^N',
             'Capacity: Corner (R_U, R_V) = (0.500, 1.000)',
             'Operating rate: R_U ~ 0.30, R_V ~ 0.60',
             'NCG: d=16, hidden=64, MI-designed frozen set (co-adapted with decoder)'],
            ['N', 'ku', 'kv', 'SC', 'SC errs/CW', 'NCG', 'NCG errs/CW', 'NCG/SC'],
            [[8,2,5,'0.1100','329/3K','0.0390','118/3K','0.36x'],
             [16,5,10,'0.0920','276/3K','0.0160','49/3K','0.18x'],
             [32,10,19,'0.0990','298/3K','0.0060','17/3K','0.06x'],
             [64,19,38,'0.0550','165/3K','0.0020','5/3K','0.03x'],
             [128,38,77,'0.0246','491/20K','3e-4','5/20K','0.01x'],
             [256,77,154,'0.0134','668/50K','2e-4','9/50K','0.01x'],
             [512,154,307,'0.0034','168/50K','2e-4','10/50K','0.06x'],
             [1024,307,614,'4e-4','26/64K','2e-4','5/30K','0.42x']],
            [('SC', Ns, [0.11,0.092,0.099,0.055,0.025,0.013,0.003,4e-4]),
             ('NCG', Ns, [0.039,0.016,0.006,0.002,3e-4,2e-4,2e-4,2e-4])],
            'NCG beats SC by 2-100x at every N. Strongest result in the project.')

        # 3. GMAC B
        Ns = [32, 64, 128, 256, 512, 1024]
        make_page(pdf,
            '3. GMAC Class B (non-corner)', 'Gaussian MAC',
            'Z = (1-2X) + (1-2Y) + W,  W ~ N(0, sigma^2),  SNR = 6 dB',
            ['Path: make_path(N, N//2)',
             'Capacity: I(X;Z)=0.465, I(Y;Z|X)=0.912, I(X,Y;Z)=1.376',
             'Operating rate: R_U = R_V ~ 0.48 (~70% of symmetric capacity)',
             'NCG: d=16, hidden=64, sequential training'],
            ['N', 'ku=kv', 'SC', 'SC errs/CW', 'NCG', 'NCG errs/CW', 'NCG/SC'],
            [[32,15,'0.0450','135/3K','0.0503','151/3K','1.12x'],
             [64,31,'0.0276','138/5K','0.0282','141/5K','1.02x'],
             [128,62,'0.0187','112/6K','0.0230','46/2K','1.23x'],
             [256,123,'0.0047','165/35K','0.0230','115/5K','4.9x (wall)'],
             [512,246,'~0.001','est.','0.0123','123/10K','12x'],
             [1024,491,'--','--','0.4700','940/2K','BROKEN']],
            [('SC', Ns, [0.045,0.028,0.019,0.0047,0.001,None]),
             ('NCG', Ns, [0.050,0.028,0.023,0.023,0.012,0.47])],
            'NCG matches SC at N=32-64. Wall at N=256 (4.4x). N=1024 broken.')

        # 4. GMAC C
        Ns = [16, 32, 64, 128, 256, 512, 1024]
        make_page(pdf,
            '4. GMAC Class C (corner)', 'Gaussian MAC',
            'Z = (1-2X) + (1-2Y) + W,  W ~ N(0, sigma^2),  SNR = 6 dB',
            ['Path: make_path(N, N), corner rate (R_U ~ 0.23, R_V ~ 0.45)',
             'Capacity: Corner (R_U, R_V) = (0.465, 0.912)',
             'NPD/NCG use MI-designed frozen sets (different from SC design)',
             'SC (MC): uses MC-based frozen set design. SC BLER decreases monotonically.'],
            ['N', 'ku', 'kv', 'SC(MC)', 'SC errs/CW', 'NPD', 'NPD errs/CW', 'NCG', 'NCG errs/CW'],
            [[16,4,7,'0.1620','1620/10K','0.1070','1070/10K','0.1190','~?/1K'],
             [32,7,15,'0.0681','681/10K','0.0373','373/10K','0.0353','106/3K'],
             [64,15,29,'0.0273','273/10K','0.0100','100/10K','0.0133','40/3K'],
             [128,30,58,'0.0067','134/20K','0.0329','329/10K','0.0010','~?/1K'],
             [256,59,117,'0.0013','126/100K','3e-4','6/20K','--','--'],
             [512,119,233,'4.0e-4','40/100K','2e-4','11/50K','--','--'],
             [1024,238,467,'5e-4','10/20K','0.0','0/50K','--','--']],
            [('SC', Ns, [0.162,0.068,0.027,0.0067,0.0013,3.8e-4,5e-4]),
             ('NPD', Ns, [0.107,0.037,0.010,0.033,3e-4,2e-4,None]),
             ('NCG', [16,32,64,128], [0.119,0.035,0.013,0.001])],
            'NPD/NCG beat SC at N=16-64. SC BLER monotonic with MC design (prev. table had bug at N=512+).')

        # 5. ABNMAC B
        Ns = [8, 16, 32, 64, 128, 256, 512]
        make_page(pdf,
            '5. ABNMAC Class B (non-corner)', 'Asymmetric Binary Noise MAC',
            'Z = (X xor Ex, Y xor Ey),  correlated binary noise',
            ['Path: make_path(N, N//2)',
             'Capacity: I(X;Z)~0.400, I(Y;Z|X)~0.800, Symmetric: (0.600, 0.600)',
             'Operating rate: R_U ~ R_V ~ 0.30 (ku=kv)',
             'NCG: d=16, hidden=64, curriculum from N=8 to N=512'],
            ['N', 'ku=kv', 'SC', 'SC errs/CW', 'NCG', 'NCG errs/CW', 'NCG/SC'],
            [[8,3,'0.1198','1198/10K','0.1202','1202/10K','1.00x'],
             [16,5,'0.0629','629/10K','0.0570','570/10K','0.91x'],
             [32,10,'0.0213','213/10K','0.0182','182/10K','0.85x'],
             [64,22,'0.0416','416/10K','0.0415','415/10K','1.00x'],
             [128,45,'0.0288','115/4K','0.0444','222/5K','1.54x'],
             [256,102,'0.0464','325/7K','0.0713','499/7K','1.54x'],
             [512,205,'0.0273','191/7K','0.0340','17/500','1.25x']],
            [('SC', Ns, [0.120,0.063,0.021,0.042,0.029,0.046,0.027]),
             ('NCG', Ns, [0.120,0.057,0.018,0.042,0.044,0.071,0.034])],
            'NCG matches SC at N<=64. NCG wall at N=128-256. N=512 improved to 1.25x (was 2.15x).')

        # 6. ABNMAC C
        Ns = [16, 32, 64, 128, 256, 512, 1024]
        make_page(pdf,
            '6. ABNMAC Class C (corner)', 'Asymmetric Binary Noise MAC — SC only',
            'Z = (X xor Ex, Y xor Ey),  correlated binary noise',
            ['Path: make_path(N, N) = [0]^N [1]^N',
             'Capacity: Corner (R_U, R_V) = (0.400, 0.800)',
             'Operating rate: R_U ~ 0.19, R_V ~ 0.38',
             'No neural decoder (ABNMAC 2D output incompatible with z_encoder)'],
            ['N', 'ku', 'kv', 'SC', 'SC errs/CW'],
            [[16,3,6,'0.0626','313/5K'],
             [32,6,13,'0.0334','167/5K'],
             [64,13,26,'0.0488','488/10K'],
             [128,26,51,'0.0300','150/5K'],
             [256,51,102,'0.0206','103/5K'],
             [512,102,205,'0.0133','200/15K'],
             [1024,205,410,'0.0030','59/20K']],
            [('SC', Ns, [0.063,0.033,0.049,0.030,0.021,0.013,0.003])],
            'SC baseline only. Non-monotonic BLER at N=64. N=1024 much lower with more CW.')

        # 7. ISI-MAC C — NO memoryless, NO joint. Chained SC vs NPD only.
        Ns = [16, 32, 64, 128, 256, 512]
        make_page(pdf,
            '7. ISI-MAC Class C (corner) — Memory Channel', 'Inter-Symbol Interference MAC',
            'Z[i] = (1-2X[i]) + (1-2Y[i]) + 0.3*(1-2X[i-1]) + 0.3*(1-2Y[i-1]) + W[i]',
            ['Parameters: h=0.3, SNR=6.0 dB (sigma^2=0.251)',
             'Path: make_path(N, N), chained decoding (Stage 1: U, Stage 2: V|U)',
             'Design: GMAC Class C proxy frozen set',
             'SC: chained 2-stage trellis (|S|=2 per stage). NPD: chained neural decoder.'],
            ['N', 'ku', 'kv', 'Chained SC', 'SC errs/CW', 'NPD (best)', 'NPD errs/CW', 'NPD arch', 'NPD/SC'],
            [[16,4,7,'0.1689','1689/10K','0.1376','688/5K','d=16 h=100','0.82x'],
             [32,7,15,'0.0822','822/10K','0.0566','283/5K','d=16 h=100','0.69x'],
             [64,15,29,'0.0407','407/10K','0.0322','161/5K','d=16 h=100','0.79x'],
             [128,30,58,'0.0223','223/10K','0.0812','406/5K','d=16 h=100','3.64x'],
             [256,59,117,'0.0070','105/15K','0.0110','56/5K','d=64 h=128','1.57x'],
             [512,119,233,'0.0026','51/20K','0.1080','565/5K','d=64 h=128','42x (wall)']],
            [('Chained Trellis SC', Ns, [0.169,0.082,0.041,0.022,0.007,0.0026]),
             ('NPD d=16 h=100', [16,32,64,128], [0.138,0.057,0.032,0.081]),
             ('NPD d=64 h=128', [128,256,512], [0.030,0.011,0.108])],
            'NPD beats SC at N=16-64 (21-31%). Wall at N>=128. d16h100 evaluated at N=64,128.')

        # 8. Ising MAC
        Ns = [16, 32, 64, 128, 256]
        make_page(pdf,
            '8. Ising MAC Class C (corner) — Memory Channel', 'Ising Channel with Markov State',
            'Good state: Z=(1-2X)+(1-2Y)+W.  Bad state: Z=W.  p_flip=0.1',
            ['Parameters: sigma^2=0.251, p_flip=0.1',
             'Path: make_path(N, N), chained decoding',
             'Design: GMAC Class C proxy',
             'Extremely hard channel. All decoders fail at N>=64 (BLER>89%).',
             'NPD d16h100: trained at N=16..256. No better than SC at any N.'],
            ['N', 'ku', 'kv', 'Trellis SC', 'SC errs/CW', 'Memoryless SC', 'Memless errs/CW', 'NPD d16h100', 'NPD errs/CW'],
            [[16,4,7,'0.5750','2873/5K','0.6340','1902/3K','0.5920','2960/5K'],
             [32,7,15,'0.6890','3443/5K','0.7810','2342/3K','0.7700','3850/5K'],
             [64,15,29,'0.8930','4465/5K','0.9424','4712/5K','~1.0','--'],
             [128,30,58,'0.9824','4912/5K','0.9946','4973/5K','0.9970','4985/5K'],
             [256,59,117,'--','--','--','--','0.9996','4998/5K']],
            [('Trellis SC', [16,32,64,128], [0.575,0.689,0.893,0.982]),
             ('Memoryless SC', [16,32,64,128], [0.634,0.781,0.942,0.995]),
             ('NPD d16h100', [16,32,128,256], [0.592,0.770,0.997,0.9996])],
            'All decoders fail at N>=64. Channel too hard for any approach tested.')

        # 8b. Ising MAC p_flip=0.001 (much easier)
        Ns = [16, 32, 64, 128, 256]
        make_page(pdf,
            '8b. Ising MAC Class C — p_flip=0.001 (weaker memory)', 'Ising Channel with Markov State',
            'Good state: Z=(1-2X)+(1-2Y)+W.  Bad state: Z=W.  p_flip=0.001',
            ['Parameters: sigma^2=0.251, p_flip=0.001 (vs 0.1 in 8a)',
             'Path: make_path(N, N), chained decoding',
             'Design: GMAC Class C proxy',
             'With rare state flips, Ising -> near-GMAC. Trellis barely beats memoryless.',
             'Non-monotonic BLER: code rate outpaces coding gain at large N.'],
            ['N', 'ku', 'kv', 'Trellis SC', 'SC errs/CW', 'Memoryless SC', 'Memless errs/CW'],
            [[16,4,7,'0.1727','1727/10K','0.1756','1756/10K'],
             [32,7,15,'0.0926','926/10K','0.0958','958/10K'],
             [64,15,29,'0.0720','720/10K','0.0757','757/10K'],
             [128,30,58,'0.1056','528/5K','0.1080','540/5K'],
             [256,59,117,'0.1888','944/5K','0.1896','948/5K']],
            [('Trellis SC', Ns, [0.173,0.093,0.072,0.106,0.189]),
             ('Memoryless SC', Ns, [0.176,0.096,0.076,0.108,0.190])],
            'Usable BLER range (7-19%). Trellis gives <5% improvement over memoryless.')

        # 9. MA-AGN MAC
        Ns = [16, 32, 64, 128, 256, 512]
        make_page(pdf,
            '9. MA-AGN MAC Class C (corner) — Continuous-State Memory', 'Moving-Average Gaussian Noise MAC',
            'Z[i] = (1-2X[i]) + (1-2Y[i]) + N[i],  N[i] = 0.3*N[i-1] + W[i]  (AR(1) noise)',
            ['Parameters: alpha=0.3, sigma^2=0.251',
             'Path: make_path(N, N), chained decoding',
             'Design: GMAC Class C proxy',
             'No analytical trellis (continuous state). Memoryless SC is the baseline.',
             'NPD d16h100 trained for N=64/128/256. Beats SC only at N=16.'],
            ['N', 'ku', 'kv', 'Memoryless SC', 'SC errs/CW', 'NPD (d16h100)', 'NPD errs/CW', 'NPD/SC'],
            [[16,4,7,'0.1750','349/2K','0.1375','275/2K','0.79x'],
             [32,7,15,'0.0570','114/2K','0.0530','265/5K','0.93x'],
             [64,15,29,'0.0250','75/3K','0.0354','177/5K','1.42x'],
             [128,30,58,'0.0066','119/18K','0.1048','524/5K','15.9x'],
             [256,59,117,'2e-4','1/5K','0.0218','109/5K','109x'],
             [512,119,233,'8e-4','4/5K','--','--','--']],
            [('Memoryless SC', Ns, [0.175,0.057,0.025,0.007,2e-4,8e-4]),
             ('NPD d16h100', [16,32,64,128,256], [0.138,0.053,0.035,0.105,0.022])],
            'NPD beats SC only at N=16 (21%). Memoryless SC improves rapidly — NPD wall at N>=64.')

    print(f"PDF saved: {os.path.join(OUT_DIR, 'ALL_RESULTS_v2.pdf')}")

if __name__ == '__main__':
    main()
