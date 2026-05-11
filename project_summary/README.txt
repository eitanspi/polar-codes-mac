PROJECT SUMMARY — Neural SC Decoder for MAC Polar Codes
========================================================
Last updated: 2026-04-14


RESULTS AT A GLANCE
-------------------

  Channel   | Class   | Decoder | N range    | vs SC         | Status
  ----------|---------|---------|------------|---------------|--------
  GMAC      | Corner  | NPD    | 16-256     | 0.56-0.80x    | WORKING (beats SC)
  GMAC      | Non-crn | NCG    | 32-128     | 0.87-1.43x    | WORKING (matches SC)
  GMAC      | Non-crn | NCG    | 256        | 4.5x          | THE WALL
  BEMAC     | Non-crn | NCG    | 16-1024    | 0.50-1.10x    | WORKING (no wall)
  BEMAC     | Corner  | SC     | 16-1024    | baseline      | NPD pending
  ABNMAC    | Both    | SC     | 16-256     | baseline      | NPD pending
  ISI-MAC   | Corner  | SC     | 16-256     | baseline      | NPD pending
  GE-MAC    | Corner  | SC     | 16-128     | baseline      | NPD pending


FOLDER CONTENTS
---------------

results/
  all_bler_results.csv               — MASTER TABLE: every result in one file
  gmac_classC_npd_vs_sc.csv          — GMAC corner: NPD vs SC
  gmac_classB_ncg_vs_sc.csv          — GMAC non-corner: NCG vs SC
  bemac_classB_ncg_vs_sc.csv         — BEMAC non-corner: NCG vs SC
  bemac_classC_sc_baseline.csv       — BEMAC corner: SC baseline
  abnmac_sc_baselines.csv            — ABNMAC: SC baselines both classes
  memory_channels_sc_baselines.csv   — Memory channels: SC baselines

plots/  (numbered for easy browsing)
  01  GMAC corner: NPD vs SC BLER curve (THE HEADLINE)
  12  GMAC non-corner: NCG vs SC (shows the N=256 wall)
  13  BEMAC non-corner: NCG vs SC (no wall, works to N=1024)
  14  Memory channels: SC baselines (3 channels, N=16-256)
  15  BEMAC corner: SC baseline (N=16-1024)
  16  ABNMAC: SC baselines both classes
  02  GMAC non-corner N=256: MI + BLER during NCG training
  03-04  MI per position: GMAC Class B and C
  05-07  MI comparison: SC vs NCG vs NPD
  08  NPD MI convergence during training
  09  Hybrid fast_ce MI convergence

docs/
  CHAINED_SC_MATH.pdf      — Math: why corner rate = two single-user problems
  MI_PER_POSITION.pdf       — MI plots across all channels/classes
  PROJECT_RESULTS.pdf       — Full results document


KEY NUMBERS
-----------
GMAC: I(X;Z)=0.465, I(Y;Z|X)=0.912, I(X,Y;Z)=1.376
Best NPD: 0.56x SC at N=32 (GMAC corner)
Best NCG: 0.50x SC at N=256 (BEMAC non-corner)
The wall: NCG at N=256 GMAC non-corner = 4.5x SC
