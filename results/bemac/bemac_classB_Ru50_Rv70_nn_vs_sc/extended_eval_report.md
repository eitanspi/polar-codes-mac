# BEMAC Class B (Ru=0.50, Rv=0.70) — Extended Evaluation at N≥256

Re-ran NN-SC and analytical SC at N=256, 512, 1024 with high codeword counts
to verify the original suspicious values:

| N | Original NN BLER | Original SC BLER | Original CW |
|---|---|---|---|
| 256 | 4e-05 | 8e-05 | 50000 |
| 512 | 0.0 | 0.0 | 10000 |
| 1024 | 1e-04 | 1e-04 | 10000 |

## Extended results

| N | NN errors | NN cw | NN BLER (95% CI) | SC errors | SC cw | SC BLER (95% CI) | NN/SC |
|---|---|---|---|---|---|---|---|
| 256 | 13 | 100000 | 1.30e-04 [7.60e-05, 2.22e-04] | 8 | 100000 | 8.00e-05 [4.05e-05, 1.58e-04] | 1.6249999999999998 |
| 512 | 3 | 50000 | 6.00e-05 [2.04e-05, 1.76e-04] | 5 | 50000 | 1.00e-04 [4.27e-05, 2.34e-04] | 0.6 |
| 1024 | 2 | 20000 | 1.00e-04 [2.74e-05, 3.65e-04] | 3 | 20000 | 1.50e-04 [5.10e-05, 4.41e-04] | 0.6666666666666667 |

Wilson 95% CI used.
