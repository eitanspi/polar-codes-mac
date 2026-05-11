#!/usr/bin/env python3
"""Run error injection in self_play mode."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import neural.train_error_inject as t
t.MODE = 'self_play'
t.P_SELF = 0.15
t.LR = 1e-4
t.CKPT_OUT = 'saved_models/n256_self_play_best.pt'
t.main()
