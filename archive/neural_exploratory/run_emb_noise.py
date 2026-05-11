#!/usr/bin/env python3
"""Run error injection in emb_noise mode."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import neural.train_error_inject as t
t.MODE = 'emb_noise'
t.EMB_NOISE = 0.1
t.LR = 1e-4
t.CKPT_OUT = 'saved_models/n256_emb_noise_best.pt'
t.main()
