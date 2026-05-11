#!/usr/bin/env python3
"""
Evaluate BLER of the fine-tuned NCG-frozen model at rate 0.594.
Also report checkpoint progression.
"""
import sys, os, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch

from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_gmac import GmacNeuralCompGraphDecoder

N = 32
SIGMA2 = 10 ** (-6.0 / 10)
CKPT_DIR = 'class_c_npd/results/ncg_r1_32_ft_ncgfrozen'


def eval_bler(model, ch, b, fu, fv, Au, Av, n_cw, batch, seed):
    rng = np.random.default_rng(seed)
    errs = 0; done = 0
    while done < n_cw:
        B = min(batch, n_cw - done)
        u = np.zeros((B, N), dtype=int); v = np.zeros((B, N), dtype=int)
        for p in Au: u[:, p - 1] = rng.integers(0, 2, B)
        for p in Av: v[:, p - 1] = rng.integers(0, 2, B)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float()
        with torch.no_grad():
            _, _, u_hat, v_hat, _ = model(z, b, fu, fv)
        u_pred = torch.stack([u_hat[p] for p in sorted(Au)], 1).numpy().astype(int)
        v_pred = torch.stack([v_hat[p] for p in sorted(Av)], 1).numpy().astype(int)
        u_true = u[:, [p - 1 for p in sorted(Au)]]
        v_true = v[:, [p - 1 for p in sorted(Av)]]
        ue = (u_pred != u_true).any(1); ve = (v_pred != v_true).any(1)
        errs += int((ue | ve).sum()); done += B
    return errs, done


def main():
    device = torch.device('cpu')
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2).to(device)
    ch = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)

    Au_sc, Av_sc, _, _, _, _, _ = design_from_file(
        'designs/gmac_B_n5_snr6dB.npz', 5, 13, 25)
    Au_ncg = sorted((set(Au_sc) - {25}) | {19})
    Av_ncg = sorted((set(Av_sc) - {22}) | {2})
    fu = {p: 0 for p in range(1, N + 1) if p not in Au_ncg}
    fv = {p: 0 for p in range(1, N + 1) if p not in Av_ncg}

    # List all checkpoints in order
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, 'iter*.pt')),
                   key=lambda p: int(os.path.basename(p).split('iter')[1].split('.')[0]))
    ckpts.append(os.path.join(CKPT_DIR, 'final.pt'))

    print(f'{"ckpt":<14} {"BLER":<10} {"95% CI":<14} err/tot')
    for ck in ckpts:
        if not os.path.exists(ck): continue
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        model.eval()
        n_cw = 20000 if ck.endswith('final.pt') else 5000
        t0 = time.time()
        errs, done = eval_bler(model, ch, b, fu, fv, Au_ncg, Av_ncg,
                               n_cw, batch=256, seed=777)
        bler = errs / done
        se = (bler * (1 - bler) / done) ** 0.5
        tag = os.path.basename(ck).replace('.pt', '')
        print(f'{tag:<14} {bler:<10.4f} ±{1.96*se:.4f}       {errs}/{done}  ({time.time()-t0:.0f}s)')

    # Also reference: fine-tuned model but evaluated on SC frozen set
    # (decoder was trained on NCG frozen; using on SC frozen is a mismatch check)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'final.pt'),
                                     map_location=device, weights_only=True))
    fu_sc = {p: 0 for p in range(1, N + 1) if p not in Au_sc}
    fv_sc = {p: 0 for p in range(1, N + 1) if p not in Av_sc}
    errs, done = eval_bler(model, ch, b, fu_sc, fv_sc,
                           sorted(Au_sc), sorted(Av_sc), 20000, 256, 777)
    bler = errs / done
    print(f'\nFinal on SC-frozen (mismatch, decoder was trained on NCG): '
          f'BLER={bler:.4f} ({errs}/{done})')

    # Reference baselines for context
    print('\nReference (from prior project results):')
    print('  GMAC Class B N=32 SC BLER           ~ 0.047')
    print('  GMAC Class B N=32 NCG BLER (SC set) ~ 0.040')


if __name__ == '__main__':
    main()
