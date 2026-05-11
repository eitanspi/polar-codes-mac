#!/usr/bin:usr/env python3
"""
Shortcut experiment (A): evaluate rate-1 NCG N=32 model with two candidate frozen sets.
  - SC:  Au=[20..32]\{25} + 25, Av=[3..16]+{20,22,23,24,26..32}  (ku=13, kv=25)
  - NCG: swap U 25->19, V 22->2 (from MI ranking)
"""
import sys, os, time, json
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
CKPT = os.path.join(os.path.dirname(__file__), '..', 'class_c_npd',
                    'results', 'ncg_r1_32', 'iter300000.pt')


def eval_bler(model, ch, path_b, frozen_u, frozen_v, Au, Av,
              n_cw, batch, seed, device):
    rng = np.random.default_rng(seed)
    errs = 0
    done = 0
    t0 = time.time()
    while done < n_cw:
        B = min(batch, n_cw - done)
        u = np.zeros((B, N), dtype=int); v = np.zeros((B, N), dtype=int)
        for p in Au: u[:, p - 1] = rng.integers(0, 2, B)
        for p in Av: v[:, p - 1] = rng.integers(0, 2, B)
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(ch.sample_batch(x, y)).float().to(device)

        with torch.no_grad():
            _, _, u_hat, v_hat, _ = model(z, path_b, frozen_u, frozen_v)

        # stack predictions
        u_pred = torch.stack([u_hat[p] for p in sorted(Au)], dim=1).cpu().numpy().astype(int)
        v_pred = torch.stack([v_hat[p] for p in sorted(Av)], dim=1).cpu().numpy().astype(int)
        u_true = u[:, [p - 1 for p in sorted(Au)]]
        v_true = v[:, [p - 1 for p in sorted(Av)]]

        ue = (u_pred != u_true).any(1)
        ve = (v_pred != v_true).any(1)
        errs += int((ue | ve).sum())
        done += B
    return errs, done, time.time() - t0


def main():
    device = torch.device('cpu')
    model = GmacNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    ch = GaussianMAC(sigma2=SIGMA2)
    b = make_path(N, N // 2)

    # --- SC-designed frozen set (Class B @ 6 dB) ---
    ku, kv = 13, 25
    Au_sc, Av_sc, _, _, _, _, _ = design_from_file(
        'designs/gmac_B_n5_snr6dB.npz', 5, ku, kv)
    Au_sc = sorted(Au_sc); Av_sc = sorted(Av_sc)

    # --- NCG frozen set = SC with 2-position swap ---
    Au_ncg = sorted((set(Au_sc) - {25}) | {19})
    Av_ncg = sorted((set(Av_sc) - {22}) | {2})

    print(f'Au_sc : {Au_sc}')
    print(f'Au_ncg: {Au_ncg}')
    print(f'Av_sc : {Av_sc}')
    print(f'Av_ncg: {Av_ncg}')
    print(f'\nrate = {(ku+kv)/(2*N):.3f}, ku={ku} kv={kv}')

    def to_frozen(Au, Av):
        fu = {p: 0 for p in range(1, N + 1) if p not in Au}
        fv = {p: 0 for p in range(1, N + 1) if p not in Av}
        return fu, fv

    N_CW = 20000
    BATCH = 256
    print(f'\nEvaluating with {N_CW} codewords per frozen set...')

    for name, Au, Av in [('SC', Au_sc, Av_sc), ('NCG', Au_ncg, Av_ncg)]:
        fu, fv = to_frozen(Au, Av)
        errs, done, elapsed = eval_bler(
            model, ch, b, fu, fv, Au, Av, N_CW, BATCH, seed=777, device=device)
        bler = errs / done
        se = (bler * (1 - bler) / done) ** 0.5
        print(f'  {name:3s}: BLER = {bler:.4f}  (±{1.96*se:.4f}, '
              f'{errs}/{done}, {elapsed:.0f}s)')


if __name__ == '__main__':
    main()
