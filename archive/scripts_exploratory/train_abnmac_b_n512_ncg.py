#!/usr/bin/env python3
"""Train NCG decoder for ABNMAC Class B N=512, warm-started from N=256 checkpoint.
Budget: ~1 hour CPU, ~4000 iters with bs=4."""
import sys, os, math, time, json
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from polar.encoder import polar_encode_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

SAVE_DIR = os.path.join(BASE, 'saved_models')
DESIGNS_DIR = os.path.join(BASE, 'designs')

N = 512
n = int(math.log2(N))
ku = 205
kv = 205
BATCH_SIZE = 4
TOTAL_ITERS = 4000
LR = 2e-4
EVAL_EVERY = 500
EVAL_CW = 300
SAVE_EVERY = 1000


def encode_z(zf):
    out = np.empty(zf.shape, dtype=np.int64)
    for idx in np.ndindex(zf.shape):
        zx, zy = zf[idx]
        out[idx] = 2 * int(zx) + int(zy)
    return out


def evaluate_nn(model, channel, N, b, Au, Av, fu, fv, n_cw, batch_size=4):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            zf_int = encode_z(zf)
            zt = torch.from_numpy(zf_int).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs, total


def main():
    channel = ABNMAC()
    design_file = os.path.join(DESIGNS_DIR, f'abnmac_B_n{n}.npz')
    Au, Av, fu, fv, pe_u, pe_v, path_i = design_from_file(design_file, n, ku, kv)
    Au = sorted(Au); Av = sorted(Av)
    b = make_path(N, path_i)

    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=4)

    # Warm start from N=256
    warm_ckpt = os.path.join(SAVE_DIR, 'ncg_abnmac_classB_N256_best.pt')
    sd = torch.load(warm_ckpt, weights_only=True, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    print(f"Loaded warm-start: {os.path.basename(warm_ckpt)}")
    print(f"N={N}, ku={ku}, kv={kv}, path_i={path_i}")
    print(f"TOTAL_ITERS={TOTAL_ITERS}, BS={BATCH_SIZE}, LR={LR}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Initial eval
    print("Initial eval...", flush=True)
    errs, total = evaluate_nn(model, channel, N, b, Au, Av, fu, fv, EVAL_CW)
    init_bler = errs / total
    print(f"  Initial BLER: {init_bler:.4f} ({errs}/{total})", flush=True)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TOTAL_ITERS, eta_min=LR * 0.01)

    rng = np.random.default_rng()
    t0 = time.time()
    losses = []
    best_bler = init_bler
    best_errs = errs

    model.train()
    for it in range(1, TOTAL_ITERS + 1):
        uf = np.zeros((BATCH_SIZE, N), dtype=int)
        vf = np.zeros((BATCH_SIZE, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH_SIZE)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH_SIZE)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        zf_int = encode_z(zf)
        zt = torch.from_numpy(zf_int).long()
        ut = torch.from_numpy(uf).float()
        vt = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _, _ = model(zt, b, fu, fv, u_true=ut, v_true=vt)

        if len(all_logits) > 0:
            logits = torch.stack(all_logits, dim=1)
            targets = torch.stack(all_targets, dim=1)
            loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            losses.append(loss.item())

        if it % EVAL_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            avg_loss = np.mean(losses[-200:]) if losses else 0
            errs, total = evaluate_nn(model, channel, N, b, Au, Av, fu, fv, EVAL_CW)
            bler = errs / total
            tag = ''
            if bler < best_bler:
                best_bler = bler
                best_errs = errs
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'ncg_abnmac_classB_N512_best.pt'))
                tag = ' *BEST*'
            print(f'  [{it:>5}/{TOTAL_ITERS}] loss={avg_loss:.4f} BLER={bler:.4f} ({errs}/{total}) '
                  f'best={best_bler:.4f} {elapsed:.1f}min{tag}', flush=True)

        if it % SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'ncg_abnmac_classB_N512_it{it}.pt'))

    # Final eval with more CW
    print("\nFinal eval (500 CW)...", flush=True)
    errs, total = evaluate_nn(model, channel, N, b, Au, Av, fu, fv, 500)
    final_bler = errs / total
    if final_bler < best_bler:
        best_bler = final_bler
        best_errs = errs
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'ncg_abnmac_classB_N512_best.pt'))
    print(f"  Final BLER: {final_bler:.4f} ({errs}/{total}), best={best_bler:.4f}", flush=True)

    elapsed_total = (time.time() - t0) / 60
    result = {
        'N': N, 'ku': ku, 'kv': kv,
        'total_iters': TOTAL_ITERS, 'batch_size': BATCH_SIZE,
        'init_bler': init_bler, 'final_bler': final_bler,
        'best_bler': best_bler, 'time_min': round(elapsed_total, 1),
        'warm_start': 'ncg_abnmac_classB_N256_best.pt'
    }
    out = os.path.join(BASE, 'results', 'abnmac_b_n512_ncg_train.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out}", flush=True)


if __name__ == '__main__':
    main()
