#!/usr/bin/env python3
"""
train_abnmac_v2.py — Train neural SC decoder for ABNMAC (Class C, analytical design).

Key changes from v1:
  - Uses Class C path (0^N 1^N) with analytical Bhattacharyya design
  - Class C is simpler (no CalcParent needed) → easier to train
  - Starts with N=8 for faster initial convergence
  - Higher initial LR with aggressive warmup
"""
import sys, os, math, time, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.encoder import polar_encode_batch
from polar.channels import ABNMAC
from polar.design import design_abnmac, make_path
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

# ABNMAC capacities: I(Z;X) ≈ 0.400, I(Z;Y|X) ≈ 0.800
# Class C rates (conservative):
RATES = {
    8:   {'ku': 3,  'kv': 6},
    16:  {'ku': 6,  'kv': 12},
    32:  {'ku': 13, 'kv': 25},
    64:  {'ku': 26, 'kv': 51},
    128: {'ku': 51, 'kv': 102},
}

def encode_z(zf):
    result = np.empty(zf.shape, dtype=np.int64)
    for idx in np.ndindex(zf.shape):
        zx, zy = zf[idx]
        result[idx] = zx * 2 + zy
    return result

def evaluate(model, channel, b, Au, Av, fu, fv, N, n_cw, batch_size=25):
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
    return errs / total

def eval_sc(channel, b, Au, Av, fu, fv, N, n_cw):
    from polar.decoder import decode_single
    from polar.encoder import polar_encode
    rng = np.random.default_rng(42)
    errs = 0
    for i in range(n_cw):
        uf = np.zeros(N, dtype=int); vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist()); y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int))
        z_list = z.tolist()
        z_tuples = [tuple(zz) if isinstance(zz, list) else zz for zz in z_list]
        u_dec, v_dec = decode_single(N, z_tuples, b, fu, fv, channel, log_domain=True)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
    return errs / n_cw

def main():
    channel = ABNMAC()
    model = PureNeuralCompGraphDecoder(d=16, hidden=64, n_layers=2, vocab_size=4)
    print(f"ABNMAC v2 — Class C (no CalcParent)")
    print(f"Parameters: {model.count_parameters():,}", flush=True)

    results = {}
    SC_BASELINES = {}

    for N in [8, 16, 32, 64, 128]:
        n = int(math.log2(N))
        rates = RATES[N]
        ku, kv = rates['ku'], rates['kv']
        Au, Av, fu, fv, _, _ = design_abnmac(n, ku, kv)
        b = make_path(N, N)  # Class C: 0^N 1^N

        iters = {8: 10000, 16: 20000, 32: 30000, 64: 50000, 128: 80000}[N]
        lr = {8: 3e-3, 16: 1e-3, 32: 5e-4, 64: 3e-4, 128: 1e-4}[N]
        bs = {8: 64, 16: 32, 32: 16, 64: 8, 128: 4}[N]

        print(f"\n{'='*60}")
        print(f"  N={N}, ku={ku}, kv={kv}, Ru={ku/N:.3f}, Rv={kv/N:.3f}")
        print(f"  Class C path, iters={iters}, lr={lr}, batch={bs}")
        print(f"{'='*60}", flush=True)

        # SC baseline
        n_sc = min(1000, 200 if N >= 128 else 1000)
        sc_bler = eval_sc(channel, b, Au, Av, fu, fv, N, n_sc)
        SC_BASELINES[N] = sc_bler
        print(f"  SC BLER: {sc_bler:.4f}", flush=True)

        init_bler = evaluate(model, channel, b, Au, Av, fu, fv, N, 300)
        print(f"  Initial NN BLER: {init_bler:.4f}", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr*0.01)
        rng = np.random.default_rng(42)
        t0 = time.time()
        losses = []; best_bler = init_bler

        model.train()
        for it in range(1, iters + 1):
            uf = np.zeros((bs, N), dtype=int); vf = np.zeros((bs, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, bs)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, bs)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            zf_int = encode_z(zf)
            zt = torch.from_numpy(zf_int).long()
            ut = torch.from_numpy(uf).float(); vt = torch.from_numpy(vf).float()

            all_logits, all_targets, _, _, _ = model(zt, b, fu, fv, u_true=ut, v_true=vt)
            if all_logits:
                logits = torch.stack(all_logits, dim=1)
                targets = torch.stack(all_targets, dim=1)
                loss = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                losses.append(loss.item())

            if it % max(1000, iters // 10) == 0:
                elapsed = time.time() - t0
                avg_loss = np.mean(losses[-200:]) if losses else 0
                bler = evaluate(model, channel, b, Au, Av, fu, fv, N, 500)
                improved = ''
                if bler < best_bler:
                    best_bler = bler
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'ncg_abnmac_N{N}.pt'))
                    improved = ' *BEST*'
                ratio = bler / max(sc_bler, 1e-8)
                print(f'  [{it:>6}/{iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                      f'(best={best_bler:.4f}, SC={sc_bler:.4f}, ratio={ratio:.2f}x) '
                      f'{elapsed/60:.0f}min{improved}', flush=True)

        final_bler = evaluate(model, channel, b, Au, Av, fu, fv, N, 1000)
        results[str(N)] = {
            'N': N, 'ku': ku, 'kv': kv,
            'nn_bler': final_bler, 'sc_bler': sc_bler,
            'best_bler': best_bler,
        }
        print(f"  Final: BLER={final_bler:.4f} (SC={sc_bler:.4f})", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'abnmac', 'abnmac_nn_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}", flush=True)

    print(f"\n{'='*60}")
    print(f"  ABNMAC Class C Results")
    print(f"{'='*60}")
    for N_s, r in results.items():
        sc = r['sc_bler']; nn = r['nn_bler']
        ratio = f"{nn/sc:.2f}x" if sc > 0 else "-"
        print(f"  N={r['N']:>4d}  SC={sc:.4f}  NN={nn:.4f}  ratio={ratio}")

if __name__ == '__main__':
    main()
