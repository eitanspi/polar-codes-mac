#!/usr/bin/env python3
"""
Priority 1: ABNMAC Class B NCG — extend curriculum from N=256 to N=512, then N=1024.
Warm-starts from ncg_abnmac_classB_N256_best.pt.
"""
import os, sys, math, time, json
import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
torch.set_num_threads(4)

from polar.encoder import polar_encode_batch
from polar.channels import ABNMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_pure_neural import PureNeuralCompGraphDecoder

D = 16; HIDDEN = 64; N_LAYERS = 2
BATCH = 8    # Small batch for large N
LR = 5e-5    # Low LR for fine-tuning from N=256
EVAL_EVERY = 2000
EVAL_CW = 300

CURRICULUM = [
    {'N': 512,  'iters': 100000, 'ku': 205, 'kv': 205, 'max_hours': 4.5},
    {'N': 1024, 'iters': 50000,  'ku': 410, 'kv': 410, 'max_hours': 3.0},
]


def encode_z(zf):
    out = np.empty(zf.shape, dtype=np.int64)
    for idx in np.ndindex(zf.shape):
        zx, zy = zf[idx]; out[idx] = 2*int(zx) + int(zy)
    return out


def evaluate_nn(model, channel, b, Au, Av, fu, fv, N, n_cw, batch_size=10):
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(batch_size, n_cw - total)
            uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf)
            zt = torch.from_numpy(encode_z(zf)).long()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / max(total, 1)


def evaluate_sc(channel, Au, Av, fu, fv, N, n_cw=200, batch_size=5):
    from polar.decoder import decode_batch
    b = make_path(N, N // 2)
    rng = np.random.default_rng(888)
    from polar.encoder import build_message_batch
    errs = 0; total = 0
    u_info_idx = np.array([p - 1 for p in Au])
    v_info_idx = np.array([p - 1 for p in Av])
    while total < n_cw:
        actual = min(batch_size, n_cw - total)
        ku = len(Au); kv = len(Av)
        info_u = rng.integers(0, 2, size=(actual, ku))
        info_v = rng.integers(0, 2, size=(actual, kv))
        U = build_message_batch(N, info_u, Au)
        V = build_message_batch(N, info_v, Av)
        X = polar_encode_batch(U)
        Y = polar_encode_batch(V)
        Z = channel.sample_batch(X, Y)
        results = decode_batch(N, Z.tolist(), b, fu, fv, channel, vectorized=True)
        for i, (u_dec, v_dec) in enumerate(results):
            u_dec_arr = np.array(u_dec)
            v_dec_arr = np.array(v_dec)
            ue = int(np.sum(u_dec_arr[u_info_idx] != info_u[i]))
            ve = int(np.sum(v_dec_arr[v_info_idx] != info_v[i]))
            if ue > 0 or ve > 0: errs += 1
        total += actual
    return errs / max(total, 1)


def train_stage(model, channel, stage, prev_ckpt=None):
    N = stage['N']; n = int(math.log2(N))
    iters = stage['iters']; ku = stage['ku']; kv = stage['kv']
    max_hours = stage.get('max_hours', 5.0)
    path_i = N // 2
    b = make_path(N, path_i)

    design_file = os.path.join(BASE, 'designs', f'abnmac_B_n{n}.npz')
    Au, Av, fu, fv, _, _, _ = design_from_file(design_file, n, ku, kv)

    print(f"\n{'='*70}", flush=True)
    print(f"  Stage N={N}, Class B, ku={ku}, kv={kv}, iters={iters}", flush=True)
    print(f"{'='*70}", flush=True)

    if prev_ckpt and os.path.exists(prev_ckpt):
        sd = torch.load(prev_ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(sd, strict=False)
        print(f"  Loaded: {os.path.basename(prev_ckpt)}", flush=True)

    # Measure initial BLER
    init_bler = evaluate_nn(model, channel, b, Au, Av, fu, fv, N, EVAL_CW)
    print(f"  Initial NN BLER: {init_bler:.4f}", flush=True)

    # SC baseline (small sample at large N)
    sc_cw = 200 if N >= 512 else 300
    print(f"  Computing SC baseline ({sc_cw} CW)...", flush=True)
    t_sc = time.time()
    sc_bler = evaluate_sc(channel, Au, Av, fu, fv, N, n_cw=sc_cw, batch_size=5)
    print(f"  SC BLER: {sc_bler:.4f} ({time.time()-t_sc:.1f}s)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=LR * 0.01)

    rng = np.random.default_rng()
    t0 = time.time()
    best_bler = init_bler
    best_path = os.path.join(BASE, 'saved_models', f'ncg_abnmac_classB_N{N}_best.pt')
    losses = []

    model.train()
    for it in range(1, iters + 1):
        uf = np.zeros((BATCH, N), dtype=int); vf = np.zeros((BATCH, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, BATCH)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, BATCH)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf)
        zt = torch.from_numpy(encode_z(zf)).long()
        ut = torch.from_numpy(uf).float(); vt = torch.from_numpy(vf).float()

        logits, targets, _, _, _ = model(zt, b, fu, fv, u_true=ut, v_true=vt)
        if not logits:
            continue
        loss = F.cross_entropy(torch.stack(logits).reshape(-1, 4),
                               torch.stack(targets).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if it % EVAL_EVERY == 0 or it == iters:
            bler = evaluate_nn(model, channel, b, Au, Av, fu, fv, N, EVAL_CW)
            recent = float(np.mean(losses[-1000:])) if losses else 0
            elapsed_min = (time.time() - t0) / 60
            elapsed_h = elapsed_min / 60
            improved = ""
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(), best_path)
                improved = " *BEST*"
            # Check time budget
            if elapsed_h > max_hours:
                print(f"  [{it:7d}/{iters}] TIME LIMIT ({max_hours}h). Best={best_bler:.4f}", flush=True)
                break
            print(f"  [{it:7d}/{iters}] loss={loss.item():.4f} (avg={recent:.4f}) "
                  f"BLER={bler:.4f} best={best_bler:.4f} SC={sc_bler:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {elapsed_min:.1f}min{improved}",
                  flush=True)

    # Final eval with 500 CW
    final_bler = evaluate_nn(model, channel, b, Au, Av, fu, fv, N, 500)
    print(f"  Stage N={N} done. Final BLER={final_bler:.4f} Best={best_bler:.4f} SC={sc_bler:.4f} "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    if best_bler < 1.0:
        torch.save(model.state_dict(), best_path)
    return best_path, best_bler, sc_bler


def main():
    channel = ABNMAC()
    model = PureNeuralCompGraphDecoder(d=D, hidden=HIDDEN, n_layers=N_LAYERS, vocab_size=4)
    print(f"  ABNMAC Class B NCG extension to N=512,1024")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    results = {'stages': []}
    prev_ckpt = os.path.join(BASE, 'saved_models', 'ncg_abnmac_classB_N256_best.pt')

    for stage in CURRICULUM:
        ckpt, best_bler, sc_bler = train_stage(model, channel, stage, prev_ckpt)
        results['stages'].append({
            'N': stage['N'], 'ku': stage['ku'], 'kv': stage['kv'],
            'iters': stage['iters'], 'best_bler': best_bler, 'sc_bler': sc_bler,
        })
        prev_ckpt = ckpt

        out = os.path.join(BASE, 'results', 'abnmac_classB_ncg_n512_n1024.json')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {out}", flush=True)

        if best_bler >= 0.5:
            print(f"\n  N={stage['N']} did not converge (BLER={best_bler:.3f}). Stopping.", flush=True)
            break

    print(f"\n  Training complete.", flush=True)


if __name__ == '__main__':
    main()
