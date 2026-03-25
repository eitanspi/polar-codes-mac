#!/usr/bin/env python3
"""
BLER vs N comparison: Neural NCG Decoder vs Analytical SC
Rate point: (Ru=0.51, Rv=0.72), Class B path, N = 16..1024
"""
import os, time, json, traceback

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polar.encoder import polar_encode_batch
from polar.channels import BEMAC
from polar.design import make_path
from polar.design_mc import design_bemac_mc
from neural.neural_comp_graph import NeuralCompGraphDecoder

SAVE = os.path.join('saved_models')
RESULTS = os.path.join('results')
os.makedirs(SAVE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

Ru_target, Rv_target = 0.51, 0.72

# ─── Curriculum training ─────────────────────────────────────────────────────

def train_curriculum(model, N, n_iters, batch_size, lr):
    b = make_path(N, N // 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iters, eta_min=lr/20)
    rng = np.random.default_rng()
    t0 = time.time()
    for it in range(1, n_iters + 1):
        u = rng.integers(0, 2, (batch_size, N))
        v = rng.integers(0, 2, (batch_size, N))
        x = polar_encode_batch(u); y = polar_encode_batch(v)
        z = torch.from_numpy(x + y).long()
        logits, targets, _, _ = model(
            z, b, {}, {},
            u_true=torch.from_numpy(u).float(),
            v_true=torch.from_numpy(v).float())
        if logits:
            loss = F.cross_entropy(
                torch.stack(logits).reshape(-1, 4),
                torch.stack(targets).reshape(-1))
        else:
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if it % max(1, n_iters // 5) == 0:
            print(f"    N={N} iter {it}/{n_iters}: loss={loss.item():.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return loss.item()

# ─── Evaluation ──────────────────────────────────────────────────────────────

def eval_nn(model, N, b, Au, Av, fu, fv, n_cw, bs):
    model.eval()
    errs = 0; total = 0; rng = np.random.default_rng(999)
    with torch.no_grad():
        while total < n_cw:
            actual = min(bs, n_cw - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au: uf[:, p-1] = rng.integers(0, 2, actual)
            for p in Av: vf[:, p-1] = rng.integers(0, 2, actual)
            x = polar_encode_batch(uf); y = polar_encode_batch(vf)
            z = torch.from_numpy(x + y).long()
            _, _, uh, vh = model(z, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i,p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i,p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / max(total, 1)

def eval_sc(N, b, Au, Av, fu, fv, n_cw):
    from polar.eval import MACEval
    sc = MACEval(BEMAC(), backend='interleaved')
    _, _, bler = sc.run(N, b, Au, Av, fu, fv,
                        n_codewords=n_cw, batch_size=min(100, n_cw),
                        verbose=False)
    return bler

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Training configs per N: (n_iters, batch_size, lr, source_model)
    configs = {
        16:   (0,     0,  0,      'ncg_N16_d16.pt'),       # already trained
        32:   (0,     0,  0,      'ncg_N32_curriculum.pt'), # already trained
        64:   (0,     0,  0,      'ncg_N64_best.pt'),       # already trained
        128:  (0,     0,  0,      'ncg_N128_curriculum.pt'),# already trained
        256:  (12000, 16, 2e-4,   'ncg_N128_curriculum.pt'),# curriculum from 128
        512:  (8000,  8,  1.5e-4, None),                    # curriculum from 256
        1024: (6000,  4,  1e-4,   None),                    # curriculum from 512
    }
    # Eval configs per N
    eval_cw = {16: 5000, 32: 5000, 64: 3000, 128: 3000,
               256: 2000, 512: 1000, 1024: 500}
    eval_bs = {16: 100, 32: 100, 64: 50, 128: 50,
               256: 25, 512: 10, 1024: 5}

    results = []
    prev_model = None

    for N in [16, 32, 64, 128, 256, 512, 1024]:
        print(f"\n{'='*60}", flush=True)
        print(f"N = {N}", flush=True)
        print(f"{'='*60}", flush=True)

        n = N.bit_length() - 1
        path_i = N // 2
        b = make_path(N, path_i)
        ku = max(1, round(Ru_target * N))
        kv = max(1, round(Rv_target * N))
        print(f"  ku={ku} (Ru={ku/N:.3f}), kv={kv} (Rv={kv/N:.3f})", flush=True)

        n_iters, bs_train, lr, src = configs[N]

        # ── Load or train model ──
        ckpt = os.path.join(SAVE, f'ncg_N{N}_bler_sweep.pt')
        model = NeuralCompGraphDecoder(d=16, hidden=64, n_layers=2)

        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, weights_only=True), strict=False)
            print(f"  Loaded existing checkpoint: {ckpt}", flush=True)
        elif n_iters == 0 and src:
            model.load_state_dict(
                torch.load(os.path.join(SAVE, src), weights_only=True),
                strict=False)
            print(f"  Loaded pre-trained: {src}", flush=True)
        else:
            # Curriculum: load from source or previous
            if src:
                model.load_state_dict(
                    torch.load(os.path.join(SAVE, src), weights_only=True),
                    strict=False)
                print(f"  Curriculum from: {src}", flush=True)
            elif prev_model is not None:
                model.load_state_dict(prev_model.state_dict())
                print(f"  Curriculum from previous N={N//2}", flush=True)
            else:
                print(f"  ERROR: no source model for N={N}", flush=True)
                continue

            try:
                print(f"  Training {n_iters} iters, batch={bs_train}, lr={lr}",
                      flush=True)
                train_curriculum(model, N, n_iters, bs_train, lr)
                torch.save(model.state_dict(), ckpt)
                print(f"  Saved: {ckpt}", flush=True)
            except Exception as e:
                print(f"  TRAINING FAILED: {e}", flush=True)
                traceback.print_exc()
                continue

        prev_model = model

        # ── Design frozen set ──
        try:
            Au, Av, fu, fv, _, _ = design_bemac_mc(
                n, ku, kv, mc_trials=min(2000, max(500, 5000 // (N // 16))),
                seed=42, verbose=False, path_i=path_i)
        except Exception as e:
            print(f"  DESIGN FAILED: {e}", flush=True)
            continue

        # ── NN eval ──
        try:
            t0 = time.time()
            nn_bler = eval_nn(model, N, b, Au, Av, fu, fv,
                              eval_cw[N], eval_bs[N])
            nn_time = time.time() - t0
            print(f"  NN BLER = {nn_bler:.4f}  [{nn_time:.0f}s]", flush=True)
        except Exception as e:
            print(f"  NN EVAL FAILED: {e}", flush=True)
            nn_bler = -1

        # ── SC eval ──
        try:
            t0 = time.time()
            sc_bler = eval_sc(N, b, Au, Av, fu, fv, eval_cw[N])
            sc_time = time.time() - t0
            print(f"  SC BLER = {sc_bler:.4f}  [{sc_time:.0f}s]", flush=True)
        except Exception as e:
            print(f"  SC EVAL FAILED: {e}", flush=True)
            sc_bler = -1

        if nn_bler >= 0 and sc_bler > 0:
            ratio = nn_bler / sc_bler
            print(f"  Ratio   = {ratio:.2f}", flush=True)
        else:
            ratio = -1

        results.append({
            'N': N, 'ku': ku, 'kv': kv,
            'Ru': ku/N, 'Rv': kv/N,
            'nn_bler': nn_bler, 'sc_bler': sc_bler, 'ratio': ratio
        })

        # Save intermediate results
        with open(os.path.join(RESULTS, 'bler_vs_N_051_072.json'), 'w') as f:
            json.dump(results, f, indent=2)

    # ── Plot ──
    print("\n\nGenerating plot...", flush=True)
    plot_results(results)
    print("DONE.", flush=True)


def plot_results(results):
    valid = [r for r in results if r['nn_bler'] >= 0 and r['sc_bler'] > 0]
    if not valid:
        print("No valid results to plot"); return

    Ns = [r['N'] for r in valid]
    nn = [max(r['nn_bler'], 2e-4) for r in valid]
    sc = [max(r['sc_bler'], 2e-4) for r in valid]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.semilogy(Ns, sc, 'b-o', linewidth=2.5, markersize=10,
                label='Analytical SC Decoder', zorder=5)
    ax.semilogy(Ns, nn, 'r--s', linewidth=2.5, markersize=10,
                label='Neural NCG Decoder', zorder=5)

    for i, r in enumerate(valid):
        if r['sc_bler'] > 0:
            ratio = r['nn_bler'] / r['sc_bler']
            ax.annotate(f'{ratio:.2f}x', (Ns[i], nn[i]),
                        textcoords='offset points', xytext=(12, 5),
                        fontsize=10, color='red', fontweight='bold')

    ax.set_xlabel('Block Length N', fontsize=14)
    ax.set_ylabel('Block Error Rate (BLER)', fontsize=14)
    ax.set_title('BEMAC Class B: Neural NCG vs Analytical SC\n'
                 r'$R_u \approx 0.51,\; R_v \approx 0.72,\;$'
                 r' Path $b = U^{N/2}V^N U^{N/2}$',
                 fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.legend(fontsize=13, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    out = os.path.join(RESULTS, 'bler_vs_N_051_072.png')
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace('.png', '.pdf'))
    plt.close()
    print(f"  Plot saved: {out}")


if __name__ == '__main__':
    main()
