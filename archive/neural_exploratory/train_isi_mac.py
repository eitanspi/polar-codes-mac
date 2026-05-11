#!/usr/bin/env python3
"""
train_isi_mac.py — Train neural SC decoder for ISI-MAC (channel with memory).

ISI-MAC: Z[i] = (1-2X[i]) + (1-2Y[i]) + h*((1-2X[i-1]) + (1-2Y[i-1])) + W[i]

Uses sliding window z_encoder that takes (z[i], z[i-1]) as input.
Trains at N=32 first, then curriculum to N=64.

This is a proof-of-concept for the "channels with memory" contribution.
"""

import sys, os, math, time, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from polar.encoder import polar_encode, polar_encode_batch, bit_reversal_perm
from polar.channels_memory import ISIMAC
from polar.design import make_path

# ISI-MAC doesn't have analytical design, so we'll use MC design
# For now, use symmetric rate ~0.40 (conservative)
RATES = {
    32:  {'ku': 13, 'kv': 13},  # ~0.40
    64:  {'ku': 26, 'kv': 26},
}

SNR_DB = 6.0
ISI_H = 0.3  # moderate ISI coefficient

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'train_isi_mac.log')


def genie_design_isi_mac(N, channel, n_trials=2000, seed=42):
    """Simple MC genie-aided design for ISI-MAC."""
    from polar.encoder import polar_encode
    rng = np.random.default_rng(seed)
    n = int(math.log2(N))
    b = make_path(N, N // 2)  # Class B

    u_err = np.zeros(N)
    v_err = np.zeros(N)

    for trial in range(n_trials):
        u = rng.integers(0, 2, size=N).tolist()
        v = rng.integers(0, 2, size=N).tolist()
        x = polar_encode(u)
        y = polar_encode(v)

        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int))
        z_list = z.tolist() if hasattr(z, 'tolist') else list(z)

        # Genie-aided decode (feed true bits)
        from polar.decoder import decode_single
        u_hat_dict = {}
        v_hat_dict = {}
        i_u, i_v = 0, 0

        # For now, just use the SC decoder with the GMAC-like interface
        # But ISI-MAC needs state — the analytical decoder can't handle this
        # So we'll do a simplified genie analysis: check if each position
        # can be decoded correctly given perfect prior decisions

        # Simple approach: decode with the GaussianMAC decoder (ignores ISI)
        # and count errors per position
        from polar.channels import GaussianMAC
        gmac = GaussianMAC(sigma2=channel.sigma2)
        try:
            u_dec, v_dec = decode_single(N, z_list, b,
                                          {}, {},  # no frozen bits
                                          gmac, log_domain=True)
            for i in range(N):
                if u_dec[i] != u[i]:
                    u_err[i] += 1
                if v_dec[i] != v[i]:
                    v_err[i] += 1
        except Exception:
            pass

    u_pe = u_err / n_trials
    v_pe = v_err / n_trials

    sorted_u = np.argsort(u_pe)
    sorted_v = np.argsort(v_pe)

    return sorted_u, sorted_v, u_pe, v_pe


def make_design(N, ku, kv, channel):
    """Create frozen sets for ISI-MAC."""
    n = int(math.log2(N))

    # Use GMAC-based GA design as a starting point (ISI is GMAC + ISI)
    from polar.design import design_gmac
    Au, Av, fu, fv, _, _ = design_gmac(n, ku, kv, channel.sigma2)
    b = make_path(N, N // 2)  # Class B

    return Au, Av, fu, fv, b


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
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = channel.sample_batch(xf, yf).astype(np.float32)
            zt = torch.from_numpy(zf).float()
            _, _, uh, vh, _ = model(zt, b, fu, fv)
            for i in range(actual):
                e = any(int(uh[p][i].item()) != uf[i, p-1] for p in Au if p in uh) or \
                    any(int(vh[p][i].item()) != vf[i, p-1] for p in Av if p in vh)
                if e: errs += 1
            total += actual
    model.train()
    return errs / total


def eval_memoryless_baseline(channel, b, Au, Av, fu, fv, N, n_cw):
    """Baseline: SC decoder treating ISI-MAC as memoryless GMAC."""
    from polar.channels import GaussianMAC
    from polar.decoder import decode_single
    gmac = GaussianMAC(sigma2=channel.sigma2)
    rng = np.random.default_rng(42)
    errs = 0
    for i in range(n_cw):
        uf = np.zeros(N, dtype=int)
        vf = np.zeros(N, dtype=int)
        for p in Au: uf[p-1] = rng.integers(0, 2)
        for p in Av: vf[p-1] = rng.integers(0, 2)
        x = polar_encode(uf.tolist())
        y = polar_encode(vf.tolist())
        z = channel.sample_batch(np.array(x, dtype=int), np.array(y, dtype=int))
        z_list = z.tolist()
        u_dec, v_dec = decode_single(N, z_list, b, fu, fv, gmac, log_domain=True)
        if any(u_dec[p-1] != uf[p-1] for p in Au) or \
           any(v_dec[p-1] != vf[p-1] for p in Av):
            errs += 1
    return errs / n_cw


def train_at_N(N, model, channel, total_iters=30000, lr=1e-3, batch_size=16,
               eval_every=2000):
    rates = RATES[N]
    ku, kv = rates['ku'], rates['kv']
    Au, Av, fu, fv, b = make_design(N, ku, kv, channel)

    print(f"\n{'='*60}")
    print(f"  ISI-MAC Training N={N}, h={ISI_H}")
    print(f"  ku={ku}, kv={kv}, Ru={ku/N:.3f}, Rv={kv/N:.3f}")
    print(f"  iters={total_iters}, lr={lr}, batch={batch_size}")
    print(f"{'='*60}")

    # Baselines
    print(f"  Memoryless SC baseline (1000 cw)...", flush=True)
    memoryless_bler = eval_memoryless_baseline(channel, b, Au, Av, fu, fv, N, 1000)
    print(f"  Memoryless SC BLER: {memoryless_bler:.4f}")

    init_bler = evaluate(model, channel, b, Au, Av, fu, fv, N, 500)
    print(f"  Initial NN BLER: {init_bler:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_iters, eta_min=lr * 0.01)

    rng = np.random.default_rng(42)
    t0 = time.time()
    losses = []
    best_bler = init_bler

    model.train()
    for it in range(1, total_iters + 1):
        uf = np.zeros((batch_size, N), dtype=int)
        vf = np.zeros((batch_size, N), dtype=int)
        for p in Au: uf[:, p-1] = rng.integers(0, 2, batch_size)
        for p in Av: vf[:, p-1] = rng.integers(0, 2, batch_size)
        xf = polar_encode_batch(uf)
        yf = polar_encode_batch(vf)
        zf = channel.sample_batch(xf, yf).astype(np.float32)
        zt = torch.from_numpy(zf).float()
        ut = torch.from_numpy(uf).float()
        vt = torch.from_numpy(vf).float()

        all_logits, all_targets, _, _, _ = model(
            zt, b, fu, fv, u_true=ut, v_true=vt)

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

        if it % eval_every == 0:
            elapsed = time.time() - t0
            avg_loss = np.mean(losses[-200:]) if losses else 0
            bler = evaluate(model, channel, b, Au, Av, fu, fv, N, 500)

            improved = ''
            if bler < best_bler:
                best_bler = bler
                torch.save(model.state_dict(),
                           os.path.join(SAVE_DIR, f'ncg_isi_mac_N{N}.pt'))
                improved = ' *BEST*'

            msg = (f'  [{it:>6}/{total_iters}] loss={avg_loss:.4f} BLER={bler:.4f} '
                   f'(best={best_bler:.4f}, memoryless_SC={memoryless_bler:.4f}) '
                   f'{elapsed/60:.0f}min{improved}')
            print(msg, flush=True)
            with open(LOG_FILE, 'a') as f:
                f.write(msg + '\n')

    final_bler = evaluate(model, channel, b, Au, Av, fu, fv, N, 1000)
    print(f"  Final BLER: {final_bler:.4f} (memoryless SC={memoryless_bler:.4f})")
    return final_bler, memoryless_bler


def main():
    channel = ISIMAC.from_snr_db(SNR_DB, h=ISI_H)

    print(f"\n{'#'*60}")
    print(f"  ISI-MAC Neural SC Decoder Training")
    print(f"  Channel: Z[i] = s_x[i]+s_y[i] + {ISI_H}*(s_x[i-1]+s_y[i-1]) + W[i]")
    print(f"  SNR={SNR_DB}dB, sigma2={channel.sigma2:.4f}")
    print(f"{'#'*60}")

    from neural.ncg_isi_mac import ISIMACNeuralDecoder
    model = ISIMACNeuralDecoder(d=16, hidden=64, n_layers=2, z_hidden=32,
                                 z_encoder_type='window')
    print(f"  Parameters: {model.count_parameters():,}")

    # Try to load GMAC pretrained weights for the tree operations
    gmac_ckpt = os.path.join(SAVE_DIR, 'ncg_gmac_mlp_N32.pt')
    if os.path.exists(gmac_ckpt):
        sd = torch.load(gmac_ckpt, map_location='cpu', weights_only=True)
        # Map tree.* keys
        tree_sd = {}
        for k, v in sd.items():
            if k.startswith('tree.'):
                tree_sd[k[5:]] = v  # strip 'tree.' prefix
        model.tree.load_state_dict(tree_sd, strict=False)
        print(f"  Loaded tree weights from GMAC N=32")

    results = {}

    for N in [32, 64]:
        iters = {32: 20000, 64: 40000}[N]
        lr = {32: 1e-3, 64: 5e-4}[N]
        bs = {32: 16, 64: 8}[N]

        nn_bler, sc_bler = train_at_N(
            N, model, channel, total_iters=iters, lr=lr,
            batch_size=bs, eval_every=max(1000, iters // 10))

        results[str(N)] = {
            'N': N, 'ku': RATES[N]['ku'], 'kv': RATES[N]['kv'],
            'nn_bler': nn_bler,
            'memoryless_sc_bler': sc_bler,
            'snr_db': SNR_DB, 'isi_h': ISI_H,
        }

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                            'isi_mac_nn_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print(f"\n{'='*60}")
    print(f"  ISI-MAC Results (h={ISI_H}, SNR={SNR_DB}dB)")
    print(f"{'='*60}")
    print(f"  {'N':>5s}  {'Memoryless SC':>14s}  {'NN-SC':>8s}  {'Improvement':>12s}")
    for N_s, r in results.items():
        sc = r['memoryless_sc_bler']
        nn = r['nn_bler']
        imp = f"{sc/nn:.2f}x" if nn > 0 else "-"
        print(f"  {r['N']:>5d}  {sc:>14.4f}  {nn:>8.4f}  {imp:>12s}")


if __name__ == '__main__':
    main()
