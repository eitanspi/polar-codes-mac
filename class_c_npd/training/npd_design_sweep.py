#!/usr/bin/env python3
"""
NPD-guided frozen set design + training sweep across N.

At each N:
  Phase 1: Train all-info NPD on the mixture channel (learns which positions it can decode)
  Phase 2: Measure per-position leaf-level MI → pick NPD-optimal info positions
  Phase 3: Retrain NPD with NPD-optimal frozen set
  Phase 4: Evaluate and compare to SC (both with genie design and NPD design)

Uses curriculum: warm-starts each N from the previous N's checkpoint.
"""
import sys, os, math, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn.functional as F
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.channels import GaussianMAC
from polar.decoder import _sc_decode_from_llr
from polar.design_mc import design_from_file

from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.eval.chain_eval import wilson_ci

SNR_DB = 6.0
SIGMA2 = 10 ** (-SNR_DB / 10)
D = 16; HIDDEN = 64; N_LAYERS = 2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Rate target: 50% of per-user capacity
IX_Z = 0.4645   # marginal capacity for U
IY_ZX = 0.9119  # conditional capacity for V
RATE_FRAC = 0.50

# SC reference
SC_REF = {
    16: 0.1626, 32: 0.0684, 64: 0.0266, 128: 0.0054, 256: 0.0020,
}

# Training schedule
SCHEDULE = {
    16:  {'phase1_iters': 30000, 'phase3_iters': 30000, 'batch': 64, 'lr': 3e-4, 'mi_samples': 20000},
    32:  {'phase1_iters': 50000, 'phase3_iters': 50000, 'batch': 64, 'lr': 3e-4, 'mi_samples': 20000},
    64:  {'phase1_iters': 80000, 'phase3_iters': 80000, 'batch': 32, 'lr': 2e-4, 'mi_samples': 20000},
    128: {'phase1_iters': 120000, 'phase3_iters': 100000, 'batch': 16, 'lr': 1e-4, 'mi_samples': 10000},
    256: {'phase1_iters': 200000, 'phase3_iters': 150000, 'batch': 8, 'lr': 1e-4, 'mi_samples': 10000},
}


def mixture_llr(z, sigma2):
    s2 = sigma2
    def lN(m): return -0.5 * (z - m) ** 2 / s2
    return np.logaddexp(lN(2.0), lN(0.0)) - np.logaddexp(lN(0.0), lN(-2.0))


def load_genie_design(n, ku, kv):
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'designs', f'gmac_C_n{n}_snr6dB.npz')
    Au, Av, fu, fv, pe_u, pe_v, _ = design_from_file(path, n, ku, kv)
    return Au, Av, pe_u, pe_v


def train_npd(model, N, Au, Av, iters, batch, lr, warm_from=None, tag=''):
    """Train NPD with fast_ce on the mixture channel."""
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    channel = GaussianMAC(sigma2=SIGMA2)
    frozen = {p - 1 for p in range(1, N + 1) if p not in Au}

    if warm_from and os.path.exists(warm_from):
        ckpt = torch.load(warm_from, weights_only=False, map_location='cpu')
        try:
            model.load_state_dict(ckpt['state_dict'])
            print(f'    warm-started from {os.path.basename(warm_from)}')
        except Exception:
            print(f'    warm-start failed, fresh start')

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(42)
    t0 = time.time()
    best_bler = 1.0
    ckpt_path = os.path.join(RESULTS_DIR, f'{tag}_best.pt')
    eval_every = max(2000, iters // 10)

    model.train()
    for it in range(1, iters + 1):
        u_msg = np.zeros((batch, N), dtype=np.int8)
        for p in Au: u_msg[:, p - 1] = rng.integers(0, 2, batch)
        x_phys = polar_encode_batch(u_msg.astype(int))
        v_msg = np.zeros((batch, N), dtype=np.int8)
        for p in Av: v_msg[:, p - 1] = rng.integers(0, 2, batch)
        y_phys = polar_encode_batch(v_msg.astype(int))
        z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
        emb = model.encode_channel(torch.from_numpy(z[:, br].astype(np.float32)).unsqueeze(-1))
        loss = model.fast_ce(emb, torch.from_numpy(x_phys[:, br]).long())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % eval_every == 0:
            bler = evaluate_npd(model, N, Au, Av, frozen, 500)
            if bler < best_bler:
                best_bler = bler
            # Always save latest (Phase 1 needs checkpoint even at BLER=1.0
            # because per-position MI is still valid)
            torch.save({'state_dict': model.state_dict(),
                        'd': D, 'hidden': HIDDEN, 'n_layers': N_LAYERS, 'z_dim': 1,
                        'N': N, 'Au': Au, 'Av': Av}, ckpt_path)
            elapsed = (time.time() - t0) / 60
            print(f'    [{it:>6}/{iters}] loss={loss.item():.4f} BLER={bler:.4f} '
                  f'(best={best_bler:.4f}) {elapsed:.1f}min', flush=True)

    return best_bler, ckpt_path


def evaluate_npd(model, N, Au, Av, frozen_u, n_cw, seed=999):
    n = int(math.log2(N)); br = bit_reversal_perm(n)
    channel = GaussianMAC(sigma2=SIGMA2)
    model.eval()
    errs = 0; total = 0
    rng = np.random.default_rng(seed); np.random.seed(seed)
    with torch.no_grad():
        while total < n_cw:
            actual = min(32, n_cw - total)
            u_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Au: u_msg[:, p - 1] = rng.integers(0, 2, actual)
            x_phys = polar_encode_batch(u_msg.astype(int))
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Av: v_msg[:, p - 1] = rng.integers(0, 2, actual)
            y_phys = polar_encode_batch(v_msg.astype(int))
            z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
            emb = model.encode_channel(torch.from_numpy(z[:, br].astype(np.float32)).unsqueeze(-1))
            u_dec = model.decode(emb, frozen_u)
            for i in range(actual):
                if any(u_dec[i, p - 1].item() != u_msg[i, p - 1] for p in Au):
                    errs += 1
            total += actual
    model.train()
    return errs / n_cw


def measure_leaf_mi(model, N, Av, n_samples=20000):
    """Measure per-position leaf-level MI from trained model."""
    n = int(math.log2(N)); br = bit_reversal_perm(n)
    channel = GaussianMAC(sigma2=SIGMA2)
    model.eval()

    leaf_bce = np.zeros(N)
    count = 0
    rng = np.random.default_rng(123); np.random.seed(123)
    batch = min(100, n_samples)

    with torch.no_grad():
        while count < n_samples:
            actual = min(batch, n_samples - count)
            u_msg = rng.integers(0, 2, (actual, N)).astype(np.int8)
            x_phys = polar_encode_batch(u_msg.astype(int))
            v_msg = np.zeros((actual, N), dtype=np.int8)
            for p in Av: v_msg[:, p - 1] = rng.integers(0, 2, actual)
            y_phys = polar_encode_batch(v_msg.astype(int))
            z = channel.sample_batch(x_phys.astype(int), y_phys.astype(int))
            emb = model.encode_channel(torch.from_numpy(z[:, br].astype(np.float32)).unsqueeze(-1))
            B, N_, d = emb.shape

            # Walk tree to leaves
            V = [torch.from_numpy(x_phys[:, br]).long()]
            E = [emb]
            for depth in range(n):
                Vo, Ve, Eo, Ee = [], [], [], []
                for vc, ec in zip(V, E):
                    Vo.append(vc[:, 0::2]); Ve.append(vc[:, 1::2])
                    Eo.append(ec[:, 0::2, :]); Ee.append(ec[:, 1::2, :])
                Vo = torch.cat(Vo, 1); Ve = torch.cat(Ve, 1)
                Eo = torch.cat(Eo, 1); Ee = torch.cat(Ee, 1)
                vt = Vo ^ Ve; vb = Ve
                nc = 2 ** depth; cs = (N_ // 2) // nc
                vtc = torch.split(vt, cs, 1); vbc = torch.split(vb, cs, 1)
                Vn = []
                for a, b in zip(vtc, vbc): Vn += [a, b]
                Vl = torch.cat(Vn[0::2], 1)
                et = model.checknode(torch.cat([Eo, Ee], -1))
                eb = model.bitnode(Eo, Ee, Vl)
                etc = torch.split(et, cs, 1); ebc = torch.split(eb, cs, 1)
                En = []
                for a, b in zip(etc, ebc): En += [a, b]
                V = Vn; E = En

            e_leaves = torch.cat(E, 1); v_leaves = torch.cat(V, 1)
            logits = model.emb2llr(e_leaves).squeeze(-1)
            bce = F.binary_cross_entropy_with_logits(logits, v_leaves.float(), reduction='none')
            leaf_bce += bce.sum(0).numpy()
            count += actual

    avg_bce = leaf_bce / count
    # Map tree order → natural order
    bce_nat = np.zeros(N)
    for tidx in range(N):
        bce_nat[br[tidx]] = avg_bce[tidx]
    mi_nat = np.log(2) - bce_nat  # MI in nats
    model.train()
    return mi_nat, bce_nat


def evaluate_sc(N, Au, Av, n_cw=2000):
    """SC BLER for Stage 1 U on mixture channel."""
    n = int(math.log2(N))
    channel = GaussianMAC(sigma2=SIGMA2)
    fu_1idx = {i: 0 for i in range(1, N + 1) if i not in Au}
    errs = 0
    rng = np.random.default_rng(999); np.random.seed(999)
    for bi in range(0, n_cw, 200):
        actual = min(200, n_cw - bi)
        uf = np.zeros((actual, N), dtype=int); vf = np.zeros((actual, N), dtype=int)
        for p in Au: uf[:, p - 1] = rng.integers(0, 2, actual)
        for p in Av: vf[:, p - 1] = rng.integers(0, 2, actual)
        xf = polar_encode_batch(uf); yf = polar_encode_batch(vf)
        z = channel.sample_batch(xf, yf)
        for i in range(actual):
            llr = mixture_llr(z[i], SIGMA2)
            u_dec = _sc_decode_from_llr(llr, fu_1idx)
            if any(u_dec[p - 1] != uf[i, p - 1] for p in Au): errs += 1
    return errs / n_cw


def run_one_N(N, ku, Av_genie, warm_phase1=None, warm_phase3=None):
    """Full pipeline for one N."""
    n = int(math.log2(N))
    cfg = SCHEDULE[N]
    print(f'\n{"="*70}')
    print(f'N={N}, ku={ku}, rate={ku/N:.3f}')
    print(f'{"="*70}')

    # All U positions as info for Phase 1
    Au_all = list(range(1, N + 1))

    # Phase 1: Train all-info
    print(f'\n  Phase 1: Train all-info NPD ({cfg["phase1_iters"]} iters)')
    torch.manual_seed(42)
    model_p1 = NPDSingleUser(d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_dim=1,
                              use_analytical_training=False)
    _, p1_ckpt = train_npd(model_p1, N, Au_all, Av_genie,
                            iters=cfg['phase1_iters'], batch=cfg['batch'], lr=cfg['lr'],
                            warm_from=warm_phase1, tag=f'npd_design_p1_N{N}')

    # Phase 2: Measure MI
    print(f'\n  Phase 2: Measuring per-position MI ({cfg["mi_samples"]} samples)')
    model_p1.load_state_dict(torch.load(p1_ckpt, weights_only=False)['state_dict'])
    mi_nat, bce_nat = measure_leaf_mi(model_p1, N, Av_genie, cfg['mi_samples'])

    # Pick top ku positions
    sorted_pos = np.argsort(-mi_nat)
    npd_Au = sorted(int(sorted_pos[i]) + 1 for i in range(ku))
    npd_frozen = {p - 1 for p in range(1, N + 1) if p not in npd_Au}

    # Compare to genie
    genie_Au, _, _, _ = load_genie_design(n, ku, round(RATE_FRAC * IY_ZX * N))
    overlap = set(npd_Au) & set(genie_Au)
    print(f'  NPD-optimal Au: {npd_Au[:10]}{"..." if len(npd_Au)>10 else ""}')
    print(f'  Genie Au:       {sorted(genie_Au)[:10]}{"..." if len(genie_Au)>10 else ""}')
    print(f'  Overlap: {len(overlap)}/{ku}')
    print(f'  NPD avoids: {sorted(set(genie_Au) - set(npd_Au))}')
    print(f'  NPD picks:  {sorted(set(npd_Au) - set(genie_Au))}')

    # Phase 3: Retrain with NPD-optimal frozen set
    print(f'\n  Phase 3: Retrain with NPD-optimal design ({cfg["phase3_iters"]} iters)')
    torch.manual_seed(42)
    model_p3 = NPDSingleUser(d=D, hidden=HIDDEN, n_layers=N_LAYERS, z_dim=1,
                              use_analytical_training=False)
    npd_bler, p3_ckpt = train_npd(model_p3, N, npd_Au, Av_genie,
                                    iters=cfg['phase3_iters'], batch=cfg['batch'], lr=cfg['lr'],
                                    warm_from=warm_phase3, tag=f'npd_design_p3_N{N}')

    # Phase 4: Evaluate all combinations
    print(f'\n  Phase 4: Evaluation')
    # NPD-design + NPD
    model_p3.load_state_dict(torch.load(p3_ckpt, weights_only=False)['state_dict'])
    npd_bler_final = evaluate_npd(model_p3, N, npd_Au, Av_genie, npd_frozen, 2000)

    # NPD-design + SC
    sc_npd_design = evaluate_sc(N, npd_Au, Av_genie, 1000)

    # Genie-design + SC (reference)
    sc_genie = evaluate_sc(N, genie_Au, Av_genie, 1000)

    print(f'\n  Results:')
    print(f'    Genie design + SC:  {sc_genie:.4f}')
    print(f'    NPD design + NPD:  {npd_bler_final:.4f}')
    print(f'    NPD design + SC:   {sc_npd_design:.4f}')
    print(f'    Ratio NPD/SC:      {npd_bler_final/max(sc_genie, 1e-5):.2f}x')

    return {
        'N': N, 'ku': ku,
        'genie_Au': sorted(genie_Au), 'npd_Au': npd_Au,
        'overlap': len(overlap),
        'sc_genie': float(sc_genie),
        'npd_bler': float(npd_bler_final),
        'sc_npd_design': float(sc_npd_design),
        'ratio': float(npd_bler_final / max(sc_genie, 1e-5)),
        'p1_ckpt': p1_ckpt, 'p3_ckpt': p3_ckpt,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_list', type=str, default='16,32,64')
    args = parser.parse_args()
    N_list = [int(x) for x in args.N_list.split(',')]

    print('NPD-guided frozen set design sweep')
    print(f'Rate: {RATE_FRAC*100:.0f}% of per-user capacity')
    print(f'N values: {N_list}')
    print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # Get Av from genie design (V frozen set stays the same)
    _, Av_genie, _, _ = load_genie_design(int(math.log2(max(N_list))),
                                           round(RATE_FRAC * IX_Z * max(N_list)),
                                           round(RATE_FRAC * IY_ZX * max(N_list)))

    all_results = {}
    prev_p1 = None; prev_p3 = None
    t_total = time.time()

    for N in N_list:
        n = int(math.log2(N))
        ku = round(RATE_FRAC * IX_Z * N)
        kv = round(RATE_FRAC * IY_ZX * N)
        _, Av_this, _, _ = load_genie_design(n, ku, kv)

        result = run_one_N(N, ku, Av_this, warm_phase1=prev_p1, warm_phase3=prev_p3)
        all_results[N] = result
        prev_p1 = result['p1_ckpt']
        prev_p3 = result['p3_ckpt']

        # Save incrementally
        with open(os.path.join(RESULTS_DIR, 'npd_design_sweep.json'), 'w') as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2, default=str)

    # Final summary
    total_min = (time.time() - t_total) / 60
    print(f'\n{"="*70}')
    print('NPD-GUIDED DESIGN SWEEP — FINAL RESULTS')
    print(f'{"="*70}')
    print(f'{"N":<6}{"ku":<6}{"SC(genie)":<12}{"NPD(npd-design)":<18}{"ratio":<8}{"overlap":<10}')
    print('-' * 70)
    for N in sorted(all_results.keys()):
        r = all_results[N]
        print(f'{N:<6}{r["ku"]:<6}{r["sc_genie"]:<12.4f}{r["npd_bler"]:<18.4f}'
              f'{r["ratio"]:<8.2f}{r["overlap"]}/{r["ku"]}')
    print(f'\nTotal wall: {total_min:.1f} min')
    print(f'Finish: {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
