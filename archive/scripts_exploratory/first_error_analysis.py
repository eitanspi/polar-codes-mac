#!/usr/bin/env python3
"""
first_error_analysis.py — Find the first step at which the NCG N=256
decoder makes a mistake (no teacher forcing) on failed blocks.

Goal: see whether failures cluster at specific steps (indicating a
polarization-level capacity issue) or spread uniformly.
"""
import os, sys, math, time
import numpy as np
import torch

# Be nice to the training job also running on this machine
torch.set_num_threads(2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from polar.encoder import polar_encode_batch
from polar.channels import GaussianMAC
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.neural_scl import SimpleMLP_Gmac


def main():
    N = 256
    PATH_I = 128
    KU = KV = 123
    SNR_DB = 6.0
    SIGMA2 = 10 ** (-SNR_DB / 10)
    N_CW = 5000
    BATCH = 50
    SEED = 4242

    # ── model ──────────────────────────────────────────────────────────────
    model = SimpleMLP_Gmac(d=16, hidden=64, n_layers=2, z_hidden=32)
    ckpt_path = os.path.join(
        os.path.dirname(__file__), '..', 'saved_models', 'ncg_gmac_mlp_N256.pt'
    )
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    fixed = {(k.replace('z_enc.', 'z_encoder.') if k.startswith('z_enc.') else k): v
             for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    print(f"[load] ckpt={ckpt_path}")
    print(f"[load] missing keys: {len(missing)}  unexpected: {len(unexpected)}")
    if missing:
        print("      first missing:", missing[:5])
    if unexpected:
        print("      first unexpected:", unexpected[:5])
    model.eval()

    # ── design + path ───────────────────────────────────────────────────────
    n = int(math.log2(N))
    design_path = os.path.join(
        os.path.dirname(__file__), '..', 'designs',
        f'gmac_B_n{n}_snr6dB.npz'
    )
    Au, Av, fu, fv, _, _, _ = design_from_file(design_path, n, KU, KV)
    print(f"[design] {design_path}")
    print(f"[design] |Au|={len(Au)} |Av|={len(Av)} |fu|={len(fu)} |fv|={len(fv)}")
    b = make_path(N, PATH_I)
    assert len(b) == 2 * N

    # Sanity: counts of 0's and 1's in b
    assert sum(1 for x in b if x == 0) == N
    assert sum(1 for x in b if x == 1) == N

    channel = GaussianMAC(sigma2=SIGMA2)

    # ── run N_CW codewords, collect first-error stats ──────────────────────
    rng = np.random.default_rng(SEED)

    Au_set = set(Au)
    Av_set = set(Av)

    failed = 0
    total = 0
    # first-error step (1..2N) for each failed block
    first_err_step = []
    first_err_user = []  # 0 = U, 1 = V
    first_err_pos = []   # polar position 1..N within its user

    n_batches = (N_CW + BATCH - 1) // BATCH
    t0 = time.time()
    with torch.no_grad():
        for bi in range(n_batches):
            actual = min(BATCH, N_CW - total)
            uf = np.zeros((actual, N), dtype=int)
            vf = np.zeros((actual, N), dtype=int)
            for p in Au:
                uf[:, p - 1] = rng.integers(0, 2, actual)
            for p in Av:
                vf[:, p - 1] = rng.integers(0, 2, actual)
            xf = polar_encode_batch(uf)
            yf = polar_encode_batch(vf)
            zf = torch.from_numpy(channel.sample_batch(xf, yf)).float()

            _, _, u_hat, v_hat, _ = model(zf, b, fu, fv)

            # Extract hard decisions into (actual, N) arrays for quick compare
            # u_hat[p] is tensor (actual,) of floats {0,1}; same for v_hat
            uh_arr = np.zeros((actual, N), dtype=int)
            vh_arr = np.zeros((actual, N), dtype=int)
            for p in range(1, N + 1):
                if p in u_hat:
                    uh_arr[:, p - 1] = u_hat[p].cpu().numpy().astype(int)
                if p in v_hat:
                    vh_arr[:, p - 1] = v_hat[p].cpu().numpy().astype(int)

            # Replay the path for each codeword to find first mismatch step.
            # path b is the SAME across codewords (and across batches).
            # We can precompute the (i_t, user) sequence once.
            # But re-do it inline for clarity:
            i_u_counter = 0
            i_v_counter = 0
            step_info = []  # list of (step_index(1..2N), gamma, i_t)
            for step_idx, gamma in enumerate(b):
                if gamma == 0:
                    i_u_counter += 1
                    step_info.append((step_idx + 1, 0, i_u_counter))
                else:
                    i_v_counter += 1
                    step_info.append((step_idx + 1, 1, i_v_counter))

            for row in range(actual):
                bad_row = False
                first_bad_step = None
                first_bad_user = None
                first_bad_pos = None
                for (s, gamma, i_t) in step_info:
                    if gamma == 0:
                        # frozen U → bit is fixed 0, skip
                        if i_t in fu:
                            continue
                        true_bit = uf[row, i_t - 1]
                        dec_bit = uh_arr[row, i_t - 1]
                        if dec_bit != true_bit:
                            bad_row = True
                            first_bad_step = s
                            first_bad_user = 0
                            first_bad_pos = i_t
                            break
                    else:
                        if i_t in fv:
                            continue
                        true_bit = vf[row, i_t - 1]
                        dec_bit = vh_arr[row, i_t - 1]
                        if dec_bit != true_bit:
                            bad_row = True
                            first_bad_step = s
                            first_bad_user = 1
                            first_bad_pos = i_t
                            break

                if bad_row:
                    failed += 1
                    first_err_step.append(first_bad_step)
                    first_err_user.append(first_bad_user)
                    first_err_pos.append(first_bad_pos)

            total += actual
            if (bi + 1) % 10 == 0 or bi == n_batches - 1:
                elapsed = time.time() - t0
                rate = total / max(elapsed, 1e-9)
                print(f"  [batch {bi+1}/{n_batches}] cw {total}/{N_CW}  "
                      f"failed={failed}  BLER={failed/max(total,1):.4f}  "
                      f"rate={rate:.1f} cw/s")

    dt = time.time() - t0
    print(f"\n[done] {N_CW} codewords in {dt:.1f}s  failed={failed}  "
          f"BLER={failed/max(N_CW,1):.4f}")

    first_err_step = np.array(first_err_step)
    first_err_user = np.array(first_err_user)
    first_err_pos = np.array(first_err_pos)

    # ── region stats ────────────────────────────────────────────────────────
    # N=256, path_i=128
    # steps 1..128   = U positions 1..128
    # steps 129..384 = V positions 1..256
    # steps 385..512 = U positions 129..256
    r_u_early = int(((first_err_step >= 1) & (first_err_step <= 128)).sum())
    r_v_mid = int(((first_err_step >= 129) & (first_err_step <= 384)).sum())
    r_u_late = int(((first_err_step >= 385) & (first_err_step <= 512)).sum())

    print("\n=== REGION SUMMARY ===")
    print(f"Total codewords:  {N_CW}")
    print(f"Failed blocks:    {failed}")
    print(f"  U-early (steps 1-128,  U pos 1-128):   {r_u_early}")
    print(f"  V       (steps 129-384, V pos 1-256):  {r_v_mid}")
    print(f"  U-late  (steps 385-512, U pos 129-256):{r_u_late}")

    # ── 64-step bin histogram ──────────────────────────────────────────────
    edges = [1, 65, 129, 193, 257, 321, 385, 449, 513]
    labels = [f"{edges[i]}-{edges[i+1]-1}" for i in range(len(edges)-1)]
    counts = [
        int(((first_err_step >= edges[i]) & (first_err_step < edges[i+1])).sum())
        for i in range(len(edges)-1)
    ]
    print("\n=== HISTOGRAM (64-step bins) ===")
    for lab, c in zip(labels, counts):
        print(f"  steps {lab:>8s}: {c}")

    # ── Top-10 individual steps ─────────────────────────────────────────────
    print("\n=== TOP-10 SINGLE STEPS WITH MOST FIRST-ERRORS ===")
    uniq, cnt = np.unique(first_err_step, return_counts=True)
    order = np.argsort(-cnt)
    top = order[:10]

    # Try to load MI data
    mi_u = mi_v = None
    try:
        mi = np.load('/tmp/ncg_mi_n256.npz')
        mi_u = mi['mi_u']
        mi_v = mi['mi_v']
    except Exception as e:
        print(f"  (MI file not available: {e})")

    print(f"  {'step':>4s}  {'gamma':>5s}  {'pos':>4s}  {'count':>5s}  {'MI':>8s}")
    for k in top:
        s = int(uniq[k])
        c = int(cnt[k])
        # decode (gamma, pos) from b
        gamma = b[s - 1]
        # count how many 0's / 1's up to step s
        if gamma == 0:
            pos = sum(1 for x in b[:s] if x == 0)
            user = 'U'
            mi_val = mi_u[pos - 1] if mi_u is not None else float('nan')
        else:
            pos = sum(1 for x in b[:s] if x == 1)
            user = 'V'
            mi_val = mi_v[pos - 1] if mi_v is not None else float('nan')
        print(f"  {s:>4d}  {user:>5s}  {pos:>4d}  {c:>5d}  {mi_val:>8.4f}")

    # Dump raw data for plotting/inspection later
    out_path = '/tmp/first_err_n256.npz'
    np.savez(out_path,
             first_err_step=first_err_step,
             first_err_user=first_err_user,
             first_err_pos=first_err_pos,
             total=N_CW,
             failed=failed)
    print(f"\n[saved] {out_path}")


if __name__ == '__main__':
    main()
