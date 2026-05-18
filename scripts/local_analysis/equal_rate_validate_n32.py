"""Equal-rate validation at N=32, path a*=19.

Design: k_U = k_V = 9 → equal rate 9/32 = 0.281 bits/use per user (sum 0.563).
Budget far below 0.694 equal-rate capacity → expect very low BLER for both
decoders, confirming NCG handles non-corner paths.

Decoders:
  1. SCT (analytical, decoder_trellis.decode_single) on path b
  2. NCG (trained on path b)
"""
import sys, os, time, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np
import torch
import torch.nn.functional as F

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode_batch
from polar.design import make_path
from polar.decoder_trellis import decode_single
from neural.ncg_isi_mac import ISIMACNeuralDecoder

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cpu")

SIGMA2 = 10**(-0.6); H = 0.3
ch = ISIMAC(sigma2=SIGMA2, h=H)
N = 32; PATH_A = 19
K_U = 9; K_V = 9

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
RES_FILE = os.path.join(OUT_DIR, "equal_rate_validate_n32.json")
CKPT = os.path.join(OUT_DIR, "ncg_models", f"ncg_isi_equal_rate_n32.pt")

NCG_ITERS = 40_000     # fewer since warm-starting from corner-path N=32 ckpt
NCG_BATCH = 32
NCG_LR = 1e-3
GRAD_CLIP = 1.0
WARM_START_CKPT = os.path.join(OUT_DIR, "ncg_models", "ncg_isi_N32.pt")

N_EVAL_SCT = 5000   # analytical is fast
N_EVAL_NCG = 5000

results = json.load(open(RES_FILE)) if os.path.exists(RES_FILE) else {}


def select_info_set(p_err, k):
    """Pick best k positions by error rate (1-indexed)."""
    sorted_p = sorted([(p, i + 1) for i, p in enumerate(p_err)])
    info = sorted([pos for _, pos in sorted_p[:k]])
    frozen = {i: 0 for i in range(1, len(p_err) + 1) if i not in info}
    return info, frozen


def gen_batch(B, N, b, Au, Av, rng):
    """Encode random messages; return (z, u, v, x, y) numpy."""
    u = rng.integers(0, 2, (B, N)).astype(np.int32)
    v = rng.integers(0, 2, (B, N)).astype(np.int32)
    x = polar_encode_batch(u); y = polar_encode_batch(v)
    z = ch.sample_batch(x, y).astype(np.float32)
    return z, u, v


def main():
    print(f"=== Equal-rate validation N={N}, path a={PATH_A}, k_U=k_V={K_U} ===")
    print(f"channel: ISI-MAC h={H}, sigma2={SIGMA2:.4f}, SNR=6 dB")

    # ── design from MC P_e at a=19 ────────────────────────────────────
    search = json.load(open(os.path.join(OUT_DIR, "equal_rate_search_results.json")))
    cand = search['candidates'][str(PATH_A)]
    peU = np.array(cand['peU']); peV = np.array(cand['peV'])

    Au, fu = select_info_set(peU, K_U)
    Av, fv = select_info_set(peV, K_V)
    print(f"\nA_U (info) = {Au}")
    print(f"A_V (info) = {Av}")
    print(f"max Pe in A_U = {max(peU[a-1] for a in Au):.4f}")
    print(f"max Pe in A_V = {max(peV[a-1] for a in Av):.4f}")

    b = make_path(N, PATH_A)

    # ── SCT eval ──────────────────────────────────────────────────────
    if 'sct' not in results or not results['sct'].get('done'):
        print(f"\n--- SCT analytical eval ({N_EVAL_SCT} CW) ---")
        rng = np.random.default_rng(11)
        errs = 0; t0 = time.time()
        for cw in range(N_EVAL_SCT):
            u = np.zeros(N, dtype=np.int32); v = np.zeros(N, dtype=np.int32)
            for p in Au: u[p-1] = rng.integers(0, 2)
            for p in Av: v[p-1] = rng.integers(0, 2)
            x = polar_encode_batch(u.reshape(1,-1))[0]
            y = polar_encode_batch(v.reshape(1,-1))[0]
            z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
            u_hat, v_hat = decode_single(N, list(z), b, fu, fv, ch)
            ue = any(int(u_hat[p-1]) != int(u[p-1]) for p in Au)
            ve = any(int(v_hat[p-1]) != int(v[p-1]) for p in Av)
            if ue or ve: errs += 1
            if (cw + 1) % max(1, N_EVAL_SCT // 10) == 0:
                print(f"  SCT {cw+1}/{N_EVAL_SCT} errs={errs} "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)
        bler_sct = errs / N_EVAL_SCT
        print(f"  SCT BLER = {bler_sct:.4f} ({errs}/{N_EVAL_SCT})")
        results['sct'] = dict(bler=bler_sct, errs=int(errs), n_cw=int(N_EVAL_SCT), done=True)
        json.dump(results, open(RES_FILE, "w"), indent=2)
    else:
        print(f"\nSCT cached: BLER={results['sct']['bler']:.4f}")

    # ── NCG training ──────────────────────────────────────────────────
    if 'ncg' not in results or not results['ncg'].get('done'):
        print(f"\n--- NCG training ({NCG_ITERS} iters, warm-start from corner-path N=32) ---")
        model = ISIMACNeuralDecoder(d=16, hidden=64, n_layers=2,
                                     z_hidden=32, z_encoder_type='window').to(device)
        if os.path.exists(WARM_START_CKPT):
            sd = torch.load(WARM_START_CKPT, map_location='cpu', weights_only=False)
            sd = sd.get('state_dict', sd)
            own = model.state_dict(); loaded = 0
            for k, w in sd.items():
                if k in own and own[k].shape == w.shape:
                    own[k] = w; loaded += 1
            model.load_state_dict(own)
            print(f"  warm-started: {loaded} tensors from {WARM_START_CKPT}")
        opt = torch.optim.Adam(model.parameters(), lr=NCG_LR)
        t0 = time.time(); ce_losses = []
        rng = np.random.default_rng(123)
        for it in range(1, NCG_ITERS + 1):
            z_t, u_t, v_t = gen_batch(NCG_BATCH, N, b, Au, Av, rng)
            z_t = torch.from_numpy(z_t)
            u_t = torch.from_numpy(u_t).long()
            v_t = torch.from_numpy(v_t).long()
            root = model.encode_z(z_t)
            all_logits, all_targets, _, _, _ = model.tree(
                z=None, b=b, frozen_u={}, frozen_v={},
                u_true=u_t, v_true=v_t, root_emb=root, distill_alpha=0.0)
            logits = torch.stack(all_logits, dim=1).reshape(-1, 4)
            targets = torch.stack(all_targets, dim=1).reshape(-1)
            ce = F.cross_entropy(logits, targets)
            opt.zero_grad(); ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ce_losses.append(ce.item())
            if it % max(1, NCG_ITERS // 50) == 0:
                avg = float(np.mean(ce_losses[-200:]))
                print(f"  iter {it}/{NCG_ITERS}  ce={avg:.4f}  "
                      f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
        torch.save({"state_dict": model.state_dict(),
                    "N": N, "a": PATH_A, "ku": K_U, "kv": K_V}, CKPT)

        # NCG eval
        print(f"\n--- NCG eval ({N_EVAL_NCG} CW) ---")
        model.eval()
        rng = np.random.default_rng(42 + N)
        errs = 0; teval0 = time.time()
        with torch.no_grad():
            for cw in range(N_EVAL_NCG):
                u = np.zeros(N, dtype=np.int32); v = np.zeros(N, dtype=np.int32)
                for p in Au: u[p-1] = rng.integers(0, 2)
                for p in Av: v[p-1] = rng.integers(0, 2)
                x = polar_encode_batch(u.reshape(1,-1))[0]
                y = polar_encode_batch(v.reshape(1,-1))[0]
                z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
                z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
                _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u=fu, frozen_v=fv)
                ue = any(int(u_hat[p].item()) != int(u[p-1]) for p in Au)
                ve = any(int(v_hat[p].item()) != int(v[p-1]) for p in Av)
                if ue or ve: errs += 1
                if (cw + 1) % max(1, N_EVAL_NCG // 10) == 0:
                    print(f"  NCG {cw+1}/{N_EVAL_NCG} errs={errs} "
                          f"({(time.time()-teval0)/60:.1f}min)", flush=True)
        bler_ncg = errs / N_EVAL_NCG
        print(f"  NCG BLER = {bler_ncg:.4f} ({errs}/{N_EVAL_NCG})")
        results['ncg'] = dict(bler=bler_ncg, errs=int(errs), n_cw=int(N_EVAL_NCG),
                              iters=NCG_ITERS, done=True)
        json.dump(results, open(RES_FILE, "w"), indent=2)

    print("\n=== SUMMARY ===")
    print(f"  SCT (analytical):  BLER = {results['sct']['bler']:.4f}  "
          f"({results['sct']['errs']}/{results['sct']['n_cw']})")
    print(f"  NCG (neural):      BLER = {results['ncg']['bler']:.4f}  "
          f"({results['ncg']['errs']}/{results['ncg']['n_cw']})")
    print(f"  rate budget: R_U=R_V={K_U}/{N}={K_U/N:.4f}  "
          f"(equal-rate capacity = 0.694)")


if __name__ == "__main__":
    main()
