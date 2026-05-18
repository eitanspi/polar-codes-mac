"""Equal-rate validation at N=64, path a*=38, k_U=k_V=18.
Per-user rate 18/64 = 0.281 (matches N=32 k=9 design).
Warm-starts NCG from corner-path N=64 ckpt.
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
N = 64; PATH_A = 38; K_U = 18; K_V = 18

OUT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis"
RES_FILE = os.path.join(OUT_DIR, "equal_rate_validate_n64.json")
CKPT = os.path.join(OUT_DIR, "ncg_models", "ncg_isi_equal_rate_n64.pt")
WARM_START_CORNER = os.path.join(OUT_DIR, "ncg_models", "ncg_isi_N64.pt")
WARM_START_N32_EQRATE = os.path.join(OUT_DIR, "ncg_models", "ncg_isi_equal_rate_n32.pt")

NCG_ITERS = 50_000   # warm-starting, larger N -> need more
NCG_BATCH = 16
NCG_LR = 1e-3
GRAD_CLIP = 1.0
N_EVAL = 3000


def select_info_set(p_err, k):
    sorted_p = sorted([(p, i + 1) for i, p in enumerate(p_err)])
    info = sorted([pos for _, pos in sorted_p[:k]])
    frozen = {i: 0 for i in range(1, len(p_err) + 1) if i not in info}
    return info, frozen


def gen_batch(B, N, rng):
    u = rng.integers(0, 2, (B, N)).astype(np.int32)
    v = rng.integers(0, 2, (B, N)).astype(np.int32)
    x = polar_encode_batch(u); y = polar_encode_batch(v)
    z = ch.sample_batch(x, y).astype(np.float32)
    return z, u, v


def main():
    print(f"=== Equal-rate validation N={N}, a={PATH_A}, k_U=k_V={K_U} ===")
    results = json.load(open(RES_FILE)) if os.path.exists(RES_FILE) else {}

    search = json.load(open(os.path.join(OUT_DIR, "equal_rate_search_n64.json")))
    cand = search['candidates'][str(PATH_A)]
    peU = np.array(cand['peU']); peV = np.array(cand['peV'])

    Au, fu = select_info_set(peU, K_U)
    Av, fv = select_info_set(peV, K_V)
    print(f"max Pe in A_U = {max(peU[a-1] for a in Au):.4f}")
    print(f"max Pe in A_V = {max(peV[a-1] for a in Av):.4f}")
    b = make_path(N, PATH_A)

    # ── SCT ────────────────────────────────────────────────────────────
    if 'sct' not in results or not results['sct'].get('done'):
        print(f"\n--- SCT eval ({N_EVAL} CW) ---")
        rng = np.random.default_rng(11)
        errs = 0; t0 = time.time()
        for cw in range(N_EVAL):
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
            if (cw + 1) % max(1, N_EVAL // 10) == 0:
                print(f"  SCT {cw+1}/{N_EVAL} errs={errs} "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)
        bler_sct = errs / N_EVAL
        print(f"  SCT BLER = {bler_sct:.4f} ({errs}/{N_EVAL})")
        results['sct'] = dict(bler=bler_sct, errs=int(errs), n_cw=int(N_EVAL), done=True)
        json.dump(results, open(RES_FILE, "w"), indent=2)
    else:
        print(f"\nSCT cached BLER={results['sct']['bler']:.4f}")

    # ── NCG ────────────────────────────────────────────────────────────
    if 'ncg' not in results or not results['ncg'].get('done'):
        print(f"\n--- NCG training ({NCG_ITERS} iters) ---")
        model = ISIMACNeuralDecoder(d=16, hidden=64, n_layers=2,
                                     z_hidden=32, z_encoder_type='window').to(device)
        warm_ckpt = (WARM_START_N32_EQRATE if os.path.exists(WARM_START_N32_EQRATE)
                     else WARM_START_CORNER)
        if os.path.exists(warm_ckpt):
            sd = torch.load(warm_ckpt, map_location='cpu', weights_only=False)
            sd = sd.get('state_dict', sd)
            own = model.state_dict(); loaded = 0
            for k, w in sd.items():
                if k in own and own[k].shape == w.shape:
                    own[k] = w; loaded += 1
            model.load_state_dict(own)
            print(f"  warm-started {loaded} tensors from {os.path.basename(warm_ckpt)}")
        opt = torch.optim.Adam(model.parameters(), lr=NCG_LR)
        t0 = time.time(); ce_losses = []
        rng = np.random.default_rng(123)
        for it in range(1, NCG_ITERS + 1):
            z_t, u_t, v_t = gen_batch(NCG_BATCH, N, rng)
            z_t = torch.from_numpy(z_t)
            u_t = torch.from_numpy(u_t).long(); v_t = torch.from_numpy(v_t).long()
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

        print(f"\n--- NCG eval ({N_EVAL} CW) ---")
        model.eval()
        rng = np.random.default_rng(42 + N)
        errs = 0; t0 = time.time()
        with torch.no_grad():
            for cw in range(N_EVAL):
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
                if (cw + 1) % max(1, N_EVAL // 10) == 0:
                    print(f"  NCG {cw+1}/{N_EVAL} errs={errs} "
                          f"({(time.time()-t0)/60:.1f}min)", flush=True)
        bler_ncg = errs / N_EVAL
        print(f"  NCG BLER = {bler_ncg:.4f} ({errs}/{N_EVAL})")
        results['ncg'] = dict(bler=bler_ncg, errs=int(errs), n_cw=int(N_EVAL),
                              iters=NCG_ITERS, done=True)
        json.dump(results, open(RES_FILE, "w"), indent=2)

    print("\n=== SUMMARY ===")
    print(f"  SCT BLER = {results['sct']['bler']:.4f}  ({results['sct']['errs']}/{results['sct']['n_cw']})")
    print(f"  NCG BLER = {results['ncg']['bler']:.4f}  ({results['ncg']['errs']}/{results['ncg']['n_cw']})")


if __name__ == "__main__":
    main()
