"""Local CPU NCG ISI chain: train N=16 → 32 → 64 → 128 with warm-starting.

Each N: train with grad clipping (no distill), eval at target rate, save ckpt.
Subsequent N reuses tree weights from previous N (channel-independent ops).
"""
import sys, os, time, json
sys.path.insert(0, "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2")
import numpy as np
import torch
import torch.nn.functional as F

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode_batch
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_isi_mac import ISIMACNeuralDecoder

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cpu")
SIGMA2 = 10**(-0.6)
ch = ISIMAC(sigma2=SIGMA2, h=0.3)

CKPT_DIR = "/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/scripts/local_analysis/ncg_models"
os.makedirs(CKPT_DIR, exist_ok=True)
RES_FILE = os.path.join(CKPT_DIR, "ncg_chain_results.json")
results = json.load(open(RES_FILE)) if os.path.exists(RES_FILE) else {}

# (N, ku, kv, n, iters, eval_cw)
CONFIGS = [
    (16,   4,   7, 4,  50_000, 5000),
    (32,   7,  15, 5,  80_000, 5000),
    (64,  15,  29, 6, 100_000, 5000),
    (128, 30,  58, 7, 120_000, 3000),
]
BATCH = 32
LR = 1e-3
GRAD_CLIP = 1.0

def gen_batch(B, N):
    rng = np.random.default_rng()
    u = rng.integers(0, 2, (B, N)).astype(np.int32)
    v = rng.integers(0, 2, (B, N)).astype(np.int32)
    x = polar_encode_batch(u); y = polar_encode_batch(v)
    z = ch.sample_batch(x, y).astype(np.float32)
    return (torch.from_numpy(z), torch.from_numpy(u).long(), torch.from_numpy(v).long())

prev_state = None
for (N, ku, kv, n, N_ITERS, n_cw) in CONFIGS:
    key = f"N{N}"
    ck_path = os.path.join(CKPT_DIR, f"ncg_isi_{key}.pt")
    if key in results and results[key].get("done"):
        print(f"  {key} done, skipping. BLER={results[key]['bler']:.4f}", flush=True)
        # load for warm-start
        prev_state = torch.load(ck_path, map_location="cpu", weights_only=False)["state_dict"]
        continue

    b = make_path(N, N)
    gmac_file = f"/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs/gmac_C_n{n}_snr6dB.npz"
    Au, Av, fu_set, fv_set, _, _, _ = design_from_file(gmac_file, n, ku, kv)
    Au = [int(p) for p in Au]; Av = [int(p) for p in Av]
    fu = {int(p): 0 for p in fu_set}; fv = {int(p): 0 for p in fv_set}

    print(f"\n=== Training NCG ISI N={N}, ku={ku}, kv={kv} ({N_ITERS} iters) ===", flush=True)
    model = ISIMACNeuralDecoder(d=16, hidden=64, n_layers=2, z_hidden=32, z_encoder_type='window').to(device)
    if prev_state is not None:
        # load only weights that match shapes
        own = model.state_dict()
        loaded = 0
        for k, w in prev_state.items():
            if k in own and own[k].shape == w.shape:
                own[k] = w; loaded += 1
        model.load_state_dict(own)
        print(f"  warm-started: {loaded} tensors from prev model", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time(); ce_losses = []
    for it in range(1, N_ITERS + 1):
        z_t, u_t, v_t = gen_batch(BATCH, N)
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
        if it % max(1, N_ITERS // 50) == 0:
            avg = float(np.mean(ce_losses[-200:]))
            print(f"  N={N} iter {it}/{N_ITERS}  ce={avg:.4f}  elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    # Eval
    print(f"  eval N={N} ({n_cw} CW)...", flush=True)
    model.eval()
    errs = 0; rng = np.random.default_rng(42 + N)
    teval0 = time.time()
    with torch.no_grad():
        for cw in range(n_cw):
            u_arr = np.zeros(N, dtype=np.int32); v_arr = np.zeros(N, dtype=np.int32)
            for p in Au: u_arr[p-1] = rng.integers(0, 2)
            for p in Av: v_arr[p-1] = rng.integers(0, 2)
            x = polar_encode_batch(u_arr.reshape(1,-1))[0]
            y = polar_encode_batch(v_arr.reshape(1,-1))[0]
            z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
            z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
            _, _, u_hat, v_hat, _ = model(z_t, b, frozen_u=fu, frozen_v=fv)
            ue = any(int(u_hat[p].item()) != int(u_arr[p-1]) for p in Au)
            ve = any(int(v_hat[p].item()) != int(v_arr[p-1]) for p in Av)
            if ue or ve: errs += 1
            if (cw+1) % max(1, n_cw // 5) == 0:
                print(f"    {cw+1}/{n_cw} errs={errs} ({(time.time()-teval0)/60:.1f}min)", flush=True)

    bler = errs / n_cw
    print(f"  >>> N={N} NCG BLER = {bler:.4f}  ({errs}/{n_cw})", flush=True)
    results[key] = dict(N=N, ku=ku, kv=kv, bler=bler, errs=errs, n_cw=n_cw,
                       iters=N_ITERS, done=True)
    json.dump(results, open(RES_FILE, "w"), indent=2)
    torch.save({"state_dict": model.state_dict(),
                "N": N, "ku": ku, "kv": kv}, ck_path)
    print(f"  saved {ck_path}", flush=True)
    prev_state = model.state_dict()

print("\n=== CHAIN DONE ===", flush=True)
for key, d in results.items():
    print(f"  {key}: NCG BLER = {d['bler']:.4f} ({d['errs']}/{d['n_cw']})", flush=True)
