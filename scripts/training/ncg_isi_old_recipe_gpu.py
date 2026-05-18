"""GPU port of the OLD NCG trainer (`ncg_isi_chain_local.py`) recipe.

Same: ISIMACNeuralDecoder, batch=32, lr=1e-3, fp32, grad clip 1.0, Adam.
Different: runs on CUDA, takes --N and --warm-start-ckpt as CLI args,
prints CE loss frequently for diagnostics.

POC mode: --iters 5000 to check if loss drops below 1.04 from warm-start.
Full run: --iters 150000 for production training.
"""
import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = "/gpfs0/bgu-haimp/users/eitansp/polar_project"
sys.path.insert(0, REPO_ROOT)

from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode_batch
from polar.design import make_path
from polar.design_mc import design_from_file
from neural.ncg_isi_mac import ISIMACNeuralDecoder

SIGMA2 = 10 ** (-0.6)
H = 0.3
RATES = {16:(4,7),32:(7,15),64:(15,29),128:(30,58),
         256:(59,117),512:(119,233),1024:(239,467)}
N_TO_N = {16:4,32:5,64:6,128:7,256:8,512:9,1024:10}


def gen_batch_gpu(B, N, ch, device):
    rng = np.random.default_rng()
    u = rng.integers(0, 2, (B, N)).astype(np.int32)
    v = rng.integers(0, 2, (B, N)).astype(np.int32)
    x = polar_encode_batch(u)
    y = polar_encode_batch(v)
    z = ch.sample_batch(x, y).astype(np.float32)
    return (torch.from_numpy(z).to(device),
            torch.from_numpy(u).long().to(device),
            torch.from_numpy(v).long().to(device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warm-start-ckpt", type=str, default=None)
    p.add_argument("--save-ckpt", type=str, default=None)
    p.add_argument("--save-every", type=int, default=0,
                   help="Save intermediate ckpt every N iters (0=off). Saves to <save-ckpt>.iter{N}.pt")
    p.add_argument("--eval-cw", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"device={device}  N={args.N}  iters={args.iters}  batch={args.batch}  lr={args.lr}", flush=True)
    torch.manual_seed(0); np.random.seed(0)

    N = args.N
    n = N_TO_N[N]
    ku, kv = RATES[N]
    ch = ISIMAC(sigma2=SIGMA2, h=H)
    b_path = make_path(N, N)

    gmac_file = f"{REPO_ROOT}/designs/gmac_C_n{n}_snr6dB.npz"
    Au, Av, fu_set, fv_set, _, _, _ = design_from_file(gmac_file, n, ku, kv)
    Au = [int(p) for p in Au]; Av = [int(p) for p in Av]
    fu = {int(p): 0 for p in fu_set}; fv = {int(p): 0 for p in fv_set}

    model = ISIMACNeuralDecoder(d=16, hidden=64, n_layers=2, z_hidden=32,
                                 z_encoder_type='window').to(device)

    if args.warm_start_ckpt:
        if not os.path.exists(args.warm_start_ckpt):
            raise FileNotFoundError(args.warm_start_ckpt)
        ck = torch.load(args.warm_start_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        own = model.state_dict()
        loaded = 0
        for k, w in sd.items():
            if k in own and own[k].shape == w.shape:
                own[k] = w; loaded += 1
        model.load_state_dict(own)
        print(f"  warm-started {loaded}/{len(own)} tensors from {args.warm_start_ckpt}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    GRAD_CLIP = 1.0

    ce_history = []
    t0 = time.time()
    model.train()
    for it in range(1, args.iters + 1):
        z_t, u_t, v_t = gen_batch_gpu(args.batch, N, ch, device)
        root = model.encode_z(z_t)
        all_logits, all_targets, _, _, _ = model.tree(
            z=None, b=b_path, frozen_u={}, frozen_v={},
            u_true=u_t, v_true=v_t, root_emb=root, distill_alpha=0.0)
        logits = torch.stack(all_logits, dim=1).reshape(-1, 4)
        targets = torch.stack(all_targets, dim=1).reshape(-1)
        ce = F.cross_entropy(logits, targets)
        opt.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        ce_history.append(ce.item())
        if it % args.log_every == 0:
            avg = float(np.mean(ce_history[-200:]))
            rate = it / (time.time() - t0)
            print(f"  it={it}/{args.iters}  ce={avg:.4f}  rate={rate:.2f}it/s  elapsed={(time.time()-t0)/60:.1f}min",
                  flush=True)
        # Intermediate save
        if args.save_every > 0 and args.save_ckpt and it % args.save_every == 0 and it < args.iters:
            inter_path = f"{args.save_ckpt}.iter{it}.pt"
            torch.save({"state_dict": model.state_dict(),
                        "N": N, "ku": ku, "kv": kv,
                        "iters": it, "ce_avg": float(np.mean(ce_history[-200:]))},
                       inter_path)
            print(f"    [saved intermediate {inter_path}]", flush=True)

    elapsed = time.time() - t0
    print(f"\n  training done: {args.iters} iters in {elapsed/60:.1f}min @ {args.iters/elapsed:.2f}it/s", flush=True)

    # Eval
    print(f"  eval at N={N} ({args.eval_cw} CW)...", flush=True)
    model.eval()
    errs = 0
    rng = np.random.default_rng(42 + N)
    teval0 = time.time()
    with torch.no_grad():
        for cw in range(args.eval_cw):
            u_arr = np.zeros(N, dtype=np.int32); v_arr = np.zeros(N, dtype=np.int32)
            for p in Au: u_arr[p-1] = rng.integers(0, 2)
            for p in Av: v_arr[p-1] = rng.integers(0, 2)
            x = polar_encode_batch(u_arr.reshape(1,-1))[0]
            y = polar_encode_batch(v_arr.reshape(1,-1))[0]
            z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
            z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)
            _, _, u_hat, v_hat, _ = model(z_t, b_path, frozen_u=fu, frozen_v=fv)
            ue = any(int(u_hat[p].item()) != int(u_arr[p-1]) for p in Au)
            ve = any(int(v_hat[p].item()) != int(v_arr[p-1]) for p in Av)
            if ue or ve: errs += 1
            if (cw+1) % max(1, args.eval_cw // 5) == 0:
                print(f"    {cw+1}/{args.eval_cw} errs={errs}  elapsed={(time.time()-teval0)/60:.1f}min", flush=True)

    bler = errs / args.eval_cw
    print(f"\n>>> N={N} NCG BLER = {bler:.5f} ({errs}/{args.eval_cw})", flush=True)

    if args.save_ckpt:
        torch.save({"state_dict": model.state_dict(),
                    "N": N, "ku": ku, "kv": kv,
                    "iters": args.iters, "bler": bler}, args.save_ckpt)
        print(f"  saved {args.save_ckpt}", flush=True)


if __name__ == "__main__":
    main()
