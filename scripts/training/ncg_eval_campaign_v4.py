"""NCG eval campaign v4 (cluster GPU, ~6 hr).

Phase 1: Tight BLER eval of N=128 NCG corner ckpt at ALL N (16..1024), 30K CW each.
Phase 2: Cross-N universality - N=64 NCG ckpt eval at all N.
Phase 3: Equal-rate truncation - N=64 equal-rate NCG ckpt eval at larger N.
Phase 4: Low-LR training POC at N=256 (does it preserve the model?).
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, "/gpfs0/bgu-haimp/users/eitansp/polar_project")
import torch
from scripts.training.ncg_isi_gpu_curriculum import _make_model, GPUDataGen, eval_bler
from polar.design import make_path
from polar.design_mc import design_from_file

device = torch.device("cuda")
SIGMA2 = 10 ** (-0.6)
H = 0.3
CONFIGS = {16:(4,7,4), 32:(7,15,5), 64:(15,29,6), 128:(30,58,7),
           256:(59,117,8), 512:(119,233,9), 1024:(239,467,10)}
OUT_PATH = "/gpfs0/bgu-haimp/users/eitansp/polar_project/ncg_v4_results.json"
LOG_PATH = "/gpfs0/bgu-haimp/users/eitansp/polar_project/ncg_v4.log"
NCW = {16:30000, 32:30000, 64:30000, 128:30000, 256:20000, 512:10000, 1024:5000}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt


def make_loaded_model(sd):
    m = _make_model(device)
    own = m.state_dict()
    loaded = 0
    for k, w in sd.items():
        if k in own and own[k].shape == w.shape:
            own[k] = w
            loaded += 1
    m.load_state_dict(own)
    return m, loaded


def eval_at_N(model, N, ku, kv, n, n_cw=10000, b_path=None,
              Au_override=None, Av_override=None):
    if Au_override is not None and Av_override is not None:
        Au = list(Au_override)
        Av = list(Av_override)
        fu = {p: 0 for p in range(1, N + 1) if p not in set(Au)}
        fv = {p: 0 for p in range(1, N + 1) if p not in set(Av)}
    else:
        Au_a, Av_a, fu_set, fv_set, _, _, _ = design_from_file(
            f"/gpfs0/bgu-haimp/users/eitansp/polar_project/designs/gmac_C_n{n}_snr6dB.npz",
            n, ku, kv)
        Au = [int(p) for p in Au_a]
        Av = [int(p) for p in Av_a]
        fu = {int(p): 0 for p in fu_set}
        fv = {int(p): 0 for p in fv_set}
    if b_path is None:
        b_path = make_path(N, N)
    datagen = GPUDataGen(N, sigma2=SIGMA2, h=H, device=device)
    bs = 128 if N <= 256 else (64 if N <= 512 else 16)
    t0 = time.time()
    bler, errs = eval_bler(model, N, Au, Av, fu, fv, b_path, datagen, n_cw, device, batch_eval=bs)
    el = time.time() - t0
    return bler, errs, n_cw, el


def phase1(out):
    log("PHASE 1: N=128 NCG ckpt at all N, high CW")
    key = "phase1_n128_truncated"
    if key not in out:
        out[key] = {}
    sd = load_ckpt("/gpfs0/bgu-haimp/users/eitansp/polar_project/scripts/local_analysis/ncg_models/ncg_isi_N128.pt")
    model, loaded = make_loaded_model(sd)
    log(f"  loaded {loaded}/42 tensors")
    model.eval()
    for N, (ku, kv, n) in CONFIGS.items():
        k = str(N)
        if k in out[key] and out[key][k].get("done"):
            continue
        log(f"  N={N} ({NCW[N]} CW)...")
        bler, errs, cw, el = eval_at_N(model, N, ku, kv, n, n_cw=NCW[N])
        log(f"  N={N} BLER={bler:.5f} ({errs}/{cw}) {el:.0f}s")
        out[key][k] = dict(N=N, ku=ku, kv=kv, bler=bler, errs=errs, n_cw=cw, elapsed_s=el, done=True)
        json.dump(out, open(OUT_PATH, "w"), indent=2)


def phase2(out):
    log("\nPHASE 2: N=64 NCG ckpt at all N (universality)")
    key = "phase2_n64_truncated"
    if key not in out:
        out[key] = {}
    path = "/gpfs0/bgu-haimp/users/eitansp/polar_project/scripts/local_analysis/ncg_models/ncg_isi_N64.pt"
    if not os.path.exists(path):
        log(f"  N=64 ckpt not found at {path}")
        return
    sd = load_ckpt(path)
    model, loaded = make_loaded_model(sd)
    log(f"  loaded {loaded}/42 tensors from N=64 ckpt")
    model.eval()
    for N, (ku, kv, n) in CONFIGS.items():
        k = str(N)
        if k in out[key] and out[key][k].get("done"):
            continue
        log(f"  N={N} ({NCW[N]} CW)...")
        bler, errs, cw, el = eval_at_N(model, N, ku, kv, n, n_cw=NCW[N])
        log(f"  N={N} BLER={bler:.5f} ({errs}/{cw}) {el:.0f}s")
        out[key][k] = dict(N=N, ku=ku, kv=kv, bler=bler, errs=errs, n_cw=cw, elapsed_s=el, done=True)
        json.dump(out, open(OUT_PATH, "w"), indent=2)


def phase3(out):
    log("\nPHASE 3: N=64 equal-rate NCG ckpt truncation")
    key = "phase3_n64_equal_rate"
    if key not in out:
        out[key] = {}
    path = "/gpfs0/bgu-haimp/users/eitansp/polar_project/scripts/local_analysis/ncg_models/ncg_isi_equal_rate_n64.pt"
    if not os.path.exists(path):
        log(f"  equal-rate ckpt not found at {path}")
        return
    sd = load_ckpt(path)
    model, loaded = make_loaded_model(sd)
    log(f"  loaded {loaded}/42 tensors from N=64 equal-rate ckpt")
    model.eval()
    # Equal-rate paths: a* = 0.594*N (from session 11)
    EQ = {64: (38, 18), 128: (76, 36), 256: (150, 72), 512: (298, 144)}
    for N in [64, 128, 256, 512]:
        k = str(N)
        if k in out[key] and out[key][k].get("done"):
            continue
        a, ku = EQ[N]
        kv = ku  # equal rate
        b_path = [0] * a + [1] * N + [0] * (N - a)
        n_cw = min(NCW.get(N, 5000), 5000)
        n = int(np.log2(N))
        log(f"  N={N} a={a} ku=kv={ku} ({n_cw} CW)...")
        try:
            bler, errs, cw, el = eval_at_N(model, N, ku, kv, n, n_cw=n_cw, b_path=b_path)
            log(f"  N={N} equal-rate BLER={bler:.5f} ({errs}/{cw}) {el:.0f}s")
            out[key][k] = dict(N=N, a=a, ku=ku, kv=kv, bler=bler, errs=errs, n_cw=cw,
                               elapsed_s=el, done=True)
        except Exception as e:
            log(f"  N={N} equal-rate FAILED: {type(e).__name__}: {e}")
            out[key][k] = dict(error=str(e))
        json.dump(out, open(OUT_PATH, "w"), indent=2)


def phase4(out):
    """Low-LR training POC: does training preserve the model?"""
    log("\nPHASE 4: Low-LR training POC at N=256")
    key = "phase4_lowlr_training"
    if key not in out:
        out[key] = {}
    # Load reference loss helper from the trainer if available
    try:
        from scripts.training.ncg_isi_gpu_curriculum import train_loop
        has_train_loop = True
    except Exception:
        has_train_loop = False
        log(f"  cannot import train_loop; skip phase 4")
        return

    base_sd = load_ckpt("/gpfs0/bgu-haimp/users/eitansp/polar_project/scripts/local_analysis/ncg_models/ncg_isi_N128.pt")
    for lr_val in [1e-5, 1e-4, 5e-4]:
        k = f"lr_{lr_val}"
        if k in out[key] and out[key][k].get("done"):
            continue
        model, loaded = make_loaded_model(base_sd)
        log(f"  lr={lr_val}: pre-train eval at N=256...")
        bler_pre, errs_pre, _, _ = eval_at_N(model, 256, 59, 117, 8, n_cw=2000)
        log(f"    pre BLER={bler_pre:.5f} ({errs_pre}/2000)")
        # Train 2K iters at N=256 with this LR
        model.train()
        N = 256
        b_path = make_path(N, N)
        datagen = GPUDataGen(N, sigma2=SIGMA2, h=H, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=lr_val)
        log(f"  lr={lr_val}: training 2000 iters at N=256 (batch 512)...")
        t0 = time.time()
        try:
            ce_losses, elapsed = train_loop(
                model, datagen, b_path, opt, 2000, device,
                amp=True, log_every=200, label=f"lr={lr_val} ")
        except TypeError:
            # Older signature
            ce_losses, elapsed = train_loop(
                model, datagen, b_path, opt, 2000, device,
                log_every=200, label=f"lr={lr_val} ")
        log(f"  lr={lr_val}: done in {elapsed:.0f}s; eval at N=256...")
        model.eval()
        bler_post, errs_post, _, _ = eval_at_N(model, 256, 59, 117, 8, n_cw=2000)
        log(f"    post BLER={bler_post:.5f} ({errs_post}/2000)")
        out[key][k] = dict(lr=lr_val, bler_pre=bler_pre, bler_post=bler_post,
                           errs_pre=errs_pre, errs_post=errs_post,
                           train_s=elapsed, done=True)
        json.dump(out, open(OUT_PATH, "w"), indent=2)


def main():
    out = {}
    if os.path.exists(OUT_PATH):
        try:
            out = json.load(open(OUT_PATH))
        except Exception:
            pass
    log("============================================")
    log("NCG v4 EVAL CAMPAIGN starting")
    log("============================================")
    t0 = time.time()
    try: phase1(out)
    except Exception as e: log(f"PHASE 1 FAILED: {e}")
    try: phase2(out)
    except Exception as e: log(f"PHASE 2 FAILED: {e}")
    try: phase3(out)
    except Exception as e: log(f"PHASE 3 FAILED: {e}")
    try: phase4(out)
    except Exception as e: log(f"PHASE 4 FAILED: {e}")
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
