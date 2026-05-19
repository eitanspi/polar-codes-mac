"""Hour-4 followup: SCT N=1024 eval-only with extra CW + NCG N=1024 latest eval.

Reuses the existing N=1024 SCT design from chained_sct_n1024_200k.json
(no need to redo the 50min design) — just runs more eval CW.

Pulls latest NCG N=1024 ckpt from cluster and evaluates locally.
"""
import os, sys, json, time, subprocess
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
if _HERE not in sys.path: sys.path.insert(0, _HERE)

LOG_DIR = "/tmp/hour4_followup"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "log.txt")
OUT_PATH = os.path.join(_HERE, "hour4_followup_results.json")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")


def sct_n1024_eval_only():
    """Reuse the existing N=1024 design (200K trials) — run more eval CW."""
    from chained_sct_4state import eval_at, RATES
    design_json = os.path.join(_HERE, "chained_sct_n1024_200k.json")
    d = json.load(open(design_json))
    Au = d["Au"]; Av = d["Av"]
    N = 1024
    ku, kv = RATES[N]
    log(f"SCT N={N} eval-only with existing design (200K trials)")
    log(f"  prior eval: 2 errs / 50000 CW (BLER 4.0e-5)")
    # 250K more CW; combined with prior 50K → 300K total
    n_more = 250000
    log(f"  running {n_more} more eval CW...")
    t0 = time.time()
    r = eval_at(N, n_more, Au, Av, 7, base_seed=1024_888)
    te = time.time() - t0
    new_errs = r["errs"]; new_cw = r["n_cw"]
    # Combine with prior
    total_errs = 2 + new_errs
    total_cw = 50000 + new_cw
    combined_bler = total_errs / total_cw
    log(f"  new alone: BLER={r['bler']:.6f} ({new_errs}/{new_cw}) in {te/60:.1f}min")
    log(f"  COMBINED (prior+new): BLER={combined_bler:.6f} ({total_errs}/{total_cw})")
    return dict(N=N, ku=ku, kv=kv,
                new_errs=int(new_errs), new_cw=int(new_cw),
                prior_errs=2, prior_cw=50000,
                total_errs=int(total_errs), total_cw=int(total_cw),
                combined_bler=combined_bler, eval_time_s=te)


def ncg_n1024_eval_latest():
    """Pull latest N=1024 ckpt from cluster, eval locally."""
    log("Pulling latest NCG N=1024 ckpt from cluster...")
    # First get the latest iter file
    cmd_ls = "ls /gpfs0/bgu-haimp/users/eitansp/polar_project/ncg_old_recipe_ckpts/ | grep n1024 | grep iter | sort -t'r' -k4 -n | tail -1"
    out = subprocess.check_output(["ssh", "bhn20", cmd_ls], text=True).strip()
    log(f"  latest: {out}")
    local_path = os.path.join("/tmp/ncg_intermediate", out)
    subprocess.run(["scp", f"bhn20:/gpfs0/bgu-haimp/users/eitansp/polar_project/ncg_old_recipe_ckpts/{out}",
                    local_path], check=True)

    # Eval at N=1024, 2K CW
    log(f"  eval N=1024, 2000 CW (single thread)...")
    import torch
    torch.set_num_threads(1)
    from polar.channels_memory import ISIMAC
    from polar.encoder import polar_encode_batch
    from polar.design import make_path
    from polar.design_mc import design_from_file
    from neural.ncg_isi_mac import ISIMACNeuralDecoder
    SIGMA2 = 10**(-0.6); H = 0.3
    N = 1024; n = 10; ku = 239; kv = 467
    ch = ISIMAC(sigma2=SIGMA2, h=H)
    b_path = make_path(N, N)
    gmac_file = f"/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs/gmac_C_n{n}_snr6dB.npz"
    Au, Av, fu_set, fv_set, _, _, _ = design_from_file(gmac_file, n, ku, kv)
    Au = [int(p) for p in Au]; Av = [int(p) for p in Av]
    fu = {int(p): 0 for p in fu_set}; fv = {int(p): 0 for p in fv_set}
    model = ISIMACNeuralDecoder(d=16, hidden=64, n_layers=2, z_hidden=32, z_encoder_type='window')
    ck = torch.load(local_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    own = model.state_dict()
    for k, w in sd.items():
        if k in own and own[k].shape == w.shape: own[k] = w
    model.load_state_dict(own); model.eval()
    iters_done = ck.get("iters", "?")
    log(f"  ckpt iters: {iters_done}")

    n_cw = 2000
    rng = np.random.default_rng(1024_111)
    errs = 0
    t0 = time.time()
    with torch.no_grad():
        for cw in range(n_cw):
            u_arr = np.zeros(N, dtype=np.int32); v_arr = np.zeros(N, dtype=np.int32)
            for p in Au: u_arr[p-1] = rng.integers(0, 2)
            for p in Av: v_arr[p-1] = rng.integers(0, 2)
            x = polar_encode_batch(u_arr.reshape(1,-1))[0]
            y = polar_encode_batch(v_arr.reshape(1,-1))[0]
            z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
            z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0)
            _, _, u_hat, v_hat, _ = model(z_t, b_path, frozen_u=fu, frozen_v=fv)
            ue = any(int(u_hat[p].item()) != int(u_arr[p-1]) for p in Au)
            ve = any(int(v_hat[p].item()) != int(v_arr[p-1]) for p in Av)
            if ue or ve: errs += 1
            if (cw+1) % 200 == 0:
                log(f"    {cw+1}/{n_cw} errs={errs} ({(time.time()-t0)/60:.1f}min)")
    te = time.time() - t0
    bler = errs / n_cw
    log(f"  NCG N=1024 ckpt={out} BLER={bler:.6f} ({errs}/{n_cw}) in {te/60:.1f}min")
    return dict(N=N, ckpt=out, iters_done=iters_done,
                n_cw=n_cw, errs=errs, bler=bler, eval_time_s=te)


def main():
    out = {}
    if os.path.exists(OUT_PATH):
        try: out = json.load(open(OUT_PATH))
        except: out = {}
    log("============================================")
    log("HOUR-4 FOLLOWUP: SCT N=1024 + NCG N=1024 eval")
    log("============================================")
    # SCT N=1024 first since it uses the multiprocessing pool (heaviest)
    if "sct_n1024_extra" not in out:
        try:
            out["sct_n1024_extra"] = sct_n1024_eval_only()
            with open(OUT_PATH, "w") as f: json.dump(out, f, indent=2)
        except Exception as e:
            log(f"SCT N=1024 FAILED: {type(e).__name__}: {e}")
            out["sct_n1024_extra"] = {"error": str(e)}
    if "ncg_n1024_latest" not in out:
        try:
            out["ncg_n1024_latest"] = ncg_n1024_eval_latest()
            with open(OUT_PATH, "w") as f: json.dump(out, f, indent=2)
        except Exception as e:
            log(f"NCG N=1024 FAILED: {type(e).__name__}: {e}")
            out["ncg_n1024_latest"] = {"error": str(e)}
    log("ALL DONE")


if __name__ == "__main__":
    main()
