"""Top up evaluation CW for SCT + NCG at each N until ≥30 errors observed.

Already covered (>=30 errs from prior runs):
  SCT N=16,32,64,128 — 30K CW each → 177-4500 errs
  NCG N=16,32,64,128 — from cluster v4 phase 1 (30K CW → 224-5045 errs)
  NCG N=1024 untrained — 76 errs / 5K CW

Phases (run in order, smallest first, biggest last for safety):
  S256  : SCT N=256 — eval up to 50K CW (target ~65 errs at BLER 1.3e-3)
  S512  : SCT N=512 — eval up to 80K CW (target ~35 errs at BLER 4.3e-4)
  N512  : NCG N=512 trained ckpt — eval up to 30K CW (target ~30 errs at BLER 1e-3)
  S1024 : SCT N=1024 — eval up to 800K CW (target ~32 errs at BLER 4e-5). BIG (~4 hr).
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
if _HERE not in sys.path: sys.path.insert(0, _HERE)

LOG_DIR = "/tmp/topup_30errors"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "log.txt")
OUT_PATH = os.path.join(_HERE, "topup_30errors_results.json")

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f: f.write(line + "\n")

WORKERS = 7

# Module-level eval chunk for SCT N=N using the 4-state-chained decoder.
def _sct_eval_chunk(args):
    from polar.channels_memory import ISIMAC
    from polar.encoder import polar_encode
    from chained_sct_4state import decode_4state_chained, SIGMA2, H_TAP
    N, fu, fv, Au, Av, seed, n_cw = args
    ch = ISIMAC(sigma2=SIGMA2, h=H_TAP)
    rng = np.random.default_rng(seed)
    Au_s = set(int(p) for p in Au); Av_s = set(int(p) for p in Av)
    errs = 0
    for _ in range(n_cw):
        u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
        for p in Au_s: u[p-1] = rng.integers(0, 2)
        for p in Av_s: v[p-1] = rng.integers(0, 2)
        x = np.array(polar_encode(u.tolist()), dtype=np.int64)
        y = np.array(polar_encode(v.tolist()), dtype=np.int64)
        z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1))[0]
        u_hat, v_hat = decode_4state_chained(z, N, fu, fv, ch)
        ue = any(int(u_hat[p-1]) != int(u[p-1]) for p in Au_s)
        ve = any(int(v_hat[p-1]) != int(v[p-1]) for p in Av_s)
        if ue or ve: errs += 1
    return errs, n_cw


def sct_eval(N, ku, kv, Au, Av, n_cw_total, base_seed):
    fu = {p: 0 for p in range(1, N+1) if p not in set(Au)}
    fv = {p: 0 for p in range(1, N+1) if p not in set(Av)}
    chunk = max(1, n_cw_total // WORKERS)
    jobs, started = [], 0
    for w in range(WORKERS):
        nt = chunk if w < WORKERS-1 else (n_cw_total - started)
        jobs.append((N, fu, fv, Au, Av, base_seed + w * 7777, nt))
        started += nt
    t0 = time.time()
    with Pool(WORKERS) as pool:
        results = pool.map(_sct_eval_chunk, jobs)
    errs = sum(r[0] for r in results); total = sum(r[1] for r in results)
    return errs, total, time.time() - t0


def design_and_eval_sct(N, target_n_cw):
    """Run a fresh MC design (if not already cached for this N), then eval."""
    from chained_sct_4state import mc_design, pick, RATES
    ku, kv = RATES[N]
    # Use a reasonable design count
    n_design = {16:30000, 32:30000, 64:30000, 128:50000, 256:50000, 512:50000, 1024:200000}[N]
    log(f"  SCT N={N}: design ({n_design} trials)...")
    t0 = time.time()
    Pe_u, Pe_v, _ = mc_design(N, n_design, WORKERS, base_seed=800000 + N)
    td = time.time() - t0
    Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
    log(f"    design done in {td/60:.1f}min; eval {target_n_cw} CW...")
    errs, total, te = sct_eval(N, ku, kv, Au, Av, target_n_cw, base_seed=810000 + N)
    bler = errs/total
    log(f"    SCT N={N}: BLER={bler:.6f} ({errs}/{total}) in {te/60:.1f}min")
    return dict(N=N, ku=ku, kv=kv, Au=Au, Av=Av,
                n_design=n_design, design_time_s=td,
                n_cw=total, errs=int(errs), bler=bler, eval_time_s=te)


def ncg_eval_n512(ckpt_path, n_cw):
    """Eval NCG N=512 trained ckpt at target CW count (single thread)."""
    import torch
    from polar.channels_memory import ISIMAC
    from polar.encoder import polar_encode_batch
    from polar.design import make_path
    from polar.design_mc import design_from_file
    from neural.ncg_isi_mac import ISIMACNeuralDecoder
    torch.set_num_threads(1)
    SIGMA2 = 10**(-0.6); H = 0.3
    N = 512; n = 9; ku = 119; kv = 233
    ch = ISIMAC(sigma2=SIGMA2, h=H)
    b_path = make_path(N, N)
    gmac_file = f"/Users/ytnspybq/PycharmProjects/polar_codes_MAC/to_git_v2/designs/gmac_C_n{n}_snr6dB.npz"
    Au, Av, fu_set, fv_set, _, _, _ = design_from_file(gmac_file, n, ku, kv)
    Au = [int(p) for p in Au]; Av = [int(p) for p in Av]
    fu = {int(p): 0 for p in fu_set}; fv = {int(p): 0 for p in fv_set}
    model = ISIMACNeuralDecoder(d=16, hidden=64, n_layers=2, z_hidden=32, z_encoder_type='window')
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    own = model.state_dict()
    for k, w in sd.items():
        if k in own and own[k].shape == w.shape: own[k] = w
    model.load_state_dict(own); model.eval()
    rng = np.random.default_rng(900512)
    t0 = time.time(); errs = 0
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
            if (cw+1) % max(1, n_cw // 10) == 0:
                el = (time.time()-t0)/60
                log(f"    NCG N=512 {cw+1}/{n_cw} errs={errs} ({el:.1f}min)")
    bler = errs / n_cw
    log(f"    NCG N=512 trained: BLER={bler:.6f} ({errs}/{n_cw}) in {(time.time()-t0)/60:.1f}min")
    return dict(N=N, ku=ku, kv=kv, ckpt=ckpt_path,
                n_cw=n_cw, errs=errs, bler=bler, elapsed_s=time.time()-t0)


def main():
    out = {}
    if os.path.exists(OUT_PATH):
        try: out = json.load(open(OUT_PATH))
        except: out = {}

    log("############################################")
    log("30-errors top-up: SCT + NCG N=512")
    log("############################################")

    # Order: smaller first (cheap), big SCT N=1024 last
    plan = [
        ("S256",  lambda: design_and_eval_sct(256,  50000)),
        ("S512",  lambda: design_and_eval_sct(512,  80000)),
        ("N512",  lambda: ncg_eval_n512("/tmp/ncg_intermediate/ncg_n512_15k_final.pt", 30000)),
        ("S1024", lambda: design_and_eval_sct(1024, 800000)),
    ]
    for key, fn in plan:
        if key in out and out[key].get("done"):
            log(f"  {key} already done, skip"); continue
        log(f"--- {key} ---")
        try:
            r = fn()
            r["done"] = True
            out[key] = r
            with open(OUT_PATH, "w") as f: json.dump(out, f, indent=2)
        except Exception as e:
            log(f"  {key} FAILED: {type(e).__name__}: {e}")
            out[key] = {"error": str(e)}
            with open(OUT_PATH, "w") as f: json.dump(out, f, indent=2)
    log("ALL DONE")


if __name__ == "__main__":
    main()
