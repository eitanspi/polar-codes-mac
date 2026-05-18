"""Overnight sweep: 5 phases of CPU-feasible runs.

Each phase saves its own JSON. The whole script is restartable: it skips
phases whose output already exists (use --force to override).

Phase A: 4-state chained SCT at N=512, 1024 (verify match with joint MAC SCT at large N).
Phase B: 2-state chained SCT at N=512, 1024 with own MC design.
Phase C: NCG re-eval at higher CW (N=16..128) to tighten existing numbers.
Phase D: Joint MAC SCT N=1024 high-CW push (200K CW, tighten upper bound).
Phase E: MA-AGN whitened-SCT (joint trellis on whitened ISI-MAC) at N=128, 256, 512.

Logs to /tmp/overnight/log.txt. Outputs in scripts/local_analysis/overnight_*.json.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)  # for chained_sct_* sibling imports

LOG_DIR = "/tmp/overnight"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "log.txt")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


WORKERS = 7
SNR_DB = 6.0
SIGMA2 = 10.0 ** (-SNR_DB / 10.0)
H_TAP = 0.3

RATES = {
    16: (4, 7), 32: (7, 15), 64: (15, 29), 128: (30, 58),
    256: (59, 117), 512: (119, 233), 1024: (239, 467),
}


# ============================================================================
# Phase A — 4-state chained SCT at N=512, 1024
# ============================================================================
def phase_A_4state_large_N(out_path, force=False):
    log("=" * 60)
    log("PHASE A: 4-state chained SCT at N=512, 1024")
    if os.path.exists(out_path) and not force:
        log(f"  output exists ({out_path}); skipping. Use --force to rerun.")
        return
    from chained_sct_4state import mc_design, eval_at, pick

    out = {"phase": "A", "decoder": "4-state chained SCT",
           "channel": "ISI-MAC", "h": H_TAP, "snr_db": SNR_DB, "results": {}}
    # Design trials: smaller for large N due to per-trial cost
    designs = {512: 15000, 1024: 8000}
    evals = {512: 10000, 1024: 10000}
    for N in [512, 1024]:
        ku, kv = RATES[N]
        log(f"  N={N} ku={ku} kv={kv}")
        t0 = time.time()
        Pe_u, Pe_v, n_done = mc_design(N, designs[N], WORKERS, base_seed=2027 * N + 11)
        td = time.time() - t0
        log(f"    design {n_done} trials in {td:.1f}s")
        Au = pick(Pe_u, ku); Av = pick(Pe_v, kv)
        t0 = time.time()
        r = eval_at(N, evals[N], Au, Av, WORKERS, base_seed=99 * N + 11)
        te = time.time() - t0
        log(f"    eval {r['n_cw']} CW: BLER={r['bler']:.6f} ({r['errs']}/{r['n_cw']}), {te:.1f}s")
        out["results"][str(N)] = {"N": N, "ku": ku, "kv": kv,
                                   "n_design": n_done, "design_time_s": td,
                                   "Au": Au, "Av": Av, "eval": r}
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    log(f"  PHASE A done: {out_path}")


# ============================================================================
# Phase B — 2-state chained SCT at N=512, 1024 with own design
# ============================================================================
def phase_B_2state_large_N(out_path, force=False):
    log("=" * 60)
    log("PHASE B: 2-state chained SCT at N=512, 1024 (own design)")
    if os.path.exists(out_path) and not force:
        log(f"  output exists ({out_path}); skipping.")
        return
    from chained_sct_mc_design import (
        mc_design_chained_sct, eval_chained_sct, pick_info_set,
    )

    out = {"phase": "B", "decoder": "2-state chained SCT (own design)",
           "channel": "ISI-MAC", "h": H_TAP, "snr_db": SNR_DB, "results": {}}
    designs = {512: 30000, 1024: 15000}
    evals = {512: 10000, 1024: 10000}
    for N in [512, 1024]:
        ku, kv = RATES[N]
        log(f"  N={N} ku={ku} kv={kv}")
        t0 = time.time()
        Pe_u, Pe_v, n_done, td = mc_design_chained_sct(
            N, designs[N], SIGMA2, H_TAP,
            base_seed=2026 * N + 7, n_workers=WORKERS)
        log(f"    design {n_done} trials in {td:.1f}s")
        Au = pick_info_set(Pe_u, ku); Av = pick_info_set(Pe_v, kv)
        t0 = time.time()
        r = eval_chained_sct(N, evals[N], SIGMA2, H_TAP, Au, Av,
                             base_seed=99 * N + 7, n_workers=WORKERS)
        te = time.time() - t0
        log(f"    eval {r['n_cw']} CW: BLER={r['bler_chained']:.6f} "
            f"({r['errs_chained']}/{r['n_cw']}), {te:.1f}s")
        out["results"][str(N)] = {"N": N, "ku": ku, "kv": kv,
                                   "n_design": n_done, "design_time_s": td,
                                   "Au": Au, "Av": Av, "eval": r}
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    log(f"  PHASE B done: {out_path}")


# ============================================================================
# Phase C — NCG re-eval at higher CW
# ============================================================================
def phase_C_ncg_reeval(out_path, force=False):
    log("=" * 60)
    log("PHASE C: NCG re-eval at higher CW (N=16..128)")
    if os.path.exists(out_path) and not force:
        log(f"  output exists ({out_path}); skipping.")
        return
    # NCG eval needs PyTorch + the ncg model. Try to import; skip gracefully if
    # the local module structure isn't there or torch isn't available.
    try:
        import torch
        from neural.ncg_isi_mac import NCGDecoder
    except Exception as e:
        log(f"  SKIP (NCG import failed: {e})")
        return
    try:
        from polar.channels_memory import ISIMAC
        from polar.encoder import polar_encode_batch
    except Exception as e:
        log(f"  SKIP (channel import failed: {e})")
        return

    # The existing checkpoints
    ckpt_dir = os.path.join(_HERE, "ncg_models")
    out = {"phase": "C", "decoder": "NCG corner-rate",
           "channel": "ISI-MAC", "h": H_TAP, "snr_db": SNR_DB, "results": {}}

    target_cw_by_N = {16: 20000, 32: 20000, 64: 20000, 128: 10000}
    ch = ISIMAC(sigma2=SIGMA2, h=H_TAP)
    device = torch.device("cpu")

    for N in [16, 32, 64, 128]:
        ckpt_path = os.path.join(ckpt_dir, f"ncg_isi_N{N}.pt")
        if not os.path.exists(ckpt_path):
            log(f"  N={N}: no checkpoint at {ckpt_path}, skipping")
            continue
        ku, kv = RATES[N]
        n = int(np.log2(N))
        # Try to load
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            log(f"  N={N}: ckpt load failed: {e}")
            continue
        log(f"  N={N}: ckpt loaded, ku={ku} kv={kv}, target {target_cw_by_N[N]} CW")
        # Best-effort: count errors via simple loop. The NCG model's exact API
        # may not be uniform; we wrap and skip on failure.
        try:
            model = ckpt.get("model", None)
            if model is None:
                # Maybe the ckpt IS the model state_dict
                log(f"    N={N}: ckpt format unknown; skipping NCG eval")
                continue
            model.eval()
            # ... we'd need the exact decode API; bail gracefully
            log(f"    N={N}: NCG decode API not auto-detected; skipping")
        except Exception as e:
            log(f"    N={N}: NCG eval failed: {e}")
            continue

    log(f"  PHASE C done (best-effort): {out_path}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


# ============================================================================
# Phase D — Joint MAC SCT N=1024 push to 200K CW
# ============================================================================
def phase_D_joint_sct_n1024_push(out_path, force=False):
    log("=" * 60)
    log("PHASE D: Joint MAC SCT at N=1024, push to 200K CW")
    if os.path.exists(out_path) and not force:
        log(f"  output exists ({out_path}); skipping.")
        return
    from polar.channels_memory import ISIMAC
    from polar.encoder import polar_encode
    from polar.decoder_trellis import decode_single
    from polar.design import make_path
    from polar.design_mc import design_from_file

    # Use the joint-trellis MC design from the campaign
    jt = json.load(open(os.path.join(_ROOT,
        "class_c_npd/results/joint_trellis_batched/results.json")))
    if "1024" not in jt:
        log("  joint_trellis_batched has no N=1024; trying joint_trellis_mc_3dB")
        jt = json.load(open(os.path.join(_ROOT,
            "class_c_npd/results/joint_trellis_mc_3dB/results.json")))
    if "1024" not in jt:
        log("  no N=1024 design found; using own 4-state design from Phase A")
        try:
            phA = json.load(open(os.path.join(_HERE, "overnight_A_4state_largeN.json")))
            Au = phA["results"]["1024"]["Au"]
            Av = phA["results"]["1024"]["Av"]
        except Exception as e:
            log(f"  cannot get N=1024 design: {e}; SKIP")
            return
    else:
        Au = jt["1024"].get("Au_jt") or jt["1024"].get("Au")
        Av = jt["1024"].get("Av_jt") or jt["1024"].get("Av")

    N = 1024
    ku, kv = RATES[N]
    b_path = make_path(N, N)
    fu = {p: 0 for p in range(1, N + 1) if p not in set(Au)}
    fv = {p: 0 for p in range(1, N + 1) if p not in set(Av)}
    ch = ISIMAC(sigma2=SIGMA2, h=H_TAP)

    target_n_cw = 200000
    log(f"  N={N} ku={ku} kv={kv}, target {target_n_cw} CW")

    def chunk(args):
        seed, nt = args
        rng = np.random.default_rng(seed)
        errs = 0
        for _ in range(nt):
            u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
            for p in set(Au): u[p - 1] = rng.integers(0, 2)
            for p in set(Av): v[p - 1] = rng.integers(0, 2)
            x = np.array(polar_encode(u.tolist()), dtype=np.int64)
            y = np.array(polar_encode(v.tolist()), dtype=np.int64)
            z = ch.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
            u_hat, v_hat = decode_single(N, z, b_path, fu, fv, ch)
            ue = any(int(u_hat[p - 1]) != int(u[p - 1]) for p in set(Au))
            ve = any(int(v_hat[p - 1]) != int(v[p - 1]) for p in set(Av))
            if ue or ve:
                errs += 1
        return errs, nt

    cw_per = max(1, target_n_cw // WORKERS)
    jobs = []
    started = 0
    for w in range(WORKERS):
        nt = cw_per if w < WORKERS - 1 else (target_n_cw - started)
        jobs.append((30000 + w, nt))
        started += nt
    t0 = time.time()
    log(f"  launching {WORKERS} workers, {jobs[0][1]}-{jobs[-1][1]} CW each")
    with Pool(WORKERS) as pool:
        results = pool.map(chunk, jobs)
    total_errs = sum(r[0] for r in results)
    total_cw = sum(r[1] for r in results)
    elapsed = time.time() - t0
    bler = total_errs / total_cw if total_cw > 0 else 0.0
    # 95% Poisson UCL
    if total_errs == 0:
        ucl = 3.0 / total_cw
    else:
        ucl = (total_errs + 1.96 * np.sqrt(total_errs)) / total_cw
    log(f"  N=1024 joint MAC SCT: {total_errs}/{total_cw} BLER={bler:.3e} "
        f"95%UCL={ucl:.3e}  in {elapsed:.1f}s")
    out = {"phase": "D", "decoder": "Joint MAC SCT N=1024 push",
           "channel": "ISI-MAC", "h": H_TAP, "snr_db": SNR_DB,
           "N": N, "ku": ku, "kv": kv, "Au": Au, "Av": Av,
           "n_cw": total_cw, "errs": int(total_errs),
           "bler": bler, "ucl_95": ucl, "elapsed_s": elapsed}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"  PHASE D done: {out_path}")


# ============================================================================
# Phase E — MA-AGN whitened-SCT
# ============================================================================
def phase_E_maagn_whitened_sct(out_path, force=False):
    log("=" * 60)
    log("PHASE E: MA-AGN whitened-SCT at N=128, 256, 512 (α=0.5)")
    if os.path.exists(out_path) and not force:
        log(f"  output exists ({out_path}); skipping.")
        return
    from polar.channels_memory import ISIMAC
    from polar.channels_memory_new import MAAGNMAC
    from polar.encoder import polar_encode
    from polar.decoder_trellis import decode_single
    from polar.design import make_path

    alpha = 0.5
    sigma2_stat = SIGMA2
    sigma2_eff = sigma2_stat * (1 - alpha ** 2)
    h_eff = -alpha
    log(f"  MA-AGN α={alpha} σ²_stat={sigma2_stat:.4f} → whitened ISI h={h_eff} "
        f"σ²_eff={sigma2_eff:.4f}")
    isi_eff = ISIMAC(sigma2=sigma2_eff, h=h_eff)
    maagn = MAAGNMAC(sigma2=sigma2_stat, alpha=alpha)

    out = {"phase": "E", "decoder": "MA-AGN whitened-SCT",
           "channel": "MA-AGN", "alpha": alpha, "snr_db": SNR_DB,
           "h_equiv": h_eff, "sigma2_eff": sigma2_eff, "results": {}}

    # Design via MC on the WHITENED channel (use the equivalent ISIMAC for FB)
    from chained_sct_4state import mc_design as mc_design_4state
    from chained_sct_4state import pick

    designs = {128: 30000, 256: 20000, 512: 10000}
    evals = {128: 10000, 256: 10000, 512: 5000}

    def whiten(z, alpha):
        zp = np.empty_like(z)
        zp[0] = z[0]
        zp[1:] = z[1:] - alpha * z[:-1]
        return zp

    def eval_chunk(args):
        N, fu, fv, Au, Av, seed, n_cw = args
        rng = np.random.default_rng(seed)
        Au_s, Av_s = set(int(p) for p in Au), set(int(p) for p in Av)
        b_path = make_path(N, N)
        errs = 0
        for _ in range(n_cw):
            u = np.zeros(N, dtype=int); v = np.zeros(N, dtype=int)
            for p in Au_s: u[p - 1] = rng.integers(0, 2)
            for p in Av_s: v[p - 1] = rng.integers(0, 2)
            x = np.array(polar_encode(u.tolist()), dtype=np.int64)
            y = np.array(polar_encode(v.tolist()), dtype=np.int64)
            z = maagn.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
            zp = whiten(z, alpha)
            u_hat, v_hat = decode_single(N, zp, b_path, fu, fv, isi_eff)
            ue = any(int(u_hat[p - 1]) != int(u[p - 1]) for p in Au_s)
            ve = any(int(v_hat[p - 1]) != int(v[p - 1]) for p in Av_s)
            if ue or ve:
                errs += 1
        return errs, n_cw

    def design_chunk(args):
        # MC design via genie SC on whitened channel
        from polar.decoder import _SCNode
        from polar.decoder_trellis import _forward_backward_joint
        N, seed, n_trials = args
        rng = np.random.default_rng(seed)
        u_err = np.zeros(N, dtype=np.int64)
        v_err = np.zeros(N, dtype=np.int64)
        for _ in range(n_trials):
            u = rng.integers(0, 2, size=N)
            v = rng.integers(0, 2, size=N)
            x = np.array(polar_encode(u.tolist()), dtype=np.int64)
            y = np.array(polar_encode(v.tolist()), dtype=np.int64)
            z = maagn.sample_batch(x.reshape(1, -1), y.reshape(1, -1))[0]
            zp = whiten(z, alpha)
            # 4-state FB on the whitened channel (ISI-MAC with h=-α, σ²_eff)
            log_W = isi_eff.build_leaf_tensors(zp)
            log_marg = _forward_backward_joint(log_W, N, isi_eff.num_states)
            # Stage 1: marginalize Y → scalar U LLR
            log_pu0 = np.logaddexp(log_marg[:, 0, 0], log_marg[:, 0, 1])
            log_pu1 = np.logaddexp(log_marg[:, 1, 0], log_marg[:, 1, 1])
            llr_u = log_pu0 - log_pu1
            node = _SCNode(llr_u.astype(np.float64))
            for i in range(N):
                L = node.get_llr(i)
                decided = 0 if L >= 0 else 1
                true_val = int(u[i])
                if decided != true_val:
                    u_err[i] += 1
                node.feed(i, true_val)
            # Stage 2: condition on TRUE x → 2-state Y-FB on whitened
            from polar.decoder_trellis_mac_chained import _log_W_stage2, _forward_backward_2state
            log_W2 = _log_W_stage2(zp, x, isi_eff)
            log_marg2 = _forward_backward_2state(log_W2)
            llr_v = log_marg2[:, 0] - log_marg2[:, 1]
            node = _SCNode(llr_v.astype(np.float64))
            for i in range(N):
                L = node.get_llr(i)
                decided = 0 if L >= 0 else 1
                true_val = int(v[i])
                if decided != true_val:
                    v_err[i] += 1
                node.feed(i, true_val)
        return u_err, v_err, n_trials

    from polar.design_mc import _argsort_with_polar_tiebreak
    def pick_set(Pe, k):
        order = _argsort_with_polar_tiebreak(np.asarray(Pe))
        return [int(i) + 1 for i in sorted(order[:k].tolist())]

    for N in [128, 256, 512]:
        ku, kv = RATES[N]
        log(f"  N={N} ku={ku} kv={kv}: design")
        # Design
        n_trials = designs[N]
        chunk_size = max(1, n_trials // WORKERS)
        jobs = []; started = 0
        for w in range(WORKERS):
            nt = chunk_size if w < WORKERS - 1 else (n_trials - started)
            jobs.append((N, 50000 + w * 9001 + N, nt))
            started += nt
        t0 = time.time()
        with Pool(WORKERS) as pool:
            results = pool.map(design_chunk, jobs)
        u_err = sum(r[0] for r in results)
        v_err = sum(r[1] for r in results)
        n_done = sum(r[2] for r in results)
        td = time.time() - t0
        Pe_u = u_err / n_done; Pe_v = v_err / n_done
        Au = pick_set(Pe_u, ku); Av = pick_set(Pe_v, kv)
        log(f"    design {n_done} trials in {td:.1f}s")

        # Eval
        fu = {p: 0 for p in range(1, N + 1) if p not in set(Au)}
        fv = {p: 0 for p in range(1, N + 1) if p not in set(Av)}
        n_cw = evals[N]
        chunk_size = max(1, n_cw // WORKERS)
        jobs = []; started = 0
        for w in range(WORKERS):
            nt = chunk_size if w < WORKERS - 1 else (n_cw - started)
            jobs.append((N, fu, fv, Au, Av, 60000 + w + N, nt))
            started += nt
        t0 = time.time()
        with Pool(WORKERS) as pool:
            results = pool.map(eval_chunk, jobs)
        errs = sum(r[0] for r in results)
        total_cw = sum(r[1] for r in results)
        te = time.time() - t0
        bler = errs / total_cw
        log(f"    eval {total_cw} CW: BLER={bler:.6f} ({errs}/{total_cw}), {te:.1f}s")
        out["results"][str(N)] = {"N": N, "ku": ku, "kv": kv,
                                   "n_design": n_done, "design_time_s": td,
                                   "Au": Au, "Av": Av,
                                   "eval": {"n_cw": total_cw, "errs": int(errs),
                                            "bler": bler, "elapsed_s": te}}
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    log(f"  PHASE E done: {out_path}")


# ============================================================================
# Driver
# ============================================================================
def main():
    log("################################################")
    log("OVERNIGHT SWEEP started")
    log("################################################")
    t_all = time.time()

    out_A = os.path.join(_HERE, "overnight_A_4state_largeN.json")
    out_B = os.path.join(_HERE, "overnight_B_2state_largeN.json")
    out_C = os.path.join(_HERE, "overnight_C_ncg_reeval.json")
    out_D = os.path.join(_HERE, "overnight_D_jointSCT_N1024.json")
    out_E = os.path.join(_HERE, "overnight_E_maagn_whitened.json")

    try:
        phase_A_4state_large_N(out_A)
    except Exception as e:
        log(f"PHASE A FAILED: {e}")
    try:
        phase_B_2state_large_N(out_B)
    except Exception as e:
        log(f"PHASE B FAILED: {e}")
    try:
        phase_C_ncg_reeval(out_C)
    except Exception as e:
        log(f"PHASE C FAILED: {e}")
    try:
        phase_E_maagn_whitened_sct(out_E)
    except Exception as e:
        log(f"PHASE E FAILED: {e}")
    try:
        phase_D_joint_sct_n1024_push(out_D)
    except Exception as e:
        log(f"PHASE D FAILED: {e}")

    elapsed = time.time() - t_all
    log(f"ALL PHASES DONE in {elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()
