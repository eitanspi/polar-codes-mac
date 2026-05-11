#!/usr/bin/env python3
"""
3-Phase NPD Design for ISI-MAC (following Aharoni et al. NPD project).

Phase 1: Train rate-1 NPD on ISI-MAC (all positions info, no frozen)
Phase 2: Measure per-position MI via fast_ce at final tree depth
Phase 3: Retrain NPD at target rate with MI-based frozen set

Also: eval NPD with the iterative SC frozen set (from CPU script).

Runs on GPU. Targets N=128, 256, 512.
"""
import sys, os, time, json, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/gpfs0/bgu-haimp/users/eitansp/polar_project")

from neural.npd_memory_mac import ChainedNPD_MAC, MemoryStageNPD, NPDTree
from polar.channels_memory import ISIMAC
from polar.encoder import polar_encode_batch, bit_reversal_perm
from polar.design_mc import _argsort_with_polar_tiebreak

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA2 = 10.0 ** (-6.0 / 10.0)
ISI_H = 0.3

BASE = "/gpfs0/bgu-haimp/users/eitansp/polar_project"
OUT = os.path.join(BASE, "class_c_npd/results/npd_3phase_isi")
os.makedirs(OUT, exist_ok=True)

LOG = os.path.join(OUT, "run.log")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ─── Batch generation ────────────────────────────────────────────────────────

def make_batch_rate1(ch, N, batch, rng):
    """Rate-1 batch: all positions are info."""
    u = rng.integers(0, 2, (batch, N)).astype(np.int8)
    v = rng.integers(0, 2, (batch, N)).astype(np.int8)
    x = polar_encode_batch(u.astype(int))
    y = polar_encode_batch(v.astype(int))
    z = ch.sample_batch(x, y).astype(np.float32)
    return u, v, z, x, y


def make_batch(ch, N, Au, Av, batch, rng):
    """Target-rate batch: only info positions are random."""
    u = np.zeros((batch, N), dtype=np.int8)
    v = np.zeros((batch, N), dtype=np.int8)
    for p in Au: u[:, p-1] = rng.integers(0, 2, batch)
    for p in Av: v[:, p-1] = rng.integers(0, 2, batch)
    x = polar_encode_batch(u.astype(int))
    y = polar_encode_batch(v.astype(int))
    z = ch.sample_batch(x, y).astype(np.float32)
    return u, v, z, x, y


# ─── Per-position MI measurement via fast_ce ─────────────────────────────────

def fast_ce_per_position(tree, emb, x_cw):
    """
    Run fast_ce but return per-position BCE at the FINAL tree depth
    (the synthetic bit-channel level).

    emb: (B, N, d) in tree order
    x_cw: (B, N) in tree order (codeword bits)

    Returns: per_pos_bce (N,) — average BCE per synthetic channel position
    """
    B, N, d = emb.shape
    n = int(math.log2(N))

    V = [x_cw]
    E = [emb]

    for depth in range(n):
        V_odds, V_evens, E_odds, E_evens = [], [], [], []
        for v_chunk, e_chunk in zip(V, E):
            V_odds.append(v_chunk[:, 0::2])
            V_evens.append(v_chunk[:, 1::2])
            E_odds.append(e_chunk[:, 0::2, :])
            E_evens.append(e_chunk[:, 1::2, :])

        V_odd = torch.cat(V_odds, dim=1)
        V_even = torch.cat(V_evens, dim=1)
        E_odd = torch.cat(E_odds, dim=1)
        E_even = torch.cat(E_evens, dim=1)

        v_top = V_odd ^ V_even
        v_bot = V_even

        num_chunks = 2 ** depth
        chunk_size = (N // 2) // num_chunks
        v_top_chunks = torch.split(v_top, chunk_size, dim=1)
        v_bot_chunks = torch.split(v_bot, chunk_size, dim=1)
        V_new = []
        for vt, vb in zip(v_top_chunks, v_bot_chunks):
            V_new.append(vt)
            V_new.append(vb)
        V_left = torch.cat(V_new[0::2], dim=1)

        e_top = tree.checknode(torch.cat([E_odd, E_even], dim=-1))
        e_bot = tree.bitnode(E_odd, E_even, V_left)

        e_top_chunks = torch.split(e_top, chunk_size, dim=1)
        e_bot_chunks = torch.split(e_bot, chunk_size, dim=1)
        E_new = []
        for et, eb in zip(e_top_chunks, e_bot_chunks):
            E_new.append(et)
            E_new.append(eb)

        V = V_new
        E = E_new

    # At final depth: V has N chunks of size 1, E has N chunks of size 1
    # Concatenate to get per-position logits
    e_final = torch.cat(E, dim=1)  # (B, N, d)
    v_final = torch.cat(V, dim=1)  # (B, N)

    logit = tree.emb2llr(e_final).squeeze(-1)  # (B, N)

    # Per-position BCE (no reduction)
    bce = F.binary_cross_entropy_with_logits(
        logit, v_final.float(), reduction='none')  # (B, N)

    return bce.mean(dim=0)  # (N,) — average over batch


def measure_mi(stage, ch, N, n_cw=50000, batch=200):
    """
    Measure per-position MI via fast_ce at rate-1.

    MI_i = H(U_i) - H(U_i | Y) = log(2) - BCE_i

    For uniform input bits, H(U_i) = log(2).
    H(U_i | Y) is approximated by the BCE from the trained model.

    Returns: mi (N,) — per-position mutual information (in nats)
    """
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.tensor(br, dtype=torch.long).to(device)

    stage.to(device)
    stage.eval()

    rng = np.random.default_rng(42)
    total_bce = torch.zeros(N, device=device)
    total = 0

    with torch.no_grad():
        while total < n_cw:
            actual = min(batch, n_cw - total)
            u, v, z, x, y = make_batch_rate1(ch, N, actual, rng)
            zt = torch.from_numpy(z).to(device)

            emb = stage.encode_channel(zt)
            emb_br = emb[:, br_t, :]
            x_br = torch.from_numpy(x[:, br]).long().to(device)

            bce = fast_ce_per_position(stage.tree, emb_br, x_br)
            total_bce += bce * actual
            total += actual

            if total % 10000 == 0:
                log(f"    MI measurement: {total}/{n_cw}")

    avg_bce = total_bce / total  # (N,) in tree order
    # MI = log(2) - BCE (for uniform input bits)
    mi_tree_order = np.log(2) - avg_bce.cpu().numpy()
    # Convert from tree order to natural order
    mi_natural = np.zeros(N)
    for i in range(N):
        mi_natural[br[i]] = mi_tree_order[i]

    return mi_natural


# ─── Training ────────────────────────────────────────────────────────────────

def train_stage1(stage, ch, N, Au, Av, frozen_set, iters, batch, tag,
                 eval_every=50000, lr=1e-3):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.tensor(br, dtype=torch.long).to(device)

    stage.to(device)
    opt = torch.optim.Adam(stage.parameters(), lr=lr)
    rng = np.random.default_rng(42)

    best_loss = float("inf")
    best_state = None
    t0 = time.time()

    stage.train()
    for it in range(1, iters + 1):
        if len(Au) == N:
            u, v, z, x, y = make_batch_rate1(ch, N, batch, rng)
        else:
            u, v, z, x, y = make_batch(ch, N, Au, Av, batch, rng)
        zt = torch.from_numpy(z).to(device)
        emb = stage.encode_channel(zt)
        emb_br = emb[:, br_t, :]
        target = torch.from_numpy(x[:, br]).long().to(device)

        loss = stage.tree.fast_ce(emb_br, target)
        if torch.isnan(loss):
            log(f"  NaN at iter {it}!")
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage.parameters(), 1.0)
        opt.step()

        if it % eval_every == 0 or it == iters:
            elapsed = (time.time() - t0) / 60
            marker = ""
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in stage.state_dict().items()}
                marker = " *BEST*"
                torch.save({"state_dict": best_state, "N": N, "Au": Au, "Av": Av,
                             "iter": it}, os.path.join(OUT, f"{tag}_best.pt"))
            if it % 100000 == 0:
                torch.save({"state_dict": {k: v.cpu() for k, v in stage.state_dict().items()},
                            "N": N, "iter": it}, os.path.join(OUT, f"{tag}_iter{it}.pt"))
            log(f"  [{tag} {it:>7}/{iters}] loss={loss.item():.4f} "
                f"(best={best_loss:.4f}) {elapsed:.0f}min{marker}")

    torch.save({"state_dict": {k: v.cpu() for k, v in stage.state_dict().items()},
                "N": N, "iter": iters}, os.path.join(OUT, f"{tag}_final.pt"))

    if best_state:
        stage.load_state_dict(best_state)
        stage.to(device)

    return best_loss


def train_stage2(stage, ch, N, Au, Av, frozen_set, iters, batch, tag,
                 eval_every=50000, lr=1e-3):
    n = int(math.log2(N))
    br = bit_reversal_perm(n)
    br_t = torch.tensor(br, dtype=torch.long).to(device)

    stage.to(device)
    opt = torch.optim.Adam(stage.parameters(), lr=lr)
    rng = np.random.default_rng(43)

    best_loss = float("inf")
    best_state = None
    t0 = time.time()

    stage.train()
    for it in range(1, iters + 1):
        u, v, z, x, y = make_batch(ch, N, Au, Av, batch, rng)
        zt = torch.from_numpy(z).to(device)
        side = torch.from_numpy((1.0-2.0*x.astype(np.float32))).unsqueeze(-1).to(device)
        emb = stage.encode_channel(zt, side=side)
        emb_br = emb[:, br_t, :]
        target = torch.from_numpy(y[:, br]).long().to(device)

        loss = stage.tree.fast_ce(emb_br, target)
        if torch.isnan(loss):
            log(f"  NaN at iter {it}!")
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stage.parameters(), 1.0)
        opt.step()

        if it % eval_every == 0 or it == iters:
            elapsed = (time.time() - t0) / 60
            marker = ""
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in stage.state_dict().items()}
                marker = " *BEST*"
                torch.save({"state_dict": best_state, "N": N, "Au": Au, "Av": Av,
                             "iter": it}, os.path.join(OUT, f"{tag}_best.pt"))
            if it % 100000 == 0:
                torch.save({"state_dict": {k: v.cpu() for k, v in stage.state_dict().items()},
                            "N": N, "iter": it}, os.path.join(OUT, f"{tag}_iter{it}.pt"))
            log(f"  [{tag} {it:>7}/{iters}] loss={loss.item():.4f} "
                f"(best={best_loss:.4f}) {elapsed:.0f}min{marker}")

    torch.save({"state_dict": {k: v.cpu() for k, v in stage.state_dict().items()},
                "N": N, "iter": iters}, os.path.join(OUT, f"{tag}_final.pt"))

    if best_state:
        stage.load_state_dict(best_state)
        stage.to(device)

    return best_loss


# ─── Eval (CPU decode) ───────────────────────────────────────────────────────

def eval_chained_bler(model, ch, N, Au, Av, fu_set, fv_set, n_cw=3000):
    n = int(math.log2(N))
    br_cpu = torch.tensor(bit_reversal_perm(n), dtype=torch.long)
    model.stage1.eval()
    model.stage2.eval()
    model.stage1.tree.cpu()
    model.stage2.tree.cpu()
    rng = np.random.default_rng(777)
    errs_u = errs_v = errs_total = 0
    with torch.no_grad():
        for cw in range(n_cw):
            u = np.zeros(N, dtype=np.int8)
            v = np.zeros(N, dtype=np.int8)
            for p in Au: u[p-1] = rng.integers(0, 2)
            for p in Av: v[p-1] = rng.integers(0, 2)
            x = polar_encode_batch(u.reshape(1,-1).astype(int))[0]
            y = polar_encode_batch(v.reshape(1,-1).astype(int))[0]
            z = ch.sample_batch(x.reshape(1,-1), y.reshape(1,-1)).astype(np.float32)
            zt = torch.from_numpy(z).to(device)
            emb1 = model.stage1.encode_channel(zt)
            u_hat = model.stage1.tree.decode(emb1.cpu()[:, br_cpu, :], fu_set)
            u_np = u_hat[0].numpy().astype(int)
            x_hat = polar_encode_batch(u_np.reshape(1,-1))[0]
            side = torch.from_numpy((1.0-2.0*x_hat.astype(np.float32)).reshape(1,-1,1)).to(device)
            emb2 = model.stage2.encode_channel(zt, side=side)
            v_hat = model.stage2.tree.decode(emb2.cpu()[:, br_cpu, :], fv_set)
            ue = any(int(u_hat[0,p-1].item()) != int(u[p-1]) for p in Au)
            ve = any(int(v_hat[0,p-1].item()) != int(v[p-1]) for p in Av)
            if ue: errs_u += 1
            if ve: errs_v += 1
            if ue or ve: errs_total += 1
            if (cw+1) % 500 == 0:
                log(f"      chained: {cw+1}/{n_cw}")
    model.stage1.tree.to(device)
    model.stage2.tree.to(device)
    return errs_total/n_cw, errs_u/n_cw, errs_v/n_cw


# ─── Main pipeline ───────────────────────────────────────────────────────────

KU_KV = {128: (30, 58), 256: (59, 117), 512: (119, 233)}

CONFIGS = [
    # (N, p1_iters, p3_iters_s1, p3_iters_s2, batch)
    (128, 500_000, 1_000_000, 500_000, 32),
    (256, 500_000, 1_000_000, 500_000, 16),
    (512, 500_000, 1_000_000, 500_000, 8),
]

total_t0 = time.time()
all_results = {}

log("=" * 70)
log("3-Phase NPD Design for ISI-MAC")
log(f"Device: {device}" + (f", GPU: {torch.cuda.get_device_name()}" if device.type == 'cuda' else ""))
log("=" * 70)

ch = ISIMAC(sigma2=SIGMA2, h=ISI_H)

for N, p1_iters, p3_s1_iters, p3_s2_iters, batch in CONFIGS:
    elapsed_h = (time.time() - total_t0) / 3600
    if elapsed_h > 11:
        log(f"TIME LIMIT ({elapsed_h:.1f}h). Stopping.")
        break

    ku, kv = KU_KV[N]
    n = int(math.log2(N))
    Au_all = list(range(1, N + 1))

    log(f"\n{'='*60}")
    log(f"N={N}: ku={ku}, kv={kv} [{elapsed_h:.1f}h elapsed]")
    log(f"{'='*60}")

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Train rate-1 model
    # ══════════════════════════════════════════════════════════════
    log(f"\n--- Phase 1: Train rate-1 NPD ({p1_iters} iters) ---")
    torch.manual_seed(42)
    model_p1 = ChainedNPD_MAC(d=16, hidden=100, n_layers=2,
                               encoder_type='bigru', gru_layers=1)
    model_p1.to(device)
    log(f"  Params: {model_p1.count_parameters():,}")

    p1_loss = train_stage1(model_p1.stage1, ch, N, Au_all, Au_all,
                           set(), iters=p1_iters, batch=batch,
                           tag=f"p1_rate1_s1_N{N}", eval_every=50000)
    log(f"  Phase 1 done: best_loss={p1_loss:.4f}")

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Measure MI per position
    # ══════════════════════════════════════════════════════════════
    log(f"\n--- Phase 2: Measure MI per position ---")
    mi = measure_mi(model_p1.stage1, ch, N, n_cw=50000, batch=200)

    # Select frozen set from MI
    sorted_by_mi = np.argsort(-mi)  # descending MI = best channels first
    Au_mi = sorted([int(i + 1) for i in sorted_by_mi[:ku]])
    Av_mi_sorted = np.argsort(-measure_mi(model_p1.stage1, ch, N, n_cw=20000, batch=200))
    # For V: use GMAC proxy (simpler, stage 2 rarely walls)
    gmac_d = np.load(os.path.join(BASE, f"designs/gmac_C_n{n}_snr{int(6)}dB.npz"))
    pe_v_gmac = gmac_d['v_error_rates']
    Av_sorted = _argsort_with_polar_tiebreak(pe_v_gmac)
    Av_mi = sorted([int(i + 1) for i in Av_sorted[:kv]])

    fu_mi_set = set(range(N)) - set(p - 1 for p in Au_mi)
    fv_mi_set = set(range(N)) - set(p - 1 for p in Av_mi)

    # Compare with GMAC proxy
    pe_u_gmac = gmac_d['u_error_rates']
    Au_gmac_sorted = _argsort_with_polar_tiebreak(pe_u_gmac)
    Au_gmac = sorted([int(i + 1) for i in Au_gmac_sorted[:ku]])
    overlap = len(set(Au_mi) & set(Au_gmac))

    log(f"  MI range: [{mi.min():.4f}, {mi.max():.4f}]")
    log(f"  Au_mi: {Au_mi}")
    log(f"  Au_gmac: {Au_gmac}")
    log(f"  Overlap: {overlap}/{ku}")

    # Save MI data
    mi_data = {"mi": mi.tolist(), "Au_mi": Au_mi, "Au_gmac": Au_gmac,
               "overlap": overlap, "ku": ku, "kv": kv}
    with open(os.path.join(OUT, f"mi_design_N{N}.json"), "w") as f:
        json.dump(mi_data, f, indent=2)

    # Save as npz design file
    np.savez(os.path.join(OUT, f"isi_mi_design_N{N}.npz"),
             u_error_rates=1.0 - mi,  # lower MI = higher "error rate" for sorting
             v_error_rates=pe_v_gmac,
             path_i=N, n_trials=50000, sigma2=SIGMA2, snr_db=6.0)

    del model_p1
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Retrain at target rate with MI-based frozen set
    # ══════════════════════════════════════════════════════════════
    log(f"\n--- Phase 3: Retrain with MI-based frozen set ---")
    torch.manual_seed(42)
    model_p3 = ChainedNPD_MAC(d=16, hidden=100, n_layers=2,
                               encoder_type='bigru', gru_layers=1)
    model_p3.to(device)

    # Stage 1
    log(f"  S1: {p3_s1_iters} iters")
    s1_loss = train_stage1(model_p3.stage1, ch, N, Au_mi, Av_mi, fu_mi_set,
                           iters=p3_s1_iters, batch=batch,
                           tag=f"p3_mi_s1_N{N}", eval_every=50000)
    log(f"  S1 done: best_loss={s1_loss:.4f}")

    # Stage 2
    log(f"  S2: {p3_s2_iters} iters")
    s2_loss = train_stage2(model_p3.stage2, ch, N, Au_mi, Av_mi, fv_mi_set,
                           iters=p3_s2_iters, batch=batch,
                           tag=f"p3_mi_s2_N{N}", eval_every=50000)
    log(f"  S2 done: best_loss={s2_loss:.4f}")

    # ══════════════════════════════════════════════════════════════
    # Eval
    # ══════════════════════════════════════════════════════════════
    log(f"\n--- Eval ---")
    bler_total, bler_u, bler_v = eval_chained_bler(
        model_p3, ch, N, Au_mi, Av_mi, fu_mi_set, fv_mi_set, n_cw=3000)
    log(f"  NPD+MI_design: BLER={bler_total:.4f} (U={bler_u:.4f} V={bler_v:.4f})")

    # SC baseline with GMAC
    from polar.decoder_trellis_mac_chained import bler_chained
    fu_gmac = {p: 0 for p in range(1, N+1) if p not in Au_gmac}
    fv_gmac = {p: 0 for p in range(1, N+1) if p not in Av_mi}
    sc_n_cw = 5000 if N <= 128 else 2000
    r_sc = bler_chained(ch, N, fu_gmac, fv_gmac, Au_gmac, Av_mi, sc_n_cw, seed=0)
    log(f"  SC+GMAC: BLER={r_sc['chained_bler']:.4f}")

    ratio = bler_total / max(r_sc['chained_bler'], 1e-9)
    log(f"\n  {'='*50}")
    log(f"  N={N} RESULTS:")
    log(f"    NPD + MI_design:   {bler_total:.4f}")
    log(f"    SC  + GMAC_design: {r_sc['chained_bler']:.4f}")
    log(f"    Ratio: {ratio:.2f}x")
    log(f"    MI overlap with GMAC: {overlap}/{ku}")
    log(f"  {'='*50}")

    all_results[str(N)] = {
        "npd_bler": bler_total, "npd_u_bler": bler_u, "npd_v_bler": bler_v,
        "sc_gmac_bler": r_sc["chained_bler"],
        "ratio": ratio, "overlap": overlap,
        "Au_mi": Au_mi, "Au_gmac": Au_gmac,
    }

    with open(os.path.join(OUT, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    del model_p3
    torch.cuda.empty_cache()

log(f"\n{'='*70}")
log(f"ALL DONE. Total: {(time.time()-total_t0)/3600:.1f}h")
log(f"{'='*70}")

log(f"\n{'N':<6} {'NPD+MI':<12} {'SC+GMAC':<12} {'Ratio':<10} {'Overlap'}")
for Ns, r in sorted(all_results.items(), key=lambda x: int(x[0])):
    log(f"{Ns:<6} {r['npd_bler']:<12.4f} {r['sc_gmac_bler']:<12.4f} {r['ratio']:<10.2f}x {r['overlap']}")
