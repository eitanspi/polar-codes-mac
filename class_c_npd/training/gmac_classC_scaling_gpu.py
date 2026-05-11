#!/usr/bin/env python3
"""GMAC Class C NPD scaling to N=512, 1024 on GPU with warm-start from N=256."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np, torch
from class_c_npd.models.npd_single_user import NPDSingleUser
from class_c_npd.training.train_stage import generate_stage1_batch, generate_stage2_batch, evaluate_stage
from class_c_npd.channels.mac_channel import build_channel
from class_c_npd.channels.frozen_sets import load_class_c_design
from class_c_npd.eval.chain_eval import chain_evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
ch = build_channel("gmac", sigma2=10**(-6.0/10))
results = {}
prev_s1 = "class_c_npd/results/npd_design_p1_N256_best.pt"
prev_s2 = "class_c_npd/results/npd_design_p3_N256_best.pt"

for N in [512, 1024]:
    n = int(np.log2(N))
    ku = max(1, round(0.5 * 0.4645 * N))
    kv = max(1, round(0.5 * 0.9119 * N))
    Au, Av, fu, fv, pe_u, pe_v = load_class_c_design("gmac", n, snr_db=6.0, ku=ku, kv=kv)
    print(f"\nN={N}, ku={len(Au)}, kv={len(Av)}", flush=True)

    for stage in [1, 2]:
        gen_fn = generate_stage1_batch if stage == 1 else generate_stage2_batch
        info = Au if stage == 1 else Av
        frozen = fu if stage == 1 else fv
        other = Av if stage == 1 else Au
        z_dim = ch.stage1_feature_dim if stage == 1 else ch.stage2_feature_dim
        sched = {512: (100000, 16, 5e-5), 1024: (50000, 8, 3e-5)}
        iters, batch, lr = sched[N]
        if stage == 2:
            iters = iters // 3
            lr = lr * 2

        model = NPDSingleUser(d=16, hidden=64, n_layers=2, z_dim=z_dim,
                               use_analytical_training=True).to(device)
        warm = prev_s1 if stage == 1 else prev_s2
        if warm and os.path.exists(warm):
            try:
                ckpt = torch.load(warm, weights_only=False, map_location=device)
                model.load_state_dict(ckpt["state_dict"])
                print(f"  S{stage}: warm-started from {warm}", flush=True)
            except Exception as e:
                print(f"  S{stage}: warm-start failed: {e}", flush=True)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.default_rng(42 + stage)
        t0 = time.time()
        best_bler = 1.0
        eval_every = 5000
        ckpt_path = f"class_c_npd/results/gmac_classC_s{stage}_N{N}.pt"

        model.train()
        for it in range(1, iters + 1):
            _, features, cw = gen_fn(ch, N, info, batch, rng, other)
            ft = torch.from_numpy(features).float().to(device)
            if ft.dim() == 2:
                ft = ft.unsqueeze(-1)
            emb = model.encode_channel(ft)
            loss = model.fast_ce(emb, torch.from_numpy(cw).long().to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if it % eval_every == 0:
                elapsed = (time.time() - t0) / 60
                # Save checkpoint every eval step (skip slow decode eval)
                ckpt_iter_path = f"class_c_npd/results/gmac_classC_s{stage}_N{N}_iter{it}.pt"
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "z_dim": z_dim, "N": N, "Au": Au, "Av": Av,
                            "iter": it, "loss": loss.item()}, ckpt_iter_path)
                # Also save as "best" (latest = best for now)
                torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "z_dim": z_dim, "N": N, "Au": Au, "Av": Av}, ckpt_path)
                print(f"  S{stage} [{it:>6}/{iters}] loss={loss.item():.4f} "
                      f"{elapsed:.1f}min  saved {ckpt_iter_path}", flush=True)

        if stage == 1:
            prev_s1 = ckpt_path if os.path.exists(ckpt_path) else prev_s1
        else:
            prev_s2 = ckpt_path if os.path.exists(ckpt_path) else prev_s2

    # Skip slow chained eval — checkpoints saved, eval later
    results[N] = {"N": N, "ku": len(Au), "kv": len(Av), "status": "trained"}
    with open("class_c_npd/results/gmac_classC_scaling_gpu.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  N={N} training done. Eval checkpoints saved.", flush=True)

print("\nDone.", flush=True)
