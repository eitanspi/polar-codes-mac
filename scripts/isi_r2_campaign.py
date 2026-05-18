"""ISI r=2 MAC NPD campaign on cluster GPU.
Channel: Z_i = s_x[i] + s_y[i] + h1*(...[i-1]) + h2*(...[i-2]) + W_i, |S|=16.
NPD trained at rate-1 + MI design, then target-rate eval. Optionally compares
to joint trellis SC (much slower at |S|=16 but still tractable for moderate N)."""
import sys, os, json, time
sys.path.insert(0, "/gpfs0/bgu-haimp/users/eitansp/polar_project")
import numpy as np
import torch

from neural.npd_memory_mac import ChainedNPD_MAC
from polar.channels_memory_new import ISIMAC2
from polar.encoder import polar_encode_batch, bit_reversal_perm
from isi_campaign import (
    train_stage1_rate1, train_stage2_rate1, measure_mi, log as _log_,
)
from npd_batched_reeval import batched_eval
from polar.design_mc import design_from_file
from polar.decoder_trellis import decode_single
from polar.design import make_path

device = torch.device("cuda")
SIGMA2 = 10**(-0.6)
H1, H2 = 0.3, 0.15   # similar relative weights to ISIMAC h=0.3 single-tap
ch = ISIMAC2(sigma2=SIGMA2, h1=H1, h2=H2)

OUT = "/gpfs0/bgu-haimp/users/eitansp/polar_project/class_c_npd/results/isi_r2_campaign"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "log.txt"); RES = os.path.join(OUT, "results.json")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")

D = 16; HIDDEN = 100; N_LAYERS = 2

CONFIGS = [
    (16,    4,   7, 4,  50_000, 15_000, 20_000),
    (32,    7,  15, 5,  75_000, 20_000, 20_000),
    (64,   15,  29, 6, 100_000, 25_000, 30_000),
    (128,  30,  58, 7, 100_000, 30_000, 30_000),
    (256,  59, 117, 8,  80_000, 25_000, 30_000),
    (512, 119, 233, 9,  80_000, 30_000, 30_000),
    (1024,239, 467,10,  50_000, 15_000, 20_000),
]

def make_model():
    return ChainedNPD_MAC(d=D, hidden=HIDDEN, n_layers=N_LAYERS,
                         encoder_type='bigru', gru_layers=1).to(device)

def main():
    log("=" * 60); log(f"ISI r=2 NPD campaign (h1={H1}, h2={H2}, SNR=6dB)")
    results = json.load(open(RES)) if os.path.exists(RES) else {}
    prev_s1 = None; prev_s2 = None
    for (N, ku, kv, n, s1i, s2i, ncw) in CONFIGS:
        key = str(N)
        if key in results and results[key].get("done"):
            log(f"  N={N} done, skip")
            ndir = os.path.join(OUT, f"N{N}")
            prev_s1 = os.path.join(ndir, "s1.pt"); prev_s2 = os.path.join(ndir, "s2.pt")
            continue
        log(f"=== N={N} (ku={ku}, kv={kv}) ===")
        model = make_model()
        # Warm-start chain
        if prev_s1 and os.path.exists(prev_s1):
            sd = torch.load(prev_s1, map_location='cpu', weights_only=False)
            sd = sd.get('state_dict', sd)
            model.stage1.load_state_dict(sd)
            log(f"  S1 warm-started from prev N")
        if prev_s2 and os.path.exists(prev_s2):
            sd = torch.load(prev_s2, map_location='cpu', weights_only=False)
            sd = sd.get('state_dict', sd)
            model.stage2.load_state_dict(sd)
            log(f"  S2 warm-started from prev N")
        ndir = os.path.join(OUT, f"N{N}"); os.makedirs(ndir, exist_ok=True)
        s1p = os.path.join(ndir, "s1.pt"); s2p = os.path.join(ndir, "s2.pt")

        log(f"  S1 {s1i}"); train_stage1_rate1(model, ch, N, n, device, s1i, s1p, progress_every=20_000)
        log(f"  S2 {s2i}"); train_stage2_rate1(model, ch, N, n, device, s2i, s2p, progress_every=10_000)
        log(f"  MI design")
        Au, _ = measure_mi(model, ch, N, n, device, stage=1, k=ku)
        Av, _ = measure_mi(model, ch, N, n, device, stage=2, k=kv)
        log(f"  Eval {ncw}")
        bs = 64 if N <= 128 else (32 if N <= 512 else 16)
        r = batched_eval(model, ch, N, n, Au, Av, ncw, batch_size=bs, seed=88000+N)
        log(f"  N={N} NPD r=2 BLER = {r['bler_chained']:.6f} ({r['errs_total']}/{r['n_cw']})")
        results[key] = dict(N=N, ku=ku, kv=kv, Au=Au, Av=Av,
                           h1=H1, h2=H2, eval=r, done=True)
        json.dump(results, open(RES, "w"), indent=2)
        prev_s1, prev_s2 = s1p, s2p
    log("DONE")

if __name__ == "__main__":
    main()
