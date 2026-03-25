"""
eval.py
=======
BER / BLER Monte Carlo evaluation pipeline for SC MAC polar codes.

Supports batched evaluation with TF/numpy vectorized encoding and
parallel decoding (via ProcessPoolExecutor or sequential).

Usage example
-------------
    from channels import BEMAC
    from design   import design_bemac, make_path
    from eval     import MACEval

    channel = BEMAC()
    eval_   = MACEval(channel, log_domain=False)   # BE-MAC: log_domain optional
    b       = make_path(N, path_i=N)               # path 0^N 1^N

    Au, Av, frozen_u, frozen_v, z_u, z_v = design_bemac(n, ku, kv=N)
    ber_u, ber_v, bler = eval_.run(N, b, Au, Av, frozen_u, frozen_v,
                                   n_codewords=500)
"""

import numpy as np
from time import time

from polar.encoder import polar_encode_batch, build_message_batch
from polar.decoder import decode_single, decode_batch


def _detect_path_i(N, b):
    """Detect path_i from path vector b. Returns -1 if unrecognised."""
    if len(b) != 2 * N:
        return -1
    pi = 0
    while pi < len(b) and b[pi] == 0:
        pi += 1
    if pi > N:
        return -1
    end_ones = pi + N
    if end_ones > 2 * N:
        return -1
    if (all(b[k] == 1 for k in range(pi, end_ones)) and
            all(b[k] == 0 for k in range(end_ones, 2 * N))):
        return pi
    return -1


class MACEval:
    """
    Monte Carlo BER / BLER evaluator for SC MAC polar codes.

    Parameters
    ----------
    channel    : MACChannel — BEMAC, ABNMAC, or GaussianMAC
    log_domain : bool — pass to decoder; set True for ABN-MAC N≥128
    n_workers  : int — parallel decode workers (1 = sequential)
    rng        : np.random.Generator or None — random state
    backend    : str — decoder backend selection:
                 'auto'        — use fastest available decoder for the path
                 'reference'   — force O(N²) decoder.py
                 'efficient'   — force efficient_decoder.py (extreme paths only)
                 'interleaved' — force decoder_interleaved.py (all paths)
    """

    def __init__(self, channel, log_domain: bool = True,
                 n_workers: int = 1, rng=None,
                 decoder_type: str = 'sc', L: int = 4,
                 backend: str = 'auto'):
        self.channel = channel
        self.log_domain = log_domain
        self.n_workers = n_workers
        self.rng = rng or np.random.default_rng()
        self.decoder_type = decoder_type
        self.L = L
        self.backend = backend

    # ── Core evaluation loop ──────────────────────────────────────────────

    def run(self, N: int, b: list, Au: list, Av: list,
            frozen_u: dict, frozen_v: dict,
            n_codewords: int = 500,
            batch_size: int = 25,
            verbose: bool = False) -> tuple:
        """
        Evaluate BER and BLER over n_codewords Monte Carlo trials.

        Parameters
        ----------
        N           : int — block length
        b           : list — path vector, length 2N
        Au          : list[int] — 1-indexed U info positions
        Av          : list[int] — 1-indexed V info positions
        frozen_u    : dict — U frozen positions
        frozen_v    : dict — V frozen positions
        n_codewords : int — total number of codewords to test
        batch_size  : int — decode batch size (balance speed vs memory)
        verbose     : bool — print periodic progress

        Returns
        -------
        ber_u, ber_v, bler : float
        """
        ku = len(Au)
        kv = len(Av)

        u_errors = 0
        v_errors = 0
        block_errors = 0
        n_done = 0
        t0 = time()

        while n_done < n_codewords:
            bs = min(batch_size, n_codewords - n_done)

            # Generate random info bits for both users
            U_info = self.rng.integers(0, 2, size=(bs, ku), dtype=np.int32)
            V_info = self.rng.integers(0, 2, size=(bs, kv), dtype=np.int32)

            # Build full message vectors (frozen bits = 0)
            U_msg = build_message_batch(N, U_info, Au)  # (bs, N)
            V_msg = build_message_batch(N, V_info, Av)  # (bs, N)

            # Polar encode both users
            X = polar_encode_batch(U_msg)  # (bs, N)
            Y = polar_encode_batch(V_msg)  # (bs, N)

            # Simulate channel — collect received vectors
            Z_list = []
            for k in range(bs):
                z_k = self.channel.sample_batch(X[k], Y[k])
                # Convert to list (handles int arrays for BE-MAC and
                # object arrays of tuples for ABN-MAC)
                z_k = z_k.tolist() if hasattr(z_k, 'tolist') else list(z_k)
                Z_list.append(z_k)

            # Decode — dispatch to appropriate backend
            decoded = self._decode_batch(
                N, Z_list, b, frozen_u, frozen_v)

            # Count errors on info positions only
            for k, (u_dec, v_dec) in enumerate(decoded):
                u_errs = sum(u_dec[p - 1] != U_msg[k, p - 1] for p in Au)
                v_errs = sum(v_dec[p - 1] != V_msg[k, p - 1] for p in Av)
                u_errors += u_errs
                v_errors += v_errs
                block_errors += (1 if u_errs > 0 or v_errs > 0 else 0)

            n_done += bs
            if verbose and time() - t0 > 30:
                ber_u = u_errors / max(1, n_done * ku)
                ber_v = v_errors / max(1, n_done * kv)
                bler = block_errors / n_done
                print(f"    {n_done}/{n_codewords}  ber_u={ber_u:.3e}  "
                      f"ber_v={ber_v:.3e}  bler={bler:.3e}")
                t0 = time()

        ber_u = u_errors / max(1, n_codewords * ku)
        ber_v = v_errors / max(1, n_codewords * kv)
        bler = block_errors / n_codewords
        return float(ber_u), float(ber_v), float(bler)

    # ── Decoder dispatch ───────────────────────────────────────────────

    def _decode_batch(self, N, Z_list, b, frozen_u, frozen_v):
        """Dispatch to the best available decoder for this (path, decoder_type, backend)."""
        path_i = _detect_path_i(N, b)
        is_extreme = path_i in (0, N)
        backend = self.backend

        # SC decoder
        if backend == 'auto':
            if is_extreme:
                backend = 'efficient'
            else:
                backend = 'interleaved'

        if backend == 'efficient' and is_extreme:
            from polar.efficient_decoder import decode_batch as eff_batch
            return eff_batch(
                N, Z_list, b, frozen_u, frozen_v,
                self.channel, self.log_domain, self.n_workers)
        elif backend == 'interleaved':
            from polar.decoder_interleaved import decode_batch as interleaved_batch
            return interleaved_batch(
                N, Z_list, b, frozen_u, frozen_v,
                self.channel, self.log_domain, self.n_workers)
        else:
            return decode_batch(
                N, Z_list, b, frozen_u, frozen_v,
                self.channel, self.log_domain, self.n_workers)


# ─────────────────────────────────────────────────────────────────────────────
#  Adaptive MC evaluation: increase n_codewords for reliable BLER estimates
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_mc_eval(N: int, b, Au, Av, frozen_u, frozen_v,
                     channel, log_domain: bool = True,
                     time_budget_s: float = 40.0,
                     min_codewords: int = 50,
                     decoder_type: str = 'sc', L: int = 4) -> tuple:
    """
    Run MC evaluation with a time budget per rate point.

    Measures the time for a small batch, then estimates how many codewords
    fit within the given time budget.

    Parameters
    ----------
    decoder_type : 'sc' or 'scl'
    L            : list size for SCL decoder (ignored for SC)

    Returns (ber_u, ber_v, bler, n_codewords_used).
    """
    from polar.decoder import decode_single

    # Time a single decode
    z_dummy = channel.sample_batch(np.zeros(N, dtype=np.int32),
                                   np.zeros(N, dtype=np.int32)).tolist()
    if isinstance(z_dummy[0], list):
        z_dummy = [tuple(zz) for zz in z_dummy]

    t0 = time()
    for _ in range(3):
        decode_single(N, z_dummy, b, frozen_u, frozen_v,
                      channel, log_domain)
    ms_per_decode = (time() - t0) / 3 * 1000

    n_codewords = max(min_codewords,
                      min(2000, int(time_budget_s * 1000 / max(1.0, ms_per_decode))))

    evaluator = MACEval(channel, log_domain=log_domain,
                        decoder_type=decoder_type, L=L)
    ber_u, ber_v, bler = evaluator.run(N, b, Au, Av, frozen_u, frozen_v,
                                       n_codewords=n_codewords,
                                       batch_size=25)
    return ber_u, ber_v, bler, n_codewords


# ─────────────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from polar.channels import BEMAC
    from polar.design import design_bemac, make_path

    print("=== MACEval self-test: BE-MAC N=16 ===")
    channel = BEMAC()
    N, n = 16, 4
    ku, kv = 4, 16
    b = make_path(N, path_i=N)  # 0^N 1^N

    Au, Av, frozen_u, frozen_v, z_u, z_v = design_bemac(n, ku, kv)
    print(f"  Au={Au}  Av={Av}")

    evaluator = MACEval(channel, log_domain=False, rng=np.random.default_rng(42))
    ber_u, ber_v, bler = evaluator.run(N, b, Au, Av, frozen_u, frozen_v,
                                       n_codewords=100, verbose=False)
    print(f"  N={N} ku={ku} kv={kv}:  ber_u={ber_u:.3e}  ber_v={ber_v:.3e}  "
          f"bler={bler:.3e}")
    print("  (expect ber_v≈0 since Z₀_v=0 for BE-MAC)")
