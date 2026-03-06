"""
_decoder_scl_base.py
====================
O(L·N²) SCL MAC decoder using array-based internals from _decoder_numba.

Internal module — users should import from decoder_scl.py instead.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from ._decoder_numba import (
    _build_z_tree,
    _coord_prob_u_log,
    _coord_prob_v_log,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-sample SCL MAC decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_single_list(N: int, z, b: list, frozen_u: dict, frozen_v: dict,
                       channel, log_domain: bool = True, L: int = 4):
    """
    SCL MAC decoder for one received vector z^N.
    Same API as decoder_scl.decode_single_list — drop-in replacement.
    """
    if not log_domain:
        raise ValueError("SCL decoder requires log_domain=True")

    z_tree = _build_z_tree(list(z))
    cache = {}

    # Each path: (u_hat_arr, v_hat_arr, path_metric, i, j)
    # u_hat / v_hat are 1-indexed np.int8 arrays of size N+1
    paths = [(np.zeros(N + 1, dtype=np.int8),
              np.zeros(N + 1, dtype=np.int8),
              0.0, 0, 0)]

    for k in range(1, 2 * N + 1):
        bk = b[k - 1]

        if bk == 0:
            # U step: decide u_{i+1}
            new_paths = []

            for (u_hat, v_hat, metric, i, j) in paths:
                i_new = i + 1
                is_frozen = i_new in frozen_u

                p0 = _coord_prob_u_log(N, i_new, j, z_tree, u_hat, v_hat,
                                       0, channel, cache)
                p1 = _coord_prob_u_log(N, i_new, j, z_tree, u_hat, v_hat,
                                       1, channel, cache)

                if is_frozen:
                    fval = frozen_u[i_new]
                    u_hat[i_new] = fval
                    m = metric + (p0 if fval == 0 else p1)
                    new_paths.append((u_hat, v_hat, m, i_new, j))
                else:
                    u_hat_0 = u_hat.copy()
                    u_hat_0[i_new] = 0
                    new_paths.append((u_hat_0, v_hat.copy(), metric + p0,
                                      i_new, j))

                    u_hat_1 = u_hat.copy()
                    u_hat_1[i_new] = 1
                    new_paths.append((u_hat_1, v_hat.copy(), metric + p1,
                                      i_new, j))

        else:
            # V step: decide v_{j+1}
            new_paths = []

            for (u_hat, v_hat, metric, i, j) in paths:
                j_new = j + 1
                is_frozen = j_new in frozen_v

                p0 = _coord_prob_v_log(N, i, j_new, z_tree, u_hat, v_hat,
                                       0, channel, cache)
                p1 = _coord_prob_v_log(N, i, j_new, z_tree, u_hat, v_hat,
                                       1, channel, cache)

                if is_frozen:
                    fval = frozen_v[j_new]
                    v_hat[j_new] = fval
                    m = metric + (p0 if fval == 0 else p1)
                    new_paths.append((u_hat, v_hat, m, i, j_new))
                else:
                    v_hat_0 = v_hat.copy()
                    v_hat_0[j_new] = 0
                    new_paths.append((u_hat.copy(), v_hat_0, metric + p0,
                                      i, j_new))

                    v_hat_1 = v_hat.copy()
                    v_hat_1[j_new] = 1
                    new_paths.append((u_hat.copy(), v_hat_1, metric + p1,
                                      i, j_new))

        # Prune to best L paths
        if len(new_paths) > L:
            new_paths.sort(key=lambda x: x[2], reverse=True)
            new_paths = new_paths[:L]

        # Metric normalization
        max_metric = max(p[2] for p in new_paths)
        if max_metric != -np.inf:
            paths = [(u, v, m - max_metric, i, j)
                     for (u, v, m, i, j) in new_paths]
        else:
            paths = new_paths

    # Return best path
    best = max(paths, key=lambda x: x[2])
    u_hat, v_hat = best[0], best[1]
    u_dec = [int(u_hat[k]) for k in range(1, N + 1)]
    v_dec = [int(v_hat[k]) for k in range(1, N + 1)]
    return u_dec, v_dec


# ─────────────────────────────────────────────────────────────────────────────
#  Batch SCL decoder
# ─────────────────────────────────────────────────────────────────────────────

def _decode_list_worker(args):
    N, z, b, frozen_u, frozen_v, channel, log_domain, L = args
    return decode_single_list(N, z, b, frozen_u, frozen_v, channel, log_domain, L)


def decode_batch_list(N: int, Z_list, b: list, frozen_u: dict, frozen_v: dict,
                      channel, log_domain: bool = True, L: int = 4,
                      n_workers: int = 1) -> list:
    if n_workers <= 1 or len(Z_list) <= 1:
        return [decode_single_list(N, z, b, frozen_u, frozen_v,
                                   channel, log_domain, L)
                for z in Z_list]

    args = [(N, z, b, frozen_u, frozen_v, channel, log_domain, L)
            for z in Z_list]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_decode_list_worker, args,
                                    chunksize=max(1, len(Z_list) // n_workers)))
    return results
