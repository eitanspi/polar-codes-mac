"""
decoder_parallel.py
===================
Numba JIT-compiled parallel SC decoder for MAC polar codes, Class B (path_i=N/2).

The entire 2N-step decode loop runs in compiled Numba code with zero Python
overhead per decoded bit. Compact flat O(N log N) storage avoids cache pressure.
Batch parallelism via prange across CPU cores.

IMPORTANT: All calc functions (_do_cl, _do_cr, _do_cp, _nav) are standalone
@njit functions taking (E, offsets, beta, N) explicitly. Do NOT use Numba
closures with mutable arrays — they produce silently incorrect results.

Public API:
    decode_parallel_single(N, log_W, b, frozen_u, frozen_v)
    decode_parallel_batch(N, log_W_batch, b, frozen_u, frozen_v)
"""

import numpy as np
from numba import njit, prange
from .encoder import bit_reversal_perm

_NEG_INF = -np.inf
_LOG_HALF = np.log(0.5)
_LOG_QUARTER = np.log(0.25)


@njit(cache=True)
def _lae(a, b):
    """Scalar logaddexp — compiled."""
    if a == -np.inf and b == -np.inf:
        return -np.inf
    if a >= b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


@njit(cache=True)
def _compute_offsets(N):
    """Compute flat storage offsets. Returns (offsets array, total size)."""
    offsets = np.zeros(2 * N, dtype=np.int64)
    total = np.int64(0)
    for beta in range(1, 2 * N):
        offsets[beta] = total
        level = 0
        tmp = beta
        while tmp > 1:
            tmp >>= 1
            level += 1
        total += (N >> level) * 4
    return offsets, total


@njit(cache=True)
def _esize(beta, N):
    """Number of tensor positions at edge beta."""
    level = 0
    tmp = beta
    while tmp > 1:
        tmp >>= 1
        level += 1
    return N >> level


@njit(cache=True)
def _do_cl(E, offsets, beta, N):
    """calcLeft at vertex beta — ALL L positions vectorized. Standalone (no closure)."""
    NINF = -np.inf
    L = _esize(2 * beta + 1, N)
    op = offsets[beta]; oL = offsets[2 * beta]; oR = offsets[2 * beta + 1]
    for i in range(L):
        j = i * 4; jL = (i + L) * 4
        t0 = E[op + j]; t1 = E[op + j + 1]; t2 = E[op + j + 2]; t3 = E[op + j + 3]
        b0 = E[op + jL]; b1 = E[op + jL + 1]; b2 = E[op + jL + 2]; b3 = E[op + jL + 3]
        r0 = E[oR + j]; r1 = E[oR + j + 1]; r2 = E[oR + j + 2]; r3 = E[oR + j + 3]
        n0 = b0 + r0; n1 = b1 + r1; n2 = b2 + r2; n3 = b3 + r3
        tot = _lae(_lae(n0, n1), _lae(n2, n3))
        if tot != NINF and np.isfinite(tot):
            n0 -= tot; n1 -= tot; n2 -= tot; n3 -= tot
        E[oL + j]     = _lae(_lae(t0 + n0, t1 + n1), _lae(t2 + n2, t3 + n3))
        E[oL + j + 1] = _lae(_lae(t1 + n0, t0 + n1), _lae(t3 + n2, t2 + n3))
        E[oL + j + 2] = _lae(_lae(t2 + n0, t3 + n1), _lae(t0 + n2, t1 + n3))
        E[oL + j + 3] = _lae(_lae(t3 + n0, t2 + n1), _lae(t1 + n2, t0 + n3))


@njit(cache=True)
def _do_cr(E, offsets, beta, N):
    """calcRight at vertex beta — ALL L positions vectorized. Standalone (no closure)."""
    NINF = -np.inf
    L = _esize(2 * beta, N)
    op = offsets[beta]; oL = offsets[2 * beta]; oR = offsets[2 * beta + 1]
    for i in range(L):
        j = i * 4; jL = (i + L) * 4
        l0 = E[oL + j]; l1 = E[oL + j + 1]; l2 = E[oL + j + 2]; l3 = E[oL + j + 3]
        t0 = E[op + j]; t1 = E[op + j + 1]; t2 = E[op + j + 2]; t3 = E[op + j + 3]
        b0 = E[op + jL]; b1 = E[op + jL + 1]; b2 = E[op + jL + 2]; b3 = E[op + jL + 3]
        c0 = _lae(_lae(l0 + t0, l1 + t1), _lae(l2 + t2, l3 + t3))
        c1 = _lae(_lae(l1 + t0, l0 + t1), _lae(l3 + t2, l2 + t3))
        c2 = _lae(_lae(l2 + t0, l3 + t1), _lae(l0 + t2, l1 + t3))
        c3 = _lae(_lae(l3 + t0, l2 + t1), _lae(l1 + t2, l0 + t3))
        n0 = b0 + c0; n1 = b1 + c1; n2 = b2 + c2; n3 = b3 + c3
        tot = _lae(_lae(n0, n1), _lae(n2, n3))
        if tot != NINF and np.isfinite(tot):
            n0 -= tot; n1 -= tot; n2 -= tot; n3 -= tot
        E[oR + j] = n0; E[oR + j + 1] = n1; E[oR + j + 2] = n2; E[oR + j + 3] = n3


@njit(cache=True)
def _do_cp(E, offsets, beta, N):
    """calcParent at vertex beta — ALL L positions vectorized. Standalone (no closure)."""
    L = _esize(2 * beta, N)
    op = offsets[beta]; oL = offsets[2 * beta]; oR = offsets[2 * beta + 1]
    for i in range(L):
        j = i * 4; jL = (i + L) * 4
        l0 = E[oL + j]; l1 = E[oL + j + 1]; l2 = E[oL + j + 2]; l3 = E[oL + j + 3]
        r0 = E[oR + j]; r1 = E[oR + j + 1]; r2 = E[oR + j + 2]; r3 = E[oR + j + 3]
        E[op + j]     = _lae(_lae(l0 + r0, l1 + r1), _lae(l2 + r2, l3 + r3))
        E[op + j + 1] = _lae(_lae(l1 + r0, l0 + r1), _lae(l3 + r2, l2 + r3))
        E[op + j + 2] = _lae(_lae(l2 + r0, l3 + r1), _lae(l0 + r2, l1 + r3))
        E[op + j + 3] = _lae(_lae(l3 + r0, l2 + r1), _lae(l1 + r2, l0 + r3))
        E[op + jL] = r0; E[op + jL + 1] = r1; E[op + jL + 2] = r2; E[op + jL + 3] = r3


@njit(cache=True)
def _nav(E, offsets, dec_head, target, N, path_buf):
    """Navigate the tree, returning new dec_head. Standalone (no closure)."""
    if dec_head == target:
        return dec_head
    up_n = 0; dn_n = 0; c = dec_head; t = target
    while c != t:
        if c > t:
            c >>= 1; path_buf[up_n] = c; up_n += 1
        else:
            path_buf[1000 + dn_n] = t; dn_n += 1; t >>= 1
    for i in range(dn_n):
        path_buf[up_n + i] = path_buf[1000 + dn_n - 1 - i]
    for pi in range(up_n + dn_n):
        b = path_buf[pi]
        if dec_head == b >> 1:
            if b & 1 == 0:
                _do_cl(E, offsets, dec_head, N)
            else:
                _do_cr(E, offsets, dec_head, N)
            dec_head = b
        elif b == dec_head >> 1:
            _do_cp(E, offsets, dec_head, N)
            dec_head = b
    return dec_head


@njit(cache=True)
def _decode_one(root_flat, N, n, offsets, total_size, frozen_u, frozen_v):
    """
    SC decode one codeword for Class B (path_i = N/2).
    Compact flat storage — O(N log N) memory.

    root_flat: length 4*N flat array (N positions x 4 values each)
    offsets: precomputed edge offsets into flat array
    total_size: total flat array size
    frozen_u, frozen_v: (N+1,) int8 arrays, -1 = not frozen
    """
    half = N >> 1
    LH = np.log(0.5)
    LQ = np.log(0.25)
    NINF = -np.inf

    E = np.full(total_size, LQ, dtype=np.float64)
    o1 = offsets[1]
    for i in range(N * 4):
        E[o1 + i] = root_flat[i]

    u_dec = np.zeros(N + 1, dtype=np.int32)
    v_dec = np.zeros(N + 1, dtype=np.int32)
    u_known = np.zeros(N + 1, dtype=np.int8)
    v_known = np.zeros(N + 1, dtype=np.int8)
    dec_head = np.int64(1)
    path_buf = np.zeros(2000, dtype=np.int64)

    i_u = 0; i_v = 0
    for step in range(2 * N):
        if step < half:
            gamma = 0
        elif step < half + N:
            gamma = 1
        else:
            gamma = 0

        if gamma == 0:
            i_u += 1; i_t = i_u
        else:
            i_v += 1; i_t = i_v

        le = i_t + N - 1; tv = le >> 1
        dec_head = _nav(E, offsets, dec_head, tv, N, path_buf)

        o = offsets[le]
        t0 = E[o]; t1 = E[o + 1]; t2 = E[o + 2]; t3 = E[o + 3]

        if le & 1 == 0:
            _do_cl(E, offsets, tv, N)
        else:
            _do_cr(E, offsets, tv, N)

        td0 = E[o]; td1 = E[o + 1]; td2 = E[o + 2]; td3 = E[o + 3]
        r0 = td0 + t0; r1 = td1 + t1; r2 = td2 + t2; r3 = td3 + t3
        tot = _lae(_lae(r0, r1), _lae(r2, r3))
        if tot != NINF and np.isfinite(tot):
            r0 -= tot; r1 -= tot; r2 -= tot; r3 -= tot

        if gamma == 0:
            fv = frozen_u[i_t]
            if fv >= 0:
                bit = np.int32(fv)
            else:
                p0 = _lae(r0, r1); p1 = _lae(r2, r3)
                bit = np.int32(1) if p1 > p0 else np.int32(0)
            u_dec[i_t] = bit; u_known[i_t] = 1
        else:
            fv = frozen_v[i_t]
            if fv >= 0:
                bit = np.int32(fv)
            else:
                p0 = _lae(r0, r2); p1 = _lae(r1, r3)
                bit = np.int32(1) if p1 > p0 else np.int32(0)
            v_dec[i_t] = bit; v_known[i_t] = 1

        ub = u_dec[i_t] if u_known[i_t] else np.int32(-1)
        vb = v_dec[i_t] if v_known[i_t] else np.int32(-1)
        E[o] = NINF; E[o + 1] = NINF; E[o + 2] = NINF; E[o + 3] = NINF
        if ub >= 0 and vb >= 0:
            E[o + ub * 2 + vb] = 0.0
        elif ub >= 0:
            E[o + ub * 2] = LH; E[o + ub * 2 + 1] = LH
        elif vb >= 0:
            E[o + vb] = LH; E[o + 2 + vb] = LH
        else:
            E[o] = LQ; E[o + 1] = LQ; E[o + 2] = LQ; E[o + 3] = LQ

    return u_dec[1:], v_dec[1:]


@njit(cache=True, parallel=True)
def _decode_batch(roots, N, n, offsets, total_size, frozen_u, frozen_v, batch):
    """Parallel batch decode using prange across CPU cores."""
    u_all = np.zeros((batch, N), dtype=np.int32)
    v_all = np.zeros((batch, N), dtype=np.int32)
    for i in prange(batch):
        u_all[i], v_all[i] = _decode_one(
            roots[i], N, n, offsets, total_size, frozen_u, frozen_v)
    return u_all, v_all


# ─────────────────────────────────────────────────────────────────────────────
#  Python helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_roots(log_W_batch, n):
    br = bit_reversal_perm(n)
    root = log_W_batch[:, br].copy()
    totals = np.logaddexp(
        np.logaddexp(root[:, :, 0, 0], root[:, :, 0, 1]),
        np.logaddexp(root[:, :, 1, 0], root[:, :, 1, 1]))
    finite = np.isfinite(totals)
    root[finite] -= totals[finite, np.newaxis, np.newaxis]
    batch, N = root.shape[0], root.shape[1]
    return root.reshape(batch, N * 4)


def _frozen_arr(frozen_dict, N):
    arr = np.full(N + 1, np.int8(-1))
    for pos, val in frozen_dict.items():
        arr[pos] = np.int8(val)
    return arr


# JIT warmup
_offs, _tsz = _compute_offsets(2)
_fu = np.full(3, np.int8(-1)); _fu[1] = 0
_fv = np.full(3, np.int8(-1))
_wr = np.full(8, _LOG_QUARTER, dtype=np.float64)
try:
    _decode_one(_wr, 2, 1, _offs, _tsz, _fu, _fv)
    _decode_batch(_wr.reshape(1, 8), 2, 1, _offs, _tsz, _fu, _fv, 1)
except Exception:
    pass


def decode_parallel_single(N, log_W, b, frozen_u, frozen_v):
    """SC MAC decoder for Class B, single codeword."""
    n = N.bit_length() - 1
    roots = _prepare_roots(log_W[np.newaxis], n)
    offsets, total_size = _compute_offsets(N)
    fu = _frozen_arr(frozen_u, N)
    fv = _frozen_arr(frozen_v, N)
    u, v = _decode_one(roots[0], N, n, offsets, total_size, fu, fv)
    return u.tolist(), v.tolist()


def decode_parallel_batch(N, log_W_batch, b, frozen_u, frozen_v):
    """SC MAC decoder for Class B, batch of codewords."""
    n = N.bit_length() - 1
    batch = log_W_batch.shape[0]
    roots = _prepare_roots(log_W_batch, n)
    offsets, total_size = _compute_offsets(N)
    fu = _frozen_arr(frozen_u, N)
    fv = _frozen_arr(frozen_v, N)
    return _decode_batch(roots, N, n, offsets, total_size, fu, fv, batch)
