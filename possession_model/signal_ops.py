"""Shared mathematical primitives and O(N) temporal window operators.

All window averages and event timers operate in O(N) time via prefix sums and respect
half-block boundaries (and optional stoppage barriers) without cross-period leakage.
"""

from typing import List, Tuple
import numpy as np

HOME = 0
AWAY = 1
DEAD = 2
NOBODY = -1
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds transform log(p / (1 - p)) with numerical clamping."""
    pc = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(pc / (1.0 - pc))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid 1 / (1 + exp(-z))."""
    zc = np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-zc))


def softplus(z: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 + exp(z)) = -log(sigmoid(-z))."""
    zc = np.asarray(z, dtype=np.float64)
    return np.maximum(zc, 0.0) + np.log1p(np.exp(-np.abs(zc)))


def box_mean_1d(arr: np.ndarray, left: int, right: int, fill: float = 0.0) -> np.ndarray:
    """O(N) finite-aware sliding mean over [t - left .. t + right]."""
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    valid = np.isfinite(arr)
    vals = np.where(valid, arr, 0.0).astype(np.float64)
    cs_v = np.concatenate(([0.0], np.cumsum(vals)))
    cs_c = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    idx = np.arange(n, dtype=np.int64)
    lo = np.maximum(0, idx - left)
    hi = np.minimum(n, idx + right + 1)
    cnt = cs_c[hi] - cs_c[lo]
    out = np.full(n, fill, dtype=np.float32)
    ok = cnt > 0
    out[ok] = ((cs_v[hi[ok]] - cs_v[lo[ok]]) / cnt[ok]).astype(np.float32)
    return out


def box_mean_blocks(
    arr: np.ndarray,
    left: int,
    right: int,
    half_blocks: List[Tuple[int, int, int]],
    fill: float = 0.0,
) -> np.ndarray:
    """Block-isolated O(N) sliding mean over [t - left .. t + right] within each half."""
    out = np.full(len(arr), fill, dtype=np.float32)
    for _, s_f, e_f in half_blocks:
        out[s_f : e_f + 1] = box_mean_1d(arr[s_f : e_f + 1], left, right, fill=fill)
    return out


def compute_barrier_bounds(barrier_1d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For a 1D boolean barrier mask of length M, return (prev_barrier_idx, next_barrier_idx) in O(M)."""
    m = len(barrier_1d)
    bar_idx = np.flatnonzero(barrier_1d)
    if len(bar_idx) == 0:
        return np.full(m, -1, dtype=np.int32), np.full(m, m, dtype=np.int32)

    grid = np.arange(m, dtype=np.int32)
    pos_r = np.searchsorted(bar_idx, grid, side="left")
    next_b = np.where(pos_r < len(bar_idx), bar_idx[np.minimum(pos_r, len(bar_idx) - 1)], m).astype(np.int32)

    pos_l = np.searchsorted(bar_idx, grid, side="right") - 1
    prev_b = np.where(pos_l >= 0, bar_idx[np.maximum(pos_l, 0)], -1).astype(np.int32)
    return prev_b, next_b


def barrier_box_mean_blocks(
    arr: np.ndarray,
    left: int,
    right: int,
    barrier: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    fill: float = 0.0,
) -> np.ndarray:
    """O(N) sliding mean over [t - left .. t + right] truncated at `barrier` frames so no information
    leaks across dead-ball stoppages.
    """
    out = np.full(len(arr), fill, dtype=np.float32)
    for _, s_f, e_f in half_blocks:
        sub = arr[s_f : e_f + 1]
        bar = barrier[s_f : e_f + 1]
        m = len(sub)
        if m == 0:
            continue
        prev_b, next_b = compute_barrier_bounds(bar)
        valid = np.isfinite(sub) & (~bar)
        vals = np.where(valid, sub, 0.0).astype(np.float64)
        cs_v = np.concatenate(([0.0], np.cumsum(vals)))
        cs_c = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))

        idx = np.arange(m, dtype=np.int32)
        lo = np.maximum(np.maximum(0, idx - left), prev_b + 1)
        hi = np.minimum(np.minimum(m, idx + right + 1), next_b)
        hi = np.maximum(lo, hi)
        cnt = cs_c[hi] - cs_c[lo]
        res = np.full(m, fill, dtype=np.float32)
        ok = cnt > 0
        res[ok] = ((cs_v[hi[ok]] - cs_v[lo[ok]]) / cnt[ok]).astype(np.float32)
        out[s_f : e_f + 1] = res
    return out


def barrier_event_timers(
    event_mask: np.ndarray,
    barrier: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    fps: float = 25.0,
    horizon_s: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dt_since_prev_s, dt_until_next_s) clipped at `horizon_s` and blocked by `barrier`."""
    n = len(event_mask)
    dt_prev = np.full(n, horizon_s, dtype=np.float32)
    dt_next = np.full(n, horizon_s, dtype=np.float32)

    for _, s_f, e_f in half_blocks:
        ev = event_mask[s_f : e_f + 1] & (~barrier[s_f : e_f + 1])
        bar = barrier[s_f : e_f + 1]
        m = len(ev)
        if m == 0:
            continue
        prev_b, next_b = compute_barrier_bounds(bar)
        ev_idx = np.flatnonzero(ev)
        if len(ev_idx) == 0:
            continue

        grid = np.arange(m, dtype=np.int32)
        pos_l = np.searchsorted(ev_idx, grid, side="right") - 1
        cand_prev = np.where(pos_l >= 0, ev_idx[np.maximum(pos_l, 0)], -1)
        ok_p = (cand_prev >= 0) & (cand_prev > prev_b) & (~bar)
        p_out = np.full(m, horizon_s, dtype=np.float32)
        p_out[ok_p] = np.minimum(horizon_s, (grid[ok_p] - cand_prev[ok_p]) / fps)

        pos_r = np.searchsorted(ev_idx, grid, side="left")
        cand_next = np.where(pos_r < len(ev_idx), ev_idx[np.minimum(pos_r, len(ev_idx) - 1)], m)
        ok_n = (cand_next < m) & (cand_next < next_b) & (~bar)
        n_out = np.full(m, horizon_s, dtype=np.float32)
        n_out[ok_n] = np.minimum(horizon_s, (cand_next[ok_n] - grid[ok_n]) / fps)

        dt_prev[s_f : e_f + 1] = p_out
        dt_next[s_f : e_f + 1] = n_out

    return dt_prev, dt_next


def runs_of_labels(labels: np.ndarray) -> List[Tuple[int, int, str]]:
    """Decompose a 1D label array into contiguous (start_idx, end_exclusive, label) tuples."""
    if len(labels) == 0:
        return []
    chg = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], chg, [len(labels)]))
    return [(int(a), int(b), str(labels[a])) for a, b in zip(bounds[:-1], bounds[1:])]
