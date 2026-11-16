"""Per-frame features, the same for the external matches and for Barça–Athletic.

A match (the dictionary of models/epts_reader.py) becomes a table with one row per frame and about 180
columns. Everything is computed from the tracking alone, which is anonymous: player positions with the
team of each slot, and the ball when it is visible.

Coordinates are flipped so that the home team always attacks towards positive x: the direction is
derived from the data of each half (where the goalkeepers stand), so it works for every source.

Time windows (past, future and centred) skip missing values and never cross half-time or the stretches
outside the halves.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epts_reader as fc  # noqa: E402  (models/epts_reader.py)
from constants import (  # noqa: E402
    FPS, SPEED_HALF, WALK, NEAR, TOUCH, WINDOWS,
    ASYMMETRIC_WINDOWS, TOUCH_COUNT_WINDOWS,
    GAP_CLIP_FRAMES, NO_BALL_INITIAL_OFFSET, MAX_STREAK,
    STREAK_RESET_GAP, BALL_INTERP_MAX_GAP,
    FIELD_BOUND_TOLERANCE, PERCENTILE_VELOCITY,
)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------

def _speed(a, period):
    """Speed (m/s) from differences centred on ±SPEED_HALF frames, NaN if an end is missing or the half changes."""
    k = SPEED_HALF
    out = np.full(a.shape, np.nan, dtype=np.float32)
    if a.shape[0] <= 2 * k:
        return out
    same = (period[2 * k:] == period[:-2 * k]) & (period[2 * k:] > 0)
    d = (a[2 * k:] - a[:-2 * k]) * (FPS / (2.0 * k))
    if d.ndim == 2:
        d[~same] = np.nan
    else:
        d = np.where(same, d, np.nan)
    out[k:-k] = d
    return out


def _segments(period):
    """[(start, end)] of the stretches with the same half > 0."""
    p = period.astype(np.int16)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], p, [0]))) != 0)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if p[a] > 0:
            out.append((a, b))
    return out


def _window_mean(f, segs, before, after):
    """Mean of the non-missing values in [t-before, t+after], within each half. NaN if there is nothing."""
    out = np.full(f.shape, np.nan, dtype=np.float32)
    for a, b in segs:
        x = f[a:b].astype(np.float64)
        ok = ~np.isnan(x)
        cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
        cn = np.concatenate(([0], np.cumsum(ok)))
        n = b - a
        t = np.arange(n)
        lo = np.clip(t - before, 0, n)
        hi = np.clip(t + after + 1, 0, n)
        cnt = cn[hi] - cn[lo]
        with np.errstate(invalid='ignore', divide='ignore'):
            out[a:b] = np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)
    return out


def _gap_position(visible, segs):
    """For each frame: frames since the ball was last seen and frames until it is seen again (within the half)."""
    since = np.full(visible.shape, np.nan, dtype=np.float32)
    until = np.full(visible.shape, np.nan, dtype=np.float32)
    for a, b in segs:
        v = visible[a:b]
        n = b - a
        idx = np.arange(n)
        last = np.maximum.accumulate(np.where(v, idx, -1))
        nxt = np.minimum.accumulate(np.where(v, idx, n)[::-1])[::-1]
        s = np.where(last >= 0, idx - last, idx + 1 + NO_BALL_INITIAL_OFFSET).astype(np.float32)   # no ball since the start
        u = np.where(nxt < n, nxt - idx, n - idx + NO_BALL_INITIAL_OFFSET).astype(np.float32)
        since[a:b] = np.minimum(s, GAP_CLIP_FRAMES)
        until[a:b] = np.minimum(u, GAP_CLIP_FRAMES)
    return since, until


def _window_sum(f, segs, before, after):
    """Sum over [t-before, t+after], within each half."""
    out = np.zeros(f.shape, dtype=np.float32)
    for a, b in segs:
        cs = np.concatenate(([0.0], np.cumsum(np.nan_to_num(f[a:b].astype(np.float64)))))
        n = b - a
        t = np.arange(n)
        out[a:b] = cs[np.clip(t + after + 1, 0, n)] - cs[np.clip(t - before, 0, n)]
    return out


def _last_next(touch, segs):
    """Team of the last touch (+1 home, -1 away) and frames since; team of the next touch and frames until."""
    T = len(touch)
    last = np.full(T, np.nan, dtype=np.float32)
    since = np.full(T, np.nan, dtype=np.float32)
    nxt = np.full(T, np.nan, dtype=np.float32)
    until = np.full(T, np.nan, dtype=np.float32)
    for a, b in segs:
        v = touch[a:b]
        n = b - a
        idx = np.arange(n)
        li = np.maximum.accumulate(np.where(v != 0, idx, -1))
        ni = np.minimum.accumulate(np.where(v != 0, idx, n)[::-1])[::-1]
        okl, okn = li >= 0, ni < n
        last[a:b] = np.where(okl, v[np.maximum(li, 0)], np.nan)
        since[a:b] = np.where(okl, np.minimum(idx - li, GAP_CLIP_FRAMES), np.nan)
        nxt[a:b] = np.where(okn, v[np.minimum(ni, n - 1)], np.nan)
        until[a:b] = np.where(okn, np.minimum(ni - idx, GAP_CLIP_FRAMES), np.nan)
    return last, since, nxt, until


def _streak(touch, segs, out_of_bounds=None):
    """How many consecutive touch runs (passes) the team of the last touch has made, with no opponent touch.

    Sign: positive for home, negative for away. It resets when the ball leaves the pitch or when more
    than STREAK_RESET_GAP frames pass without any touch (dead ball / stoppage).
    """
    out = np.zeros(len(touch), dtype=np.float32)
    oob = np.nan_to_num(out_of_bounds, nan=0.0) > 0.5 if out_of_bounds is not None else np.zeros(len(touch), dtype=bool)
    for a, b in segs:
        cur, count, prev, gap = 0.0, 0, 0.0, 0
        v = touch[a:b]
        ob = oob[a:b]
        for i in range(b - a):
            if ob[i]:
                cur, count, prev, gap = 0.0, 0, 0.0, 0
                out[a + i] = 0.0
                continue
            x = v[i]
            if x != 0:
                if gap > STREAK_RESET_GAP:
                    cur, count, prev = 0.0, 0, 0.0
                gap = 0
                if x != prev:
                    if x == cur:
                        count += 1
                    else:
                        cur, count = x, 1
                prev = x
            else:
                gap += 1
                if prev != 0:
                    prev = 0.0
                if gap > STREAK_RESET_GAP:
                    cur, count = 0.0, 0
            out[a + i] = cur * count
    return np.clip(out, -MAX_STREAK, MAX_STREAK)


def _interp_short_gaps(bx, by, bx_ff, by_ff, ball_out, segs):
    """Linear interpolation of the ball path (bx_lin, by_lin) in short gaps (<= BALL_INTERP_MAX_GAP)
    when both ends are inside the pitch; otherwise it falls back to the forward fill (bx_ff, by_ff).
    """
    bx_lin = bx_ff.copy()
    by_lin = by_ff.copy()
    vis = ~np.isnan(bx)
    in_b = vis & (np.nan_to_num(ball_out, nan=1.0) < 0.5)
    for a, b in segs:
        v = np.flatnonzero(vis[a:b])
        if len(v) < 2:
            continue
        for i0, i1 in zip(v[:-1], v[1:]):
            gap = i1 - i0 - 1
            if 0 < gap <= BALL_INTERP_MAX_GAP and in_b[a + i0] and in_b[a + i1]:
                w = np.linspace(0.0, 1.0, gap + 2, dtype=np.float32)[1:-1]
                bx_lin[a + i0 + 1:a + i1] = (1.0 - w) * bx[a + i0] + w * bx[a + i1]
                by_lin[a + i0 + 1:a + i1] = (1.0 - w) * by[a + i0] + w * by[a + i1]
    return bx_lin, by_lin


def _nanmin_rows(a):
    """Row minimum ignoring NaN; NaN if the whole row is missing."""
    if a.shape[1] == 0:
        return np.full(a.shape[0], np.nan, dtype=np.float32)
    filled = np.where(np.isnan(a), np.inf, a)
    m = filled.min(axis=1)
    return np.where(np.isinf(m), np.nan, m).astype(np.float32)


def _second_min_rows(a):
    if a.shape[1] < 2:
        return np.full(a.shape[0], np.nan, dtype=np.float32)
    filled = np.where(np.isnan(a), np.inf, a)
    part = np.partition(filled, 1, axis=1)[:, 1]
    return np.where(np.isinf(part), np.nan, part).astype(np.float32)


def attack_sign(match):
    """For each half, +1 if the home team attacks towards positive x, -1 otherwise.

    It looks at where the goalkeepers stand: the deepest home player (minimum x) and the most advanced one
    (maximum x), averaged over the half, compared with those of the other team.
    """
    px, team, period = match['px'], match['team'], match['period']
    signs = {}
    for half in (1, 2):
        rows = period == half
        if not rows.any():
            continue
        h = px[rows][:, team == 0]
        a = px[rows][:, team == 1]
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                h_mid = np.nanmedian(np.nanmin(h, axis=1)) + np.nanmedian(np.nanmax(h, axis=1))
                a_mid = np.nanmedian(np.nanmin(a, axis=1)) + np.nanmedian(np.nanmax(a, axis=1))
        signs[half] = 1.0 if h_mid < a_mid else -1.0
    return signs


# --------------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------------

def features(match):
    """DataFrame (frame × features) in float32. Home always attacks towards positive x."""
    import warnings
    warnings.simplefilter('ignore', category=RuntimeWarning)

    period = match['period']
    T = len(period)
    L, W = float(match['length']), float(match['width'])
    signs = attack_sign(match)
    s = np.zeros(T, dtype=np.float32)
    for half, sign in signs.items():
        s[period == half] = sign

    px = match['px'] * s[:, None]
    py = match['py'] * s[:, None]          # 180° rotation: y changes sign too, so left and right are preserved
    bx = match['bx'] * s
    by = match['by'] * s
    team = match['team']
    H, A = team == 0, team == 1
    segs = _segments(period)

    f = {}
    vis = ~np.isnan(bx)
    f['ball_seen'] = vis.astype(np.float32)
    f['bx'] = bx
    f['by_abs'] = np.abs(by)
    f['dist_goal_line'] = L / 2 - np.abs(bx)
    f['dist_touchline'] = W / 2 - np.abs(by)
    f['ball_out'] = np.where(vis, ((np.abs(bx) > L / 2 + FIELD_BOUND_TOLERANCE) | (np.abs(by) > W / 2 + FIELD_BOUND_TOLERANCE)).astype(np.float32), np.nan)
    bvx, bvy = _speed(bx, period), _speed(by, period)
    bspeed = np.hypot(bvx, bvy)
    f['ball_speed'] = bspeed
    f['ball_vx'] = bvx
    # acceleration: change of speed over ±2 frames (kicks, touches, bounces)
    f['ball_acc'] = np.hypot(_speed(bvx, period), _speed(bvy, period))
    since, until = _gap_position(vis, segs)
    f['since_ball_seen'] = since
    f['until_ball_seen'] = until
    f['ball_gap'] = since + until

    # players
    vx, vy = _speed(px, period), _speed(py, period)
    sp = np.hypot(vx, vy)
    seen = ~np.isnan(px)
    f['n_home'] = seen[:, H].sum(1).astype(np.float32)
    f['n_away'] = seen[:, A].sum(1).astype(np.float32)
    f['speed_mean'] = np.nanmean(sp, axis=1)
    f['speed_home'] = np.nanmean(sp[:, H], axis=1)
    f['speed_away'] = np.nanmean(sp[:, A], axis=1)
    f['speed_max'] = np.nanmax(np.where(np.isnan(sp), -1, sp), axis=1)
    f['speed_max'][f['speed_max'] < 0] = np.nan
    f['walking_share'] = np.nanmean(np.where(np.isnan(sp), np.nan, (sp < WALK).astype(np.float32)), axis=1)
    f['speed_p90'] = np.nanpercentile(sp, PERCENTILE_VELOCITY, axis=1).astype(np.float32)

    for name, m in (('home', H), ('away', A)):
        x, y = px[:, m], py[:, m]
        f['cx_' + name] = np.nanmean(x, axis=1)
        f['cy_' + name] = np.nanmean(y, axis=1)
        f['sx_' + name] = np.nanstd(x, axis=1)
        f['sy_' + name] = np.nanstd(y, axis=1)
        f['xmin_' + name] = np.nanmin(x, axis=1)
        f['xmax_' + name] = np.nanmax(x, axis=1)
        f['vx_' + name] = np.nanmean(vx[:, m], axis=1)
    f['cx_diff'] = f['cx_home'] - f['cx_away']
    f['vx_diff'] = f['vx_home'] - f['vx_away']
    f['cx_pack'] = 0.5 * (f['cx_home'] + f['cx_away'])
    f['vx_pack'] = 0.5 * (f['vx_home'] + f['vx_away'])
    f['sx_diff'] = f['sx_home'] - f['sx_away']
    f['sy_diff'] = f['sy_home'] - f['sy_away']
    v_mean_mag = np.hypot(np.nanmean(vx, axis=1), np.nanmean(vy, axis=1))
    f['speed_alignment'] = v_mean_mag / (f['speed_mean'] + 0.5)
    in_own_H = np.nanmean(np.where(np.isnan(px[:, H]), np.nan, (px[:, H] < 1.0).astype(np.float32)), axis=1)
    in_own_A = np.nanmean(np.where(np.isnan(px[:, A]), np.nan, (px[:, A] > -1.0).astype(np.float32)), axis=1)
    f['own_half_share'] = 0.5 * (in_own_H + in_own_A)
    # defensive line: the second-to-last player towards the own goal (the goalkeeper is the last)
    hx = np.where(np.isnan(px[:, H]), np.inf, px[:, H])
    ax = np.where(np.isnan(px[:, A]), -np.inf, px[:, A])
    hsorted = np.sort(hx, axis=1)
    asorted = -np.sort(-ax, axis=1)
    f['def_line_home'] = np.where(np.isfinite(hsorted[:, 1]), hsorted[:, 1], np.nan)
    f['def_line_away'] = np.where(np.isfinite(asorted[:, 1]), asorted[:, 1], np.nan)

    # ball and players
    d = np.hypot(px - bx[:, None], py - by[:, None])
    dH, dA = _nanmin_rows(d[:, H]), _nanmin_rows(d[:, A])
    f['d_home'] = dH
    f['d_away'] = dA
    f['d2_home'] = _second_min_rows(d[:, H])
    f['d2_away'] = _second_min_rows(d[:, A])
    f['d_diff'] = dH - dA
    f['near_home'] = np.where(vis, (d[:, H] < NEAR).sum(1), np.nan).astype(np.float32)
    f['near_away'] = np.where(vis, (d[:, A] < NEAR).sum(1), np.nan).astype(np.float32)
    closest_home = np.where(np.isnan(dH) | np.isnan(dA), np.nan, (dH < dA) + 0.5 * (dH == dA)).astype(np.float32)
    # when the ball is missing: distance from its last seen position and from the linear interpolation in short gaps
    bx_ff = pd.Series(bx).groupby(period).ffill().to_numpy(np.float32)
    by_ff = pd.Series(by).groupby(period).ffill().to_numpy(np.float32)
    dff = np.hypot(px - bx_ff[:, None], py - by_ff[:, None])
    f['dff_home'] = _nanmin_rows(dff[:, H])
    f['dff_away'] = _nanmin_rows(dff[:, A])
    f['dff_diff'] = f['dff_home'] - f['dff_away']
    f['bx_ff'] = bx_ff
    bx_lin, by_lin = _interp_short_gaps(bx, by, bx_ff, by_ff, f['ball_out'], segs)
    dlin = np.hypot(px - bx_lin[:, None], py - by_lin[:, None])
    f['dlin_home'] = _nanmin_rows(dlin[:, H])
    f['dlin_away'] = _nanmin_rows(dlin[:, A])
    f['dlin_diff'] = f['dlin_home'] - f['dlin_away']
    f['bx_lin'] = bx_lin
    # speed of the closest player of each team (a player carrying the ball moves slowly with the ball close)
    dH_full = np.where(np.isnan(d[:, H]), np.inf, d[:, H])
    dA_full = np.where(np.isnan(d[:, A]), np.inf, d[:, A])
    iH, iA = dH_full.argmin(1), dA_full.argmin(1)
    spH, spA = sp[:, H], sp[:, A]
    r = np.arange(T)
    f['speed_closest_home'] = np.where(np.isnan(dH), np.nan, spH[r, iH])
    f['speed_closest_away'] = np.where(np.isnan(dA), np.nan, spA[r, iA])
    # radial speed towards the (interpolated) ball of the closest player of each team
    dlinH_full = np.where(np.isnan(dlin[:, H]), np.inf, dlin[:, H])
    dlinA_full = np.where(np.isnan(dlin[:, A]), np.inf, dlin[:, A])
    iH_lin, iA_lin = dlinH_full.argmin(1), dlinA_full.argmin(1)
    pxH, pyH, vxH, vyH = px[:, H][r, iH_lin], py[:, H][r, iH_lin], vx[:, H][r, iH_lin], vy[:, H][r, iH_lin]
    pxA, pyA, vxA, vyA = px[:, A][r, iA_lin], py[:, A][r, iA_lin], vx[:, A][r, iA_lin], vy[:, A][r, iA_lin]
    uxH = (bx_lin - pxH) / (f['dlin_home'] + 0.1)
    uyH = (by_lin - pyH) / (f['dlin_home'] + 0.1)
    uxA = (bx_lin - pxA) / (f['dlin_away'] + 0.1)
    uyA = (by_lin - pyA) / (f['dlin_away'] + 0.1)
    f['v_approach_home'] = np.where(np.isnan(f['dlin_home']), np.nan, vxH * uxH + vyH * uyH)
    f['v_approach_away'] = np.where(np.isnan(f['dlin_away']), np.nan, vxA * uxA + vyA * uyA)
    f['v_approach_diff'] = f['v_approach_home'] - f['v_approach_away']

    # touches: ball seen within TOUCH metres of a player. The rules talk about control
    # ("keeps the ball for a few seconds", "more than two consecutive passes"): we need who touched last,
    # who touches next, and how many consecutive touches the same team makes
    tH = vis & (np.nan_to_num(dH, nan=99) < TOUCH)
    tA = vis & (np.nan_to_num(dA, nan=99) < TOUCH)
    dh, da = np.nan_to_num(dH, nan=99), np.nan_to_num(dA, nan=99)
    touch = np.where(tH & (~tA | (dh < da)), 1.0, np.where(tA & (~tH | (da < dh)), -1.0, 0.0)).astype(np.float32)
    last_t, since_t, next_t, until_t = _last_next(touch, segs)
    f['last_touch'] = last_t
    f['since_last_touch'] = since_t
    f['next_touch'] = next_t
    f['until_next_touch'] = until_t
    f['touch_changes'] = np.where(np.isnan(last_t) | np.isnan(next_t), np.nan, (last_t != next_t).astype(np.float32))
    starts = np.zeros(T, dtype=np.float32)            # start of a run of touches (a received pass)
    starts[1:] = (touch[1:] != 0) & (touch[1:] != touch[:-1])
    starts[0] = touch[0] != 0
    for w in TOUCH_COUNT_WINDOWS:
        f['touches_home_p{}'.format(w)] = _window_sum(starts * (touch > 0), segs, w, 0)
        f['touches_away_p{}'.format(w)] = _window_sum(starts * (touch < 0), segs, w, 0)
        f['touches_home_f{}'.format(w)] = _window_sum(starts * (touch > 0), segs, 0, w)
        f['touches_away_f{}'.format(w)] = _window_sum(starts * (touch < 0), segs, 0, w)
        f['hold_home_c{}'.format(w)] = _window_sum((touch > 0).astype(np.float32), segs, w, w)
        f['hold_away_c{}'.format(w)] = _window_sum((touch < 0).astype(np.float32), segs, w, w)
    f['last_streak'] = _streak(touch, segs, f['ball_out'])

    # time windows
    base = {
        'd_diff': f['d_diff'], 'closest_home': closest_home, 'ball_seen': f['ball_seen'],
        'speed_mean': f['speed_mean'], 'walking_share': f['walking_share'], 'ball_speed': f['ball_speed'],
        'ball_vx': f['ball_vx'], 'ball_acc': f['ball_acc'], 'bx': bx, 'vx_diff': f['vx_diff'],
        'cx_diff': f['cx_diff'], 'cx_pack': f['cx_pack'], 'vx_pack': f['vx_pack'],
        'dff_diff': f['dff_diff'], 'dlin_diff': f['dlin_diff'], 'v_approach_diff': f['v_approach_diff'],
        'speed_alignment': f['speed_alignment'], 'ball_out': f['ball_out'],
        'd_min': np.fmin(dH, dA),
    }
    for name, v in base.items():
        for w in WINDOWS:
            f['{}_c{}'.format(name, w)] = _window_mean(v, segs, w, w)
    for name in ('d_diff', 'dlin_diff', 'closest_home', 'ball_seen', 'speed_mean', 'ball_speed', 'bx', 'vx_diff', 'cx_pack', 'vx_pack'):
        for w in ASYMMETRIC_WINDOWS:
            f['{}_p{}'.format(name, w)] = _window_mean(base[name], segs, w, -1)
            f['{}_f{}'.format(name, w)] = _window_mean(base[name], segs, -1, w)
    for k in list(f):
        f[k] = np.asarray(f[k], dtype=np.float32)
    df = pd.DataFrame(f)
    df.loc[period == 0, :] = np.nan
    return df


# --------------------------------------------------------------------------------------------------
# Team swap: the same action seen from the other side
# --------------------------------------------------------------------------------------------------

def swap_teams(df):
    """The features as if home and away were swapped (pitch rotated by 180°).

    It doubles the training examples and makes the model symmetric: H becomes A and vice versa.
    """
    out = df.copy()
    cols = list(df.columns)
    neg = []
    for c in cols:
        root = re.sub(r'_[cpf]\d+$', '', c)
        if root in ('bx', 'ball_vx', 'bx_ff', 'bx_lin', 'd_diff', 'sx_diff', 'sy_diff',
                    'dff_diff', 'dlin_diff', 'v_approach_diff', 'cx_pack', 'vx_pack'):
            neg.append(c)
    neg += [c for c in cols if c in ('last_touch', 'next_touch', 'last_streak')]
    for c in neg:
        out[c] = -df[c]
    for c in cols:
        if c.startswith(('touches_home', 'hold_home')):
            o = c.replace('_home', '_away')
            out[c], out[o] = df[o].values, df[c].values
    # closest_home becomes 1 - closest_home
    for c in cols:
        if c.startswith('closest_home'):
            out[c] = 1.0 - df[c]
    pairs = [('n_home', 'n_away'), ('speed_home', 'speed_away'), ('d_home', 'd_away'), ('d2_home', 'd2_away'),
             ('near_home', 'near_away'), ('dff_home', 'dff_away'), ('dlin_home', 'dlin_away'),
             ('speed_closest_home', 'speed_closest_away'), ('v_approach_home', 'v_approach_away')]
    for a, b in pairs:
        if a in out.columns and b in out.columns:
            out[a], out[b] = df[b].values, df[a].values
    for stat in ('cx', 'cy', 'sx', 'sy', 'vx'):
        a, b = stat + '_home', stat + '_away'
        if a in out.columns and b in out.columns:
            sign = -1.0 if stat in ('cx', 'cy', 'vx') else 1.0
            out[a], out[b] = sign * df[b].values, sign * df[a].values
    # xmin/xmax: rotating the pitch turns the minimum into the maximum
    out['xmin_home'], out['xmax_home'] = -df['xmax_away'].values, -df['xmin_away'].values
    out['xmin_away'], out['xmax_away'] = -df['xmax_home'].values, -df['xmin_home'].values
    out['def_line_home'], out['def_line_away'] = -df['def_line_away'].values, -df['def_line_home'].values
    return out
