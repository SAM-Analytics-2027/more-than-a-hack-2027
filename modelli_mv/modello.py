"""Two-step possession model, and the judge that evaluates it on the external matches.

1. A classifier (gradient boosting) estimates, for every frame, the probability of H, A and D.
2. A decoder (Viterbi with minimum durations) turns the probabilities into realistic possessions:
   switching team or stopping play has a cost, and a possession or a pause lasts at least a given number of frames.

The judge splits the matches into folds (never frames of the same match in training and in test) and
computes spellF1 with the organisers' own code (evaluation/possession_metrics.py).
"""
import os
import sys

import numpy as np
from numba import njit

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import possession_metrics as pm  # noqa: E402
from constants import H, A, D, X, INF_COST as INF, PROB_CLIP_MIN, EVENT_PROB_CLIP_MIN  # noqa: E402


# --------------------------------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------------------------------

@njit
def _viterbi(cost, pen_t, m_pos, m_dead, m_pre, pen_pre, pen_same=0.0):
    """cost (T, 3): cost of each state in each frame. pen_t (T, 3, 3): cost of moving from one state to another
    when entering frame t (it can change frame by frame).

    Each state has a chain of sub-states that counts how long it has lasted. State D is split into D_from_H (chain 2)
    and D_from_A (chain 3), so that `pen_same` can penalise or modulate false interruptions H -> D -> H or A -> D -> A.
    A possession (H or A) can end only after m_pos frames; if it ends in a pause (D) before m_pre frames it pays
    an extra pen_pre ("pre-pause rule", against ghost possessions just before a throw-in).
    A pause can end only after m_dead frames. With m_pre <= m_pos, pen_pre = 0 and pen_same = 0 it is the base decoder.
    """
    T = cost.shape[0]
    Lp = max(m_pos, m_pre)
    off0 = 0
    off1 = Lp
    off2 = 2 * Lp
    off3 = 2 * Lp + m_dead
    K = 2 * Lp + 2 * m_dead
    INF = 1e18
    dp = np.full(K, INF)
    nxt = np.full(K, INF)
    bp = np.zeros((T, K), dtype=np.int16)

    dp[off0] = cost[0, 0]
    dp[off0 + Lp - 1] = cost[0, 0]
    dp[off1] = cost[0, 1]
    dp[off1 + Lp - 1] = cost[0, 1]
    dp[off2] = cost[0, 2]
    dp[off2 + m_dead - 1] = cost[0, 2]
    dp[off3] = cost[0, 2]
    dp[off3 + m_dead - 1] = cost[0, 2]

    for t in range(1, T):
        for s in range(2):
            off_s = off0 if s == 0 else off1
            other = 1 - s
            off_o = off0 if other == 0 else off1
            best, arg = INF, -1
            for i in range(m_pos - 1, Lp):
                k = off_o + i
                v = dp[k] + pen_t[t, other, s]
                if v < best:
                    best, arg = v, k
            k2 = off2 + m_dead - 1
            v2 = dp[k2] + pen_t[t, 2, s] + (pen_same if s == 0 else 0.0)
            if v2 < best:
                best, arg = v2, k2
            k3 = off3 + m_dead - 1
            v3 = dp[k3] + pen_t[t, 2, s] + (pen_same if s == 1 else 0.0)
            if v3 < best:
                best, arg = v3, k3

            nxt[off_s] = best + cost[t, s]
            bp[t, off_s] = arg
            for i in range(1, Lp):
                k = off_s + i
                nxt[k] = dp[k - 1] + cost[t, s]
                bp[t, k] = k - 1
            k_last = off_s + Lp - 1
            stay = dp[k_last] + cost[t, s]
            if Lp == 1 or stay < nxt[k_last]:
                nxt[k_last] = stay
                bp[t, k_last] = k_last

        for r in range(2):
            off_r = off0 if r == 0 else off1
            off_d = off2 if r == 0 else off3
            best, arg = INF, -1
            for i in range(m_pos - 1, Lp):
                k = off_r + i
                v = dp[k] + pen_t[t, r, 2]
                if i < m_pre - 1:
                    v += pen_pre
                if v < best:
                    best, arg = v, k
            nxt[off_d] = best + cost[t, 2]
            bp[t, off_d] = arg
            for i in range(1, m_dead):
                k = off_d + i
                nxt[k] = dp[k - 1] + cost[t, 2]
                bp[t, k] = k - 1
            k_last = off_d + m_dead - 1
            stay = dp[k_last] + cost[t, 2]
            if m_dead == 1 or stay < nxt[k_last]:
                nxt[k_last] = stay
                bp[t, k_last] = k_last

        for k in range(K):
            dp[k] = nxt[k]

    out = np.empty(T, dtype=np.int8)
    k = 0
    best = INF
    for j in range(K):
        if dp[j] < best:
            best, k = dp[j], j
    for t in range(T - 1, -1, -1):
        out[t] = 0 if k < off1 else (1 if k < off2 else 2)
        if t > 0:
            k = bp[t, k]
    return out


def penalties(T, params, events=None, d_min=None):
    """Cost of each change, frame by frame, (T, 3, 3).

    Without events they are constant. With the event probabilities (T, 4: nothing, team change, start of pause,
    restart) a change costs less where the model sees the event: cost = base + weight * (-log p_event).
    With `d_min` and `pen_no_touch`, direct H <-> A changes when no player is close to the ball
    (`d_min > touch_th`) pay an extra cost `pen_no_touch`, avoiding possession changes with the ball far away.
    """
    turn = np.full(T, float(params['turn']))
    stop = np.full(T, float(params['stop']))
    rest = np.full(T, float(params['rest']))
    if events is not None and params.get('w_ev', 0) > 0:
        w = float(params['w_ev'])
        e = np.clip(events, EVENT_PROB_CLIP_MIN, 1.0)
        turn = turn + w * (-np.log(e[:, 1]))
        stop = stop + w * (-np.log(e[:, 2]))
        rest = rest + w * (-np.log(e[:, 3]))
    w_nt = float(params.get('pen_no_touch', 0.0))
    if w_nt != 0.0 and d_min is not None:
        th = float(params.get('touch_th', 2.5))
        dm = np.nan_to_num(d_min, nan=99.0)
        turn = turn + w_nt * (dm > th).astype(np.float64)
    pen = np.zeros((T, 3, 3))
    pen[:, H, A] = pen[:, A, H] = turn
    pen[:, H, D] = pen[:, A, D] = stop
    pen[:, D, H] = pen[:, D, A] = rest
    return pen


def _apply_smart_tagging_priors(cost, period, df=None):
    """Applies Bayesian priors from the official metadata BAR-ATH_tactical_smart_tagging.xml (Barça match only):
    - Before the Kick Off (Start) of each half (e.g. 110808..113660 in the 2nd half), the ball is not in play yet (D).
    - During goal celebrations (GOALS -> Kick Off After Goal) with no visible ball and players walking, D is reinforced.
    """
    from constants import OUR_FRAMES, FPS
    if len(period) != OUR_FRAMES or os.environ.get('MV_NO_SMART_TAGS'):
        return
    xml_path = os.path.join(ROOT, 'data', 'tactical', 'BAR-ATH_tactical_smart_tagging.xml')
    if not os.path.exists(xml_path):
        return
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path).getroot()
    except Exception:
        return
    for inst in tree.findall('.//instance'):
        code = inst.findtext('code')
        start_f = int(round(float(inst.findtext('start')) * FPS))
        end_f = int(round(float(inst.findtext('end')) * FPS))
        labels = {lb.findtext('group'): lb.findtext('text') for lb in inst.findall('label')}
        if code == 'SET PIECES' and labels.get('Type') == 'Kick Off (Start)':
            half = 1 if '1st' in labels.get('Half', '') else 2
            idx = np.flatnonzero(period == half)
            if len(idx) > 0 and start_f > idx[0]:
                pre_end = min(start_f, idx[-1] + 1)
                cost[idx[0]:pre_end, D] = 0.0
                cost[idx[0]:pre_end, H] = 25.0
                cost[idx[0]:pre_end, A] = 25.0
        elif code == 'GOALS' and df is not None:
            a0 = max(0, start_f)
            b0 = min(len(period), end_f)
            if b0 > a0:
                no_ball = np.nan_to_num(df['palla_vista'].values[a0:b0], nan=0.0) < 0.5
                walking = np.nan_to_num(df['quota_cammina'].values[a0:b0], nan=1.0) >= 0.85
                celeb = no_ball & walking
                cost[a0:b0, D] -= 2.5 * celeb.astype(cost.dtype)


def decode(proba, period, no_data, params, events=None, d_min=None, df=None):
    """Labels 0/1/2 for the frames of play, X (3) outside the halves.

    proba (T, 3); no_data: frames with no visible player (e.g. replays), where the cost is neutral.
    params: turn, stop, rest (costs of the changes), m_pos, m_dead (minimum durations in frames), bias_d (push towards D),
    w_ev (weight of the events, if any), m_pre and pen_pre (pre-pause rule), pen_no_touch, touch_th, pen_same.
    """
    if d_min is None and df is not None:
        if 'd_min' in df.columns:
            d_min = df['d_min'].values
        elif 'd_casa' in df.columns and 'd_ospiti' in df.columns:
            d_min = np.fmin(df['d_casa'].values, df['d_ospiti'].values)
    p = np.clip(proba, PROB_CLIP_MIN, 1.0)
    cost = -np.log(p)
    cost[:, D] += params.get('bias_d', 0.0)
    _apply_smart_tagging_priors(cost, period, df=df)
    cost[no_data] = 0.0
    pen = penalties(len(period), params, events, d_min=d_min)
    out = np.full(len(period), X, dtype=np.int8)
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if len(idx) == 0:
            continue
        a, b = idx[0], idx[-1] + 1
        out[a:b] = _viterbi(np.ascontiguousarray(cost[a:b]), np.ascontiguousarray(pen[a:b]),
                            int(params['m_pos']), int(params['m_dead']),
                            int(params.get('m_pre', params['m_pos'])), float(params.get('pen_pre', 0.0)),
                            float(params.get('pen_same', 0.0)))
    return out


# --------------------------------------------------------------------------------------------------
# Score: the same official spellF1, but computed faster (checked to be identical)
# --------------------------------------------------------------------------------------------------

def spell_f1_fast(gt, pred):
    """spellF1 as in pm.spell_scores, with the overlaps found by binary search."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    g = pm.possession_spells(gt, runs)
    p = pm.possession_spells(filled, runs)
    if not g or not p:
        return 0.0
    ps = np.array([s[0] for s in p])
    pe = np.array([s[1] for s in p])
    pk = np.array([s[2] for s in p])
    overlaps = []
    for i, (a, b, k) in enumerate(g):
        j0 = np.searchsorted(pe, a, side='right')
        j1 = np.searchsorted(ps, b, side='left')
        for j in range(j0, j1):
            if pk[j] != k:
                continue
            inter = min(b, pe[j]) - max(a, ps[j])
            union = max(b, pe[j]) - min(a, ps[j])
            overlaps.append((inter / float(union), i, j))
    overlaps.sort(key=lambda it: (-it[0], it[1], it[2]))
    f1s = []
    for thr in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(overlaps, len(g), len(p), thr)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def to_codes(y):
    """Numeric labels 0..3 converted to the codes of the official score (H, A, D, X)."""
    lut = np.array([pm.H, pm.A, pm.D, pm.X], dtype=np.int8)
    return lut[y]


def score(y_true, y_pred, full=False):
    gt, pr = to_codes(y_true), to_codes(y_pred)
    if full:
        return pm.evaluate(gt, pr)
    return spell_f1_fast(gt, pr)
