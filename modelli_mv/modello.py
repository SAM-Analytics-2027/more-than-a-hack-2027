"""Modello a due passi per il possesso, e il giudice che lo valuta sulle partite esterne.

1. Un classificatore (gradient boosting) stima per ogni frame la probabilità di H, A e D.
2. Un decodificatore (Viterbi con durate minime) trasforma le probabilità in possessi realistici:
   cambiare squadra o fermare il gioco costa, e un possesso o una pausa durano almeno un tot di frame.

Il giudice divide le partite in gruppi (mai frame della stessa partita in allenamento e in prova) e
calcola lo spellF1 con lo stesso codice degli organizzatori (evaluation/possession_metrics.py).
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
# Decodificatore
# --------------------------------------------------------------------------------------------------

@njit
def _viterbi(cost, pen_t, m_pos, m_dead, m_pre, pen_pre):
    """cost (T, 3): costo di ogni stato in ogni frame. pen_t (T, 3, 3): costo di passare da uno stato all'altro
    entrando nel frame t (può cambiare frame per frame).

    Ogni stato ha una catena di sottostati che conta quanto dura. Un possesso (H o A) può finire solo dopo
    m_pos frame; se finisce in una pausa (D) prima di m_pre frame paga pen_pre in più: è la "regola prima
    delle pause" (idea di Stefano), contro i possessi fantasma subito prima di un fallo laterale.
    Una pausa può finire solo dopo m_dead frame. Con m_pre <= m_pos e pen_pre = 0 è il decodificatore semplice.
    """
    T = cost.shape[0]
    Lp = max(m_pos, m_pre)
    m = np.array([Lp, Lp, m_dead])
    off = np.array([0, Lp, 2 * Lp])
    K = 2 * Lp + m_dead
    INF = 1e18
    dp = np.full(K, INF)
    nxt = np.full(K, INF)
    bp = np.zeros((T, K), dtype=np.int16)
    for s in range(3):
        dp[off[s]] = cost[0, s]
        dp[off[s] + m[s] - 1] = cost[0, s]          # si può anche partire "a metà" di un possesso
    for t in range(1, T):
        for s in range(3):
            # ingresso in s da un altro stato
            best, arg = INF, -1
            for r in range(3):
                if r == s:
                    continue
                if r == 2:
                    k = off[2] + m[2] - 1
                    v = dp[k] + pen_t[t, 2, s]
                    if v < best:
                        best, arg = v, k
                else:
                    for i in range(m_pos - 1, Lp):
                        k = off[r] + i
                        v = dp[k] + pen_t[t, r, s]
                        if s == 2 and i < m_pre - 1:
                            v += pen_pre
                        if v < best:
                            best, arg = v, k
            nxt[off[s]] = best + cost[t, s]
            bp[t, off[s]] = arg
            # avanzamento lungo la catena
            for i in range(1, m[s]):
                k = off[s] + i
                nxt[k] = dp[k - 1] + cost[t, s]
                bp[t, k] = k - 1
            # l'ultimo sottostato può anche restare
            k = off[s] + m[s] - 1
            stay = dp[k] + cost[t, s]
            if m[s] == 1 or stay < nxt[k]:
                nxt[k] = stay
                bp[t, k] = k
        for k in range(K):
            dp[k] = nxt[k]
    out = np.empty(T, dtype=np.int8)
    k = 0
    best = INF
    for j in range(K):
        if dp[j] < best:
            best, k = dp[j], j
    for t in range(T - 1, -1, -1):
        s = 0 if k < off[1] else (1 if k < off[2] else 2)
        out[t] = s
        if t > 0:
            k = bp[t, k]
    return out


def penalties(T, params, events=None):
    """Costi dei cambi frame per frame, (T, 3, 3).

    Senza eventi sono costanti. Con le probabilità degli eventi (T, 4: niente, cambio squadra, inizio pausa,
    ripresa) un cambio costa meno dove il modello vede l'evento: costo = base + peso * (-log p_evento).
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
    pen = np.zeros((T, 3, 3))
    pen[:, H, A] = pen[:, A, H] = turn
    pen[:, H, D] = pen[:, A, D] = stop
    pen[:, D, H] = pen[:, D, A] = rest
    return pen


def decode(proba, period, no_data, params, events=None):
    """Etichette 0/1/2 per i frame di gioco, X (3) fuori dai tempi.

    proba (T, 3); no_data: frame senza nessun giocatore visibile (es. replay), lì il costo è neutro.
    params: turn, stop, rest (costi dei cambi), m_pos, m_dead (durate minime in frame), bias_d (spinta verso D),
    w_ev (peso degli eventi, se ci sono), m_pre e pen_pre (regola prima delle pause).
    """
    p = np.clip(proba, PROB_CLIP_MIN, 1.0)
    cost = -np.log(p)
    cost[:, D] += params.get('bias_d', 0.0)
    cost[no_data] = 0.0
    pen = penalties(len(period), params, events)
    out = np.full(len(period), X, dtype=np.int8)
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if len(idx) == 0:
            continue
        a, b = idx[0], idx[-1] + 1
        out[a:b] = _viterbi(np.ascontiguousarray(cost[a:b]), np.ascontiguousarray(pen[a:b]),
                            int(params['m_pos']), int(params['m_dead']),
                            int(params.get('m_pre', params['m_pos'])), float(params.get('pen_pre', 0.0)))
    return out


# --------------------------------------------------------------------------------------------------
# Punteggio: lo stesso spellF1 ufficiale, ma calcolato più in fretta (verificato uguale)
# --------------------------------------------------------------------------------------------------

def spell_f1_fast(gt, pred):
    """spellF1 come pm.spell_scores, con le sovrapposizioni trovate per ricerca binaria."""
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
    """Etichette numeriche 0..3 nei codici del punteggio ufficiale (H, A, D, X)."""
    lut = np.array([pm.H, pm.A, pm.D, pm.X], dtype=np.int8)
    return lut[y]


def score(y_true, y_pred, full=False):
    gt, pr = to_codes(y_true), to_codes(y_pred)
    if full:
        return pm.evaluate(gt, pr)
    return spell_f1_fast(gt, pr)
