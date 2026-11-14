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
def _viterbi(cost, pen_t, m_pos, m_dead, m_pre, pen_pre, pen_same=0.0):
    """cost (T, 3): costo di ogni stato in ogni frame. pen_t (T, 3, 3): costo di passare da uno stato all'altro
    entrando nel frame t (può cambiare frame per frame).

    Ogni stato ha una catena di sottostati che conta quanto dura. Lo stato D è sdoppiato in D_from_H (catena 2)
    e D_from_A (catena 3) per penalizzare o modulare via `pen_same` le false interruzioni H -> D -> H o A -> D -> A.
    Un possesso (H o A) può finire solo dopo m_pos frame; se finisce in una pausa (D) prima di m_pre frame paga
    pen_pre in più ("regola prima delle pause", contro i possessi fantasma subito prima di un fallo laterale).
    Una pausa può finire solo dopo m_dead frame. Con m_pre <= m_pos, pen_pre = 0 e pen_same = 0 coincide col base.
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
    """Costi dei cambi frame per frame, (T, 3, 3).

    Senza eventi sono costanti. Con le probabilità degli eventi (T, 4: niente, cambio squadra, inizio pausa,
    ripresa) un cambio costa meno dove il modello vede l'evento: costo = base + peso * (-log p_evento).
    Con `d_min` e `pen_no_touch`, i cambi diretti H <-> A quando nessun giocatore è vicino alla palla
    (`d_min > touch_th`) pagano un costo aggiuntivo `pen_no_touch`, evitando cambi di possesso a palla lontana.
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
    """Applica i prior bayesiani dai metadati ufficiali BAR-ATH_tactical_smart_tagging.xml (solo per il Barça):
    - Prima del Kick Off (Start) di ciascun tempo (es. 110808..113660 nel 2° tempo), la palla non è ancora in gioco (D).
    - Nelle esultanze post-gol (GOALS -> Kick Off After Goal) senza palla visibile e con giocatori al passo, rafforza D.
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
    """Etichette 0/1/2 per i frame di gioco, X (3) fuori dai tempi.

    proba (T, 3); no_data: frame senza nessun giocatore visibile (es. replay), lì il costo è neutro.
    params: turn, stop, rest (costi dei cambi), m_pos, m_dead (durate minime in frame), bias_d (spinta verso D),
    w_ev (peso degli eventi, se ci sono), m_pre e pen_pre (regola prima delle pause), pen_no_touch, touch_th, pen_same.
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
