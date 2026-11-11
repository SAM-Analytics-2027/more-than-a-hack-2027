"""Caratteristiche per frame, uguali per le partite esterne e per Barça–Athletic.

Una partita (il dizionario di modelli_mv/formato_comune.py) diventa una tabella con una riga per frame
e circa 100 colonne. Tutto si calcola solo dal tracking, che è anonimo: posizioni dei giocatori con la
squadra di ogni slot, e la palla quando si vede.

Le coordinate si girano in modo che la squadra di casa attacchi sempre verso x positive: la direzione
si ricava dai dati di ogni tempo (dove stanno i portieri), così vale per tutte le fonti.

Le finestre nel tempo (passato, futuro e centrate) saltano i valori mancanti e non attraversano mai
l'intervallo né i tratti fuori dai tempi.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formato_comune as fc  # noqa: E402  (modelli_mv/formato_comune.py)

FPS = 25
SPEED_HALF = 2           # velocità su ±2 frame (0,16 s)
WALK = 2.0               # m/s: sotto questa velocità un giocatore "cammina"
NEAR = 3.0               # m: giocatori "vicini" alla palla
TOUCH = 1.8              # m: palla "toccata" (la palla sporcata è spostata di circa 0,8 m)
WINDOWS = (12, 37, 112)  # mezze finestre: ±0,5 s, ±1,5 s, ±4,5 s


# --------------------------------------------------------------------------------------------------
# Strumenti
# --------------------------------------------------------------------------------------------------

def _speed(a, period):
    """Velocità (m/s) da differenze centrate su ±SPEED_HALF frame, NaN se manca un estremo o cambia tempo."""
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
    """[(inizio, fine)] dei tratti con lo stesso tempo di gioco > 0."""
    p = period.astype(np.int16)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], p, [0]))) != 0)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if p[a] > 0:
            out.append((a, b))
    return out


def _window_mean(f, segs, before, after):
    """Media dei valori non mancanti in [t-before, t+after], dentro ogni tempo. NaN se non c'è niente."""
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
    """Per ogni frame: frame trascorsi dall'ultima palla vista e frame fino alla prossima (dentro il tempo)."""
    since = np.full(visible.shape, np.nan, dtype=np.float32)
    until = np.full(visible.shape, np.nan, dtype=np.float32)
    for a, b in segs:
        v = visible[a:b]
        n = b - a
        idx = np.arange(n)
        last = np.maximum.accumulate(np.where(v, idx, -1))
        nxt = np.minimum.accumulate(np.where(v, idx, n)[::-1])[::-1]
        s = np.where(last >= 0, idx - last, idx + 1 + 250).astype(np.float32)   # senza palla dall'inizio
        u = np.where(nxt < n, nxt - idx, n - idx + 250).astype(np.float32)
        since[a:b] = np.minimum(s, 1500)
        until[a:b] = np.minimum(u, 1500)
    return since, until


def _window_sum(f, segs, before, after):
    """Somma in [t-before, t+after], dentro ogni tempo."""
    out = np.zeros(f.shape, dtype=np.float32)
    for a, b in segs:
        cs = np.concatenate(([0.0], np.cumsum(np.nan_to_num(f[a:b].astype(np.float64)))))
        n = b - a
        t = np.arange(n)
        out[a:b] = cs[np.clip(t + after + 1, 0, n)] - cs[np.clip(t - before, 0, n)]
    return out


def _last_next(touch, segs):
    """Squadra dell'ultimo tocco (+1 casa, -1 ospiti) e frame passati; squadra del prossimo e frame che mancano."""
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
        since[a:b] = np.where(okl, np.minimum(idx - li, 1500), np.nan)
        nxt[a:b] = np.where(okn, v[np.minimum(ni, n - 1)], np.nan)
        until[a:b] = np.where(okn, np.minimum(ni - idx, 1500), np.nan)
    return last, since, nxt, until


def _streak(touch, segs):
    """Quante serie di tocchi consecutive (passaggi) ha fatto la squadra dell'ultimo tocco, senza tocchi avversari.

    Segno: positivo se è la casa, negativo se sono gli ospiti.
    """
    out = np.zeros(len(touch), dtype=np.float32)
    for a, b in segs:
        cur, count, prev = 0.0, 0, 0.0
        v = touch[a:b]
        for i in range(b - a):
            x = v[i]
            if x != 0 and x != prev:
                if x == cur:
                    count += 1
                else:
                    cur, count = x, 1
            if x != 0:
                prev = x
            elif prev != 0:
                prev = 0.0
            out[a + i] = cur * count
    return np.clip(out, -20, 20)


def _nanmin_rows(a):
    """Minimo per riga ignorando i NaN; NaN se la riga è tutta mancante."""
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
    """Per ogni tempo, +1 se la squadra di casa attacca verso x positive, -1 altrimenti.

    Si guarda dove stanno i portieri: il giocatore di casa più indietro (x minima) e quello più avanti
    (x massima), mediati sul tempo, confrontati con quelli dell'altra squadra.
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
# Caratteristiche
# --------------------------------------------------------------------------------------------------

def features(match):
    """DataFrame (frame × caratteristiche) in float32. Casa attacca sempre verso x positive."""
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
    py = match['py'] * s[:, None]          # giro di 180°: anche y cambia segno, così destra e sinistra restano
    bx = match['bx'] * s
    by = match['by'] * s
    team = match['team']
    H, A = team == 0, team == 1
    segs = _segments(period)

    f = {}
    vis = ~np.isnan(bx)
    f['palla_vista'] = vis.astype(np.float32)
    f['bx'] = bx
    f['by_abs'] = np.abs(by)
    f['dist_linea_fondo'] = L / 2 - np.abs(bx)
    f['dist_linea_lato'] = W / 2 - np.abs(by)
    f['palla_fuori'] = np.where(vis, ((np.abs(bx) > L / 2 + 0.5) | (np.abs(by) > W / 2 + 0.5)).astype(np.float32), np.nan)
    bvx, bvy = _speed(bx, period), _speed(by, period)
    bspeed = np.hypot(bvx, bvy)
    f['palla_vel'] = bspeed
    f['palla_vx'] = bvx
    # accelerazione: cambio di velocità su ±2 frame (calci, tocchi, rimbalzi)
    f['palla_acc'] = np.hypot(_speed(bvx, period), _speed(bvy, period))
    since, until = _gap_position(vis, segs)
    f['da_ultima_palla'] = since
    f['a_prossima_palla'] = until
    f['buco_palla'] = since + until

    # giocatori
    vx, vy = _speed(px, period), _speed(py, period)
    sp = np.hypot(vx, vy)
    seen = ~np.isnan(px)
    f['n_casa'] = seen[:, H].sum(1).astype(np.float32)
    f['n_ospiti'] = seen[:, A].sum(1).astype(np.float32)
    f['vel_media'] = np.nanmean(sp, axis=1)
    f['vel_casa'] = np.nanmean(sp[:, H], axis=1)
    f['vel_ospiti'] = np.nanmean(sp[:, A], axis=1)
    f['vel_max'] = np.nanmax(np.where(np.isnan(sp), -1, sp), axis=1)
    f['vel_max'][f['vel_max'] < 0] = np.nan
    f['quota_cammina'] = np.nanmean(np.where(np.isnan(sp), np.nan, (sp < WALK).astype(np.float32)), axis=1)
    f['vel_p90'] = np.nanpercentile(sp, 90, axis=1).astype(np.float32)

    for name, m in (('casa', H), ('ospiti', A)):
        x, y = px[:, m], py[:, m]
        f['cx_' + name] = np.nanmean(x, axis=1)
        f['cy_' + name] = np.nanmean(y, axis=1)
        f['sx_' + name] = np.nanstd(x, axis=1)
        f['sy_' + name] = np.nanstd(y, axis=1)
        f['xmin_' + name] = np.nanmin(x, axis=1)
        f['xmax_' + name] = np.nanmax(x, axis=1)
        f['vx_' + name] = np.nanmean(vx[:, m], axis=1)
    f['cx_diff'] = f['cx_casa'] - f['cx_ospiti']
    f['vx_diff'] = f['vx_casa'] - f['vx_ospiti']
    f['sx_diff'] = f['sx_casa'] - f['sx_ospiti']
    f['sy_diff'] = f['sy_casa'] - f['sy_ospiti']
    # linea difensiva: il penultimo giocatore verso la propria porta (il portiere è l'ultimo)
    hx = np.where(np.isnan(px[:, H]), np.inf, px[:, H])
    ax = np.where(np.isnan(px[:, A]), -np.inf, px[:, A])
    hsorted = np.sort(hx, axis=1)
    asorted = -np.sort(-ax, axis=1)
    f['linea_casa'] = np.where(np.isfinite(hsorted[:, 1]), hsorted[:, 1], np.nan)
    f['linea_ospiti'] = np.where(np.isfinite(asorted[:, 1]), asorted[:, 1], np.nan)

    # palla e giocatori
    d = np.hypot(px - bx[:, None], py - by[:, None])
    dH, dA = _nanmin_rows(d[:, H]), _nanmin_rows(d[:, A])
    f['d_casa'] = dH
    f['d_ospiti'] = dA
    f['d2_casa'] = _second_min_rows(d[:, H])
    f['d2_ospiti'] = _second_min_rows(d[:, A])
    f['d_diff'] = dH - dA
    f['vicini_casa'] = np.where(vis, (d[:, H] < NEAR).sum(1), np.nan).astype(np.float32)
    f['vicini_ospiti'] = np.where(vis, (d[:, A] < NEAR).sum(1), np.nan).astype(np.float32)
    closest_home = np.where(np.isnan(dH) | np.isnan(dA), np.nan, (dH < dA) + 0.5 * (dH == dA)).astype(np.float32)
    # quando la palla manca: distanza dall'ultima posizione vista (la palla non si teletrasporta)
    bx_ff = pd.Series(bx).groupby(period).ffill().to_numpy(np.float32)
    by_ff = pd.Series(by).groupby(period).ffill().to_numpy(np.float32)
    dff = np.hypot(px - bx_ff[:, None], py - by_ff[:, None])
    f['dff_casa'] = _nanmin_rows(dff[:, H])
    f['dff_ospiti'] = _nanmin_rows(dff[:, A])
    f['dff_diff'] = f['dff_casa'] - f['dff_ospiti']
    f['bx_ff'] = bx_ff
    # velocità del giocatore più vicino di ogni squadra (chi conduce va piano con la palla vicina)
    dH_full = np.where(np.isnan(d[:, H]), np.inf, d[:, H])
    dA_full = np.where(np.isnan(d[:, A]), np.inf, d[:, A])
    iH, iA = dH_full.argmin(1), dA_full.argmin(1)
    spH, spA = sp[:, H], sp[:, A]
    r = np.arange(T)
    f['vel_vicino_casa'] = np.where(np.isnan(dH), np.nan, spH[r, iH])
    f['vel_vicino_ospiti'] = np.where(np.isnan(dA), np.nan, spA[r, iA])

    # tocchi: palla vista entro TOUCH metri da un giocatore. Le regole parlano di controllo
    # ("tiene la palla qualche secondo", "più di due passaggi di fila"): servono chi ha toccato per ultimo,
    # chi tocca dopo, e quanti tocchi di fila fa la stessa squadra
    tH = vis & (np.nan_to_num(dH, nan=99) < TOUCH)
    tA = vis & (np.nan_to_num(dA, nan=99) < TOUCH)
    dh, da = np.nan_to_num(dH, nan=99), np.nan_to_num(dA, nan=99)
    touch = np.where(tH & (~tA | (dh < da)), 1.0, np.where(tA & (~tH | (da < dh)), -1.0, 0.0)).astype(np.float32)
    last_t, since_t, next_t, until_t = _last_next(touch, segs)
    f['ultimo_tocco'] = last_t
    f['da_ultimo_tocco'] = since_t
    f['prossimo_tocco'] = next_t
    f['a_prossimo_tocco'] = until_t
    f['tocco_cambia'] = np.where(np.isnan(last_t) | np.isnan(next_t), np.nan, (last_t != next_t).astype(np.float32))
    starts = np.zeros(T, dtype=np.float32)            # inizio di una serie di tocchi (un passaggio ricevuto)
    starts[1:] = (touch[1:] != 0) & (touch[1:] != touch[:-1])
    starts[0] = touch[0] != 0
    for w in (50, 125):
        f['tocchi_casa_p{}'.format(w)] = _window_sum(starts * (touch > 0), segs, w, 0)
        f['tocchi_ospiti_p{}'.format(w)] = _window_sum(starts * (touch < 0), segs, w, 0)
        f['tocchi_casa_f{}'.format(w)] = _window_sum(starts * (touch > 0), segs, 0, w)
        f['tocchi_ospiti_f{}'.format(w)] = _window_sum(starts * (touch < 0), segs, 0, w)
        f['tenuta_casa_c{}'.format(w)] = _window_sum((touch > 0).astype(np.float32), segs, w, w)
        f['tenuta_ospiti_c{}'.format(w)] = _window_sum((touch < 0).astype(np.float32), segs, w, w)
    f['serie_ultimo'] = _streak(touch, segs)

    # finestre nel tempo
    base = {
        'd_diff': f['d_diff'], 'vicino_casa': closest_home, 'palla_vista': f['palla_vista'],
        'vel_media': f['vel_media'], 'quota_cammina': f['quota_cammina'], 'palla_vel': f['palla_vel'],
        'palla_vx': f['palla_vx'], 'palla_acc': f['palla_acc'], 'bx': bx, 'vx_diff': f['vx_diff'],
        'cx_diff': f['cx_diff'], 'dff_diff': f['dff_diff'], 'palla_fuori': f['palla_fuori'],
        'd_min': np.fmin(dH, dA),
    }
    for name, v in base.items():
        for w in WINDOWS:
            f['{}_c{}'.format(name, w)] = _window_mean(v, segs, w, w)
    for name in ('d_diff', 'vicino_casa', 'palla_vista', 'vel_media', 'palla_vel', 'bx', 'vx_diff'):
        for w in (25, 75):
            f['{}_p{}'.format(name, w)] = _window_mean(base[name], segs, w, -1)
            f['{}_f{}'.format(name, w)] = _window_mean(base[name], segs, -1, w)
    for k in list(f):
        f[k] = np.asarray(f[k], dtype=np.float32)
    df = pd.DataFrame(f)
    df.loc[period == 0, :] = np.nan
    return df


# --------------------------------------------------------------------------------------------------
# Scambio delle squadre: la stessa azione vista dall'altra parte
# --------------------------------------------------------------------------------------------------

def swap_teams(df):
    """Le caratteristiche come se casa e ospiti fossero scambiati (campo girato di 180°).

    Serve a raddoppiare gli esempi e a rendere il modello simmetrico: H diventa A e viceversa.
    """
    out = df.copy()
    cols = list(df.columns)
    neg = []
    for c in cols:
        root = re.sub(r'_[cpf]\d+$', '', c)
        if root in ('bx', 'palla_vx', 'bx_ff', 'd_diff', 'sx_diff', 'sy_diff', 'dff_diff'):  # cx_diff e vx_diff non cambiano
            neg.append(c)
    neg += [c for c in cols if c in ('ultimo_tocco', 'prossimo_tocco', 'serie_ultimo')]
    for c in neg:
        out[c] = -df[c]
    for c in cols:
        if c.startswith(('tocchi_casa', 'tenuta_casa')):
            o = c.replace('_casa', '_ospiti')
            out[c], out[o] = df[o].values, df[c].values
    # vicino_casa diventa 1 - vicino_casa
    for c in cols:
        if c.startswith('vicino_casa'):
            out[c] = 1.0 - df[c]
    pairs = [('n_casa', 'n_ospiti'), ('vel_casa', 'vel_ospiti'), ('d_casa', 'd_ospiti'), ('d2_casa', 'd2_ospiti'),
             ('vicini_casa', 'vicini_ospiti'), ('dff_casa', 'dff_ospiti'), ('vel_vicino_casa', 'vel_vicino_ospiti')]
    for a, b in pairs:
        out[a], out[b] = df[b].values, df[a].values
    for stat in ('cx', 'cy', 'sx', 'sy', 'vx'):
        a, b = stat + '_casa', stat + '_ospiti'
        sign = -1.0 if stat in ('cx', 'cy', 'vx') else 1.0
        out[a], out[b] = sign * df[b].values, sign * df[a].values
    # xmin/xmax: girando il campo il minimo diventa il massimo
    out['xmin_casa'], out['xmax_casa'] = -df['xmax_ospiti'].values, -df['xmin_ospiti'].values
    out['xmin_ospiti'], out['xmax_ospiti'] = -df['xmax_casa'].values, -df['xmin_casa'].values
    out['linea_casa'], out['linea_ospiti'] = -df['linea_ospiti'].values, -df['linea_casa'].values
    return out
