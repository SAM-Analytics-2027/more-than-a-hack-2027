"""Prove del modello sulle 30 partite esterne, sempre su partite lasciate fuori dall'allenamento.

    python modelli_mv/esperimento.py allena  --nome v2                    allena su tutte le fonti (5 gruppi)
    python modelli_mv/esperimento.py allena  --nome v2_dm --fonti dfl metrica      solo DFL e Metrica
    python modelli_mv/esperimento.py allena  --nome v2_ind --indicatore            tutte, con la colonna fonte_tv
    python modelli_mv/esperimento.py eventi  --nome ev_v2                 il modello degli eventi
    python modelli_mv/esperimento.py decodifica --nome v2 [--eventi ev_v2]   regola il decodificatore e dà i punteggi
    python modelli_mv/esperimento.py annidato   --nome v2 [--eventi ev_v2]   lo stesso, regolando senza il gruppo giudicato

Le probabilità "fuori gruppo" di ogni partita si salvano in data/cache_mv/oof_<nome>.npz.
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import caratteristiche as C  # noqa: E402
import dati  # noqa: E402
import modello as M  # noqa: E402
from constants import (  # noqa: E402
    N_FOLDS, STEP, SEED, META,
    SHORT_SPELL_WEIGHT_MIN, SHORT_SPELL_WEIGHT_MAX,
    EV_HALF, EVENT_SUBSAMPLE_POS, EVENT_SUBSAMPLE_NEG,
    EVENT_WEIGHT_POS, EVENT_WEIGHT_NEG,
    HGB_PARAMS, DEFAULT, GRID, LETTERS,
)


def folds(items):
    """Gruppi di partite intere, con le fonti distribuite in modo uniforme."""
    out = {}
    for src in dati.SOURCES:
        ids = [it for it in items if it[0] == src]
        rng = np.random.default_rng(SEED)
        rng.shuffle(ids)
        for i, it in enumerate(ids):
            out[it] = i % N_FOLDS
    return out


def feature_cols(df):
    return [c for c in df.columns if c not in META]


def short_spell_weights(y, period):
    """Peso di ogni frame: i possessi brevi pesano di più, quelli lunghi di meno (idea di Stefano).

    w = sqrt(durata mediana / durata del possesso), tra 1/3 e 3; i frame di palla ferma pesano 1.
    """
    w = np.ones(len(y), dtype=np.float32)
    spells = []
    for half in (1, 2):
        idx = np.flatnonzero(period == half)
        if len(idx) == 0:
            continue
        a0, b0 = idx[0], idx[-1] + 1
        sub = y[a0:b0]
        cut = np.flatnonzero(sub[1:] != sub[:-1]) + 1
        for a, b in zip(np.concatenate(([0], cut)), np.concatenate((cut, [len(sub)]))):
            if sub[a] in (0, 1):
                spells.append((a0 + a, a0 + b))
    if spells:
        med = np.median([b - a for a, b in spells])
        for a, b in spells:
            w[a:b] = np.clip(np.sqrt(med / (b - a)), SHORT_SPELL_WEIGHT_MIN, SHORT_SPELL_WEIGHT_MAX)
    return w


WEIGHTS = False        # True: più peso ai possessi brevi
NO_RICOCHETS = False   # True: allena sulle etichette DFL senza rimpalli (etichette.py)
TV_WEIGHT = 1.0        # peso dei frame SkillCorner in allenamento (1/4 = valgono come 5 partite)


def training_rows(df, cols, swap=True):
    keep = (df['y'].values < 3) & (df['period'].values > 0)
    w_all = short_spell_weights(df['y'].values, df['period'].values) if WEIGHTS else np.ones(len(df), np.float32)
    sub = df.loc[keep].iloc[::STEP]
    w = w_all[keep][::STEP]
    X = sub[cols]
    y = sub['y'].values.astype(np.int8)
    if not swap:
        return X, y, w
    Xs = C.swap_teams(X)[cols]
    ys = y.copy()
    ys[y == 0], ys[y == 1] = 1, 0
    return pd.concat([X, Xs], ignore_index=True), np.concatenate([y, ys]), np.concatenate([w, w])


def make_model():
    return HistGradientBoostingClassifier(**HGB_PARAMS)


def predict_sym(model, df, cols):
    """Media della previsione diretta e di quella a squadre scambiate (rimesse a posto)."""
    X = df[cols]
    p1 = model.predict_proba(X)
    p2 = model.predict_proba(C.swap_teams(X)[cols])[:, [1, 0, 2]]
    return (p1 + p2) / 2


def add_source_flag(data):
    """Colonna 'fonte_tv': 1 per le partite SkillCorner (tracking dalle riprese tv), 0 per le altre e per il Barça."""
    for it, df in data.items():
        df['fonte_tv'] = np.float32(it[0] == 'skillcorner')
    return data


def train(name, train_sources, flag=False):
    data = dati.load_all()
    if flag:
        add_source_flag(data)
    if NO_RICOCHETS:
        import etichette
        for it in data:
            if it[0] == 'dfl':      # solo per allenare: il giudizio usa sempre le etichette originali (meta_of)
                data[it] = data[it].assign(y=etichette.load(it[1]))
    items = sorted(data)
    fold_of = folds(items)
    cols = feature_cols(next(iter(data.values())))
    oof = {}
    for k in range(N_FOLDS):
        t = time.time()
        tr = [it for it in items if fold_of[it] != k and it[0] in train_sources]
        te = [it for it in items if fold_of[it] == k]
        parts = [training_rows(data[it], cols) for it in tr]
        X = pd.concat([p[0] for p in parts], ignore_index=True)
        y = np.concatenate([p[1] for p in parts])
        w = np.concatenate([p[2] * (TV_WEIGHT if it[0] == 'skillcorner' else 1.0) for p, it in zip(parts, tr)])
        model = make_model().fit(X, y, sample_weight=w)
        for it in te:
            oof[it] = predict_sym(model, data[it], cols).astype(np.float32)
        print('gruppo {}: {} partite in allenamento ({} righe), {} iterazioni, {:.0f} s'.format(
            k, len(tr), len(y), model.n_iter_, time.time() - t), flush=True)
    path = os.path.join(dati.CACHE, 'oof_{}.npz'.format(name))
    np.savez_compressed(path, **{'{}__{}'.format(*it): p for it, p in oof.items()})
    print('salvate', path)


def event_labels(y):
    """0 niente, 1 cambio squadra, 2 inizio pausa, 3 ripresa, intorno a ogni cambio dell'etichetta vera."""
    codes = M.to_codes(y)
    ev = np.zeros(len(y), dtype=np.int8)
    runs = M.pm.valid_runs(codes)
    for f, a, b in M.pm.transitions(codes, runs):
        if {a, b} == {M.pm.H, M.pm.A}:
            k = 1
        elif b == M.pm.D:
            k = 2
        else:
            k = 3
        ev[max(f - EV_HALF, 0):f + EV_HALF + 1] = k
    return ev


def train_events(name):
    """Il modello degli eventi: per ogni frame, la probabilità che lì cambi la squadra, inizi una pausa o riprenda il gioco."""
    data = dati.load_all()
    items = sorted(data)
    fold_of = folds(items)
    cols = feature_cols(next(iter(data.values())))
    oof = {}
    for k in range(N_FOLDS):
        t = time.time()
        Xs, ys, ws = [], [], []
        for it in [i for i in items if fold_of[i] != k]:
            df = data[it]
            y = df['y'].values
            ev = event_labels(y)
            ok = (y < 3) & (df['period'].values > 0)
            idx_ev = np.flatnonzero(ok & (ev > 0))[::EVENT_SUBSAMPLE_POS]
            idx_no = np.flatnonzero(ok & (ev == 0))[::EVENT_SUBSAMPLE_NEG]
            idx = np.concatenate([idx_ev, idx_no])
            w = np.concatenate([np.full(len(idx_ev), EVENT_WEIGHT_POS), np.full(len(idx_no), EVENT_WEIGHT_NEG)])
            X = df[cols].iloc[idx]
            Xs += [X, C.swap_teams(X)[cols]]
            ys += [ev[idx], ev[idx]]
            ws += [w, w]
        model = make_model().fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys),
                                 sample_weight=np.concatenate(ws))
        for it in [i for i in items if fold_of[i] == k]:
            X = data[it][cols]
            p = (model.predict_proba(X) + model.predict_proba(C.swap_teams(X)[cols])) / 2
            oof[it] = p.astype(np.float32)
        print('gruppo {}: {} iterazioni, {:.0f} s'.format(k, model.n_iter_, time.time() - t), flush=True)
    path = os.path.join(dati.CACHE, 'oof_{}.npz'.format(name))
    np.savez_compressed(path, **{'{}__{}'.format(*it): p for it, p in oof.items()})
    print('salvate', path)


def load_oof(name):
    z = np.load(os.path.join(dati.CACHE, 'oof_{}.npz'.format(name)))
    return {tuple(k.split('__')): z[k] for k in z.files}


def evaluate(oof, data_meta, params, items):
    out = {}
    for it in items:
        y, per, nd, ev = data_meta[it]
        pred = M.decode(oof[it], per, nd, params, ev)
        out[it] = M.score(y, pred)
    return out


def summary(scores):
    by = {}
    for (src, _), v in scores.items():
        by.setdefault(src, []).append(v)
    per_src = {s: float(np.mean(v)) for s, v in by.items()}
    return float(np.mean(list(per_src.values()))), per_src


def tune(oof, data_meta, items, start=None, rounds=2):
    """Ricerca per coordinate: un parametro alla volta, il valore migliore, poi il giro successivo.

    Obiettivo: media delle tre fonti (ogni fonte pesa uguale, non ogni partita).
    """
    params = dict(start or DEFAULT)
    best, _ = summary(evaluate(oof, data_meta, params, items))
    grid = dict(GRID)
    if any(data_meta[it][3] is not None for it in items):
        grid['w_ev'] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        params.setdefault('w_ev', 0.0)
    for r in range(rounds):
        for key, values in grid.items():
            for v in values:
                trial = dict(params, **{key: v})
                s, _ = summary(evaluate(oof, data_meta, trial, items))
                if s > best + 1e-6:
                    best, params = s, trial
        print('  giro {}: {:.4f} {}'.format(r + 1, best, params), flush=True)
    return params, best


def meta_of(items, events=None):
    out = {}
    for it in items:
        df = dati.load(*it)
        nd = (df['n_casa'].values + df['n_ospiti'].values) == 0
        out[it] = (df['y'].values, df['period'].values, nd, None if events is None else events[it])
    return out


def decode_report(name, ev_name=None):
    oof = load_oof(name)
    items = sorted(oof)
    meta = meta_of(items, load_oof(ev_name) if ev_name else None)
    raw = {}
    for it in items:
        y, per, nd, _ = meta[it]
        raw[it] = M.score(y, np.where(per > 0, oof[it].argmax(1), 3).astype(np.int8))
    print('senza decodificatore (argmax frame per frame):', summary(raw))
    print('decodificatore di partenza:', summary(evaluate(oof, meta, DEFAULT, items)))
    params, best = tune(oof, meta, items)
    scores = evaluate(oof, meta, params, items)
    tot, per_src = summary(scores)
    print('regolato:', round(tot, 4), {k: round(v, 4) for k, v in per_src.items()})
    for it in items:
        print('   {:12s} {:10s} {:.3f}'.format(it[0], it[1], scores[it]))
    tag = name + ('+' + ev_name if ev_name else '')
    with open(os.path.join(dati.CACHE, 'risultati_{}.json'.format(tag)), 'w') as f:
        json.dump({'params': params, 'media_fonti': tot, 'per_fonte': per_src,
                   'per_partita': {'{}__{}'.format(*k): v for k, v in scores.items()}}, f, indent=1)
    return params


def nested_report(name, ev_name=None):
    """Punteggio onesto: i parametri del decodificatore si regolano senza il gruppo su cui si giudica."""
    oof = load_oof(name)
    items = sorted(oof)
    fold_of = folds(items)
    meta = meta_of(items, load_oof(ev_name) if ev_name else None)
    scores = {}
    for k in range(N_FOLDS):
        tr = [it for it in items if fold_of[it] != k]
        te = [it for it in items if fold_of[it] == k]
        params, _ = tune(oof, meta, tr, rounds=1)
        scores.update(evaluate(oof, meta, params, te))
    tot, per_src = summary(scores)
    print('annidato:', round(tot, 4), {k: round(v, 4) for k, v in per_src.items()})
    return scores


EXT_RESULTS = os.path.join(C.fc.DATA_DIR, 'risultati_esterne')


def _write_labels(path, codes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({'frame': np.arange(len(codes)), 'label': LETTERS[codes]}).to_csv(path, index=False)


def export_external(name, params_file, out_name):
    """Risposte "fuori allenamento" sulle 10 partite DFL e Metrica, per la pagina della dashboard sulle partite esterne.

    Una cartella per modello: data/risultati_esterne/<out_name>/<partita>.csv, nello stesso formato del file del Barça.
    """
    oof = load_oof(name)
    params = json.load(open(os.path.join(dati.CACHE, params_file)))['params']
    items = [it for it in sorted(oof) if it[0] in ('dfl', 'metrica')]
    meta = meta_of(items)
    for it in items:
        _, per, nd, _ = meta[it]
        _write_labels(os.path.join(EXT_RESULTS, out_name, it[1] + '.csv'), M.decode(oof[it], per, nd, params))
    print('scritte', len(items), 'partite in', os.path.join(EXT_RESULTS, out_name))


def export_baseline(window=50):
    """La baseline degli organizzatori (starter/baseline.py) sulle partite esterne sporcate: squadra del giocatore
    più vicino alla palla, poi voto di maggioranza su ±window frame; D dove nessuno ha la palla."""
    for it in [i for i in dati.external_matches() if i[0] in ('dfl', 'metrica')]:
        df = dati.load(*it)
        dh, da = df['d_casa'].values, df['d_ospiti'].values
        code = np.where(np.isnan(dh) & np.isnan(da), -1, np.where(np.nan_to_num(dh, nan=1e9) <= np.nan_to_num(da, nan=1e9), 0, 1))
        hc = np.concatenate(([0], np.cumsum(code == 0)))
        ac = np.concatenate(([0], np.cumsum(code == 1)))
        i = np.arange(len(code))
        lo, hi = np.maximum(0, i - window), np.minimum(len(code), i + window + 1)
        h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
        out = np.where(h + a > 0, np.where(h >= a, 0, 1), 2).astype(np.int8)
        out[df['period'].values == 0] = 3
        _write_labels(os.path.join(EXT_RESULTS, 'baseline_smooth{}'.format(window), it[1] + '.csv'), out)
    print('baseline scritta')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cosa', choices=['allena', 'eventi', 'decodifica', 'annidato', 'esporta', 'baseline'])
    ap.add_argument('--parametri', help='per "esporta": file json dei parametri, in data/cache_mv/')
    ap.add_argument('--come', help='per "esporta": nome della cartella in data/risultati_esterne/')
    ap.add_argument('--eventi', default=None, help='probabilità degli eventi da usare nel decodificatore')
    ap.add_argument('--nome', default='tutte')
    ap.add_argument('--fonti', nargs='+', default=list(dati.SOURCES))
    ap.add_argument('--indicatore', action='store_true', help='aggiunge la colonna fonte_tv (SkillCorner sì/no)')
    ap.add_argument('--pesi-brevi', action='store_true', help='più peso ai frame dei possessi brevi')
    ap.add_argument('--senza-rimbalzi', action='store_true', help='allena sulle etichette DFL senza rimpalli')
    ap.add_argument('--peso-tv', type=float, default=1.0, help='peso dei frame SkillCorner (es. 0.25)')
    a = ap.parse_args()
    WEIGHTS = a.pesi_brevi
    NO_RICOCHETS = a.senza_rimbalzi
    TV_WEIGHT = a.peso_tv
    if a.cosa == 'allena':
        train(a.nome, set(a.fonti), a.indicatore)
    elif a.cosa == 'annidato':
        nested_report(a.nome, a.eventi)
    elif a.cosa == 'eventi':
        train_events(a.nome)
    elif a.cosa == 'esporta':
        export_external(a.nome, a.parametri, a.come)
    elif a.cosa == 'baseline':
        export_baseline()
    else:
        decode_report(a.nome, a.eventi)
