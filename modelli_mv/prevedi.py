"""Allena il modello finale su tutte le partite esterne e scrive le risposte per Barça–Athletic.

    python modelli_mv/prevedi.py --nome dm --parametri risultati_dm_v2_dm.json --fonti dfl metrica
    python modelli_mv/prevedi.py --nome ind --parametri risultati_dm_v2_ind.json --indicatore
    python modelli_mv/prevedi.py --nome media --parametri risultati_dm_media_dm_ind.json --media dm ind

I parametri del decodificatore sono quelli regolati sulle partite esterne (un file json in data/cache_mv/).
Il file va in evaluation/results/mv_<nome>.csv (ignorato da GitHub) e si controlla con
python evaluation/validate_submission.py evaluation/results/mv_<nome>.csv
La risposta della gara non entra mai: né nell'allenamento, né nella scelta dei parametri.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import caratteristiche as C  # noqa: E402
import dati  # noqa: E402
import esperimento as E  # noqa: E402
import modello as M  # noqa: E402
from constants import LETTERS  # noqa: E402


def fit_all(cols, data, events=False):
    Xs, ys, ws = [], [], []
    for it, df in data.items():
        if events:
            y = df['y'].values
            ev = E.event_labels(y)
            ok = (y < 3) & (df['period'].values > 0)
            idx_ev = np.flatnonzero(ok & (ev > 0))[::2]
            idx_no = np.flatnonzero(ok & (ev == 0))[::16]
            idx = np.concatenate([idx_ev, idx_no])
            w = np.concatenate([np.full(len(idx_ev), 2.0), np.full(len(idx_no), 16.0)])
            X = df[cols].iloc[idx]
            Xs += [X, C.swap_teams(X)[cols]]
            ys += [ev[idx], ev[idx]]
            ws += [w, w]
        else:
            X, y, w = E.training_rows(df, cols)
            Xs.append(X)
            ys.append(y)
            ws.append(w * (E.TV_WEIGHT if it[0] == 'skillcorner' else 1.0))
    model = E.make_model().fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys),
                               sample_weight=np.concatenate(ws))
    v = model.validation_score_
    print('{} iterazioni; perdita di validazione a 100/250/ultima: {:.4f} {:.4f} {:.4f}'.format(
        model.n_iter_, v[min(100, len(v) - 1)], v[min(250, len(v) - 1)], v[-1]), flush=True)
    return model


def main(name, params_file, sources, flag=False, ev_name=None):
    data = {it: df for it, df in dati.load_all().items() if it[0] in sources}
    target = dati.load('barca', 'BAR-ATH')
    if flag:
        E.add_source_flag(data)
        target['fonte_tv'] = np.float32(0)      # il Barça ha una telecamera tattica, non le riprese tv
    cols = E.feature_cols(next(iter(data.values())))
    tag = name
    params = json.load(open(os.path.join(dati.CACHE, params_file)))['params']
    print('partite', len(data), 'parametri', params, flush=True)
    model = fit_all(cols, data)
    proba = E.predict_sym(model, target, cols)
    events = None
    if ev_name:
        ev_model = fit_all(cols, data, events=True)
        X = target[cols]
        events = (ev_model.predict_proba(X) + ev_model.predict_proba(C.swap_teams(X)[cols])) / 2
    period = target['period'].values
    no_data = (target['n_casa'].values + target['n_ospiti'].values) == 0
    pred = M.decode(proba, period, no_data, params, events)
    out = os.path.join(C.ROOT, 'evaluation', 'results', 'mv_{}.csv'.format(tag.replace('+', '_')))
    pd.DataFrame({'frame': np.arange(len(pred)), 'label': LETTERS[pred]}).to_csv(out, index=False)
    np.save(os.path.join(dati.CACHE, 'barca_proba_{}.npy'.format(name)), proba)
    print('scritto', out, {k: int((LETTERS[pred] == k).sum()) for k in 'HADX'})


def combine(names, params_file, out_name):
    """Media delle probabilità di più modelli finali già allenati, poi il decodificatore."""
    target = dati.load('barca', 'BAR-ATH')
    proba = np.mean([np.load(os.path.join(dati.CACHE, 'barca_proba_{}.npy'.format(n))) for n in names], axis=0)
    params = json.load(open(os.path.join(dati.CACHE, params_file)))['params']
    no_data = (target['n_casa'].values + target['n_ospiti'].values) == 0
    pred = M.decode(proba, target['period'].values, no_data, params)
    out = os.path.join(C.ROOT, 'evaluation', 'results', 'mv_{}.csv'.format(out_name))
    pd.DataFrame({'frame': np.arange(len(pred)), 'label': LETTERS[pred]}).to_csv(out, index=False)
    print('scritto', out, params, {k: int((LETTERS[pred] == k).sum()) for k in 'HADX'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nome', required=True, help='nome del file di risposte')
    ap.add_argument('--parametri', required=True, help='file json dei parametri, in data/cache_mv/')
    ap.add_argument('--fonti', nargs='+', default=list(dati.SOURCES))
    ap.add_argument('--indicatore', action='store_true')
    ap.add_argument('--eventi', action='store_true')
    ap.add_argument('--media', nargs='+', help='non allena: fa la media dei modelli finali con questi nomi')
    ap.add_argument('--peso-tv', type=float, default=1.0, help='peso dei frame SkillCorner (es. 0.25)')
    ap.add_argument('--pesi-brevi', action='store_true', help='più peso ai frame dei possessi brevi')
    a = ap.parse_args()
    E.TV_WEIGHT = a.peso_tv
    E.WEIGHTS = a.pesi_brevi
    if a.media:
        combine(a.media, a.parametri, a.nome)
    else:
        main(a.nome, a.parametri, set(a.fonti), a.indicatore, 'ev' if a.eventi else None)
