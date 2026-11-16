"""Trains the final model on all the external matches and writes the answers for Barça–Athletic.

    python modelli_mv/prevedi.py --name dm --params results_dm_v2_dm.json --sources dfl metrica
    python modelli_mv/prevedi.py --name ind --params results_dm_v2_ind.json --source-flag
    python modelli_mv/prevedi.py --name mean --params results_dm_mean_dm_ind.json --mean dm ind

The decoder parameters are the ones tuned on the external matches (a json file in data/cache_mv/).
The file goes to evaluation/results/mv_<name>.csv (ignored by GitHub) and is checked with
python evaluation/validate_submission.py evaluation/results/mv_<name>.csv
The competition answer is never used: neither in training nor in the choice of the parameters.
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
    print('{} iterations; validation loss at 100/250/last: {:.4f} {:.4f} {:.4f}'.format(
        model.n_iter_, v[min(100, len(v) - 1)], v[min(250, len(v) - 1)], v[-1]), flush=True)
    return model


def main(name, params_file, sources, flag=False, ev_name=None):
    data = {it: df for it, df in dati.load_all().items() if it[0] in sources}
    target = dati.load('barca', 'BAR-ATH')
    if flag:
        E.add_source_flag(data)
        target['fonte_tv'] = np.float32(0)      # Barça has a tactical camera, not TV footage
    cols = E.feature_cols(next(iter(data.values())))
    tag = name
    params = json.load(open(os.path.join(dati.CACHE, params_file)))['params']
    print('matches', len(data), 'parameters', params, flush=True)
    model = fit_all(cols, data)
    proba = E.predict_sym(model, target, cols)
    events = None
    if ev_name:
        ev_model = fit_all(cols, data, events=True)
        X = target[cols]
        events = (ev_model.predict_proba(X) + ev_model.predict_proba(C.swap_teams(X)[cols])) / 2
    period = target['period'].values
    no_data = (target['n_casa'].values + target['n_ospiti'].values) == 0
    pred = M.decode(proba, period, no_data, params, events, df=target)
    out = os.path.join(C.ROOT, 'evaluation', 'results', 'mv_{}.csv'.format(tag.replace('+', '_')))
    pd.DataFrame({'frame': np.arange(len(pred)), 'label': LETTERS[pred]}).to_csv(out, index=False)
    np.save(os.path.join(dati.CACHE, 'barca_proba_{}.npy'.format(name)), proba)
    print('wrote', out, {k: int((LETTERS[pred] == k).sum()) for k in 'HADX'})


def combine(names, params_file, out_name):
    """Mean of the probabilities of several final models already trained, then the decoder."""
    target = dati.load('barca', 'BAR-ATH')
    proba = np.mean([np.load(os.path.join(dati.CACHE, 'barca_proba_{}.npy'.format(n))) for n in names], axis=0)
    params = json.load(open(os.path.join(dati.CACHE, params_file)))['params']
    no_data = (target['n_casa'].values + target['n_ospiti'].values) == 0
    pred = M.decode(proba, target['period'].values, no_data, params, df=target)
    out = os.path.join(C.ROOT, 'evaluation', 'results', 'mv_{}.csv'.format(out_name))
    pd.DataFrame({'frame': np.arange(len(pred)), 'label': LETTERS[pred]}).to_csv(out, index=False)
    print('wrote', out, params, {k: int((LETTERS[pred] == k).sum()) for k in 'HADX'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='name of the answers file')
    ap.add_argument('--params', required=True, help='json file with the parameters, in data/cache_mv/')
    ap.add_argument('--sources', nargs='+', default=list(dati.SOURCES))
    ap.add_argument('--source-flag', action='store_true')
    ap.add_argument('--events', action='store_true')
    ap.add_argument('--mean', nargs='+', help='does not train: averages the final models with these names')
    ap.add_argument('--tv-weight', type=float, default=1.0, help='weight of the SkillCorner frames (e.g. 0.25)')
    ap.add_argument('--short-weights', action='store_true', help='more weight to the frames of short possessions')
    a = ap.parse_args()
    E.TV_WEIGHT = a.tv_weight
    E.WEIGHTS = a.short_weights
    if a.mean:
        combine(a.mean, a.params, a.name)
    else:
        main(a.name, a.params, set(a.sources), a.source_flag, 'ev' if a.events else None)
