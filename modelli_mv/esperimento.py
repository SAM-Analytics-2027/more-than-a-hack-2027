"""Model experiments on the 30 external matches, always on matches left out of training.

    python modelli_mv/esperimento.py train   --name v2                     trains on all sources (5 folds)
    python modelli_mv/esperimento.py train   --name v2_dm --sources dfl metrica      DFL and Metrica only
    python modelli_mv/esperimento.py train   --name v2_ind --source-flag             all, with the fonte_tv column
    python modelli_mv/esperimento.py events  --name ev_v2                  the event model
    python modelli_mv/esperimento.py decode  --name v2 [--events ev_v2]    tunes the decoder and gives the scores
    python modelli_mv/esperimento.py nested  --name v2 [--events ev_v2]    the same, tuning without the judged fold

The out-of-fold probabilities of each match are saved in data/cache_mv/oof_<name>.npz.
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
    """Folds of whole matches, with the sources spread evenly (always on the canonical IDs of the 30 matches)."""
    canon = sorted(dati.external_matches())
    out = {}
    for src in dati.SOURCES:
        ids = [it for it in canon if it[0] == src]
        rng = np.random.default_rng(SEED)
        rng.shuffle(ids)
        for i, it in enumerate(ids):
            out[it] = i % N_FOLDS
    return {it: out[it] for it in items if it in out}


def feature_cols(df):
    return [c for c in df.columns if c not in META]


def short_spell_weights(y, period):
    """Weight of each frame: short possessions weigh more, long ones less.

    w = sqrt(median duration / possession duration), between 1/3 and 3; dead-ball frames weigh 1.
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


WEIGHTS = False        # True: more weight to short possessions
NO_RICOCHETS = False   # True: trains on the DFL labels without ricochets (etichette.py)
TV_WEIGHT = 1.0        # weight of the SkillCorner frames in training (1/4 = they count as 5 matches)


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
    """Mean of the direct prediction and of the one with the teams swapped (swapped back)."""
    X = df[cols]
    p1 = model.predict_proba(X)
    p2 = model.predict_proba(C.swap_teams(X)[cols])[:, [1, 0, 2]]
    return (p1 + p2) / 2


def add_source_flag(data):
    """Column 'fonte_tv': 1 for the SkillCorner matches (tracking from TV footage), 0 for the others and for Barça-Athletic."""
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
            if it[0] == 'dfl':      # training only: evaluation always uses the original labels (meta_of)
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
        print('fold {}: {} training matches ({} rows), {} iterations, {:.0f} s'.format(
            k, len(tr), len(y), model.n_iter_, time.time() - t), flush=True)
    path = os.path.join(dati.CACHE, 'oof_{}.npz'.format(name))
    np.savez_compressed(path, **{'{}__{}'.format(*it): p for it, p in oof.items()})
    print('saved', path)


def event_labels(y):
    """0 nothing, 1 team change, 2 start of pause, 3 restart, around every change of the true label."""
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
    """The event model: for each frame, the probability that the team changes, a pause starts or play restarts there."""
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
        print('fold {}: {} iterations, {:.0f} s'.format(k, model.n_iter_, time.time() - t), flush=True)
    path = os.path.join(dati.CACHE, 'oof_{}.npz'.format(name))
    np.savez_compressed(path, **{'{}__{}'.format(*it): p for it, p in oof.items()})
    print('saved', path)


def load_oof(name):
    z = np.load(os.path.join(dati.CACHE, 'oof_{}.npz'.format(name)))
    return {tuple(k.split('__')): z[k] for k in z.files}


def evaluate(oof, data_meta, params, items):
    out = {}
    for it in items:
        m = data_meta[it]
        y, per, nd, ev = m[:4]
        d_min = m[4] if len(m) > 4 else None
        pred = M.decode(oof[it], per, nd, params, ev, d_min=d_min)
        out[it] = M.score(y, pred)
    return out


def summary(scores):
    by = {}
    for (src, _), v in scores.items():
        by.setdefault(src, []).append(v)
    per_src = {s: float(np.mean(v)) for s, v in by.items()}
    return float(np.mean(list(per_src.values()))), per_src


def tune(oof, data_meta, items, start=None, rounds=2):
    """Coordinate search: one parameter at a time, the best value, then the next round.

    Objective: mean over the matches (if DFL+Metrica only) or over the sources.
    """
    params = dict(start or DEFAULT)
    best = float(np.mean(list(evaluate(oof, data_meta, params, items).values())))
    grid = dict(GRID)
    if any(data_meta[it][3] is not None for it in items):
        grid['w_ev'] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        params.setdefault('w_ev', 0.0)
    for r in range(rounds):
        for key, values in grid.items():
            for v in values:
                trial = dict(params, **{key: v})
                s = float(np.mean(list(evaluate(oof, data_meta, trial, items).values())))
                if s > best + 1e-6:
                    best, params = s, trial
        print('  round {}: {:.4f} {}'.format(r + 1, best, params), flush=True)
    return params, best


def meta_of(items, events=None):
    out = {}
    for it in items:
        df = dati.load(*it)
        nd = (df['n_casa'].values + df['n_ospiti'].values) == 0
        dm = np.fmin(df['d_casa'].values, df['d_ospiti'].values) if 'd_casa' in df.columns else None
        out[it] = (df['y'].values, df['period'].values, nd, None if events is None else events[it], dm)
    return out


def decode_report(name, ev_name=None):
    oof = load_oof(name)
    items = sorted(oof)
    meta = meta_of(items, load_oof(ev_name) if ev_name else None)
    raw = {}
    for it in items:
        y, per, nd, _ = meta[it]
        raw[it] = M.score(y, np.where(per > 0, oof[it].argmax(1), 3).astype(np.int8))
    print('without decoder (frame-by-frame argmax):', summary(raw))
    print('starting decoder:', summary(evaluate(oof, meta, DEFAULT, items)))
    params, best = tune(oof, meta, items)
    scores = evaluate(oof, meta, params, items)
    tot, per_src = summary(scores)
    print('tuned:', round(tot, 4), {k: round(v, 4) for k, v in per_src.items()})
    for it in items:
        print('   {:12s} {:10s} {:.3f}'.format(it[0], it[1], scores[it]))
    tag = name + ('+' + ev_name if ev_name else '')
    with open(os.path.join(dati.CACHE, 'results_{}.json'.format(tag)), 'w') as f:
        json.dump({'params': params, 'mean_sources': tot, 'per_source': per_src,
                   'per_match': {'{}__{}'.format(*k): v for k, v in scores.items()}}, f, indent=1)
    return params


def nested_report(name, ev_name=None):
    """Honest score: the decoder parameters are tuned without the fold being judged."""
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
    print('nested:', round(tot, 4), {k: round(v, 4) for k, v in per_src.items()})
    return scores


EXT_RESULTS = os.path.join(C.fc.DATA_DIR, 'risultati_esterne')


def _write_labels(path, codes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({'frame': np.arange(len(codes)), 'label': LETTERS[codes]}).to_csv(path, index=False)


def export_external(name, params_file, out_name):
    """Out-of-training answers on the 10 DFL and Metrica matches, for the dashboard page on the external matches.

    One folder per model: data/risultati_esterne/<out_name>/<match>.csv, in the same format as the Barça-Athletic file.
    """
    oof = load_oof(name)
    params = json.load(open(os.path.join(dati.CACHE, params_file)))['params']
    items = [it for it in sorted(oof) if it[0] in ('dfl', 'metrica')]
    meta = meta_of(items)
    for it in items:
        _, per, nd, _ = meta[it]
        _write_labels(os.path.join(EXT_RESULTS, out_name, it[1] + '.csv'), M.decode(oof[it], per, nd, params))
    print('wrote', len(items), 'matches in', os.path.join(EXT_RESULTS, out_name))


def export_baseline(window=50):
    """The organisers' baseline (starter/baseline.py) on the degraded external matches: team of the player
    closest to the ball, then a majority vote over ±window frames; D where nobody has the ball."""
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
    print('baseline written')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['train', 'events', 'decode', 'nested', 'export', 'baseline'])
    ap.add_argument('--params', help='for "export": json file with the parameters, in data/cache_mv/')
    ap.add_argument('--out', help='for "export": name of the folder in data/risultati_esterne/')
    ap.add_argument('--events', default=None, help='event probabilities to use in the decoder')
    ap.add_argument('--name', default='all')
    ap.add_argument('--sources', nargs='+', default=list(dati.SOURCES))
    ap.add_argument('--source-flag', action='store_true', help='adds the fonte_tv column (SkillCorner yes/no)')
    ap.add_argument('--short-weights', action='store_true', help='more weight to the frames of short possessions')
    ap.add_argument('--no-ricochets', action='store_true', help='trains on the DFL labels without ricochets')
    ap.add_argument('--tv-weight', type=float, default=1.0, help='weight of the SkillCorner frames (e.g. 0.25)')
    a = ap.parse_args()
    WEIGHTS = a.short_weights
    NO_RICOCHETS = a.no_ricochets
    TV_WEIGHT = a.tv_weight
    if a.what == 'train':
        train(a.name, set(a.sources), a.source_flag)
    elif a.what == 'nested':
        nested_report(a.name, a.events)
    elif a.what == 'events':
        train_events(a.name)
    elif a.what == 'export':
        export_external(a.name, a.params, a.out)
    elif a.what == 'baseline':
        export_baseline()
    else:
        decode_report(a.name, a.events)
