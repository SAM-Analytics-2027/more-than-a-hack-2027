"""Several model families, with the same data, the same folds of matches and the same judge.

Common setup (the winning one from the 9 November experiments):
    - degraded tracking of all 30 matches, original labels;
    - fonte_tv column (SkillCorner yes/no) and SkillCorner frames at weight 1/4;
    - more weight to short possessions; team swap to double the examples;
    - 5 folds of whole matches (esperimento.folds): every match is predicted without being seen.

Families:
    xgb      XGBoost, number of trees chosen on held-out matches (never on the judged one)
    lgbm     LightGBM, same
    cascade  two LightGBM models: "dead ball?" and "which team?" (live ball only)
    net      non-causal convolutional neural network on the sequence at 5 frames per second (±25 s of context)

The out-of-fold probabilities go to data/cache_mv/oof_<family>.npz and are judged like the others
(scegli.py). xgb, lgbm and cascade also need xgboost and lightgbm:

    python modelli_mv/famiglie.py xgb
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import caratteristiche as C  # noqa: E402
import dati  # noqa: E402
import esperimento as E  # noqa: E402
from constants import TV_WEIGHT, SEED, THREADS, XGB_PARAMS, LGBM_PARAMS  # noqa: E402


# --------------------------------------------------------------------------------------------------
# Common data
# --------------------------------------------------------------------------------------------------

def load():
    data = E.add_source_flag(dati.load_all())
    items = sorted(data)
    cols = E.feature_cols(next(iter(data.values())))
    return data, items, cols, E.folds(items)


def rows(data, items, cols):
    """Training rows (one frame every 8 for DFL/Metrica, every 16 at weight 2*TV_WEIGHT for SkillCorner, plus team swap)."""
    E.WEIGHTS = True
    Xs, ys, ws = [], [], []
    for it in items:
        X, y, w = E.training_rows(data[it], cols)
        if it[0] == 'skillcorner':
            X = X.iloc[::2]
            y = y[::2]
            w = w[::2] * (2.0 * TV_WEIGHT)
        Xs.append(X)
        ys.append(y)
        ws.append(w)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


def split_inner(train_items, fold_of, k):
    """Training matches held out to decide when to stop: those of the next fold."""
    val = [it for it in train_items if fold_of[it] == (k + 1) % E.N_FOLDS]
    return [it for it in train_items if it not in val], val


def save(name, oof, rounds=None):
    if rounds:
        with open(os.path.join(dati.CACHE, 'trees_{}.json'.format(name)), 'w') as f:
            json.dump(rounds, f)
    path = os.path.join(dati.CACHE, 'oof_{}.npz'.format(name))
    np.savez_compressed(path, **{'{}__{}'.format(*it): p.astype(np.float32) for it, p in oof.items()})
    print('saved', path, flush=True)


def sym_predict(predict, df, cols):
    """Mean of the direct prediction and of the one with the teams swapped (swapped back)."""
    X = df[cols]
    p1 = predict(X)
    p2 = predict(C.swap_teams(X)[cols])[:, [1, 0, 2]]
    return (p1 + p2) / 2


# --------------------------------------------------------------------------------------------------
# Boosting: XGBoost and LightGBM
# --------------------------------------------------------------------------------------------------

def fit_xgb(Xtr, ytr, wtr, Xva, yva, wva):
    import xgboost as xgb
    params = dict(XGB_PARAMS)
    dtr = xgb.DMatrix(Xtr.values, label=ytr, weight=wtr)
    dva = xgb.DMatrix(Xva.values, label=yva, weight=wva)
    booster = xgb.train(params, dtr, num_boost_round=2000, evals=[(dva, 'val')],
                        early_stopping_rounds=60, verbose_eval=False)
    best = booster.best_iteration + 1
    return booster, best, lambda X: booster.predict(xgb.DMatrix(X.values), iteration_range=(0, best))


def fit_lgbm(Xtr, ytr, wtr, Xva, yva, wva, objective='multiclass', n_class=3):
    import lightgbm as lgb
    params = dict(LGBM_PARAMS, objective=objective)
    if objective == 'multiclass':
        params['num_class'] = n_class
    dtr = lgb.Dataset(Xtr.values, label=ytr, weight=wtr, free_raw_data=True)
    dva = lgb.Dataset(Xva.values, label=yva, weight=wva, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(80, verbose=False)])
    best = booster.best_iteration

    def predict(X):
        p = booster.predict(X.values, num_iteration=best)
        return p if p.ndim == 2 else p[:, None]
    return booster, best, predict


def run_boosting(kind):
    data, items, cols, fold_of = load()
    oof, rounds = {}, []
    for k in range(E.N_FOLDS):
        t = time.time()
        train = [it for it in items if fold_of[it] != k]
        test = [it for it in items if fold_of[it] == k]
        tr, va = split_inner(train, fold_of, k)
        Xtr, ytr, wtr = rows(data, tr, cols)
        Xva, yva, wva = rows(data, va, cols)
        fit = fit_xgb if kind == 'xgb' else fit_lgbm
        _, best, predict = fit(Xtr, ytr, wtr, Xva, yva, wva)
        rounds.append(int(best))
        for it in test:
            oof[it] = sym_predict(predict, data[it], cols)
        print('{} fold {}: {} trees, {:.0f} s'.format(kind, k, best, time.time() - t), flush=True)
    save(kind, oof, rounds)


def run_cascade():
    """P(D) from a "dead ball?" model, P(H | live ball) from a "which team?" model."""
    data, items, cols, fold_of = load()
    oof, rounds = {}, []
    for k in range(E.N_FOLDS):
        t = time.time()
        train = [it for it in items if fold_of[it] != k]
        test = [it for it in items if fold_of[it] == k]
        tr, va = split_inner(train, fold_of, k)
        Xtr, ytr, wtr = rows(data, tr, cols)
        Xva, yva, wva = rows(data, va, cols)
        _, bd, pdead = fit_lgbm(Xtr, (ytr == 2).astype(int), wtr, Xva, (yva == 2).astype(int), wva, objective='binary')
        a_tr, a_va = ytr < 2, yva < 2
        _, bt, phome = fit_lgbm(Xtr[a_tr], (ytr[a_tr] == 0).astype(int), wtr[a_tr],
                                Xva[a_va], (yva[a_va] == 0).astype(int), wva[a_va], objective='binary')

        def predict(X):
            d = pdead(X)[:, 0]
            h = phome(X)[:, 0]
            return np.stack([(1 - d) * h, (1 - d) * (1 - h), d], axis=1)
        for it in test:
            oof[it] = sym_predict(predict, data[it], cols)
        rounds.append([int(bd), int(bt)])
        print('cascade fold {}: {} + {} trees, {:.0f} s'.format(k, bd, bt, time.time() - t), flush=True)
    save('cascade', oof, rounds)


# --------------------------------------------------------------------------------------------------
# Convolutional neural network on the sequence: the code is in rete.py (needs PyTorch)
# --------------------------------------------------------------------------------------------------

def train_net(data, train, cols, seed):
    """Trains the network on the matches `train`; returns a function that gives (T, 3) probabilities for a match."""
    import rete
    model = rete.train(data, train, cols, seed)
    return (lambda df, it: rete.predict(model, df, it)), model['per_epoch']


def run_net():
    data, items, cols, fold_of = load()
    oof = {}
    folds_to_run = [0] if os.environ.get('MV_SMOKE') else range(E.N_FOLDS)
    for k in folds_to_run:
        t = time.time()
        train = [it for it in items if fold_of[it] != k]
        test = [it for it in items if fold_of[it] == k]
        predict, per_epoch = train_net(data, train, cols, SEED + k)
        for it in test:
            oof[it] = predict(data[it], it)
        print('net fold {}: {} steps per epoch, {:.0f} s'.format(k, per_epoch, time.time() - t), flush=True)
    if os.environ.get('MV_SMOKE'):
        print('smoke test ok:', {it[1]: oof[it].shape for it in list(oof)[:2]})
        return
    save('net', oof)


def run_net_seed(seed_offset, name=None):
    """The same network with another random seed: if the score changes a lot, the result was luck.

    seed_offset 0 gives the same seeds as run_net (SEED + fold).
    """
    data, items, cols, fold_of = load()
    oof = {}
    for k in range(E.N_FOLDS):
        train = [it for it in items if fold_of[it] != k]
        predict, _ = train_net(data, train, cols, SEED + 1000 * seed_offset + k)
        for it in [i for i in items if fold_of[i] == k]:
            oof[it] = predict(data[it], it)
        print('net seed {} fold {} done'.format(seed_offset, k), flush=True)
    save(name or 'net_seed{}'.format(seed_offset), oof)


def run_net_source_out():
    """"Never-seen source" test for the network: trained on one source (+ SkillCorner at weight 1/4), judged on the other."""
    data, items, cols, _ = load()
    oof = {}
    for test_src, train_src in (('dfl', 'metrica'), ('metrica', 'dfl')):
        train = [it for it in items if it[0] in (train_src, 'skillcorner')]
        predict, _ = train_net(data, train, cols, SEED)
        for it in [i for i in items if i[0] == test_src]:
            oof[it] = predict(data[it], it)
        print('net source out: test on {} done'.format(test_src), flush=True)
    save('net_source_out', oof)


# --------------------------------------------------------------------------------------------------
# Final models, trained on all 30 matches, and answers for Barça-Athletic
# --------------------------------------------------------------------------------------------------

def run_final(kind, seed_offset=0):
    """Trains on all the matches with the median number of trees of the folds and saves the Barça-Athletic probabilities.

    For the network, seed_offset picks the seed: the versions with different seeds are then averaged.
    """
    data, items, cols, _ = load()
    target = dati.load('barca', 'BAR-ATH')
    target['fonte_tv'] = np.float32(0)          # Barça-Athletic has a tactical camera, not TV footage
    t = time.time()
    if kind == 'net':
        predict, _ = train_net(data, items, cols, SEED + 1000 * seed_offset)
        proba = predict(target, ('barca', 'BAR-ATH'))
    else:
        rounds = json.load(open(os.path.join(dati.CACHE, 'trees_{}.json'.format(kind))))
        X, y, w = rows(data, items, cols)
        if kind == 'xgb':
            import xgboost as xgb
            n = int(np.median(rounds))
            params = dict(objective='multi:softprob', num_class=3, tree_method='hist', max_depth=8, eta=0.06,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=50, reg_lambda=2.0,
                          nthread=THREADS, seed=SEED)
            booster = xgb.train(params, xgb.DMatrix(X.values, label=y, weight=w), num_boost_round=n)
            proba = sym_predict(lambda Z: booster.predict(xgb.DMatrix(Z.values)), target, cols)
        else:
            import lightgbm as lgb

            def fit(objective, Xf, yf, wf, n):
                params = dict(objective=objective, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                              num_threads=THREADS, seed=SEED, verbose=-1)
                if objective == 'multiclass':
                    params['num_class'] = 3
                return lgb.train(params, lgb.Dataset(Xf.values, label=yf, weight=wf), num_boost_round=n)
            if kind == 'lgbm':
                b = fit('multiclass', X, y, w, int(np.median(rounds)))
                proba = sym_predict(lambda Z: b.predict(Z.values), target, cols)
            else:
                nd = int(np.median([r[0] for r in rounds]))
                nt = int(np.median([r[1] for r in rounds]))
                bd = fit('binary', X, (y == 2).astype(int), w, nd)
                alive = y < 2
                bt = fit('binary', X[alive], (y[alive] == 0).astype(int), w[alive], nt)

                def predict(Z):
                    d, h = bd.predict(Z.values), bt.predict(Z.values)
                    return np.stack([(1 - d) * h, (1 - d) * (1 - h), d], axis=1)
                proba = sym_predict(predict, target, cols)
    name = kind if not seed_offset else '{}_seed{}'.format(kind, seed_offset)
    np.save(os.path.join(dati.CACHE, 'barca_proba_{}.npy'.format(name)), proba.astype(np.float32))
    print('final {}: Barça-Athletic probabilities saved ({:.0f} s)'.format(kind, time.time() - t), flush=True)


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'final':
        run_final(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif what == 'seed':
        run_net_seed(int(sys.argv[2]))
    elif what == 'source_out':
        run_net_source_out()
    else:
        {'xgb': lambda: run_boosting('xgb'), 'lgbm': lambda: run_boosting('lgbm'),
         'cascade': run_cascade, 'net': run_net}[what]()
