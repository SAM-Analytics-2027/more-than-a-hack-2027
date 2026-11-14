"""Più famiglie di modelli, con gli stessi dati, gli stessi gruppi di partite e lo stesso giudice.

Impostazione comune (quella vincente delle prove del 9 novembre):
    - tracking sporcato di tutte e 30 le partite, etichette originali;
    - colonna fonte_tv (SkillCorner sì/no) e frame SkillCorner a peso 1/4;
    - più peso ai possessi brevi; scambio delle squadre per raddoppiare gli esempi;
    - 5 gruppi di partite intere (esperimento.folds): ogni partita è prevista senza essere vista.

Famiglie:
    xgb      XGBoost, numero di alberi scelto su partite tenute da parte (mai su quella giudicata)
    lgbm     LightGBM, idem
    cascata  due modelli LightGBM come Stefano: "palla ferma?" e "di quale squadra?" (solo a palla viva)
    rete     rete neurale convoluzionale non causale sulla sequenza a 5 frame al secondo (±25 s di contesto)

Le probabilità "fuori gruppo" vanno in data/cache_mv/oof_<famiglia>.npz e si giudicano come le altre
(scegli.py). xgb, lgbm e cascata vogliono anche xgboost e lightgbm (non servono per il modello finale):

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
# Dati comuni
# --------------------------------------------------------------------------------------------------

def load():
    data = E.add_source_flag(dati.load_all())
    items = sorted(data)
    cols = E.feature_cols(next(iter(data.values())))
    return data, items, cols, E.folds(items)


def rows(data, items, cols):
    """Righe di allenamento (un frame ogni 8 per DFL/Metrica, ogni 16 a peso 2*TV_WEIGHT per SkillCorner, più scambio squadre)."""
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
    """Partite di allenamento tenute da parte per decidere quando fermarsi: quelle del gruppo successivo."""
    val = [it for it in train_items if fold_of[it] == (k + 1) % E.N_FOLDS]
    return [it for it in train_items if it not in val], val


def save(name, oof, rounds=None):
    if rounds:
        with open(os.path.join(dati.CACHE, 'alberi_{}.json'.format(name)), 'w') as f:
            json.dump(rounds, f)
    path = os.path.join(dati.CACHE, 'oof_{}.npz'.format(name))
    np.savez_compressed(path, **{'{}__{}'.format(*it): p.astype(np.float32) for it, p in oof.items()})
    print('salvate', path, flush=True)


def sym_predict(predict, df, cols):
    """Media della previsione diretta e di quella a squadre scambiate (rimesse a posto)."""
    X = df[cols]
    p1 = predict(X)
    p2 = predict(C.swap_teams(X)[cols])[:, [1, 0, 2]]
    return (p1 + p2) / 2


# --------------------------------------------------------------------------------------------------
# Boosting: XGBoost e LightGBM
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
        print('{} gruppo {}: {} alberi, {:.0f} s'.format(kind, k, best, time.time() - t), flush=True)
    save(kind, oof, rounds)


def run_cascade():
    """Come Stefano: P(D) da un modello "palla ferma?", P(H | palla viva) da un modello "di quale squadra?"."""
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
        print('cascata gruppo {}: {} + {} alberi, {:.0f} s'.format(k, bd, bt, time.time() - t), flush=True)
    save('cascata', oof, rounds)


# --------------------------------------------------------------------------------------------------
# Rete neurale convoluzionale sulla sequenza: il codice sta in rete.py (serve PyTorch)
# --------------------------------------------------------------------------------------------------

def train_net(data, train, cols, seed):
    """Allena la rete sulle partite `train`; restituisce una funzione che dà (T, 3) probabilità per una partita."""
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
        print('rete gruppo {}: {} passi per epoca, {:.0f} s'.format(k, per_epoch, time.time() - t), flush=True)
    if os.environ.get('MV_SMOKE'):
        print('prova veloce ok:', {it[1]: oof[it].shape for it in list(oof)[:2]})
        return
    save('rete', oof)


def run_net_seed(seed_offset, name=None):
    """La stessa rete con un altro seme casuale: se il punteggio cambia molto, il risultato era fortuna.

    seed_offset 0 dà gli stessi semi di run_net (SEED + gruppo).
    """
    data, items, cols, fold_of = load()
    oof = {}
    for k in range(E.N_FOLDS):
        train = [it for it in items if fold_of[it] != k]
        predict, _ = train_net(data, train, cols, SEED + 1000 * seed_offset + k)
        for it in [i for i in items if fold_of[i] == k]:
            oof[it] = predict(data[it], it)
        print('rete seme {} gruppo {} fatto'.format(seed_offset, k), flush=True)
    save(name or 'rete_seme{}'.format(seed_offset), oof)


def run_net_source_out():
    """Prova "fonte mai vista" per la rete: allenata su una fonte (+ SkillCorner a peso 1/4), giudicata sull'altra."""
    data, items, cols, _ = load()
    oof = {}
    for test_src, train_src in (('dfl', 'metrica'), ('metrica', 'dfl')):
        train = [it for it in items if it[0] in (train_src, 'skillcorner')]
        predict, _ = train_net(data, train, cols, SEED)
        for it in [i for i in items if i[0] == test_src]:
            oof[it] = predict(data[it], it)
        print('rete fonte fuori: prova su {} fatta'.format(test_src), flush=True)
    save('rete_fonte_fuori', oof)


# --------------------------------------------------------------------------------------------------
# Modelli finali, allenati su tutte e 30 le partite, e risposte per il Barça
# --------------------------------------------------------------------------------------------------

def run_final(kind, seed_offset=0):
    """Allena su tutte le partite con il numero di alberi mediano dei gruppi e salva le probabilità del Barça.

    Per la rete, seed_offset sceglie il seme: le versioni con semi diversi si fanno poi in media.
    """
    data, items, cols, _ = load()
    target = dati.load('barca', 'BAR-ATH')
    target['fonte_tv'] = np.float32(0)          # il Barça ha una telecamera tattica, non le riprese tv
    t = time.time()
    if kind == 'rete':
        predict, _ = train_net(data, items, cols, SEED + 1000 * seed_offset)
        proba = predict(target, ('barca', 'BAR-ATH'))
    else:
        rounds = json.load(open(os.path.join(dati.CACHE, 'alberi_{}.json'.format(kind))))
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
    name = kind if not seed_offset else '{}_seme{}'.format(kind, seed_offset)
    np.save(os.path.join(dati.CACHE, 'barca_proba_{}.npy'.format(name)), proba.astype(np.float32))
    print('finale {}: probabilità del Barça salvate ({:.0f} s)'.format(kind, time.time() - t), flush=True)


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'finale':
        run_final(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif what == 'seme':
        run_net_seed(int(sys.argv[2]))
    elif what == 'fonte_fuori':
        run_net_source_out()
    else:
        {'xgb': lambda: run_boosting('xgb'), 'lgbm': lambda: run_boosting('lgbm'),
         'cascata': run_cascade, 'rete': run_net}[what]()
