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
(confronto_dati.py). Gira nell'ambiente conda 'mtah' (xgboost, lightgbm, pytorch):

    /Users/matteovezzoli/anaconda3/envs/mtah/bin/python modelli_mv/famiglie.py xgb
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

TV_WEIGHT = 0.25
SEED = 2027
THREADS = 8


# --------------------------------------------------------------------------------------------------
# Dati comuni
# --------------------------------------------------------------------------------------------------

def load():
    data = E.add_source_flag(dati.load_all())
    items = sorted(data)
    cols = E.feature_cols(next(iter(data.values())))
    return data, items, cols, E.folds(items)


def rows(data, items, cols):
    """Righe di allenamento (un frame ogni 8, più lo scambio delle squadre) con i pesi."""
    E.WEIGHTS = True
    Xs, ys, ws = [], [], []
    for it in items:
        X, y, w = E.training_rows(data[it], cols)
        Xs.append(X)
        ys.append(y)
        ws.append(w * (TV_WEIGHT if it[0] == 'skillcorner' else 1.0))
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
    params = dict(objective='multi:softprob', num_class=3, tree_method='hist', max_depth=8, eta=0.06,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=50, reg_lambda=2.0,
                  eval_metric='mlogloss', nthread=THREADS, seed=SEED)
    dtr = xgb.DMatrix(Xtr.values, label=ytr, weight=wtr)
    dva = xgb.DMatrix(Xva.values, label=yva, weight=wva)
    booster = xgb.train(params, dtr, num_boost_round=2000, evals=[(dva, 'val')],
                        early_stopping_rounds=60, verbose_eval=False)
    best = booster.best_iteration + 1
    return booster, best, lambda X: booster.predict(xgb.DMatrix(X.values), iteration_range=(0, best))


def fit_lgbm(Xtr, ytr, wtr, Xva, yva, wva, objective='multiclass', n_class=3):
    import lightgbm as lgb
    params = dict(objective=objective, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  num_threads=THREADS, seed=SEED, verbose=-1)
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
# Rete neurale ricorrente sulla sequenza
# --------------------------------------------------------------------------------------------------

STRIDE = 5        # 5 frame al secondo: la rete guarda la sequenza a 0,2 s di passo
CROP = 300        # tratti di 60 s per allenare
EPOCHS = int(os.environ.get('MV_EPOCHS', 20))


_NET_CACHE = {}


def _net_raw(df, it, cols):
    """Una volta per partita: i frame a 5 al secondo di ogni tempo, diretti e a squadre scambiate (non normalizzati)."""
    if it in _NET_CACHE:
        return _NET_CACHE[it]
    per = df['period'].values
    y_all = df['y'].values.astype(np.int64)
    w_all = E.short_spell_weights(df['y'].values, per) * (TV_WEIGHT if it[0] == 'skillcorner' else 1.0)
    out = []
    for half in (1, 2):
        idx = np.flatnonzero(per == half)
        if len(idx) < CROP * STRIDE:
            continue
        sel = idx[::STRIDE]
        sub = df[cols].iloc[sel]
        xo = sub.values.astype(np.float32)
        xs = C.swap_teams(sub)[cols].values.astype(np.float32)
        y = y_all[sel]
        ys = np.where(y == 0, 1, np.where(y == 1, 0, y))
        out.append((xo, xs, y, ys, w_all[sel].astype(np.float32), sel))
    _NET_CACHE[it] = out
    return out


def _net_seqs(df, it, cols, med, iqr, swap=False):
    """Per ogni tempo: (caratteristiche normalizzate + indicatori dei mancanti, etichette, pesi, frame)."""
    out = []
    for xo, xs, y, ys, w, sel in _net_raw(df, it, cols):
        x = ((xs if swap else xo) - med) / iqr
        miss = np.isnan(x)
        x = np.clip(np.nan_to_num(x, nan=0.0), -6, 6)
        x = np.concatenate([x, miss.astype(np.float32)], axis=1).astype(np.float32)
        out.append((x, ys if swap else y, w, sel))
    return out


def train_net(data, train, cols, seed):
    """Allena la rete sulle partite `train`; restituisce una funzione che dà (T, 3) probabilità per una partita."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    allX = np.concatenate([data[it][cols].values[::40] for it in train])
    med = np.nanmedian(allX, axis=0)
    iqr = np.nanpercentile(allX, 75, axis=0) - np.nanpercentile(allX, 25, axis=0)
    iqr = np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, 1.0)
    med = np.where(np.isfinite(med), med, 0.0)
    seqs = []
    for it in train:
        seqs += _net_seqs(data[it], it, cols, med, iqr) + _net_seqs(data[it], it, cols, med, iqr, swap=True)

    class Block(nn.Module):
        """Convoluzione "dilatata": guarda frame sempre più lontani, avanti e indietro, con un collegamento diretto."""
        def __init__(self, c, d):
            super().__init__()
            self.conv = nn.Conv1d(c, c, 3, padding=d, dilation=d)
            self.norm = nn.BatchNorm1d(c)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.15)

        def forward(self, x):
            return x + self.drop(self.act(self.norm(self.conv(x))))

    class Net(nn.Module):
        """Rete convoluzionale non causale: ogni momento vede ±127 passi (±25 s) intorno a sé."""
        def __init__(self, n_in, c=96):
            super().__init__()
            self.inp = nn.Conv1d(n_in, c, 1)
            self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 16, 32, 64)])
            self.out = nn.Conv1d(c, 3, 1)

        def forward(self, x):
            return self.out(self.blocks(self.inp(x.transpose(1, 2)))).transpose(1, 2)

    net = Net(seqs[0][0].shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lens = np.array([len(q[0]) for q in seqs], dtype=float)
    per_epoch = max(1, int(lens.sum()) // (CROP * 32))
    if os.environ.get('MV_SMOKE'):
        per_epoch = 3
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * per_epoch)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        net.train()
        for _ in range(per_epoch):
            xb, yb, wb = [], [], []
            for si in rng.choice(len(seqs), size=32, p=lens / lens.sum()):
                x, y, w, _ = seqs[si]
                a = rng.integers(0, len(x) - CROP + 1)
                xb.append(x[a:a + CROP])
                yb.append(y[a:a + CROP])
                wb.append(w[a:a + CROP])
            xb = torch.from_numpy(np.stack(xb)).to(dev)
            yb = torch.from_numpy(np.stack(yb)).to(dev)
            wb = torch.from_numpy(np.stack(wb)).to(dev)
            logits = net(xb)
            valid = (yb < 3).float() * wb
            loss = nn.functional.cross_entropy(logits.reshape(-1, 3), yb.clamp(max=2).reshape(-1), reduction='none')
            loss = (loss * valid.reshape(-1)).sum() / valid.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    net.eval()

    def predict(df, it):
        T = len(df)
        probs = []
        for swap in (False, True):
            full = np.full((T, 3), 1 / 3, dtype=np.float32)
            for x, _, _, sel in _net_seqs(df, it, cols, med, iqr, swap=swap):
                with torch.no_grad():
                    p = torch.softmax(net(torch.from_numpy(x[None]).to(dev)), -1)[0].cpu().numpy()
                if swap:
                    p = p[:, [1, 0, 2]]
                a, b = sel[0], sel[-1] + 1      # da 5 a 25 frame al secondo: interpolazione tra i frame vicini
                grid = np.arange(a, b)
                full[a:b] = np.stack([np.interp(grid, sel, p[:, c]) for c in range(3)], axis=1)
            probs.append(full)
        return (probs[0] + probs[1]) / 2
    return predict, per_epoch


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


def run_net_seed(seed_offset):
    """La stessa rete con un altro seme casuale: se il punteggio cambia molto, il risultato era fortuna."""
    data, items, cols, fold_of = load()
    oof = {}
    for k in range(E.N_FOLDS):
        train = [it for it in items if fold_of[it] != k]
        predict, _ = train_net(data, train, cols, SEED + 1000 * seed_offset + k)
        for it in [i for i in items if fold_of[i] == k]:
            oof[it] = predict(data[it], it)
        print('rete seme {} gruppo {} fatto'.format(seed_offset, k), flush=True)
    save('rete_seme{}'.format(seed_offset), oof)


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
