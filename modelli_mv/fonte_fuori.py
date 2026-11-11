"""Le SkillCorner aiutano a generalizzare a una fonte MAI vista? (la situazione del Barça)

Si lascia fuori un'intera fonte e la si usa come prova:
    prova su DFL:     allenamento su Metrica, con o senza le SkillCorner
    prova su Metrica: allenamento su DFL,     con o senza le SkillCorner
Le SkillCorner si provano in tre modi: mescolate alla pari, con la colonna fonte_tv, e con la colonna ma a peso 1/4
(valgono come 5 partite). Il decodificatore è lo stesso per tutti (parametri di v2_dm); per completezza si riporta
anche il punteggio con il decodificatore regolato sulla prova stessa, uguale per tutti (un tetto, non un risultato).

    python modelli_mv/fonte_fuori.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dati  # noqa: E402
import esperimento as E  # noqa: E402

CONFIGS = [
    ('senza SkillCorner', 0.0, False),
    ('+ SkillCorner mescolate', 1.0, False),
    ('+ SkillCorner con colonna', 1.0, True),
    ('+ SkillCorner con colonna, peso 1/4', 0.25, True),
]
PAIRS = [('dfl', 'metrica'), ('metrica', 'dfl')]     # (fonte di prova, fonte di allenamento)


def main():
    data = dati.load_all()
    params = json.load(open(os.path.join(dati.CACHE, 'risultati_dm_v2_dm.json')))['params']
    base_cols = E.feature_cols(next(iter(data.values())))
    meta = E.meta_of(sorted(data))
    out = {}
    for test_src, train_src in PAIRS:
        test = [it for it in sorted(data) if it[0] == test_src]
        for label, w_sc, flag in CONFIGS:
            t = time.time()
            cols = base_cols + (['fonte_tv'] if flag else [])
            train = [it for it in sorted(data) if it[0] == train_src or (w_sc > 0 and it[0] == 'skillcorner')]
            Xs, ys, ws = [], [], []
            for it in train:
                df = data[it].assign(fonte_tv=np.float32(it[0] == 'skillcorner'))
                X, y, w = E.training_rows(df, cols)
                Xs.append(X)
                ys.append(y)
                ws.append(w * (w_sc if it[0] == 'skillcorner' else 1.0))
            model = E.make_model().fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys),
                                       sample_weight=np.concatenate(ws))
            proba = {it: E.predict_sym(model, data[it].assign(fonte_tv=np.float32(0)), cols).astype(np.float32)
                     for it in test}
            fixed = E.evaluate(proba, meta, params, test)
            _, tuned_score = E.tune(proba, meta, test, start=params, rounds=1)
            key = '{} | {}'.format(test_src, label)
            out[key] = {'fisso': float(np.mean(list(fixed.values()))), 'regolato_sulla_prova': tuned_score,
                        'per_partita': {it[1]: v for it, v in fixed.items()}}
            print('prova su {:8s} allenato su {:8s} {:36s} decodificatore fisso {:.4f} | regolato sulla prova {:.4f}  ({:.0f} s)'.format(
                test_src, train_src, label, out[key]['fisso'], tuned_score, time.time() - t), flush=True)
    with open(os.path.join(dati.CACHE, 'risultati_fonte_fuori.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
