"""Do the SkillCorner matches help to generalise to a NEVER-seen source? (the Barça situation)

A whole source is left out and used as the test:
    test on DFL:      training on Metrica, with or without SkillCorner
    test on Metrica:  training on DFL,     with or without SkillCorner
SkillCorner is tried in three ways: mixed in at equal weight, with the fonte_tv column, and with the column but at
weight 1/4 (they count as 5 matches). The decoder is the same for all (parameters of v2_dm); for completeness the
score with the decoder tuned on the test itself is also reported, the same for all (a ceiling, not a result).

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
    ('without SkillCorner', 0.0, False),
    ('+ SkillCorner mixed in', 1.0, False),
    ('+ SkillCorner with column', 1.0, True),
    ('+ SkillCorner with column, weight 1/4', 0.25, True),
]
PAIRS = [('dfl', 'metrica'), ('metrica', 'dfl')]     # (test source, training source)


def main():
    data = dati.load_all()
    params = json.load(open(os.path.join(dati.CACHE, 'results_dm_v2_dm.json')))['params']
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
            out[key] = {'fixed': float(np.mean(list(fixed.values()))), 'tuned_on_test': tuned_score,
                        'per_match': {it[1]: v for it, v in fixed.items()}}
            print('test on {:8s} trained on {:8s} {:36s} fixed decoder {:.4f} | tuned on the test {:.4f}  ({:.0f} s)'.format(
                test_src, train_src, label, out[key]['fixed'], tuned_score, time.time() - t), flush=True)
    with open(os.path.join(dati.CACHE, 'results_source_out.json'), 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
