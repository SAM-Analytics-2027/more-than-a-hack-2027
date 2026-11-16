"""Comparison of all the candidates and of their averages, with the same judge.

For each candidate (out-of-fold probabilities in data/cache_mv/oof_<name>.npz):
    - the decoder (Viterbi) is tuned on the 10 DFL + Metrica matches;
    - the scores are given: mean over the 10 matches, per source, worst match, SkillCorner;
    - for the averages (ensembles) all equal-weight combinations of the base candidates are tried.
Then, for the best ones, the "nested" score: the decoder is tuned without the fold of matches being judged.

    python modelli_mv/scegli.py v3_hgb xgb lgbm cascade net
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dati  # noqa: E402
import esperimento as E  # noqa: E402
from constants import DM_SOURCES as DM  # noqa: E402


def score_candidate(oof, meta, items, start=None):
    dm = [it for it in items if it[0] in DM]
    sc = [it for it in items if it[0] == 'skillcorner']
    params, _ = E.tune(oof, meta, dm, start=start, rounds=1)
    s = E.evaluate(oof, meta, params, dm)
    _, per = E.summary(s)
    return {
        'params': params,
        'mean10': float(np.mean(list(s.values()))),
        'dfl': per['dfl'], 'metrica': per['metrica'],
        'worst': float(min(s.values())),
        'skillcorner': float(np.mean(list(E.evaluate(oof, meta, params, sc).values()))) if sc else float('nan'),
        'per_match': {it[1]: v for it, v in s.items()},
    }


def nested(oof, meta, items):
    dm = [it for it in items if it[0] in DM]
    fold_of = E.folds(items)
    out = {}
    for k in range(E.N_FOLDS):
        tr = [it for it in dm if fold_of[it] != k]
        te = [it for it in dm if fold_of[it] == k]
        if not te:
            continue
        params, _ = E.tune(oof, meta, tr, rounds=1)
        out.update(E.evaluate(oof, meta, params, te))
    return float(np.mean(list(out.values()))), E.summary(out)[1]


def main(names):
    base = {n: E.load_oof(n) for n in names}
    items = sorted(next(iter(base.values())))
    meta = E.meta_of(items)
    results = {}
    for n, oof in base.items():
        results[n] = score_candidate(oof, meta, items)
        print('{:28s} mean10 {:.4f}  DFL {:.4f}  Metrica {:.4f}  worst {:.4f}  SkillCorner {:.4f}'.format(
            n, results[n]['mean10'], results[n]['dfl'], results[n]['metrica'], results[n]['worst'],
            results[n]['skillcorner']), flush=True)
    best_single = max(results, key=lambda n: results[n]['mean10'])
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            oof = {it: np.mean([base[n][it] for n in combo], axis=0) for it in items}
            key = 'mean(' + '+'.join(combo) + ')'
            results[key] = score_candidate(oof, meta, items, start=results[best_single]['params'])
            print('{:28s} mean10 {:.4f}  DFL {:.4f}  Metrica {:.4f}  worst {:.4f}  SkillCorner {:.4f}'.format(
                key, results[key]['mean10'], results[key]['dfl'], results[key]['metrica'],
                results[key]['worst'], results[key]['skillcorner']), flush=True)
    ranking = sorted(results, key=lambda n: -results[n]['mean10'])
    print('\nRANKING (mean over the 10 DFL + Metrica matches, decoder tuned on them)')
    for i, n in enumerate(ranking[:10], 1):
        print('{:2d}. {:40s} {:.4f}  DFL {:.4f}  Metrica {:.4f}  worst {:.4f}'.format(
            i, n, results[n]['mean10'], results[n]['dfl'], results[n]['metrica'], results[n]['worst']))
    print('\nNESTED SCORE of the top 4 (decoder tuned without the judged fold)')
    for n in ranking[:4]:
        combo = n[len('mean('):-1].split('+') if n.startswith('mean(') else [n]
        oof = {it: np.mean([base[c][it] for c in combo], axis=0) for it in items}
        tot, per = nested(oof, meta, items)
        results[n]['nested'] = tot
        print('   {:40s} {:.4f}  DFL {:.4f}  Metrica {:.4f}'.format(n, tot, per['dfl'], per['metrica']), flush=True)
    with open(os.path.join(dati.CACHE, 'results_selection.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
