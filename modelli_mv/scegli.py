"""Confronto di tutti i candidati e delle loro medie, con lo stesso giudice.

Per ogni candidato (probabilità "fuori gruppo" in data/cache_mv/oof_<nome>.npz):
    - si regola l'arbitro (Viterbi) sulle 10 partite DFL + Metrica;
    - si danno i punteggi: media sulle 10 partite, per fonte, partita peggiore, SkillCorner;
    - per le medie (ensemble) si provano tutte le combinazioni a pesi uguali dei candidati base.
Poi, per i migliori, il punteggio "annidato": l'arbitro è regolato senza il gruppo di partite giudicato.

    python modelli_mv/scegli.py v3_hgb xgb lgbm cascata rete
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
        'media10': float(np.mean(list(s.values()))),
        'dfl': per['dfl'], 'metrica': per['metrica'],
        'peggiore': float(min(s.values())),
        'skillcorner': float(np.mean(list(E.evaluate(oof, meta, params, sc).values()))) if sc else float('nan'),
        'per_partita': {it[1]: v for it, v in s.items()},
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
        print('{:28s} media10 {:.4f}  DFL {:.4f}  Metrica {:.4f}  peggiore {:.4f}  SkillCorner {:.4f}'.format(
            n, results[n]['media10'], results[n]['dfl'], results[n]['metrica'], results[n]['peggiore'],
            results[n]['skillcorner']), flush=True)
    best_single = max(results, key=lambda n: results[n]['media10'])
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            oof = {it: np.mean([base[n][it] for n in combo], axis=0) for it in items}
            key = 'media(' + '+'.join(combo) + ')'
            results[key] = score_candidate(oof, meta, items, start=results[best_single]['params'])
            print('{:28s} media10 {:.4f}  DFL {:.4f}  Metrica {:.4f}  peggiore {:.4f}  SkillCorner {:.4f}'.format(
                key, results[key]['media10'], results[key]['dfl'], results[key]['metrica'],
                results[key]['peggiore'], results[key]['skillcorner']), flush=True)
    ranking = sorted(results, key=lambda n: -results[n]['media10'])
    print('\nCLASSIFICA (media sulle 10 partite DFL + Metrica, arbitro regolato su di esse)')
    for i, n in enumerate(ranking[:10], 1):
        print('{:2d}. {:40s} {:.4f}  DFL {:.4f}  Metrica {:.4f}  peggiore {:.4f}'.format(
            i, n, results[n]['media10'], results[n]['dfl'], results[n]['metrica'], results[n]['peggiore']))
    print('\nPUNTEGGIO ANNIDATO dei primi 4 (arbitro regolato senza il gruppo giudicato)')
    for n in ranking[:4]:
        combo = n[len('media('):-1].split('+') if n.startswith('media(') else [n]
        oof = {it: np.mean([base[c][it] for c in combo], axis=0) for it in items}
        tot, per = nested(oof, meta, items)
        results[n]['annidato'] = tot
        print('   {:40s} {:.4f}  DFL {:.4f}  Metrica {:.4f}'.format(n, tot, per['dfl'], per['metrica']), flush=True)
    with open(os.path.join(dati.CACHE, 'risultati_scelta.json'), 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
