"""Più partite aiutano? Stesso modello allenato su 10 partite (DFL + Metrica) o su tutte e 30.

Si confrontano sulle stesse partite di prova (DFL e Metrica, sempre lasciate fuori), con il decodificatore
regolato nello stesso modo: solo su DFL e Metrica.

    python modelli_mv/esperimento.py allena --nome v2_dm --fonti dfl metrica
    python modelli_mv/confronto_dati.py v2 v2_dm
"""
import json
import os
import sys

import numpy as np

import dati
import esperimento as E


def main(names):
    test = None
    for name in names:
        oof = E.load_oof(name)
        items = sorted(oof)
        meta = E.meta_of(items)
        dm = [it for it in items if it[0] in ('dfl', 'metrica')]
        sc = [it for it in items if it[0] == 'skillcorner']
        params, _ = E.tune(oof, meta, dm, rounds=1)
        s_dm = E.evaluate(oof, meta, params, dm)
        s_sc = E.evaluate(oof, meta, params, sc)
        _, per = E.summary(s_dm)
        with open(os.path.join(dati.CACHE, 'risultati_dm_{}.json'.format(name)), 'w') as f:
            json.dump({'params': params, 'per_fonte': per, 'skillcorner': float(np.mean(list(s_sc.values()))),
                       'per_partita': {'{}__{}'.format(*k): v for k, v in {**s_dm, **s_sc}.items()}}, f, indent=1)
        print('{:8s} DFL {:.4f}  Metrica {:.4f}  | SkillCorner {:.4f}   parametri {}'.format(
            name, per['dfl'], per['metrica'], np.mean(list(s_sc.values())), params))
        if test is None:
            test = s_dm
        else:
            d = np.array([s_dm[k] - test[k] for k in test])
            print('   rispetto a {}: {:+.4f} in media, {} partite su {} cambiano in meglio'.format(
                names[0], d.mean(), int((d > 0).sum()), len(d)))


if __name__ == '__main__':
    main(sys.argv[1:])
