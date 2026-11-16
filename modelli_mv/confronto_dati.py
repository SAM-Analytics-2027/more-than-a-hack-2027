"""Do more matches help? The same model trained on 10 matches (DFL + Metrica) or on all 30.

They are compared on the same test matches (DFL and Metrica, always left out), with the decoder
tuned in the same way: on DFL and Metrica only.

    python modelli_mv/esperimento.py train --name v2_dm --sources dfl metrica
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
        with open(os.path.join(dati.CACHE, 'results_dm_{}.json'.format(name)), 'w') as f:
            json.dump({'params': params, 'per_source': per, 'skillcorner': float(np.mean(list(s_sc.values()))),
                       'per_match': {'{}__{}'.format(*k): v for k, v in {**s_dm, **s_sc}.items()}}, f, indent=1)
        print('{:8s} DFL {:.4f}  Metrica {:.4f}  | SkillCorner {:.4f}   parameters {}'.format(
            name, per['dfl'], per['metrica'], np.mean(list(s_sc.values())), params))
        if test is None:
            test = s_dm
        else:
            d = np.array([s_dm[k] - test[k] for k in test])
            print('   compared with {}: {:+.4f} on average, {} matches out of {} improve'.format(
                names[0], d.mean(), int((d > 0).sum()), len(d)))


if __name__ == '__main__':
    main(sys.argv[1:])
