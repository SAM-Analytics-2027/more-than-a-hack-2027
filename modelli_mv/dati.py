"""Le 30 partite esterne (e Barça–Athletic) trasformate in tabelle di caratteristiche, salvate in data/cache_mv/.

Il tracking viene sempre dalla versione SPORCATA, le etichette dalla PULITA (come nel LEGGIMI).
La cache sta dentro data/, quindi GitHub non la vede.

    python modelli_mv/dati.py        prepara la cache (circa 2 minuti, in parallelo)
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import caratteristiche as C  # noqa: E402

BASE = os.path.join(C.fc.DATA_DIR, 'altre_partite')
CACHE = os.path.join(C.fc.DATA_DIR, 'cache_mv')
VERSION = 'v2'          # cambia quando cambiano le caratteristiche: v1 senza i tocchi


def _feat_dir():
    return os.path.join(CACHE, VERSION) if VERSION != 'v1' else CACHE
SOURCES = ('dfl', 'metrica', 'skillcorner')
CODE = {'H': 0, 'A': 1, 'D': 2, 'X': 3}


def external_matches():
    """[(fonte, id)] delle partite esterne pronte."""
    out = []
    for src in SOURCES:
        for path in sorted(glob.glob(os.path.join(BASE, src + '_sporcate', '*'))):
            if os.path.isdir(path):
                out.append((src, os.path.basename(path)))
    return out


def _path(src, mid):
    return os.path.join(_feat_dir(), '{}__{}.parquet'.format(src, mid))


def build_one(item):
    src, mid = item
    out = _path(src, mid)
    if os.path.exists(out):
        return out
    if src == 'barca':
        m = C.fc.read_ours()
        labels = np.full(len(m['period']), 'X')       # la risposta della gara non si usa mai
    else:
        m = C.fc.read_epts(os.path.join(BASE, src + '_sporcate', mid),
                           labels_folder=os.path.join(BASE, src + '_pulite', mid))
        labels = m['labels']
    df = C.features(m)
    df.insert(0, 'y', np.array([CODE[c] for c in labels], dtype=np.int8))
    df.insert(0, 'period', m['period'].astype(np.int8))
    df.insert(0, 'frame', np.arange(len(df), dtype=np.int32))
    os.makedirs(_feat_dir(), exist_ok=True)
    df.to_parquet(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)
    return out


def load(src, mid):
    return pd.read_parquet(_path(src, mid))


def load_all(items=None):
    """{(fonte, id): DataFrame} per le partite esterne."""
    items = items or external_matches()
    return {it: load(*it) for it in items}


if __name__ == '__main__':
    items = external_matches() + [('barca', 'BAR-ATH')]
    print(len(items), 'partite')
    with ProcessPoolExecutor(max_workers=4) as ex:
        for path in ex.map(build_one, items):
            print('pronta', os.path.basename(path), flush=True)
