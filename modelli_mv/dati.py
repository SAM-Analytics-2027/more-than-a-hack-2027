"""Le 30 partite esterne (e Barça–Athletic) trasformate in tabelle di caratteristiche, salvate in data/cache_mv/.

Il tracking viene sempre dalla versione SPORCATA, le etichette dalla PULITA:
    data/altre_partite/<fonte>_sporcate/<id>/<id>_atd.txt e _atd.xml     (l'input del modello)
    data/altre_partite/<fonte>_pulite/<id>/<id>_possession_gt.csv        (la risposta giusta, solo per allenare)
Le partite sono un elenco fisso (MATCHES): cartelle in più dentro data/altre_partite/ non cambiano niente.

La cache sta dentro data/, quindi GitHub non la vede. Il suo nome contiene un'impronta del codice che calcola le
caratteristiche: se il codice cambia, la cache si rifà da sola e non si usa mai una cache vecchia.

    python modelli_mv/dati.py        prepara la cache (circa 2 minuti, in parallelo)
"""
import hashlib
import os
import sys
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caratteristiche as C  # noqa: E402

from constants import VERSION, SOURCES, CODE, BARCA, MATCHES  # noqa: E402

BASE = os.path.join(C.fc.DATA_DIR, 'altre_partite')
CACHE = os.path.join(C.fc.DATA_DIR, 'cache_mv')

_CODE_HASH = None


def code_hash():
    """Impronta dei due file che calcolano le caratteristiche (a capo Windows e Mac contano uguale)."""
    global _CODE_HASH
    if _CODE_HASH is None:
        h = hashlib.sha256()
        for name in ('caratteristiche.py', 'formato_comune.py', 'constants.py', 'dati.py'):
            with open(os.path.join(HERE, name), 'rb') as f:
                h.update(f.read().replace(b'\r\n', b'\n'))
        _CODE_HASH = h.hexdigest()[:8]
    return _CODE_HASH


def _feat_dir(legacy=False):
    return os.path.join(CACHE, '{}_{}_{}'.format(VERSION, code_hash(), 'legacy' if legacy else 'current'))


def external_matches():
    """[(fonte, id)] delle 30 partite esterne, sempre le stesse e nello stesso ordine."""
    return [(src, mid) for src in SOURCES for mid in MATCHES[src]]


def input_files(item):
    """I file letti per una partita, relativi a data/ (per controllare che tutti abbiano gli stessi)."""
    src, mid = item
    if item != BARCA and item not in external_matches():
        raise ValueError('Partita non riconosciuta: {!r}'.format(item))
    if src == 'barca':
        return ['tactical/BAR-ATH_tactical_atd.txt', 'tactical/BAR-ATH_tactical_atd.xml']
    return ['altre_partite/{}_sporcate/{}/{}_atd.txt'.format(src, mid, mid),
            'altre_partite/{}_sporcate/{}/{}_atd.xml'.format(src, mid, mid),
            'altre_partite/{}_pulite/{}/{}_possession_gt.csv'.format(src, mid, mid)]


@lru_cache(maxsize=256)
def _file_hash(path, size, modified_ns):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def input_hash(item):
    """The cache follows actual input contents, including the external label CSV."""
    h = hashlib.sha256()
    for rel in input_files(item):
        path = os.path.abspath(os.path.join(C.fc.DATA_DIR, rel))
        stat = os.stat(path)
        h.update(rel.encode('utf-8'))
        h.update(_file_hash(path, stat.st_size, stat.st_mtime_ns).encode('ascii'))
    return h.hexdigest()[:16]


def _path(src, mid, legacy=False):
    return os.path.join(_feat_dir(legacy), '{}__{}_{}.parquet'.format(src, mid, input_hash((src, mid))))


def table(item, legacy=False):
    """La tabella delle caratteristiche di una partita (frame, period, y, caratteristiche), senza cache."""
    src, mid = item
    input_files(item)  # reject arbitrary paths and unknown target matches
    if src == 'barca':
        m = C.fc.read_ours()
        labels = np.full(len(m['period']), 'X')       # la risposta della gara non si usa mai
    else:
        m = C.fc.read_epts(os.path.join(BASE, src + '_sporcate', mid),
                           labels_folder=os.path.join(BASE, src + '_pulite', mid))
        labels = m['labels']
    df = C.features(m, legacy=legacy)
    df.insert(0, 'y', np.array([CODE[c] for c in labels], dtype=np.int8))
    df.insert(0, 'period', m['period'].astype(np.int8))
    df.insert(0, 'frame', np.arange(len(df), dtype=np.int32))
    return df


def build_one(item, legacy=False):
    out = _path(*item, legacy=legacy)
    if os.path.exists(out):
        return out
    df = table(item, legacy=legacy)
    os.makedirs(_feat_dir(legacy), exist_ok=True)
    tmp = out + '.{}.tmp'.format(os.getpid())
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return out


def build_all(items=None, workers=4, legacy=False):
    """Prepara la cache delle partite che non ce l'hanno ancora."""
    items = external_matches() if items is None else list(items)
    if workers < 1:
        raise ValueError('workers deve essere positivo')
    todo = [it for it in items if not os.path.exists(_path(*it, legacy=legacy))]
    if not todo:
        return
    print('preparo le caratteristiche di {} partite in {}'.format(len(todo), _feat_dir(legacy)), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path in ex.map(build_one, todo, [legacy] * len(todo)):
            print('  pronta', os.path.basename(path), flush=True)


def load(src, mid, legacy=False):
    return pd.read_parquet(build_one((src, mid), legacy=legacy))


def load_all(items=None, legacy=False):
    """{(fonte, id): DataFrame} per le partite esterne."""
    items = external_matches() if items is None else items
    return {it: load(*it, legacy=legacy) for it in items}


if __name__ == '__main__':
    build_all()
