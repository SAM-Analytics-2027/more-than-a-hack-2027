"""The 30 external matches (and Barça–Athletic) turned into feature tables, saved in data/cache/.

The tracking always comes from the DEGRADED version, the labels from the CLEAN one:
    data/external/<source>_degraded/<id>/<id>_atd.txt and _atd.xml     (the model input)
    data/external/<source>_clean/<id>/<id>_possession_gt.csv           (the right answer, for training only)
The matches are a fixed list (MATCHES): extra folders inside data/external/ change nothing.

The cache lives inside data/, so GitHub does not see it. Its name contains a fingerprint of the code that computes
the features: if that code changes, the cache is rebuilt automatically and an old cache is never used.

    python models/dataset.py       builds the cache (about 2 minutes, in parallel)
"""
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as C  # noqa: E402

from constants import VERSION, SOURCES, CODE, BARCA, MATCHES  # noqa: E402

BASE = os.path.join(C.fc.DATA_DIR, 'external')
if not os.path.isdir(BASE) and os.path.isdir(os.path.join(C.fc.DATA_DIR, 'training')):
    BASE = os.path.join(C.fc.DATA_DIR, 'training')
CACHE = os.path.join(C.fc.DATA_DIR, 'cache')

_CODE_HASH = None


def code_hash():
    """Fingerprint of the two files that compute the features (Windows and Mac line endings count the same)."""
    global _CODE_HASH
    if _CODE_HASH is None:
        h = hashlib.sha256()
        for name in ('features.py', 'epts_reader.py'):
            with open(os.path.join(HERE, name), 'rb') as f:
                h.update(f.read().replace(b'\r\n', b'\n'))
        _CODE_HASH = h.hexdigest()[:8]
    return _CODE_HASH


def _feat_dir():
    return os.path.join(CACHE, '{}_{}'.format(VERSION, code_hash()))


def external_matches():
    """[(source, id)] of the 30 external matches, always the same ones and in the same order."""
    return [(src, mid) for src in SOURCES for mid in MATCHES[src]]


def input_files(item):
    """The files read for one match, relative to data/ (to check that everyone has the same ones)."""
    src, mid = item
    if src == 'barca':
        return ['tactical/BAR-ATH_tactical_atd.txt', 'tactical/BAR-ATH_tactical_atd.xml']
    return ['external/{}_degraded/{}/{}_atd.txt'.format(src, mid, mid),
            'external/{}_degraded/{}/{}_atd.xml'.format(src, mid, mid),
            'external/{}_clean/{}/{}_possession_gt.csv'.format(src, mid, mid)]


def _path(src, mid):
    return os.path.join(_feat_dir(), '{}__{}.parquet'.format(src, mid))


def table(item):
    """The feature table of one match (frame, period, y, features), without the cache."""
    src, mid = item
    if src == 'barca':
        m = C.fc.read_ours()
        labels = np.full(len(m['period']), 'X')       # the competition answer is never used
    else:
        m = C.fc.read_epts(os.path.join(BASE, src + '_degraded', mid),
                           labels_folder=os.path.join(BASE, src + '_clean', mid))
        labels = m['labels']
    df = C.features(m)
    df.insert(0, 'y', np.array([CODE[c] for c in labels], dtype=np.int8))
    df.insert(0, 'period', m['period'].astype(np.int8))
    df.insert(0, 'frame', np.arange(len(df), dtype=np.int32))
    return df


def build_one(item):
    out = _path(*item)
    if os.path.exists(out):
        return out
    df = table(item)
    os.makedirs(_feat_dir(), exist_ok=True)
    df.to_parquet(out + '.tmp', index=False)
    os.replace(out + '.tmp', out)
    return out


def _has_raw(item):
    src, mid = item
    if src == 'barca':
        return os.path.exists(os.path.join(C.fc.DATA_DIR, 'tactical', 'BAR-ATH_tactical_atd.txt'))
    return os.path.exists(os.path.join(BASE, src + '_degraded', mid, '{}_atd.txt'.format(mid)))


def build_all(items=None, workers=4):
    """Builds the cache for the matches that do not have it yet."""
    items = items or external_matches() + [BARCA]
    todo = [it for it in items if not os.path.exists(_path(*it)) and _has_raw(it)]
    if not todo:
        return
    print('computing the features of {} matches in {}'.format(len(todo), _feat_dir()), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path in ex.map(build_one, todo):
            print('  ready', os.path.basename(path), flush=True)


def load(src, mid):
    return pd.read_parquet(_path(src, mid))


def load_all(items=None):
    """{(source, id): DataFrame} for the external matches."""
    items = items or external_matches()
    avail = [it for it in items if os.path.exists(_path(*it)) or _has_raw(it)]
    return {it: load(*it) for it in avail}


if __name__ == '__main__':
    build_all()
