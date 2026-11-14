"""Viste riproducibili del tracking *sporcato* esterno, prima delle feature.

``originale`` indica la sporcata fornita, senza ulteriori perturbazioni: il tracking
pulito non entra mai nel modello. Dalle cartelle pulite si legge soltanto il CSV
delle etichette. Ogni perturbazione usa esclusivamente le coordinate della propria
partita, senza ground truth o statistiche apprese dalle altre partite/fold.
"""

import argparse
import hashlib
import json
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import gaussian_filter1d

import caratteristiche as C
import constants as K
import dati

TRAIN_VIEWS = ('originale', 'occlusioni', 'posizione')
TEST_VIEW = 'stress'
VIEW_PROB = (0.50, 0.25, 0.25)
CACHE_SCHEMA = 'dirty-tracking-v2'
_FILE_HASHES = {}


def rng_for(item, view, seed=K.SEED):
    return np.random.default_rng([seed, zlib.crc32('/'.join(item).encode()), zlib.crc32(view.encode())])


def _spans(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def _occlude(ball_x, ball_y, player_x, player_y, rng, fps, length, width):
    """Buchi contigui più frequenti vicino a linee e giocatori, senza usare H/A/D."""
    seen = np.isfinite(ball_x) & np.isfinite(ball_y)
    if not seen.any():
        return
    gaps = [b - a for a, b in _spans(~seen) if a and b < len(seen)]
    typical = np.clip(np.median(gaps) if gaps else fps / 2, fps / 5, 2 * fps)
    distance = np.hypot(player_x - ball_x[:, None], player_y - ball_y[:, None])
    nearby = (distance < 2).sum(axis=1)
    edge = (np.abs(ball_x) > length / 2 - 5) | (np.abs(ball_y) > width / 2 - 5)
    weights = np.where(seen, 1.0 + nearby + 2.0 * edge, 0)
    count = min(int(seen.sum()), rng.poisson(seen.sum() / (8 * fps)))
    starts = rng.choice(len(seen), count, replace=False, p=weights / weights.sum())
    durations = np.clip(rng.lognormal(np.log(typical), 0.7, count), 2, 4 * fps).astype(int)
    for a, duration in zip(starts, durations):
        ball_x[a:a + duration] = ball_y[a:a + duration] = np.nan

    # Piccoli dropout di singole tracce. Il portiere e i giocatori lontani restano spesso visibili.
    for a, duration in zip(starts[::2], durations[::2]):
        slots = np.flatnonzero(np.isfinite(player_x[a]) & np.isfinite(player_y[a]))
        if len(slots) < 3:
            continue
        weights = 1 / (3 + distance[a, slots])
        slot = rng.choice(slots, p=weights / weights.sum())
        b = min(len(player_x), a + int(min(duration, 2 * fps)))
        # Una traccia visibile al momento iniziale può essere l'unica superstite
        # pochi frame dopo. Non creare nuovi replay/tratti senza dati: tutte le
        # viste devono mantenere gli stessi segmenti temporali del campione.
        visible = np.isfinite(player_x[a:b]) & np.isfinite(player_y[a:b])
        rows = np.flatnonzero(visible[:, slot] & (visible.sum(axis=1) > 1)) + a
        player_x[rows, slot] = player_y[rows, slot] = np.nan


def _shift(x, y, rng, fps, multiplier):
    """Errore di localizzazione liscio, in metri, con scala ricavata dal tracking esterno."""
    acceleration = np.hypot(np.diff(x, n=2), np.diff(y, n=2))
    usable = acceleration[np.isfinite(acceleration)]
    if not len(usable):
        return
    # La curvatura per frame misura sia movimento sia rumore. Il limite evita di simulare salti di più metri.
    scale = multiplier * np.clip(4 * np.quantile(usable, .80), .12, .50)
    for axis in (x, y):
        drift = gaussian_filter1d(rng.standard_normal(len(axis)), sigma=fps / 4, mode='reflect')
        axis += scale * drift / max(drift.std(), 1e-6)


def transform(match, item, view):
    """Mantiene frame, periodi ed etichette. Opera su copie e non attraversa i tratti senza giocatori."""
    if view not in (*TRAIN_VIEWS, TEST_VIEW):
        raise ValueError(view)
    if view == 'originale':
        return match
    if not np.isfinite(match['fps']) or match['fps'] <= 0:
        raise ValueError('La frequenza del tracking deve essere positiva e finita')
    rng = rng_for(item, view)
    bx, by, px, py = (match[key].copy() for key in ('bx', 'by', 'px', 'py'))
    observed = (np.isfinite(px) & np.isfinite(py)).any(axis=1)
    for half in (1, 2):
        for a, b in _spans(observed & (match['period'] == half)):
            if b - a < match['fps']:
                continue
            x, y = bx[a:b], by[a:b]
            if view in ('posizione', TEST_VIEW):
                _shift(x, y, rng, match['fps'], 1.35 if view == TEST_VIEW else 1.0)
            if view in ('occlusioni', TEST_VIEW):
                _occlude(x, y, px[a:b], py[a:b], rng, match['fps'], match['length'], match['width'])
    return dict(match, bx=bx, by=by, px=px, py=py)


def _input(item):
    src, mid = item
    if src not in K.MATCHES or mid not in K.MATCHES[src]:
        raise ValueError(f'Partita esterna non riconosciuta: {item}')
    return Path(dati.BASE) / (src + '_sporcate') / mid


def _labels_input(item):
    _input(item)  # stessa whitelist delle partite esterne
    return Path(dati.BASE) / (item[0] + '_pulite') / item[1]


def _input_files(item):
    folder = _input(item)
    mid = item[1]
    return (folder / f'{mid}_atd.txt', folder / f'{mid}_atd.xml',
            _labels_input(item) / f'{mid}_possession_gt.csv')


def _file_hash(path):
    """SHA-256 completo, memoizzato solo mentre identità/stat del file restano uguali.

    Il contenuto, non la data del download, identifica gli input. La memoizzazione
    evita di rileggere gigabyte a ogni accesso a una singola vista. Un nuovo
    processo ricalcola sempre l'impronta; prepare la aggiorna esplicitamente.
    """
    path = Path(path).resolve()
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                 stat.st_dev, stat.st_ino)
    cached = _FILE_HASHES.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    digest = h.hexdigest()
    _FILE_HASHES[path] = (signature, digest)
    return digest


def provenance(item):
    """Provenienza riproducibile; nessuna coordinata o etichetta esportata."""
    code = hashlib.sha256()
    for module in (__file__, K.__file__, C.__file__, C.fc.__file__, dati.__file__):
        code.update(Path(module).name.encode())
        code.update(Path(module).read_bytes().replace(b'\r\n', b'\n'))
    files = _input_files(item)
    return {
        'schema': CACHE_SCHEMA, 'tracking': 'sporcate', 'labels': 'pulite',
        'item': list(item), 'code_sha256': code.hexdigest(),
        'inputs_sha256': {str(p.relative_to(dati.BASE)).replace('\\', '/'): _file_hash(p)
                          for p in files},
        'numpy': np.__version__, 'scipy': scipy.__version__,
    }


def _cache_key(item):
    raw = json.dumps(provenance(item), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_path(item, view):
    if view not in (*TRAIN_VIEWS, TEST_VIEW):
        raise ValueError(f'Vista non riconosciuta: {view}')
    return Path(dati.CACHE) / 'augmentation' / f'{item[0]}__{item[1]}_{_cache_key(item)}' / f'{view}.parquet'


def _build(item):
    views = [view for view in (*TRAIN_VIEWS, TEST_VIEW) if not cache_path(item, view).is_file()]
    if not views:
        return item
    match = C.fc.read_epts(_input(item), labels_folder=_labels_input(item))
    if match['labels'] is None:
        raise ValueError(f'Etichette mancanti: {item}')
    if len(match['labels']) != len(match['period']):
        raise ValueError(f'Etichette non allineate ai frame: {item}')
    if not np.isin(match['labels'], tuple(K.CODE)).all():
        raise ValueError(f'Etichette non valide: {item}')
    labels = np.array([K.CODE[c] for c in match['labels']], dtype=np.int8)
    for view in views:
        frame = C.features(transform(match, item, view))
        frame.insert(0, 'y', labels)
        frame.insert(0, 'period', match['period'])
        frame.insert(0, 'frame', np.arange(len(frame), dtype=np.int32))
        frame['fonte_tv'] = np.float32(item[0] == 'skillcorner')
        path = cache_path(item, view)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
            tmp = Path(handle.name)
        try:
            frame.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    path = cache_path(item, 'originale').parent / 'provenienza.json'
    with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
        tmp = Path(handle.name)
    try:
        tmp.write_text(json.dumps(provenance(item), indent=2), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return item


def prepare(items, workers):
    if workers < 1:
        raise ValueError('workers deve essere positivo')
    # Rileggere le impronte a inizio preparazione anche se un file fosse stato
    # sostituito preservando esplicitamente timestamp e dimensione.
    _FILE_HASHES.clear()
    if workers == 1:
        for item in items:
            print('Feature pronte', *_build(item), flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for item in executor.map(_build, items):
            print('Feature pronte', *item, flush=True)


def load(item, view='originale'):
    return pd.read_parquet(cache_path(item, view))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    prepare(dati.external_matches(), args.workers)
