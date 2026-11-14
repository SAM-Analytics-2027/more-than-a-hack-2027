"""Verifica riproducibile delle cache augmentation delle sole partite esterne.

Non prepara cache, non allena modelli e non legge alcun materiale di Barça–Athletic.
Il report contiene solo esiti e conteggi aggregati, senza coordinate o sequenze GT.

    python modelli_mv/valida_augmentation.py
    python modelli_mv/valida_augmentation.py --output data/cache_mv/validazione.json
    python modelli_mv/valida_augmentation.py --ricomputa

``--ricomputa`` rilegge gli ATD sporcati e ricalcola esattamente le quattro viste:
è più costoso della verifica standard, che controlla cache, provenienza e GT.
"""

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

import augmentation as A
import constants as K
import dati


VIEWS = (*A.TRAIN_VIEWS, A.TEST_VIEW)
META = ('frame', 'period', 'y')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ground_truth(item, n_frames):
    """Le sole etichette pulite; consente CSV sparsi con X nei frame non presenti."""
    path = A._labels_input(item) / f'{item[1]}_possession_gt.csv'
    frame = pd.read_csv(path, dtype=str)
    _require(set(frame.columns) == {'frame', 'label'}, 'Header GT diverso da frame,label')
    numbers = pd.to_numeric(frame['frame'], errors='raise').to_numpy()
    _require(np.isfinite(numbers).all(), 'Frame GT non finiti')
    _require(np.equal(numbers, np.floor(numbers)).all(), 'Frame GT non interi')
    _require(((numbers >= 0) & (numbers < n_frames)).all(), 'Frame GT fuori intervallo')
    indices = numbers.astype(np.int64)
    _require(len(np.unique(indices)) == len(indices), 'Frame GT duplicati')
    labels = frame['label'].to_numpy()
    _require(np.isin(labels, tuple(K.CODE)).all(), 'GT con etichette non valide')
    y = np.full(n_frames, K.X, dtype=np.int8)
    y[indices] = np.array([K.CODE[label] for label in labels], dtype=np.int8)
    return y


def _missing_players(frame):
    return (frame['n_casa'] + frame['n_ospiti']).to_numpy() == 0


def validate_match(item, *, recompute=False):
    """Verifica una partita della whitelist; solleva ValueError al primo errore."""
    provenance = A.provenance(item)
    _require(provenance['tracking'] == 'sporcate', 'La provenienza non dichiara tracking sporcato')
    _require(provenance['labels'] == 'pulite', 'La provenienza non dichiara etichette pulite')
    folder = A.cache_path(item, VIEWS[0]).parent
    stored = json.loads((folder / 'provenienza.json').read_text(encoding='utf-8'))
    _require(stored == provenance, 'Manifest di provenienza diverso da input/codice correnti')
    expected_paths = {
        f'{item[0]}_sporcate/{item[1]}/{item[1]}_atd.txt',
        f'{item[0]}_sporcate/{item[1]}/{item[1]}_atd.xml',
        f'{item[0]}_pulite/{item[1]}/{item[1]}_possession_gt.csv',
    }
    _require(set(stored['inputs_sha256']) == expected_paths,
             'Il manifest non contiene esattamente ATD sporcati e GT pulita')

    base = A.load(item, VIEWS[0])
    _require(base.columns.is_unique, 'Colonne duplicate nella cache')
    _require(set((*META, 'n_casa', 'n_ospiti')).issubset(base.columns),
             'Colonne obbligatorie mancanti')
    _require(len(base) > 0, 'Cache vuota')
    _require(np.array_equal(base['frame'].to_numpy(), np.arange(len(base))),
             'Frame non consecutivi da zero')
    _require(np.isin(base['period'], (0, 1, 2)).all(), 'Periodi non validi')
    _require(np.array_equal(base['y'].to_numpy(), _ground_truth(item, len(base))),
             'Le etichette della cache differiscono dal CSV pulito')
    metadata = base[list(META)].to_numpy(copy=True)
    no_players = _missing_players(base)
    columns = list(base.columns)
    expected_dtypes = tuple(str(dtype) for dtype in base.dtypes)
    n_frames = len(base)
    del base

    match = None
    original = None
    if recompute:
        match = A.C.fc.read_epts(A._input(item), labels_folder=A._labels_input(item))
        original = {key: value.copy() for key, value in match.items()
                    if isinstance(value, np.ndarray)}

    views = []
    for view in VIEWS:
        frame = A.load(item, view)
        _require(len(frame) == n_frames, f'{view}: numero di frame differente')
        _require(list(frame.columns) == columns, f'{view}: colonne o ordine differenti')
        _require(tuple(str(dtype) for dtype in frame.dtypes) == expected_dtypes,
                 f'{view}: tipi delle colonne differenti')
        _require(np.array_equal(frame[list(META)].to_numpy(), metadata),
                 f'{view}: frame, periodi o etichette modificati')
        _require(all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes),
                 f'{view}: colonne non numeriche')
        _require(not np.isinf(frame.to_numpy()).any(), f'{view}: valori Inf nelle feature')
        _require(np.array_equal(_missing_players(frame), no_players),
                 f'{view}: nuovi tratti senza giocatori')
        _require(np.equal(frame['fonte_tv'].to_numpy(), float(item[0] == 'skillcorner')).all(),
                 f'{view}: fonte_tv incoerente con la fonte')
        if recompute:
            transformed = A.transform(match, item, view)
            recomputed = A.C.features(transformed)
            feature_columns = [col for col in columns if col not in (*META, 'fonte_tv')]
            _require(list(recomputed.columns) == feature_columns,
                     f'{view}: schema ricomputato differente')
            _require(np.array_equal(frame[feature_columns].to_numpy(),
                                    recomputed.to_numpy(), equal_nan=True),
                     f'{view}: feature ricomputate diverse dalla cache')
            for key in ('period', 'team', 'labels'):
                _require(np.array_equal(transformed[key], original[key]),
                         f'{view}: {key} modificato dalla trasformazione')
            for key, saved in original.items():
                actual = match[key]
                equal = (np.array_equal(actual, saved, equal_nan=True)
                         if np.issubdtype(actual.dtype, np.number)
                         else np.array_equal(actual, saved))
                _require(equal, f'{view}: array di input modificato: {key}')
            del recomputed, transformed
        views.append({'view': view, 'rows': len(frame), 'columns': len(frame.columns),
                      'ok': True, 'recomputed': recompute})
        del frame
    return {'source': item[0], 'match': item[1], 'cache': folder.name,
            'rows': n_frames, 'columns': len(columns), 'ok': True, 'views': views}


def validate_all(*, recompute=False):
    A._FILE_HASHES.clear()  # nessuna impronta memoizzata da una verifica precedente
    items = dati.external_matches()
    results = []
    for item in items:
        try:
            result = validate_match(item, recompute=recompute)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            result = {'source': item[0], 'match': item[1], 'ok': False,
                      'error': f'{type(exc).__name__}: {exc}'}
        results.append(result)
        status = 'OK' if result['ok'] else 'ERRORE'
        print(f'{status}: {item[0]}/{item[1]}', flush=True)
    good = [result for result in results if result['ok']]
    return {
        'schema': 'augmentation-validation-v1',
        'tracking': 'sporcate', 'labels': 'pulite', 'recomputed': recompute,
        'expected_matches': len(items), 'checked_matches': len(results),
        'passed_matches': len(good), 'failed_matches': len(results) - len(good),
        'checked_views': len(good) * len(VIEWS), 'views': list(VIEWS),
        'total_frames_per_view': sum(result['rows'] for result in good),
        'ok': len(good) == len(items), 'matches': results,
    }


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
        tmp = Path(handle.name)
    try:
        tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(dati.CACHE) / 'validazione_augmentation.json')
    parser.add_argument('--ricomputa', action='store_true',
                        help='Ricalcola le feature da ATD sporcati e confronta esattamente tutte le viste')
    args = parser.parse_args()
    report = validate_all(recompute=args.ricomputa)
    _write_report(args.output, report)
    print(f"Partite valide: {report['passed_matches']}/{report['expected_matches']}; "
          f"viste valide: {report['checked_views']}; report: {args.output}", flush=True)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
