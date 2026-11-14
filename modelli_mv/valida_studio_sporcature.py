"""Check every generated training view and independently recompute selected views.

No fitting, decoder selection or public ground truth. Held-out distribution
checks are descriptive, performed after the generator was fixed.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import augmentation_realiste as A
import constants as K
import dati
import studio_augmentation as S


@lru_cache(maxsize=1)
def _read_original_info(path, signature):
    """Cache only metadata of the current match, never its full feature table."""
    columns = pq.read_schema(path).names
    if 'fonte_tv' not in columns:
        columns.append('fonte_tv')
    return pd.read_parquet(path, columns=['frame', 'period', 'y']), columns


def _original_info(item):
    path = Path(dati._path(*item, legacy=True))
    if not path.is_file():
        raise ValueError(f'Cache originale mancante, nessuna rigenerazione automatica: {path}')
    stat = path.stat()
    return _read_original_info(str(path), (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def validate_diagnostics(record, frame, *, fps=K.FPS, expected=None):
    """Validate sidecar fields independently of its parquet hash.

    The sidecar's feature SHA does not protect its descriptive statistics. All
    fields are checked, observable counts/gaps are recalculated from features,
    and selected coordinate recomputations verify the complete dictionary.
    """
    scalar = ('ball_missing_fraction', 'ball_gaps_per_90min', 'visible_players_mean')
    count = ('active_frames', 'ball_error_observed_frames')
    quantiles = ('ball_gap_seconds_p25_p50_p75', 'ball_error_metres_p50_p90_p99')
    if not isinstance(record, dict) or set(record) != set((*scalar, *count, *quantiles)):
        raise ValueError('Schema delle diagnostiche non valido')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Frequenza delle diagnostiche non valida')
    for key in count:
        value = record[key]
        if type(value) is not int or value < 0 or value > len(frame):
            raise ValueError(f'Conteggio diagnostico non valido: {key}')
    for key in scalar:
        value = record[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'Diagnostica non finita o negativa: {key}')
    if record['ball_missing_fraction'] > 1 or record['ball_error_observed_frames'] > record['active_frames']:
        raise ValueError('Quote/conteggi delle diagnostiche incoerenti')
    for key in quantiles:
        values = np.asarray(record[key])
        if (values.shape != (3,) or not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values).all() or (values < 0).any() or (np.diff(values) < 0).any()):
            raise ValueError(f'Quantili diagnostici non validi: {key}')
    players = (frame['n_casa'] + frame['n_ospiti']).to_numpy()
    period = frame['period'].to_numpy()
    active = (period > 0) & (players > 0)
    seen = frame['palla_vista'].to_numpy() > 0
    gaps = [b - a for a, b in A.C._segments(np.where(active & ~seen, period, 0))]
    n = max(1, int(active.sum()))
    actual = {'active_frames': int(active.sum()),
              'ball_missing_fraction': float((active & ~seen).sum() / n),
              'ball_gaps_per_90min': float(len(gaps) * 90 * 60 * fps / n),
              'ball_gap_seconds_p25_p50_p75': (np.quantile(gaps, [.25, .5, .75]) / fps).tolist() if gaps else [0., 0., 0.],
              'visible_players_mean': float(players[active].astype(np.float64).mean()) if active.any() else 0.}
    for key, value in actual.items():
        if not np.allclose(record[key], value, rtol=0, atol=1e-8):
            raise ValueError(f'Diagnostica diversa dalle feature: {key}')
    if expected is not None and record != expected:
        raise ValueError('Diagnostiche diverse dalla ricomputazione delle coordinate')
    return True


def _validate_profile(profile, *, train, excluded=()):
    A.validate_profile(profile, train_items=train, excluded_items=excluded, verify_inputs=True)
    if profile['manifest']['code_sha256'] != A._code_hash():
        raise ValueError('Profilo prodotto da un altro codice del generatore')


def validate_view(item, profile, seed, replica, *, recompute=False):
    path = A.cache_path(item, profile, seed, replica, legacy=True)
    sidecar = path.with_suffix('.json')
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f'Vista di training mancante: {path}')
    record = S.read_json(sidecar)
    manifest = A.view_manifest(item, profile, seed, replica, legacy=True)
    if record['manifest'] != manifest or record['parquet_sha256'] != A._hash(path):
        raise ValueError(f'Vista o provenienza alterata: {path}')
    frame = pd.read_parquet(path)
    original, original_columns = _original_info(item)
    for name in ('frame', 'period', 'y'):
        if not np.array_equal(frame[name], original[name]):
            raise ValueError(f'Metadato {name} diverso: {path}')
    if list(frame.columns) != original_columns:
        raise ValueError(f'Colonne diverse: {path}')
    if np.isinf(frame.select_dtypes(include=[np.number]).to_numpy()).any():
        raise ValueError(f'Inf nella vista: {path}')
    fps = profile.get('sources', {}).get(item[0], profile.get('fallback', {})).get('fps', K.FPS)
    validate_diagnostics(record['diagnostics'], frame, fps=fps)
    if recompute:
        clean = A.C.fc.read_epts(A._folder(item, 'pulite'))
        transformed = A.transform(clean, item, profile, seed, replica)
        fresh = A.C.features(transformed, legacy=True)
        fresh.insert(0, 'y', np.asarray([K.CODE[y] for y in clean['labels']], dtype=np.int8))
        fresh.insert(0, 'period', clean['period'].astype(np.int8))
        fresh.insert(0, 'frame', np.arange(len(fresh), dtype=np.int32))
        fresh['fonte_tv'] = np.float32(item[0] == 'skillcorner')
        pd.testing.assert_frame_equal(frame, fresh, check_exact=True)
        validate_diagnostics(record['diagnostics'], frame, fps=fps,
                             expected=A.diagnostics(transformed, clean))
    return {'item': list(item), 'seed': seed, 'replica': replica, 'rows': len(frame),
            'path': str(path), 'sha256': record['parquet_sha256'], 'recomputed': recompute,
            'diagnostics': record['diagnostics']}


def validate_run(root, *, recompute=False, include_final=False):
    root = Path(root)
    config, plan = S.read_json(root / 'config.json'), S.read_json(root / 'piano.json')
    if config['smoke'] or config['sources'] != S.source_hashes():
        raise ValueError('Esperimento smoke o codice modificato')
    S.verify_complete(root, config, plan)
    records, profiles = [], []
    for replicate, seed in enumerate(config['seeds']):
        for key, fit in plan['fits'].items():
            folder = S.fit_path(root, 'risporcata', replicate, key)
            profile = S.read_json(folder / 'profilo.json')
            train = [tuple(it) for it in fit['train']]
            excluded = [tuple(it) for it in fit['test']]
            _validate_profile(profile, train=train, excluded=excluded)
            profiles.append({'key': key, 'replicate': replicate, 'profile_sha256': profile['profile_sha256']})
            representatives = {next(it for it in train if it[0] == source)
                               for source in {it[0] for it in train}}
            for item in train:
                for replica in range(config['replicas']):
                    redo = recompute and replicate == 0 and len(fit['excluded']) == 1 and replica == 0 and item in representatives
                    records.append(dict(fit=key, training_replicate=replicate,
                        **validate_view(item, profile, seed, replica, recompute=redo)))
            print(f'CHECK {key} r{replicate}: {len(train)*config["replicas"]} viste', flush=True)
    if include_final:
        target = Path(S.R.model_dir(S.model_name(root, 'risporcata', 0)))
        profile = S.read_json(target / 'profilo.json')
        items = [tuple(it) for it in plan['items']]
        _validate_profile(profile, train=items)
        profiles.append({'key': 'finale', 'replicate': 0, 'profile_sha256': profile['profile_sha256']})
        for item in items:
            for replica in range(config['replicas']):
                records.append(dict(fit='finale', training_replicate=0,
                    **validate_view(item, profile, config['seeds'][0], replica)))
    return {'ok': True, 'profiles_checked': len(profiles), 'views_checked': len(records),
            'unique_profiles_checked': len({profile['profile_sha256'] for profile in profiles}),
            'rows_checked': sum(r['rows'] for r in records),
            'views_recomputed_exactly': sum(r['recomputed'] for r in records),
            'recompute_policy': 'first train match per source, each outer fold, seed0 replica0',
            'profiles': profiles, 'views': records,
            'interpretation': 'Technical checks; source errors are synthetic, not independent camera ground truth.'}


def held_out_diagnostics(root):
    root = Path(root)
    config, plan = S.read_json(root / 'config.json'), S.read_json(root / 'piano.json')
    rows = []
    for outer in plan['outer']:
        profile = S.read_json(S.fit_path(root, 'risporcata', 0, outer['refit']) / 'profilo.json')
        _validate_profile(profile, train=[tuple(it) for it in plan['fits'][outer['refit']]['train']],
                          excluded=[tuple(it) for it in outer['test']])
        for item in map(tuple, outer['test']):
            clean = A.C.fc.read_epts(A._folder(item, 'pulite'))
            dirty = A.C.fc.read_epts(A._folder(item, 'sporcate'))
            replicas = [A.diagnostics(A.transform(clean, item, profile, config['seeds'][0], i), clean)
                        for i in range(config['replicas'])]
            rows.append({'item': list(item), 'fold': outer['fold'],
                         'reference': A.diagnostics(dirty, clean), 'generated': replicas})
            print(f'DIAGNOSTICA held-out {item}', flush=True)
    return {'kind': 'held-out descriptive distribution diagnostic; never tuning or selection', 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--esperimento', required=True)
    parser.add_argument('--ricomputa', action='store_true')
    parser.add_argument('--finale', action='store_true')
    parser.add_argument('--diagnostica', action='store_true')
    args = parser.parse_args()
    if not S.P.valid_name(args.esperimento):
        parser.error('Nome non valido')
    root = Path(dati.CACHE) / 'augmentation_runs' / args.esperimento
    result = validate_run(root, recompute=args.ricomputa, include_final=args.finale)
    if args.diagnostica:
        result['held_out'] = held_out_diagnostics(root)
    S.P._write(root / 'validazione_sporcature.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('views','profiles','held_out')}, indent=2))


if __name__ == '__main__':
    main()
