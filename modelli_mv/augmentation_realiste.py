"""Nuove sporcature da tracking pulito esterno, con parametri stimati nel train.

Il riferimento appreso e' la sporcatura sintetica preesistente, NON una misura
indipendente dell'errore di una telecamera. Non vengono importati i suoi parametri
calibrati sulla partita di gara. Nessuna etichetta entra nel fit o nelle trasformazioni.
Le mancanze gia' presenti nelle sorgenti pulite (Metrica/SkillCorner incluse) restano.
"""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import zlib

import numpy as np
import pandas as pd
from numba import njit

import caratteristiche as C
import constants as K
import dati

SCHEMA = 'clean-external-corruption-v1'
AGE_SECONDS = np.array([.2, .5, 1., 2., 4., 8., 16., 32., 64.])
BALL_CONTEXTS = 24
PLAYER_CONTEXTS = 40  # geometry/motion x whether the corrupted ball is missing
SNIPPET_SECONDS = 2.0
SNIPPETS_PER_MATCH = 96


def _item(item):
    item = tuple(item)
    if len(item) != 2 or item[0] not in K.MATCHES or item[1] not in K.MATCHES[item[0]]:
        raise ValueError(f'Solo partite esterne della whitelist: {item!r}')
    return item


def _folder(item, kind):
    source, mid = _item(item)
    return Path(dati.BASE) / f'{source}_{kind}' / mid


def _files(item, labels=False):
    _, mid = _item(item)
    paths = [folder / f'{mid}_atd.{ext}' for folder in
             (_folder(item, 'pulite'), _folder(item, 'sporcate')) for ext in ('txt', 'xml')]
    if labels:
        paths.append(_folder(item, 'pulite') / f'{mid}_possession_gt.csv')
    return paths


@lru_cache(maxsize=512)
def _hash_file(path, signature):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _hash(path):
    path = Path(path).resolve()
    s = path.stat()
    return _hash_file(str(path), (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino))


def _code_hash():
    h = hashlib.sha256()
    for path in (__file__, C.__file__, C.fc.__file__, K.__file__):
        h.update(Path(path).name.encode())
        h.update(Path(path).read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def input_manifest(item, labels=False):
    return {str(p.relative_to(dati.BASE)).replace('\\', '/'): _hash(p)
            for p in _files(item, labels)}


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
        tmp = Path(handle.name)
    try:
        tmp.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _spans(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return zip(edges[::2], edges[1::2])


def _active(match):
    return (match['period'] > 0) & (np.isfinite(match['px']) & np.isfinite(match['py'])).any(axis=1)


def _validate_pair(clean, dirty):
    if (not np.array_equal(clean['period'], dirty['period']) or
            not np.isclose(clean['fps'], dirty['fps']) or
            not np.isclose(clean['length'], dirty['length']) or
            not np.isclose(clean['width'], dirty['width'])):
        raise ValueError('Coppia pulita/sporcata non allineata per frame, tempi, fps o campo')
    if clean['fps'] <= 0 or not np.isfinite(clean['fps']):
        raise ValueError('fps non validi')


def _geometry(match):
    """Contesti da sole coordinate pulite: moto, zona e affollamento."""
    px, py, period = match['px'], match['py'], match['period']
    speed = np.hypot(C._speed(px, period), C._speed(py, period))
    valid_speed = np.isfinite(speed)
    mean_speed = np.divide(np.nansum(speed, axis=1), valid_speed.sum(axis=1),
                           out=np.zeros(len(period), float), where=valid_speed.sum(axis=1) > 0)
    bx, by = match['bx'], match['by']
    outside = (np.abs(bx) > match['length'] / 2) | (np.abs(by) > match['width'] / 2)
    box = (np.abs(bx) > match['length'] / 2 - 16.5) & (np.abs(by) < 20.16)
    sideline = np.abs(by) > match['width'] / 2 - 5
    zone = np.where(outside, 0, np.where(box, 1, np.where(sideline, 2, 3)))
    ball = 4 * np.searchsorted([.5, 1., 1.5, 2., 3.], mean_speed) + zone
    nearest = np.full(px.shape, np.inf, dtype=np.float32)
    for a in range(0, len(px), 1024):
        b = min(a + 1024, len(px))
        distance = np.hypot(px[a:b, :, None] - px[a:b, None, :],
                            py[a:b, :, None] - py[a:b, None, :])
        distance[:, np.arange(px.shape[1]), np.arange(px.shape[1])] = np.inf
        distance[~np.isfinite(distance)] = np.inf
        nearest[a:b] = distance.min(axis=2)
    player = 4 * np.searchsorted([1., 2., 4., 8.], nearest) + np.searchsorted(
        [.5, 1.5, 3.], np.nan_to_num(speed, nan=0.))
    return ball.astype(np.int16), player.astype(np.int16)


def _player_presence(clean, dirty):
    """Riallinea slot anonimi tramite coordinate e squadra, senza identita'/etichette.

    Il generatore esistente non altera coordinate dei giocatori: la tolleranza
    di 1 cm copre solo la quantizzazione dei file. Non si assumono slot uguali.
    """
    present = np.zeros(clean['px'].shape, bool)
    for team in (0, 1):
        ci = np.flatnonzero(clean['team'] == team)
        di = np.flatnonzero(dirty['team'] == team)
        if not len(ci) or not len(di):
            continue
        for a in range(0, len(present), 1024):
            b = min(a + 1024, len(present))
            distance = np.hypot(clean['px'][a:b, ci, None] - dirty['px'][a:b, None, di],
                                clean['py'][a:b, ci, None] - dirty['py'][a:b, None, di])
            present[a:b, ci] = (distance < .01).any(axis=2)
    return present


@njit(cache=True)
def _counts(state, eligible, context, period, age_edges, ncontexts):
    n, slots = state.shape
    counts = np.zeros((2, ncontexts, len(age_edges) + 1, 2), np.int64)
    initial = np.zeros((ncontexts, 2), np.int64)
    for j in range(slots):
        age = 0
        for i in range(n):
            if not eligible[i, j]:
                age = 0
                continue
            s, c = int(state[i, j]), int(context[i, j])
            if i == 0 or not eligible[i - 1, j] or period[i] != period[i - 1]:
                initial[c, s] += 1
                age = 1
                continue
            previous = int(state[i - 1, j])
            k = np.searchsorted(age_edges, age, side='right')
            counts[previous, c, k, int(s != previous)] += 1
            age = age + 1 if s == previous else 1
    return counts, initial


def _lagged_ball(match, lag):
    out = np.c_[match['bx'], match['by']].copy()
    if lag == 0:
        return out
    valid = np.isfinite(out).all(axis=1) & _active(match)
    for a, b in C._segments(np.where(valid, match['period'], 0)):
        if b - a > lag:
            out[a + lag:b] = out[a:b - lag].copy()
    return out


def _error_bank(clean, dirty, rng):
    valid = _active(clean) & np.isfinite(dirty['bx']) & np.isfinite(dirty['by'])
    observed = np.c_[dirty['bx'], dirty['by']]
    candidates = []
    for lag in range(int(round(.20 * clean['fps'])) + 1):
        residual = observed - _lagged_ball(clean, lag)
        use = valid & np.isfinite(residual).all(axis=1)
        score = float(np.mean(np.square(residual[use]))) if use.any() else np.inf
        candidates.append(score)
    lag = int(np.argmin(candidates))
    residual = observed - _lagged_ball(clean, lag)
    valid &= np.isfinite(residual).all(axis=1)
    width = max(2, int(round(SNIPPET_SECONDS * clean['fps'])))
    starts = []
    for a, b in C._segments(np.where(valid, clean['period'], 0)):
        if b - a >= width:
            starts.extend(range(a, b - width + 1, max(1, width // 2)))
    if starts:
        selected = rng.choice(starts, min(SNIPPETS_PER_MATCH, len(starts)), replace=False)
        bank = np.stack([residual[a:a + width] for a in selected])
    elif valid.any():
        # Degenerate short records retain measured marginal error, never invented scale.
        samples = residual[valid]
        selected = rng.choice(len(samples), min(SNIPPETS_PER_MATCH, len(samples)), replace=False)
        bank = np.repeat(samples[selected, None, :], width, axis=1)
    else:
        bank = np.zeros((1, width, 2), np.float32)
    return lag, np.round(bank, 5).tolist()


def summarize_pair(clean, dirty, seed=2027):
    """Statistica sufficiente di UNA coppia: nessun accesso a labels."""
    _validate_pair(clean, dirty)
    active = _active(clean)
    ball_context, player_context = _geometry(clean)
    ball_eligible = active & np.isfinite(clean['bx']) & np.isfinite(clean['by'])
    ball_missing = ~(np.isfinite(dirty['bx']) & np.isfinite(dirty['by']))
    player_context = player_context + 20 * ball_missing[:, None].astype(np.int16)
    player_eligible = active[:, None] & np.isfinite(clean['px']) & np.isfinite(clean['py'])
    player_missing = ~_player_presence(clean, dirty)
    edges = AGE_SECONDS * clean['fps']
    bc, bi = _counts(ball_missing[:, None], ball_eligible[:, None], ball_context[:, None],
                     clean['period'], edges, BALL_CONTEXTS)
    pc, pi = _counts(player_missing, player_eligible, player_context,
                     clean['period'], edges, PLAYER_CONTEXTS)
    lag, bank = _error_bank(clean, dirty, np.random.default_rng(seed))
    return {'fps': float(clean['fps']), 'ball_counts': bc.tolist(), 'ball_initial': bi.tolist(),
            'player_counts': pc.tolist(), 'player_initial': pi.tolist(),
            'ball_lag': lag, 'ball_error_bank': bank,
            'player_slots_per_team': [int((dirty['team'] == team).sum()) for team in (0, 1)],
            'anonymous_slots': bool(clean['px'].shape[1] != dirty['px'].shape[1]),
            'clean_diagnostics': diagnostics(clean), 'dirty_diagnostics': diagnostics(dirty)}


def pair_summary(item):
    item = _item(item)
    manifest = {'schema': SCHEMA, 'code_sha256': _code_hash(), 'inputs': input_manifest(item)}
    path = Path(dati.CACHE) / 'augmentation_realiste' / 'pairs' / f'{item[0]}__{item[1]}_{_digest(manifest)}.json'
    if path.is_file():
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['manifest'] != manifest or data['summary_sha256'] != _digest(data['summary']):
            raise ValueError(f'Cache coppia corrotta: {path}')
        return data['summary']
    clean = C.fc.read_epts(_folder(item, 'pulite'))
    dirty = C.fc.read_epts(_folder(item, 'sporcate'))
    summary = summarize_pair(clean, dirty, seed=zlib.crc32('/'.join(item).encode()))
    _atomic_json(path, {'manifest': manifest, 'summary': summary, 'summary_sha256': _digest(summary)})
    return summary


def _pool(summaries):
    result = {'fps': summaries[0]['fps'], 'ball_lags': [s['ball_lag'] for s in summaries],
              'ball_error_bank': [row for s in summaries for row in s['ball_error_bank']],
              'player_slots_per_team': np.max([s['player_slots_per_team'] for s in summaries], axis=0).tolist(),
              'anonymous_slots': any(s['anonymous_slots'] for s in summaries)}
    if any(s['fps'] != result['fps'] for s in summaries):
        raise ValueError('Ricampionare esplicitamente sorgenti con fps differenti')
    for name in ('ball', 'player'):
        counts = sum(np.asarray(s[name + '_counts'], np.float64) for s in summaries)
        total = counts.sum(axis=-1)
        global_counts = counts.sum(axis=1)
        state_counts = global_counts.sum(axis=1)
        state_prior = np.divide(state_counts[:, 1], state_counts.sum(axis=1),
                                out=np.zeros(2), where=state_counts.sum(axis=1) > 0)
        # A simulation can reach ages absent from finite training records. Zero
        # at an unseen age would create an absorbing state (a half-long gap).
        # Back off to the observed state exit rate, not an invented probability.
        prior = np.divide(global_counts[..., 1], global_counts.sum(axis=-1),
                          out=np.broadcast_to(state_prior[:, None], global_counts.shape[:2]).copy(),
                          where=global_counts.sum(axis=-1) > 0)
        # Four seconds of source-level backoff stabilize sparse geometry/age cells.
        prior_weight = 4 * result['fps']
        rate = (counts[..., 1] + prior_weight * prior[:, None, :]) / (total + prior_weight)
        initial = sum(np.asarray(s[name + '_initial'], np.float64) for s in summaries)
        denom = initial.sum(axis=1)
        p = initial[:, 1].sum() / max(1., initial.sum())
        result[name + '_hazard'] = rate.tolist()
        result[name + '_initial_missing'] = ((initial[:, 1] + p) / (denom + 1)).tolist()
    return result


def fit_profile(train_items):
    """Fit/cache esclusivamente sulle coppie passate; nessun insieme globale implicito."""
    items = sorted({_item(it) for it in train_items})
    if not items:
        raise ValueError('Il profilo richiede almeno una partita esterna di training')
    manifest = {'schema': SCHEMA, 'code_sha256': _code_hash(), 'train_items': [list(it) for it in items],
                'inputs': {'/'.join(it): input_manifest(it) for it in items}}
    path = Path(dati.CACHE) / 'augmentation_realiste' / 'profiles' / f'{_digest(manifest)}.json'
    if path.is_file():
        profile = json.loads(path.read_text(encoding='utf-8'))
        if profile['manifest'] != manifest or profile['profile_sha256'] != _digest(
                {k: v for k, v in profile.items() if k != 'profile_sha256'}):
            raise ValueError(f'Profilo corrotto: {path}')
        validate_profile(profile, train_items=items)
        return profile
    summaries = [(it, pair_summary(it)) for it in items]
    profile = {'manifest': manifest, 'schema': SCHEMA,
               'reference': 'preexisting synthetic dirty external tracking; not camera ground truth',
               'sources': {source: _pool([s for it, s in summaries if it[0] == source])
                           for source in sorted({it[0] for it in items})},
               'fallback': _pool([s for _, s in summaries])}
    profile['profile_sha256'] = _digest(profile)
    validate_profile(profile, train_items=items)
    _atomic_json(path, profile)
    return profile


def validate_profile(profile, train_items=None, excluded_items=(), verify_inputs=False):
    """Fail closed su provenienza, split, valori e integrita' del generatore.

    ``excluded_items`` rende esplicito il controllo anti-leakage del chiamante.
    ``verify_inputs`` rilegge le impronte dei soli input elencati nel training.
    """
    if profile.get('schema') != SCHEMA or profile.get('manifest', {}).get('schema') != SCHEMA:
        raise ValueError('Schema del profilo non riconosciuto')
    if profile.get('profile_sha256') != _digest({k: v for k, v in profile.items() if k != 'profile_sha256'}):
        raise ValueError('Impronta del profilo non valida')
    manifest = profile['manifest']
    items = [_item(item) for item in manifest['train_items']]
    if not items or len(set(items)) != len(items):
        raise ValueError('Training del profilo vuoto o con duplicati')
    if train_items is not None and set(items) != {_item(item) for item in train_items}:
        raise ValueError('Il training del profilo non coincide con quello richiesto')
    if set(items) & {_item(item) for item in excluded_items}:
        raise ValueError('Leakage: una partita esclusa compare nel profilo')
    if set(manifest['inputs']) != {'/'.join(item) for item in items}:
        raise ValueError('Manifest input non corrispondente alle partite di training')
    if set(profile['sources']) != {item[0] for item in items}:
        raise ValueError('Fonti del profilo diverse da quelle di training')
    if verify_inputs:
        for item in items:
            if manifest['inputs']['/'.join(item)] != input_manifest(item):
                raise ValueError(f'Input del profilo modificato: {item}')
    for params in [profile['fallback'], *profile['sources'].values()]:
        fps = params['fps']
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Frequenza del profilo non valida')
        for name, contexts in (('ball', BALL_CONTEXTS), ('player', PLAYER_CONTEXTS)):
            hazard = np.asarray(params[name + '_hazard'])
            initial = np.asarray(params[name + '_initial_missing'])
            if hazard.shape != (2, contexts, len(AGE_SECONDS) + 1) or initial.shape != (contexts,):
                raise ValueError('Forma delle probabilita del profilo non valida')
            if any(not np.isfinite(x).all() or (x < 0).any() or (x > 1).any()
                   for x in (hazard, initial)):
                raise ValueError('Probabilita del profilo non valida')
        bank = np.asarray(params['ball_error_bank'])
        if bank.ndim != 3 or bank.shape[0] < 1 or bank.shape[1] < 2 or bank.shape[2] != 2 or not np.isfinite(bank).all():
            raise ValueError('Traiettorie di errore non valide')
        lags = np.asarray(params['ball_lags'])
        if not lags.size or not np.isfinite(lags).all() or (lags < 0).any() or (lags != lags.astype(int)).any():
            raise ValueError('Ritardi del profilo non validi')
        slots = np.asarray(params['player_slots_per_team'])
        if slots.shape != (2,) or not np.isfinite(slots).all() or (slots < 1).any() or (slots != slots.astype(int)).any():
            raise ValueError('Capacita degli slot non valida')
    return True


@njit(cache=True)
def _simulate(eligible, context, period, age_edges, hazard, initial, uniforms):
    n, slots = eligible.shape
    missing = np.zeros((n, slots), np.bool_)
    for j in range(slots):
        state, age = 0, 0
        for i in range(n):
            if not eligible[i, j]:
                state, age = 0, 0
                continue
            c = int(context[i, j])
            if i == 0 or not eligible[i - 1, j] or period[i] != period[i - 1]:
                state = int(uniforms[i, j] < initial[c])
                age = 1
            else:
                k = np.searchsorted(age_edges, age, side='right')
                if uniforms[i, j] < hazard[state, c, k]:
                    state, age = 1 - state, 1
                else:
                    age += 1
            missing[i, j] = state == 1
    return missing


def _sample_error(bank, n, rng):
    """Bootstrap di traiettorie d'errore, con giunzioni continue e scala preservata."""
    bank = np.asarray(bank, np.float32)
    width = bank.shape[1]
    hop = max(1, width // 2)
    output = np.zeros((n + width, 2), np.float64)
    norm = np.zeros(n + width, np.float64)
    weight = np.sin(np.pi * (np.arange(width) + .5) / width)
    for a in range(-hop, n, hop):
        snippet = bank[int(rng.integers(len(bank)))].copy()
        # Reflection of both axes avoids memorizing one source match's directional bias.
        snippet *= rng.choice([-1., 1.], size=2)
        lo, hi = max(0, a), min(n, a + width)
        local = slice(lo - a, hi - a)
        output[lo:hi] += weight[local, None] * snippet[local]
        norm[lo:hi] += weight[local] ** 2
    return (output[:n] / np.sqrt(np.maximum(norm[:n, None], 1e-12))).astype(np.float32)


def _anonymize_players(px, py, team, period, capacities, rng):
    """Ogni segmento visibile va in uno slot libero della stessa squadra.

    Le coordinate e il numero di giocatori per frame non cambiano. La capacita'
    viene dalle sole coppie di training; si amplia soltanto se necessario per
    rappresentare il numero di giocatori effettivamente osservati nella sorgente.
    """
    seen = np.isfinite(px) & np.isfinite(py)
    capacities = np.asarray(capacities, int).copy()
    for t in (0, 1):
        maximum = int(seen[:, team == t].sum(axis=1).max(initial=0))
        capacities[t] = max(capacities[t], 2 * maximum)
    output_team = np.repeat(np.arange(2, dtype=np.int8), capacities)
    ox = np.full((len(period), int(capacities.sum())), np.nan, np.float32)
    oy = np.full_like(ox, np.nan)
    offsets = [0, capacities[0]]
    free = [np.zeros(capacities[t], np.int64) for t in (0, 1)]
    segments = []
    for j in range(px.shape[1]):
        # Even period=0 observations are transported without changing their
        # coordinates; segmentation at each period boundary avoids slot bridges.
        segment_period = np.where(seen[:, j], period.astype(np.int16) + 1, 0)
        for a, b in C._segments(segment_period):
            segments.append((a, b, j))
    for a, b, j in sorted(segments):
        t = int(team[j])
        available = np.flatnonzero(free[t] <= a)
        if not len(available):
            raise ValueError('Capacita slot insufficiente per i segmenti osservati')
        slot = int(rng.choice(available))
        free[t][slot] = b + 1  # one empty frame before another trajectory
        dest = offsets[t] + slot
        ox[a:b, dest], oy[a:b, dest] = px[a:b, j], py[a:b, j]
    return ox, oy, output_team


def transform(clean, item, profile, seed=2027, replica=0):
    """Sporca una copia delle coordinate pulite; labels/frame/period/squadre invariati."""
    item = _item(item)
    if seed < 0 or replica < 0:
        raise ValueError('Semi e indici replica devono essere non negativi')
    validate_profile(profile)
    params = profile['sources'].get(item[0], profile['fallback'])
    if not np.isclose(clean['fps'], params['fps']):
        raise ValueError('Frequenza del profilo diversa dalla partita')
    rng = np.random.default_rng([int(seed), int(replica), zlib.crc32('/'.join(item).encode())])
    active = _active(clean)
    bc, pc = _geometry(clean)
    be = active & np.isfinite(clean['bx']) & np.isfinite(clean['by'])
    pe = active[:, None] & np.isfinite(clean['px']) & np.isfinite(clean['py'])
    edges = AGE_SECONDS * clean['fps']
    bm = _simulate(be[:, None], bc[:, None], clean['period'], edges,
                   np.asarray(params['ball_hazard']), np.asarray(params['ball_initial_missing']),
                   rng.random((len(active), 1)))[:, 0]
    pc = pc + 20 * (bm | ~np.isfinite(clean['bx']) | ~np.isfinite(clean['by']))[:, None].astype(np.int16)
    pm = _simulate(pe, pc, clean['period'], edges,
                   np.asarray(params['player_hazard']), np.asarray(params['player_initial_missing']),
                   rng.random(pe.shape))
    # Never fabricate a replay by deleting the last observed player.
    empty = active & ~(pe & ~pm).any(axis=1)
    rows = np.flatnonzero(empty)
    if len(rows):
        pm[rows, np.argmax(pe[rows], axis=1)] = False
    bx, by = clean['bx'].copy(), clean['by'].copy()
    lag = int(rng.choice(params['ball_lags']))
    shifted = _lagged_ball(clean, lag)
    error_bank = np.asarray(params['ball_error_bank'], np.float32)
    for a, b in C._segments(np.where(be, clean['period'], 0)):
        error = _sample_error(error_bank, b - a, rng)
        bx[a:b] = shifted[a:b, 0] + error[:, 0]
        by[a:b] = shifted[a:b, 1] + error[:, 1]
    bx[bm] = by[bm] = np.nan
    px, py = clean['px'].copy(), clean['py'].copy()
    px[pm] = py[pm] = np.nan
    team = clean['team']
    if params['anonymous_slots']:
        px, py, team = _anonymize_players(px, py, team, clean['period'], params['player_slots_per_team'], rng)
    return dict(clean, bx=bx, by=by, px=px, py=py, team=team)


def diagnostics(match, clean=None):
    """Descrizione senza etichette; adatta a held-out DOPO aver fissato il generatore."""
    active = _active(match)
    n = max(1, int(active.sum()))
    seen = np.isfinite(match['bx']) & np.isfinite(match['by'])
    gaps = [b - a for a, b in C._segments(np.where(active & ~seen, match['period'], 0))]
    players = (np.isfinite(match['px']) & np.isfinite(match['py'])).sum(axis=1)
    out = {'active_frames': int(active.sum()), 'ball_missing_fraction': float((active & ~seen).sum() / n),
           'ball_gaps_per_90min': float(len(gaps) * 90 * 60 * match['fps'] / n),
           'ball_gap_seconds_p25_p50_p75': (np.quantile(gaps, [.25, .5, .75]) / match['fps']).tolist() if gaps else [0., 0., 0.],
           'visible_players_mean': float(players[active].mean()) if active.any() else 0.}
    if clean is not None:
        _validate_pair(clean, match)
        distance = np.hypot(match['bx'] - clean['bx'], match['by'] - clean['by'])
        use = active & np.isfinite(distance)
        out['ball_error_metres_p50_p90_p99'] = np.quantile(distance[use], [.5, .9, .99]).tolist() if use.any() else [0., 0., 0.]
        out['ball_error_observed_frames'] = int(use.sum())
    return out


def view_manifest(item, profile, seed=2027, replica=0, legacy=False):
    _item(item)
    return {'schema': SCHEMA, 'code_sha256': _code_hash(), 'item': list(item),
            'profile_sha256': _digest(profile), 'seed': int(seed), 'replica': int(replica),
            'legacy': bool(legacy), 'inputs': input_manifest(item, labels=True),
            'numpy': np.__version__, 'pandas': pd.__version__}


def cache_path(item, profile, seed=2027, replica=0, legacy=False):
    manifest = view_manifest(item, profile, seed, replica, legacy)
    return Path(dati.CACHE) / 'augmentation_realiste' / 'views' / f'{item[0]}__{item[1]}_{_digest(manifest)}.parquet'


def load_view(item, profile, seed=2027, replica=0, legacy=False):
    """Feature lazy/cache di una replica pulita->sporcata, mai input pulito al modello."""
    manifest = view_manifest(item, profile, seed, replica, legacy)
    path = Path(dati.CACHE) / 'augmentation_realiste' / 'views' / f'{item[0]}__{item[1]}_{_digest(manifest)}.parquet'
    meta_path = path.with_suffix('.json')
    if path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if metadata['manifest'] != manifest or metadata['parquet_sha256'] != _hash(path):
            raise ValueError(f'Cache feature corrotta: {path}')
        return pd.read_parquet(path)
    clean = C.fc.read_epts(_folder(item, 'pulite'))
    if clean['labels'] is None or len(clean['labels']) != len(clean['period']):
        raise ValueError('Etichette esterne mancanti o non allineate')
    corrupted = transform(clean, item, profile, seed, replica)
    frame = C.features(corrupted, legacy=legacy)
    frame.insert(0, 'y', np.array([K.CODE[label] for label in clean['labels']], dtype=np.int8))
    frame.insert(0, 'period', clean['period'].astype(np.int8))
    frame.insert(0, 'frame', np.arange(len(frame), dtype=np.int32))
    frame['fonte_tv'] = np.float32(item[0] == 'skillcorner')
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
        tmp = Path(handle.name)
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    _atomic_json(meta_path, {'manifest': manifest, 'parquet_sha256': _hash(path),
                             'diagnostics': diagnostics(corrupted, clean)})
    return frame


def _prepare_one(args):
    item, profile, seeds, replicas, legacy = args
    for seed in seeds:
        for replica in range(replicas):
            load_view(item, profile, seed, replica, legacy)
    return item


def prepare_views(items, profile, seeds=(2027,), replicas=3, legacy=False, workers=1):
    if replicas < 1 or workers < 1 or not seeds:
        raise ValueError('Servono semi, almeno una replica e almeno un worker')
    tasks = [(tuple(item), profile, tuple(seeds), replicas, legacy) for item in items]
    if workers == 1:
        for task in tasks:
            _prepare_one(task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_prepare_one, tasks))
