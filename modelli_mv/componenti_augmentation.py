"""Controlled train/predict primitives for the clean-source augmentation study.

No dataset is loaded implicitly and no decoder is calibrated here. Every caller
supplies whole-match training and (optionally) validation partitions. The legacy
network reproduces the normalization, sampling and crop recipe in HEAD before
the augmentation work; feature, context, normalization, segmentation and
inference changes are independently selectable. Existing frozen artifacts and
the ordinary ``rete``/``riproduci`` entry points are left untouched.

``views(item)`` (or ``views[item]``) returns additional *corrupted* feature tables,
excluding the original dirty table in ``data[item]``. Replicas may have different
observable gaps. Sequence boundaries are therefore computed separately for each
view, while the number of optimizer updates is determined by the original data.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

import caratteristiche as C
import constants as K
import dati
import esperimento as E
import rete


@dataclass(frozen=True)
class Config:
    feature_version: int = 1
    halo: bool = False
    normalization: str = 'legacy'
    segment_version: int = 1
    inference_version: int = 1

    def __post_init__(self):
        if self.feature_version not in (1, 2) or self.segment_version not in (1, 2):
            raise ValueError('feature_version e segment_version devono essere 1 o 2')
        if self.inference_version not in (1, 2) or self.normalization not in ('legacy', 'symmetric'):
            raise ValueError('inferenza o normalizzazione non riconosciuta')


PRESETS = {
    'base': Config(),
    'feature': Config(feature_version=2),
    'contesto': Config(halo=True),
    'normalizzazione': Config(normalization='symmetric'),
    'segmentazione': Config(segment_version=2),
    'inferenza': Config(inference_version=2),
    'moderno': Config(2, True, 'symmetric', 2, 2),
}


def _table(data, item):
    return data(item) if callable(data) else data[item]


def _external(items):
    items = [tuple(it) for it in items]
    if not items or len(set(items)) != len(items):
        raise ValueError('partite di training vuote o duplicate')
    if not set(items).issubset(set(dati.external_matches())):
        raise ValueError('training e calibrazione ammessi soltanto su partite esterne note')
    return items


def _replicas(views, item, base, cols):
    frames = [] if views is None else list(_table(views, item))
    for frame in frames:
        if len(frame) != len(base) or list(frame[cols].columns) != list(cols):
            raise ValueError('dimensioni o colonne della replica incoerenti')
        for key in ('frame', 'period', 'y'):
            if key in base and (key not in frame or not np.array_equal(base[key], frame[key])):
                raise ValueError(f'la replica modifica {key}')
    return frames


def _probabilities(n_replicas, original_probability):
    if not 0 <= original_probability <= 1:
        raise ValueError('probabilità originale fuori da [0, 1]')
    return np.r_[original_probability, np.full(n_replicas, (1 - original_probability) / n_replicas)] if n_replicas else np.ones(1)


def _item_rng(seed, item):
    # Stable across ordering and processes; Python hash() is intentionally avoided.
    digest = hashlib.sha256('/'.join(item).encode('utf-8')).digest()
    return np.random.default_rng([int(seed), int.from_bytes(digest[:4], 'little'), 117])


def tree_rows(data, items, cols, *, views=None, original_probability=.5, seed=2027):
    """Same baseline row indices/budget as HEAD, with independent replica draws."""
    xs, ys, ws = [], [], []
    for item in items:
        frame = _table(data, item)
        valid = frame['y'].isin((0, 1, 2)) & frame['period'].isin((1, 2))
        sel = np.flatnonzero(valid.to_numpy())[::K.STEP]
        if not len(sel):
            raise ValueError(f'nessuna riga valida: {item}')
        x = frame.iloc[sel][cols].copy()
        replicas = _replicas(views, item, frame, cols)
        if replicas:
            choice = _item_rng(seed, item).choice(len(replicas) + 1, len(sel), p=_probabilities(len(replicas), original_probability))
            for view, table in enumerate(replicas, 1):
                take = np.flatnonzero(choice == view)
                x.iloc[take] = table.iloc[sel[take]][cols].to_numpy()
        y = frame['y'].to_numpy()[sel].astype(np.int8)
        w = E.short_spell_weights(frame['y'].to_numpy(), frame['period'].to_numpy())[sel]
        w = w * K.SOURCE_WEIGHTS[item[0]]
        xs.extend((x, C.swap_teams(x)[cols]))
        ys.extend((y, np.where(y < 2, 1 - y, y)))
        ws.extend((w, w))
    if not xs:
        raise ValueError('nessuna partita di training')
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


def _normalizer(samples, kind):
    if kind == 'symmetric':
        return rete.normalizer(samples)
    if not samples or not any(len(x) for x in samples):
        raise ValueError('nessuna riga per normalizzare')
    all_x = np.concatenate(samples)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(all_x, axis=0)
        iqr = np.nanpercentile(all_x, 75, axis=0) - np.nanpercentile(all_x, 25, axis=0)
    return np.where(np.isfinite(med), med, 0.), np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, 1.)


def _normalise(x, med, iqr, kind):
    x = (x - med) / iqr
    miss = np.isnan(x) if kind == 'legacy' else ~np.isfinite(x)
    if kind != 'legacy':
        x[miss] = np.nan
    x = np.clip(np.nan_to_num(x, nan=0.), *K.NET_FEATURE_CLIP)
    return np.concatenate((x, miss.astype(np.float32)), axis=1).astype(np.float32)


def prepare_network(data, items, cols, config=Config(), *, views=None, original_probability=.5):
    """Prepare independent per-view sequences; normalizer fits ORIGINAL train only.

    Keeping the original normalizer makes the augmentation intervention explicit:
    it changes training inputs, not the scaler. Modern normalization is its own
    ablation. Data/replica loaders are consumed one match at a time.
    """
    samples, pools, replica_count, reference_length = [], [], None, 0
    for item in items:
        frame = _table(data, item)
        # The historical optimizer budget depends only on whole-half lengths.
        # Modern segmentation must not silently add/remove gradient updates.
        for half in (1, 2):
            count = int((frame['period'].to_numpy() == half).sum())
            if count >= rete.CROP * rete.STRIDE:
                reference_length += 2 * ((count + rete.STRIDE - 1) // rete.STRIDE)
        if config.normalization == 'legacy':
            samples.append(frame[cols].to_numpy()[::K.NORM_SUBSAMPLE_STEP])
        else:
            sub = frame.loc[(frame['period'] > 0) & frame['y'].between(0, 2), cols].iloc[::K.NORM_SUBSAMPLE_STEP]
            samples.extend((sub.to_numpy(), C.swap_teams(sub)[cols].to_numpy()))
        replicas = _replicas(views, item, frame, cols)
        if replica_count is None:
            replica_count = len(replicas)
            pools = [[] for _ in range(replica_count + 1)]
        if len(replicas) != replica_count:
            raise ValueError('numero di repliche diverso fra partite')
        for view, table in enumerate([frame, *replicas]):
            raw = rete.raw_sequences(table, cols, K.SOURCE_WEIGHTS[item[0]], legacy=config.segment_version == 1)
            # HEAD ordering: all original halves, then all swapped halves.
            for side in (0, 1):
                for xo, xs, y, ys, w, sel in raw:
                    pools[view].append((xo if side == 0 else xs, y if side == 0 else ys, w, sel))
    if not pools or any(not pool for pool in pools):
        raise ValueError('nessuna sequenza valida in una vista di training')
    med, iqr = _normalizer(samples, config.normalization)
    lengths = [np.asarray([len(q[0]) for q in pool], dtype=float) for pool in pools]
    return dict(med=med, iqr=iqr, pools=pools,
                probabilities=[lens / lens.sum() for lens in lengths],
                view_probability=_probabilities(replica_count, original_probability),
                steps=max(1, reference_length // (rete.CROP * K.BATCH_SIZE)),
                config=asdict(config), cols=list(cols))


def network_batch(prepared, rng, view_rng, config, *, batch_size=None):
    """Produce a batch without network state, useful for controlled regression checks."""
    batch_size = K.BATCH_SIZE if batch_size is None else batch_size
    pools, probabilities = prepared['pools'], prepared['probabilities']
    if len(pools) == 1:
        views = np.zeros(batch_size, dtype=int)
        ids = rng.choice(len(pools[0]), size=batch_size, p=probabilities[0])
    else:
        views = view_rng.choice(len(pools), size=batch_size, p=prepared['view_probability'])
        # One uniform per sample keeps timeline sampling independent of the
        # replica RNG. Each replica has its own observable sequence boundaries.
        u = rng.random(batch_size)
        ids = np.asarray([np.searchsorted(np.cumsum(probabilities[v]), q, side='right') for v, q in zip(views, u)])
    xb, yb, wb = [], [], []
    for view, si in zip(views, ids):
        x, y, w, _ = pools[view][int(si)]
        start = int(rng.integers(0, max(1, len(x) - rete.CROP + 1)))
        context = sum(K.DILATIONS) if config.halo else 0
        lo, hi = max(0, start - context), min(len(x), start + rete.CROP + context)
        # Normalize only the real samples. Padding is all zero, exactly as in
        # the modern ordinary rete.train path; it never receives loss.
        actual = _normalise(x[lo:hi], prepared['med'], prepared['iqr'], config.normalization)
        size = rete.CROP + 2 * context
        xc = np.zeros((size, 2 * x.shape[1]), dtype=np.float32)
        offset = lo - (start - context)
        xc[offset:offset + len(actual)] = actual
        yc = np.full(size, K.X, dtype=np.int64)
        wc = np.zeros(size, dtype=np.float32)
        n = min(rete.CROP, len(y) - start)
        yc[context:context + n], wc[context:context + n] = y[start:start + n], w[start:start + n]
        xb.append(xc)
        yb.append(yc)
        wb.append(wc)
    return xb, yb, wb


def train_network(prepared, cols, seed, config=Config(), *, device=None, epochs=20, steps=None, progress=None):
    if prepared['config'] != asdict(config) or prepared['cols'] != list(cols):
        raise ValueError('preparazione e configurazione della rete diverse')
    view_rng = np.random.default_rng([int(seed), 881])
    def batch(rng):
        return network_batch(prepared, rng, view_rng, config)
    result = rete.fit_batches(cols, seed, device or rete.device(), prepared['med'], prepared['iqr'],
                              prepared['steps'] if steps is None else steps, batch,
                              raw=False, epochs=epochs, progress=progress)
    result['inference_version'] = config.inference_version
    result['ablation_config'] = asdict(config)
    return result


def fit_tree(data, items, cols, *, iterations=None, val_data=None, val_items=None,
             views=None, original_probability=.5, seed=2027, max_iter=500):
    """No random-row early stopping: fixed rounds OR disjoint external matches."""
    items = _external(items)
    if iterations is not None and (val_data is not None or val_items is not None):
        raise ValueError('scegliere iterazioni fisse oppure validation a partite')
    if iterations is None:
        if val_data is None or val_items is None:
            raise ValueError('specificare iterazioni fisse o validation a partite disgiunte')
        val_items = _external(val_items)
        if set(items) & set(val_items):
            raise ValueError('partite di training e validation sovrapposte')
    elif int(iterations) != iterations or iterations < 1:
        raise ValueError('iterazioni HGB non valide')
    x, y, w = tree_rows(data, items, cols, views=views, original_probability=original_probability, seed=seed)
    model = E.make_model().set_params(random_state=int(seed), validation_fraction=None)
    if iterations is not None:
        model.set_params(early_stopping=False, max_iter=int(iterations))
        model.fit(x, y, sample_weight=w)
    else:
        xv, yv, wv = tree_rows(val_data, val_items, cols)
        model.set_params(early_stopping=True, max_iter=int(max_iter))
        model.fit(x, y, sample_weight=w, X_val=xv, y_val=yv, sample_weight_val=wv)
    return model


def _write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_manifest(folder):
    """Fail closed on incomplete writes or altered model files."""
    folder = Path(folder)
    manifest = folder / 'manifest.json'
    if not manifest.is_file():
        raise ValueError(f'manifest dei componenti mancante: {folder}')
    expected = json.loads(manifest.read_text(encoding='utf-8'))
    if 'modello.json' not in expected or 'hgb.joblib' not in expected:
        raise ValueError('manifest dei componenti incompleto')
    for name, digest in expected.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or _sha256(path) != digest:
            raise ValueError(f'componente mancante o modificato: {name}')
    info = json.loads((folder / 'modello.json').read_text(encoding='utf-8'))
    required = {'modello.json', 'hgb.joblib', *(f'rete_seme{i}.pt' for i in range(len(info['semi_rete'])))}
    if set(expected) != required:
        raise ValueError('manifest non corrispondente alle reti dichiarate')
    return expected


def fit_components(data, items, cols, folder, net_seeds, config=Config(), *,
                   hgb_iterations=None, val_data=None, val_items=None, tree_seed=2027,
                   epochs=20, steps=None, device=None, views=None,
                   original_probability=.5, progress=None):
    """Train and atomically save a fresh ensemble, compatible with riproduci loaders.

    Resuming and source fingerprints belong to the experiment driver. Refuse to
    replace completed model metadata; a new recipe must use a new folder.
    """
    items = _external(items)
    seeds = [int(seed) for seed in net_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('specificare semi distinti per almeno una rete')
    folder = Path(folder)
    if (folder / 'modello.json').exists():
        raise FileExistsError(f'modello già completato: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    model = fit_tree(data, items, cols, iterations=hgb_iterations, val_data=val_data,
                     val_items=val_items, views=views, original_probability=original_probability, seed=tree_seed)
    tmp = folder / 'hgb.tmp'
    joblib.dump(model, tmp, compress=3)
    tmp.replace(folder / 'hgb.joblib')
    prepared = prepare_network(data, items, cols, config, views=views, original_probability=original_probability)
    for index, seed in enumerate(seeds):
        net = train_network(prepared, cols, seed, config, device=device, epochs=epochs, steps=steps, progress=progress)
        tmp = folder / f'rete_seme{index}.tmp'
        rete.save(net, tmp)
        tmp.replace(folder / f'rete_seme{index}.pt')
        del net
    info = dict(partite=['/'.join(it) for it in items], semi_rete=seeds, colonne=list(cols),
                feature_version=config.feature_version, ablation_config=asdict(config),
                dispositivo_allenamento=str(device or rete.device()),
                hgb_iterazioni=int(model.n_iter_), hgb_seme=int(tree_seed),
                hgb_validation=[] if val_items is None else ['/'.join(it) for it in val_items],
                epoche=int(epochs), passi_per_epoca=prepared['steps'] if steps is None else int(steps),
                viste=len(prepared['pools']), probabilita_viste=prepared['view_probability'].tolist(),
                normalizzazione_su='sole_sporcate_originali_di_training')
    _write_json(folder / 'modello.json', info)
    names = ['modello.json', 'hgb.joblib', *(f'rete_seme{i}.pt' for i in range(len(seeds)))]
    _write_json(folder / 'manifest.json', {name: _sha256(folder / name) for name in names})
    return load_components(folder)


def load_components(folder):
    folder = Path(folder)
    verify_model_manifest(folder)
    info = json.loads((folder / 'modello.json').read_text(encoding='utf-8'))
    model = joblib.load(folder / 'hgb.joblib')
    nets = [rete.load(folder / f'rete_seme{i}.pt') for i in range(len(info['semi_rete']))]
    return model, nets, info['colonne']


def predict_components(components, frame, item, *, inference_version=None):
    """Unrounded probabilities; inference-only ablation can reuse fitted weights."""
    model, nets, cols = components
    parts = [E.predict_sym(model, frame, cols)]
    for net in nets:
        if inference_version is not None:
            if inference_version not in (1, 2):
                raise ValueError('versione inferenza non valida')
            net = dict(net, inference_version=inference_version)
        parts.append(rete.predict(net, frame, item))
    p = np.mean([part.astype(np.float64) for part in parts], axis=0)
    if p.shape != (len(frame), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1., atol=1e-5):
        raise ValueError('probabilità ensemble non valide')
    return p
