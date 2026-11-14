"""Protocollo annidato e controlli del confronto clean -> dirty.

I cinque fold sono quelli storici di ``esperimento.folds``. Per ogni fold
esterno k, le probabilità di calibrazione di j sono prodotte escludendo k e j
dal training. Le dieci coppie non ordinate sono riutilizzabili in entrambe le
direzioni, senza addestrare venti volte gli stessi insiemi. Il modello che
prevede k viene poi allenato su tutte le partite fuori da k.

Questo modulo non legge tracking, etichette o risultati pubblici. Le sintesi
sono di sviluppo; le repliche dello stesso incontro non sono nuovi incontri.
"""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

SCHEMA = 'augmentation-nested-v1'
SOURCES = ('dfl', 'metrica', 'skillcorner')
TACTICAL = ('dfl', 'metrica')
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _item(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('Ogni partita richiede fonte e identificativo')
    source, name = value
    if source not in SOURCES or not isinstance(name, str) or not name or '/' in name or '\\' in name:
        raise ValueError('Partita esterna non valida')
    return source, name


def _items(values):
    result = [_item(it) for it in values]
    if len(result) != len(set(result)):
        raise ValueError('Partite duplicate')
    return sorted(result)


def _lists(values):
    return [list(it) for it in sorted(values)]


def fit_key(excluded):
    values = sorted(excluded)
    if (not values or len(values) != len(set(values))
            or any(type(v) is not int or v < 0 for v in values)):
        raise ValueError('Fold esclusi non validi')
    return 'excl_' + '_'.join(str(v) for v in values)


def build_plan(items, fold_of=None):
    """Crea un piano serializzabile JSON, con 15 fit per cinque fold.

    ``fold_of`` può avere chiavi tuple oppure stringhe ``fonte/id``. Se omesso
    usa la medesima assegnazione del riferimento congelato, senza rimescolare.
    """
    items = _items(items)
    if fold_of is None:
        import esperimento
        fold_of = esperimento.folds(items)
    mapping = {}
    for key, value in fold_of.items():
        item = _item(key.split('/') if isinstance(key, str) else key)
        if item in mapping or type(value) is not int or value < 0:
            raise ValueError('Assegnazione fold non valida')
        mapping[item] = value
    folds = sorted(set(mapping.values()))
    if set(mapping) != set(items) or len(folds) < 3 or folds != list(range(len(folds))):
        raise ValueError('Il piano richiede copertura esatta e almeno tre fold contigui')
    fits = {}
    for size in (1, 2):
        for excluded in itertools.combinations(folds, size):
            train = [it for it in items if mapping[it] not in excluded]
            test = [it for it in items if mapping[it] in excluded]
            if not train or not test:
                raise ValueError('Fit con training o test vuoto')
            fits[fit_key(excluded)] = {'excluded': list(excluded),
                                       'train': _lists(train), 'test': _lists(test)}
    outer = []
    for k in folds:
        calibration = [it for it in items if mapping[it] != k and it[0] in TACTICAL]
        if not calibration:
            raise ValueError('Calibrazione DFL/Metrica vuota')
        outer.append({'fold': k, 'refit': fit_key([k]),
                      'inner_sources': {fit_key([k, j]): _lists(it for it in items if mapping[it] == j)
                                        for j in folds if j != k},
                      'calibration': _lists(calibration),
                      'test': _lists(it for it in items if mapping[it] == k)})
    return {'schema': SCHEMA, 'items': _lists(items),
            'fold_of': {'/'.join(it): mapping[it] for it in items},
            'fits': fits, 'outer': outer}


def validate_plan(plan):
    """Rifiuta piani alterati, inclusa la sostituzione con OOF globali."""
    expected = build_plan(plan['items'], plan['fold_of'])
    if plan != expected:
        raise ValueError('Piano annidato incoerente o modificato')
    return True


def outer_plan(plan, fold):
    validate_plan(plan)
    found = [entry for entry in plan['outer'] if entry['fold'] == fold]
    if len(found) != 1:
        raise ValueError('Fold esterno non riconosciuto')
    return found[0]


def validate_fit_provenance(plan, key, metadata, *, generator_required=False):
    """Verifica gli insiemi effettivamente letti per fitting e normalizzazione.

    La provenienza del generatore è vuota per la baseline e coincide con il
    training del fit aumentato. ``early_stopping`` deve essere False: il budget
    HGB è fissato prima del confronto. ``test`` è facoltativo ma, se presente,
    deve coincidere con le partite escluse previste dal piano.
    """
    validate_plan(plan)
    if key not in plan['fits']:
        raise ValueError('Fit non riconosciuto')
    spec = plan['fits'][key]
    expected = _items(spec['train'])
    for field in ('train', 'normalization_train', 'label_train'):
        if field not in metadata or _items(metadata[field]) != expected:
            raise ValueError(f'Provenienza {field} diversa dal training previsto')
    generated = expected if generator_required else []
    if 'generator_train' not in metadata or _items(metadata['generator_train']) != generated:
        raise ValueError('Il generatore usa partite non autorizzate dal fit')
    if metadata.get('early_stopping') is not False:
        raise ValueError('Occorre un budget HGB fisso, senza early stopping')
    if 'test' in metadata and _items(metadata['test']) != _items(spec['test']):
        raise ValueError('Test del fit diverso dal piano')
    return True


def select_inner_predictions(plan, fold, predictions_by_fit):
    """Estrae le OOF DFL/Metrica che escludono il test esterno dal modello.

    Input: ``{fit_key: {(fonte, id): array_probabilita}}``. La struttura dei
    file del fit va verificata con ``validate_fit_provenance`` prima del load.
    """
    spec = outer_plan(plan, fold)
    expected = set(_items(spec['calibration']))
    selected = {}
    for key, ids in spec['inner_sources'].items():
        needed = [it for it in _items(ids) if it in expected]
        if not needed:
            continue
        if key not in predictions_by_fit:
            raise ValueError(f'Predizioni interne mancanti: {key}')
        source = predictions_by_fit[key]
        for item in needed:
            if item not in source or item in selected:
                raise ValueError('Copertura delle probabilità interne non valida')
            value = np.asarray(source[item])
            if (value.ndim != 2 or value.shape[1] != 3 or not len(value)
                    or not np.isfinite(value).all() or (value < 0).any()
                    or (value > 1).any()
                    or not np.allclose(value.sum(axis=1), 1, rtol=0, atol=5e-4)):
                raise ValueError('Probabilità interne non valide')
            selected[item] = value
    if set(selected) != expected:
        raise ValueError('Calibrazione incompleta')
    return selected


def validate_completion(plan, completed, recipes, replicates, *, smoke=False):
    """Il finale richiede tutti i fit per ogni ricetta e replica dichiarata.

    ``completed`` contiene tuple ``(recipe, replicate, fit_key)``; la presenza
    non sostituisce la verifica dei manifest di ciascun fit.
    """
    validate_plan(plan)
    if smoke:
        raise ValueError('Uno smoke test non può essere promosso a modello finale')
    if not recipes or not replicates or len(set(recipes)) != len(recipes) or len(set(replicates)) != len(replicates):
        raise ValueError('Ricette o repliche mancanti/duplicate')
    expected = set(itertools.product(recipes, replicates, plan['fits']))
    actual = [tuple(value) for value in completed]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Confronto incompleto, duplicato o con fit inattesi')
    return True


def validate_score_report(plan, report):
    """Rifiuta report troncati anche quando tutte le ricette perdono gli stessi match."""
    validate_plan(plan)
    expected = set(_items(plan['items']))
    covered = []
    tactical = []
    if not isinstance(report.get('mode'), str) or not report['mode']:
        raise ValueError('Ricetta del report mancante')
    if type(report.get('replicate')) is not int or report['replicate'] < 0:
        raise ValueError('Replica del report non valida')
    for row in report['rows']:
        item = _item((row['source'], row['match']))
        if (item not in expected or row.get('fold') != plan['fold_of']['/'.join(item)]
                or row.get('mode') != report['mode'] or row.get('replicate') != report['replicate']):
            raise ValueError('Partita, fold, ricetta o replica incoerenti nel report')
        metrics = np.asarray([row['spellF1'], row['macroF1']], float)
        if not np.isfinite(metrics).all() or (metrics < 0).any() or (metrics > 1).any():
            raise ValueError('Metriche del report non valide')
        covered.append(item)
        if item[0] in TACTICAL:
            tactical.append(metrics)
    if len(covered) != len(set(covered)) or set(covered) != expected:
        raise ValueError('Copertura partite del report incompleta o duplicata')
    if set(report['decoders']) != {str(entry['fold']) for entry in plan['outer']}:
        raise ValueError('Copertura decoder incompleta')
    actual = np.mean(tactical, axis=0)
    reported = np.asarray([report['mean10']['spellF1'], report['mean10']['macroF1']], float)
    if not np.isfinite(reported).all() or not np.allclose(actual, reported, rtol=0, atol=1e-12):
        raise ValueError('La media del report non coincide con le partite DFL/Metrica')
    return True


def paired_summary(rows, baseline, *, bootstrap_seed=2027, n_boot=2000, ensemble_rows=None):
    """Confronto appaiato, unità di aggregazione = partita DFL/Metrica.

    Ogni riga richiede recipe, replicate (int), source, match, spellF1, macroF1.
    La CI bootstrap ricampiona partite, mantenendo insieme le loro repliche;
    è descrittiva e condizionata ai semi osservati, non una conferma indipendente.
    Le eventuali ``ensemble_rows`` hanno le medesime colonne tranne replicate:
    sono punteggi calcolati dopo l'ensemble delle probabilità e decodifica.
    """
    if type(n_boot) is not int or n_boot < 0:
        raise ValueError('Numero bootstrap non valido')
    values = {}
    for row in rows:
        item = _item((row['source'], row['match']))
        if item[0] not in TACTICAL:
            continue
        replica = row['replicate']
        recipe = row['recipe']
        if type(replica) is not int or replica < 0 or not isinstance(recipe, str) or not recipe:
            raise ValueError('Ricetta o replica non valida')
        key = recipe, replica, item
        metrics = np.asarray([row['spellF1'], row['macroF1']], float)
        if key in values or not np.isfinite(metrics).all() or (metrics < 0).any() or (metrics > 1).any():
            raise ValueError('Riga di risultati duplicata o non valida')
        values[key] = metrics
    recipes = sorted({key[0] for key in values})
    if baseline not in recipes:
        raise ValueError('Baseline mancante')
    replicate_ids = sorted({key[1] for key in values})
    items = sorted({key[2] for key in values})
    expected = set(itertools.product(recipes, replicate_ids, items))
    if set(values) != expected:
        raise ValueError('Il confronto appaiato richiede le stesse partite e repliche per ogni ricetta')
    tensor = {recipe: np.array([[values[recipe, rep, it] for it in items]
                               for rep in replicate_ids]) for recipe in recipes}
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.integers(len(items), size=(n_boot, len(items))) if n_boot else None
    metric_names = ('spellF1', 'macroF1')
    output = {'baseline': baseline, 'matches': _lists(items), 'replicates': replicate_ids,
              'scale': '0_to_1', 'recipes': {},
              'uncertainty': 'Bootstrap descrittivo per partita, repliche tenute insieme; '
                             'condizionato ai semi osservati. CV di sviluppo, non test indipendente.'}
    for recipe in recipes:
        data = tensor[recipe]
        delta = data - tensor[baseline]
        seed_means, seed_deltas = data.mean(axis=1), delta.mean(axis=1)
        per_match_delta = delta.mean(axis=0)
        record = {'mean': dict(zip(metric_names, data.mean(axis=(0, 1)).tolist())),
                  'paired_delta': dict(zip(metric_names, delta.mean(axis=(0, 1)).tolist())),
                  'replicates': [{'replicate': rep,
                                  **dict(zip(metric_names, seed_means[i].tolist())),
                                  'paired_delta': dict(zip(metric_names, seed_deltas[i].tolist()))}
                                 for i, rep in enumerate(replicate_ids)],
                  'per_match': [{'source': it[0], 'match': it[1],
                                 **dict(zip(metric_names, data[:, i].mean(axis=0).tolist())),
                                 'paired_delta': dict(zip(metric_names, per_match_delta[i].tolist()))}
                                for i, it in enumerate(items)],
                  'delta_sd_across_replicates': dict(zip(metric_names, seed_deltas.std(axis=0, ddof=1).tolist()))
                  if len(replicate_ids) > 1 else None}
        if n_boot:
            interval = np.quantile(per_match_delta[samples].mean(axis=1), [.025, .975], axis=0)
            record['paired_match_bootstrap_ci95'] = {metric: interval[:, i].tolist()
                                                    for i, metric in enumerate(metric_names)}
        output['recipes'][recipe] = record
    if ensemble_rows is not None:
        ensemble = {}
        for row in ensemble_rows:
            item = _item((row['source'], row['match']))
            if item[0] not in TACTICAL:
                continue
            key = row['recipe'], item
            score = np.asarray([row['spellF1'], row['macroF1']], float)
            if key in ensemble or not np.isfinite(score).all() or (score < 0).any() or (score > 1).any():
                raise ValueError('Punteggio ensemble duplicato o non valido')
            ensemble[key] = score
        if set(ensemble) != set(itertools.product(recipes, items)):
            raise ValueError('Ensemble incompleto')
        output['ensemble'] = {recipe: dict(zip(metric_names, np.mean([ensemble[recipe, it] for it in items], axis=0).tolist()))
                              for recipe in recipes}
        output['ensemble_note'] = 'Scoring separato delle probabilità aggregate e decodificate; non media dei punteggi delle repliche.'
    return output


def source_fingerprints(paths, *, project_root=PROJECT_ROOT):
    """Hash byte-per-byte dei sorgenti; rifiuta file fuori dal progetto."""
    root = Path(project_root).resolve()
    result = {}
    for value in paths:
        path = Path(value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Sorgente fuori progetto o mancante: {path}')
        name = path.relative_to(root).as_posix()
        if name in result:
            raise ValueError('Sorgente duplicato')
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise ValueError('Elenco sorgenti vuoto')
    return dict(sorted(result.items()))


def archive_sources(destination, paths, *, project_root=PROJECT_ROOT):
    """Conserva sorgenti esatti e manifest prima del training, senza sovrascrivere.

    Per preservare un esperimento precedente, chiamare PRIMA di modificare i
    suoi sorgenti. Un hash del vecchio codice senza la copia non lo ricostruisce.
    """
    paths = list(paths)
    root = Path(project_root).resolve()
    destination = Path(destination).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError('Snapshot fuori dal progetto')
    manifest = source_fingerprints(paths, project_root=root)
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Snapshot già associato ad altri sorgenti')
    for name, expected in manifest.items():
        target = destination / name
        content = (root / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Sorgente cambiato durante lo snapshot')
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError('Snapshot esistente modificato')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_bytes(content)
        temporary.replace(target)
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary.replace(manifest_path)
    return manifest
