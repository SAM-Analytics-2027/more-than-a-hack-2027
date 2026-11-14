"""Confronto su tracking sporcato, con train/calibrazione/test a partite disgiunte.

La vista ``originale`` è il tracking SPORCATO senza perturbazioni aggiuntive.
La vista stressata è solo diagnostica: non determina Viterbi o ricetta vincente.
Il confronto non legge mai la partita di gara né la sua ground truth pubblica.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import augmentation as A
import constants as K
import dati
import esperimento as E
import modello as M
import rete
import riproduci as R

MODES = {
    'base': (False, False),
    'tracking': (True, False),
    'transizioni': (False, True),
    'combinato': (True, True),
}
FOCUSED_CROPS = .25
EVAL_VIEWS = ('originale', A.TEST_VIEW)
PROTOCOL_VERSION = 2
CALIBRATION_VIEW = 'originale'
MACRO_FLOOR = .90


def valid_name(name):
    return bool(name) and all(c.isalnum() or c in '_-' for c in name)


def code_fingerprints():
    """Tutti i componenti che influenzano training, inferenza e valutazione."""
    modules = (A, K, dati, A.C, A.C.fc, rete, E, M, M.pm, R)
    paths = [Path(__file__), *(Path(module.__file__) for module in modules)]
    return {path.name: hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for path in paths}


def cache_fingerprints(items):
    return {'/'.join(it): A.cache_path(it, 'originale').parent.name for it in items}


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def split_fold(items, fold_of, k):
    """Tre insiemi di partite, senza duplicati o viste della stessa gara ai due lati."""
    if len(set(items)) != len(items) or set(fold_of) != set(items):
        raise ValueError('Elenco partite o assegnazione dei fold non valido')
    if k not in range(K.N_FOLDS) or set(fold_of.values()) - set(range(K.N_FOLDS)):
        raise ValueError('Indice fold non valido')
    val = [it for it in items if fold_of[it] == (k + 1) % K.N_FOLDS]
    test = [it for it in items if fold_of[it] == k]
    train = [it for it in items if it not in val and it not in test]
    if not all((train, val, test)):
        raise ValueError('Ogni fold richiede partite di training, calibrazione e test')
    return train, val, test


def _rows(items, cols, tracking):
    xs, ys, ws = [], [], []
    for it in items:
        frame = A.load(it)
        keep = (frame['y'].to_numpy() < K.X) & (frame['period'].to_numpy() > 0)
        # Mantiene un budget fisso ma non allinea tutte le partite alla fase zero.
        rng = A.rng_for(it, 'righe_hgb')
        valid = np.flatnonzero(keep)
        sel = valid[int(rng.integers(min(K.STEP, len(valid))))::K.STEP] if len(valid) else valid
        if not len(sel):
            raise ValueError(f'Nessuna etichetta valida per allenare HGB: {it}')
        x = frame.iloc[sel][cols].copy()
        if tracking:
            choices = rng.choice(len(A.TRAIN_VIEWS), len(sel), p=A.VIEW_PROB)
            for i, view in enumerate(A.TRAIN_VIEWS[1:], 1):
                rows = np.flatnonzero(choices == i)
                x.iloc[rows] = A.load(it, view).iloc[sel[rows]][cols].to_numpy()
        y = frame['y'].to_numpy()[sel]
        w = E.short_spell_weights(frame['y'].to_numpy(), frame['period'].to_numpy())[sel]
        xs.extend((x, A.C.swap_teams(x)[cols]))
        ys.extend((y, np.where(y < K.D, 1 - y, y)))
        ws.extend((w * K.SOURCE_WEIGHTS[it[0]],) * 2)
    return pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


def _focus(frame, sel):
    """Indici a 5 Hz di cambi validi, senza superare confini di tempo o etichette X."""
    y = frame['y'].to_numpy()
    if not len(sel):
        return np.empty(0, dtype=int)
    period = frame['period'].to_numpy()
    candidates = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] < K.X) & (y[:-1] < K.X)
                               & (period[1:] == period[:-1]) & (period[1:] > 0)) + 1
    candidates = candidates[(candidates > sel[0]) & (candidates <= sel[-1])]
    return np.unique(np.searchsorted(sel, candidates))


def _sequences(items, cols, tracking):
    samples, sequences = [], []
    for it in items:
        frame = A.load(it)
        sample = frame.loc[frame['period'] > 0].iloc[::K.NORM_SUBSAMPLE_STEP][cols]
        # La normalizzazione vede esattamente la simmetria usata in allenamento.
        samples.extend((sample.to_numpy(copy=True), A.C.swap_teams(sample)[cols].to_numpy(copy=True)))
        raw = rete.raw_sequences(frame, cols, K.SOURCE_WEIGHTS[it[0]])
        views = [raw]
        if tracking:
            for view in A.TRAIN_VIEWS[1:]:
                views.append(rete.raw_sequences(A.load(it, view), cols, K.SOURCE_WEIGHTS[it[0]]))
        for i, (_, _, y, swapped, w, sel) in enumerate(raw):
            focus = _focus(frame, sel)
            for side, labels in enumerate((y, swapped)):
                sequences.append((tuple(v[i][side] for v in views), labels, w, focus))
    if not sequences:
        raise ValueError('Nessuna sequenza valida nelle partite di training')
    return rete.normalizer(samples), sequences


def _train_net(prepared, cols, seed, device, epochs, focused, smoke):
    (med, iqr), sequences = prepared
    lengths = np.asarray([len(seq[1]) for seq in sequences])
    probability = lengths / lengths.sum()
    steps = 3 if smoke else max(1, int(lengths.sum()) // (rete.CROP * K.BATCH_SIZE))
    view_rng, focus_rng = (np.random.default_rng([seed, stream]) for stream in (1, 2))
    tracking = len(sequences[0][0]) > 1

    def batch(rng):
        xb, yb, wb = [], [], []
        ids = rng.choice(len(sequences), size=K.BATCH_SIZE, p=probability)
        views = view_rng.choice(len(A.TRAIN_VIEWS), K.BATCH_SIZE, p=A.VIEW_PROB) if tracking else np.zeros(K.BATCH_SIZE, int)
        for i, view in zip(ids, views):
            xs, y, w, events = sequences[i]
            start = rng.integers(0, max(1, len(y) - rete.CROP + 1))
            if focused and len(events) and focus_rng.random() < FOCUSED_CROPS:
                middle = focus_rng.choice(events) + focus_rng.integers(-rete.CROP // 6, rete.CROP // 6 + 1)
                start = int(np.clip(middle - rete.CROP // 2, 0, max(0, len(y) - rete.CROP)))
            xc, yc, wc = rete.training_crop(xs[view], y, w, start)
            xb.append(xc)
            yb.append(yc)
            wb.append(wc)
        return xb, yb, wb

    def progress(epoch, total):
        if epoch in (1, total) or epoch % 5 == 0:
            print(f'    rete {seed}, epoca {epoch}/{total}', flush=True)

    return rete.fit_batches(cols, seed, device, med, iqr, steps, batch,
                            raw=True, epochs=epochs, progress=progress)


def _fit_tree(path, items, val_rows, cols, tracking, smoke):
    if path.is_file():
        return joblib.load(path)
    x, y, w = _rows(items, cols, tracking)
    xv, yv, wv = val_rows
    model = E.make_model()
    if smoke:
        model.set_params(max_iter=4)
    model.fit(x, y, sample_weight=w, X_val=xv, y_val=yv, sample_weight_val=wv)
    tmp = path.with_suffix('.tmp')
    joblib.dump(model, tmp, compress=3)
    tmp.replace(path)
    print(f'    HGB {path.stem}: {model.n_iter_} iterazioni', flush=True)
    return model


def _fit_nets(path, modes, items, cols, tracking, seeds, device, epochs, smoke):
    prepared = None
    for mode in modes:
        _, focused = MODES[mode]
        folder = path / mode
        folder.mkdir(exist_ok=True)
        for i, seed in enumerate(seeds):
            weights = folder / f'rete_seme{i}.pt'
            if weights.is_file():
                continue
            if prepared is None:
                prepared = _sequences(items, cols, tracking)
            print(f'    {mode}: rete {i + 1}/{len(seeds)}', flush=True)
            net = _train_net(prepared, cols, seed, device, epochs, focused, smoke)
            tmp = weights.with_suffix('.tmp')
            rete.save(net, tmp)
            tmp.replace(weights)
            del net


def _probabilities(tree, nets, items, cols):
    predictions = {}
    for it in items:
        for view in EVAL_VIEWS:
            frame = A.load(it, view)
            predictions[it, view] = R.ensemble_proba((tree, nets, cols), frame, (*it, view)).astype(np.float32)
            rete._RAW.clear()
    return predictions


def _meta(items):
    meta = {}
    for it in items:
        for view in EVAL_VIEWS:
            frame = pd.read_parquet(A.cache_path(it, view), columns=['y', 'period', 'n_casa', 'n_ospiti'])
            meta[it, view] = (frame['y'].to_numpy(), frame['period'].to_numpy(),
                              (frame['n_casa'] + frame['n_ospiti']).to_numpy() == 0)
    return meta


def _objective(spell, macro, macro_floor=MACRO_FLOOR):
    """Prima soddisfa il vincolo frame; oltre la soglia ottimizza i possessi."""
    if macro >= macro_floor:
        return (1, float(spell), float(macro))
    return (0, float(macro), float(spell))


def _calibrate(probs, meta, items, rounds, macro_floor=MACRO_FLOOR):
    """Media per partita DFL+Metrica sporcata, mai stress o test del fold.

    Ogni ricetta calibra le proprie probabilità. Si decodifica prima di inserire
    il separatore dell'intervallo: il separatore è una convenzione del giudice,
    non un frame sintetico da dare al Viterbi.
    """
    tactical = [it for it in items if it[0] in K.DM]
    if not tactical:
        raise ValueError('La calibrazione richiede almeno una partita DFL o Metrica')

    def evaluate(params):
        spells, macros = [], []
        for it in tactical:
            y, period, no_data = meta[it, CALIBRATION_VIEW]
            decoded = M.decode(probs[it, CALIBRATION_VIEW], period, no_data, params)
            gt = M.to_codes(R._gap(y, period))
            pred = M.to_codes(R._gap(decoded, period))
            spells.append(M.spell_f1_fast(gt, pred))
            macros.append(M.pm.macro_f1(M.pm.scored_confusion(M.pm.confusion(gt, pred))))
        return float(np.mean(spells)), float(np.mean(macros))

    params = dict(K.DEFAULT)
    best = _objective(*evaluate(params), macro_floor)
    for r in range(rounds):
        changed = False
        for key, values in K.GRID.items():
            for value in values:
                trial = dict(params, **{key: value})
                candidate = _objective(*evaluate(trial), macro_floor)
                # Evita cambi dovuti al rumore numerico o a pareggi esatti.
                if candidate > best and any(abs(a - b) > 1e-8 for a, b in zip(candidate, best)):
                    best, params, changed = candidate, trial, True
        spell, macro = evaluate(params)
        print(f'    Viterbi giro {r + 1}: spellF1 {spell:.4f}, macroF1 {macro:.4f}', flush=True)
        if not changed:
            break
    return params


def _score(prob, info, params):
    y, period, no_data = info
    decoded = M.decode(prob, period, no_data, params)
    gt = M.to_codes(R._gap(y, period))
    pred = M.to_codes(R._gap(decoded, period))
    pm = M.pm
    runs = pm.valid_runs(gt)
    g = pm.possession_spells(gt, runs)
    p = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    overlap = pm.spell_overlaps(g, p)
    flags = pm.matched_gt_flags(overlap, len(g), .5)
    short = np.array([b - a < 2 * K.FPS for a, b, _ in g], dtype=bool)
    return {
        'spellF1': M.spell_f1_fast(gt, pred),
        'macroF1': float(pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))),
        'possessi': len(g), 'predetti': len(p), 'abbinati50': int(flags.sum()),
        'brevi': int(short.sum()), 'brevi_abbinati50': int((flags & short).sum()),
    }


def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def _save_probabilities(path, probabilities):
    """Conserva soltanto predizioni esterne, riusabili senza ripetere l'inferenza."""
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **{'__'.join((*it, view)): p for (it, view), p in probabilities.items()})
    tmp.replace(path)


def load_probabilities(path):
    with np.load(path, allow_pickle=False) as archive:
        return {((parts[0], parts[1]), parts[2]): archive[key]
                for key in archive.files for parts in [key.split('__')]}


def _verify_artifacts(path, result):
    artifacts = result.get('artefatti')
    if not artifacts:
        raise ValueError(f'Fold senza manifest degli artefatti: {path}; usa un nuovo esperimento')
    for relative, expected in artifacts.items():
        artifact = (path / relative).resolve()
        if not artifact.is_relative_to(path.resolve()):
            raise ValueError('Percorso artefatto fuori dal fold')
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f'Artefatto mancante o modificato: {artifact}')


def _export_fold(path, result, run_name):
    """CSV dashboard da predizioni test e Viterbi calibrato senza quelle partite."""
    items = [tuple(it) for it in result['test'] if it[0] in K.DM]
    meta = _meta(items)
    for mode in result['modi']:
        folder = Path(R.EXT_RESULTS) / f'{run_name}__{mode}'
        provenance = folder / 'provenienza.json'
        description = {'protocollo': PROTOCOL_VERSION, 'esperimento': run_name, 'modo': mode,
                       'tracking': 'sporcato', 'validazione': 'train_calibrazione_test_disgiunti',
                       'punteggio': 'cross_validation_di_sviluppo_non_test_indipendente'}
        if provenance.exists() and json.loads(provenance.read_text(encoding='utf-8')) != description:
            raise ValueError(f'Output dashboard appartiene a un altro esperimento: {folder}')
        _write(provenance, description)
        probabilities = load_probabilities(path / mode / 'probabilita.npz')
        for it in items:
            _, period, no_data = meta[it, CALIBRATION_VIEW]
            codes = M.decode(probabilities[it, CALIBRATION_VIEW], period, no_data, result['viterbi'][mode])
            target = folder / f'{it[1]}.csv'
            temporary = target.with_suffix('.tmp')
            R._write_csv(str(temporary), codes)
            temporary.replace(target)


def _run_fold(path, items, fold_of, k, modes, cols, args, device):
    path.mkdir(exist_ok=True)
    out = path / 'risultati.json'
    if out.is_file():
        result = json.loads(out.read_text(encoding='utf-8'))
        expected = split_fold(items, fold_of, k)
        actual = tuple([tuple(it) for it in result[key]] for key in ('train', 'calibrazione', 'test'))
        if expected != actual or set(result['modi']) != set(modes):
            raise ValueError(f'Partizioni o ricette incoerenti nel fold {k}')
        _verify_artifacts(path, result)
        _export_fold(path, result, args.nome)
        return result
    train, val, test = split_fold(items, fold_of, k)
    print(f'Fold {k + 1}/{K.N_FOLDS}: {len(train)} train, {len(val)} calibrazione, {len(test)} test', flush=True)
    seeds = [K.SEED + 1000 * s + k for s in range(args.n_reti)]
    calibration_rows = _rows(val, cols, tracking=False)
    meta = _meta(val + test)
    results, parameters = {}, {}
    for tracking in (False, True):
        selected = [mode for mode in modes if MODES[mode][0] == tracking]
        if not selected:
            continue
        tree = _fit_tree(path / ('hgb_tracking.joblib' if tracking else 'hgb_base.joblib'),
                         train, calibration_rows, cols, tracking, args.smoke)
        _fit_nets(path, selected, train, cols, tracking, seeds, device, args.epoche, args.smoke)
        for mode in selected:
            nets = [rete.load(path / mode / f'rete_seme{i}.pt') for i in range(len(seeds))]
            probs = _probabilities(tree, nets, val + test, cols)
            params = _calibrate(probs, meta, val, 1 if args.smoke else 2, args.macro_floor)
            parameters[mode] = params
            _save_probabilities(path / mode / 'probabilita.npz', probs)
            _write(path / mode / 'modello.json', {
                'partite': ['/'.join(it) for it in train], 'semi_rete': seeds, 'colonne': cols,
                'feature_version': 2, 'ricetta': mode, 'calibrazione': ['/'.join(it) for it in val],
                'test': ['/'.join(it) for it in test], 'tracking': 'sporcato',
                'viterbi': params, 'dispositivo_allenamento': str(device),
            })
            rows = []
            for it in test:
                for view in EVAL_VIEWS:
                    metrics = _score(probs[it, view], meta[it, view], params)
                    rows.append(dict(fonte=it[0], partita=it[1], vista=view, **metrics))
            results[mode] = rows
            regular = [r for r in rows if r['vista'] == CALIBRATION_VIEW and r['fonte'] in K.DM]
            spell = np.mean([r['spellF1'] for r in regular])
            macro = np.mean([r['macroF1'] for r in regular])
            print(f'    {mode}: spellF1 {spell:.4f}, macroF1 {macro:.4f}, test sporcato DFL + Metrica', flush=True)
    artifact_paths = sorted([*path.glob('*.joblib'), *path.glob('*/*.pt'),
                             *path.glob('*/probabilita.npz'), *path.glob('*/modello.json')])
    result = {'fold': k, 'train': train, 'calibrazione': val, 'test': test,
              'viterbi': parameters, 'modi': results,
              'artefatti': {p.relative_to(path).as_posix(): _sha256(p) for p in artifact_paths}}
    _write(out, result)
    _export_fold(path, result, args.nome)
    return result


def _summary(folds, modes):
    result = {}
    for mode in modes:
        rows = pd.DataFrame([row for fold in folds for row in fold['modi'][mode]])
        result[mode] = {}
        for view in EVAL_VIEWS:
            result[mode][view] = {}
            for label, sources in [('tattiche', K.DM), *[(src, (src,)) for src in K.SOURCES]]:
                subset = rows[(rows['vista'] == view) & rows['fonte'].isin(sources)]
                if subset.empty:
                    continue
                result[mode][view][label] = {
                    'partite': int(len(subset)),
                    'spellF1': float(subset['spellF1'].mean()),
                    'macroF1': float(subset['macroF1'].mean()),
                    'spellF1_peggiore': float(subset['spellF1'].min()),
                    'precisione50': float(subset['abbinati50'].sum() / max(1, subset['predetti'].sum())),
                    'recall_brevi50': float(subset['brevi_abbinati50'].sum() / max(1, subset['brevi'].sum())),
                }
    return result


def validate_completed_report(report, config, items):
    """Il finale richiede cinque fold veri, copertura esatta e tutte le ricette."""
    if config.get('protocollo') != PROTOCOL_VERSION or config.get('tracking') != 'sporcato':
        raise ValueError('Confronto precedente al protocollo sporcato: occorre un nuovo esperimento')
    if config['smoke']:
        raise ValueError('Un esperimento smoke non può produrre il modello finale')
    folds = report['fold']
    if len(folds) != K.N_FOLDS or sorted(f['fold'] for f in folds) != list(range(K.N_FOLDS)):
        raise ValueError('Servono tutti i fold, senza duplicati')
    fold_of = E.folds(items)
    for fold in folds:
        expected = split_fold(items, fold_of, fold['fold'])
        actual = tuple([tuple(it) for it in fold[key]] for key in ('train', 'calibrazione', 'test'))
        if actual != expected or set(fold['modi']) != set(config['modi']):
            raise ValueError('Partizioni o ricette del confronto non coerenti')
        for mode in config['modi']:
            expected_rows = {(src, mid, view) for src, mid in expected[2] for view in EVAL_VIEWS}
            rows = fold['modi'][mode]
            keys = [(r['fonte'], r['partita'], r['vista']) for r in rows]
            if len(keys) != len(set(keys)) or set(keys) != expected_rows:
                raise ValueError('Risultati mancanti o duplicati nel confronto')
            if not all(np.isfinite(r[m]) and 0 <= r[m] <= 1 for r in rows for m in ('spellF1', 'macroF1')):
                raise ValueError('Metriche non valide nel confronto')
    return sorted(folds, key=lambda f: f['fold'])


def select_mode(folds, modes, macro_floor=MACRO_FLOOR):
    """Selezione esterna dichiarata come sviluppo, senza consultare il Barça."""
    summary = _summary(folds, modes)
    scores = {mode: summary[mode][CALIBRATION_VIEW]['tattiche'] for mode in modes}
    # In caso di pareggio conserva la ricetta più semplice nell'ordine MODES.
    chosen = max(modes, key=lambda mode: _objective(scores[mode]['spellF1'], scores[mode]['macroF1'], macro_floor))
    return chosen, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nome', required=True)
    parser.add_argument('--modi', nargs='+', choices=list(MODES), default=list(MODES))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--n-reti', type=int, default=K.N_NETS)
    parser.add_argument('--epoche', type=int, default=K.EPOCHS)
    parser.add_argument('--fold', type=int, choices=range(K.N_FOLDS))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--macro-floor', type=float, default=MACRO_FLOOR,
                        help='vincolo macroF1 nella sola calibrazione esterna (default 0.90)')
    args = parser.parse_args()
    if not valid_name(args.nome):
        parser.error('--nome deve contenere solo lettere, cifre, trattini o underscore')
    if min(args.workers, args.threads, args.n_reti, args.epoche) < 1:
        parser.error('worker, thread, reti ed epoche devono essere positivi')
    if not 0 <= args.macro_floor <= 1:
        parser.error('--macro-floor deve essere compreso tra 0 e 1')
    modes = [mode for mode in MODES if mode in args.modi]
    items = sorted(dati.external_matches())
    fold_of = E.folds(items)
    root = Path(dati.CACHE) / 'augmentation_runs' / args.nome
    root.mkdir(parents=True, exist_ok=True)
    print('Cache delle feature esterne', flush=True)
    A.prepare(items, args.workers)
    device = rete.device()
    config = {
        'protocollo': PROTOCOL_VERSION, 'tracking': 'sporcato', 'feature_version': 2,
        'modi': modes, 'n_reti': args.n_reti, 'epoche': args.epoche, 'smoke': args.smoke,
        'macro_floor': args.macro_floor, 'calibration_view': CALIBRATION_VIEW,
        'obiettivo': 'media_per_partita_DFL_Metrica',
        'fold': {'/'.join(it): fold_of[it] for it in items},
        'pesi_fonti': K.SOURCE_WEIGHTS, 'probabilita_viste': dict(zip(A.TRAIN_VIEWS, A.VIEW_PROB)),
        'quota_crop_transizioni': FOCUSED_CROPS, 'device': str(device), 'versioni': R._versions(),
        'cache': cache_fingerprints(items), 'codice': code_fingerprints(),
    }
    config_path = root / 'config.json'
    if config_path.exists() and config != json.loads(config_path.read_text(encoding='utf-8')):
        parser.error(f'configurazione differente per {root}, usa un altro nome')
    _write(config_path, config)
    torch.set_num_threads(args.threads)
    cols = E.feature_cols(A.load(items[0]))
    start = time.monotonic()
    with threadpool_limits(limits=args.threads):
        folds = range(K.N_FOLDS) if args.fold is None else (args.fold,)
        for k in folds:
            _run_fold(root / f'fold{k}', items, fold_of, k, modes, cols, args, device)
    finished = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('fold*/risultati.json'))]
    summary = _summary(finished, modes)
    _write(root / 'risultati.json', {'fold_completati': len(finished), 'sintesi': summary, 'fold': finished,
                                    'interpretazione': 'Cross-validation di sviluppo. La ricetta viene selezionata '
                                                      'su questi risultati: non sono un test finale indipendente.'})
    print(json.dumps(summary, indent=2), flush=True)
    print(f'{len(finished)}/{K.N_FOLDS} fold, {(time.monotonic() - start) / 60:.1f} min. {root}', flush=True)


if __name__ == '__main__':
    main()
