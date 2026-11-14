"""Clean-source augmentation: controlled ablations and genuinely nested CV.

Every model sees dirty inputs only. Corruption profiles are fitted exclusively
on the actual training matches of each fit. The fixed ten dirty benchmark games
are never regenerated. No public ground truth is read by this runner.

python modelli_mv/studio_augmentation.py --nome aug_pulite_v3 --fase tutto
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import constants as K
import dati
import esperimento as E
import modello as M
import rete
import riproduci as R
import prova_augmentation as P
import componenti_augmentation as C
import protocollo_augmentation as Q

MODES = ('base', 'risporcata')
ABLATIONS = ('feature', 'contesto', 'normalizzazione', 'segmentazione', 'inferenza', 'moderno')
REPLICATE_SEEDS = (2027, 12027, 22027)
SCHEMA = 'clean-source-nested-v1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_hashes():
    here = Path(__file__).parent
    names = ('studio_augmentation.py', 'augmentation_realiste.py', 'componenti_augmentation.py',
             'protocollo_augmentation.py', 'constants.py', 'dati.py', 'caratteristiche.py',
             'formato_comune.py', 'rete.py', 'modello.py', 'esperimento.py', 'riproduci.py',
             'prova_augmentation.py')
    paths = [here / name for name in names]
    paths.append(here.parent / 'evaluation' / 'possession_metrics.py')
    return {p.relative_to(here.parent).as_posix(): P._sha256(p) for p in paths}


def A_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_original(item, feature_version=1):
    df = dati.load(*item, legacy=feature_version == 1)
    df['fonte_tv'] = np.float32(item[0] == 'skillcorner')
    return df


def info(df):
    return (df['y'].to_numpy(), df['period'].to_numpy(),
            (df['n_casa'].to_numpy() + df['n_ospiti'].to_numpy()) == 0)


def metadata(items):
    return {it: info(load_original(it)) for it in items}


def save_probabilities(path, probabilities):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **{'__'.join(it): p for it, p in probabilities.items()})
    tmp.replace(path)


def load_probabilities(path):
    with np.load(path, allow_pickle=False) as f:
        return {tuple(name.split('__')): f[name] for name in f.files}


def verify_manifest(folder, result):
    P._verify_artifacts(Path(folder), {'artefatti': result['artifacts']})
    if 'reused_weights' in result:
        source = Path(result['reused_weights']['path'])
        if P._sha256(source / 'completato.json') != result['reused_weights']['manifest_sha256']:
            raise ValueError('Pesi riutilizzati diversi dalla prova di inferenza')
        verify_manifest(source, read_json(source / 'completato.json'))


def verify_fit(root, plan, mode, replicate, key):
    folder = fit_path(root, mode, replicate, key)
    result = read_json(folder / 'completato.json')
    Q.validate_fit_provenance(plan, key, result['provenance'], generator_required=mode == 'risporcata')
    config = read_json(root / 'config.json')
    preset = mode[3:] if mode.startswith('ab_') else 'base'
    if (result['provenance'].get('seed') != config['seeds'][replicate]
            or result['provenance'].get('preset') != asdict(C.PRESETS[preset])
            or result['provenance'].get('augmented') != (mode == 'risporcata')
            or result['provenance'].get('excluded') != plan['fits'][key]['excluded']):
        raise ValueError('Fit di un altro seme, componente o fold')
    verify_manifest(folder, result)
    return result


def verify_complete(root, config, plan):
    completed = []
    for mode in MODES:
        for replicate in range(len(config['seeds'])):
            for key in plan['fits']:
                verify_fit(root, plan, mode, replicate, key)
                completed.append((mode, replicate, key))
    Q.validate_completion(plan, completed, MODES, range(len(config['seeds'])), smoke=config['smoke'])


def checked_decoder(root, plan, mode, replicate, outer):
    path = root / 'decoders' / mode / f'r{replicate}' / f'fold{outer["fold"]}.json'
    recorded = read_json(path)
    for key in outer['inner_sources']:
        verify_fit(root, plan, mode, replicate, key)
    expected = {'calibration': outer['calibration'], 'excluded_outer_fold': outer['fold'],
        'inner_sources': outer['inner_sources'], 'input_hashes': {
            key: P._sha256(fit_path(root, mode, replicate, key) / 'completato.json')
            for key in outer['inner_sources']}}
    if {key: recorded.get(key) for key in expected} != expected:
        raise ValueError('Decoder di un altro insieme di calibrazione')
    if recorded.get('params_sha256') != A_digest(recorded['params']):
        raise ValueError('Parametri del decoder alterati')
    return recorded['params']


def model_name(root, mode, replicate):
    return f'{root.name}__{mode}_r{replicate}'


def fit_path(root, mode, replicate, key):
    return root / 'fits' / mode / f'r{replicate}' / key


def _profile(train, folder):
    import augmentation_realiste as A
    path = folder / 'profilo.json'
    profile = A.fit_profile(train)
    A.validate_profile(profile, train_items=train, verify_inputs=True)
    if path.exists() and read_json(path) != profile:
        raise ValueError(f'Profilo differente da quello registrato: {path}')
    P._write(path, profile)
    return profile


def _view_loader(profile, seed, replicas, feature_version):
    import augmentation_realiste as A
    def views(item):
        result = []
        for replica in range(replicas):
            df = A.load_view(item, profile, seed=seed, replica=replica, legacy=feature_version == 1)
            df['fonte_tv'] = np.float32(item[0] == 'skillcorner')
            result.append(df)
        return result
    return views


def fit_one(root, config, plan, mode, replicate, key, preset='base'):
    folder = fit_path(root, mode, replicate, key)
    done = folder / 'completato.json'
    augmented = mode == 'risporcata'
    specification = plan['fits'][key]
    train = [tuple(it) for it in specification['train']]
    test = [tuple(it) for it in specification['test']]
    seed = config['seeds'][replicate]
    component_config = C.PRESETS[preset]
    provenance = dict(train=train, normalization_train=train, label_train=train,
                      generator_train=train if augmented else [], excluded=specification['excluded'],
                      preset=asdict(component_config), seed=seed, augmented=augmented,
                      early_stopping=False)
    # JSON roundtrip makes tuple/list comparison explicit during resume.
    provenance = json.loads(json.dumps(provenance))
    Q.validate_fit_provenance(plan, key, provenance, generator_required=augmented)
    if done.exists():
        result = read_json(done)
        if result['provenance'] != provenance:
            raise ValueError(f'Provenienza diversa: {folder}')
        verify_manifest(folder, result)
        return result
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    print(f'FIT {mode} r{replicate} {key}: train={len(train)}, test={len(test)}', flush=True)
    P._write(root / 'progresso.json', {'mode': mode, 'replicate': replicate, 'fit': key,
                                       'stage': 'training', 'started_unix': time.time()})
    loader = lambda it: load_original(it, component_config.feature_version)
    cols = E.feature_cols(loader(train[0]))
    views = None
    if augmented:
        import augmentation_realiste as A
        profile = _profile(train, folder)
        A.prepare_views(train, profile, seeds=(seed,), replicas=config['replicas'],
                        legacy=component_config.feature_version == 1, workers=config['augmentation_workers'])
        views = _view_loader(profile, seed, config['replicas'], component_config.feature_version)
    weights = folder / 'modello'
    if (weights / 'modello.json').exists():
        # This marker is written only after every weight has been saved.
        recorded = read_json(weights / 'modello.json')
        if (recorded['partite'] != ['/'.join(it) for it in train]
                or recorded['ablation_config'] != asdict(component_config)
                or recorded['semi_rete'] != [seed + 1000*i for i in range(config['n_nets'])]
                or recorded['epoche'] != config['epochs']
                or recorded['hgb_iterazioni'] != config['hgb_iterations']
                or recorded['viste'] != (1 + config['replicas'] if augmented else 1)):
            raise ValueError('Pesi parziali da un altro training')
        P._verify_artifacts(weights, {'artefatti': read_json(weights / 'manifest.json')})
        components = C.load_components(weights)
    elif preset == 'inferenza':
        source = fit_path(root, 'base', replicate, key)
        verify_manifest(source, read_json(source / 'completato.json'))
        components = C.load_components(source / 'modello')
    else:
        components = C.fit_components(loader, train, cols, weights,
            [seed + 1000 * i for i in range(config['n_nets'])], component_config,
            hgb_iterations=config['hgb_iterations'], tree_seed=seed,
            epochs=config['epochs'], steps=2 if config['smoke'] else None,
            device=rete.device(), views=views, original_probability=config['original_probability'],
            progress=lambda epoch, total: print(f'  epoca {epoch}/{total}', flush=True) if epoch == total else None)
    probabilities = {}
    for it in test:
        # Only original, fixed dirty inputs are evaluated.
        probabilities[it] = np.round(C.predict_components(components, loader(it), it,
            inference_version=2 if preset == 'inferenza' else None), R.DECIMALS)
        print(f'  previsto {it[0]}/{it[1]}', flush=True)
    save_probabilities(folder / 'probabilita.npz', probabilities)
    paths = [folder / 'probabilita.npz', *sorted(weights.glob('*'))]
    if augmented:
        paths.append(folder / 'profilo.json')
    result = {'provenance': provenance, 'elapsed_seconds': time.monotonic() - t0,
              'artifacts': {p.relative_to(folder).as_posix(): P._sha256(p) for p in paths if p.is_file()}}
    if preset == 'inferenza':
        result['reused_weights'] = {'path': str(source), 'manifest_sha256': P._sha256(source / 'completato.json')}
    P._write(done, result)
    del components, probabilities, views
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f'DONE {mode} r{replicate} {key}: {result["elapsed_seconds"] / 60:.1f} min', flush=True)
    return result


def score_replicate(root, config, plan, mode, replicate, *, ablation=False):
    output = root / 'scores' / f'{mode}_r{replicate}.json'
    rows, decoders = [], {}
    items = [tuple(it) for it in plan['items']]
    meta = metadata(items)
    dashboard_name = model_name(root, mode, replicate)
    destination = Path(R.EXT_RESULTS) / dashboard_name
    destination.mkdir(parents=True, exist_ok=True)
    P._write(destination / 'provenienza.json', {'schema': SCHEMA, 'experiment': root.name,
        'mode': mode, 'replicate': replicate, 'smoke': config['smoke'],
        'evaluation': 'ablation_fixed_base_decoder' if ablation else 'nested_cross_validation_development',
        'test_input': 'existing_dirty_fixed', 'independent_final_test': False})
    for outer in plan['outer']:
        k = outer['fold']
        fitfolder = fit_path(root, mode, replicate, outer['refit'])
        verify_fit(root, plan, mode, replicate, outer['refit'])
        predictions = load_probabilities(fitfolder / 'probabilita.npz')
        decoder_path = root / 'decoders' / mode / f'r{replicate}' / f'fold{k}.json'
        if ablation:
            params = checked_decoder(root, plan, 'base', replicate, outer)
        else:
            inner = {}
            for key in outer['inner_sources']:
                folder = fit_path(root, mode, replicate, key)
                verify_fit(root, plan, mode, replicate, key)
                inner[key] = load_probabilities(folder / 'probabilita.npz')
            calibration = Q.select_inner_predictions(plan, k, inner)
            tactical = sorted(calibration)
            expected_decoder = json.loads(json.dumps({'calibration': tactical,
                'excluded_outer_fold': k, 'inner_sources': outer['inner_sources'],
                'input_hashes': {key: P._sha256(fit_path(root, mode, replicate, key) / 'completato.json')
                                 for key in outer['inner_sources']}}))
            if decoder_path.exists():
                recorded = read_json(decoder_path)
                if {key: recorded.get(key) for key in expected_decoder} != expected_decoder:
                    raise ValueError('Provenienza del decoder diversa dalla calibrazione attuale')
                if recorded.get('params_sha256') != A_digest(recorded['params']):
                    raise ValueError('Parametri decoder alterati')
                params = recorded['params']
            else:
                print(f'CALIBRA {mode} r{replicate} fold{k}: {len(tactical)} partite', flush=True)
                params = P._calibrate({(it, 'originale'): p for it, p in calibration.items()},
                    {(it, 'originale'): meta[it] for it in tactical}, tactical,
                    rounds=2, macro_floor=config['macro_floor'])
                P._write(decoder_path, dict(expected_decoder, params=params, params_sha256=A_digest(params)))
            del inner, calibration
        decoders[str(k)] = params
        for item in map(tuple, outer['test']):
            scores = P._score(predictions[item], meta[item], params)
            row = dict(mode=mode, replicate=replicate, fold=k, source=item[0], match=item[1], **scores)
            if item[0] in K.DM:
                codes = M.decode(predictions[item], meta[item][1], meta[item][2], params)
                row['csv_sha256'] = R._write_csv(str(destination / f'{item[1]}.csv'), codes)
            rows.append(row)
    tactical = [r for r in rows if r['source'] in K.DM]
    report = {'mode': mode, 'replicate': replicate, 'rows': rows, 'decoders': decoders,
              'diagnostic_ablation': ablation, 'dashboard_model': dashboard_name,
              'mean10': {m: float(np.mean([row[m] for row in tactical])) for m in ('spellF1', 'macroF1')}}
    P._write(output, report)
    print(f'SCORE {mode} r{replicate}: {report["mean10"]}', flush=True)
    return report


def collect_report(root, config):
    reports = [read_json(p) for p in sorted((root / 'scores').glob('*.json'))]
    nested = [r for r in reports if not r['diagnostic_ablation']]
    complete = {(r['mode'], r['replicate']) for r in nested} == {
        (mode, replicate) for mode in MODES for replicate in range(len(config['seeds']))}
    summary = {}
    for mode in MODES:
        selected = [r for r in nested if r['mode'] == mode]
        if selected:
            summary[mode] = {metric: {'mean': float(np.mean([r['mean10'][metric] for r in selected])),
                'std_across_runs': float(np.std([r['mean10'][metric] for r in selected], ddof=1)) if len(selected)>1 else None,
                'runs': [r['mean10'][metric] for r in selected]} for metric in ('spellF1', 'macroF1')}
    result = dict(schema=SCHEMA, complete=complete, smoke=config['smoke'], summary=summary,
                  reports=reports, interpretation='Development CV; repeated model selection on these games is not an independent test.')
    if complete:
        # Match-clustered uncertainty is descriptive with only ten benchmark games.
        rows = [r for report in nested for r in report['rows'] if r['source'] in K.DM]
        result['paired'] = Q.paired_summary([dict(row, recipe=row['mode']) for row in rows], baseline='base')
    P._write(root / 'risultati.json', result)
    return result


def validate_results(root, config, plan, *, final=False):
    import valida_risultati_augmentation as V
    report = collect_report(root, config)
    if not report['complete'] or report['smoke']:
        raise ValueError('Confronto incompleto o smoke: nessuna validazione finale')
    verify_complete(root, config, plan)
    dashboard = V._dashboard()
    checked = []
    for score in report['reports']:
        Q.validate_score_report(plan, score)
        expected_keys = {tuple(it) for it in plan['items']}
        keys = [(row['source'], row['match']) for row in score['rows']]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError('Risultati mancanti o duplicati')
        for outer in plan['outer']:
            verify_fit(root, plan, score['mode'], score['replicate'], outer['refit'])
            decoder_mode = 'base' if score['diagnostic_ablation'] else score['mode']
            if checked_decoder(root, plan, decoder_mode, score['replicate'], outer) != score['decoders'][str(outer['fold'])]:
                raise ValueError('Decoder del report differente dagli artefatti')
        for row in score['rows']:
            if row['source'] not in K.DM:
                continue
            src, mid = row['source'], row['match']
            clean = Path(dati.BASE) / f'{src}_pulite' / mid
            csv = Path(R.EXT_RESULTS) / score['dashboard_model'] / f'{mid}.csv'
            checked.append({'model': score['dashboard_model'], 'match': mid, **V.validate_csv(
                csv, clean / f'{mid}_possession_gt.csv', clean / f'{mid}_atd.xml', row, dashboard)})
    models = {}
    if final:
        for mode in MODES:
            name = model_name(root, mode, 0)
            models[name] = V.validate_model(name, public=False)
    result = dict(ok=all(r['ok'] for r in checked) and all(m['ok'] for m in models.values()),
                  csv=checked, models=models, source_hashes=source_hashes(),
                  report_sha256=P._sha256(root / 'risultati.json'))
    P._write(root / 'validazione.json', result)
    print(f'VALIDAZIONE: {len(checked)} CSV, {len(models)} modelli, ok={result["ok"]}', flush=True)
    return result


def finalize(root, config, plan):
    report = collect_report(root, config)
    if not report['complete'] or report['smoke']:
        raise ValueError('Servono tutti i fold, le ricette e le repliche complete non-smoke')
    verify_complete(root, config, plan)
    items = [tuple(it) for it in plan['items']]
    meta = metadata(items)
    # Replicate zero is declared in the initial config, never selected by score.
    seed = config['seeds'][0]
    for mode in MODES:
        name = model_name(root, mode, 0)
        target = Path(R.model_dir(name))
        target.mkdir(parents=True, exist_ok=True)
        if (target / 'manifest.json').exists():
            P._verify_artifacts(target, {'artefatti': read_json(target / 'manifest.json')})
        recipe = {'schema': SCHEMA, 'experiment': root.name, 'mode': mode, 'replicate': 0,
                  'replicate_selection': 'preregistered_zero_not_best_seed', 'config': config,
                  'report_sha256': P._sha256(root / 'risultati.json')}
        if (target / 'ricetta.json').exists() and read_json(target / 'ricetta.json') != recipe:
            raise ValueError(f'Ricetta finale diversa: {target}')
        P._write(target / 'ricetta.json', recipe)
        params_path = target / 'parametri_viterbi.json'
        oof_hashes = {outer['refit']: P._sha256(fit_path(root, mode, 0, outer['refit']) / 'completato.json')
                      for outer in plan['outer']}
        if params_path.exists() and read_json(params_path).get('oof_hashes') != oof_hashes:
            raise ValueError('Parametri finali da altre probabilità OOF')
        if not params_path.exists():
            oof = {}
            for outer in plan['outer']:
                oof.update(load_probabilities(fit_path(root, mode, 0, outer['refit']) / 'probabilita.npz'))
            tactical = sorted(it for it in oof if it[0] in K.DM)
            params = P._calibrate({(it, 'originale'): oof[it] for it in tactical},
                {(it, 'originale'): meta[it] for it in tactical}, tactical, rounds=2,
                macro_floor=config['macro_floor'])
            P._write(params_path, {'params': params, 'params_sha256': A_digest(params), 'oof_hashes': oof_hashes,
                'origin': 'All external OOF, final inference only; never benchmark scoring'})
        elif read_json(params_path).get('params_sha256') != A_digest(read_json(params_path)['params']):
            raise ValueError('Parametri finali alterati')
        folder = target / 'finale'
        if (folder / 'modello.json').exists():
            P._verify_artifacts(folder, {'artefatti': read_json(folder / 'manifest.json')})
        if not (folder / 'modello.json').exists():
            views = _view_loader(_profile(items, target), seed, config['replicas'], 1) if mode == 'risporcata' else None
            components = C.fit_components(load_original, items, E.feature_cols(load_original(items[0])),
                folder, [seed + 1000 * i for i in range(config['n_nets'])], C.PRESETS['base'],
                hgb_iterations=config['hgb_iterations'], tree_seed=seed, epochs=config['epochs'],
                device=rete.device(), views=views, original_probability=config['original_probability'],
                progress=lambda epoch,total: print(f'FINALE {mode}: {epoch}/{total}', flush=True) if epoch==total else None)
            del components, views
            gc.collect()
        paths = [params_path, target / 'ricetta.json', *sorted(folder.iterdir())]
        if (target / 'profilo.json').exists():
            paths.append(target / 'profilo.json')
        P._write(target / 'manifest.json', {p.relative_to(target).as_posix(): P._sha256(p) for p in paths if p.is_file()})
    # Both recipes and all weights fixed before touching target tracking.
    P._write(root / 'modelli_fissati_prima_del_pubblico.json', {
        mode: {'name': model_name(root, mode, 0), 'manifest_sha256': P._sha256(
            Path(R.model_dir(model_name(root, mode, 0))) / 'manifest.json')} for mode in MODES})
    for mode in MODES:
        name = model_name(root, mode, 0)
        R.predict(root=R.model_dir(name), name=name, write=True, report_public=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nome', required=True)
    parser.add_argument('--fase', choices=('tutto', 'fits', 'score', 'ablazioni', 'finale', 'valida'), default='tutto')
    parser.add_argument('--modo', choices=MODES)
    parser.add_argument('--replica', type=int, choices=range(3))
    parser.add_argument('--fit', help='single fit, e.g. excl_0_1; smoke or resumable scheduling')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not P.valid_name(args.nome) or args.threads < 1:
        parser.error('Nome o thread non valido')
    import augmentation_realiste as A
    items = sorted(dati.external_matches())
    plan = Q.build_plan(items)
    root = Path(dati.CACHE) / 'augmentation_runs' / args.nome
    root.mkdir(parents=True, exist_ok=True)
    config = dict(schema=SCHEMA, seeds=list(REPLICATE_SEEDS), n_nets=1 if args.smoke else 3,
        epochs=1 if args.smoke else 20, hgb_iterations=4 if args.smoke else 500,
        smoke=args.smoke, replicas=3, original_probability=.5, macro_floor=.90, augmentation_workers=3,
        final_replicate=0, ablations=list(ABLATIONS), ablation_decoder='nested_base_same_fold',
        sources=source_hashes(), original_inputs={'/'.join(it): dati.input_hash(it) for it in items},
        clean_and_dirty_inputs={'/'.join(it): A.input_manifest(it, labels=True) for it in items},
        versions=R._versions(), device=str(rete.device()), threads=args.threads)
    config_path = root / 'config.json'
    if config_path.exists() and read_json(config_path) != config:
        parser.error('Configurazione, codice o dati modificati: usare un nuovo nome')
    P._write(config_path, config)
    P._write(root / 'piano.json', plan)
    snapshot = root / 'sorgenti'
    snapshot.mkdir(exist_ok=True)
    for name in config['sources']:
        destination = snapshot / name
        if destination.exists() and P._sha256(destination) != config['sources'][name]:
            raise ValueError('Snapshot sorgenti modificato')
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((Path(__file__).parent.parent / name).read_bytes())
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    modes = (args.modo,) if args.modo else MODES
    replicates = (args.replica,) if args.replica is not None else range(len(config['seeds']))
    with threadpool_limits(limits=args.threads):
        if args.fase in ('fits', 'tutto'):
            keys = [args.fit] if args.fit else sorted(plan['fits'], key=lambda key: (-len(plan['fits'][key]['excluded']), key))
            for replicate in replicates:
                for mode in modes:
                    for key in keys:
                        fit_one(root, config, plan, mode, replicate, key)
                    if not args.fit:
                        score_replicate(root, config, plan, mode, replicate)
                        collect_report(root, config)
        if args.fase == 'score':
            for replicate in replicates:
                for mode in modes:
                    score_replicate(root, config, plan, mode, replicate)
            collect_report(root, config)
        if args.fase in ('ablazioni', 'tutto'):
            for preset in ABLATIONS:
                mode = f'ab_{preset}'
                for outer in plan['outer']:
                    fit_one(root, config, plan, mode, 0, outer['refit'], preset=preset)
                score_replicate(root, config, plan, mode, 0, ablation=True)
            collect_report(root, config)
        if args.fase in ('finale', 'tutto'):
            validate_results(root, config, plan)
            finalize(root, config, plan)
            validate_results(root, config, plan, final=True)
        if args.fase == 'valida':
            validate_results(root, config, plan, final=True)


if __name__ == '__main__':
    main()
