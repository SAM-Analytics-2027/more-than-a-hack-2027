"""Seleziona la ricetta su dati esterni, allena sullo sporcato e infine predice il Barça.

Il confronto esterno usa sempre il Viterbi del rispettivo fold, calibrato senza
le partite giudicate. Il Viterbi finale viene ricalibrato sulle probabilità OOF
esterne e NON viene usato per riportare un nuovo punteggio su quelle stesse gare.
"""

import argparse
import json
import shutil
from pathlib import Path
from statistics import median

import joblib
import torch
from threadpoolctl import threadpool_limits

import augmentation as A
import constants as K
import dati
import esperimento as E
import prova_augmentation as P
import rete
import riproduci as R


def _calibrate(root, comparison, folds, items, mode, macro_floor):
    path = root / 'parametri_viterbi.json'
    if path.exists():
        return
    print('Taratura finale del Viterbi sulle predizioni fuori gruppo DFL + Metrica sporcate', flush=True)
    probs = {}
    for fold in folds:
        held_out = {tuple(it) for it in fold['test'] if it[0] in K.DM}
        cached = P.load_probabilities(comparison / f"fold{fold['fold']}" / mode / 'probabilita.npz')
        probs.update({key: value for key, value in cached.items() if key[0] in held_out})
    tactical = [it for it in items if it[0] in K.DM]
    params = P._calibrate(probs, P._meta(tactical), tactical, rounds=2, macro_floor=macro_floor)
    P._write(path, {
        'params': params,
        'origine': 'Predizioni OOF esterne, ricetta selezionata, tracking sporcato senza stress aggiuntivo.',
        'uso': 'Inferenza finale; non usato per il benchmark esterno sul quale è calibrato.',
    })


def _fit(folder, items, cols, iterations, epochs, seeds, device, mode):
    folder.mkdir(exist_ok=True)
    tracking, focused = P.MODES[mode]
    tree_path = folder / 'hgb.joblib'
    if not tree_path.exists():
        print(f'HGB finale: {iterations} iterazioni, {len(items)} partite esterne sporcate', flush=True)
        x, y, w = P._rows(items, cols, tracking=tracking)
        tree = E.make_model().set_params(max_iter=iterations, early_stopping=False)
        tree.fit(x, y, sample_weight=w)
        tmp = tree_path.with_suffix('.tmp')
        joblib.dump(tree, tmp, compress=3)
        tmp.replace(tree_path)
        del x, y, w, tree

    prepared = None
    for i, seed in enumerate(seeds):
        path = folder / f'rete_seme{i}.pt'
        if path.exists():
            continue
        if prepared is None:
            prepared = P._sequences(items, cols, tracking=tracking)
        print(f'Rete finale {i + 1}/{len(seeds)}', flush=True)
        model = P._train_net(prepared, cols, seed, device, epochs, focused=focused, smoke=False)
        tmp = path.with_suffix('.tmp')
        rete.save(model, tmp)
        tmp.replace(path)
        del model
    P._write(folder / 'modello.json', {
        'partite': ['/'.join(it) for it in items], 'semi_rete': seeds,
        'colonne': cols, 'dispositivo_allenamento': str(device), 'feature_version': 2,
        'ricetta': mode, 'tracking': 'sporcato', 'pesi_fonti': K.SOURCE_WEIGHTS,
    })


def _export_benchmark(comparison, folds, run_name, mode, model_name):
    """Rigenera il benchmark annidato dai suoi artefatti, poi crea l'alias finale."""
    for fold in folds:
        P._export_fold(comparison / f"fold{fold['fold']}", fold, run_name)
    source = Path(R.EXT_RESULTS) / f'{run_name}__{mode}'
    destination = Path(R.EXT_RESULTS) / model_name
    if source.resolve() == destination.resolve():
        return
    description = {'protocollo': P.PROTOCOL_VERSION, 'esperimento': run_name, 'modo': mode,
                   'alias_di': source.name, 'tracking': 'sporcato',
                   'validazione': 'train_calibrazione_test_disgiunti'}
    manifest = destination / 'provenienza.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != description:
        raise ValueError(f'Nome dashboard già usato da un altro esperimento: {destination}')
    if destination.exists() and not manifest.exists() and any(destination.iterdir()):
        raise ValueError(f'Cartella dashboard esistente priva di provenienza: {destination}')
    P._write(manifest, description)
    for path in source.glob('*.csv'):
        shutil.copyfile(path, destination / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--esperimento', default='confronto_augmentation')
    parser.add_argument('--nome', help='nome nuovo; default <esperimento>__<ricetta selezionata>')
    parser.add_argument('--modo', choices=list(P.MODES),
                        help='ricetta esplicita già valutata; default selezione automatica sulle esterne')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--solo-allena', action='store_true', help='fissa il modello senza leggere il match di gara')
    args = parser.parse_args()
    if not P.valid_name(args.esperimento) or (args.nome is not None and not P.valid_name(args.nome)):
        parser.error('nome esperimento/modello non valido')
    if args.threads < 1:
        parser.error('--threads deve essere positivo')
    comparison = Path(dati.CACHE) / 'augmentation_runs' / args.esperimento
    config = json.loads((comparison / 'config.json').read_text(encoding='utf-8'))
    report = json.loads((comparison / 'risultati.json').read_text(encoding='utf-8'))
    items = sorted(dati.external_matches())
    try:
        folds = P.validate_completed_report(report, config, items)
        if (P.cache_fingerprints(items) != config['cache']
                or K.SOURCE_WEIGHTS != config['pesi_fonti']
                or P.code_fingerprints() != config['codice']):
            raise ValueError('Dati, codice o pesi delle fonti differiscono dal confronto')
        for fold in folds:
            P._verify_artifacts(comparison / f"fold{fold['fold']}", fold)
    except ValueError as exc:
        parser.error(str(exc))
    mode, candidate_scores = P.select_mode(folds, config['modi'], config['macro_floor'])
    if args.modo is not None:
        if args.modo not in config['modi']:
            parser.error('la ricetta richiesta non è stata valutata nel confronto')
        mode = args.modo
    name = args.nome or f'{args.esperimento}__{mode}'
    if name == R.NAME:
        parser.error('il riferimento congelato non può essere sovrascritto')
    print(f'Ricetta scelta sulle esterne: {mode}. Modello: {name}', flush=True)
    tracking, _ = P.MODES[mode]
    tree_name = 'hgb_tracking.joblib' if tracking else 'hgb_base.joblib'
    trees = [joblib.load(comparison / f"fold{fold['fold']}" / tree_name) for fold in folds]
    iterations = int(median(tree.n_iter_ for tree in trees))
    del trees
    device = rete.device()
    root = Path(R.model_dir(name))
    recipe = {
        'protocollo': P.PROTOCOL_VERSION, 'confronto': args.esperimento,
        'config_confronto': config, 'ricetta_selezionata': mode,
        'selezione': 'esplicita_su_confronto_esterno' if args.modo else 'automatica_su_confronto_esterno',
        'metriche_selezione': candidate_scores, 'hgb_iterazioni': iterations,
        'report_sha256': P._sha256(comparison / 'risultati.json'),
        'codice_finale_sha256': P._sha256(Path(__file__)),
        'dispositivo': str(device), 'versioni': R._versions(), 'feature_version': 2,
    }
    recipe_path = root / 'ricetta.json'
    if recipe_path.exists():
        if recipe != json.loads(recipe_path.read_text(encoding='utf-8')):
            parser.error(f'ricetta differente da quella già salvata in {root}; usa un nuovo nome')
    elif root.exists() and any(root.iterdir()):
        parser.error(f'cartella modello esistente senza provenienza: {root}; usa un nuovo nome')
    root.mkdir(parents=True, exist_ok=True)
    P._write(recipe_path, recipe)
    A.prepare(items, workers=1)
    cols = E.feature_cols(A.load(items[0]))
    seeds = [K.SEED + 1000 * i for i in range(config['n_reti'])]
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        _calibrate(root, comparison, folds, items, mode, config['macro_floor'])
        _fit(root / 'finale', items, cols, iterations, config['epoche'], seeds, device, mode)
        _export_benchmark(comparison, folds, args.esperimento, mode, name)
        P._write(root / 'benchmark_esterne.json', {
            'esperimento': args.esperimento, 'modo': mode, 'nome_dashboard': name,
            'sintesi': candidate_scores[mode],
            'nota': 'OOF con calibrazione esclusa dal test di ciascun fold; CV di sviluppo, '
                    'non stima su un test indipendente dalla selezione della ricetta.',
        })
        artifact_paths = [root / 'parametri_viterbi.json', *sorted((root / 'finale').iterdir())]
        P._write(root / 'manifest.json', {
            p.relative_to(root).as_posix(): P._sha256(p) for p in artifact_paths if p.is_file()
        })
        if not args.solo_allena:
            print('Modello, ricetta e Viterbi fissati. Inferenza su Barça–Athletic.', flush=True)
            R.predict(root, name, write=True)


if __name__ == '__main__':
    main()
