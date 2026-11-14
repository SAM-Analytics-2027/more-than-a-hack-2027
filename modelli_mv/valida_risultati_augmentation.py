"""Convalida artefatti augmentation e CSV con il giudice ufficiale e la dashboard.

Non allena e non calibra. Le ground truth lette sono esclusivamente quelle
esterne. Il controllo pubblico dei cinque minuti richiede --pubblico esplicito.

    python modelli_mv/valida_risultati_augmentation.py --esperimento RUN
    python modelli_mv/valida_risultati_augmentation.py --esperimento RUN --modelli MODEL
"""

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import numpy as np

import constants as K
import dati
import prova_augmentation as P
import riproduci as R

TOLERANCE = 1e-10


def _dashboard():
    """Carica soltanto funzioni: nessuna scansione della GT pubblica all'import."""
    spec = importlib.util.spec_from_file_location('augmentation_dashboard_validation',
                                                  Path(__file__).resolve().parents[1] / 'dashboard' / 'app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_csv(path, gt_path, xml_path, expected, dashboard=None):
    """Giudica il CSV su disco usando il percorso indipendente della dashboard."""
    dashboard = dashboard or _dashboard()
    pm = dashboard.pm
    ground_truth = pm.read_label_csv(gt_path)
    problem = pm.validate_submission(ground_truth, len(ground_truth))
    if problem:
        raise ValueError(f'Ground truth esterna non valida: {problem}')
    frame = pm.read_label_csv(path)
    problem = pm.validate_submission(frame, len(ground_truth))
    if problem:
        raise ValueError(f'CSV non valido: {problem}')
    gt, pred = pm.label_codes(ground_truth), pm.label_codes(frame)
    halves = dashboard._ext_halves(str(xml_path), len(gt))
    actual = dashboard._ext_compare(gt, pred, halves, K.FPS)
    for report_key, score_key in (('spellF1', 'spell_f1_mean'), ('macroF1', 'macro_f1')):
        if not np.isclose(actual[score_key], expected[report_key], rtol=0, atol=TOLERANCE):
            raise ValueError(f'{report_key}: CSV/dashboard={actual[score_key]:.12f}, '
                             f'report={expected[report_key]:.12f}')
    if actual['n_spells'] != expected['predetti'] or actual['spells_matched_50'] != expected['abbinati50']:
        raise ValueError('Conteggi dei possessi o abbinamenti diversi dal report')
    gt_summary = dashboard._ext_gt_summary(gt, halves, K.FPS)
    if gt_summary['n_spells'] != expected['possessi']:
        raise ValueError('Conteggio dei possessi GT diverso dal report')
    return {'ok': True, 'rows': len(frame), 'sha256': P._sha256(path),
            'spellF1': float(actual['spell_f1_mean']), 'macroF1': float(actual['macro_f1']),
            'possessi': int(gt_summary['n_spells']), 'predetti': int(actual['n_spells']),
            'abbinati50': int(actual['spells_matched_50'])}


def validate_experiment(name):
    comparison = Path(dati.CACHE) / 'augmentation_runs' / name
    config = json.loads((comparison / 'config.json').read_text(encoding='utf-8'))
    report = json.loads((comparison / 'risultati.json').read_text(encoding='utf-8'))
    items = sorted(dati.external_matches())
    folds = P.validate_completed_report(report, config, items)
    if (P.cache_fingerprints(items) != config['cache']
            or P.code_fingerprints() != config['codice']
            or K.SOURCE_WEIGHTS != config['pesi_fonti']):
        raise ValueError('Cache, codice o pesi delle fonti diversi dal confronto')
    summary = P._summary(folds, config['modi'])
    if summary != report['sintesi']:
        raise ValueError('La sintesi non coincide con i risultati dei fold')
    for fold in folds:
        P._verify_artifacts(comparison / f"fold{fold['fold']}", fold)

    dashboard = _dashboard()
    rows, per_mode = [], {}
    for mode in config['modi']:
        expected_rows = [row for fold in folds for row in fold['modi'][mode]
                         if row['vista'] == P.CALIBRATION_VIEW and row['fonte'] in K.DM]
        for expected in expected_rows:
            src, mid = expected['fonte'], expected['partita']
            clean = Path(dati.BASE) / f'{src}_pulite' / mid
            path = Path(R.EXT_RESULTS) / f'{name}__{mode}' / f'{mid}.csv'
            try:
                result = validate_csv(path, clean / f'{mid}_possession_gt.csv',
                                      clean / f'{mid}_atd.xml', expected, dashboard)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
            rows.append({'mode': mode, 'source': src, 'match': mid, **result})
        mode_rows = [row for row in rows if row['mode'] == mode and row['ok']]
        complete = len(mode_rows) == len(expected_rows)
        per_mode[mode] = {
            'ok': complete, 'expected_csv': len(expected_rows), 'valid_csv': len(mode_rows),
            'spellF1': float(np.mean([row['spellF1'] for row in mode_rows])) if mode_rows else None,
            'macroF1': float(np.mean([row['macroF1'] for row in mode_rows])) if mode_rows else None,
        }
        if complete:
            for metric in ('spellF1', 'macroF1'):
                target = report['sintesi'][mode][P.CALIBRATION_VIEW]['tattiche'][metric]
                if not np.isclose(per_mode[mode][metric], target, rtol=0, atol=TOLERANCE):
                    raise ValueError(f'Media {metric} non coerente per {mode}')
    return {'ok': all(row['ok'] for row in rows), 'folds': len(folds),
            'expected_csv': len(rows), 'valid_csv': sum(row['ok'] for row in rows),
            'modes': per_mode, 'csv': rows, 'report_sha256': P._sha256(comparison / 'risultati.json')}


def validate_model(name, *, public=False):
    """Rigenera il CSV e confronta SHA-256; in caso di errore ripristina l'originale."""
    root = Path(R.model_dir(name))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    P._verify_artifacts(root, {'artefatti': manifest})
    csv = Path(R.RESULTS) / f'{name}.csv'
    if not csv.is_file():
        raise ValueError(f'CSV finale mancante: {csv}')
    expected = P._sha256(csv)
    expected_manifest = root / 'attesi.json'
    if expected_manifest.exists():
        recorded = json.loads(expected_manifest.read_text(encoding='utf-8'))['csv_sha256']
        if recorded != expected:
            raise ValueError('Il CSV esistente differisce dalla sua impronta attesa')
    with NamedTemporaryFile(dir=csv.parent, prefix=csv.stem + '_validation_', suffix='.tmp', delete=False) as handle:
        backup = Path(handle.name)
    shutil.copyfile(csv, backup)
    valid = False
    try:
        # Cattura log del runner: la CLI di validazione stampa solo il riepilogo.
        with contextlib.redirect_stdout(io.StringIO()):
            result = R.predict(root=root, name=name, write=False, report_public=public)
        actual = P._sha256(csv)
        if actual != expected:
            raise ValueError(f'CSV non riproducibile: atteso {expected}, rigenerato {actual}')
        valid = True
    finally:
        if not valid:
            backup.replace(csv)
        else:
            backup.unlink(missing_ok=True)
    output = {'ok': True, 'csv_sha256': expected, 'reproduced_identically': True,
              'public_checked': public, 'artifact_files': len(manifest)}
    if public:
        output['finestra_pubblica'] = result['finestra_pubblica']
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--esperimento', required=True)
    parser.add_argument('--modelli', nargs='*', default=[])
    parser.add_argument('--pubblico', action='store_true',
                        help='controllo finale pubblico esplicito, mai scelta o tuning')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not P.valid_name(args.esperimento) or not all(P.valid_name(name) for name in args.modelli):
        parser.error('nome esperimento o modello non valido')
    if args.pubblico and not args.modelli:
        parser.error('--pubblico richiede almeno un modello finale')
    output = args.output or Path(dati.CACHE) / 'augmentation_runs' / args.esperimento / 'validazione_risultati.json'
    result = {'schema': 'augmentation-results-validation-v1', 'experiment': args.esperimento,
              'tolerance': TOLERANCE, 'public_requested': args.pubblico}
    try:
        result['external'] = validate_experiment(args.esperimento)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['external'] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    result['models'] = {}
    for name in args.modelli:
        try:
            result['models'][name] = validate_model(name, public=args.pubblico)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            result['models'][name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    result['ok'] = result['external']['ok'] and all(model['ok'] for model in result['models'].values())
    P._write(output, result)
    external = result['external']
    print(f"Validazione {'OK' if result['ok'] else 'FALLITA'}: "
          f"CSV esterni {external.get('valid_csv', 0)}/{external.get('expected_csv', 0)}, "
          f"modelli riproducibili {sum(model['ok'] for model in result['models'].values())}/{len(args.modelli)}.",
          flush=True)
    if 'modes' in external:
        print(json.dumps(external['modes'], indent=2), flush=True)
    print(f'Report: {output}', flush=True)
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
