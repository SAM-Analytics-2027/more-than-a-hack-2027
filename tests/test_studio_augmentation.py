"""Resume, provenienza e copertura del runner, senza training o dati di gara."""

import contextlib
import copy
from dataclasses import asdict
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modelli_mv'))
import augmentation_realiste as A
import dati
import protocollo_augmentation as Q
import studio_augmentation as S
import valida_risultati_augmentation as V


class StudyIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'study'
        self.root.mkdir()
        self.plan = Q.build_plan(dati.external_matches())
        self.config = {'seeds': [2027], 'smoke': False}
        S.P._write(self.root / 'config.json', self.config)

    def write_fit(self, key, mode='base'):
        folder = S.fit_path(self.root, mode, 0, key)
        spec = self.plan['fits'][key]
        train = copy.deepcopy(spec['train'])
        provenance = {field: copy.deepcopy(train) for field in ('train', 'normalization_train', 'label_train')}
        provenance.update(generator_train=train if mode == 'risporcata' else [],
                          early_stopping=False, seed=2027, augmented=mode == 'risporcata',
                          preset=asdict(S.C.PRESETS['base']), excluded=spec['excluded'])
        probabilities = {tuple(it): np.tile([.5, .3, .2], (4, 1)) for it in spec['test']}
        S.save_probabilities(folder / 'probabilita.npz', probabilities)
        record = {'provenance': provenance, 'artifacts': {'probabilita.npz': S.P._sha256(folder / 'probabilita.npz')}}
        S.P._write(folder / 'completato.json', record)
        return folder, record

    def write_decoder(self, fold=0):
        outer = self.plan['outer'][fold]
        for key in outer['inner_sources']:
            self.write_fit(key)
        params = dict(S.K.DEFAULT)
        record = {'params': params, 'params_sha256': S.A_digest(params),
                  'calibration': outer['calibration'], 'excluded_outer_fold': fold,
                  'inner_sources': outer['inner_sources'], 'input_hashes': {
                      key: S.P._sha256(S.fit_path(self.root, 'base', 0, key) / 'completato.json')
                      for key in outer['inner_sources']}}
        path = self.root / 'decoders' / 'base' / 'r0' / f'fold{fold}.json'
        S.P._write(path, record)
        return outer, path, record

    def test_fit_resume_rejects_different_seed_components_and_changed_probabilities(self):
        key = 'excl_0_1'
        folder, record = self.write_fit(key)
        self.assertEqual(S.verify_fit(self.root, self.plan, 'base', 0, key), record)
        for field, value in [('seed', 12027), ('augmented', True),
                             ('preset', asdict(S.C.PRESETS['feature'])), ('excluded', [0, 2])]:
            bad = copy.deepcopy(record)
            bad['provenance'][field] = value
            S.P._write(folder / 'completato.json', bad)
            with self.assertRaises(ValueError):
                S.verify_fit(self.root, self.plan, 'base', 0, key)
        S.P._write(folder / 'completato.json', record)
        (folder / 'probabilita.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            S.verify_fit(self.root, self.plan, 'base', 0, key)

    def test_augmented_fit_rejects_a_profile_trained_on_test_game(self):
        key = 'excl_0'
        folder, record = self.write_fit(key, 'risporcata')
        self.assertTrue(S.verify_fit(self.root, self.plan, 'risporcata', 0, key))
        record['provenance']['generator_train'].append(self.plan['fits'][key]['test'][0])
        S.P._write(folder / 'completato.json', record)
        with self.assertRaisesRegex(ValueError, 'generatore'):
            S.verify_fit(self.root, self.plan, 'risporcata', 0, key)

    def test_decoder_resume_rejects_changed_params_or_input_fit(self):
        outer, path, record = self.write_decoder()
        self.assertEqual(S.checked_decoder(self.root, self.plan, 'base', 0, outer), record['params'])
        damaged = copy.deepcopy(record)
        damaged['params']['stop'] += 1
        S.P._write(path, damaged)
        with self.assertRaisesRegex(ValueError, 'alterati'):
            S.checked_decoder(self.root, self.plan, 'base', 0, outer)
        S.P._write(path, record)
        key = next(iter(outer['inner_sources']))
        done = S.fit_path(self.root, 'base', 0, key) / 'completato.json'
        changed = S.read_json(done)
        changed['elapsed_seconds'] = 123
        S.P._write(done, changed)
        with self.assertRaisesRegex(ValueError, 'calibrazione'):
            S.checked_decoder(self.root, self.plan, 'base', 0, outer)

    def test_completion_requires_all_nested_fit_artifacts(self):
        for mode in S.MODES:
            for key in self.plan['fits']:
                self.write_fit(key, mode)
        S.verify_complete(self.root, self.config, self.plan)
        removed = S.fit_path(self.root, 'risporcata', 0, 'excl_1_2') / 'completato.json'
        removed.unlink()
        with self.assertRaises(FileNotFoundError):
            S.verify_complete(self.root, self.config, self.plan)

    def test_validation_rejects_equal_truncation_of_all_score_reports(self):
        for mode in S.MODES:
            item = self.plan['items'][0]
            row = {'mode': mode, 'replicate': 0, 'fold': self.plan['fold_of']['/'.join(item)],
                   'source': item[0], 'match': item[1], 'spellF1': .7, 'macroF1': .93}
            report = {'mode': mode, 'replicate': 0, 'diagnostic_ablation': False,
                      'dashboard_model': mode, 'rows': [row],
                      'decoders': {str(k): {} for k in range(5)},
                      'mean10': {'spellF1': .7, 'macroF1': .93}}
            S.P._write(self.root / 'scores' / f'{mode}_r0.json', report)
        with patch.object(S, 'verify_complete'), patch.object(V, '_dashboard'):
            with self.assertRaisesRegex(ValueError, 'Copertura'):
                S.validate_results(self.root, self.config, self.plan)

    def test_reused_inference_weights_are_bound_to_source_manifest(self):
        source, source_record = self.write_fit('excl_0')
        folder, record = self.write_fit('excl_1')
        record['reused_weights'] = {'path': str(source),
                                    'manifest_sha256': S.P._sha256(source / 'completato.json')}
        S.verify_manifest(folder, record)
        source_record['elapsed_seconds'] = 1
        S.P._write(source / 'completato.json', source_record)
        with self.assertRaisesRegex(ValueError, 'riutilizzati'):
            S.verify_manifest(folder, record)


class StudyMainResumeTests(unittest.TestCase):
    def test_changed_clean_source_refuses_resume_before_training_or_target_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {'clean': 'first'}
            def manifest(item, labels=False):
                return {'clean_atd.txt': state['clean'], 'dirty_atd.txt': 'fixed', 'gt.csv': 'fixed'}
            argv = ['studio_augmentation.py', '--nome', 'synthetic', '--fase', 'fits',
                    '--modo', 'base', '--replica', '0', '--fit', 'excl_0', '--smoke']
            with patch.object(sys, 'argv', argv), patch.object(S.dati, 'CACHE', tmp), \
                    patch.object(S.dati, 'input_hash', return_value='dirty-and-gt-fixed'), \
                    patch.object(A, 'input_manifest', side_effect=manifest), \
                    patch.object(S, 'source_hashes', return_value={}), \
                    patch.object(S.R, '_versions', return_value={}), \
                    patch.object(S.rete, 'device', return_value='cpu'), \
                    patch.object(S.torch, 'set_num_threads'), \
                    patch.object(S, 'fit_one') as fit, \
                    patch.object(S.R, 'predict', side_effect=AssertionError('target read')), \
                    patch.object(S.R, '_public_window', side_effect=AssertionError('public GT read')):
                S.main()
                fit.assert_called_once()
                state['clean'] = 'changed'
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    S.main()
                self.assertEqual(error.exception.code, 2)
                fit.assert_called_once()


if __name__ == '__main__':
    unittest.main()
