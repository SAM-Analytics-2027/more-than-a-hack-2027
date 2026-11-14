"""Regressions for exact external exports and the final-model public-check lock."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modelli_mv'))
import valida_csv_studio as V


class ExactCSVTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'match.csv'
        self.period = np.array([0] + [1] * 18 + [0])
        self.no_data = np.zeros(20, dtype=bool)
        self.probabilities = np.tile([.98, .01, .01], (20, 1))
        self.codes = np.array([3] + [0] * 18 + [3])
        self.digest = V.S.R._write_csv(str(self.path), self.codes)
        self.params = dict(V.S.K.DEFAULT)

    def check(self, digest=None, probabilities=None):
        return V.validate_one_csv(self.path, self.digest if digest is None else digest,
                                  self.probabilities if probabilities is None else probabilities,
                                  self.period, self.no_data, self.params)

    def test_exact_csv_is_verified_without_rewriting(self):
        before = self.path.stat().st_mtime_ns
        result = self.check()
        self.assertTrue(result['reproduced_identically'])
        self.assertEqual(result['decoded_sha256'], self.digest)
        self.assertEqual(self.path.stat().st_mtime_ns, before)

    def test_modified_csv_is_rejected_by_recorded_hash(self):
        self.path.write_bytes(self.path.read_bytes().replace(b'1,H', b'1,A', 1))
        with self.assertRaisesRegex(ValueError, 'Impronta'):
            self.check()

    def test_wrong_predictions_are_rejected_even_if_hash_is_updated(self):
        bad = self.codes.copy()
        bad[2] = 1
        digest = V.S.R._write_csv(str(self.path), bad)
        with self.assertRaisesRegex(ValueError, 'etichette'):
            self.check(digest)

    def test_wrong_frames_are_rejected_even_if_hash_is_updated(self):
        frame = pd.read_csv(self.path)
        frame.loc[1, 'frame'] = 100
        frame.to_csv(self.path, index=False, lineterminator='\n')
        with self.assertRaisesRegex(ValueError, 'Frame'):
            self.check(V.S.P._sha256(self.path))

    def test_equivalent_labels_with_different_csv_bytes_are_rejected(self):
        self.path.write_bytes(self.path.read_bytes().replace(b'\n', b'\r\n'))
        with self.assertRaisesRegex(ValueError, 'byte per byte'):
            self.check(V.S.P._sha256(self.path))

    def test_invalid_probabilities_are_rejected(self):
        for bad in (self.probabilities[:-1], self.probabilities * 2,
                    np.full_like(self.probabilities, np.nan)):
            with self.subTest(shape=bad.shape), self.assertRaisesRegex(ValueError, 'Probabilità'):
                self.check(probabilities=bad)


class StudyCSVTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'study'
        self.root.mkdir()
        self.destination = Path(self.tmp.name) / 'csv'
        items = [(source, f'{source}{fold}') for source in ('dfl', 'metrica', 'skillcorner')
                 for fold in range(5)]
        self.plan = V.S.Q.build_plan(items, {it: int(it[1][-1]) for it in items})
        self.config = {'smoke': False, 'sources': {}, 'seeds': list(V.S.REPLICATE_SEEDS),
                       'ablations': list(V.S.ABLATIONS), 'ablation_decoder': 'nested_base_same_fold',
                       'original_inputs': {'/'.join(it): 'input' for it in items},
                       'clean_and_dirty_inputs': {'/'.join(it): {} for it in items}}
        V.S.P._write(self.root / 'config.json', self.config)
        V.S.P._write(self.root / 'piano.json', self.plan)
        self.params = dict(V.S.K.DEFAULT)
        self.probabilities = np.tile([.98, .01, .01], (3, 1))
        self.decoder_calls = []
        modes = [(mode, rep) for mode in V.S.MODES for rep in range(3)]
        modes += [(f'ab_{name}', 0) for name in V.S.ABLATIONS]
        for mode, rep in modes:
            name = V.S.model_name(self.root, mode, rep)
            rows = []
            for item in self.plan['items']:
                row = dict(mode=mode, replicate=rep, source=item[0], match=item[1],
                           fold=self.plan['fold_of']['/'.join(item)], spellF1=.7, macroF1=.93)
                if item[0] in V.S.K.DM:
                    row['csv_sha256'] = V.S.R._write_csv(
                        str(self.destination / name / f'{item[1]}.csv'), np.zeros(3, dtype=np.int8))
                rows.append(row)
            report = dict(mode=mode, replicate=rep, diagnostic_ablation=mode.startswith('ab_'),
                          dashboard_model=name, rows=rows, mean10={'spellF1': .7, 'macroF1': .93},
                          decoders={str(i): self.params for i in range(5)})
            V.S.P._write(self.root / 'scores' / f'{mode}_r{rep}.json', report)
        patches = [patch.object(V.S, 'source_hashes', return_value={}),
                   patch.object(V.S.dati, 'input_hash', return_value='input'),
                   patch.object(V.A, 'input_manifest', return_value={}),
                   patch.object(V.S, 'verify_complete'), patch.object(V.S, 'verify_fit'),
                   patch.object(V.S, 'checked_decoder', side_effect=self.decoder),
                   patch.object(V.S, 'load_probabilities', side_effect=self.predictions),
                   patch.object(V.S, 'load_original', side_effect=lambda it: pd.DataFrame(
                       {'frame': np.arange(3), 'period': np.ones(3, dtype=int),
                        'n_casa': np.ones(3), 'n_ospiti': np.ones(3)})),
                   patch.object(V.S.R, 'EXT_RESULTS', str(self.destination)),
                   patch.object(V.S.R, '_public_window', side_effect=AssertionError('public GT read'))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def decoder(self, root, plan, mode, rep, outer):
        self.decoder_calls.append((mode, rep, outer['fold']))
        return self.params

    def predictions(self, path):
        fit = self.plan['fits'][path.parent.name]
        return {tuple(it): self.probabilities for it in fit['test']}

    def test_all_120_csv_and_fixed_base_decoders_for_ablations(self):
        result = V.validate_exported_csv(self.root)
        self.assertEqual(result['csv_checked'], 120)
        self.assertEqual(result['models_checked'], 12)
        self.assertFalse(result['public_ground_truth_read'])
        self.assertEqual(sum(mode == 'base' for mode, _, _ in self.decoder_calls), 45)
        self.assertEqual(sum(mode == 'risporcata' for mode, _, _ in self.decoder_calls), 15)
        V.S.verify_complete.assert_called_once()
        self.assertEqual(V.S.verify_fit.call_count, 60)

    def test_missing_ablation_report_refuses_completion(self):
        (self.root / 'scores' / 'ab_feature_r0.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Copertura report'):
            V.validate_exported_csv(self.root)

    def test_wrong_ablation_decoder_is_rejected(self):
        path = self.root / 'scores' / 'ab_feature_r0.json'
        report = V.S.read_json(path)
        report['decoders']['0']['stop'] += 1
        V.S.P._write(path, report)
        with self.assertRaisesRegex(ValueError, 'Decoder'):
            V.validate_exported_csv(self.root)

    def test_changed_input_is_rejected_before_reading_predictions(self):
        self.config['original_inputs']['dfl/dfl0'] = 'different'
        V.S.P._write(self.root / 'config.json', self.config)
        with self.assertRaisesRegex(ValueError, 'Input esterni'):
            V.validate_exported_csv(self.root)
        V.S.load_probabilities.assert_not_called()


class FinalLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'study'
        self.root.mkdir()
        self.csv_root = Path(self.tmp.name) / 'csv'
        self.model_root = Path(self.tmp.name) / 'models'
        lock, validation = {}, {'ok': True, 'models': {}}
        for mode in V.S.MODES:
            name = V.S.model_name(self.root, mode, 0)
            folder = self.model_root / name
            folder.mkdir(parents=True)
            (folder / 'weights').write_bytes(b'fixed model weights')
            V.S.P._write(folder / 'manifest.json', {'weights': V.S.P._sha256(folder / 'weights')})
            digest = V.S.R._write_csv(str(self.csv_root / f'{name}.csv'), np.zeros(3, dtype=np.int8))
            V.S.P._write(folder / 'attesi.json', {'csv_sha256': digest})
            lock[mode] = {'name': name, 'manifest_sha256': V.S.P._sha256(folder / 'manifest.json')}
            validation['models'][name] = {'ok': True, 'reproduced_identically': True, 'csv_sha256': digest}
        V.S.P._write(self.root / 'modelli_fissati_prima_del_pubblico.json', lock)
        V.S.P._write(self.root / 'validazione.json', validation)
        for item in [patch.object(V.S.R, 'model_dir', side_effect=lambda name: self.model_root / name),
                     patch.object(V.S.R, 'RESULTS', str(self.csv_root)),
                     patch.object(V.S.R, '_public_window', side_effect=AssertionError('public GT read'))]:
            item.start()
            self.addCleanup(item.stop)

    def test_before_after_snapshot_requires_unchanged_artifacts(self):
        before = V.verify_final_lock(self.root)
        self.assertEqual(V.verify_final_lock(self.root, expected_snapshot=before), before)
        modified = copy.deepcopy(before)
        modified['validation_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'durante'):
            V.verify_final_lock(self.root, expected_snapshot=modified)

    def test_weight_modification_is_rejected(self):
        name = V.S.model_name(self.root, 'base', 0)
        (self.model_root / name / 'weights').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            V.verify_final_lock(self.root)

    def test_csv_modification_is_rejected(self):
        name = V.S.model_name(self.root, 'risporcata', 0)
        V.S.R._write_csv(str(self.csv_root / f'{name}.csv'), np.ones(3, dtype=np.int8))
        with self.assertRaisesRegex(ValueError, 'CSV finale'):
            V.verify_final_lock(self.root)

    def test_incomplete_final_lock_is_rejected(self):
        path = self.root / 'modelli_fissati_prima_del_pubblico.json'
        value = V.S.read_json(path)
        del value['base']
        V.S.P._write(path, value)
        with self.assertRaisesRegex(ValueError, 'incompleti'):
            V.verify_final_lock(self.root)


if __name__ == '__main__':
    unittest.main()
