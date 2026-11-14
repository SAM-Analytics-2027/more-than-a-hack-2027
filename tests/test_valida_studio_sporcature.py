"""Independent validator checks with synthetic cache files only."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modelli_mv'))
import augmentation_realiste as A
import constants as K
import dati
import valida_studio_sporcature as V


class CorruptionCacheValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'synthetic.parquet'
        self.item = dati.external_matches()[0]
        self.manifest = {'seed': 2027, 'replica': 0, 'item': list(self.item)}
        n = 12
        self.clean = dict(period=np.ones(n, np.int8), team=np.array([0, 0, 1, 1]),
            px=np.tile([-40., -10., 10., 40.], (n, 1)).astype(np.float32),
            py=np.tile([-5., 3., -2., 6.], (n, 1)).astype(np.float32),
            bx=np.arange(n, dtype=np.float32), by=np.full(n, 3., np.float32),
            length=105., width=68., fps=25, labels=np.full(n, 'H'))
        self.frame = A.C.features(self.clean, legacy=True)
        self.frame.insert(0, 'y', np.full(n, K.H, np.int8))
        self.frame.insert(0, 'period', self.clean['period'])
        self.frame.insert(0, 'frame', np.arange(n, dtype=np.int32))
        self.frame['fonte_tv'] = np.float32(self.item[0] == 'skillcorner')
        self.original = self.frame.copy(deep=True)
        self.diagnostics = A.diagnostics(self.clean, self.clean)
        self.write()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(A, 'cache_path', return_value=self.path))
        self.stack.enter_context(patch.object(A, 'view_manifest', return_value=self.manifest))
        self.stack.enter_context(patch.object(V, '_original_info', return_value=(
            self.original[['frame', 'period', 'y']], list(self.original.columns))))
        self.stack.enter_context(patch.object(A.C.fc, 'read_epts', return_value=self.clean))
        self.stack.enter_context(patch.object(A, 'transform', return_value=self.clean))

    def write(self):
        self.frame.to_parquet(self.path, index=False)
        record = {'manifest': self.manifest, 'parquet_sha256': A._hash(self.path),
                  'diagnostics': self.diagnostics}
        self.path.with_suffix('.json').write_text(json.dumps(record), encoding='utf-8')

    def test_valid_existing_cache_and_exact_recomputation(self):
        result = V.validate_view(self.item, {}, 2027, 0, recompute=True)
        self.assertEqual(result['rows'], len(self.frame))
        self.assertTrue(result['recomputed'])
        self.assertEqual(result['diagnostics'], self.diagnostics)

    def test_missing_generated_view_is_never_regenerated(self):
        self.path.unlink()
        with patch.object(A, 'load_view', side_effect=AssertionError('must not regenerate')) as loader:
            with self.assertRaisesRegex(ValueError, 'mancante'):
                V.validate_view(self.item, {}, 2027, 0)
        loader.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_hash_mismatch_fails_before_reading_table(self):
        self.path.write_bytes(b'changed')
        with patch.object(pd, 'read_parquet', side_effect=AssertionError('must not parse bad hash')):
            with self.assertRaisesRegex(ValueError, 'alterata'):
                V.validate_view(self.item, {}, 2027, 0)

    def test_wrong_seed_provenance_rejected_even_with_valid_table_hash(self):
        record = json.loads(self.path.with_suffix('.json').read_text())
        record['manifest']['seed'] = 999
        self.path.with_suffix('.json').write_text(json.dumps(record), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'alterata'):
            V.validate_view(self.item, {}, 2027, 0)

    def test_changed_ground_truth_rejected_even_with_new_hash(self):
        self.frame.loc[3, 'y'] = K.A
        self.write()
        with self.assertRaisesRegex(ValueError, 'Metadato y diverso'):
            V.validate_view(self.item, {}, 2027, 0)

    def test_infinity_rejected_even_with_new_hash(self):
        self.frame.loc[3, 'bx'] = np.inf
        self.write()
        with self.assertRaisesRegex(ValueError, 'Inf'):
            V.validate_view(self.item, {}, 2027, 0)

    def test_recomputation_detects_finite_but_wrong_features(self):
        self.frame.loc[3, 'bx'] += 1
        self.write()
        with self.assertRaises(AssertionError):
            V.validate_view(self.item, {}, 2027, 0, recompute=True)

    def test_nonfinite_or_missing_diagnostic_is_rejected(self):
        self.diagnostics['ball_error_metres_p50_p90_p99'][1] = float('nan')
        self.write()
        with self.assertRaisesRegex(ValueError, 'Quantili diagnostici'):
            V.validate_view(self.item, {}, 2027, 0)
        del self.diagnostics['ball_error_metres_p50_p90_p99']
        self.write()
        with self.assertRaisesRegex(ValueError, 'Schema delle diagnostiche'):
            V.validate_view(self.item, {}, 2027, 0)

    def test_observable_diagnostic_is_recomputed_for_every_cache(self):
        self.diagnostics['ball_missing_fraction'] = .25
        self.write()
        with self.assertRaisesRegex(ValueError, 'diversa dalle feature'):
            V.validate_view(self.item, {}, 2027, 0)

    def test_coordinate_recomputation_checks_error_diagnostics(self):
        self.diagnostics['ball_error_metres_p50_p90_p99'] = [1., 2., 3.]
        self.write()
        with self.assertRaisesRegex(ValueError, 'Diagnostiche diverse dalla ricomputazione'):
            V.validate_view(self.item, {}, 2027, 0, recompute=True)

    def test_profile_code_mismatch_rejected_after_payload_checks(self):
        with patch.object(A, 'validate_profile', return_value=True), patch.object(A, '_code_hash', return_value='now'):
            with self.assertRaisesRegex(ValueError, 'altro codice'):
                V._validate_profile({'manifest': {'code_sha256': 'old'}}, train=[self.item])

    def test_missing_original_cache_is_not_regenerated(self):
        # Call the unpatched implementation: setUp patches _original_info for
        # the synthetic view tests, so use a fresh direct alias from __wrapped__
        # is not applicable. Restore only this mock for this isolated check.
        self.stack.close()
        with patch.object(dati, '_path', return_value=str(Path(self.temp.name) / 'absent.parquet')):
            with patch.object(dati, 'build_one', side_effect=AssertionError('must not rebuild')) as builder:
                with self.assertRaisesRegex(ValueError, 'nessuna rigenerazione automatica'):
                    V._original_info(self.item)
                builder.assert_not_called()

    def test_final_profile_count_distinguishes_checks_from_unique_profiles(self):
        root = Path(self.temp.name) / 'run'
        root.mkdir()
        other = dati.external_matches()[1]
        config = dict(smoke=False, sources={'a': 'b'}, seeds=[11, 22], replicas=1)
        plan = {'items': [list(self.item), list(other)], 'fits': {'excl_0': {
            'train': [list(self.item)], 'test': [list(other)], 'excluded': [0]}}}
        for name, value in (('config.json', config), ('piano.json', plan)):
            (root / name).write_text(json.dumps(value), encoding='utf-8')
        for replicate in range(2):
            folder = V.S.fit_path(root, 'risporcata', replicate, 'excl_0')
            folder.mkdir(parents=True)
            (folder / 'profilo.json').write_text(json.dumps({'profile_sha256': 'same_cv_profile'}), encoding='utf-8')
        target = root / 'final_model'
        target.mkdir()
        (target / 'profilo.json').write_text(json.dumps({'profile_sha256': 'final_profile'}), encoding='utf-8')
        with patch.object(V.S, 'source_hashes', return_value=config['sources']), \
             patch.object(V.S, 'verify_complete'), patch.object(V, '_validate_profile'), \
             patch.object(V.S.R, 'model_dir', return_value=str(target)), \
             patch.object(V, 'validate_view', return_value={'rows': 12, 'recomputed': False}):
            result = V.validate_run(root, include_final=True)
        self.assertEqual(result['profiles_checked'], 3)
        self.assertEqual(result['unique_profiles_checked'], 2)
        self.assertEqual(result['views_checked'], 4)


if __name__ == '__main__':
    unittest.main()
