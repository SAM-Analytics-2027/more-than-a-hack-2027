"""Synthetic regression tests for split integrity, official metrics and external selection."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modelli_mv'))
import constants as K
import dati
import esperimento as E
import finale_augmentation as F
import modello as M
import prova_augmentation as P


def row(src, mid, spell=.8, macro=.94, view='originale'):
    return {'fonte': src, 'partita': mid, 'vista': view, 'spellF1': spell,
            'macroF1': macro, 'possessi': 10, 'predetti': 10, 'abbinati50': 8,
            'brevi': 3, 'brevi_abbinati50': 2}


class PipelineTests(unittest.TestCase):
    def test_splits_are_disjoint_and_cover_every_external_match(self):
        items = sorted(dati.external_matches())
        assignment = E.folds(items)
        held_out = []
        for k in range(K.N_FOLDS):
            train, calibration, test = P.split_fold(items, assignment, k)
            self.assertFalse(set(train) & set(calibration))
            self.assertFalse(set(train) & set(test))
            self.assertFalse(set(calibration) & set(test))
            self.assertEqual(set(items), set(train + calibration + test))
            held_out.extend(test)
        self.assertCountEqual(items, held_out)

    def test_splits_reject_duplicate_match(self):
        items = sorted(dati.external_matches())
        with self.assertRaises(ValueError):
            P.split_fold(items + items[:1], E.folds(items), 0)

    def test_focus_does_not_treat_halftime_or_x_as_an_event(self):
        frame = pd.DataFrame({'y': [0, 0, 1, 1, 3, 0, 2, 2],
                              'period': [1, 1, 2, 2, 2, 2, 2, 2]})
        np.testing.assert_array_equal(P._focus(frame, np.arange(8)), [6])

    def test_score_matches_official_scorer_with_halftime_and_abstention(self):
        y = np.array([0] * 40 + [2] * 20 + [1] * 40 + [1] * 40 + [2] * 20 + [0] * 40, dtype=np.int8)
        pred = y.copy()
        pred[25:30] = 1
        pred[80:85] = 3
        period = np.repeat([1, 2], 100)
        probabilities = np.full((len(y), 3), 1 / 3)
        with patch.object(M, 'decode', return_value=pred):
            actual = P._score(probabilities, (y, period, np.zeros(len(y), bool)), K.DEFAULT)
        expected = M.pm.evaluate(M.to_codes(P.R._gap(y, period)), M.to_codes(P.R._gap(pred, period)))
        self.assertAlmostEqual(actual['spellF1'], expected['spell_f1_mean'], places=12)
        self.assertAlmostEqual(actual['macroF1'], expected['macro_f1'], places=12)

    def test_calibration_ignores_stress_and_skillcorner(self):
        tactical = ('dfl', 'synthetic')
        tv = ('skillcorner', 'synthetic_tv')
        original = np.array([[.9, .05, .05], [.9, .05, .05]])
        stress = np.array([[.05, .9, .05], [.05, .9, .05]])
        probs = {(tactical, 'originale'): original, (tactical, P.A.TEST_VIEW): stress,
                 (tv, 'originale'): stress}
        info = (np.zeros(2, np.int8), np.ones(2, np.int8), np.zeros(2, bool))
        meta = {key: info for key in probs}
        seen = []

        def decode(p, *args):
            seen.append(p)
            return np.zeros(2, np.int8)

        with patch.object(M, 'decode', side_effect=decode), patch.object(K, 'GRID', {}):
            P._calibrate(probs, meta, [tactical, tv], rounds=1)
        self.assertTrue(seen)
        self.assertTrue(all(p is original for p in seen))

    def test_selection_weights_matches_instead_of_sources(self):
        base = [row('dfl', f'd{i}', .9) for i in range(7)] + [row('metrica', f'm{i}', .4) for i in range(3)]
        augmented = [row('dfl', f'd{i}', .7) for i in range(7)] + [row('metrica', f'm{i}', .7) for i in range(3)]
        chosen, scores = P.select_mode([{'modi': {'base': base, 'combinato': augmented}}], ['base', 'combinato'])
        self.assertEqual(chosen, 'base')
        self.assertAlmostEqual(scores['base']['spellF1'], .75)

    def test_selection_obeys_macro_floor_and_ignores_stress(self):
        base = [row('dfl', 'one', .80, .91), row('dfl', 'one', .1, .1, P.A.TEST_VIEW)]
        augmented = [row('dfl', 'one', .95, .89), row('dfl', 'one', 1, 1, P.A.TEST_VIEW)]
        chosen, _ = P.select_mode([{'modi': {'base': base, 'combinato': augmented}}], ['base', 'combinato'])
        self.assertEqual(chosen, 'base')

    def _complete_report(self):
        items = sorted(dati.external_matches())
        assignment = E.folds(items)
        folds = []
        for k in range(K.N_FOLDS):
            train, calibration, test = P.split_fold(items, assignment, k)
            folds.append({'fold': k, 'train': train, 'calibrazione': calibration, 'test': test,
                          'modi': {'base': [row(src, mid, view=view) for src, mid in test for view in P.EVAL_VIEWS]}})
        config = {'protocollo': P.PROTOCOL_VERSION, 'tracking': 'sporcato', 'smoke': False, 'modi': ['base']}
        return items, config, {'fold': folds}

    def test_completed_report_rejects_leakage_and_duplicate_rows(self):
        items, config, report = self._complete_report()
        self.assertEqual(len(P.validate_completed_report(report, config, items)), K.N_FOLDS)
        leaked = copy.deepcopy(report)
        leaked['fold'][0]['train'].append(leaked['fold'][0]['test'][0])
        with self.assertRaises(ValueError):
            P.validate_completed_report(leaked, config, items)
        duplicate = copy.deepcopy(report)
        duplicate['fold'][0]['modi']['base'].append(duplicate['fold'][0]['modi']['base'][0])
        with self.assertRaises(ValueError):
            P.validate_completed_report(duplicate, config, items)

    def test_completed_report_rejects_old_clean_protocol_and_smoke(self):
        items, config, report = self._complete_report()
        for overrides in ({'tracking': 'pulito'}, {'smoke': True}, {'protocollo': 1}):
            with self.assertRaises(ValueError):
                P.validate_completed_report(report, dict(config, **overrides), items)

    def test_probability_archive_roundtrip_and_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'probabilita.npz'
            original = {(('metrica', 'Sample_Game_1'), 'originale'): np.array([[.1, .2, .7]], np.float32)}
            P._save_probabilities(path, original)
            restored = P.load_probabilities(path)
            self.assertEqual(set(restored), set(original))
            np.testing.assert_array_equal(next(iter(restored.values())), next(iter(original.values())))
            manifest = {'artefatti': {'probabilita.npz': P._sha256(path)}}
            P._verify_artifacts(root, manifest)
            path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                P._verify_artifacts(root, manifest)
            with self.assertRaises(ValueError):
                P._verify_artifacts(root, {'artefatti': {'../outside': 'bad'}})

    def test_final_calibration_reads_only_held_out_probabilities(self):
        train, held_out = ('dfl', 'train'), ('metrica', 'test')
        folds = [{'fold': 0, 'test': [held_out]}]
        cached = {(train, 'originale'): np.ones((2, 3)), (held_out, 'originale'): np.ones((2, 3))}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(P, 'load_probabilities', return_value=cached), patch.object(P, '_meta', return_value={}), \
                    patch.object(P, '_calibrate', return_value=dict(K.DEFAULT)) as calibrate:
                F._calibrate(Path(tmp), Path(tmp), folds, [held_out], 'base', .90)
            supplied = calibrate.call_args.args[0]
            self.assertEqual(set(supplied), {(held_out, 'originale')})
            self.assertTrue((Path(tmp) / 'parametri_viterbi.json').exists())


if __name__ == '__main__':
    unittest.main()
