"""Controlli indipendenti su annidamento, provenienza e unità statistica."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modelli_mv'))
import dati
import esperimento as E
import protocollo_augmentation as P


class NestedProtocolTests(unittest.TestCase):
    def setUp(self):
        self.items = sorted(dati.external_matches())
        self.assignment = E.folds(self.items)
        self.plan = P.build_plan(self.items, self.assignment)

    def test_pair_reuse_preserves_true_nested_training_exclusions(self):
        self.assertEqual(len(self.plan['fits']), 15)
        counted = []
        for outer in self.plan['outer']:
            k = outer['fold']
            outer_test = {tuple(it) for it in outer['test']}
            refit = self.plan['fits'][outer['refit']]
            self.assertEqual(set(map(tuple, refit['train'])), set(self.items) - outer_test)
            predicted = []
            for key, held_out in outer['inner_sources'].items():
                train = set(map(tuple, self.plan['fits'][key]['train']))
                inner_test = set(map(tuple, held_out))
                self.assertFalse(train & outer_test)
                self.assertFalse(train & inner_test)
                self.assertEqual(train | outer_test | inner_test, set(self.items))
                self.assertIn(k, self.plan['fits'][key]['excluded'])
                predicted.extend(inner_test)
                counted.append(key)
            self.assertCountEqual(predicted, set(self.items) - outer_test)
        pair_keys = [key for key, value in self.plan['fits'].items() if len(value['excluded']) == 2]
        self.assertTrue(all(counted.count(key) == 2 for key in pair_keys))
        self.assertEqual([len(o['calibration']) for o in self.plan['outer']], [7, 7, 8, 9, 9])
        self.assertEqual(sum(len(self.plan['fits'][o['refit']]['train']) for o in self.plan['outer']) / 5, 24)

    def test_plan_survives_json_and_keeps_original_fold_assignment(self):
        self.assertTrue(P.validate_plan(json.loads(json.dumps(self.plan))))
        self.assertEqual(P.build_plan(list(reversed(self.items))), self.plan)

    def test_rejects_global_oof_substitution_and_invalid_items(self):
        damaged = copy.deepcopy(self.plan)
        outer = damaged['outer'][0]
        key = next(iter(outer['inner_sources']))
        outer['inner_sources']['excl_1'] = outer['inner_sources'].pop(key)
        with self.assertRaises(ValueError):
            P.validate_plan(damaged)
        for items in (self.items + self.items[:1], [('barca', 'target')], [('dfl', '../target')]):
            with self.assertRaises(ValueError):
                P.build_plan(items)

    def test_metadata_detects_generator_normalizer_and_label_leakage(self):
        key = 'excl_0_1'
        spec = self.plan['fits'][key]
        metadata = {field: copy.deepcopy(spec['train'])
                    for field in ('train', 'normalization_train', 'label_train', 'generator_train')}
        metadata['early_stopping'] = False
        self.assertTrue(P.validate_fit_provenance(self.plan, key, metadata, generator_required=True))
        for field in ('train', 'normalization_train', 'label_train', 'generator_train'):
            damaged = copy.deepcopy(metadata)
            damaged[field].append(spec['test'][0])
            with self.assertRaisesRegex(ValueError, 'Provenienza|generatore'):
                P.validate_fit_provenance(self.plan, key, damaged, generator_required=True)
        metadata['generator_train'] = []
        self.assertTrue(P.validate_fit_provenance(self.plan, key, metadata))
        metadata['early_stopping'] = True
        with self.assertRaises(ValueError):
            P.validate_fit_provenance(self.plan, key, metadata)

    def test_selects_only_double_heldout_tactical_probabilities(self):
        outer = P.outer_plan(self.plan, 0)
        predictions = {}
        for index, (key, spec) in enumerate(self.plan['fits'].items()):
            predictions[key] = {tuple(it): np.tile([.1 + index / 100, .2, .7 - index / 100], (4, 1))
                                for it in spec['test']}
        selected = P.select_inner_predictions(self.plan, 0, predictions)
        self.assertEqual(set(selected), set(map(tuple, outer['calibration'])))
        for key, targets in outer['inner_sources'].items():
            for item in map(tuple, targets):
                if item[0] in P.TACTICAL:
                    self.assertIs(selected[item], predictions[key][item])
        missing = copy.deepcopy(predictions)
        needed_key = next(key for key, ids in outer['inner_sources'].items()
                          if any(it[0] in P.TACTICAL for it in ids))
        del missing[needed_key]
        with self.assertRaises(ValueError):
            P.select_inner_predictions(self.plan, 0, missing)
        damaged = copy.deepcopy(predictions)
        target = next(it for it in map(tuple, outer['inner_sources'][needed_key]) if it[0] in P.TACTICAL)
        damaged[needed_key][target][0, 0] = np.nan
        with self.assertRaises(ValueError):
            P.select_inner_predictions(self.plan, 0, damaged)

    def test_completion_refuses_smoke_missing_and_duplicate_fit(self):
        recipes, replicas = ['base', 'aug'], [0, 1, 2]
        finished = [(mode, seed, key) for mode in recipes for seed in replicas for key in self.plan['fits']]
        self.assertTrue(P.validate_completion(self.plan, finished, recipes, replicas))
        for incomplete in (finished[:-1], finished + finished[:1]):
            with self.assertRaises(ValueError):
                P.validate_completion(self.plan, incomplete, recipes, replicas)
        with self.assertRaises(ValueError):
            P.validate_completion(self.plan, finished, recipes, replicas, smoke=True)

    def test_score_report_requires_every_match_and_correct_fold_and_mean(self):
        report = {'mode': 'base', 'replicate': 0,
                  'rows': [{'mode': 'base', 'replicate': 0, 'source': src, 'match': mid,
                            'fold': self.assignment[src, mid], 'spellF1': .7, 'macroF1': .93}
                           for src, mid in self.items],
                  'decoders': {str(k): {} for k in range(5)},
                  'mean10': {'spellF1': .7, 'macroF1': .93}}
        self.assertTrue(P.validate_score_report(self.plan, report))
        for field, bad in [('rows', report['rows'][:-1]), ('rows', report['rows'] + report['rows'][:1]),
                           ('decoders', {'0': {}}), ('mean10', {'spellF1': .8, 'macroF1': .93})]:
            with self.assertRaises(ValueError):
                P.validate_score_report(self.plan, dict(report, **{field: bad}))
        for field, value in [('fold', 99), ('replicate', 1), ('mode', 'aug'), ('spellF1', np.nan)]:
            bad = copy.deepcopy(report)
            bad['rows'][0][field] = value
            with self.assertRaises(ValueError):
                P.validate_score_report(self.plan, bad)


class PairedSummaryTests(unittest.TestCase):
    def rows(self):
        rows = []
        for recipe in ('base', 'aug'):
            for replica in range(3):
                for source, match in [('dfl', 'a'), ('metrica', 'b')]:
                    delta = (.02 if match == 'a' else -.01) if recipe == 'aug' else 0
                    rows.append({'recipe': recipe, 'replicate': replica, 'source': source,
                                 'match': match, 'spellF1': .7 + replica * .01 + delta,
                                 'macroF1': .93 + delta / 2})
        return rows

    def test_pairs_by_match_and_reports_seed_variation_separately(self):
        summary = P.paired_summary(self.rows(), 'base', n_boot=100)
        aug = summary['recipes']['aug']
        self.assertAlmostEqual(aug['paired_delta']['spellF1'], .005)
        self.assertEqual(len(aug['per_match']), 2)
        self.assertEqual(len(aug['replicates']), 3)
        low, high = aug['paired_match_bootstrap_ci95']['spellF1']
        self.assertAlmostEqual(low, -.01)
        self.assertAlmostEqual(high, .02)
        self.assertAlmostEqual(aug['delta_sd_across_replicates']['spellF1'], 0)
        self.assertEqual(summary, P.paired_summary(list(reversed(self.rows())), 'base', n_boot=100))

    def test_duplicate_replication_does_not_shrink_match_bootstrap_interval(self):
        original = self.rows()
        one_replica = [row for row in original if row['replicate'] == 0]
        a = P.paired_summary(one_replica, 'base', n_boot=100)
        b = P.paired_summary(original, 'base', n_boot=100)
        np.testing.assert_allclose(a['recipes']['aug']['paired_match_bootstrap_ci95']['spellF1'],
                                   b['recipes']['aug']['paired_match_bootstrap_ci95']['spellF1'])

    def test_requires_exact_paired_coverage_and_separate_ensemble_scores(self):
        for bad in (self.rows()[:-1], self.rows() + self.rows()[:1]):
            with self.assertRaises(ValueError):
                P.paired_summary(bad, 'base')
        ensemble = [dict(row, spellF1=.99) for row in self.rows() if row['replicate'] == 0]
        summary = P.paired_summary(self.rows(), 'base', ensemble_rows=ensemble)
        self.assertEqual(summary['ensemble']['aug']['spellF1'], .99)
        self.assertNotEqual(summary['recipes']['aug']['mean']['spellF1'], .99)


class SourceSnapshotTests(unittest.TestCase):
    def test_archives_exact_code_and_refuses_changed_or_tampered_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'model.py'
            source.write_bytes(b'x = 1\r\n')
            snapshot = root / 'snapshot'
            hashes = P.archive_sources(snapshot, [source], project_root=root)
            self.assertEqual((snapshot / 'model.py').read_bytes(), source.read_bytes())
            self.assertEqual(P.archive_sources(snapshot, [source], project_root=root), hashes)
            source.write_bytes(b'x = 2\n')
            with self.assertRaises(ValueError):
                P.archive_sources(snapshot, [source], project_root=root)
            source.write_bytes(b'x = 1\r\n')
            (snapshot / 'model.py').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                P.archive_sources(snapshot, [source], project_root=root)
            with self.assertRaises(ValueError):
                P.source_fingerprints([Path(__file__)], project_root=root)


if __name__ == '__main__':
    unittest.main()
