"""Controlled ablations: synthetic external-shaped data, no competition labels."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modelli_mv'))
import caratteristiche as C
import componenti_augmentation as A
import constants as K
import dati
import esperimento as E
import rete


def table(n=3200):
    t = np.arange(n, dtype=np.float32)
    m = dict(period=np.ones(n, np.int8), team=np.array([0, 0, 1, 1]),
             px=np.tile([-40., -10., 10., 40.], (n, 1)).astype(np.float32) + t[:, None] * .001,
             py=np.tile([-5., 3., -2., 6.], (n, 1)).astype(np.float32),
             bx=-10 + t * .001, by=np.full(n, 3., np.float32), length=105., width=68., fps=25)
    df = C.features(m, legacy=True)
    df['frame'], df['period'], df['y'] = np.arange(n), m['period'], ((t // 200) % 3).astype(np.int8)
    df['fonte_tv'] = np.float32(0)
    return df, E.feature_cols(df)


class ControlledComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.item = dati.external_matches()[0]
        cls.other = dati.external_matches()[1]
        cls.df, cls.cols = table()
        cls.data = {cls.item: cls.df}

    def test_legacy_tree_rows_equal_historical_recipe(self):
        previous = E.WEIGHTS
        try:
            E.WEIGHTS = True
            x, y, w = E.training_rows(self.df, self.cols)
        finally:
            E.WEIGHTS = previous
        actual = A.tree_rows(self.data, [self.item], self.cols)
        pd.testing.assert_frame_equal(actual[0], x)
        np.testing.assert_array_equal(actual[1], y)
        np.testing.assert_array_equal(actual[2], w * K.SOURCE_WEIGHTS[self.item[0]])

    def test_legacy_normalizer_and_batch_equal_head_algorithm(self):
        config = A.Config()
        prepared = A.prepare_network(self.data, [self.item], self.cols, config)
        x = self.df[self.cols].to_numpy()[::K.NORM_SUBSAMPLE_STEP]
        with np.errstate(invalid='ignore'):
            med = np.nanmedian(x, axis=0)
            iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
        med = np.where(np.isfinite(med), med, 0.)
        iqr = np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, 1.)
        np.testing.assert_array_equal(prepared['med'], med)
        np.testing.assert_array_equal(prepared['iqr'], iqr)
        # Independent reconstruction of the pre-augmentation HEAD crop loop.
        normalized = []
        for raw, y, w, _ in prepared['pools'][0]:
            z = (raw - med) / iqr
            miss = np.isnan(z)
            z = np.clip(np.nan_to_num(z, nan=0.), *K.NET_FEATURE_CLIP)
            normalized.append((np.concatenate((z, miss.astype(np.float32)), axis=1).astype(np.float32), y, w))
        rng = np.random.default_rng(47)
        lens = np.asarray([len(x[0]) for x in normalized], dtype=float)
        expected = [[], [], []]
        for index in rng.choice(len(normalized), size=4, p=lens / lens.sum()):
            z, y, w = normalized[index]
            start = rng.integers(0, len(z) - rete.CROP + 1)
            for dest, values in zip(expected, (z, y, w)):
                dest.append(values[start:start + rete.CROP])
        actual = A.network_batch(prepared, np.random.default_rng(47), np.random.default_rng(999), config, batch_size=4)
        for before, after in zip(expected, actual):
            np.testing.assert_array_equal(np.stack(before), np.stack(after))

    def test_each_intervention_has_explicit_independent_config(self):
        base = asdict(A.PRESETS['base'])
        for name in ('feature', 'contesto', 'normalizzazione', 'segmentazione', 'inferenza'):
            changed = sum(base[k] != v for k, v in asdict(A.PRESETS[name]).items())
            self.assertEqual(changed, 1, name)
        with self.assertRaises(ValueError):
            A.Config(normalization='typo')

    def test_replica_never_changes_metadata_or_original_data(self):
        replica = self.df.copy(deep=True)
        replica.loc[100, 'y'] = 2
        with self.assertRaisesRegex(ValueError, 'modifica y'):
            A.tree_rows(self.data, [self.item], self.cols, views={self.item: [replica]})
        replica['y'] = self.df['y']
        replica['bx'] += 20
        before = self.df.copy(deep=True)
        one = A.tree_rows(self.data, [self.item], self.cols, views={self.item: [replica]}, seed=3)
        two = A.tree_rows(self.data, [self.item], self.cols, views=lambda it: [replica], seed=3)
        pd.testing.assert_frame_equal(one[0], two[0])
        pd.testing.assert_frame_equal(before, self.df)
        self.assertTrue(np.any(one[0]['bx'].to_numpy() != A.tree_rows(self.data, [self.item], self.cols)[0]['bx'].to_numpy()))

    def test_augmented_views_have_own_boundaries_and_fixed_budget(self):
        replica = self.df.copy(deep=True)
        replica.loc[1000:1500, ['n_casa', 'n_ospiti', 'palla_vista']] = 0
        config = A.Config(segment_version=2, halo=True)
        base = A.prepare_network(self.data, [self.item], self.cols, config)
        aug = A.prepare_network(self.data, [self.item], self.cols, config, views={self.item: [replica]})
        self.assertEqual(aug['steps'], base['steps'])
        self.assertEqual(len(aug['pools'][0]), 2)
        self.assertEqual(len(aug['pools'][1]), 4)
        for *_, sel in aug['pools'][1]:
            self.assertFalse(np.any((sel >= 1000) & (sel <= 1500)))
            self.assertFalse(sel[0] < 1000 and sel[-1] > 1500)
        np.testing.assert_array_equal(aug['med'], base['med'])
        np.testing.assert_array_equal(aug['iqr'], base['iqr'])

    def test_segmentation_ablation_keeps_original_optimizer_budget(self):
        fragmented = self.df.copy(deep=True)
        fragmented.loc[400:2300, ['n_casa', 'n_ospiti', 'palla_vista']] = 0
        data = {self.item: fragmented}
        base = A.prepare_network(data, [self.item], self.cols, A.Config())
        segmented = A.prepare_network(data, [self.item], self.cols, A.Config(segment_version=2))
        self.assertEqual(base['steps'], segmented['steps'])
        self.assertGreater(sum(len(q[0]) for q in base['pools'][0]), sum(len(q[0]) for q in segmented['pools'][0]))

    def test_component_manifest_rejects_missing_and_changed_weights(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            with self.assertRaisesRegex(ValueError, 'manifest.*mancante'):
                A.verify_model_manifest(folder)
            A._write_json(folder / 'modello.json', {'semi_rete': [2027]})
            (folder / 'hgb.joblib').write_bytes(b'tree')
            (folder / 'rete_seme0.pt').write_bytes(b'net')
            names = ['modello.json', 'hgb.joblib', 'rete_seme0.pt']
            A._write_json(folder / 'manifest.json', {n: A._sha256(folder / n) for n in names})
            self.assertEqual(len(A.verify_model_manifest(folder)), 3)
            (folder / 'rete_seme0.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'modificato'):
                A.verify_model_manifest(folder)

    def test_short_sequences_halo_padding_never_receive_loss(self):
        df = self.df.iloc[:30].copy()
        config = A.Config(halo=True, segment_version=2)
        prepared = A.prepare_network({self.item: df}, [self.item], self.cols, config)
        xb, yb, wb = A.network_batch(prepared, np.random.default_rng(3), np.random.default_rng(4), config, batch_size=1)
        halo = sum(K.DILATIONS)
        self.assertEqual(xb[0].shape, (rete.CROP + 2 * halo, 2 * len(self.cols)))
        self.assertTrue(np.isfinite(xb[0]).all())
        self.assertTrue(np.all(wb[0][:halo] == 0))
        self.assertTrue(np.all(wb[0][halo + 6:] == 0))
        self.assertTrue(np.all(yb[0][wb[0] == 0] == K.X))

    def test_hgb_requires_explicit_rounds_or_disjoint_matches(self):
        with self.assertRaisesRegex(ValueError, 'iterazioni fisse'):
            A.fit_tree(self.data, [self.item], self.cols)
        with self.assertRaisesRegex(ValueError, 'sovrapposte'):
            A.fit_tree(self.data, [self.item], self.cols, val_data=self.data, val_items=[self.item])
        with self.assertRaisesRegex(ValueError, 'soltanto su partite esterne'):
            A.fit_tree(self.data, [dati.BARCA], self.cols, iterations=500)

    def test_fixed_hgb_disables_random_row_early_stopping(self):
        class FakeTree:
            def __init__(self):
                self.params, self.fitted = {}, None
            def set_params(self, **kwargs):
                self.params.update(kwargs)
                return self
            def fit(self, *args, **kwargs):
                self.fitted = kwargs
                return self
        tree = FakeTree()
        with patch.object(E, 'make_model', return_value=tree):
            got = A.fit_tree(self.data, [self.item], self.cols, iterations=500, seed=31)
        self.assertIs(got, tree)
        self.assertEqual(tree.params, dict(random_state=31, validation_fraction=None, early_stopping=False, max_iter=500))
        self.assertNotIn('X_val', tree.fitted)

    def test_raw_predictions_are_not_rounded_and_override_does_not_mutate(self):
        a = np.tile([.123456789, .234567891, .641975320], (7, 1))
        b = np.tile([.222222221, .333333333, .444444446], (7, 1))
        model = {'inference_version': 1}
        with patch.object(E, 'predict_sym', return_value=a), patch.object(rete, 'predict', return_value=b) as predict:
            got = A.predict_components((object(), [model], self.cols), self.df.iloc[:7], self.item, inference_version=2)
        np.testing.assert_array_equal(got, (a + b) / 2)
        self.assertEqual(predict.call_args.args[0]['inference_version'], 2)
        self.assertEqual(model['inference_version'], 1)


if __name__ == '__main__':
    unittest.main()
