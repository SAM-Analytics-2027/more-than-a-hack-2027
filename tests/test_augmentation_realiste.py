"""Corruzione clean->dirty: indipendenza da GT/test e invarianti osservabili."""
import copy
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

ITEM = ('dfl', 'J03WMX')
OTHER = ('dfl', 'J03WN1')


def fixture():
    n, slots = 1600, 6
    t = np.arange(n, dtype=np.float32)
    px = np.tile(np.linspace(-25, 25, slots, dtype=np.float32), (n, 1))
    py = np.tile(np.linspace(-15, 15, slots, dtype=np.float32), (n, 1))
    px += .5 * np.sin(t[:, None] / 23)
    py += .5 * np.sin(t[:, None] / 17)
    period = np.r_[np.ones(750), np.zeros(100), np.full(750, 2)].astype(np.int8)
    px[1000:1040] = py[1000:1040] = np.nan
    bx = (3 * np.sin(t / 17)).astype(np.float32)
    by = (4 * np.cos(t / 13)).astype(np.float32)
    bx[1250:1270] = by[1250:1270] = np.nan
    clean = dict(name='fixture', fps=25, length=105., width=68., period=period,
                 px=px, py=py, bx=bx, by=by, team=np.array([0, 0, 0, 1, 1, 1]),
                 labels=np.resize(np.array(['H', 'D', 'A']), n))
    dirty = copy.deepcopy(clean)
    lagged = A._lagged_ball(clean, 2)
    dirty['bx'] = lagged[:, 0] + .4 * np.sin(t / 10)
    dirty['by'] = lagged[:, 1] + .3 * np.cos(t / 10)
    for a, b in ((100, 200), (400, 470), (900, 980), (1400, 1550)):
        dirty['bx'][a:b] = dirty['by'][a:b] = np.nan
    dirty['px'][150:220, 0] = dirty['py'][150:220, 0] = np.nan
    dirty['px'][450:530, 3] = dirty['py'][450:530, 3] = np.nan
    order = [2, 0, 1, 5, 3, 4]
    dirty['px'], dirty['py'] = dirty['px'][:, order], dirty['py'][:, order]
    return clean, dirty


def profile_fixture():
    clean, dirty = fixture()
    params = A._pool([A.summarize_pair(clean, dirty)])
    profile = {'schema': A.SCHEMA, 'manifest': {'schema': A.SCHEMA,
               'train_items': [list(ITEM)], 'inputs': {'/'.join(ITEM): {}}, 'code_sha256': 'fixture'},
               'sources': {'dfl': params}, 'fallback': copy.deepcopy(params)}
    profile['profile_sha256'] = A._digest(profile)
    return profile


class TransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = profile_fixture()

    def test_reproducibility_labels_and_input_immutability(self):
        clean, _ = fixture()
        before = copy.deepcopy(clean)
        first = A.transform(clean, ITEM, self.profile, seed=72, replica=0)
        repeated = A.transform(clean, ITEM, self.profile, seed=72, replica=0)
        other_gt = A.transform(dict(clean, labels=np.full(len(clean['period']), 'X')),
                               ITEM, self.profile, seed=72, replica=0)
        for key in ('bx', 'by', 'px', 'py'):
            np.testing.assert_array_equal(first[key], repeated[key])
            np.testing.assert_array_equal(first[key], other_gt[key])
            np.testing.assert_array_equal(clean[key], before[key])
        for key in ('team', 'period', 'labels'):
            np.testing.assert_array_equal(first[key], clean[key])

    def test_independent_replicas_and_seeds(self):
        clean, _ = fixture()
        a = A.transform(clean, ITEM, self.profile, seed=5, replica=0)
        b = A.transform(clean, ITEM, self.profile, seed=5, replica=1)
        c = A.transform(clean, ITEM, self.profile, seed=6, replica=0)
        self.assertFalse(np.array_equal(a['bx'], b['bx'], equal_nan=True))
        self.assertFalse(np.array_equal(a['bx'], c['bx'], equal_nan=True))

    def test_no_reconstruction_or_fabricated_replays(self):
        clean, _ = fixture()
        out = A.transform(clean, ITEM, self.profile)
        active = A._active(clean)
        for key in ('bx', 'by', 'px', 'py'):
            self.assertTrue(np.isnan(out[key][np.isnan(clean[key])]).all())
            np.testing.assert_array_equal(out[key][~active], clean[key][~active])
            self.assertFalse(np.isinf(out[key]).any())
        np.testing.assert_array_equal(A._active(out), active)

    def test_error_remains_temporally_correlated(self):
        params = self.profile['sources']['dfl']
        error = A._sample_error(params['ball_error_bank'], 10000, np.random.default_rng(7))
        for axis in (0, 1):
            self.assertGreater(np.corrcoef(error[:-1, axis], error[1:, axis])[0, 1], .90)

    def test_anonymous_slots_preserve_geometry_teams_and_continuous_tracks(self):
        clean, dirty = fixture()
        px, py, team = A._anonymize_players(dirty['px'], dirty['py'], dirty['team'],
                                            dirty['period'], [6, 6], np.random.default_rng(7))
        for t in (0, 1):
            for row in range(len(px)):
                before = sorted(zip(dirty['px'][row, dirty['team'] == t], dirty['py'][row, dirty['team'] == t]))
                after = sorted(zip(px[row, team == t], py[row, team == t]))
                before = sorted(xy for xy in before if np.isfinite(xy).all())
                after = sorted(xy for xy in after if np.isfinite(xy).all())
                np.testing.assert_array_equal(after, before)
        valid = np.isfinite(px[1:]) & np.isfinite(px[:-1])
        self.assertLess(np.abs(np.diff(px, axis=0))[valid].max(), .1)

    def test_static_slot_permutation_preserves_features(self):
        clean, _ = fixture()
        order = [2, 0, 1, 5, 3, 4]
        permuted = dict(clean, px=clean['px'][:, order], py=clean['py'][:, order], team=clean['team'][order])
        for legacy in (False, True):
            np.testing.assert_allclose(A.C.features(clean, legacy=legacy).to_numpy(),
                                       A.C.features(permuted, legacy=legacy).to_numpy(),
                                       rtol=2e-5, atol=2e-5, equal_nan=True)

    def test_no_invented_noise_when_pair_has_none(self):
        clean, _ = fixture()
        params = A._pool([A.summarize_pair(clean, clean)])
        profile = copy.deepcopy(self.profile)
        profile['sources']['dfl'] = params
        profile['fallback'] = params
        profile['profile_sha256'] = A._digest({k: v for k, v in profile.items() if k != 'profile_sha256'})
        out = A.transform(clean, ITEM, profile)
        for key in ('bx', 'by', 'px', 'py'):
            np.testing.assert_array_equal(out[key], clean[key])

    def test_reject_target_bad_seed_and_wrong_frequency(self):
        clean, _ = fixture()
        with self.assertRaises(ValueError):
            A.transform(clean, ('barca', 'BAR-ATH'), self.profile)
        with self.assertRaises(ValueError):
            A.transform(clean, ITEM, self.profile, seed=-1)
        with self.assertRaises(ValueError):
            A.transform(dict(clean, fps=10), ITEM, self.profile)


class FitAndProvenanceTests(unittest.TestCase):
    def test_player_alignment_uses_geometry_and_team_not_slot(self):
        clean, dirty = fixture()
        presence = A._player_presence(clean, dirty)
        expected = np.isfinite(clean['px'])
        expected[150:220, 0] = False
        expected[450:530, 3] = False
        np.testing.assert_array_equal(presence, expected)

    def test_fit_and_diagnostics_are_label_independent(self):
        clean, dirty = fixture()
        original = A.summarize_pair(clean, dirty)
        altered = A.summarize_pair(dict(clean, labels=None), dict(dirty, labels='ignored'))
        self.assertEqual(original, altered)
        self.assertEqual(A.diagnostics(clean), A.diagnostics(dict(clean, labels=None)))
        self.assertEqual(original['ball_lag'], 2)

    def test_misaligned_pairs_fail(self):
        clean, dirty = fixture()
        for changed in (dict(dirty, fps=10), dict(dirty, width=70),
                        dict(dirty, period=np.zeros_like(dirty['period']))):
            with self.assertRaises(ValueError):
                A.summarize_pair(clean, changed)

    def test_unseen_age_bins_do_not_make_missing_state_absorbing(self):
        summary = A.summarize_pair(*fixture())
        params = A._pool([summary])
        for name in ('ball', 'player'):
            counts = np.asarray(summary[name + '_counts'])
            supported = counts.sum(axis=(1, 3)) > 0
            self.assertFalse(supported[1, -1])
            hazard = np.asarray(params[name + '_hazard'])
            self.assertTrue((hazard[1, :, -1] > 0).all())

    def test_fit_reads_only_explicit_training_items_and_caches(self):
        summary = A.summarize_pair(*fixture())
        with tempfile.TemporaryDirectory() as tmp, patch.object(A.dati, 'CACHE', tmp), \
                patch.object(A, 'input_manifest', return_value={'tracking': 'sha'}), \
                patch.object(A, 'pair_summary', return_value=summary) as reader:
            profile = A.fit_profile([ITEM])
            reader.assert_called_once_with(ITEM)
            self.assertEqual(profile['manifest']['train_items'], [list(ITEM)])
            self.assertTrue(A.validate_profile(profile, train_items=[ITEM], excluded_items=[OTHER]))
            self.assertEqual(A.fit_profile([ITEM]), profile)
            reader.assert_called_once_with(ITEM)
            with self.assertRaises(ValueError):
                A.validate_profile(profile, excluded_items=[ITEM])
            with self.assertRaises(ValueError):
                A.validate_profile(profile, train_items=[OTHER])
            with self.assertRaises(ValueError):
                A.fit_profile([])

    def test_profile_integrity_and_input_drift_fail(self):
        profile = profile_fixture()
        profile['sources']['dfl']['ball_lags'][0] += 1
        with self.assertRaises(ValueError):
            A.validate_profile(profile)
        profile = profile_fixture()
        with patch.object(A, 'input_manifest', return_value={'changed': 'sha'}):
            with self.assertRaises(ValueError):
                A.validate_profile(profile, verify_inputs=True)

    def test_feature_cache_tracks_profile_inputs_seed_and_feature_version(self):
        profile = profile_fixture()
        with patch.object(A, 'input_manifest', return_value={'tracking': 'a'}):
            first = A.cache_path(ITEM, profile)
            self.assertNotEqual(first, A.cache_path(ITEM, profile, seed=42))
            self.assertNotEqual(first, A.cache_path(ITEM, profile, replica=1))
            self.assertNotEqual(first, A.cache_path(ITEM, profile, legacy=True))
        with patch.object(A, 'input_manifest', return_value={'tracking': 'b'}):
            self.assertNotEqual(first, A.cache_path(ITEM, profile))

    def test_feature_cache_provenance_and_corruption_detection(self):
        profile = profile_fixture()
        clean, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(A.dati, 'CACHE', tmp), \
                patch.object(A, 'input_manifest', return_value={'tracking': 'sha'}), \
                patch.object(A.C.fc, 'read_epts', return_value=clean) as reader, \
                patch.object(A.C, 'features', side_effect=lambda m, legacy=False: pd.DataFrame({'bx': m['bx']})):
            frame = A.load_view(ITEM, profile)
            reader.assert_called_once_with(A._folder(ITEM, 'pulite'))
            np.testing.assert_array_equal(frame['period'], clean['period'])
            np.testing.assert_array_equal(frame['y'], [A.K.CODE[x] for x in clean['labels']])
            pd.testing.assert_frame_equal(A.load_view(ITEM, profile), frame)
            path = A.cache_path(ITEM, profile)
            path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                A.load_view(ITEM, profile)


if __name__ == '__main__':
    unittest.main()
