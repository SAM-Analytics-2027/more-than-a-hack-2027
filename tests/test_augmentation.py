"""Invarianti dell'augmentation su dati sintetici, senza leggere la partita di gara."""

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modelli_mv'))
import augmentation as A


ITEM = ('dfl', 'J03WMX')


def match_fixture():
    n, slots = 900, 6
    t = np.arange(n, dtype=np.float32)
    px = np.tile(np.linspace(-20, 20, slots, dtype=np.float32), (n, 1))
    py = np.tile(np.linspace(-10, 10, slots, dtype=np.float32), (n, 1))
    px += np.sin(t[:, None] / 30)
    bx, by = np.sin(t / 20).astype(np.float32), np.cos(t / 30).astype(np.float32)
    bx[50:75] = by[50:75] = np.nan
    period = np.r_[np.ones(430), np.zeros(40), np.full(430, 2)].astype(np.int8)
    # Replay interno al secondo tempo, indipendente dalla verità H/A/D.
    px[650:690] = py[650:690] = np.nan
    return dict(fps=25, length=105.0, width=68.0, period=period,
                px=px, py=py, bx=bx, by=by, team=np.array([0, 0, 0, 1, 1, 1]),
                labels=np.array(['H', 'A', 'D'] * 300))


class TransformTests(unittest.TestCase):
    def test_deterministic_immutable_and_label_independent(self):
        match = match_fixture()
        saved = copy.deepcopy(match)
        other_labels = dict(match, labels=np.full(len(match['period']), 'X'))
        for view in (*A.TRAIN_VIEWS, A.TEST_VIEW):
            first = A.transform(match, ITEM, view)
            second = A.transform(match, ITEM, view)
            unrelated = A.transform(other_labels, ITEM, view)
            for name in ('bx', 'by', 'px', 'py'):
                np.testing.assert_array_equal(first[name], second[name])
                np.testing.assert_array_equal(first[name], unrelated[name])
                np.testing.assert_array_equal(match[name], saved[name])
            np.testing.assert_array_equal(first['period'], saved['period'])
            np.testing.assert_array_equal(first['team'], saved['team'])
            np.testing.assert_array_equal(first['labels'], saved['labels'])

    def test_preserves_gaps_and_never_restores_missing_ball(self):
        match = match_fixture()
        observed = (np.isfinite(match['px']) & np.isfinite(match['py'])).any(axis=1)
        untouched = (match['period'] == 0) | ~observed
        for view in (*A.TRAIN_VIEWS, A.TEST_VIEW):
            out = A.transform(match, ITEM, view)
            for name in ('bx', 'by', 'px', 'py'):
                np.testing.assert_array_equal(out[name][untouched], match[name][untouched])
            after = (np.isfinite(out['px']) & np.isfinite(out['py'])).any(axis=1)
            np.testing.assert_array_equal(after, observed)
            self.assertTrue(np.isnan(out['bx'][np.isnan(match['bx'])]).all())

    def test_dropout_cannot_remove_last_visible_player_later(self):
        class ForcedRandom:
            def poisson(self, rate):
                return 2

            def choice(self, candidates, size=None, **kwargs):
                return np.array([0, 1]) if size is not None else candidates[0]

            def lognormal(self, mean, sigma, count):
                return np.full(count, 20)

        bx, by = np.zeros(40), np.zeros(40)
        px, py = np.zeros((40, 3)), np.zeros((40, 3))
        px[2:, 1:] = py[2:, 1:] = np.nan
        A._occlude(bx, by, px, py, ForcedRandom(), 25, 105, 68)
        self.assertTrue((np.isfinite(px) & np.isfinite(py)).any(axis=1).all())
        self.assertTrue(np.isnan(px[:2, 0]).all())

    def test_no_ball_or_no_players_is_safe(self):
        for missing in ('ball', 'players'):
            match = match_fixture()
            keys = ('bx', 'by') if missing == 'ball' else ('px', 'py')
            for key in keys:
                match[key][:] = np.nan
            out = A.transform(match, ITEM, 'stress')
            for key in keys:
                self.assertTrue(np.isnan(out[key]).all())

    def test_unknown_view_and_invalid_frequency_fail(self):
        match = match_fixture()
        with self.assertRaises(ValueError):
            A.transform(match, ITEM, '../anything')
        with self.assertRaises(ValueError):
            A.transform(dict(match, fps=0), ITEM, 'stress')


class SourceAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / 'altre_partite'
        self.patches = [patch.object(A.dati, 'BASE', str(self.base)),
                        patch.object(A.dati, 'CACHE', str(self.root / 'cache'))]
        for p in self.patches:
            p.start()
        A._FILE_HASHES.clear()
        for path in A._input_files(ITEM):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name, encoding='utf-8')
        self.clean_tracking = A._labels_input(ITEM) / f'{ITEM[1]}_atd.txt'
        self.clean_tracking.write_text('must never be used as input', encoding='utf-8')

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        A._FILE_HASHES.clear()
        self.tmp.cleanup()

    def test_input_routing_and_provenance_exclude_clean_tracking(self):
        self.assertEqual(A._input(ITEM).parent.name, 'dfl_sporcate')
        self.assertEqual(A._labels_input(ITEM).parent.name, 'dfl_pulite')
        manifest = A.provenance(ITEM)
        self.assertEqual(manifest['tracking'], 'sporcate')
        files = tuple(manifest['inputs_sha256'])
        self.assertEqual(sum('_pulite/' in p for p in files), 1)
        self.assertTrue(next(p for p in files if '_pulite/' in p).endswith('_possession_gt.csv'))
        key = A._cache_key(ITEM)
        self.clean_tracking.write_text('changed clean coordinates', encoding='utf-8')
        self.assertEqual(key, A._cache_key(ITEM))

    def test_key_depends_on_input_contents_not_download_time(self):
        path = A._input_files(ITEM)[0]
        key = A._cache_key(ITEM)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertEqual(key, A._cache_key(ITEM))
        path.write_text('new dirty coordinates', encoding='utf-8')
        self.assertNotEqual(key, A._cache_key(ITEM))
        key = A._cache_key(ITEM)
        A._input_files(ITEM)[2].write_text('new labels', encoding='utf-8')
        self.assertNotEqual(key, A._cache_key(ITEM))

    def test_build_reads_dirty_coordinates_and_clean_labels(self):
        match = match_fixture()

        def features(m):
            return pd.DataFrame({'synthetic_x': m['bx']})

        with patch.object(A.C.fc, 'read_epts', return_value=match) as reader, \
                patch.object(A.C, 'features', side_effect=features):
            A._build(ITEM)
        reader.assert_called_once_with(A._input(ITEM), labels_folder=A._labels_input(ITEM))
        frame = A.load(ITEM)
        np.testing.assert_array_equal(frame['synthetic_x'], match['bx'])
        np.testing.assert_array_equal(frame['y'], [A.K.CODE[c] for c in match['labels']])
        self.assertTrue((A.cache_path(ITEM, 'originale').parent / 'provenienza.json').is_file())
        for view in (*A.TRAIN_VIEWS, A.TEST_VIEW):
            got = A.load(ITEM, view)
            np.testing.assert_array_equal(got[['frame', 'period', 'y']], frame[['frame', 'period', 'y']])

    def test_reject_target_match_invalid_view_and_workers(self):
        with self.assertRaises(ValueError):
            A._input(('barca', 'BAR-ATH'))
        with self.assertRaises(ValueError):
            A.cache_path(ITEM, '../anything')
        with self.assertRaises(ValueError):
            A.prepare([], workers=0)


if __name__ == '__main__':
    unittest.main()
