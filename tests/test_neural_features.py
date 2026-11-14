"""Synthetic regression tests: no competition data or ground truth is read."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modelli_mv'))
import caratteristiche as C
import formato_comune as fc
import rete


def match(n=40):
    t = np.arange(n, dtype=np.float32)
    return dict(period=np.ones(n, dtype=np.int8), team=np.array([0, 0, 1, 1]),
                px=np.tile([-40., -10., 10., 40.], (n, 1)).astype(np.float32) + t[:, None] * .01,
                py=np.tile([-5., 3., -2., 6.], (n, 1)).astype(np.float32),
                bx=-10. + t * .01, by=np.full(n, 3., np.float32),
                length=105., width=68., fps=25)


def table(n=40):
    m = match(n)
    frame = C.features(m)
    frame['period'] = m['period']
    frame['y'] = np.zeros(n, dtype=np.int64)
    return frame, [col for col in frame if col not in ('period', 'y')]


class FeatureTests(unittest.TestCase):
    def test_swap_matches_recomputed_geometry_and_is_involution(self):
        original = match()
        swapped = copy.deepcopy(original)
        swapped['team'] = 1 - swapped['team']
        before = C.features(original)
        expected = C.features(swapped)
        actual = C.swap_teams(before)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(C.swap_teams(actual), before, equal_nan=True)

    def test_speed_does_not_cross_missing_slot_or_period_break(self):
        x = np.arange(20, dtype=np.float32)
        period = np.ones(20, dtype=np.int8)
        x[8] = np.nan
        self.assertTrue(np.isnan(C._speed(x, period)[7]))
        self.assertTrue(np.isfinite(C._speed(x, period, legacy=True)[7]))
        x[8] = 8
        period[8] = 0
        self.assertTrue(np.isnan(C._speed(x, period)[7]))

    def test_features_do_not_bridge_no_tracking_gap(self):
        m = match()
        m['px'][10] = m['py'][10] = np.nan
        m['bx'][10:15] = m['by'][10:15] = np.nan
        current = C.features(m)
        legacy = C.features(m, legacy=True)
        self.assertTrue(np.isnan(current.loc[11, 'bx_ff']))
        self.assertTrue(np.isfinite(legacy.loc[11, 'bx_ff']))
        self.assertTrue(np.isnan(current.loc[11, 'ultimo_tocco']))

    def test_warning_filters_are_unchanged(self):
        before = list(warnings.filters)
        C.features(match())
        self.assertEqual(warnings.filters, before)


class NetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, cols, version):
        net = rete.Net(2 * len(cols), c=4)
        with torch.no_grad():
            for p in net.parameters():
                p.zero_()
            net.out.bias.copy_(torch.tensor([2., 0., -1.]))
        return dict(net=net.eval(), med=np.zeros(len(cols), np.float32),
                    iqr=np.ones(len(cols), np.float32), cols=cols, seed=7,
                    inference_version=version)

    def test_short_segment_and_unsampled_tail_are_predicted(self):
        frame, cols = table(18)
        model = self.model(cols, 2)
        actual = rete.predict(model, frame, ('dfl', 'synthetic'))
        expected = torch.softmax(torch.tensor([2., 0., -1.]), 0).numpy()
        expected = (expected + expected[[1, 0, 2]]) / 2
        np.testing.assert_allclose(actual, np.tile(expected, (len(frame), 1)), atol=1e-7)
        legacy = rete.predict(self.model(cols, 1), frame, ('dfl', 'synthetic'))
        np.testing.assert_allclose(legacy, np.full_like(legacy, 1 / 3))

    def test_legacy_tail_stays_compatible(self):
        frame, cols = table(1504)
        legacy = rete.predict(self.model(cols, 1), frame, ('dfl', 'synthetic'))
        modern = rete.predict(self.model(cols, 2), frame, ('dfl', 'synthetic'))
        np.testing.assert_allclose(legacy[-3:], np.full((3, 3), 1 / 3), atol=1e-7)
        self.assertGreater(float(modern[-1, 0]), .45)

    def test_raw_data_never_reuses_another_view(self):
        frame, cols = table()
        first = rete._raw(frame, ('dfl', 'same'), cols)[0][0]
        changed = frame.copy()
        changed['bx'] = 42.
        second = rete._raw(changed, ('dfl', 'same'), cols)[0][0]
        self.assertFalse(np.array_equal(first[:, cols.index('bx')], second[:, cols.index('bx')]))

    def test_segments_split_on_observed_gap_without_using_labels(self):
        frame, cols = table(30)
        frame.loc[10:14, ['n_casa', 'n_ospiti', 'palla_vista']] = 0
        original = rete.raw_sequences(frame, cols, 1.)
        frame['y'] = 3
        hidden_labels = rete.raw_sequences(frame, cols, 1.)
        self.assertEqual(len(original), 2)
        for a, b in zip(original, hidden_labels):
            np.testing.assert_array_equal(a[-1], b[-1])
        self.assertLess(original[0][-1][-1], 10)
        self.assertGreaterEqual(original[1][-1][0], 15)

    def test_halo_is_context_only_and_short_sequences_are_padded(self):
        x = np.arange(60, dtype=np.float32).reshape(20, 3)
        y = np.arange(20) % 3
        w = np.ones(20, np.float32)
        xc, yc, wc = rete.training_crop(x, y, w, 5, length=6, context=2)
        np.testing.assert_array_equal(xc, x[3:13])
        np.testing.assert_array_equal(yc[2:8], y[5:11])
        np.testing.assert_array_equal(wc, [0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
        xc, yc, wc = rete.training_crop(x[:3], y[:3], w[:3], 0, length=6, context=2)
        self.assertEqual(xc.shape, (10, 3))
        self.assertEqual(float(wc.sum()), 3.)
        self.assertTrue(np.isnan(xc[:2]).all())
        self.assertTrue((yc[5:] == 3).all())

    def test_normalizer_handles_all_missing_and_infinite_columns(self):
        med, iqr = rete.normalizer([np.array([[np.nan, 1., np.inf], [np.nan, 3., 2.]])])
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue((iqr > 0).all())
        self.assertEqual(med[0], 0.)
        self.assertEqual(med[2], 2.)

    def test_checkpoint_round_trip_preserves_version_and_architecture(self):
        frame, cols = table()
        model = self.model(cols, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.pt'
            rete.save(model, path)
            loaded = rete.load(path)
        self.assertEqual(loaded['inference_version'], 2)
        self.assertEqual(loaded['net'].inp.out_channels, 4)
        np.testing.assert_array_equal(rete.predict(model, frame, ('dfl', 's')),
                                      rete.predict(loaded, frame, ('dfl', 's')))


class LoaderTests(unittest.TestCase):
    def fixture(self, folder, *, rate=25, rows=None, labels='frame,label\n0,H\n'):
        xml = '''<main xmlns="urn:test"><Team id="h"><Name>Home</Name></Team>
        <Team id="a"><Name>Away</Name></Team><Player teamId="h"/><Player teamId="a"/>
        <DataFormatSpecification endFrame="3"/><FieldSize><Width>105</Width><Height>68</Height></FieldSize>
        <FrameRate>{rate}</FrameRate><ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
        <ProviderParameter><Name>first_half_end</Name><Value>3</Value></ProviderParameter></main>'''.format(rate=rate)
        Path(folder, 'test_atd.xml').write_text(xml)
        if rows is None:
            rows = ['{}:0.1,0.2,0;0.8,0.9,0:0.5,0.5,0'.format(i) for i in range(1, 4)]
        Path(folder, 'test_atd.txt').write_text('\n'.join(rows))
        Path(folder, 'test_possession_gt.csv').write_text(labels)

    def test_namespaced_xml_and_single_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            m = fc.read_epts(tmp)
        self.assertEqual(m['home'], 'Home')
        np.testing.assert_array_equal(m['labels'], ['H', 'X', 'X'])
        np.testing.assert_array_equal(m['period'], [1, 1, 1])

    def test_duplicate_and_out_of_range_tracking_frames_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for frames in ((1, 1), (0, 1), (1, 4)):
                self.fixture(tmp, rows=['{}:0,0,0;1,1,0:0.5,0.5,0'.format(i) for i in frames])
                with self.assertRaises(ValueError):
                    fc.read_epts(tmp)

    def test_invalid_labels_or_fps_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for labels in ('frame,label\n0,Q\n', 'frame,label\n0,H\n0,A\n', 'frame,label\n-1,H\n'):
                self.fixture(tmp, labels=labels)
                with self.assertRaises(ValueError):
                    fc.read_epts(tmp)
            self.fixture(tmp, rate=10)
            with self.assertRaises(ValueError):
                fc.read_epts(tmp)


if __name__ == '__main__':
    unittest.main()
