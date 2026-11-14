"""Independent, exhaustive checks of duration decoding and official scoring parity."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modelli_mv'))
import modello as M
import esperimento as E


class DecoderTests(unittest.TestCase):
    def test_duration_one_can_change_state(self):
        p = np.array([[.99, .005, .005], [.005, .99, .005], [.005, .005, .99]])
        params = dict(turn=0., stop=0., rest=0., m_pos=1, m_dead=1)
        np.testing.assert_array_equal(M.decode(p, np.ones(3), np.zeros(3, bool), params), [0, 1, 2])

    def test_optimal_cost_matches_exhaustive_paths(self):
        rng = np.random.default_rng(42)
        for m_pos, m_dead in ((1, 1), (2, 3), (3, 2)):
            for n in (1, 4, 6):
                cost = rng.uniform(0, 3, (n, 3))
                params = dict(turn=.3, stop=.7, rest=.5, m_pos=m_pos, m_dead=m_dead)
                pen = M.penalties(n, params)
                def objective(path):
                    cuts = np.r_[0, np.flatnonzero(np.diff(path)) + 1, n]
                    for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                        # Both boundary spells are censored by the observation window.
                        if j not in (0, len(cuts) - 2) and b - a < (m_dead if path[a] == 2 else m_pos):
                            return np.inf
                    return sum(cost[i, s] for i, s in enumerate(path)) + sum(
                        pen[i, path[i - 1], path[i]] for i in range(1, n) if path[i - 1] != path[i])
                pred = M._viterbi(cost, pen, m_pos, m_dead, m_pos, 0.)
                best = min(objective(z) for z in itertools.product(range(3), repeat=n))
                self.assertAlmostEqual(objective(pred), best, places=10)

    def test_gap_is_never_overwritten(self):
        p = np.full((7, 3), .01)
        p[:, 1] = .98
        period = np.array([1, 1, 0, 0, 1, 2, 2])
        pred = M.decode(p, period, np.zeros(7, bool), dict(turn=1, stop=1, rest=1, m_pos=2, m_dead=2))
        np.testing.assert_array_equal(pred, [1, 1, 3, 3, 1, 1, 1])

    def test_bad_inputs_fail(self):
        params = dict(turn=1, stop=1, rest=1, m_pos=0, m_dead=2)
        with self.assertRaises(ValueError):
            M.decode(np.ones((2, 3)) / 3, np.ones(2), np.zeros(2, bool), params)
        with self.assertRaises(ValueError):
            M.to_codes(np.array([-1, 0]))

    def test_fast_scorer_matches_official(self):
        rng = np.random.default_rng(2027)
        for _ in range(40):
            gt = np.repeat(rng.integers(0, 4, 35), rng.integers(1, 20, 35)).astype(np.int8)
            pred = rng.integers(0, 4, len(gt), dtype=np.int8)
            official = M.pm.evaluate(gt, pred)['spell_f1_mean']
            self.assertAlmostEqual(M.spell_f1_fast(gt, pred), official, places=14)

    def test_half_separator(self):
        y = np.array([0, 0, 0, 0], dtype=np.int8)
        period = np.array([1, 1, 2, 2])
        self.assertEqual(M.score_periods(y, y, period, full=True)['n_gt_spells'], 2)

    def test_fold_membership_order_invariant(self):
        items = [('dfl', str(i)) for i in range(7)] + [('metrica', str(i)) for i in range(3)]
        self.assertEqual(E.folds(items), E.folds(list(reversed(items))))


if __name__ == '__main__':
    unittest.main()
