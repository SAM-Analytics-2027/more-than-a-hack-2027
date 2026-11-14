"""The acceptance checks must reject malformed or non-reproducible CSVs."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'modelli_mv'))
import modello as M
import riproduci as R
import valida_risultati_augmentation as V


class OutputValidationTests(unittest.TestCase):
    def fixture(self, folder):
        labels = np.array(['H'] * 30 + ['D'] * 20 + ['A'] * 30 + ['H'] * 30)
        table = pd.DataFrame({'frame': np.arange(len(labels)), 'label': labels})
        gt, csv, xml = (Path(folder) / name for name in ('gt.csv', 'pred.csv', 'tracking.xml'))
        table.to_csv(gt, index=False)
        table.to_csv(csv, index=False)
        xml.write_text('<Root><ProviderParameter><Name>first_half_start</Name><Value>1</Value>'
                       '</ProviderParameter><ProviderParameter><Name>first_half_end</Name>'
                       '<Value>110</Value></ProviderParameter></Root>')
        expected = dict(spellF1=1., macroF1=1., predetti=3, possessi=3, abbinati50=3)
        return csv, gt, xml, expected

    def test_official_dashboard_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            result = V.validate_csv(*self.fixture(temp))
        self.assertTrue(result['ok'])
        self.assertEqual(result['rows'], 110)

    def test_corrupt_frame_order_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            csv, gt, xml, expected = self.fixture(temp)
            frame = pd.read_csv(csv)
            frame.loc[0, 'frame'] = 1
            frame.to_csv(csv, index=False)
            with self.assertRaises(ValueError):
                V.validate_csv(csv, gt, xml, expected)

    def test_stale_report_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            csv, gt, xml, expected = self.fixture(temp)
            expected['spellF1'] = .99
            with self.assertRaises(ValueError):
                V.validate_csv(csv, gt, xml, expected)

    def test_regeneration_default_never_requests_public_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text('{}')
            csv = root / 'model.csv'
            csv.write_text('frame,label\n0,H\n')
            with patch.object(R, 'model_dir', return_value=str(root)), patch.object(R, 'RESULTS', str(root)), \
                 patch.object(V.P, '_verify_artifacts'), patch.object(R, 'predict', return_value={}) as predict:
                result = V.validate_model('model')
            self.assertTrue(result['ok'])
            self.assertFalse(predict.call_args.kwargs['report_public'])

    def test_failed_regeneration_restores_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text('{}')
            csv = root / 'model.csv'
            original = b'frame,label\n0,H\n'
            csv.write_bytes(original)
            def regenerate(**kwargs):
                csv.write_text('frame,label\n0,A\n')
                return {}
            with patch.object(R, 'model_dir', return_value=str(root)), patch.object(R, 'RESULTS', str(root)), \
                 patch.object(V.P, '_verify_artifacts'), patch.object(R, 'predict', side_effect=regenerate):
                with self.assertRaises(ValueError):
                    V.validate_model('model')
            self.assertEqual(csv.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
