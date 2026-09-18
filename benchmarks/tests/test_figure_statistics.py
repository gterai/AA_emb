"""Check manuscript multiplicity families and reference figure input validation."""
import sys
from pathlib import Path
import unittest
import tempfile
import csv
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from summarize_go_slim_holdout import holm_adjust
from plot_recent_models import read_values

class FigureStatistics(unittest.TestCase):
    def test_holm_step_down_preserves_order(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.01, 0.005, 0.9]),
                                   [0.08, 0.03, 0.02, 0.9])
        np.testing.assert_allclose(holm_adjust([0.02, 0.021, 0.9]), [0.06, 0.06, 0.9])

    def test_main_reference_family(self):
        for metric in ['pearson', 'spearman']:
            path = ROOT / f'reference_results/main_comparison/FigMain_{metric}_paired_ttests.tsv'
            with path.open() as f: rows = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual(len(rows), 8)
            np.testing.assert_allclose([float(r['p_adj']) for r in rows],
                                       holm_adjust([float(r['p_raw']) for r in rows]))
            for r in rows:
                p = float(r['p_adj'])
                self.assertEqual(r['significance'], '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'n.s.')

    def test_recent_models_reject_duplicate_seed(self):
        source = ROOT / 'reference_results/recent_models/per_seed_summary.tsv'
        self.assertEqual(len(read_values(source)), 5)
        with source.open() as f: rows = list(csv.DictReader(f, delimiter='\t'))
        rows[1]['seed'] = rows[0]['seed']
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.tsv'
            with p.open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
                w.writeheader(); w.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'expected seeds'):
                read_values(p)
