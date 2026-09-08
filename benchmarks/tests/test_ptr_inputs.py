"""PTR parsing and publication checks, using synthetic data only."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ptr_validation', ROOT/'scripts/ptr_validation.py')
ptr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ptr)


class PTRInputs(unittest.TestCase):
    def source_text(self):
        header = ['EnsemblTranscriptID'] + [f'T{i}_PTR' for i in range(29)]
        return '\t'.join(header)+'\n'+'\t'.join(['ENST1.2','NA']+['1.25']*28)+'\n'

    def test_zip_and_tsv_preserve_log_values_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            tsv = Path(d)/'Table_EV3.tsv';tsv.write_text(self.source_text())
            archive = Path(d)/'source.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('Table_EV3/Table_EV3.tsv',self.source_text())
            a, labels, _ = ptr.load_table(tsv);b, labels2, _ = ptr.load_table(archive)
            self.assertEqual(labels,labels2)
            np.testing.assert_equal(a['ENST1'],b['ENST1'])
            self.assertTrue(np.isnan(a['ENST1'][0]))
            self.assertEqual(a['ENST1'][1],1.25)

    def test_duplicate_stable_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'Table_EV3.tsv';text=self.source_text();p.write_text(text+text.splitlines()[1]+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):ptr.load_table(p)

    def test_distributed_predictions_have_no_source_measurements(self):
        for threshold,n in [('c50',4687),('c70',4679),('c90',4668)]:
            with (ROOT/f'reference_predictions/ptr_validation/{threshold}.tsv').open() as f:
                reader=csv.DictReader(f,delimiter='\t')
                self.assertEqual(reader.fieldnames,['transcript_id','mrna_only_median_te','mrna_t5u_median_te'])
                self.assertEqual(len(list(reader)),n)


if __name__=='__main__':unittest.main()
