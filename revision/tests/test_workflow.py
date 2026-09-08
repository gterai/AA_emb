"""Checks for fixed-partition publication and run-command construction (no GPU)."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validate_revision', ROOT/'scripts/validate_revision.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class WorkflowTests(unittest.TestCase):
    def test_all_published_splits_and_checksums(self):
        self.assertEqual(len(validation.validate_assets()), 9926)

    def test_duplicate_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'split.txt'
            path.write_text('a train\na test\nb validation\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate ID'):
                validation.read_split(path)

    def test_run_matrix_and_protein_only_ablation(self):
        for analysis, count in [('protein', 90), ('function', 120)]:
            with tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, str(ROOT/'scripts/run_revision.py'),
                    analysis, '--dry-run', '--output', str(Path(directory)/'runs')],
                    text=True, capture_output=True, check=True)
                commands = [line for line in result.stdout.splitlines() if '--metrics_tsv' in line]
                self.assertEqual(len(commands), count)
                self.assertTrue(all("--model_fname ''" in line for line in commands))
                if analysis == 'protein':
                    self.assertEqual(sum('--abl_type m' in line for line in commands), 30)
                else:
                    self.assertTrue(all('--abl_type m' not in line for line in commands))

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'mrna_only/c50/seed_0').mkdir(parents=True)
            result = subprocess.run([sys.executable, str(ROOT/'scripts/run_revision.py'), 'protein',
                '--dry-run', '--groups', 'c50', '--seeds', '0', '--conditions', 'mrna_only',
                '--output', directory], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Run already exists', result.stderr)


if __name__ == '__main__':
    unittest.main()
