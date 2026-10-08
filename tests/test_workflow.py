"""Checks for fixed-partition validation and run-command construction (no GPU)."""
import importlib.util
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validate_benchmarks', ROOT/'tests/check_data.py')
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
        for analysis, count in [('primary', 150), ('protein', 90), ('function', 120), ('pfam', 100)]:
            with tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, str(ROOT/'training/run_paper.py'),
                    analysis, '--dry-run', '--output', str(Path(directory)/'runs')],
                    text=True, capture_output=True, check=True)
                commands = [line for line in result.stdout.splitlines() if '--metrics_tsv' in line]
                self.assertEqual(len(commands), count)
                for command in commands:
                    tokens = shlex.split(command)
                    self.assertTrue(Path(tokens[1]).is_file())
                    self.assertTrue(Path(tokens[tokens.index('--input_class_fname') + 1]).is_file())
                self.assertTrue(all("--model_fname ''" in line for line in commands))
                if analysis == 'protein':
                    self.assertEqual(sum('--abl_type m' in line for line in commands), 30)
                elif analysis == 'pfam':
                    self.assertEqual(sum('--emb_name emb_T5u' in line for line in commands), 50)
                    self.assertTrue(all('--abl_type' not in line for line in commands))
                    self.assertTrue(all('partitions/pfam_holdout/' in line for line in commands))
                    for command in commands:
                        tokens = shlex.split(command)
                        self.assertEqual(tokens[tokens.index('--epoch') + 1], '200')
                        self.assertEqual(tokens[tokens.index('--lr') + 1], '0.0001')
                        self.assertEqual(tokens[tokens.index('--s_bat') + 1], '100')
                elif analysis == 'primary':
                    self.assertEqual(sum('--abl_type m' in line for line in commands), 60)
                    self.assertEqual(sum('--abl_type p' in line for line in commands), 10)
                    self.assertTrue(all('partitions/mrna_c80/' in line for line in commands))
                else:
                    self.assertTrue(all('--abl_type m' not in line for line in commands))

    def test_pfam_selection_and_invalid_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(ROOT/'training/run_paper.py'), 'pfam',
                       '--groups', 'PF00018', '--seeds', '0', '--device', 'cuda',
                       '--output', directory, '--dry-run']
            result = subprocess.run(command, text=True, capture_output=True, check=True)
            lines = [line for line in result.stdout.splitlines() if '--metrics_tsv' in line]
            self.assertEqual(len(lines), 2)
            self.assertTrue(all('PF00018/seed_0/class.txt' in line for line in lines))
            self.assertTrue(all('--device cuda' in line for line in lines))
            invalid = subprocess.run(command + ['--conditions', 't5u_only'], text=True, capture_output=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn('Invalid or duplicate conditions', invalid.stderr)
            (Path(directory)/'mrna_only/PF00018/seed_0').mkdir(parents=True)
            existing = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(existing.returncode, 0)
            self.assertIn('Run already exists', existing.stderr)

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'mrna_only/c50/seed_0').mkdir(parents=True)
            result = subprocess.run([sys.executable, str(ROOT/'training/run_paper.py'), 'protein',
                '--dry-run', '--groups', 'c50', '--seeds', '0', '--conditions', 'mrna_only',
                '--output', directory], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Run already exists', result.stderr)


if __name__ == '__main__':
    unittest.main()
