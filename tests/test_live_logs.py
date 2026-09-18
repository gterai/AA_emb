"""Check live delivery, separate saved logs, and propagation of training failures."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_runner', ROOT/'training/run_paper.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LiveLogs(unittest.TestCase):
    def test_live_output_and_both_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            marker = run/'received'
            class LiveTerminal(io.StringIO):
                def write(self, text):
                    result = super().write(text)
                    if 'started' in text:
                        marker.touch()
                    return result
            out, err = LiveTerminal(), io.StringIO()
            # The child cannot finish until its first line reaches the terminal.
            code = """import sys,time
from pathlib import Path
print('started')
end=time.monotonic()+10
while not Path(sys.argv[1]).exists():
    if time.monotonic()>end: raise RuntimeError('output was not streamed live')
    time.sleep(.01)
for i in range(2000):
    print('out',i)
    print('err',i,file=sys.stderr)
"""
            with patch.object(runner.sys, 'stdout', out), patch.object(runner.sys, 'stderr', err):
                runner.run_with_live_logs([sys.executable, '-c', code, str(marker)], run, dict(os.environ))
            self.assertEqual(out.getvalue(), (run/'stdout.txt').read_text())
            self.assertEqual(err.getvalue(), (run/'stderr.txt').read_text())
            self.assertIn('out 1999', out.getvalue())
            self.assertIn('err 1999', err.getvalue())

    def test_failure_is_visible_and_propagated(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            out, err = io.StringIO(), io.StringIO()
            with patch.object(runner.sys, 'stdout', out), patch.object(runner.sys, 'stderr', err):
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    runner.run_with_live_logs([sys.executable, '-c',
                        "import sys; print('failed',file=sys.stderr); sys.exit(7)"], run, dict(os.environ))
            self.assertEqual(caught.exception.returncode, 7)
            self.assertEqual(err.getvalue(), 'failed\n')
            self.assertEqual((run/'stderr.txt').read_text(), 'failed\n')
