"""Ensure strict validation covers every reference field without real embeddings."""
import ast
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import check_data


class InputHashTests(unittest.TestCase):
    def test_all_reference_fields_and_optional_strict_mode(self):
        reference = json.loads((check_data.ROOT / 'prepare_input/input_reference.json').read_text())
        record = {key: np.zeros(ast.literal_eval(next(iter(shapes))), dtype=np.float32)
                  for key, shapes in reference['key_shapes'].items()}
        self.assertEqual(len(record), 12)
        hashes = {}
        for key, array in record.items():
            payload = ('test_id' + str(array.dtype) + str(array.shape)).encode() + array.tobytes()
            hashes[key] = hashlib.sha256(payload).hexdigest()
        reference['key_content_sha256'] = hashes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'prepare_input').mkdir()
            (root / 'partitions/metadata').mkdir(parents=True)
            (root / 'prepare_input/input_reference.json').write_text(json.dumps(reference))
            (root / 'partitions/metadata/cohort.tsv').write_text('transcript_id_original\ntest_id\n')
            path = root / 'input.pkl.gz'

            def check(current, strict=True):
                with gzip.open(path, 'wb') as handle:
                    pickle.dump(({'test_id': current}, reference['te_columns']), handle)
                with patch.object(check_data, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                    check_data.validate_input(path, strict_hashes=strict)

            check(record)
            for key in record:
                with self.subTest(key=key):
                    changed = {**record, key: record[key].copy()}
                    changed[key].flat[0] = 1
                    with self.assertRaisesRegex(ValueError, f'{key}: differs from reference'):
                        check(changed)
            missing = {key: value for key, value in record.items() if key != 'emb_dipep'}
            with self.assertRaisesRegex(ValueError, 'missing reference fields: emb_dipep'):
                check(missing)
            core = {key: record[key] for key in ('oht', 'TE', 'y_mask', 'emb_T5u')}
            check(core, strict=False)
            typed = {**record, 'emb_aacom': record['emb_aacom'].astype(np.float64)}
            with self.assertRaisesRegex(ValueError, 'emb_aacom: differs from reference'):
                check(typed)
            reshaped = {**record, 'emb_dipep': record['emb_dipep'].reshape(20, 20)}
            with self.assertRaisesRegex(ValueError, 'emb_dipep: differs from reference'):
                check(reshaped)
