#!/usr/bin/env python3
"""Regression tests for manuscript masked dataset-level loss aggregation."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "training"
sys.path.insert(0, str(EVALUATION))
os.environ.setdefault("HOME", str(Path.home()))
spec = importlib.util.spec_from_file_location(
    "eval_multiemb", EVALUATION / "eval_multiemb.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class MaskedLossAggregationTest(unittest.TestCase):
    def test_dataset_mean_uses_valid_target_counts(self):
        # Batch 1: three observed targets with total loss 6.
        loss1 = torch.tensor([[1.0, 2.0], [3.0, 100.0]])
        mask1 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        sum1, count1, mean1 = module.masked_loss_statistics(loss1, mask1)

        # Batch 2: one observed target with total loss 10.
        loss2 = torch.tensor([[10.0, 100.0]])
        mask2 = torch.tensor([[0.0, 1.0]])
        sum2, count2, mean2 = module.masked_loss_statistics(loss2, mask2)

        corrected = (sum1 + sum2) / (count1 + count2)
        self.assertEqual(corrected.item(), 4.0)

        # The former sample-count weighting gives a different value (14/3).
        former = (mean1 * 2 + mean2 * 1) / 3
        self.assertAlmostEqual(former.item(), 14.0 / 3.0, places=6)
        self.assertNotEqual(corrected.item(), former.item())

    def test_all_missing_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no observed target"):
            module.masked_loss_statistics(
                torch.ones((2, 2)), torch.ones((2, 2))
            )


if __name__ == "__main__":
    unittest.main()
