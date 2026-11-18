"""Tests for experimental rolling temporal split boundaries."""
import unittest
import pandas as pd
from src.data.build_rolling_evaluation_splits import rolling_splits


class RollingSplitTests(unittest.TestCase):
    def test_overlapping_folds_but_disjoint_within_fold(self):
        day = 86_400_000
        df = pd.DataFrame({
            "evaluation_id": range(14),
            "user_id": [1] * 14,
            "time_ms": [i * day + 1000 for i in range(14)],
        })
        folds = list(rolling_splits(df, train_days=5, validation_days=2,
                                    test_days=2, step_days=1))
        self.assertGreaterEqual(len(folds), 2)
        for _, splits, manifest in folds:
            self.assertEqual(manifest["counts"], {"train": 5, "validation": 2, "test": 2})
            a, b, c = [set(s.evaluation_id) for s in splits.values()]
            self.assertFalse(a & b or b & c or a & c)
        self.assertTrue(set(folds[0][1]["train"].evaluation_id) &
                        set(folds[1][1]["train"].evaluation_id))

    def test_embargo_and_no_complete_folds(self):
        day = 86_400_000
        df = pd.DataFrame({"evaluation_id": range(10), "user_id": [1]*10,
                           "time_ms": [i*day+1000 for i in range(10)]})
        folds = list(rolling_splits(df, train_days=3, validation_days=2,
                                    test_days=2, step_days=1, embargo_hours=24))
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0][2]["counts"], {"train": 3, "validation": 2, "test": 2})

    def test_reject_duplicate_ids(self):
        df = pd.DataFrame({"evaluation_id": [1, 1], "user_id": [1, 1],
                           "time_ms": [1, 2]})
        with self.assertRaises(ValueError):
            list(rolling_splits(df))


if __name__ == "__main__":
    unittest.main()
