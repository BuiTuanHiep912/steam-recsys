import unittest

import pandas as pd

from src.data.split import find_cutoffs, sanity_check, split_by_time


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.interactions = pd.DataFrame({
            "u_idx": [0] * 100,
            "i_idx": list(range(100)),
            "hours": [1.0] * 100,
            "ts": list(range(100)),
        })

    def test_cutoffs_achieve_requested_interaction_ratios(self):
        cut_train, cut_valid = find_cutoffs(self.interactions, 0.8, 0.1)
        train, valid, test = split_by_time(
            self.interactions, cut_train, cut_valid
        )

        self.assertEqual((cut_train, cut_valid), (80, 90))
        self.assertEqual(
            (len(train), len(valid), len(test)),
            (80, 10, 10),
        )
        self.assertGreaterEqual(test["ts"].min(), cut_valid)
        sanity_check(train, valid, test, cut_train, cut_valid)

    def test_cutoffs_keep_same_timestamp_rows_together(self):
        tied = pd.DataFrame({
            "u_idx": range(100),
            "i_idx": range(100),
            "ts": [0] * 80 + [1] * 10 + [2] * 10,
        })

        cut_train, cut_valid = find_cutoffs(tied, 0.8, 0.1)
        train, valid, test = split_by_time(tied, cut_train, cut_valid)

        self.assertEqual((cut_train, cut_valid), (1, 2))
        self.assertEqual(
            (len(train), len(valid), len(test)),
            (80, 10, 10),
        )

    def test_sanity_check_rejects_test_interaction_before_cutoff(self):
        train, valid, test = split_by_time(self.interactions, 80, 90)
        early_test = test.assign(ts=89)

        with self.assertRaisesRegex(ValueError, "test"):
            sanity_check(train, valid, early_test, 80, 90)


if __name__ == "__main__":
    unittest.main()
