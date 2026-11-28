import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.preprocess import (
    build_index_maps,
    build_items,
    cut_time_window,
    filter_positive,
    k_core,
    load_raw,
    report_stats,
)


class PreprocessTests(unittest.TestCase):
    def test_load_raw_uses_typed_columns_and_keeps_positive_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            pd.DataFrame({
                "user_id": [1, 1],
                "app_id": [10, 11],
                "is_recommended": [True, False],
                "hours": [1.5, 2.5],
                "date": ["2022-01-01", "2022-01-02"],
            }).to_csv(raw_dir / "recommendations.csv", index=False)

            loaded = load_raw(raw_dir)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(str(loaded["user_id"].dtype), "int32")
        self.assertEqual(str(loaded["hours"].dtype), "float32")
        self.assertTrue(loaded["is_recommended"].all())

    def test_positive_deduplication_keeps_latest_and_time_window(self):
        df = pd.DataFrame({
            "user_id": [1, 1, 1, 2],
            "app_id": [10, 10, 11, 10],
            "is_recommended": [True, True, False, True],
            "hours": [1.0, 3.0, 4.0, 2.0],
            "date": ["2021-01-01", "2022-01-01", "2023-01-01", "2020-01-01"],
        })

        positive = filter_positive(df)
        windowed = cut_time_window(positive, "2021-01-01")

        self.assertEqual(
            windowed[["user_id", "app_id", "hours"]].values.tolist(),
            [[1, 10, 3.0]],
        )

    def test_k_core_repeats_until_user_and_item_counts_are_stable(self):
        df = pd.DataFrame({
            "user_id": [1, 1, 2, 2, 3, 3, 4],
            "app_id":  [10, 11, 10, 11, 11, 12, 12],
        })

        result = k_core(df, k=2)

        self.assertEqual(
            result[["user_id", "app_id"]].values.tolist(),
            [[1, 10], [1, 11], [2, 10], [2, 11]],
        )

    def test_indices_are_contiguous_and_maps_include_game_titles(self):
        interactions = pd.DataFrame({
            "user_id": [20, 10],
            "app_id": [200, 100],
        })
        indexed, maps = build_index_maps(interactions)
        self.assertEqual(indexed["u_idx"].tolist(), [1, 0])
        self.assertEqual(indexed["i_idx"].tolist(), [1, 0])
        self.assertEqual(maps["i_idx_to_app_id"], [100, 200])

        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            pd.DataFrame({
                "app_id": [100, 200],
                "title": ["Game A", "Game B"],
                "date_release": ["2020-01-01", "2021-01-01"],
                "win": [True, True],
                "mac": [False, False],
                "linux": [False, False],
                "price_final": [1.0, 2.0],
            }).to_csv(raw_dir / "games.csv", index=False)
            (raw_dir / "games_metadata.json").write_text(
                "\n".join([
                    json.dumps({
                        "app_id": 100, "description": "A", "tags": ["Action"]
                    }),
                    json.dumps({
                        "app_id": 200, "description": "B", "tags": ["RPG"]
                    }),
                ]),
                encoding="utf-8",
            )

            items = build_items(raw_dir, maps["i_idx_to_app_id"])
            maps["i_idx_to_title"] = (
                items.sort_values("i_idx")["title"].tolist()
            )

        self.assertEqual(maps["i_idx_to_title"], ["Game A", "Game B"])
        self.assertEqual(items["i_idx"].tolist(), [0, 1])

    def test_stats_report_sparsity_before_and_after_core(self):
        before = pd.DataFrame({
            "user_id": [1, 1, 2, 3],
            "app_id": [10, 11, 10, 11],
        })
        after = before.iloc[[0, 2, 3]]

        stats = report_stats(before, after)

        self.assertEqual(stats["stage"].tolist(), ["before_k_core", "after_k_core"])
        self.assertAlmostEqual(stats["sparsity"].iloc[0], 1 / 3)
        self.assertEqual(stats["sparsity"].iloc[1], 0.5)


if __name__ == "__main__":
    unittest.main()
