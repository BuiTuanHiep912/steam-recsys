"""Tiền xử lý dữ liệu thô Steam. NGƯỜI A — ngày 3-5.

Đầu vào : data/raw/recommendations.csv, games.csv, games_metadata.json
Đầu ra  : data/interim/items.parquet, data/interim/maps.json
          data/interim/interactions.parquet  (chưa chia split)

Chạy:
    python -m src.data.preprocess --config config/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_raw(raw_dir: Path, dtypes: dict | None = None) -> pd.DataFrame:
    """Đọc recommendations.csv an toàn về bộ nhớ.

    TODO
    ----
    * Chỉ định dtype cho từng cột: int32 cho ID, float32 cho hours.
    * Chỉ đọc các cột thật sự cần: user_id, app_id, is_recommended, hours, date.
    * Cân nhắc polars nếu pandas vẫn chậm.
    """
    raise NotImplementedError


def filter_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Giữ is_recommended == True, khử trùng cặp (user, game) giữ bản mới nhất."""
    raise NotImplementedError


def cut_time_window(df: pd.DataFrame, start: str) -> pd.DataFrame:
    """Chỉ giữ review từ mốc `start` trở đi.

    Đây là quyết định thu hẹp phạm vi, không phải thủ thuật giấu dữ liệu.
    Ghi rõ trong báo cáo số dòng trước và sau khi cắt.
    """
    raise NotImplementedError


def k_core(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Lọc k-core LẶP đến khi ổn định.

    Lọc một lượt là chưa đủ: bỏ user thưa sẽ làm một số game tụt xuống dưới
    ngưỡng, và ngược lại. Lặp cho đến khi không còn gì bị loại.
    """
    raise NotImplementedError


def build_index_maps(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ánh xạ user_id và app_id về chỉ số liên tục 0..n-1.

    Trả về (df đã thêm cột u_idx/i_idx, dict để ghi ra maps.json).
    """
    raise NotImplementedError


def build_items(raw_dir: Path, keep_app_ids: np.ndarray) -> pd.DataFrame:
    """Ghép games.csv và games_metadata.json theo ITEM_SCHEMA trong contracts.py.

    Chỉ giữ những game còn lại sau k-core. Cột `tags` nối bằng dấu "|".
    """
    raise NotImplementedError


def report_stats(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Bảng thống kê trước và sau lọc — đưa thẳng vào báo cáo.

    Gồm: số người dùng, số game, số tương tác, độ thưa, trung vị tương tác
    mỗi người dùng. Độ thưa = 1 - |R| / (|U| * |I|).
    """
    raise NotImplementedError


def main(config_path: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    main(p.parse_args().config)
