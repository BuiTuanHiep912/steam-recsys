"""Mô hình dựa trên nội dung. NGƯỜI A — ngày 15-17.

Dùng cho RQ3 (game mới phát hành) và cho đường xử lý game mới trong demo.

Chạy thử:
    python -m src.content.content_based --data data_fake
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


class ContentBased:
    """Gợi ý theo độ giống nội dung. Tuân thủ giao diện Recommender.

    name : str
    fit(train_df, confidence=None)
    recommend(user_ids, k=10, exclude_seen=True) -> np.ndarray
    """

    name = "ContentBased"

    def __init__(self, items: pd.DataFrame, max_features: int = 5000):
        """items: data/interim/items.parquet, cần cột `tags` và `description`."""
        self.items = items
        self.max_features = max_features
        self.V = None        # (n_items, n_features) ma trận TF-IDF đã chuẩn hóa
        self.profiles = None  # (n_users, n_features) hồ sơ người dùng

    def _vectorize_items(self) -> None:
        """Bước 1: TF-IDF trên tags + description.

        TODO
        ----
        * Ghép: items["tags"].str.replace("|", " ") + " " + items["description"]
        * TfidfVectorizer(max_features=self.max_features)
        * sklearn.preprocessing.normalize để mỗi hàng có độ dài 1
        * Giữ đúng thứ tự theo i_idx — đây là lỗi dễ mắc nhất.
        """
        raise NotImplementedError

    def fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None = None) -> None:
        """Bước 2: hồ sơ người dùng = trung bình có trọng số vector game đã thích.

            q_u = sum_{i in I_u} c_ui * v_i  /  sum_{i in I_u} c_ui

        Mẫu số để tổng trọng số bằng 1, nên q_u nằm cùng thang với v_i.
        confidence=None thì coi mọi trọng số bằng 1.
        """
        raise NotImplementedError

    def recommend(self, user_ids, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        """Bước 3: cosine giữa hồ sơ và vector game.

        Đã chuẩn hóa cả hai về độ dài 1 thì cosine chính là tích vô hướng.
        """
        raise NotImplementedError

    def recommend_for_new_user(self, liked_i_idx, k: int = 10) -> np.ndarray:
        """Dành cho DEMO: người xem demo không có trong tập huấn luyện.

        Dựng hồ sơ tạm từ danh sách game họ vừa chọn, rồi cho điểm như thường.
        """
        raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_fake")
    raise SystemExit("TODO: chạy thử trên dữ liệu giả và in top-10 cho vài user")
