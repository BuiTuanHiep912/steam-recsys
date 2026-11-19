"""Lớp cơ sở cho mọi mô hình. NGƯỜI B — ngày 1-2.

Mọi mô hình trong thư mục này, và cả ContentBased của A, đều tuân thủ giao diện
Recommender trong contracts.py. C chỉ gọi fit và recommend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BaseRecommender:
    """Xử lý phần chung mà mô hình nào cũng cần.

    Lớp con cài đặt _score_users và đặt lại `name`.
    """

    name = "Base"

    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self._seen: dict[int, set[int]] | None = None

    def fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None = None) -> None:
        """Lưu lịch sử để exclude_seen hoạt động, rồi gọi _fit của lớp con.

        TODO
        ----
        * self._seen = train_df.groupby("u_idx").i_idx.apply(set).to_dict()
        * gọi self._fit(train_df, confidence)
        """
        raise NotImplementedError

    def _fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None) -> None:
        raise NotImplementedError("lớp con cài đặt")

    def _score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """Trả về mảng điểm (len(user_ids), n_items). Lớp con cài đặt."""
        raise NotImplementedError("lớp con cài đặt")

    def recommend(self, user_ids, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        """Chấm điểm, loại game đã xem, lấy top-k.

        TODO
        ----
        * scores = self._score_users(np.asarray(user_ids))
        * nếu exclude_seen: đặt -inf vào các ô đã xem
        * np.argpartition rồi sắp xếp lại — nhanh hơn argsort toàn bộ
        * trả về int32, đúng kích thước (len(user_ids), k), không có -1 hay NaN
        * người dùng không có trong train vẫn phải trả về đủ k cột
        """
        raise NotImplementedError
