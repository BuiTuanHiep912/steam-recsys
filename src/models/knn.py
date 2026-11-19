"""ItemKNN — lọc cộng tác láng giềng. NGƯỜI B — ngày 4-5.

Hai game giống nhau nếu hay được cùng một nhóm người chơi.

    sim(i,j) = |U_i giao U_j| / (sqrt|U_i| * sqrt|U_j|)
    s(u,i)   = tổng sim(i,j) với j chạy qua lịch sử của u

Giá trị riêng của mô hình này: nó giải thích được. "Gợi ý Dead Cells vì bạn đã chơi
Hades" — các mô hình nhân tố ẩn không làm được điều đó. Vì vậy demo dùng nó.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseRecommender


class ItemKNN(BaseRecommender):
    name = "ItemKNN"

    def __init__(self, n_users: int, n_items: int, k_neighbors: int = 50):
        super().__init__(n_users, n_items)
        self.k_neighbors = k_neighbors
        self.sim = None  # ma trận tương đồng, thưa

    def _fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None) -> None:
        """Dùng implicit.nearest_neighbours.CosineRecommender.

        TODO
        ----
        * Dựng ma trận thưa (n_users, n_items) từ train_df.
        * CosineRecommender(K=self.k_neighbors).fit(X)
        * Nhớ kiểm tra chiều của ma trận mà phiên bản implicit đang cài mong đợi.
        """
        raise NotImplementedError

    def _score_users(self, user_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def export_similarity(self, path: str) -> None:
        """Xuất ma trận tương đồng cho A dùng trong demo. Hạn: ngày 5.

        Lưu dạng thưa (scipy.sparse.save_npz) để file không quá lớn.
        """
        raise NotImplementedError

    def top_similar_pairs(self, items: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """In vài cặp game giống nhau nhất — để kiểm tra bằng mắt.

        Đây là tiêu chí Xong khi của việc này: các cặp phải trông hợp lý với
        người đọc. Nếu ra toàn cặp vô nghĩa thì có lỗi ở đâu đó.
        """
        raise NotImplementedError
