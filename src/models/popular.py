"""MostPopular — ngưỡng sàn. NGƯỜI B — ngày 3.

Không cá nhân hóa: mọi người dùng nhận cùng một danh sách.

Vì sao bắt buộc phải có: nếu một mô hình phức tạp không vượt được MostPopular,
nghĩa là nó chưa học được gì ngoài độ phổ biến. Rất nhiều bài báo bị phát hiện
mắc đúng lỗi này.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseRecommender


class MostPopular(BaseRecommender):
    name = "MostPopular"

    def _fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None) -> None:
        """Đếm số tương tác mỗi game TỪ TẬP HUẤN LUYỆN.

        TODO
        ----
        * self.pop = np.bincount(train_df.i_idx, minlength=self.n_items)
        * KHÔNG dùng cột user_reviews trong games.csv: đó là ảnh chụp năm 2023,
          đã bao gồm cả tương tác nằm trong tập kiểm thử. Dùng nó là rò rỉ dữ liệu.
        """
        raise NotImplementedError

    def _score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """Điểm giống nhau cho mọi người dùng: lặp lại vector độ phổ biến."""
        raise NotImplementedError
