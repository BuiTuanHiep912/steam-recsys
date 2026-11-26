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
        """Không cần học thêm: BaseRecommender.fit đã đếm số tương tác mỗi game
        TỪ TẬP HUẤN LUYỆN vào self._pop. Dùng lại để chỉ có một nguồn đếm.

        KHÔNG dùng cột user_reviews trong games.csv: đó là ảnh chụp năm 2023,
        đã bao gồm cả tương tác nằm trong tập kiểm thử. Dùng nó là rò rỉ dữ liệu.

        confidence bị bỏ qua (contracts.py: mô hình không dùng thì không báo lỗi).
        """

    def _score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """Điểm giống nhau cho mọi người dùng: lặp lại vector độ phổ biến.

        broadcast_to không tốn bộ nhớ; BaseRecommender.recommend tự sao chép
        trước khi sửa nên self._pop không bị ghi đè.
        """
        return np.broadcast_to(self._pop, (len(user_ids), self.n_items))
