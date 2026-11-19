"""WMF / ALS — mô hình chính của RQ1. NGƯỜI B — ngày 8-10.

Đây là mô hình DUY NHẤT trong đồ án dùng được trọng số tin cậy từng ô, nên toàn bộ
thí nghiệm về giờ chơi chạy trên nó.

Hàm mục tiêu (Hu, Koren, Volinsky 2008):

    min_{X,Y}  sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + lambda(||X||^2 + ||Y||^2)

Khác phân rã ma trận cổ điển ở hai chỗ: có trọng số c_ui cho từng ô, và tổng chạy
qua TOÀN BỘ ma trận kể cả ô trống. Ô trống vẫn tham gia huấn luyện với p = 0 nhưng
trọng số chỉ bằng 1, nên mô hình được phép cãi lại giả định "không thích" mà gần
như không bị phạt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseRecommender


class WMF(BaseRecommender):
    name = "WMF"

    def __init__(self, n_users: int, n_items: int, factors: int = 64,
                 regularization: float = 0.05, iterations: int = 20,
                 seed: int = 42):
        super().__init__(n_users, n_items)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed
        self.model = None

    def _fit(self, train_df: pd.DataFrame, confidence: np.ndarray | None) -> None:
        """Huấn luyện bằng implicit.als.AlternatingLeastSquares.

        TODO
        ----
        * Dựng ma trận thưa với GIÁ TRỊ LÀ CONFIDENCE, không phải 1.
          C = csr_matrix((confidence, (u_idx, i_idx)), shape=(n_users, n_items))
        * AlternatingLeastSquares(factors=..., regularization=...,
                                  iterations=..., alpha=1.0, random_state=seed)

        CẠM BẪY alpha
        -------------
        Thư viện implicit có tham số `alpha` riêng và một số phiên bản tự nhân ma
        trận đầu vào với nó. Nếu bạn vừa tự tính confidence (đã có alpha bên trong)
        vừa truyền alpha=40 vào constructor thì alpha bị áp dụng HAI LẦN.

        Cách an toàn: tự tính confidence bằng build_confidence của A, rồi đặt
        alpha=1.0. Kiểm tra lại docstring của đúng phiên bản đang cài.
        """
        raise NotImplementedError

    def _score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """s(u,i) = x_u . y_i"""
        raise NotImplementedError

    def fold_in(self, liked_i_idx, confidence=None) -> np.ndarray:
        """Tính vector cho một người dùng MỚI mà không huấn luyện lại. Dành cho demo.

            x_u = (Y^T C^u Y + lambda I)^-1 Y^T C^u p_u

        Trong implicit có thể gọi model.recommend(..., recalculate_user=True).
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Xuất item_factors và user_factors cho demo. Hạn: ngày 14."""
        raise NotImplementedError
