"""Khung đánh giá. NGƯỜI C — ngày 4-5.

Nhận một Recommender bất kỳ và chấm điểm. A và B gọi run_eval khi chạy thí nghiệm.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def build_truth(test_df: pd.DataFrame, user_ids: np.ndarray) -> dict[int, set[int]]:
    """{chỉ số hàng trong user_ids -> tập i_idx đúng}.

    Chú ý: khóa là CHỈ SỐ HÀNG, không phải u_idx. Đây là chỗ dễ nhầm nhất.
    """
    raise NotImplementedError


def rank_all(
    model,
    user_ids: np.ndarray,
    k: int = 20,
    item_subset: np.ndarray | None = None,
) -> np.ndarray:
    """Sinh dự đoán cho một danh sách người dùng.

    item_subset giới hạn catalog được xếp hạng — dùng cho RQ3 khi chỉ chấm riêng
    nhóm game mới phát hành. None nghĩa là toàn bộ catalog.

    TODO
    ----
    * Chia lô người dùng để không nổ bộ nhớ: catalog 50 nghìn game nhân vài chục
      nghìn người dùng là ma trận rất lớn.
    * KHÔNG lấy mẫu 100 negative khi đánh giá. Cách đó cho điểm cao hơn nhiều và
      không so sánh được với các công trình khác.
    """
    raise NotImplementedError


def run_eval(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_list: Iterable[int] = (10, 20),
    item_subset: np.ndarray | None = None,
) -> dict[str, float]:
    """Tiện ích gộp rank_all + evaluate.

    Chỉ chấm trên những người dùng có ít nhất một tương tác trong test_df.

    TODO
    ----
    * user_ids = test_df.u_idx.unique()
    * preds = rank_all(model, user_ids, k=max(k_list), item_subset=item_subset)
    * truth = build_truth(test_df, user_ids)
    * item_pop = np.bincount(train_df.i_idx, minlength=n_items)
    * n_users  = train_df.u_idx.nunique()      # số user TRONG TRAIN, cho novelty
    * return evaluate(preds, truth, k_list, item_pop, n_items, n_users)
    """
    raise NotImplementedError


def log_run(results: dict, config: dict, path: str = "results/runs.csv") -> None:
    """Ghi một dòng vào results/runs.csv kèm siêu tham số và seed.

    Mỗi con số trong báo cáo phải truy được về một dòng ở đây.
    """
    raise NotImplementedError
