"""
Hợp đồng giao diện giữa ba người — ĐÓNG BĂNG sau buổi họp ngày 1.

File này chỉ KHAI BÁO, không cài đặt. Ai đổi gì trong đây phải báo cả ba người.

    A  sở hữu  build_confidence   (src/data/signals.py)
    B  sở hữu  Recommender        (src/models/*.py)
    C  sở hữu  evaluate, rank_all (src/eval/*.py)

Quy ước chung
-------------
* Mọi chỉ số người dùng và game là số nguyên liên tục 0..n-1 (u_idx, i_idx).
  ID gốc của Steam (app_id) chỉ xuất hiện trong items.parquet và maps.json.
* Mọi thời điểm là Unix timestamp, đơn vị giây, kiểu int64.
* Mọi mảng điểm số và trọng số là np.float32.
* Seed mặc định của cả nhóm: 42.
"""

from __future__ import annotations

from typing import Iterable, Literal, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Lược đồ dữ liệu
# --------------------------------------------------------------------------

SEED = 42

#: data/splits/{train,valid,test}.parquet — chỉ chứa tương tác DƯƠNG
#: (is_recommended == True đã được lọc ở bước tiền xử lý).
INTERACTION_SCHEMA: Mapping[str, str] = {
    "u_idx": "int32",   # 0..n_users-1
    "i_idx": "int32",   # 0..n_items-1
    "hours": "float32",  # số giờ chơi tại thời điểm viết review, >= 0
    "ts": "int64",      # Unix timestamp của review, đơn vị giây
}

#: data/interim/items.parquet — một dòng mỗi game còn lại sau khi lọc k-core.
ITEM_SCHEMA: Mapping[str, str] = {
    "i_idx": "int32",
    "app_id": "int64",
    "title": "string",
    "date_release": "int64",   # Unix timestamp; dùng để chia nhóm cold-start
    "price_final": "float32",  # USD
    "win": "bool",
    "mac": "bool",
    "linux": "bool",
    "tags": "string",          # các thẻ nối bằng dấu "|"
    "description": "string",
}

#: data/interim/maps.json
#: {
#:   "n_users": int,
#:   "n_items": int,
#:   "cutoff_train": int,   # ts: train là ts <  cutoff_train
#:   "cutoff_valid": int,   # ts: valid là cutoff_train <= ts < cutoff_valid
#:   "seed": int
#: }
MAPS_FILE = "data/interim/maps.json"

ConfidenceScheme = Literal["binary", "log", "linear", "capped"]


# --------------------------------------------------------------------------
# 2. A — sinh trọng số tin cậy
# --------------------------------------------------------------------------

def build_confidence(
    df: pd.DataFrame,
    scheme: ConfidenceScheme = "log",
    alpha: float = 40.0,
    cap_hours: float = 100.0,
) -> np.ndarray:
    """Trả về mảng float32 cùng độ dài và cùng thứ tự với ``df``.

    ``df`` là một split đọc từ parquet, đúng INTERACTION_SCHEMA.

    Công thức theo từng scheme, với h = df["hours"]::

        binary  ->  1 + alpha
        log     ->  1 + alpha * ln(1 + h)
        linear  ->  1 + alpha * h
        capped  ->  1 + alpha * ln(1 + min(h, cap_hours))

    Giá trị trả về là trọng số của Ô DƯƠNG. Ô trống luôn có trọng số 1 và
    do thư viện tự hiểu, không nằm trong mảng này.

    Lưu ý: mọi giá trị đều > 0 và không phụ thuộc thứ tự dòng.
    """
    raise NotImplementedError("A cài đặt trong src/data/signals.py")


# --------------------------------------------------------------------------
# 3. B — mô hình
# --------------------------------------------------------------------------

class Recommender(Protocol):
    """Giao diện chung của MỌI mô hình, kể cả content-based của A.

    C gọi đúng hai phương thức này và không được biết gì thêm về bên trong.
    """

    name: str

    def fit(
        self,
        train_df: pd.DataFrame,
        confidence: np.ndarray | None = None,
    ) -> None:
        """Huấn luyện trên một split.

        ``confidence`` cùng độ dài với ``train_df`` khi mô hình dùng được
        trọng số tin cậy (WMF, ItemKNN, content-based). Mô hình không dùng
        thì bỏ qua tham số này, KHÔNG báo lỗi.
        """
        ...

    def recommend(
        self,
        user_ids: Sequence[int] | np.ndarray,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        """Trả về mảng int32 kích thước (len(user_ids), k) chứa i_idx.

        Hàng thứ r ứng với user_ids[r], các cột đã xếp hạng giảm dần.
        ``exclude_seen=True`` loại mọi game người đó đã tương tác trong
        tập huấn luyện.

        Người dùng không có trong tập huấn luyện vẫn phải trả về đủ k cột
        (ví dụ: lấy theo độ phổ biến). Không được trả về -1 hay NaN.
        """
        ...


# --------------------------------------------------------------------------
# 4. C — đánh giá
# --------------------------------------------------------------------------

def rank_all(
    model: Recommender,
    user_ids: np.ndarray,
    k: int = 20,
    item_subset: np.ndarray | None = None,
) -> np.ndarray:
    """Sinh dự đoán cho một danh sách người dùng.

    ``item_subset`` giới hạn catalog được xếp hạng — dùng cho RQ3 khi chỉ
    chấm riêng nhóm game mới phát hành. None nghĩa là toàn bộ catalog.
    """
    raise NotImplementedError("C cài đặt trong src/eval/evaluate.py")


def evaluate(
    preds: np.ndarray,
    truth: Mapping[int, set[int]],
    k_list: Iterable[int] = (10, 20),
    item_pop: np.ndarray | None = None,
    n_items: int | None = None,
) -> dict[str, float]:
    """Chấm điểm một mảng dự đoán. HÀM THUẦN — không đọc file, không gọi model.

    Nhờ vậy C kiểm thử được bằng một ví dụ tính tay trên giấy.

    preds    : (n_users, k) các i_idx đã xếp hạng, hàng r ứng với user thứ r
    truth    : {chỉ số hàng -> tập i_idx đúng trong tập kiểm thử}
    item_pop : số tương tác mỗi game TRONG TẬP HUẤN LUYỆN, cần cho novelty
    n_items  : kích thước catalog, cần cho coverage

    Khóa trả về::

        ndcg@10, ndcg@20, recall@10, recall@20, mrr,
        coverage@10, novelty@10

    Mọi giá trị là float thường, không phải np.float32 — để ghi thẳng ra CSV.
    """
    raise NotImplementedError("C cài đặt trong src/eval/metrics.py")


def run_eval(
    model: Recommender,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_list: Iterable[int] = (10, 20),
    item_subset: np.ndarray | None = None,
) -> dict[str, float]:
    """Tiện ích gộp rank_all + evaluate. B và A gọi hàm này khi chạy thí nghiệm.

    Chỉ chấm trên những người dùng có ít nhất một tương tác trong test_df.
    """
    raise NotImplementedError("C cài đặt trong src/eval/evaluate.py")
