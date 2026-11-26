"""Phân tích chuyên sâu. NGƯỜI C — ngày 11-14 và 18-20.

Một bảng điểm số không giải thích được hệ thống hoạt động ra sao. Phần này mới là
nội dung chính của báo cáo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- ngày 11-14

def popularity_bias(preds: np.ndarray, item_pop: np.ndarray,
                    n_groups: int = 3) -> pd.DataFrame:
    """Gợi ý rơi vào nhóm game phổ biến hay nhóm đuôi dài?

    Chia catalog thành n_groups nhóm theo độ phổ biến trong tập huấn luyện, tính
    tỷ lệ gợi ý rơi vào từng nhóm, so với tỷ lệ của chúng trong tập huấn luyện.

    Nếu mô hình chính xác nhất cũng là mô hình gợi ý game phổ biến nhiều nhất, đó
    là kết quả cần nói rõ chứ không phải điều đáng giấu.
    """
    raise NotImplementedError


def sample_recommendations(model, train_df: pd.DataFrame, items: pd.DataFrame,
                           user_ids: np.ndarray, k: int = 10) -> str:
    """In lịch sử của vài người dùng cạnh top-10 gợi ý, kèm TÊN GAME.

    Đây là việc nên làm SỚM NHẤT, trước khi tối ưu bất cứ thứ gì. Cách này phát
    hiện lỗi nhanh hơn mọi độ đo — ví dụ gợi ý toàn bản DLC của game vừa chơi, hoặc
    gợi ý giống hệt nhau cho mọi người (dấu hiệu mô hình chỉ học được độ phổ biến).
    """
    raise NotImplementedError


# ---------------------------------------------------------------- ngày 18-20

def split_items_by_release(items: pd.DataFrame, cutoff_ts: int
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Chia catalog thành (game cũ, game mới) theo date_release so với mốc cắt train.

    Dùng cho RQ3: game "mới" là game lọc cộng tác gần như không có dữ liệu để dựa vào.

    Mốc cắt phải khớp với tập model đã học:
      * học trên train          -> cutoff_ts = maps["cutoff_train"]
      * học trên train+valid    -> cutoff_ts = maps["cutoff_valid"]   (chấm test)
    """
    raise NotImplementedError


def rq3_table(models: dict, train_df, test_df, items: pd.DataFrame,
              cutoff_ts: int) -> pd.DataFrame:
    """Bảng kết quả RQ3: hai cột (game cũ, game mới) x hai dòng (WMF, content-based).

    Gọi run_eval hai lần cho mỗi mô hình, mỗi lần với một item_subset khác nhau.
    train_df là tập gộp train+valid; MỌI mô hình (kể cả content-based của A) học lại
    trên tập này. cutoff_ts = maps["cutoff_valid"].
    """
    raise NotImplementedError


def profile_entropy(train_df: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    """Độ rộng sở thích của mỗi người dùng.

        H(u) = - sum_t pi_t * log(pi_t)

    pi_t là tỷ lệ thẻ t trong các game thuộc lịch sử của u. Người chơi chuyên một
    dòng game có H thấp, người chơi tạp có H cao.

    Dùng lại cột `tags` trong items.parquet, tách bằng dấu "|".
    """
    raise NotImplementedError


def metrics_by_breadth(preds, truth, entropy: pd.Series, item_pop,
                       n_groups: int = 3) -> pd.DataFrame:
    """nDCG@10 VÀ novelty theo ba nhóm độ rộng sở thích.

    Phải báo cả hai độ đo. Kỳ vọng nhóm hẹp có nDCG cao nhưng novelty thấp: mô hình
    đúng một cách tầm thường, chỉ đoán được thứ người ta đằng nào cũng tìm thấy.

    Nếu ra như vậy thì đó là phát hiện đáng viết: nDCG cao không đồng nghĩa với gợi
    ý có giá trị.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- ngày 21

def make_all_figures(runs: pd.DataFrame, out_dir: str = "reports/figures") -> None:
    """Toàn bộ biểu đồ cho báo cáo, cùng một bảng màu và cùng cỡ chữ.

    Bốn hình cần có:
      1. nDCG@10 theo alpha, mỗi hàm confidence một đường (kết quả RQ1)
      2. Phân bố long-tail của catalog
      3. Bảng RQ3
      4. nDCG và novelty theo nhóm entropy
    """
    raise NotImplementedError
