"""Sinh trọng số tin cậy. NGƯỜI A — ngày 8-10.

Đây là trái tim của RQ1. B gọi hàm này cho mọi lần chạy thí nghiệm, nên chữ ký
phải khớp đúng contracts.py và không được đổi sau ngày 1.

Nhắc lại hai tín hiệu, đừng gộp làm một:

    p_ui  = 1 nếu có tương tác, 0 nếu không     <- ĐÍCH ĐẾN, luôn nhị phân
    c_ui  = 1 + alpha * log(1 + hours)          <- TRỌNG SỐ của ô đó

Đặt độ lớn giờ chơi vào p_ui là sai lầm nặng nhất có thể mắc: khi đó mô hình học
dự đoán thời lượng chơi chứ không phải mức độ ưa thích, và game dài luôn đứng đầu
bảng gợi ý bất kể người dùng có hợp thể loại hay không.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_confidence(
    df: pd.DataFrame,
    scheme: str = "log",
    alpha: float = 40.0,
    cap_hours: float = 100.0,
) -> np.ndarray:
    """Trả về mảng float32 cùng độ dài và cùng thứ tự với df.

    Công thức, với h = df["hours"]::

        binary  ->  1 + alpha                              (mọi ô dương như nhau)
        log     ->  1 + alpha * ln(1 + h)                  (giả thuyết chính)
        linear  ->  1 + alpha * h                          (đối chứng: vì sao cần log)
        capped  ->  1 + alpha * ln(1 + min(h, cap_hours))  (chặn tài khoản cực đoan)

    TODO
    ----
    * Dùng np.log1p, không tự viết log(1+x).
    * Ép kiểu float32 trước khi trả về.
    * Kiểm tra mọi giá trị > 0.
    """
    raise NotImplementedError


def normalize_hours_per_game(df: pd.DataFrame) -> np.ndarray:
    """Chuẩn hóa giờ chơi theo trung vị của CHÍNH game đó. Dành cho RQ2 (tùy chọn).

        h~_ui = ln(1 + h_ui) / median_{j chơi i} ln(1 + h_ji)

    Ý tưởng: 40 giờ cho một game nhập vai là ít, cho một game indie là rất nhiều.
    Chuẩn hóa theo game cho phép so sánh giữa các thể loại.

    Chỉ làm nếu RQ1 xong trước ngày 15.
    """
    raise NotImplementedError
