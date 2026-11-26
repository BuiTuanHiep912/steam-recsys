"""STUB tạm của build_confidence. NGƯỜI B — ngày 8 (PLANB mục 7).

Chỉ để B chạy WMF trước khi A xong src/data/signals.py. Đúng chữ ký và công
thức trong contracts.py. Khi A xong:

    1. đổi import sang  from src.data.signals import build_confidence
    2. xóa file này
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMES = ("binary", "log", "linear", "capped")


def build_confidence(
    df: pd.DataFrame,
    scheme: str = "log",
    alpha: float = 40.0,
    cap_hours: float = 100.0,
) -> np.ndarray:
    """Trả về mảng float32 cùng độ dài và cùng thứ tự với df.

    Công thức, với h = df["hours"]::

        binary  ->  1 + alpha
        log     ->  1 + alpha * ln(1 + h)
        linear  ->  1 + alpha * h
        capped  ->  1 + alpha * ln(1 + min(h, cap_hours))
    """
    h = df["hours"].to_numpy(dtype=np.float64)
    if scheme == "binary":
        c = np.full(len(h), 1.0 + alpha)
    elif scheme == "log":
        c = 1.0 + alpha * np.log1p(h)
    elif scheme == "linear":
        c = 1.0 + alpha * h
    elif scheme == "capped":
        c = 1.0 + alpha * np.log1p(np.minimum(h, cap_hours))
    else:
        raise ValueError(f"scheme phải thuộc {SCHEMES}, nhận {scheme!r}")
    c = c.astype(np.float32)
    if not (c > 0).all():
        raise ValueError("confidence phải > 0 ở mọi dòng")
    return c
