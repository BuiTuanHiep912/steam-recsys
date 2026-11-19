"""Chia train/valid/test theo thời gian. NGƯỜI A — ngày 5.

Đầu vào : data/interim/interactions.parquet
Đầu ra  : data/splits/{train,valid,test}.parquet, cập nhật maps.json

Chia theo thời gian chứ không ngẫu nhiên: mô hình chỉ được biết quá khứ khi dự
đoán tương lai. Chia ngẫu nhiên sẽ cho mô hình nhìn thấy tương tác tương lai của
chính người dùng đó, làm điểm số cao giả tạo.

Chạy:
    python -m src.data.split --config config/default.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd


def find_cutoffs(df: pd.DataFrame, train_ratio: float, valid_ratio: float
                 ) -> tuple[int, int]:
    """Tìm hai mốc thời gian sao cho tỷ lệ theo SỐ TƯƠNG TÁC xấp xỉ mục tiêu.

    Không chia theo khoảng thời gian đều nhau — lượng review mỗi tháng khác nhau.
    Dùng phân vị của cột ts.
    """
    raise NotImplementedError


def split_by_time(df: pd.DataFrame, cut_train: int, cut_valid: int
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train: ts < cut_train | valid: cut_train <= ts < cut_valid | test: ts >= cut_valid"""
    raise NotImplementedError


def sanity_check(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                 cut_train: int, cut_valid: int) -> None:
    """Kiểm tra bắt buộc trước khi ghi file.

    TODO
    ----
    * Không có dòng nào trong test có ts < cut_valid.
    * Ba split rời nhau và hợp lại bằng tập gốc.
    * Có bao nhiêu người dùng trong test cũng xuất hiện trong train? In ra —
      đây là con số C cần khi chấm điểm.
    """
    raise NotImplementedError


def main(config_path: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    main(p.parse_args().config)
