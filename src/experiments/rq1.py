"""RQ1: giờ chơi làm trọng số tin cậy có giúp không? NGƯỜI B — ngày 11-14.

16 lần chạy = 4 hàm confidence x 4 giá trị alpha. Mọi thứ khác cố định.

Chạy:
    python -m src.experiments.rq1 --config config/default.yaml --data data/splits
"""

from __future__ import annotations

import argparse

import pandas as pd


def run_one(train_df, valid_df, scheme: str, alpha: float, cfg: dict) -> dict:
    """Một lần chạy: dựng confidence, huấn luyện WMF, chấm trên tập kiểm định.

    TODO
    ----
    * conf = build_confidence(train_df, scheme=scheme, alpha=alpha)
    * model = WMF(...); model.fit(train_df, confidence=conf)
    * metrics = run_eval(model, train_df, valid_df)
    * trả về dict gồm scheme, alpha, mọi siêu tham số, seed, và các độ đo
    """
    raise NotImplementedError


def sweep(train_df, valid_df, cfg: dict) -> pd.DataFrame:
    """Quét toàn bộ lưới và ghi vào results/runs.csv.

    Ghi dần sau MỖI lần chạy, đừng đợi hết 16 lần. Máy sập giữa chừng thì không
    mất hết.
    """
    raise NotImplementedError


def pick_best(runs: pd.DataFrame) -> dict:
    """Chọn cấu hình tốt nhất theo nDCG@10 TRÊN TẬP KIỂM ĐỊNH."""
    raise NotImplementedError


def final_test(train_df, test_df, best: dict, cfg: dict) -> dict:
    """Chạy MỘT LẦN DUY NHẤT trên tập kiểm thử với cấu hình đã chọn.

    Đây là con số vào báo cáo. Không chạy lại, không tinh chỉnh thêm sau bước này.
    """
    raise NotImplementedError


def main(config_path: str, data_dir: str) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--data", default="data/splits")
    a = p.parse_args()
    main(a.config, a.data)
