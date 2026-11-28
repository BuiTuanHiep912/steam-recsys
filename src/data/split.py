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
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def find_cutoffs(df: pd.DataFrame, train_ratio: float, valid_ratio: float
                 ) -> tuple[int, int]:
    """Tìm hai mốc thời gian sao cho tỷ lệ theo SỐ TƯƠNG TÁC xấp xỉ mục tiêu.

    Không chia theo khoảng thời gian đều nhau — lượng review mỗi tháng khác nhau.
    Dùng phân vị của cột ts.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio phải nằm giữa 0 và 1")
    if not 0 < valid_ratio < 1 or train_ratio + valid_ratio >= 1:
        raise ValueError("valid_ratio phải dương và train_ratio + valid_ratio < 1")
    if df.empty or df["ts"].isna().any():
        raise ValueError("Cần dữ liệu ts không rỗng và không có giá trị thiếu")

    timestamps = df["ts"].to_numpy(dtype=np.int64)
    unique_ts, counts = np.unique(timestamps, return_counts=True)
    if len(unique_ts) < 3:
        raise ValueError("Cần ít nhất 3 mốc ts khác nhau để tạo đủ ba split")

    # Chọn ranh giới tại timestamp có thật, tránh tách các review cùng thời điểm.
    before_timestamp = np.concatenate(([0], np.cumsum(counts)[:-1]))
    train_candidates = np.arange(1, len(unique_ts) - 1)
    target_train = train_ratio * len(df)
    train_boundary_idx = train_candidates[
        np.argmin(np.abs(before_timestamp[train_candidates] - target_train))
    ]

    valid_candidates = np.arange(train_boundary_idx + 1, len(unique_ts))
    target_valid = (train_ratio + valid_ratio) * len(df)
    valid_boundary_idx = valid_candidates[
        np.argmin(np.abs(before_timestamp[valid_candidates] - target_valid))
    ]
    return int(unique_ts[train_boundary_idx]), int(unique_ts[valid_boundary_idx])


def split_by_time(df: pd.DataFrame, cut_train: int, cut_valid: int
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train: ts < cut_train | valid: cut_train <= ts < cut_valid | test: ts >= cut_valid"""
    if cut_train >= cut_valid:
        raise ValueError("cut_train phải nhỏ hơn cut_valid")
    train = df.loc[df["ts"] < cut_train].copy()
    valid = df.loc[(df["ts"] >= cut_train) & (df["ts"] < cut_valid)].copy()
    test = df.loc[df["ts"] >= cut_valid].copy()
    return train, valid, test


def sanity_check(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                 cut_train: int, cut_valid: int) -> None:
    """Kiểm tra ranh giới, tính rời nhau và báo số user test đã có trong train."""
    if cut_train >= cut_valid:
        raise ValueError("cut_train phải nhỏ hơn cut_valid")
    if (train["ts"] >= cut_train).any():
        raise ValueError("train có tương tác tại/sau cutoff_train")
    if ((valid["ts"] < cut_train) | (valid["ts"] >= cut_valid)).any():
        raise ValueError("valid có tương tác nằm ngoài hai cutoff")
    if (test["ts"] < cut_valid).any():
        raise ValueError("test có tương tác sớm hơn cutoff_valid")

    combined = pd.concat([train, valid, test], ignore_index=True)
    if combined.duplicated(["u_idx", "i_idx", "ts"]).any():
        raise ValueError("Các split có tương tác trùng lặp")

    shared_users = len(set(train["u_idx"]) & set(test["u_idx"]))
    print(f"Test users also present in train: {shared_users:,}")


def main(config_path: str) -> None:
    config_file = Path(config_path).resolve()
    with config_file.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    project_dir = config_file.parent.parent

    def configured_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_dir / path

    interim_dir = configured_path(config["paths"]["interim"])
    split_dir = configured_path(config["paths"]["splits"])
    split_dir.mkdir(parents=True, exist_ok=True)

    interactions = pd.read_parquet(interim_dir / "interactions.parquet")
    items = pd.read_parquet(
        interim_dir / "items.parquet", columns=["i_idx", "date_release"]
    )
    if items["i_idx"].duplicated().any() or items["date_release"].isna().any():
        raise ValueError("items.parquet cần i_idx duy nhất và date_release cho mọi game")
    item_indices = np.sort(items["i_idx"].to_numpy(dtype=np.int64))
    if not np.array_equal(item_indices, np.arange(len(item_indices))):
        raise ValueError("items.parquet i_idx phải phủ liên tục từ 0 đến n_items-1")

    train_ratio = float(config["split"]["train_ratio"])
    valid_ratio = float(config["split"]["valid_ratio"])
    cut_train, cut_valid = find_cutoffs(interactions, train_ratio, valid_ratio)
    train, valid, test = split_by_time(interactions, cut_train, cut_valid)
    sanity_check(train, valid, test, cut_train, cut_valid)
    if len(train) + len(valid) + len(test) != len(interactions):
        raise ValueError("Ba split không bảo toàn số tương tác đầu vào")
    if not interactions["i_idx"].isin(item_indices).all():
        raise ValueError("interactions.parquet có i_idx không có trong items.parquet")

    maps_path = interim_dir / "maps.json"
    with maps_path.open(encoding="utf-8") as file:
        maps = json.load(file)
    maps["cutoff_train"] = cut_train
    maps["cutoff_valid"] = cut_valid

    for name, frame in (("train", train), ("valid", valid), ("test", test)):
        frame.to_parquet(split_dir / f"{name}.parquet", index=False)
    maps_path.write_text(
        json.dumps(maps, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = len(interactions)
    ratios = [len(frame) / total for frame in (train, valid, test)]
    print(
        f"Interactions train/valid/test: "
        f"{len(train):,}/{len(valid):,}/{len(test):,} "
        f"({ratios[0]:.1%}/{ratios[1]:.1%}/{ratios[2]:.1%})"
    )
    print(f"Cutoffs: train={cut_train}, valid={cut_valid}")
    print(
        f"Per-game date_release remains available in "
        f"{interim_dir / 'items.parquet'}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="steam-recsys\\config\\default.yaml")
    main(p.parse_args().config)
