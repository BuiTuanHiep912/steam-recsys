"""Tiền xử lý dữ liệu thô Steam. NGƯỜI A — ngày 3-5.

Đầu vào : data/raw/recommendations.csv, games.csv, games_metadata.json
Đầu ra  : data/interim/items.parquet, data/interim/maps.json
          data/interim/interactions.parquet  (chưa chia split)

Chạy:
    python -m src.data.preprocess --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


RAW_COLUMNS = ["user_id", "app_id", "is_recommended", "hours", "date"]
RAW_DTYPES = {
    "user_id": "int32",
    "app_id": "int32",
    "is_recommended": "bool",
    "hours": "float32",
    "date": "string",
}
READ_CHUNK_SIZE = 1_000_000


def load_raw(raw_dir: Path, dtypes: dict | None = None) -> pd.DataFrame:
    """Đọc từng khối, chỉ giữ cột cần thiết và giảm bộ nhớ nhờ dtype tường minh."""
    path = raw_dir / "recommendations.csv"
    chunk_dtypes = RAW_DTYPES | (dtypes or {})
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=RAW_COLUMNS,
        dtype=chunk_dtypes,
        chunksize=READ_CHUNK_SIZE,
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="raise", utc=True)
        chunk = chunk.loc[chunk["is_recommended"].eq(True)].copy()
        if chunk["date"].isna().any():
            raise ValueError("recommendations.csv chứa date trống")
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame({
            "user_id": pd.Series(dtype=chunk_dtypes["user_id"]),
            "app_id": pd.Series(dtype=chunk_dtypes["app_id"]),
            "is_recommended": pd.Series(dtype=chunk_dtypes["is_recommended"]),
            "hours": pd.Series(dtype=chunk_dtypes["hours"]),
            "date": pd.Series(dtype="datetime64[ns, UTC]"),
        })
    return pd.concat(chunks, ignore_index=True)


def filter_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Giữ is_recommended == True, khử trùng cặp (user, game) giữ bản mới nhất."""
    positive = df.loc[df["is_recommended"].eq(True)].copy()
    if positive.empty:
        return positive.drop(columns=["is_recommended"], errors="ignore")

    positive["date"] = pd.to_datetime(positive["date"], errors="raise", utc=True)
    if positive["date"].isna().any():
        raise ValueError("recommendations.csv chứa date trống")
    positive = positive.sort_values("date", kind="mergesort")
    positive = positive.drop_duplicates(["user_id", "app_id"], keep="last")
    return positive.drop(columns=["is_recommended"])


def cut_time_window(df: pd.DataFrame, start: str) -> pd.DataFrame:
    """Chỉ giữ review từ mốc `start` trở đi.

    Đây là quyết định thu hẹp phạm vi, không phải thủ thuật giấu dữ liệu.
    Ghi rõ trong báo cáo số dòng trước và sau khi cắt.
    """
    dates = pd.to_datetime(df["date"], errors="raise", utc=True)
    cutoff = pd.to_datetime(start, errors="raise", utc=True)
    if dates.isna().any():
        raise ValueError("recommendations.csv chứa date trống")
    return df.loc[dates >= cutoff].copy()


def k_core(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Lọc k-core LẶP đến khi ổn định.

    Lọc một lượt là chưa đủ: bỏ user thưa sẽ làm một số game tụt xuống dưới
    ngưỡng, và ngược lại. Lặp cho đến khi không còn gì bị loại.
    """
    if k < 1:
        raise ValueError("k phải lớn hơn hoặc bằng 1")
    current = df.copy()
    while not current.empty:
        user_counts = current["user_id"].value_counts()
        item_counts = current["app_id"].value_counts()
        keep = (
            current["user_id"].map(user_counts).ge(k)
            & current["app_id"].map(item_counts).ge(k)
        )
        filtered = current.loc[keep]
        if len(filtered) == len(current):
            return filtered.reset_index(drop=True)
        current = filtered
    return current.reset_index(drop=True)


def build_index_maps(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ánh xạ user_id và app_id về chỉ số liên tục 0..n-1.

    Trả về (df đã thêm cột u_idx/i_idx, dict để ghi ra maps.json).
    """
    result = df.copy()
    user_ids = np.sort(result["user_id"].unique())
    app_ids = np.sort(result["app_id"].unique())
    result["u_idx"] = pd.Categorical(
        result["user_id"], categories=user_ids, ordered=True
    ).codes.astype(np.int32)
    result["i_idx"] = pd.Categorical(
        result["app_id"], categories=app_ids, ordered=True
    ).codes.astype(np.int32)
    maps = {
        "n_users": int(len(user_ids)),
        "n_items": int(len(app_ids)),
        "u_idx_to_user_id": user_ids.astype(np.int64).tolist(),
        "i_idx_to_app_id": app_ids.astype(np.int64).tolist(),
    }
    return result, maps


def build_items(raw_dir: Path, keep_app_ids: np.ndarray) -> pd.DataFrame:
    """Ghép games.csv và games_metadata.json theo ITEM_SCHEMA trong contracts.py.

    Chỉ giữ những game còn lại sau k-core. Cột `tags` nối bằng dấu "|".
    """
    keep_app_ids = np.asarray(keep_app_ids, dtype=np.int64)
    games = pd.read_csv(
        raw_dir / "games.csv",
        usecols=[
            "app_id", "title", "date_release", "win", "mac", "linux",
            "price_final",
        ],
        dtype={
            "app_id": "int64",
            "title": "string",
            "date_release": "string",
            "win": "bool",
            "mac": "bool",
            "linux": "bool",
            "price_final": "float32",
        },
    )
    games = games.loc[games["app_id"].isin(keep_app_ids)].copy()
    if games["app_id"].duplicated().any():
        raise ValueError("games.csv chứa app_id trùng lặp")
    missing = np.setdiff1d(keep_app_ids, games["app_id"].to_numpy())
    if len(missing):
        raise ValueError(
            f"Không tìm thấy {len(missing)} app_id trong games.csv; "
            f"ví dụ: {missing[:5].tolist()}"
        )
    if games["title"].isna().any():
        raise ValueError("games.csv thiếu title cho game còn lại sau lọc")

    games["date_release"] = pd.to_datetime(
        games["date_release"], errors="coerce", utc=True
    )
    release_ts = games["date_release"].astype("int64") // 1_000_000_000
    games["date_release"] = release_ts.where(games["date_release"].notna(), -1)

    metadata = {}
    metadata_path = raw_dir / "games_metadata.json"
    wanted = set(keep_app_ids.tolist())
    with metadata_path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            app_id = record.get("app_id")
            if app_id in wanted:
                metadata[app_id] = (
                    record.get("description") or "",
                    "|".join(record.get("tags") or []),
                )
    games["description"] = games["app_id"].map(
        lambda app_id: metadata.get(app_id, ("", ""))[0]
    ).astype("string")
    games["tags"] = games["app_id"].map(
        lambda app_id: metadata.get(app_id, ("", ""))[1]
    ).astype("string")

    order = pd.DataFrame({
        "app_id": keep_app_ids,
        "i_idx": np.arange(len(keep_app_ids), dtype=np.int32),
    })
    items = order.merge(games, on="app_id", how="left", validate="one_to_one")
    return items[
        [
            "i_idx", "app_id", "title", "date_release", "price_final",
            "win", "mac", "linux", "tags", "description",
        ]
    ].astype({
        "i_idx": "int32",
        "app_id": "int64",
        "title": "string",
        "date_release": "int64",
        "price_final": "float32",
        "win": "bool",
        "mac": "bool",
        "linux": "bool",
        "tags": "string",
        "description": "string",
    })


def report_stats(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Bảng thống kê trước và sau lọc — đưa thẳng vào báo cáo.

    Gồm: số người dùng, số game, số tương tác, độ thưa, trung vị tương tác
    mỗi người dùng. Độ thưa = 1 - |R| / (|U| * |I|).
    """
    rows = []
    for stage, frame in (("before_k_core", before), ("after_k_core", after)):
        n_users = int(frame["user_id"].nunique())
        n_items = int(frame["app_id"].nunique())
        n_interactions = int(len(frame))
        possible = n_users * n_items
        rows.append({
            "stage": stage,
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": n_interactions,
            "sparsity": 1.0 - n_interactions / possible if possible else np.nan,
            "median_interactions_per_user": (
                float(frame.groupby("user_id").size().median())
                if n_users else np.nan
            ),
        })
    return pd.DataFrame(rows)


def main(config_path: str) -> None:
    config_file = Path(config_path).resolve()
    with config_file.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    project_dir = config_file.parent.parent

    def configured_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_dir / path

    raw_dir = configured_path(config["paths"]["raw"])
    interim_dir = configured_path(config["paths"]["interim"])
    reports_dir = project_dir / "reports"
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    positive = load_raw(raw_dir)
    deduplicated = filter_positive(positive)
    windowed = cut_time_window(
        deduplicated, config["preprocess"]["time_window_start"]
    )
    core = k_core(windowed, int(config["preprocess"]["k_core"]))
    stats = report_stats(windowed, core)
    stats.to_csv(reports_dir / "preprocess_stats.csv", index=False)
    print(stats.to_string(index=False))

    indexed, maps = build_index_maps(core)
    items = build_items(raw_dir, np.asarray(maps["i_idx_to_app_id"]))
    maps["i_idx_to_title"] = items.sort_values("i_idx")["title"].tolist()
    maps["seed"] = int(config.get("seed", 42))

    timestamps = indexed["date"].astype("int64") // 1_000_000_000
    if indexed["hours"].isna().any() or (indexed["hours"] < 0).any():
        raise ValueError("recommendations.csv chứa hours trống hoặc âm")
    interactions = pd.DataFrame({
        "u_idx": indexed["u_idx"].astype("int32"),
        "i_idx": indexed["i_idx"].astype("int32"),
        "hours": indexed["hours"].astype("float32"),
        "ts": timestamps.astype("int64"),
    })
    interactions.to_parquet(interim_dir / "interactions.parquet", index=False)
    items.to_parquet(interim_dir / "items.parquet", index=False)
    (interim_dir / "maps.json").write_text(
        json.dumps(maps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote processed data and maps to {interim_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="steam-recsys\\config\\default.yaml")
    main(p.parse_args().config)
