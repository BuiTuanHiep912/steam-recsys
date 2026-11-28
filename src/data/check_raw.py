"""Kiểm tra dữ liệu thô trước khi tiền xử lý. NGƯỜI A — ngày 1.

Chạy cái này NGAY sau khi tải xong, đừng đợi đến ngày 3. Nó bắt sớm ba lỗi hay gặp:
tải thiếu file, giải nén dở dang, và bộ dữ liệu trên Kaggle đã đổi tên cột so với
lúc nhóm lên kế hoạch.

    python -m src.data.check_raw --raw data/raw

Script chỉ ĐỌC, không sửa gì. Chạy trong vài giây vì nó chỉ đọc vài dòng đầu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Cột mong đợi theo mô tả bộ dữ liệu. Nếu Kaggle đổi, script báo chứ không chết —
# A cập nhật lại danh sách này rồi báo cả nhóm.
EXPECTED = {
    "recommendations.csv": [
        "app_id", "helpful", "funny", "date", "is_recommended",
        "hours", "user_id", "review_id",
    ],
    "games.csv": [
        "app_id", "title", "date_release", "win", "mac", "linux",
        "rating", "positive_ratio", "user_reviews",
        "price_final", "price_original", "discount", "steam_deck",
    ],
    "users.csv": ["user_id", "products", "reviews"],
}
METADATA = "games_metadata.json"


def human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def check_csv(path: Path, expected: list[str]) -> bool:
    if not path.exists():
        print(f"  THIẾU  {path.name}")
        return False

    head = pd.read_csv(path, nrows=5)
    cols = list(head.columns)
    print(f"  OK     {path.name:<24} {human(path.stat().st_size):>8}  "
          f"{len(cols)} cột")

    missing = [c for c in expected if c not in cols]
    extra = [c for c in cols if c not in expected]
    if missing:
        print(f"         thiếu cột: {missing}")
    if extra:
        print(f"         cột lạ:    {extra}")
    return not missing


def count_rows(path: Path) -> int:
    """Đếm dòng mà không nạp cả file vào bộ nhớ."""
    with path.open("rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def main(raw_dir: str) -> None:
    raw = Path(raw_dir)
    print(f"Kiểm tra {raw.resolve()}\n")

    ok = True
    for name, cols in EXPECTED.items():
        ok &= check_csv(raw / name, cols)

    meta = raw / METADATA
    if meta.exists():
        with meta.open(encoding="utf-8") as f:
            first = json.loads(f.readline())
        print(f"  OK     {METADATA:<24} {human(meta.stat().st_size):>8}  "
              f"khóa: {list(first)}")
    else:
        print(f"  THIẾU  {METADATA}")
        ok = False

    rec = raw / "recommendations.csv"
    if rec.exists():
        n = count_rows(rec) - 1
        print(f"\n  recommendations.csv có {n:,} dòng")
        print(f"  nạp thẳng bằng pandas mặc định tốn khoảng "
              f"{human(n * 100)} RAM — đó là lý do phải chỉ định dtype")

    print("\n" + ("Đủ file, chạy preprocess được." if ok else
                  "CHƯA ĐỦ — xem hướng dẫn tải trong src/data/README.md"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="steam-recsys/data/raw")
    main(p.parse_args().raw)
